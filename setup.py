#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    author="Ruairi OSullivan",
    author_email="ruairi.osullivan.work@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Utility functions for working with scikit-learn",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="sklearn_utils",
    name="sklearn_utils",
    packages=find_packages(include=["sklearn_utils", "sklearn_utils.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Ruairi-osul/sklearn-utils",
    version="0.0.1",
    zip_safe=False,
)
