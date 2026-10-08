# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


def _validate_package_contents():
    # TODO: When ir2vec python bindings add support for Windows and MacOS,
    # we should update the expected patterns to include the
    # appropriate file extensions for each platform, e.g.:
    # "ir2vec/ir2vec.cpython-*.pyd", "ir2vec/ir2vec.cpython-*.dylib"],
    _EXPECTED_PACKAGE_CONTENTS = {
        "bindings module": ["ir2vec/ir2vec.cpython-*.so"],
        "seed embedding vocabulary file": ["ir2vec/seedEmbeddingVocab75D.json"],
    }

    for description, patterns in _EXPECTED_PACKAGE_CONTENTS.items():
        if not any(glob.glob(p) for p in patterns):
            raise FileNotFoundError(
                f"Missing {description}: none of {patterns} found in ir2vec/. "
                "Run the build step before packaging."
            )


_validate_package_contents()

setup(distclass=BinaryDistribution)
