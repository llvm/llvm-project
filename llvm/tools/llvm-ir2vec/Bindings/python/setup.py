# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
import os
import warnings
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


if (
    not glob.glob("ir2vec/*.so")
    + glob.glob("ir2vec/*.pyd")
    + glob.glob("ir2vec/*.dylib")
):
    warnings.warn(
        "No native module (.so/.pyd/.dylib) found in ir2vec/. "
        "Run the build step before invoking pip wheel.",
        stacklevel=1,
    )

# Version tracks the LLVM release this package was built against.
# Format: <llvm_major>.<llvm_minor>.<llvm_patch>[.<binding_patch>]
# Examples:
#   "20.1.0"   - first release built against LLVM 20.1.0
#   "20.1.0.1" - binding-layer fix, same LLVM 20.1.0
#   "20.1.1"   - built against a different LLVM 20.1.1
#
# Must be a plain PEP 440 version with no local segment (+...)
# so it can be published to PyPI.
#
# "0.0.0.dev0" signals a local dev build where version was not injected.
version = os.environ.get("IR2VEC_VERSION", "0.0.0.dev0")

setup(
    version=version,
    package_data={
        "ir2vec": ["*.so", "*.pyd", "*.dylib"],
        "ir2vec.vocab_data": ["*.json"],
    },
    distclass=BinaryDistribution,
)
