# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
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

setup(distclass=BinaryDistribution)
