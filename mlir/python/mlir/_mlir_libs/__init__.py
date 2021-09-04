# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

import os

_this_dir = os.path.dirname(__file__)


def get_lib_dirs() -> Sequence[str]:
  """Gets the lib directory for linking to shared libraries.

  On some platforms, the package may need to be built specially to export
  development libraries.
  """
  return [_this_dir]


def get_include_dirs() -> Sequence[str]:
  """Gets the include directory for compiling against exported C libraries.

  Depending on how the package was build, development C libraries may or may
  not be present.
  """
  return [os.path.join(_this_dir, "include")]
