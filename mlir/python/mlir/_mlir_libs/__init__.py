# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import os

__all__ = [
  "load_extension",
  "preload_dependency",
]

_this_dir = os.path.dirname(__file__)

def load_extension(name):
  return importlib.import_module(f".{name}", __package__)


def preload_dependency(public_name):
  # TODO: Implement this hook to pre-load DLLs with ctypes on Windows.
  pass
