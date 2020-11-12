#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import platform

_is_windows = platform.system() == "Windows"
_this_directory = os.path.dirname(__file__)

# The standard LLVM build/install tree for Windows is laid out as:
#   bin/
#     MLIRPublicAPI.dll
#   python/
#     _mlir.*.pyd (dll extension)
#     mlir/
#       _dlloader.py (this file)
# First check the python/ directory level for DLLs co-located with the pyd
# file, and then fall back to searching the bin/ directory.
# TODO: This should be configurable at some point.
_dll_search_path = [
  os.path.join(_this_directory, ".."),
  os.path.join(_this_directory, "..", "..", "bin"),
]

# Stash loaded DLLs to keep them alive.
_loaded_dlls = []

def preload_dependency(public_name):
  """Preloads a dylib by its soname or DLL name.

  On Windows and Linux, doing this prior to loading a dependency will populate
  the library in the flat namespace so that a subsequent library that depend
  on it will resolve to this preloaded version.

  On OSX, resolution is completely path based so this facility no-ops. On
  Linux, as long as RPATHs are setup properly, resolution is path based but
  this facility can still act as an escape hatch for relocatable distributions.
  """
  if _is_windows:
    _preload_dependency_windows(public_name)


def _preload_dependency_windows(public_name):
  dll_basename = public_name + ".dll"
  found_path = None
  for search_dir in _dll_search_path:
    candidate_path = os.path.join(search_dir, dll_basename)
    if os.path.exists(candidate_path):
      found_path = candidate_path
      break

  if found_path is None:
    raise RuntimeError(
      f"Unable to find dependency DLL {dll_basename} in search "
      f"path {_dll_search_path}")

  import ctypes
  _loaded_dlls.append(ctypes.CDLL(found_path))
