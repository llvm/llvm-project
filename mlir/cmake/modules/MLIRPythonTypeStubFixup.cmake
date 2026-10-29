# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

file(READ "${INPUT_FILE}" _content)
string(REPLACE "\r\n" "\n" _content "${_content}")
# DefaultingPyMlirContext and DefaultingPyLocation describe their types using
# bare _mlir.ir names even when the extension is imported under a package.
# Normalize both those names and fully-qualified native IR references.
foreach(_module
    "${MLIR_PYTHON_PACKAGE_PREFIX}._mlir_libs._mlir.ir"
    "_mlir.ir"
    "${MLIR_PYTHON_PACKAGE_PREFIX}.ir")
  string(REPLACE "import ${_module}\n" "" _content "${_content}")
  string(REPLACE "${_module}" "_ir" _content "${_content}")
endforeach()
if(NOT _content MATCHES "(^|[^A-Za-z0-9_.])_mlir\\.")
  string(REPLACE "import _mlir\n" "" _content "${_content}")
endif()
file(WRITE "${OUTPUT_FILE}"
  "from ${MLIR_PYTHON_PACKAGE_PREFIX} import ir as _ir\n${_content}")
