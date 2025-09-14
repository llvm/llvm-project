set(LLVM_TARGET_DEFINITIONS python_test_ops.td)
mlir_tablegen(lib/PythonTestDialect.h.inc -gen-dialect-decls)
mlir_tablegen(lib/PythonTestDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(lib/PythonTestOps.h.inc -gen-op-decls)
mlir_tablegen(lib/PythonTestOps.cpp.inc -gen-op-defs)
mlir_tablegen(lib/PythonTestAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(lib/PythonTestAttributes.cpp.inc -gen-attrdef-defs)
mlir_tablegen(lib/PythonTestTypes.h.inc -gen-typedef-decls)
mlir_tablegen(lib/PythonTestTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRPythonTestIncGen)

add_subdirectory(lib)

set(MLIR_PYTHON_TEST_DEPENDS MLIRPythonModules)
if(NOT MLIR_STANDALONE_BUILD)
  list(APPEND MLIR_PYTHON_TEST_DEPENDS FileCheck count)
endif()
add_lit_testsuite(check-mlir-python "Running the MLIR Python regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_PYTHON_TEST_DEPENDS}
)
