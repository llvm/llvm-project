set(LLVM_TARGET_DEFINITIONS python_test_ops.td)
mlir_tablegen(lib/PythonTestDialect.h.inc -gen-dialect-decls)
mlir_tablegen(lib/PythonTestDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(lib/PythonTestOps.h.inc -gen-op-decls)
mlir_tablegen(lib/PythonTestOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRPythonTestIncGen)

add_subdirectory(lib)
