set(LLVM_TARGET_DEFINITIONS ArithOps.td)
mlir_tablegen(ArithOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ArithOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ArithOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=arith)
mlir_tablegen(ArithOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=arith)
add_mlir_dialect(ArithOps arith)
add_mlir_doc(ArithOps ArithOps Dialects/ -gen-dialect-doc)
add_mlir_interface(ArithOpsInterfaces)

if(XCODE)
  # The Xcode generator needs a dependency between these two, one way or
  # another, since both targets generate the same file:
  # tools/mlir/include/mlir/Dialect/Arith/IR/ArithOpsEnums.h.inc
  add_dependencies(MLIRArithOpsInterfacesIncGen MLIRArithOpsIncGen)
endif()