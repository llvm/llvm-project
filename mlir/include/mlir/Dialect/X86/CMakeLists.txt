add_mlir_dialect(X86 x86)
add_mlir_doc(X86 X86 Dialects/ -gen-dialect-doc -dialect=x86)

set(LLVM_TARGET_DEFINITIONS X86Enums.td)
mlir_tablegen(X86Enums.h.inc -gen-enum-decls)
mlir_tablegen(X86Enums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIRX86EnumsIncGen)

add_mlir_interface(X86Interfaces)
add_dependencies(MLIRX86IncGen MLIRX86InterfacesIncGen MLIRX86EnumsIncGen)

add_subdirectory(TransformOps)
