add_mlir_dialect(X86 x86)
add_mlir_doc(X86 X86 Dialects/ -gen-dialect-doc -dialect=x86)

add_mlir_interface(X86Interfaces)
add_dependencies(MLIRX86IncGen MLIRX86InterfacesIncGen)

add_subdirectory(TransformOps)
