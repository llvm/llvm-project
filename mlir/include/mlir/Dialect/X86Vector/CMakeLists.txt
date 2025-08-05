add_mlir_dialect(X86Vector x86vector)
add_mlir_doc(X86Vector X86Vector Dialects/ -gen-dialect-doc -dialect=x86vector)

add_mlir_interface(X86VectorInterfaces)
add_dependencies(MLIRX86VectorIncGen MLIRX86VectorInterfacesIncGen)
