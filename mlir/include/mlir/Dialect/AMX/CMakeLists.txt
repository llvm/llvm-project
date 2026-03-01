add_mlir_dialect(AMX amx)
add_mlir_doc(AMX AMX Dialects/ -gen-dialect-doc -dialect=amx)

add_mlir_interface(AMXInterfaces)
add_dependencies(MLIRAMXIncGen MLIRAMXInterfacesIncGen)
