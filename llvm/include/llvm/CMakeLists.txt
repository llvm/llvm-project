# Use LLVM_HEADERS for each `tablegen(LLVM)` in subdirectories
# LLVM_HEADERS_TABLEGEN points to `llvm-min-tblgen`
set(LLVM_TABLEGEN_PROJECT LLVM_HEADERS)

add_subdirectory(CodeGen)
add_subdirectory(IR)
add_subdirectory(Support)
add_subdirectory(Frontend)
add_subdirectory(TargetParser)

# If we're doing an out-of-tree build, copy a module map for generated
# header files into the build area.
if (NOT "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  configure_file(module.modulemap.build module.modulemap COPYONLY)
endif (NOT "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
