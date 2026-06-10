# Use LLVM_HEADERS for each `tablegen(LLVM)` in subdirectories
# LLVM_HEADERS_TABLEGEN points to `llvm-min-tblgen`
set(LLVM_TABLEGEN_PROJECT LLVM_HEADERS)

# Read instrumentor runtime helper header to a string for use in
# instrumentor pass
file(READ
  ${CMAKE_CURRENT_SOURCE_DIR}/Transforms/IPO/InstrumentorRuntimeHelper.h
  LLVM_INSTRUMENTOR_RUNTIME_HELPER)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Transforms/IPO/InstrumentorVariables.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/Transforms/IPO/InstrumentorVariables.inc @ONLY)

add_subdirectory(Analysis)
add_subdirectory(CodeGen)
add_subdirectory(IR)
add_subdirectory(Support)
add_subdirectory(Frontend)
add_subdirectory(TargetParser)
