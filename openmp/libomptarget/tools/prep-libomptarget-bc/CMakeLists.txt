set(LLVM_LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  IRReader
  Linker
  Object
  Support
  TransformUtils
  IPO
  )
add_llvm_tool(prep-libomptarget-bc prep-libomptarget-bc.cpp)
llvm_update_compile_flags(prep-libomptarget-bc)
include_directories( ${LLVM_INCLUDE_DIR} ${LLVM_MAIN_INCLUDE_DIR})
