include_directories( ${CMAKE_SOURCE_DIR}/lib/Target/AMDGPU/ )

set(LLVM_LINK_COMPONENTS
  AllTargetsAsmPrinters
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  CodeGen
  MC
  MCDisassembler
  Object
  Support
  Symbolize
  )

add_llvm_tool(amdgpu-objdump
  amdgpu-objdump.cpp
  )
