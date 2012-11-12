add_llvm_library(LLVMX86Desc
  X86AsmBackend.cpp
  X86MCTargetDesc.cpp
  X86MCAsmInfo.cpp
  X86MCCodeEmitter.cpp
  X86MachObjectWriter.cpp
  X86ELFObjectWriter.cpp
  X86WinCOFFObjectWriter.cpp
  )

add_dependencies(LLVMX86Desc X86CommonTableGen)

# Hack: we need to include 'main' target directory to grab private headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/..)
