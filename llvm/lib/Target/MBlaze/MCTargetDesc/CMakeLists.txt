add_llvm_library(LLVMMBlazeDesc
  MBlazeAsmBackend.cpp
  MBlazeMCAsmInfo.cpp
  MBlazeMCCodeEmitter.cpp
  MBlazeMCTargetDesc.cpp
  MBlazeELFObjectWriter.cpp
  )

add_dependencies(LLVMMBlazeDesc MBlazeCommonTableGen)
