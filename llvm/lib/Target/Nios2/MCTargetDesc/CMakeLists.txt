#MCTargetDesc / CMakeLists.txt
add_llvm_library(LLVMNios2Desc
                 Nios2AsmBackend.cpp
                 Nios2ELFObjectWriter.cpp
                 Nios2MCAsmInfo.cpp
                 Nios2MCExpr.cpp
                 Nios2MCTargetDesc.cpp
                 Nios2TargetStreamer.cpp)

