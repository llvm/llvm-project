# MCTargetDesc/CMakeLists.txt
add_llvm_library(LLVMP2Desc
    P2MCAsmInfo.cpp
    P2BaseInfo.cpp
    P2InstPrinter.cpp
    P2ELFObjectWriter.cpp
    P2AsmBackend.cpp
    P2MCCodeEmitter.cpp
    P2MCTargetDesc.cpp
    P2TargetStreamer.cpp
    P2ELFStreamer.cpp
)