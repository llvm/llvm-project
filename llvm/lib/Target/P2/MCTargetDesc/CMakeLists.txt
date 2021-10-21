# MCTargetDesc/CMakeLists.txt
add_llvm_component_library(LLVMP2Desc
    P2MCAsmInfo.cpp
    P2BaseInfo.cpp
    P2InstPrinter.cpp
    P2ELFObjectWriter.cpp
    P2AsmBackend.cpp
    P2MCCodeEmitter.cpp
    P2MCTargetDesc.cpp
    P2TargetStreamer.cpp
    P2ELFStreamer.cpp

    LINK_COMPONENTS
    MC
    P2Info
    Support
  
    ADD_TO_COMPONENT
    P2
)

add_dependencies(LLVMP2Desc P2CommonTableGen)