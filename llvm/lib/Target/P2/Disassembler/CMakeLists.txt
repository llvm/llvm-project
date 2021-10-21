add_llvm_component_library(LLVMP2Disassembler
    P2Disassembler.cpp

    LINK_COMPONENTS
    MCDisassembler
    P2Info
    Support

    ADD_TO_COMPONENT
    P2
)

