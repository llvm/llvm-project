add_llvm_component_library(LLVMP2AsmParser
    P2AsmParser.cpp

    LINK_COMPONENTS
    MC
    MCParser
    P2Desc
    P2Info
    Support

    ADD_TO_COMPONENT
    P2
)