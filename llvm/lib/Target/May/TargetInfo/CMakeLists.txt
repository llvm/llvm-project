add_llvm_component_library(LLVMMayInfo
        MayTargetInfo.cpp

        LINK_COMPONENTS
        MC
        Support

        ADD_TO_COMPONENT
        May
)