add_llvm_component_library(LLVMDPUDisassembler
        DPUDisassembler.cpp
        DPUInstructionDecoder.cpp
        DPUOperandDecoder.cpp

        LINK_COMPONENTS
        MC
        MCDisassembler
        DPUInfo
        Support

        ADD_TO_COMPONENT
        DPU
        )
