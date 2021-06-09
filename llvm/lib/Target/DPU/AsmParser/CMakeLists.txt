add_llvm_component_library(LLVMDPUAsmParser
        DPUAsmParser.cpp

        LINK_COMPONENTS
        MC 
        MCParser
        DPUDesc
        DPUInfo
        Support

        ADD_TO_COMPONENT
        DPU
  )
