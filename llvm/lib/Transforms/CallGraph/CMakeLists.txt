add_llvm_library(LLVMCallGraph MODULE
    CallGraph.cpp

    #DEPENDS
    #intrinsics_gen
    
    PLUGIN_TOOL
    opt
    )
