add_lldb_library(lldbPluginSourceLocatorDebuginfod PLUGIN
        SourceLocatorDebuginfod.cpp

        LINK_LIBS
        lldbCore
        lldbHost
        LLVMDebuginfod
)
