add_lldb_tool(yaml2macho-core
    CoreSpec.cpp
    LCNoteWriter.cpp
    MemoryWriter.cpp
    ThreadWriter.cpp
    Utility.cpp
    yaml2macho.cpp

    LINK_COMPONENTS
      Support
    LINK_LIBS
      lldbUtility
      ${LLDB_SYSTEM_LIBS}
)
