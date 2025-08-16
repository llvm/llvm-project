add_lldb_tool(yaml2macho-core
    main.cpp
    yaml2corespec.cpp
    LCNoteWriter.cpp
    MemoryWriter.cpp
    ThreadWriter.cpp
    Utility.cpp

    LINK_COMPONENTS
      Support
    LINK_LIBS
      ${LLDB_SYSTEM_LIBS}
)
