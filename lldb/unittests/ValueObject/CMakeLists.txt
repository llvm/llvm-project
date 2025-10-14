add_lldb_unittest(LLDBValueObjectTests
  DumpValueObjectOptionsTests.cpp
  DILLexerTests.cpp
  DynamicValueObjectLocalBuffer.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbValueObject
    lldbPluginPlatformLinux
    lldbPluginScriptInterpreterNone
    LLVMTestingSupport
  )
