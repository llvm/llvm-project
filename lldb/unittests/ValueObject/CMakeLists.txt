add_lldb_unittest(LLDBValueObjectTests
  DumpValueObjectOptionsTests.cpp
  DILLexerTests.cpp

  LINK_LIBS
    lldbValueObject
    lldbPluginPlatformLinux
    lldbPluginScriptInterpreterNone
    LLVMTestingSupport

  LINK_COMPONENTS
    Support
  )
