add_lldb_unittest(LLDBFormatterTests
  FormatManagerTests.cpp
  FormattersContainerTest.cpp
  FormatterBytecodeTest.cpp
  StringPrinterTests.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
  )
