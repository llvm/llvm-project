add_lldb_unittest(LLDBFormatterTests
  FormatManagerTests.cpp
  FormattersContainerTest.cpp
  FormatterBytecodeTest.cpp
  StringPrinterTests.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility

  LINK_COMPONENTS
    Support
  )
