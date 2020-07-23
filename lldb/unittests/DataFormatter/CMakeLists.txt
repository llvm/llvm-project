add_lldb_unittest(LLDBFormatterTests
  FormatManagerTests.cpp
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
