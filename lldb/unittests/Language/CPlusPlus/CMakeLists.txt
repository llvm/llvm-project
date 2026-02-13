add_lldb_unittest(LanguageCPlusPlusTests
  CPlusPlusLanguageTest.cpp
  LibStdcppTupleTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginCPlusPlusLanguage
    lldbPluginTypeSystemClang
    LLVMTestingSupport
  )
