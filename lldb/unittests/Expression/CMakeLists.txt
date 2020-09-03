set(SWIFT_SOURCES SwiftParserTest.cpp)
set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
endif()

add_lldb_unittest(ExpressionTests
  ClangParserTest.cpp
  ClangExpressionDeclMapTest.cpp
  DiagnosticManagerTest.cpp
  DWARFExpressionTest.cpp
  CppModuleConfigurationTest.cpp
  ClangParserTest.cpp

  ${SWIFT_SOURCES}

  LINK_LIBS
    lldbCore
    lldbPluginExpressionParserClang
    lldbPluginTypeSystemClang
    lldbUtility
    lldbUtilityHelpers
    LLVMTestingSupport
  )
