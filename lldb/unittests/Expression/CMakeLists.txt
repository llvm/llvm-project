set(SWIFT_SOURCES SwiftParserTest.cpp)
set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
endif()

add_lldb_unittest(ExpressionTests
  ClangParserTest.cpp
  DiagnosticManagerTest.cpp
  DWARFExpressionTest.cpp
  CppModuleConfigurationTest.cpp

  ${SWIFT_SOURCES}

  LINK_LIBS
    lldbCore
    lldbPluginExpressionParserClang
    lldbUtility
    lldbUtilityHelpers
    LLVMTestingSupport
  )
