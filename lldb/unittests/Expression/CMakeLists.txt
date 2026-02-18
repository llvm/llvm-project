if ("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(LINK_COMPONENTS_AARCH AArch64)
  set(LINK_LIBS_AARCH lldbPluginABIAArch64)
endif()

add_lldb_unittest(ExpressionTests
  ClangParserTest.cpp
  ClangExpressionDeclMapTest.cpp
  DiagnosticManagerTest.cpp
  DWARFExpressionTest.cpp
  CppModuleConfigurationTest.cpp
  ExpressionTest.cpp
  ValueMatcher.cpp

  LINK_COMPONENTS
    Support
    ${LINK_COMPONENTS_AARCH}
  LINK_LIBS
    lldbCore
    lldbPluginObjectFileELF
    lldbPluginObjectFileWasm
    lldbPluginSymbolVendorWasm
    lldbPluginPlatformLinux
    lldbPluginExpressionParserClang
    lldbPluginTypeSystemClang
    lldbUtility
    lldbUtilityHelpers
    lldbSymbolHelpers
    LLVMTestingSupport
    ${LINK_LIBS_AARCH}
  )

if ("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(ExpressionTests PRIVATE ARCH_AARCH64)
endif()
