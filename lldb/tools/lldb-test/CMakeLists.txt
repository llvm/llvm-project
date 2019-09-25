get_property(LLDB_ALL_PLUGINS GLOBAL PROPERTY LLDB_PLUGINS)

add_lldb_tool(lldb-test
  FormatUtil.cpp
  lldb-test.cpp
  SystemInitializerTest.cpp

  LINK_LIBS
    lldbBase
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInitialization
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    ${LLDB_ALL_PLUGINS}
    ${host_lib}

  LINK_COMPONENTS
    Support
  )
  
foreach (t ${LLVM_TARGETS_TO_BUILD})
  set_property(SOURCE SystemInitializerTest.cpp APPEND PROPERTY
    COMPILE_DEFINITIONS "LLVM_TARGET_${t}_BUILT")
endforeach()

include_directories(${LLDB_SOURCE_DIR}/source)
