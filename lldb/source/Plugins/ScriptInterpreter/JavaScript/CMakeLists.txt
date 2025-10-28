if(NOT LLDB_ENABLE_JAVASCRIPT)
  return()
endif()

find_package(V8)

if(NOT V8_FOUND)
  message(FATAL_ERROR "V8 JavaScript engine not found. JavaScript scripting will not be available.")
  return()
endif()

add_lldb_library(lldbPluginScriptInterpreterJavaScript PLUGIN
  JavaScript.cpp
  ScriptInterpreterJavaScript.cpp
  ${CMAKE_BINARY_DIR}/tools/lldb/bindings/javascript/LLDBWrapJavaScript.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbHost
    lldbInterpreter
    lldbTarget
    lldbUtility

  LINK_COMPONENTS
    Support

  CLANG_LIBS
    clangBasic
)

target_include_directories(lldbPluginScriptInterpreterJavaScript PUBLIC
  ${V8_INCLUDE_DIR}
)

# Link against V8
if(V8_LIBRARIES)
  target_link_libraries(lldbPluginScriptInterpreterJavaScript PRIVATE ${V8_LIBRARIES})
endif()
