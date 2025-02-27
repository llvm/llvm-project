if (HAVE_LIBPTHREAD)
  list(APPEND extra_libs pthread)
endif ()


if(APPLE)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lldb-dap-Info.plist.in
    ${CMAKE_CURRENT_BINARY_DIR}/lldb-dap-Info.plist
    )
  # Inline info plist in binary (use target_link_options for this as soon as CMake 3.13 is available)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-sectcreate,__TEXT,__info_plist,${CMAKE_CURRENT_BINARY_DIR}/lldb-dap-Info.plist")
endif()

# We need to include the llvm components we depend on manually, as liblldb does
# not re-export those.
set(LLVM_LINK_COMPONENTS Support)
set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(LLDBDAPOptionsTableGen)
add_lldb_tool(lldb-dap
  lldb-dap.cpp
  Breakpoint.cpp
  BreakpointBase.cpp
  DAP.cpp
  EventHelper.cpp
  ExceptionBreakpoint.cpp
  FifoFiles.cpp
  FunctionBreakpoint.cpp
  InstructionBreakpoint.cpp
  IOStream.cpp
  JSONUtils.cpp
  LLDBUtils.cpp
  OutputRedirector.cpp
  ProgressEvent.cpp
  RunInTerminal.cpp
  SourceBreakpoint.cpp
  Protocol.cpp
  Watchpoint.cpp

  Handler/ResponseHandler.cpp
  Handler/AttachRequestHandler.cpp
  Handler/BreakpointLocationsHandler.cpp
  Handler/CompileUnitsRequestHandler.cpp
  Handler/CompletionsHandler.cpp
  Handler/ConfigurationDoneRequestHandler.cpp
  Handler/ContinueRequestHandler.cpp
  Handler/DataBreakpointInfoRequestHandler.cpp
  Handler/DisassembleRequestHandler.cpp
  Handler/DisconnectRequestHandler.cpp
  Handler/EvaluateRequestHandler.cpp
  Handler/ExceptionInfoRequestHandler.cpp
  Handler/InitializeRequestHandler.cpp
  Handler/LaunchRequestHandler.cpp
  Handler/LocationsRequestHandler.cpp
  Handler/ModulesRequestHandler.cpp
  Handler/NextRequestHandler.cpp
  Handler/PauseRequestHandler.cpp
  Handler/ReadMemoryRequestHandler.cpp
  Handler/RequestHandler.cpp
  Handler/RestartRequestHandler.cpp
  Handler/ScopesRequestHandler.cpp
  Handler/SetBreakpointsRequestHandler.cpp
  Handler/SetDataBreakpointsRequestHandler.cpp
  Handler/SetExceptionBreakpointsRequestHandler.cpp
  Handler/SetFunctionBreakpointsRequestHandler.cpp
  Handler/SetInstructionBreakpointsRequestHandler.cpp
  Handler/SetVariableRequestHandler.cpp
  Handler/SourceRequestHandler.cpp
  Handler/StackTraceRequestHandler.cpp
  Handler/StepInRequestHandler.cpp
  Handler/StepInTargetsRequestHandler.cpp
  Handler/StepOutRequestHandler.cpp
  Handler/TestGetTargetBreakpointsRequestHandler.cpp
  Handler/ThreadsRequestHandler.cpp
  Handler/VariablesRequestHandler.cpp

  LINK_LIBS
    liblldb
    lldbHost
    ${extra_libs}

  LINK_COMPONENTS
    Option
    Support
  )

target_include_directories(lldb-dap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(LLDB_DAP_WELCOME_MESSAGE)
  target_compile_definitions(lldb-dap
    PRIVATE
    -DLLDB_DAP_WELCOME_MESSAGE=\"${LLDB_DAP_WELCOME_MESSAGE}\")
endif()

if(LLDB_BUILD_FRAMEWORK)
  # In the build-tree, we know the exact path to the framework directory.
  # The installed framework can be in different locations.
  lldb_setup_rpaths(lldb-dap
    BUILD_RPATH
      "${LLDB_FRAMEWORK_ABSOLUTE_BUILD_DIR}"
    INSTALL_RPATH
      "@loader_path/../../../SharedFrameworks"
      "@loader_path/../../System/Library/PrivateFrameworks"
      "@loader_path/../../Library/PrivateFrameworks"
  )
endif()
