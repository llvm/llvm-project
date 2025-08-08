# We need to include the llvm components we depend on manually, as liblldb does
# not re-export those.
set(LLVM_LINK_COMPONENTS Support)
set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(LLDBDAPOptionsTableGen)

add_lldb_library(lldbDAP
  Breakpoint.cpp
  BreakpointBase.cpp
  CommandPlugins.cpp
  DAP.cpp
  DAPError.cpp
  DAPLog.cpp
  EventHelper.cpp
  ExceptionBreakpoint.cpp
  FifoFiles.cpp
  FunctionBreakpoint.cpp
  InstructionBreakpoint.cpp
  JSONUtils.cpp
  LLDBUtils.cpp
  OutputRedirector.cpp
  ProgressEvent.cpp
  ProtocolUtils.cpp
  RunInTerminal.cpp
  SourceBreakpoint.cpp
  Transport.cpp
  Variables.cpp
  Watchpoint.cpp

  Handler/ResponseHandler.cpp
  Handler/AttachRequestHandler.cpp
  Handler/BreakpointLocationsHandler.cpp
  Handler/CancelRequestHandler.cpp
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
  Handler/WriteMemoryRequestHandler.cpp

  Protocol/DAPTypes.cpp
  Protocol/ProtocolBase.cpp
  Protocol/ProtocolEvents.cpp
  Protocol/ProtocolTypes.cpp
  Protocol/ProtocolRequests.cpp

  LINK_COMPONENTS
    Option
    Support
  LINK_LIBS
    liblldb
    lldbHost
  )

target_include_directories(lldbDAP
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if(LLDB_DAP_WELCOME_MESSAGE)
  target_compile_definitions(lldbDAP
    PRIVATE
    -DLLDB_DAP_WELCOME_MESSAGE=\"${LLDB_DAP_WELCOME_MESSAGE}\")
endif()

add_subdirectory(tool)
