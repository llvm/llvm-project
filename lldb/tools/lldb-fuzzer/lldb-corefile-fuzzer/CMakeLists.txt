set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_fuzzer(lldb-corefile-fuzzer
  EXCLUDE_FROM_ALL
  lldb-corefile-fuzzer.cpp
  )

if(TARGET lldb-corefile-fuzzer)
  target_include_directories(lldb-corefile-fuzzer PRIVATE ..)
  target_include_directories(lldb-corefile-fuzzer PRIVATE ${LLDB_SOURCE_ROOT})
  target_link_libraries(lldb-corefile-fuzzer
    PRIVATE
    lldbCore
    lldbTarget
    lldbPluginObjectFileELF
    lldbPluginObjectFileMachO
    lldbPluginPlatformLinux
    lldbPluginPlatformMacOSX
    lldbPluginProcessElfCore
    lldbPluginProcessMachCore
    lldbPluginScriptInterpreterNone
    lldbFuzzerUtils
    )

  set(corefile_artifacts
    ${CMAKE_BINARY_DIR}/fuzzer-artifacts/corefile-artifacts)

  add_custom_command(TARGET lldb-corefile-fuzzer PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${corefile_artifacts}
    )

  add_custom_target(fuzz-lldb-corefile
    COMMENT "Running the LLDB corefile fuzzer..."
    WORKING_DIRECTORY ${corefile_artifacts}
    COMMAND $<TARGET_FILE:lldb-corefile-fuzzer>
      -artifact_prefix=corefile-
      -dict=${CMAKE_CURRENT_SOURCE_DIR}/corefile.dict
    USES_TERMINAL
    )
  set_target_properties(fuzz-lldb-corefile PROPERTIES FOLDER "LLDB/Fuzzer")
endif()
