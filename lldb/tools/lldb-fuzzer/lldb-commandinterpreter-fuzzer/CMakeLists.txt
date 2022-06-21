set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_fuzzer(lldb-commandinterpreter-fuzzer
  EXCLUDE_FROM_ALL
  lldb-commandinterpreter-fuzzer.cpp
  )

if(TARGET lldb-commandinterpreter-fuzzer)
  target_include_directories(lldb-commandinterpreter-fuzzer PRIVATE ..)
  target_link_libraries(lldb-commandinterpreter-fuzzer
    PRIVATE
    liblldb
    )

  # This will create a directory specifically for the fuzzer's artifacts, go to that
  # directory and run the fuzzer from there. When the fuzzer exits the input
  # artifact that caused it to exit will be written to a directory within the
  # build directory
  add_custom_target(fuzz-lldb-commandinterpreter
    COMMENT "Running the LLDB command interpreter fuzzer..."
    COMMAND mkdir -p ${CMAKE_BINARY_DIR}/fuzzer-artifacts/commandinterpreter-artifacts &&
    cd ${CMAKE_BINARY_DIR}/fuzzer-artifacts/commandinterpreter-artifacts
    && $<TARGET_FILE:lldb-commandinterpreter-fuzzer> -dict=${CMAKE_CURRENT_SOURCE_DIR}/inputdictionary.txt  -only_ascii=1 -artifact_prefix=commandinterpreter-
    USES_TERMINAL
    )
endif()
