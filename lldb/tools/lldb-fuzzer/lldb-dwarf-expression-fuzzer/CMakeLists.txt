set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_fuzzer(lldb-dwarf-expression-fuzzer
  EXCLUDE_FROM_ALL
  lldb-dwarf-expression-fuzzer.cpp
  )

if(TARGET lldb-dwarf-expression-fuzzer)
  target_include_directories(lldb-dwarf-expression-fuzzer PRIVATE ..)
  target_include_directories(lldb-dwarf-expression-fuzzer PRIVATE ${LLDB_SOURCE_ROOT})
  target_link_libraries(lldb-dwarf-expression-fuzzer
    PRIVATE
    lldbCore
    lldbPluginExpressionParserClang
    lldbPluginPlatformLinux
    lldbPluginTypeSystemClang
    lldbFuzzerUtils
    )

  add_custom_command(TARGET lldb-dwarf-expression-fuzzer PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/fuzzer-artifacts/dwarf-expression-artifacts
    )

  add_custom_target(fuzz-lldb-dwarf-expression
    COMMENT "Running the LLDB DWARF expression evaluator fuzzer..."
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/fuzzer-artifacts/dwarf-expression-artifacts
    COMMAND $<TARGET_FILE:lldb-dwarf-expression-fuzzer> -artifact_prefix=dwarf-expression-
    USES_TERMINAL
    )
  set_target_properties(fuzz-lldb-dwarf-expression PROPERTIES FOLDER "LLDB/Fuzzer")
endif()
