if( LLVM_USE_SANITIZE_COVERAGE )
  include_directories(BEFORE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Fuzzer)

  set(LLVM_LINK_COMPONENTS
      AllTargetsAsmPrinters
      AllTargetsAsmParsers
      AllTargetsDescs
      AllTargetsInfos
      MC
      MCParser
      Support
      )
  add_llvm_tool(llvm-mc-assemble-fuzzer 
                llvm-mc-assemble-fuzzer.cpp)
  target_link_libraries(llvm-mc-assemble-fuzzer
                        LLVMFuzzer
                        )
endif()
