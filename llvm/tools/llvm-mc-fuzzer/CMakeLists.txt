if( LLVM_USE_SANITIZE_COVERAGE )
  include_directories(BEFORE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Fuzzer)

  set(LLVM_LINK_COMPONENTS
      AllTargetsDescs
      AllTargetsDisassemblers
      AllTargetsInfos
      MC
      MCDisassembler
      Support
      )
  add_llvm_tool(llvm-mc-fuzzer 
                llvm-mc-fuzzer.cpp)
  target_link_libraries(llvm-mc-fuzzer
                        LLVMFuzzer
                        )
endif()
