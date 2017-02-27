if( LLVM_USE_SANITIZE_COVERAGE )
  include_directories(BEFORE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Fuzzer)

  set(LLVM_LINK_COMPONENTS
      AllTargetsAsmPrinters
      AllTargetsDescs
      AllTargetsDisassemblers
      AllTargetsInfos
      MC
      MCDisassembler
      MCParser
      Support
      )
  add_llvm_tool(llvm-mc-disassemble-fuzzer 
                llvm-mc-disassemble-fuzzer.cpp)

  target_link_libraries(llvm-mc-disassemble-fuzzer
                        LLVMFuzzer
                        )
endif()
