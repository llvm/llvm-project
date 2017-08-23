if( LLVM_USE_SANITIZE_COVERAGE )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")

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
endif()
