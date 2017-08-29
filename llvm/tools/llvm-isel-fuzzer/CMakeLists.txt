if( LLVM_USE_SANITIZE_COVERAGE )
  include_directories(BEFORE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Fuzzer)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")
  set(LLVM_LINK_COMPONENTS
      ${LLVM_TARGETS_TO_BUILD}
      Analysis
      AsmPrinter
      CodeGen
      Core
      FuzzMutate
      IRReader
      MC
      ScalarOpts
      SelectionDAG
      Support
      Target
  )
  add_llvm_tool(llvm-isel-fuzzer
                llvm-isel-fuzzer.cpp)
endif()
