if(LLVM_BINDINGS MATCHES "go")
  set(binpath ${CMAKE_BINARY_DIR}/bin/llvm-go${CMAKE_EXECUTABLE_SUFFIX})
  add_custom_command(OUTPUT ${binpath}
    COMMAND ${GO_EXECUTABLE} build -o ${binpath} llvm-go.go
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/llvm-go.go
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Go executable llvm-go")
  add_custom_target(llvm-go ALL DEPENDS ${binpath})
endif()
