set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_llvm_executable(mlir-sdbm-api-test
  sdbm-api-test.cpp
)

llvm_update_compile_flags(mlir-sdbm-api-test)

target_link_libraries(mlir-sdbm-api-test
  PRIVATE
  MLIRIR
  MLIRSDBM
  MLIRSupport
)

target_include_directories(mlir-sdbm-api-test PRIVATE ..)
