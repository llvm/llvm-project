add_library(MathTest STATIC
  CommandLineExtras.cpp DeviceContext.cpp DeviceResources.cpp ErrorHandling.cpp TestConfig.cpp)

target_include_directories(MathTest PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")

if(NOT LLVM_REQUIRES_RTTI)
  target_compile_options(MathTest PUBLIC -fno-rtti)
endif()

include(FindLibcCommonUtils)
target_link_libraries(MathTest PUBLIC LLVMOffload LLVMSupport LLVMDemangle llvm-libc-common-utilities)
