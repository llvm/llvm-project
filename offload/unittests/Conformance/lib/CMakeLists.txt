add_library(MathTest STATIC
  CommandLineExtras.cpp DeviceContext.cpp DeviceResources.cpp ErrorHandling.cpp TestConfig.cpp)

target_include_directories(MathTest PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")

include(FindLibcCommonUtils)
target_link_libraries(MathTest PUBLIC LLVMOffload LLVMSupport LLVMDemangle llvm-libc-common-utilities)
