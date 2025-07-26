add_library(MathTest STATIC DeviceContext.cpp DeviceResources.cpp ErrorHandling.cpp)

include(FindLibcCommonUtils)

target_include_directories(MathTest PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(MathTest PUBLIC -fno-rtti)
target_link_libraries(MathTest PUBLIC LLVMOffload LLVMSupport LLVMDemangle llvm-libc-common-utilities)
