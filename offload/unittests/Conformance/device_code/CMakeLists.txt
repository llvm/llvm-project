add_offload_test_device_code(LLVMLibm.c llvm-libm -stdlib -fno-builtin)

set(OFFLOAD_CONFORMANCE_DEVICE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
