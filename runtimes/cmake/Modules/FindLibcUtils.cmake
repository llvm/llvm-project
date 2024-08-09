add_library(llvm-libc-shared-utilities INTERFACE)
# TODO: Reorganize the libc shared section so that it can be included without
# adding the root "libc" directory to the include path.
# TODO: Find a better way to solve the problem that ${CMAKE_SOURCE_DIR} is
# rooted in the runtimes directory, which is why we need the ".."
target_include_directories(llvm-libc-shared-utilities INTERFACE ${CMAKE_SOURCE_DIR}/../libc) 
target_compile_definitions(llvm-libc-shared-utilities INTERFACE LIBC_NAMESPACE=__llvm_libc_shared_utils)
target_compile_features(llvm-libc-shared-utilities INTERFACE cxx_std_17)
