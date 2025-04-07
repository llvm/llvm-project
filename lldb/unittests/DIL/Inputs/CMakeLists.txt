# Build `test_binary.cc` and put the binary in the Inputs folder, 
# allowing `lldb_private::GetInputFilePath` to find it.
# Projects that must be enabled: clang;lldb;lld
# Runtimes that must be enabled: libcxx;libcxxabi;libunwind
if ("libcxx" IN_LIST LLVM_ENABLE_RUNTIMES)
  if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
    set(LIBCXX_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE})
    set(LIBCXX_GENERATED_INCLUDE_DIR "${LLVM_BINARY_DIR}/include/c++/v1")
    set(LIBCXX_GENERATED_INCLUDE_TARGET_DIR "${LLVM_BINARY_DIR}/include/${LLVM_DEFAULT_TARGET_TRIPLE}/c++/v1")
  else()
    set(LIBCXX_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBCXX_LIBDIR_SUFFIX})
    set(LIBCXX_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include/c++/v1")
  endif()

  if(DEFINED LIBCXX_GENERATED_INCLUDE_TARGET_DIR)
    set(INCLUDE_TARGET_DIR_OPTION "-cxx-isystem" "${LIBCXX_GENERATED_INCLUDE_TARGET_DIR}")
  endif()

  get_target_property(EXE_PATH DILTests RUNTIME_OUTPUT_DIRECTORY)
  add_custom_command(
    OUTPUT test_binary.bin
    COMMAND $<TARGET_FILE:clang> ${CMAKE_CURRENT_SOURCE_DIR}/test_binary.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_extern.cpp
            -O0 -g -std=c++17 -fuse-ld=lld -B$<TARGET_FILE:lld>
            -nostdlib++ -nostdinc++ -cxx-isystem ${LIBCXX_GENERATED_INCLUDE_DIR}
            ${INCLUDE_TARGET_DIR_OPTION}
            -L${LIBCXX_LIBRARY_DIR} -Wl,-rpath,${LIBCXX_LIBRARY_DIR} -lc++
            -o ${EXE_PATH}/Inputs/test_binary.bin
    DEPENDS test_binary.cpp test_extern.cpp clang lld
  )
  add_custom_target(test_binary
    DEPENDS test_binary.bin
  )
else()
  message(FATAL_ERROR "libcxx runtime must be enabled.")
endif()
