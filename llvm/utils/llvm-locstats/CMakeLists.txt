if (LLVM_BUILD_UTILS AND LLVM_BUILD_TOOLS)
  add_custom_command(
    OUTPUT ${LLVM_TOOLS_BINARY_DIR}/llvm-locstats
    DEPENDS ${LLVM_MAIN_SRC_DIR}/utils/llvm-locstats/llvm-locstats.py
    COMMAND ${CMAKE_COMMAND} -E copy ${LLVM_MAIN_SRC_DIR}/utils/llvm-locstats/llvm-locstats.py ${LLVM_TOOLS_BINARY_DIR}/llvm-locstats
    COMMENT "Copying llvm-locstats into ${LLVM_TOOLS_BINARY_DIR}"
    )
  add_custom_target(llvm-locstats ALL
    DEPENDS ${LLVM_TOOLS_BINARY_DIR}/llvm-locstats
    )
  set_target_properties(llvm-locstats PROPERTIES FOLDER "Tools")
endif()
