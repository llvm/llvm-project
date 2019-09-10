if (LLVM_BUILD_UTILS)
  add_custom_target(llvm-locstats ALL
    COMMAND  ${CMAKE_COMMAND} -E copy ${LLVM_MAIN_SRC_DIR}/utils/llvm-locstats/llvm-locstats.py ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llvm-locstats
    )
  set_target_properties(llvm-locstats PROPERTIES FOLDER "Utils")
endif()
