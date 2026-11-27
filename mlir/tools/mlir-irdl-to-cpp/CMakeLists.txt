add_llvm_executable(mlir-irdl-to-cpp
    mlir-irdl-to-cpp.cpp
  )
mlir_target_link_libraries(mlir-irdl-to-cpp
  PRIVATE
  MLIRTargetIRDLToCpp
  )

setup_host_tool(mlir-irdl-to-cpp MLIR_IRDL_TO_CPP MLIR_IRDL_TO_CPP_EXE MLIR_IRDL_TO_CPP_TARGET)

if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(TARGETS mlir-irdl-to-cpp
          RUNTIME DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
          COMPONENT mlir-irdl-to-cpp)
  if(NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-mlir-irdl-to-cpp
                             DEPENDS mlir-irdl-to-cpp
                             COMPONENT mlir-irdl-to-cpp)
  endif()
endif()
