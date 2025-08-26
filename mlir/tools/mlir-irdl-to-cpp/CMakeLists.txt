add_llvm_executable(mlir-irdl-to-cpp
    mlir-irdl-to-cpp.cpp
  )
mlir_target_link_libraries(mlir-irdl-to-cpp
  PRIVATE
  MLIRTargetIRDLToCpp
  )

# Set up a native build when cross-compiling.
if(LLVM_USE_HOST_TOOLS)
  build_native_tool(
    mlir-irdl-to-cpp
    MLIR_IRDL_TO_CPP_EXE

    # Native tool must depend on target tool so that the native tool gets
    # properly rebuilt when the target tool changes.
    DEPENDS mlir-irdl-to-cpp
  )
  add_custom_target(mlir-irdl-to-cpp-host DEPENDS ${MLIR_IRDL_TO_CPP_EXE})
  set(MLIR_IRDL_TO_CPP_TARGET mlir-irdl-to-cpp-host)
else()
  set(MLIR_IRDL_TO_CPP_EXE $<TARGET_FILE:mlir-irdl-to-cpp>)
  set(MLIR_IRDL_TO_CPP_TARGET mlir-irdl-to-cpp)
endif()

# Save the executable path and target name to the cache to expose it globally.
set(MLIR_IRDL_TO_CPP_EXE "${MLIR_IRDL_TO_CPP_EXE}" CACHE INTERNAL "")
set(MLIR_IRDL_TO_CPP_TARGET "${MLIR_IRDL_TO_CPP_TARGET}" CACHE INTERNAL "")
