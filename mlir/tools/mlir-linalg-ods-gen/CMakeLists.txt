set(LLVM_LINK_COMPONENTS
  Core
  Support
  )
add_llvm_tool(mlir-linalg-ods-gen
  mlir-linalg-ods-gen.cpp
)
llvm_update_compile_flags(mlir-linalg-ods-gen)
target_link_libraries(mlir-linalg-ods-gen PRIVATE
  MLIRSupport
  MLIRIR
  )

set(MLIR_LINALG_ODS_GEN mlir-linalg-ods-gen CACHE
    STRING "Native mlir-linalg-ods-gen executable. Saves building one when cross-compiling.")

set(MLIR_LINALG_ODS_GEN_EXE ${MLIR_LINALG_ODS_GEN} PARENT_SCOPE)
set(MLIR_LINALG_ODS_GEN_TARGET mlir-linalg-ods-gen PARENT_SCOPE)

if(LLVM_USE_HOST_TOOLS)
  if ("${MLIR_LINALG_ODS_GEN_EXE}" STREQUAL mlir-linalg-ods-gen)
    build_native_tool(mlir-linalg-ods-gen MLIR_LINALG_ODS_GEN_EXE DEPENDS mlir-linalg-ods-gen)
    set(MLIR_LINALG_ODS_GEN_EXE ${MLIR_LINALG_ODS_GEN_EXE} PARENT_SCOPE)

    add_custom_target(mlir-linalg-ods-gen-host DEPENDS ${MLIR_LINALG_ODS_GEN_EXE})
    set(MLIR_LINALG_ODS_GEN_TARGET mlir-linalg-ods-gen-host DEPENDS PARENT_SCOPE)

    if(NOT LLVM_BUILD_UTILS)
      set_target_properties(mlir-linalg-ods-gen PROPERTIES EXCLUDE_FROM_ALL ON)
    endif()
  endif()
endif()
