set(LLVM_LINK_COMPONENTS
  Support
  )


set(LIBS
  MLIRParser
  MLIRSupport
  MLIRIR
)

add_mlir_tool(mlir-cat
  mlir-cat.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${LIBS}
  )
target_link_libraries(mlir-cat PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-cat)
mlir_check_all_link_libraries(mlir-cat)

list(APPEND LIBS
  MLIROptLib
  MLIRPass
)
add_mlir_tool(mlir-minimal-opt
  mlir-minimal-opt.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${LIBS}
  )
target_link_libraries(mlir-minimal-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-minimal-opt)
mlir_check_all_link_libraries(mlir-minimal-opt)


list(APPEND LIBS
  MLIROptLib
  MLIRPass
  MLIRTransforms
)
add_mlir_tool(mlir-minimal-opt-canonicalize
  mlir-minimal-opt-canonicalize.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${LIBS}
  )
target_link_libraries(mlir-minimal-opt-canonicalize PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-minimal-opt-canonicalize)
mlir_check_all_link_libraries(mlir-minimal-opt-canonicalize)

