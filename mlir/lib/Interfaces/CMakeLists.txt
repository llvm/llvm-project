set(LLVM_OPTIONAL_SOURCES
  CallInterfaces.cpp
  ControlFlowInterfaces.cpp
  DerivedAttributeOpInterface.cpp
  InferTypeOpInterface.cpp
  LoopLikeInterface.cpp
  SideEffects.cpp
  )

add_mlir_library(MLIRCallInterfaces
  CallInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces
  )
add_dependencies(MLIRCallInterfaces
  MLIRCallInterfacesIncGen
  )
target_link_libraries(MLIRCallInterfaces
  PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRControlFlowInterfaces
  ControlFlowInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces
  )
add_dependencies(MLIRControlFlowInterfaces
  MLIRControlFlowInterfacesIncGen
  )
target_link_libraries(MLIRControlFlowInterfaces
  PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRDerivedAttributeOpInterface
  DerivedAttributeOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces
  )
add_dependencies(MLIRDerivedAttributeOpInterface
  MLIRDerivedAttributeOpInterfaceIncGen
  )
target_link_libraries(MLIRDerivedAttributeOpInterface
  PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRInferTypeOpInterface
  InferTypeOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces
  )
add_dependencies(MLIRInferTypeOpInterface
  MLIRInferTypeOpInterfaceIncGen
  )
target_link_libraries(MLIRInferTypeOpInterface
  PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRLoopLikeInterface
  LoopLikeInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces
  )
add_dependencies(MLIRLoopLikeInterface
  MLIRLoopLikeInterfaceIncGen
  )
target_link_libraries(MLIRLoopLikeInterface
  PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRSideEffects
  SideEffects.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces
  )
add_dependencies(MLIRSideEffects
  MLIRSideEffectOpInterfacesIncGen
  )
target_link_libraries(MLIRSideEffects
  PUBLIC
  MLIRIR
  )
