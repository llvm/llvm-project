set(LLVM_OPTIONAL_SOURCES
  CallInterfaces.cpp
  ControlFlowInterfaces.cpp
  DerivedAttributeOpInterface.cpp
  InferTypeOpInterface.cpp
  SideEffects.cpp
  )

add_llvm_library(MLIRCallInterfaces
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

add_llvm_library(MLIRControlFlowInterfaces
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

add_llvm_library(MLIRDerivedAttributeOpInterface
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

add_llvm_library(MLIRInferTypeOpInterface
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

add_llvm_library(MLIRSideEffects
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
