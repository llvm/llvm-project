set(LLVM_OPTIONAL_SOURCES
  CallInterfaces.cpp
  ControlFlowInterfaces.cpp
  DerivedAttributeOpInterface.cpp
  InferTypeOpInterface.cpp
  LoopLikeInterface.cpp
  SideEffectInterfaces.cpp
  ViewLikeInterface.cpp
  )

add_mlir_library(MLIRCallInterfaces
  CallInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRCallInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRControlFlowInterfaces
  ControlFlowInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRControlFlowInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRDerivedAttributeOpInterface
  DerivedAttributeOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRDerivedAttributeOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRInferTypeOpInterface
  InferTypeOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRInferTypeOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRLoopLikeInterface
  LoopLikeInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRLoopLikeInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRSideEffectInterfaces
  SideEffectInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRSideEffectInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRViewLikeInterface
  ViewLikeInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRViewLikeInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
