set(LLVM_OPTIONAL_SOURCES
  CallInterfaces.cpp
  CastInterfaces.cpp
  ControlFlowInterfaces.cpp
  CopyOpInterface.cpp
  DataLayoutInterfaces.cpp
  DerivedAttributeOpInterface.cpp
  DestinationStyleOpInterface.cpp
  FunctionImplementation.cpp
  FunctionInterfaces.cpp
  InferIntRangeInterface.cpp
  InferTypeOpInterface.cpp
  LoopLikeInterface.cpp
  MemorySlotInterfaces.cpp
  ParallelCombiningOpInterface.cpp
  RuntimeVerifiableOpInterface.cpp
  ShapedOpInterfaces.cpp
  SideEffectInterfaces.cpp
  TilingInterface.cpp
  ValueBoundsOpInterface.cpp
  VectorInterfaces.cpp
  ViewLikeInterface.cpp
  )

function(add_mlir_interface_library name)
  add_mlir_library(MLIR${name}
    ${name}.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

    DEPENDS
    MLIR${name}IncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
endfunction(add_mlir_interface_library)


add_mlir_interface_library(CallInterfaces)
add_mlir_interface_library(CastInterfaces)
add_mlir_interface_library(ControlFlowInterfaces)
add_mlir_interface_library(CopyOpInterface)
add_mlir_interface_library(DataLayoutInterfaces)
add_mlir_interface_library(DerivedAttributeOpInterface)
add_mlir_interface_library(DestinationStyleOpInterface)

add_mlir_library(MLIRFunctionInterfaces
  FunctionInterfaces.cpp
  FunctionImplementation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRFunctionInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_interface_library(InferIntRangeInterface)
add_mlir_interface_library(InferTypeOpInterface)

add_mlir_library(MLIRLoopLikeInterface
  LoopLikeInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRLoopLikeInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFunctionInterfaces
)

add_mlir_interface_library(MemorySlotInterfaces)
add_mlir_interface_library(ParallelCombiningOpInterface)
add_mlir_interface_library(RuntimeVerifiableOpInterface)
add_mlir_interface_library(ShapedOpInterfaces)
add_mlir_interface_library(SideEffectInterfaces)
add_mlir_interface_library(TilingInterface)
add_mlir_interface_library(VectorInterfaces)
add_mlir_interface_library(ViewLikeInterface)

add_mlir_library(MLIRValueBoundsOpInterface
  ValueBoundsOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRDestinationStyleOpInterface
  MLIRValueBoundsOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRDestinationStyleOpInterface
  MLIRIR
  )

add_subdirectory(Utils)
