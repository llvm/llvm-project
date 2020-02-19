add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
add_public_tablegen_target(ToyCh6CombineIncGen)

add_toy_chapter(toyc-ch6
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  )

add_dependencies(toyc-ch6 ToyCh6ShapeInferenceInterfaceIncGen)
add_dependencies(toyc-ch6 ToyCh6OpsIncGen)
add_dependencies(toyc-ch6 ToyCh6CombineIncGen)
add_dependencies(toyc-ch6 MLIRCallOpInterfacesIncGen)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch6
  PRIVATE
    MLIRAllDialects
    MLIRAffineToStandard
    MLIRAnalysis
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMIR
    MLIRLoopToStandard
    MLIRParser
    MLIRPass
    MLIRStandardToLLVM
    MLIRTargetLLVMIR
    MLIRTransforms
    )

