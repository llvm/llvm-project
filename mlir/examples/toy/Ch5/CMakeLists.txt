add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
add_public_tablegen_target(ToyCh5CombineIncGen)

add_toy_chapter(toyc-ch5
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp

  DEPENDS
  ToyCh5ShapeInferenceInterfaceIncGen
  ToyCh5OpsIncGen
  ToyCh5CombineIncGen
  MLIRCallOpInterfacesIncGen
  )

include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(toyc-ch5
  PRIVATE
    ${dialect_libs}
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms)
