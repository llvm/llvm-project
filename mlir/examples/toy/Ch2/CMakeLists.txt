# For a better template to copy, see examples/standalone
add_subdirectory(include)

add_embedded_lit_tests(EmbeddedLitTestsGen 
  "include/toy/TestPasses.td"
  "/disk1/kjain/workspace/llvm-project/mlir/examples/toy/Ch2/jkshtj"
)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_toy_chapter(toyc-ch2
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp

  DEPENDS
  ToyCh2OpsIncGen
  EmbeddedLitTestsGen
  )

include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch2
  PRIVATE
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)
