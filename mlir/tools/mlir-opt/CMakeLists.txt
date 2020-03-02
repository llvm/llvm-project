set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

set(LIB_LIBS
  MLIRAnalysis
  MLIRLLVMIR
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
)
add_llvm_library(MLIRMlirOptMain
  mlir-opt.cpp
)
target_link_libraries(MLIRMlirOptMain
  ${LIB_LIBS}
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRLoopOpsTransforms
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerSupport
  MLIRQuantizerTransforms
  MLIRSPIRV
  MLIRSPIRVTestPasses
  MLIRSPIRVTransforms
  MLIRTransforms
  MLIRTransformUtils
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRSupport
  MLIRIR
  MLIROptLib
  LLVMSupport
  LLVMCore
  LLVMAsmParser
)

add_llvm_tool(mlir-opt
 mlir-opt.cpp
)
# Manually expand the target library, since our MLIR libraries
# aren't plugged into the LLVM dependency tracking. If we don't
# do this then we can't insert the CodeGen library after ourselves
llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)
# Prepend LLVM in front of every target, this is how the library
# are named with CMake
SET(targets_to_link)
FOREACH(t ${TARGET_LIBS})
  LIST(APPEND targets_to_link "LLVM${t}")
ENDFOREACH(t)

llvm_update_compile_flags(mlir-opt)
target_link_libraries(mlir-opt PRIVATE ${LIBS} ${targets_to_link})
