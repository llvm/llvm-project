set(LLVM_LINK_COMPONENTS
  Support
  )
  
include(CMakeDependentOption)
include(GNUInstallDirs)

set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include/ ) # --includedir
set(CLANG_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../clang/include/ ) # --includedir
set(CLANG_INCLUDE_DIR_INC ${CMAKE_BINARY_DIR}/tools/clang/include/ ) # --includedir
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include/)
include_directories(SYSTEM ${CLANG_INCLUDE_DIR})
include_directories(SYSTEM ${CLANG_INCLUDE_DIR_INC})
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})


include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)

# Make sure that our source directory is on the current cmake module path so
# that we can include cmake files from this directory.
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

include(CMakeParseArguments)
include(AddOBS)

include_directories(include/)
#add_obs_executable(obstest 
#    obs-ir/obs.cpp 
#    obs-ir/Dialect.cpp
#    obs-ir/MLIRGen.cpp
#    parser/AST.cpp
#  DEPENDS
#    OBSOpGen
#)

#target_link_libraries(obstest
#  PRIVATE
#  MLIRAnalysis
#  MLIRFunctionInterfaces
#  MLIRIR
#  MLIRParser
#  MLIRSideEffectInterfaces
#  MLIRTransforms
#)


add_obs_executable(codegen
    codegen/OBSGen.cpp
    codegen/CodeGenAction.cpp
    codegen/CodeGen.cpp
  DEPENDS
    OBSOpGen
)

target_link_libraries(codegen
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
)


