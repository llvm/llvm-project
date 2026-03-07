set(LLVM_LINK_COMPONENTS Support)
set(LIBS MLIRSupport)

add_llvm_executable(mlir-src-sharder mlir-src-sharder.cpp)

set_target_properties(mlir-src-sharder PROPERTIES FOLDER "MLIR/Tablegenning")
target_link_libraries(mlir-src-sharder PRIVATE ${LIBS})

set(MLIR_SRC_SHARDER_TABLEGEN_EXE mlir-src-sharder PARENT_SCOPE)

mlir_check_all_link_libraries(mlir-src-sharder)
