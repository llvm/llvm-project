
build_root=/mnt/data01/zmz/workspace/07ascendnpu/llvm/llvm-project/build
mlir_src_root=/mnt/data01/zmz/workspace/07ascendnpu/llvm/llvm-project/mlir

${build_root}/bin/mlir-tblgen -gen-op-defs ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/