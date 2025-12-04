export PATH=/mnt/data01/zmz/workspace/07ascendnpu/llvm/llvm-project/build/bin/:$PATH

# toyc-ch3 ./invalid.mlir -emit=mlir  2>&1 

toyc-ch3 ./scalar.toy -emit=mlir 2>&1 # | FileCheck ./scalar.toy
toyc-ch3 ./scalar.toy -emit=mlir 2>&1 | FileCheck ./scalar.toy

# toyc-ch3 ./codegen.toy -emit=mlir -mlir-print-debuginfo 2>&1 # | FileCheck ./codegen.toy

# toyc-ch3 ./codegen.toy -emit=error 2>&1 # | FileCheck ./codegen.toy

toyc-ch3 ./codegen.toy -emit=mlir 2>&1 | FileCheck ./codegen.toy

toyc-ch3 ./transpose_transpose.toy -emit=mlir -opt       # 添加优化选项，可以执行SimplifyRedundantTranspose，将两个transpose给消除掉

toyc-ch3 ./transpose_transpose.toy -emit=mlir

toyc-ch3 ./trivial_reshape.toy -emit=mlir
toyc-ch3 ./trivial_reshape.toy -emit=mlir -opt

