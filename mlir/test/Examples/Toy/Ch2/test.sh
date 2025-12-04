export PATH=/mnt/data01/zmz/workspace/07ascendnpu/llvm/llvm-project/build/bin/:$PATH

toyc-ch2 ./invalid.mlir -emit=mlir 2>&1 

toyc-ch2 ./scalar.toy -emit=mlir 2>&1 # | FileCheck ./scalar.toy

toyc-ch2 ./codegen.toy -emit=mlir -mlir-print-debuginfo 2>&1 # | FileCheck ./codegen.toy

# toyc-ch2 ./codegen.toy -emit=error 2>&1 # | FileCheck ./codegen.toy
