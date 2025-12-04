export PATH=/mnt/data01/zmz/workspace/07ascendnpu/llvm/llvm-project/build/bin/:$PATH

toyc-ch1 ./empty.toy -emit=ast 2>&1 | FileCheck ./empty.toy

toyc-ch1 ./ast.toy -emit=ast 2>&1 | FileCheck ./ast.toy

toyc-ch1 ./ast.toy -emit=ast