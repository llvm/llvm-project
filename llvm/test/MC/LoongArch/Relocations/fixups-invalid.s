# RUN: not llvm-mc --filetype=obj %s --triple=loongarch32 -o /dev/null 2>&1 \
# RUN:     | FileCheck %s
# RUN: not llvm-mc --filetype=obj %s --triple=loongarch64 -o /dev/null 2>&1 \
# RUN:     | FileCheck %s

.byte foo   # CHECK: [[#@LINE]]:7: error: 1-byte data relocations not supported
.2byte foo  # CHECK: [[#@LINE]]:8: error: 2-byte data relocations not supported
