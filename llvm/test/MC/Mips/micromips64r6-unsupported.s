# RUN: not --crash llvm-mc -filetype=obj -triple=mips64-unknown-linux -mattr=+micromips \
# RUN: -mcpu=mips64r6 %s 2>&1 | FileCheck %s -check-prefix=CHECK-OPTION
# RUN: not llvm-mc -filetype=obj -triple=mips64-unknown-linux -mcpu=mips64r6 \
# RUN: %s 2>&1 | FileCheck %s -check-prefix=CHECK-MM-DIRECTIVE
# RUN: not llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN: %s 2>&1 | FileCheck %s -check-prefix=CHECK-DIRECTIVE

# CHECK-OPTION: LLVM ERROR: microMIPS64R6 is not supported

.set micromips
# CHECK-MM-DIRECTIVE: :[[@LINE-1]]:6: error: .set micromips directive is not supported with MIPS64R6

.set mips64r6
.set arch=mips64r6
# CHECK-DIRECTIVE: :[[@LINE-2]]:6: error: MIPS64R6 is not supported with microMIPS
# CHECK-DIRECTIVE: :[[@LINE-2]]:19: error: mips64r6 does not support microMIPS
