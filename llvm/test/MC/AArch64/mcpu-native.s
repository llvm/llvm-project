# REQUIRES: aarch64-registered-target,system-linux,aarch64-host
# RUN: llvm-mc -triple=aarch64 -filetype=obj -o %t -mcpu=native %s 2> %t.stderr
# RUN: FileCheck --allow-empty %s < %t.stderr

# CHECK-NOT: {{.+}}
