// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s
// CHECK: Undefined temporary symbol

.addrsig
.addrsig_sym .Lundef
