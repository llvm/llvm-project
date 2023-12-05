@ RUN: not llvm-mc -filetype=obj -o /dev/null %s 2>&1 -triple=thumbv7   | FileCheck %s
@ RUN: not llvm-mc -filetype=obj -o /dev/null %s 2>&1 -triple=thumbebv7 | FileCheck %s

   ldrd r0, r1, foo

@ CHECK: :[[#@LINE-2]]:4: error: unsupported relocation type
