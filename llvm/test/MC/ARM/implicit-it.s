@ RUN: not llvm-mc -triple thumbv7a--none-eabi -arm-implicit-it=never  < %s 2>%t | FileCheck %s --check-prefix=CHECK
@ RUN:     FileCheck %s < %t --check-prefix=THUMB-STDERR
@ RUN: not llvm-mc -triple   armv7a--none-eabi -arm-implicit-it=never  < %s 2>%t | FileCheck %s --check-prefix=CHECK --check-prefix=ARM
@ RUN:     FileCheck %s < %t --check-prefix=ARM-STDERR

@ RUN: not llvm-mc -triple thumbv7a--none-eabi -arm-implicit-it=always < %s      | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB
@ RUN: not llvm-mc -triple   armv7a--none-eabi -arm-implicit-it=always < %s      | FileCheck %s --check-prefix=CHECK --check-prefix=ARM

@ RUN: not llvm-mc -triple thumbv7a--none-eabi -arm-implicit-it=arm    < %s 2>%t | FileCheck %s --check-prefix=CHECK
@ RUN:     FileCheck %s < %t --check-prefix=THUMB-STDERR
@ RUN: not llvm-mc -triple   armv7a--none-eabi -arm-implicit-it=arm    < %s      | FileCheck %s --check-prefix=CHECK --check-prefix=ARM

@ RUN: not llvm-mc -triple thumbv7a--none-eabi                     < %s 2>%t | FileCheck %s --check-prefix=CHECK
@ RUN:     FileCheck %s < %t --check-prefix=THUMB-STDERR
@ RUN: not llvm-mc -triple   armv7a--none-eabi                     < %s      | FileCheck %s --check-prefix=CHECK --check-prefix=ARM

@ RUN: not llvm-mc -triple thumbv7a--none-eabi -arm-implicit-it=thumb  < %s      | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB
@ RUN: not llvm-mc -triple   armv7a--none-eabi -arm-implicit-it=thumb  < %s 2>%t | FileCheck %s --check-prefix=CHECK --check-prefix=ARM
@ RUN:     FileCheck %s < %t --check-prefix=ARM-STDERR

@ A single conditional instruction without IT block
  .section test1
@ CHECK-LABEL: test1
  addeq r0, r0, #1
@ THUMB: it eq
@ THUMB: addeq r0, r0, #1
@ ARM:   addeq r0, r0, #1
@ THUMB-STDERR: error: predicated instructions must be in IT block
@ ARM-STDERR: warning: predicated instructions should be in IT block

@ A single conditional instruction with IT block
  .section test2
@ CHECK-LABEL: test2
  it eq
  addeq r0, r0, #1
@ THUMB: it eq
@ THUMB: addeq r0, r0, #1
@ ARM:   addeq r0, r0, #1
@ THUMB-STDERR-NOT: error:
@ ARM-STDERR-NOT: warning:

@ A single conditional instruction with IT block, but incorrect condition
  .section test3
@ CHECK-LABEL: test3
  it eq
  addgt r0, r0, #1
@ THUMB-STDERR: error: incorrect condition in IT block
@ ARM-STDERR:   error: incorrect condition in IT block

@ Multiple conditional instructions in an IT block, inverted and non-inverted conditions
  .section test4
@ CHECK-LABEL: test4
  itete gt
  addgt r0, r0, #1
  addle r0, r0, #1
  addgt r0, r0, #1
  addle r0, r0, #1
@ THUMB: itete gt
@ CHECK: addgt r0, r0, #1
@ CHECK: addle r0, r0, #1
@ CHECK: addgt r0, r0, #1
@ CHECK: addle r0, r0, #1
@ THUMB-STDERR-NOT: error:
@ ARM-STDERR-NOT: warning:

@ Incorrectly inverted condition on the second slot of an IT block
  .section test5
@ CHECK-LABEL: test5
  itt eq
  addeq r0, r0, #1
  addne r0, r0, #1
@ THUMB-STDERR: error: incorrect condition in IT block
@ ARM-STDERR:   error: incorrect condition in IT block
