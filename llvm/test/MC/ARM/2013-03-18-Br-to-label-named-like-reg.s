@ RUN: llvm-mc -triple arm-eabi %s -o - | FileCheck %s

@ CHECK: test:
@ CHECK: bl r1
test:
  bl r1
