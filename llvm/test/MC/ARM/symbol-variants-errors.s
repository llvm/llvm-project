@ RUN: not llvm-mc < %s -triple armv7-none-linux-gnueabi 2>&1 | FileCheck %s

@ check for invalid variant
f1:
  bl bar(blargh)
@CHECK: error: invalid variant 'blargh'
@CHECK:  bl bar(blargh)
@CHECK:                ^

@ check for missing closed paren
f2:
  .word bar(got
@CHECK: :[[#@LINE-1]]:16: error: expected ')'
@CHECK:  .word bar(got
@CHECK:               ^

@ check for invalid symbol before variant end
f3:
  .word bar(got+2)
@CHECK: :[[#@LINE-1]]:16: error: expected ')'
@CHECK:  .word bar(got+2)
@CHECK:               ^
