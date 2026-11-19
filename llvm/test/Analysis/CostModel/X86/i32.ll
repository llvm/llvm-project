; RUN: opt < %s  -passes="print<cost-model>" -disable-output -mtriple=i386 -mcpu=corei7-avx 2>&1 | FileCheck %s

;CHECK: cost of 0 {{.*}} ret
define i32 @no_info(i32 %arg) {
  %e = add i64 undef, undef
  ret i32 undef
}
