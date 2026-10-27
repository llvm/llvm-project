; RUN: opt -mtriple=riscv64 -mattr=+v,+zbb -passes="print<cost-model>" -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=riscv64 -mattr=+zbb -passes="print<cost-model>" -disable-output %s 2>&1 | FileCheck %s

declare i16 @llvm.sadd.sat.i16(i16, i16)
declare i16 @llvm.ssub.sat.i16(i16, i16)
declare i16 @llvm.uadd.sat.i16(i16, i16)
declare i16 @llvm.usub.sat.i16(i16, i16)

; CHECK-LABEL: Printing analysis 'Cost Model Analysis' for function 'signed_add':
; CHECK: Cost Model: Found an estimated cost of 3 for instruction: {{.*}}@llvm.sadd.sat.i16
define i16 @signed_add(i16 %a, i16 %b) {
  %r = call i16 @llvm.sadd.sat.i16(i16 %a, i16 %b)
  ret i16 %r
}

; CHECK-LABEL: Printing analysis 'Cost Model Analysis' for function 'signed_sub':
; CHECK: Cost Model: Found an estimated cost of 3 for instruction: {{.*}}@llvm.ssub.sat.i16
define i16 @signed_sub(i16 %a, i16 %b) {
  %r = call i16 @llvm.ssub.sat.i16(i16 %a, i16 %b)
  ret i16 %r
}

; CHECK-LABEL: Printing analysis 'Cost Model Analysis' for function 'unsigned_add':
; CHECK: Cost Model: Found an estimated cost of 2 for instruction: {{.*}}@llvm.uadd.sat.i16
define i16 @unsigned_add(i16 %a, i16 %b) {
  %r = call i16 @llvm.uadd.sat.i16(i16 %a, i16 %b)
  ret i16 %r
}

; CHECK-LABEL: Printing analysis 'Cost Model Analysis' for function 'unsigned_sub':
; CHECK: Cost Model: Found an estimated cost of 2 for instruction: {{.*}}@llvm.usub.sat.i16
define i16 @unsigned_sub(i16 %a, i16 %b) {
  %r = call i16 @llvm.usub.sat.i16(i16 %a, i16 %b)
  ret i16 %r
}
