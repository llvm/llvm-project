; A GEP of an address made by inttoptr becomes integer arithmetic and its own
; inttoptr in the Metal pipeline: the AIR writer gave the inttoptr one pointee
; from its uses, and a byte GEP storing a float next to one storing an i32 made
; Metal's reader reject the library ("Unexpected bitcode file"). The portable
; AIR keeps the GEPs.
; RUN: opt -mtriple=air -passes=air-legalize-unsupported-ir -S %s | FileCheck %s
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s --check-prefix=PORTABLE
; RUN: llc -mtriple=air -filetype=obj %s -o /dev/null

; CHECK-LABEL: define void @k(
; CHECK: [[A4:%.*]] = add i64 %a, 4
; CHECK: [[F:%.*]] = inttoptr i64 [[A4]] to ptr addrspace(1)
; CHECK: store float 0.000000e+00, ptr addrspace(1) [[F]]
; CHECK: [[A8:%.*]] = add i64 %a, 8
; CHECK: [[G:%.*]] = inttoptr i64 [[A8]] to ptr addrspace(1)
; CHECK: store i32 7, ptr addrspace(1) [[G]]
; PORTABLE-LABEL: define void @k(
; PORTABLE: getelementptr i8
define void @k(i64 %a) {
  %p = inttoptr i64 %a to ptr addrspace(1)
  %f = getelementptr i8, ptr addrspace(1) %p, i64 4
  store float 0.0, ptr addrspace(1) %f
  %g = getelementptr i8, ptr addrspace(1) %p, i64 8
  store i32 7, ptr addrspace(1) %g
  ret void
}
