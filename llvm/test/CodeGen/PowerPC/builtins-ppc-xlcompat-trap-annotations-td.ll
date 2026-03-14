; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   --ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   --ppc-asm-full-reg-names -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-aix \
; RUN:   --ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s -check-prefix=AIX
; RUN: llc -mtriple=powerpc64-unknown-aix -filetype=obj -o %t_64.o < %s
; RUN: llvm-readobj --exception-section %t_64.o | FileCheck %s --check-prefix=OBJ64

; Check that we do not crash in object mode
; OBJ64:       Exception section {
; OBJ64-NEXT:    Symbol: .test__tdw_annotation

!1 = !{!"ppc-trap-reason", !"1", !"2"}
declare void @llvm.ppc.trapd(i64 %a)
declare void @llvm.ppc.tdw(i64 %a, i64 %b, i32 immarg)

define dso_local void @test__trapd_annotation(i64 %a) {
; CHECK-LABEL: test__trapd_annotation:
; CHECK:       # %bb.0:
; CHECK-NEXT:    tdi 24, r3, 0
; CHECK-NEXT:    blr
;
; AIX-LABEL: test__trapd_annotation:
; AIX:       # %bb.0:
; AIX-NEXT:  L..tmp0:
; AIX-NEXT:    .except .test__trapd_annotation, 1, 2
; AIX-NEXT:    tdi 24, r3, 0
; AIX-NEXT:    blr
  call void @llvm.ppc.trapd(i64 %a), !annotation !1
  ret void
}

define dso_local void @test__tdw_annotation(i64 %a) {
; CHECK-LABEL: test__tdw_annotation:
; CHECK:       # %bb.0:
; CHECK-NEXT:    tdi 0, r3, 4
; CHECK-NEXT:    blr
;
; AIX-LABEL: test__tdw_annotation:
; AIX:       # %bb.0:
; AIX-NEXT:  L..tmp1:
; AIX-NEXT:    .except .test__tdw_annotation, 1, 2
; AIX-NEXT:    tdi 0, r3, 4
; AIX-NEXT:    blr
  call void @llvm.ppc.tdw(i64 4, i64 %a, i32 0), !annotation !1
  ret void
}
