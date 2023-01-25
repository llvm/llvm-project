; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   --ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   --ppc-asm-full-reg-names -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu \
; RUN:   --ppc-asm-full-reg-names -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-aix \
; RUN:   --ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s -check-prefix=AIX
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-aix \
; RUN:   --ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s -check-prefix=AIX
; RUN: llc -mtriple=powerpc-ibm-aix-xcoff -filetype=obj -o %t_32.o < %s
; RUN: llvm-readobj --exception-section %t_32.o | FileCheck %s --check-prefix=OBJ

; Check that we do not crash in object mode
; OBJ:       Exception section {
; OBJ-NEXT:    Symbol: .test__trap_annotation (3)

!1 = !{!"ppc-trap-reason", !"1", !"2"}
declare void @llvm.ppc.trap(i32 %a)
declare void @llvm.ppc.tw(i32 %a, i32 %b, i32 immarg)
define dso_local void @test__trap_annotation(i32 %a) {
; CHECK-LABEL: test__trap_annotation:
; CHECK:       # %bb.0:
; CHECK-NEXT:    twi 24, r3, 0
; CHECK-NEXT:    blr
;
; AIX-LABEL: test__trap_annotation:
; AIX:       # %bb.0:
; AIX-NEXT:  L..tmp0:
; AIX-NEXT:    .except .test__trap_annotation, 1, 2
; AIX-NEXT:    twi 24, r3, 0
; AIX-NEXT:    blr
  call void @llvm.ppc.trap(i32 %a), !annotation !1
  ret void
}

define dso_local void @test__tw_annotation(i32 %a) {
; CHECK-LABEL: test__tw_annotation:
; CHECK:       # %bb.0:
; CHECK-NEXT:    twi 0, r3, 4
; CHECK-NEXT:    blr
;
; AIX-LABEL: test__tw_annotation:
; AIX:       # %bb.0:
; AIX-NEXT:  L..tmp1:
; AIX-NEXT:    .except .test__tw_annotation, 1, 2
; AIX-NEXT:    twi 0, r3, 4
; AIX-NEXT:    blr
  call void @llvm.ppc.tw(i32 4, i32 %a, i32 0), !annotation !1
  ret void
}
