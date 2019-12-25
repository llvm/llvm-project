; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

define void @_Z1fv() #0 {
entry:
  %0 = call i8* @llvm.eh.dwarf.cfa(i32 0)
  call void @_Z1gPv(i8* %0)
  ret void

; CHECK-LABEL: @_Z1fv
; CHECK: stdu 1, -[[SS:[0-9]+]](1)
; CHECK: .cfi_def_cfa_offset [[SS]]
; CHECK: mr 31, 1
; CHECK: .cfi_def_cfa_register r31
; CHECK: addi 3, 31, [[SS]]
; CHECK-NEXT: bl _Z1gPv
; CHECK: blr
}

declare void @_Z1gPv(i8*)

; Function Attrs: nounwind
declare i8* @llvm.eh.dwarf.cfa(i32) #1

attributes #0 = { "frame-pointer"="all" "target-cpu"="ppc64le" }
attributes #1 = { nounwind }

