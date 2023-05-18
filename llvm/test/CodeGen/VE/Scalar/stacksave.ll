; RUN: llc < %s -mtriple=ve | FileCheck %s

; Function Attrs: noinline nounwind optnone
define ptr @stacksave() {
; CHECK-LABEL: stacksave:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, %s11
; CHECK-NEXT:    or %s11, 0, %s9
  %ret = call ptr @llvm.stacksave()
  ret ptr %ret
}

; Function Attrs: noinline nounwind optnone
define void @stackrestore(ptr %ptr) {
; CHECK-LABEL: stackrestore:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  call void @llvm.stackrestore(ptr %ptr)
  ret void
}

; Function Attrs: nounwind
declare ptr @llvm.stacksave()
; Function Attrs: nounwind
declare void @llvm.stackrestore(ptr)
