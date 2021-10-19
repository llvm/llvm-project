; RUN: llc < %s -mtriple=ve | FileCheck %s

; Function Attrs: noinline nounwind optnone
define i8* @stacksave() {
; CHECK-LABEL: stacksave:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, %s11
; CHECK-NEXT:    or %s11, 0, %s9
  %ret = call i8* @llvm.stacksave()
  ret i8* %ret
}

; Function Attrs: noinline nounwind optnone
define void @stackrestore(i8* %ptr) {
; CHECK-LABEL: stackrestore:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  call void @llvm.stackrestore(i8* %ptr)
  ret void
}

; Function Attrs: nounwind
declare i8* @llvm.stacksave()
; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*)
