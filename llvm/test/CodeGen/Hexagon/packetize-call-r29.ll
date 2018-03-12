; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that the assignment to r29 does not occur in the same packet as the call.

; CHECK: call
; CHECK: }
; CHECK: r29 = #0

define protected void @f0(i8* %a0, i8* %a1, ...) local_unnamed_addr {
b0:
  call void @llvm.va_start(i8* nonnull undef)
  call void @f1()
  call void @llvm.stackrestore(i8* null)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.va_start(i8*) #0

declare protected void @f1() local_unnamed_addr

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #0

attributes #0 = { nounwind }
