; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s

@X = constant {i8, i8 } { i8 0, i8 0 }
@XA = alias i8, getelementptr inbounds ({ i8, i8 }, ptr @X, i32 0, i32 1)

define void @f(ptr %p) align 2 {
entry:
  store ptr @XA, ptr %p, align 4
  ret void
}

; CHECK: f:
; CHECK: ldr r{{.*}}, [[L:.*]]
; CHECK: [[L]]:
; CHECK-NEXT: .long XA
; CHECK: XA = X+1
