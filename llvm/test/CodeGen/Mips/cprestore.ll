; DISABLE: llc -mtriple=mipsel < %s | FileCheck %s
; RUN: false
; XFAIL: *

; CHECK: .set macro
; CHECK: .set at
; CHECK-NEXT: .cprestore
; CHECK: .set noat
; CHECK-NEXT: .set nomacro

%struct.S = type { [16384 x i32] }

define void @foo2() nounwind {
entry:
  %s = alloca %struct.S, align 4
  call void @foo1(ptr byval(%struct.S) %s)
  ret void
}

declare void @foo1(ptr byval(%struct.S))
