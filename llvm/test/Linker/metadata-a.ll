; RUN: llvm-link %s %p/metadata-b.ll -S -o - | FileCheck %s

; CHECK: define void @foo(i32 %a)
; CHECK: ret void, !attach !0
; CHECK: define void @goo(i32 %b)
; CHECK: ret void, !attach !1
; CHECK: !0 = !{i32 524334, ptr @foo}
; CHECK: !1 = !{i32 524334, ptr @goo}

define void @foo(i32 %a) nounwind {
entry:
  ret void, !attach !0
}

!0 = !{i32 524334, ptr @foo}
