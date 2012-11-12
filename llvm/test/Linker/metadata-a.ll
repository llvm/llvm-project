; RUN: llvm-link %s %p/metadata-b.ll -S -o - | FileCheck %s

; CHECK: define void @foo(i32 %a)
; CHECK: ret void, !attach !0, !also !{i32 %a}
; CHECK: define void @goo(i32 %b)
; CHECK: ret void, !attach !1, !and !{i32 %b}
; CHECK: !0 = metadata !{i32 524334, void (i32)* @foo}
; CHECK: !1 = metadata !{i32 524334, void (i32)* @goo}

define void @foo(i32 %a) nounwind {
entry:
  ret void, !attach !0, !also !{ i32 %a }
}

!0 = metadata !{i32 524334, void (i32)* @foo}
