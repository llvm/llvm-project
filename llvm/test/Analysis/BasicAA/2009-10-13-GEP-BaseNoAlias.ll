; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; If GEP base doesn't alias Z, then GEP doesn't alias Z.
; rdar://7282591

@Y = common global i32 0
@Z = common global i32 0

; CHECK: Function: foo
; CHECK:   NoAlias: i32* %P, i32* @Z

define void @foo(i32 %cond) nounwind {
entry:
  %a = alloca i32
  %tmp = icmp ne i32 %cond, 0
  br i1 %tmp, label %bb, label %bb1

bb:
  %b = getelementptr i32, i32* %a, i32 0
  br label %bb2

bb1:
  br label %bb2

bb2:
  %P = phi i32* [ %b, %bb ], [ @Y, %bb1 ]
  %tmp1 = load i32, i32* @Z, align 4
  store i32 123, i32* %P, align 4
  %tmp2 = load i32, i32* @Z, align 4
  br label %return

return:
  ret void
}
