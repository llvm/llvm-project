; PR45220
; RUN: opt -S -jump-threading < %s | FileCheck %s

declare i1 @NOP()

define dso_local i32 @f(i1 %b, i1 %u) sanitize_memory {
entry:
  br i1 %b, label %if.end, label %if.else

if.else:
  %call = call i1 @NOP()
  br label %if.end

if.end:
; Check that both selects in this BB are still in place,
; and were not replaced with a conditional branch.
; CHECK:      phi
; CHECK-NEXT: phi
; CHECK-NEXT: select
; CHECK-NEXT: select
; CHECK-NEXT: ret
  %u1 = phi i1 [ true, %if.else ], [ %u, %entry ]
  %v = phi i1 [ %call, %if.else ], [ false, %entry ]
  %s = select i1 %u1, i32 22, i32 0
  %v1 = select i1 %v, i32 %s, i32 42
  ret i32 %v1
}

