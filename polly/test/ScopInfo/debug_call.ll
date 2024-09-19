; RUN: opt %loadNPMPolly -polly-debug-func=dbg_printf '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s -match-full-lines
;
; Check that the call to dbg_printf is accepted as a debug-function.
;
declare void @dbg_printf(ptr, ...)

define void @func(i32 %n) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      call void (ptr, ...) @dbg_printf(ptr null, i32 %j)
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_body[i0] : 0 <= i0 < n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_body[i0] -> [i0] };
; CHECK-NEXT: }
