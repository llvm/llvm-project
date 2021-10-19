; RUN: opt %loadPolly -polly-codegen-trace-stmts -polly-codegen-trace-scalars -polly-codegen -S < %s | FileCheck %s
;

define void @func(i32 %n, double* %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      store double 0.0, double* %A_idx
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: @0 = private unnamed_addr addrspace(4) constant [10 x i8] c"Stmt_body\00"
; CHECK: @1 = private unnamed_addr addrspace(4) constant [2 x i8] c"(\00"
; CHECK: @2 = private unnamed_addr addrspace(4) constant [2 x i8] c")\00"
; CHECK: @3 = private unnamed_addr addrspace(4) constant [2 x i8] c"\0A\00"
; CHECK: @4 = private unnamed_addr constant [12 x i8] c"%s%s%ld%s%s\00"

; CHECK:      polly.stmt.body:
; CHECK:        call i32 (...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @4, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([10 x i8], [10 x i8] addrspace(4)* @0, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @1, i32 0, i32 0), i64 %polly.indvar, i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @2, i32 0, i32 0), i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* @3, i32 0, i32 0))
; CHECK-NEXT:   call i32 @fflush(i8* null)
