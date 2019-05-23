; RUN: opt < %s -globalopt -S | FileCheck %s

%struct.Expr = type { [1 x i32], i32 }

@e = internal global %struct.Expr zeroinitializer, align 4
; CHECK-NOT: @e = internal global %struct.Expr zeroinitializer, align 4

define dso_local i32 @foo(i32 %i) {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %arrayidx = getelementptr inbounds [1 x i32], [1 x i32]* getelementptr inbounds (%struct.Expr, %struct.Expr* @e, i32 0, i32 0), i32 0, i32 %0
  store i32 57005, i32* %arrayidx, align 4
  %1 = load i32, i32* getelementptr inbounds (%struct.Expr, %struct.Expr* @e, i32 0, i32 1), align 4
  ret i32 %1
; CHECK:  ret i32 0
}
