; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

@a = dso_local local_unnamed_addr global [10 x i32] zeroinitializer, align 4

; CHECK-LABEL: Function: select_in_gep
; CHECK: NoAlias: i32* %arrayidx, i32* getelementptr inbounds ([10 x i32], ptr @a, i64 0, i64 3)
define i32 @select_in_gep(i1 %c)  {
entry:
  %cond = select i1 %c, i64 2, i64 1
  %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @a, i64 0, i64 3), align 4
  %arrayidx = getelementptr inbounds [10 x i32], ptr @a, i64 0, i64 %cond
  store i32 %0, ptr %arrayidx, align 4
  %1 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @a, i64 0, i64 3), align 4
  ret i32 %1
}
