; RUN: llc --mtriple=loongarch32 -mattr=+d < %s
; RUN: llc --mtriple=loongarch64 -mattr=+d < %s

;; This should not crash the code generator.

@.str.2 = external dso_local unnamed_addr constant [69 x i8], align 1

define dso_local void @main() {
entry:
  %n0 = alloca [2 x [3 x i32]], align 4
  %0 = load i32, ptr poison, align 4
  %idxprom15 = sext i32 %0 to i64
  %arrayidx16 = getelementptr inbounds [2 x [3 x i32]], ptr %n0, i64 0, i64 %idxprom15
  %arrayidx17 = getelementptr inbounds [3 x i32], ptr %arrayidx16, i64 0, i64 0
  %1 = load i32, ptr %arrayidx17, align 4
  call void (ptr, ...) @printf(ptr noundef @.str.2, i32 noundef signext %1)
  ret void
}

declare void @printf(ptr, ...)
