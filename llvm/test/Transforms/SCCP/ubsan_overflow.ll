; RUN: opt -passes=sccp -S %s | FileCheck %s

@0 = private unnamed_addr constant [16 x i8] c"\01\00\00\00\01\01\00\00\01\01\01\00\01\01\01\01"

; CHECK-LABEL: @foo
define i8 @foo() {
entry:
  %0 = add nuw nsw i64 0, -1
  %1 = lshr i64 %0, 1
  %2 = getelementptr inbounds [4 x [4 x i8]], [4 x [4 x i8]]* bitcast ([16 x i8]* @0 to [4 x [4 x i8]]*), i64 0, i64 0, i64 %1
  %3 = load i8, i8* %2, align 1
  ret i8 %3
}
