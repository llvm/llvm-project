; RUN: opt < %s -passes=sccp -S | FileCheck %s

@0 = private unnamed_addr constant [2 x i32] [i32 -1, i32 1]
@"\01??_7A@@6B@" = unnamed_addr alias i32, getelementptr inbounds ([2 x i32], ptr @0, i32 0, i32 1)

; CHECK: ret i32 1

define i32 @main() {
  %a = load i32, ptr @"\01??_7A@@6B@"
  ret i32 %a
}
