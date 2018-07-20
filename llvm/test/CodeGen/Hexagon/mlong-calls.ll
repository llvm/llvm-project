; RUN: llc -hexagon-long-calls -march=hexagon -enable-save-restore-long=true < %s | FileCheck %s

; CHECK: call ##f1
; CHECK: jump ##__restore

; Function Attrs: minsize nounwind
define i64 @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = add nsw i32 %a0, 5
  %v1 = tail call i64 @f1(i32 %v0) #1
  %v2 = sext i32 %a1 to i64
  %v3 = add nsw i64 %v1, %v2
  ret i64 %v3
}

; Function Attrs: minsize nounwind
declare i64 @f1(i32) #0

; Function Attrs: nounwind
define i64 @f2(i32 %a0, i32 %a1) #1 {
b0:
  %v0 = add nsw i32 %a0, 5
  %v1 = tail call i64 @f1(i32 %v0) #1
  ret i64 %v1
}

; Function Attrs: noreturn nounwind
define i64 @f3(i32 %a0, i32 %a1) #2 {
b0:
  %v0 = add nsw i32 %a0, 5
  %v1 = tail call i64 @f4(i32 %v0) #2
  unreachable
}

; Function Attrs: noreturn
declare i64 @f4(i32) #3

attributes #0 = { minsize nounwind }
attributes #1 = { nounwind }
attributes #2 = { noreturn nounwind }
attributes #3 = { noreturn }
