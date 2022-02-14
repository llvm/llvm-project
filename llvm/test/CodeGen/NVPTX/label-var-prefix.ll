; RUN: llc < %s -march=nvptx64  | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK: .u32 LBB0_2
@LBB0_2 = global i32 zeroinitializer
; CHECK-NOT: LBB0_2

declare i64 @foo(i64 %a, i64 %b, i64 %c)
declare i64 @bar(i64 %a, i64 %b, i64 %c)
define i64 @baz(i64 %a, i64 %b, i64 %c) {
entry:
  %0 = icmp eq i64 %a, 0
  br i1 %0, label %L1, label %L2

; CHECK: $L__BB{{[0-9_]+}}:
L1:
  %1 = call i64 @foo(i64 %a, i64 %b, i64 %c)
  ret i64 %1
; CHECK: $L__BB{{[0-9_]+}}:
L2:
  %2 = call i64 @bar(i64 %a, i64 %b, i64 %c)
  ret i64 %2
}
