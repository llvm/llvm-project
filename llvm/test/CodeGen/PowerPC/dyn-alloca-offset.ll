; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare i64 @llvm.get.dynamic.area.offset.i64()

declare i64 @bar(i64)

attributes #0 = { nounwind }

; Function Attrs: nounwind sanitize_address uwtable
define signext i64 @foo(i32 signext %N, i32 signext %M) #0 {
  %1 = alloca i64, align 32
  %dynamic_area_offset = call i64 @llvm.get.dynamic.area.offset.i64()
  %2 = call i64 @bar(i64 %dynamic_area_offset)
  ret i64 %2

; CHECK-DAG: li [[REG1:[0-9]+]], 112
; CHECK: blr

}
