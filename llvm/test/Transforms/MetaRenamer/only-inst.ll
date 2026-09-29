; RUN: opt -passes=metarenamer -rename-only-inst=1 -S < %s | FileCheck %s
target triple = "x86_64-pc-linux-gnu"

; CHECK: %struct.foo_xxx = type { i32, float, %struct.bar_xxx }
; CHECK: %struct.bar_xxx = type { i32, double }
%struct.bar_xxx = type { i32, double }
%struct.foo_xxx = type { i32, float, %struct.bar_xxx }

; CHECK: @global_3_xxx = common global
@global_3_xxx = common global i32 0, align 4

; CHECK-LABEL: func_4_xxx
; CHECK-NOT: %1
; CHECK-NOT: %2
; CHECK-NOT: %3
; CHECK-NOT: %4
; CHECK: %.int_arg = call i64 @len()
define void @func_4_xxx(ptr sret(%struct.foo_xxx) %agg.result) nounwind uwtable ssp {
  %1 = alloca %struct.foo_xxx, align 8
  store i32 1, ptr %1, align 4
  %2 = getelementptr inbounds %struct.foo_xxx, ptr %1, i32 0, i32 1
  store float 2.000000e+00, ptr %2, align 4
  %3 = getelementptr inbounds %struct.foo_xxx, ptr %1, i32 0, i32 2
  store i32 3, ptr %3, align 4
  %4 = getelementptr inbounds %struct.bar_xxx, ptr %3, i32 0, i32 1
  store double 4.000000e+00, ptr %4, align 8
  %.int_arg = call i64 @len()
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.result, ptr align 8 %1, i64 %.int_arg, i1 false)
  ret void
}

declare i64 @len()
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
