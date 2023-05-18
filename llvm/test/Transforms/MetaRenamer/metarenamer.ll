; RUN: opt -passes=metarenamer -S < %s | FileCheck %s

; CHECK: target triple {{.*}}
; CHECK-NOT: {{^x*}}xxx{{^x*}}
; CHECK: ret i32 6

target triple = "x86_64-pc-linux-gnu"

%struct.bar_xxx = type { i32, double }
%struct.foo_xxx = type { i32, float, %struct.bar_xxx }

@func_5_xxx.static_local_3_xxx = internal global i32 3, align 4
@global_3_xxx = common global i32 0, align 4

@func_7_xxx = weak alias i32 (...), ptr @aliased_func_7_xxx

define i32 @aliased_func_7_xxx(...) {
  ret i32 0
}

define i32 @func_3_xxx() nounwind uwtable ssp {
  ret i32 3
}

define void @func_4_xxx(ptr sret(%struct.foo_xxx) %agg.result) nounwind uwtable ssp {
  %1 = alloca %struct.foo_xxx, align 8
  store i32 1, ptr %1, align 4
  %2 = getelementptr inbounds %struct.foo_xxx, ptr %1, i32 0, i32 1
  store float 2.000000e+00, ptr %2, align 4
  %3 = getelementptr inbounds %struct.foo_xxx, ptr %1, i32 0, i32 2
  store i32 3, ptr %3, align 4
  %4 = getelementptr inbounds %struct.bar_xxx, ptr %3, i32 0, i32 1
  store double 4.000000e+00, ptr %4, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.result, ptr align 8 %1, i64 24, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

define i32 @func_5_xxx(i32 %arg_1_xxx, i32 %arg_2_xxx, i32 %arg_3_xxx, i32 %arg_4_xxx) nounwind uwtable ssp {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %local_1_xxx = alloca i32, align 4
  %local_2_xxx = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %arg_1_xxx, ptr %1, align 4
  store i32 %arg_2_xxx, ptr %2, align 4
  store i32 %arg_3_xxx, ptr %3, align 4
  store i32 %arg_4_xxx, ptr %4, align 4
  store i32 1, ptr %local_1_xxx, align 4
  store i32 2, ptr %local_2_xxx, align 4
  store i32 0, ptr %i, align 4
  br label %5

; <label>:5                                       ; preds = %9, %0
  %6 = load i32, ptr %i, align 4
  %7 = icmp slt i32 %6, 10
  br i1 %7, label %8, label %12

; <label>:8                                       ; preds = %5
  br label %9

; <label>:9                                       ; preds = %8
  %10 = load i32, ptr %i, align 4
  %11 = add nsw i32 %10, 1
  store i32 %11, ptr %i, align 4
  br label %5

; <label>:12                                      ; preds = %5
  %13 = load i32, ptr %local_1_xxx, align 4
  %14 = load i32, ptr %1, align 4
  %15 = add nsw i32 %13, %14
  %16 = load i32, ptr %local_2_xxx, align 4
  %17 = add nsw i32 %15, %16
  %18 = load i32, ptr %2, align 4
  %19 = add nsw i32 %17, %18
  %20 = load i32, ptr @func_5_xxx.static_local_3_xxx, align 4
  %21 = add nsw i32 %19, %20
  %22 = load i32, ptr %3, align 4
  %23 = add nsw i32 %21, %22
  %24 = load i32, ptr %4, align 4
  %25 = add nsw i32 %23, %24
  ret i32 %25
}

define i32 @varargs_func_6_xxx(i32 %arg_1_xxx, i32 %arg_2_xxx, ...) nounwind uwtable ssp {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 %arg_1_xxx, ptr %1, align 4
  store i32 %arg_2_xxx, ptr %2, align 4
  ret i32 6
}

declare noalias ptr @malloc(i64)
declare void @free(ptr nocapture)

define void @dont_rename_lib_funcs() {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[TMP:%.*]] = call ptr @malloc(i64 23)
; CHECK-NEXT:    call void @free(ptr [[TMP]])
; CHECK-NEXT:    ret void
;
  %x = call ptr @malloc(i64 23)
  call void @free(ptr %x)
  ret void
}
