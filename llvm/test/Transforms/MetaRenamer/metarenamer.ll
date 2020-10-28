; RUN: opt -metarenamer -S < %s | FileCheck %s
; RUN: opt -passes=metarenamer -S < %s | FileCheck %s

; CHECK: target triple {{.*}}
; CHECK-NOT: {{^x*}}xxx{{^x*}}
; CHECK: ret i32 6

target triple = "x86_64-pc-linux-gnu"

%struct.bar_xxx = type { i32, double }
%struct.foo_xxx = type { i32, float, %struct.bar_xxx }

@func_5_xxx.static_local_3_xxx = internal global i32 3, align 4
@global_3_xxx = common global i32 0, align 4

@func_7_xxx = weak alias i32 (...), i32 (...)* @aliased_func_7_xxx

define i32 @aliased_func_7_xxx(...) {
  ret i32 0
}

define i32 @func_3_xxx() nounwind uwtable ssp {
  ret i32 3
}

define void @func_4_xxx(%struct.foo_xxx* sret %agg.result) nounwind uwtable ssp {
  %1 = alloca %struct.foo_xxx, align 8
  %2 = getelementptr inbounds %struct.foo_xxx, %struct.foo_xxx* %1, i32 0, i32 0
  store i32 1, i32* %2, align 4
  %3 = getelementptr inbounds %struct.foo_xxx, %struct.foo_xxx* %1, i32 0, i32 1
  store float 2.000000e+00, float* %3, align 4
  %4 = getelementptr inbounds %struct.foo_xxx, %struct.foo_xxx* %1, i32 0, i32 2
  %5 = getelementptr inbounds %struct.bar_xxx, %struct.bar_xxx* %4, i32 0, i32 0
  store i32 3, i32* %5, align 4
  %6 = getelementptr inbounds %struct.bar_xxx, %struct.bar_xxx* %4, i32 0, i32 1
  store double 4.000000e+00, double* %6, align 8
  %7 = bitcast %struct.foo_xxx* %agg.result to i8*
  %8 = bitcast %struct.foo_xxx* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %7, i8* align 8 %8, i64 24, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

define i32 @func_5_xxx(i32 %arg_1_xxx, i32 %arg_2_xxx, i32 %arg_3_xxx, i32 %arg_4_xxx) nounwind uwtable ssp {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %local_1_xxx = alloca i32, align 4
  %local_2_xxx = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %arg_1_xxx, i32* %1, align 4
  store i32 %arg_2_xxx, i32* %2, align 4
  store i32 %arg_3_xxx, i32* %3, align 4
  store i32 %arg_4_xxx, i32* %4, align 4
  store i32 1, i32* %local_1_xxx, align 4
  store i32 2, i32* %local_2_xxx, align 4
  store i32 0, i32* %i, align 4
  br label %5

; <label>:5                                       ; preds = %9, %0
  %6 = load i32, i32* %i, align 4
  %7 = icmp slt i32 %6, 10
  br i1 %7, label %8, label %12

; <label>:8                                       ; preds = %5
  br label %9

; <label>:9                                       ; preds = %8
  %10 = load i32, i32* %i, align 4
  %11 = add nsw i32 %10, 1
  store i32 %11, i32* %i, align 4
  br label %5

; <label>:12                                      ; preds = %5
  %13 = load i32, i32* %local_1_xxx, align 4
  %14 = load i32, i32* %1, align 4
  %15 = add nsw i32 %13, %14
  %16 = load i32, i32* %local_2_xxx, align 4
  %17 = add nsw i32 %15, %16
  %18 = load i32, i32* %2, align 4
  %19 = add nsw i32 %17, %18
  %20 = load i32, i32* @func_5_xxx.static_local_3_xxx, align 4
  %21 = add nsw i32 %19, %20
  %22 = load i32, i32* %3, align 4
  %23 = add nsw i32 %21, %22
  %24 = load i32, i32* %4, align 4
  %25 = add nsw i32 %23, %24
  ret i32 %25
}

define i32 @varargs_func_6_xxx(i32 %arg_1_xxx, i32 %arg_2_xxx, ...) nounwind uwtable ssp {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 %arg_1_xxx, i32* %1, align 4
  store i32 %arg_2_xxx, i32* %2, align 4
  ret i32 6
}

declare noalias i8* @malloc(i32)
declare void @free(i8* nocapture)

define void @dont_rename_lib_funcs() {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[TMP:%.*]] = call i8* @malloc(i32 23)
; CHECK-NEXT:    call void @free(i8* [[TMP]])
; CHECK-NEXT:    ret void
;
  %x = call i8* @malloc(i32 23)
  call void @free(i8* %x)
  ret void
}
