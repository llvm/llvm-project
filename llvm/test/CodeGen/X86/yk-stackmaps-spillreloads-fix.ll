; RUN: llc -stop-after fix-stackmaps-spill-reloads --yk-stackmap-spillreloads-fix < %s  | FileCheck %s

; CHECK-LABEL: name:            main
; CHECK-LABEL: bb.0 (%ir-block.1):
; CHECK-LABEL: CALL64pcrel32 target-flags(x86-plt) @foo2,
; CHECK-NEXT: STACKMAP 1, 0, renamable $ebx, 3, renamable $r14d, 3, 1, 4, $rbp, -48, 3, renamable $r12d, 3, 1, 4, $rbp, -52, 3, renamable $r15d, 3, renamable $r13d, 3, implicit-def dead early-clobber $r11

@.str = private unnamed_addr constant [13 x i8] c"%d %d %d %d\0A\00", align 1

define dso_local i32 @foo(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6) #0 {
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  store i32 %0, ptr %9, align 4
  store i32 %1, ptr %10, align 4
  store i32 %2, ptr %11, align 4
  store i32 %3, ptr %12, align 4
  store i32 %4, ptr %13, align 4
  store i32 %5, ptr %14, align 4
  store i32 %6, ptr %15, align 4
  %16 = load i32, ptr %9, align 4
  %17 = load i32, ptr %10, align 4
  %18 = load i32, ptr %11, align 4
  %19 = load i32, ptr %12, align 4
  %20 = load i32, ptr %13, align 4
  %21 = load i32, ptr %14, align 4
  %22 = load i32, ptr %15, align 4
  %23 = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %16, i32 noundef %17, i32 noundef %18, i32 noundef %19, i32 noundef %20, i32 noundef %21, i32 noundef %22)
  %24 = load i32, ptr %8, align 4
  ret i32 %24
}

declare i32 @printf(ptr noundef, ...) #2

define dso_local i32 @main(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %10 = load i32, ptr %2, align 4
  %11 = mul nsw i32 %10, 1
  store i32 %11, ptr %3, align 4
  %12 = load i32, ptr %2, align 4
  %13 = mul nsw i32 %12, 2
  store i32 %13, ptr %4, align 4
  %14 = load i32, ptr %2, align 4
  %15 = mul nsw i32 %14, 3
  store i32 %15, ptr %5, align 4
  %16 = load i32, ptr %2, align 4
  %17 = mul nsw i32 %16, 4
  store i32 %17, ptr %6, align 4
  %18 = load i32, ptr %2, align 4
  %19 = mul nsw i32 %18, 5
  store i32 %19, ptr %7, align 4
  %20 = load i32, ptr %2, align 4
  %21 = mul nsw i32 %20, 6
  store i32 %21, ptr %8, align 4
  %22 = load i32, ptr %2, align 4
  %23 = mul nsw i32 %22, 7
  store i32 %23, ptr %9, align 4
  %24 = call i32 @foo2(i32 noundef %23, i32 noundef %21, i32 noundef %19, i32 noundef %17, i32 noundef %15, i32 noundef %13, i32 noundef %11)
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0, i32 %11, i32 %13, i32 %15, i32 %17, i32 %19, i32 %21, i32 %23)
  %25 = mul nsw i32 %23, 5
  %26 = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %11, i32 noundef %13, i32 noundef %15, i32 noundef %17, i32 noundef %19, i32 noundef %21, i32 noundef %25)
  ret i32 0
}

declare void @foo2(...)
declare void @llvm.experimental.stackmap(i64, i32, ...)

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
