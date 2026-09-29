; RUN: llc --enable-machine-outliner --mtriple=loongarch32-unknown-linux < %s | FileCheck %s
@x = dso_local global i32 0, align 4

define dso_local i32 @check_boundaries() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  store i32 0, ptr %2, align 4
  %6 = load i32, ptr %2, align 4
  %7 = icmp ne i32 %6, 0
  br i1 %7, label %9, label %8

  store i32 1, ptr %2, align 4
  store i32 2, ptr %3, align 4
  store i32 3, ptr %4, align 4
  store i32 4, ptr %5, align 4
  br label %10

  store i32 1, ptr %4, align 4
  br label %10

  %11 = load i32, ptr %2, align 4
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %14, label %13

  store i32 1, ptr %2, align 4
  store i32 2, ptr %3, align 4
  store i32 3, ptr %4, align 4
  store i32 4, ptr %5, align 4
  br label %15

  store i32 1, ptr %4, align 4
  br label %15

  ret i32 0
}

define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4

  store i32 0, ptr %1, align 4
  store i32 0, ptr @x, align 4
  store i32 1, ptr %2, align 4
  store i32 2, ptr %3, align 4
  store i32 3, ptr %4, align 4
  store i32 4, ptr %5, align 4
  store i32 1, ptr @x, align 4
  call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  store i32 1, ptr %2, align 4
  store i32 2, ptr %3, align 4
  store i32 3, ptr %4, align 4
  store i32 4, ptr %5, align 4
  ret i32 0
}

attributes #0 = { noredzone nounwind ssp uwtable "frame-pointer"="all" }
