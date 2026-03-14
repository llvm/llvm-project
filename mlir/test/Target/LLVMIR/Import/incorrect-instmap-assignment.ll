; RUN: mlir-translate --import-llvm %s | FileCheck %s

; This test file is meant to saturate `instMap` used in the translation
; and force it to resize.

; This test is primarily used to make sure it doesn't bail out with non-zero
; exit code. Thus, we only wrote minimum level of checks.

%my_struct = type {i32, i32}
@gvar = external global %my_struct

; CHECK: llvm.func @f(%arg0: i32, %arg1: i32)
define void @f(i32 %0, i32 %1) {
  %3 = add i32 %0, %1
  %4 = add i32 %1, %3
  %5 = add i32 %3, %4
  %6 = add i32 %4, %5
  %7 = add i32 %5, %6
  %8 = add i32 %6, %7
  %9 = add i32 %7, %8
  %10 = add i32 %8, %9
  %11 = add i32 %9, %10
  %12 = add i32 %10, %11
  %13 = add i32 %11, %12
  %14 = add i32 %12, %13
  %15 = add i32 %13, %14
  %16 = add i32 %14, %15
  %17 = add i32 %15, %16
  %18 = add i32 %16, %17
  %19 = add i32 %17, %18
  %20 = add i32 %18, %19
  %21 = add i32 %19, %20
  %22 = add i32 %20, %21
  %23 = add i32 %21, %22
  %24 = add i32 %22, %23
  %25 = add i32 %23, %24
  %26 = add i32 %24, %25
  %27 = add i32 %25, %26
  %28 = add i32 %26, %27
  %29 = add i32 %27, %28
  %30 = add i32 %28, %29
  %31 = add i32 %29, %30
  %32 = add i32 %30, %31
  %33 = add i32 %31, %32
  %34 = add i32 %32, %33
  %35 = add i32 %33, %34
  %36 = add i32 %34, %35
  %37 = add i32 %35, %36
  %38 = add i32 %36, %37
  %39 = add i32 %37, %38
  %40 = add i32 %38, %39
  %41 = add i32 %39, %40
  %42 = add i32 %40, %41
  %43 = add i32 %41, %42
  %44 = add i32 %42, %43
  %45 = add i32 %43, %44
  %46 = add i32 %44, %45
  %47 = add i32 %45, %46
  %48 = add i32 %46, %47
  %49 = add i32 %47, %48
  %50 = add i32 %48, %49
  %51 = add i32 %49, %50
  %52 = add i32 %50, %51
  %53 = add i32 %51, %52
  %54 = add i32 %52, %53
  %55 = add i32 %53, %54
  %56 = add i32 %54, %55
  %57 = add i32 %55, %56
  %58 = add i32 %56, %57
  %59 = add i32 %57, %58
  %60 = add i32 %58, %59
  %61 = add i32 %59, %60
  %62 = add i32 %60, %61
  %63 = add i32 %61, %62
  %64 = add i32 %62, %63
  %65 = add i32 %63, %64
  %66 = add i32 %64, %65
  %67 = add i32 %65, %66
  %68 = add i32 %66, %67
  %69 = add i32 %67, %68
  %70 = add i32 %68, %69
  %71 = add i32 %69, %70
  %72 = add i32 %70, %71
  %73 = add i32 %71, %72
  %74 = add i32 %72, %73
  %75 = add i32 %73, %74
  %76 = add i32 %74, %75
  %77 = add i32 %75, %76
  %78 = add i32 %76, %77
  %79 = add i32 %77, %78
  %80 = add i32 %78, %79
  %81 = add i32 %79, %80
  %82 = add i32 %80, %81
  %83 = add i32 %81, %82
  %84 = add i32 %82, %83
  %85 = add i32 %83, %84
  %86 = add i32 %84, %85
  %87 = add i32 %85, %86
  %88 = add i32 %86, %87
  %89 = add i32 %87, %88
  %90 = add i32 %88, %89
  %91 = add i32 %89, %90
  %92 = add i32 %90, %91
  %93 = add i32 %91, %92
  %94 = add i32 %92, %93
  %95 = load i32, ptr @gvar
  %96 = add i32 %1, %95
  ret void
}
