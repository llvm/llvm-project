; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Make sure that we don't generate a std r, 0(0) -- the memory address cannot
; be stored in r0.
; CHECK-LABEL: @test1
; CHECK-NOT: std {{[0-9]+}}, 0(0) 
; CHECK: blr

define void @test1({ i8*, void (i8*, i8*)* } %fn_arg) {
  %fn = alloca { i8*, void (i8*, i8*)* }
  %sp = alloca i8*, align 8
  %regs = alloca [18 x i64], align 8
  store { i8*, void (i8*, i8*)* } %fn_arg, { i8*, void (i8*, i8*)* }* %fn
  %1 = bitcast [18 x i64]* %regs to i64*
  call void asm sideeffect "std  14, $0", "=*m"(i64* %1)
  %2 = bitcast [18 x i64]* %regs to i8*
  %3 = getelementptr i8, i8* %2, i32 8
  %4 = bitcast i8* %3 to i64*
  call void asm sideeffect "std  15, $0", "=*m"(i64* %4)
  %5 = bitcast [18 x i64]* %regs to i8*
  %6 = getelementptr i8, i8* %5, i32 16
  %7 = bitcast i8* %6 to i64*
  call void asm sideeffect "std  16, $0", "=*m"(i64* %7)
  %8 = bitcast [18 x i64]* %regs to i8*
  %9 = getelementptr i8, i8* %8, i32 24
  %10 = bitcast i8* %9 to i64*
  call void asm sideeffect "std  17, $0", "=*m"(i64* %10)
  %11 = bitcast [18 x i64]* %regs to i8*
  %12 = getelementptr i8, i8* %11, i32 32
  %13 = bitcast i8* %12 to i64*
  call void asm sideeffect "std  18, $0", "=*m"(i64* %13)
  %14 = bitcast [18 x i64]* %regs to i8*
  %15 = getelementptr i8, i8* %14, i32 40
  %16 = bitcast i8* %15 to i64*
  call void asm sideeffect "std  19, $0", "=*m"(i64* %16)
  %17 = bitcast [18 x i64]* %regs to i8*
  %18 = getelementptr i8, i8* %17, i32 48
  %19 = bitcast i8* %18 to i64*
  call void asm sideeffect "std  20, $0", "=*m"(i64* %19)
  %20 = bitcast [18 x i64]* %regs to i8*
  %21 = getelementptr i8, i8* %20, i32 56
  %22 = bitcast i8* %21 to i64*
  call void asm sideeffect "std  21, $0", "=*m"(i64* %22)
  %23 = bitcast [18 x i64]* %regs to i8*
  %24 = getelementptr i8, i8* %23, i32 64
  %25 = bitcast i8* %24 to i64*
  call void asm sideeffect "std  22, $0", "=*m"(i64* %25)
  %26 = bitcast [18 x i64]* %regs to i8*
  %27 = getelementptr i8, i8* %26, i32 72
  %28 = bitcast i8* %27 to i64*
  call void asm sideeffect "std  23, $0", "=*m"(i64* %28)
  %29 = bitcast [18 x i64]* %regs to i8*
  %30 = getelementptr i8, i8* %29, i32 80
  %31 = bitcast i8* %30 to i64*
  call void asm sideeffect "std  24, $0", "=*m"(i64* %31)
  %32 = bitcast [18 x i64]* %regs to i8*
  %33 = getelementptr i8, i8* %32, i32 88
  %34 = bitcast i8* %33 to i64*
  call void asm sideeffect "std  25, $0", "=*m"(i64* %34)
  %35 = bitcast [18 x i64]* %regs to i8*
  %36 = getelementptr i8, i8* %35, i32 96
  %37 = bitcast i8* %36 to i64*
  call void asm sideeffect "std  26, $0", "=*m"(i64* %37)
  %38 = bitcast [18 x i64]* %regs to i8*
  %39 = getelementptr i8, i8* %38, i32 104
  %40 = bitcast i8* %39 to i64*
  call void asm sideeffect "std  27, $0", "=*m"(i64* %40)
  %41 = bitcast [18 x i64]* %regs to i8*
  %42 = getelementptr i8, i8* %41, i32 112
  %43 = bitcast i8* %42 to i64*
  call void asm sideeffect "std  28, $0", "=*m"(i64* %43)
  %44 = bitcast [18 x i64]* %regs to i8*
  %45 = getelementptr i8, i8* %44, i32 120
  %46 = bitcast i8* %45 to i64*
  call void asm sideeffect "std  29, $0", "=*m"(i64* %46)
  %47 = bitcast [18 x i64]* %regs to i8*
  %48 = getelementptr i8, i8* %47, i32 128
  %49 = bitcast i8* %48 to i64*
  call void asm sideeffect "std  30, $0", "=*m"(i64* %49)
  %50 = bitcast [18 x i64]* %regs to i8*
  %51 = getelementptr i8, i8* %50, i32 136
  %52 = bitcast i8* %51 to i64*
  call void asm sideeffect "std  31, $0", "=*m"(i64* %52)
  %53 = getelementptr { i8*, void (i8*, i8*)* }, { i8*, void (i8*, i8*)* }* %fn, i32 0, i32 1
  %.funcptr = load void (i8*, i8*)*, void (i8*, i8*)** %53
  %54 = getelementptr { i8*, void (i8*, i8*)* }, { i8*, void (i8*, i8*)* }* %fn, i32 0, i32 0
  %.ptr = load i8*, i8** %54
  %55 = load i8*, i8** %sp
  call void %.funcptr(i8* %.ptr, i8* %55)
  ret void
}

