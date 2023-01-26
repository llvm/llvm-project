; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Make sure that we don't generate a std r, 0(0) -- the memory address cannot
; be stored in r0.
; CHECK-LABEL: @test1
; CHECK-NOT: std {{[0-9]+}}, 0(0) 
; CHECK: blr

define void @test1({ ptr, ptr } %fn_arg) {
  %fn = alloca { ptr, ptr }
  %sp = alloca ptr, align 8
  %regs = alloca [18 x i64], align 8
  store { ptr, ptr } %fn_arg, ptr %fn
  call void asm sideeffect "std  14, $0", "=*m"(ptr elementtype(i64) %regs)
  %1 = getelementptr i8, ptr %regs, i32 8
  call void asm sideeffect "std  15, $0", "=*m"(ptr elementtype(i64) %1)
  %2 = getelementptr i8, ptr %regs, i32 16
  call void asm sideeffect "std  16, $0", "=*m"(ptr elementtype(i64) %2)
  %3 = getelementptr i8, ptr %regs, i32 24
  call void asm sideeffect "std  17, $0", "=*m"(ptr elementtype(i64) %3)
  %4 = getelementptr i8, ptr %regs, i32 32
  call void asm sideeffect "std  18, $0", "=*m"(ptr elementtype(i64) %4)
  %5 = getelementptr i8, ptr %regs, i32 40
  call void asm sideeffect "std  19, $0", "=*m"(ptr elementtype(i64) %5)
  %6 = getelementptr i8, ptr %regs, i32 48
  call void asm sideeffect "std  20, $0", "=*m"(ptr elementtype(i64) %6)
  %7 = getelementptr i8, ptr %regs, i32 56
  call void asm sideeffect "std  21, $0", "=*m"(ptr elementtype(i64) %7)
  %8 = getelementptr i8, ptr %regs, i32 64
  call void asm sideeffect "std  22, $0", "=*m"(ptr elementtype(i64) %8)
  %9 = getelementptr i8, ptr %regs, i32 72
  call void asm sideeffect "std  23, $0", "=*m"(ptr elementtype(i64) %9)
  %10 = getelementptr i8, ptr %regs, i32 80
  call void asm sideeffect "std  24, $0", "=*m"(ptr elementtype(i64) %10)
  %11 = getelementptr i8, ptr %regs, i32 88
  call void asm sideeffect "std  25, $0", "=*m"(ptr elementtype(i64) %11)
  %12 = getelementptr i8, ptr %regs, i32 96
  call void asm sideeffect "std  26, $0", "=*m"(ptr elementtype(i64) %12)
  %13 = getelementptr i8, ptr %regs, i32 104
  call void asm sideeffect "std  27, $0", "=*m"(ptr elementtype(i64) %13)
  %14 = getelementptr i8, ptr %regs, i32 112
  call void asm sideeffect "std  28, $0", "=*m"(ptr elementtype(i64) %14)
  %15 = getelementptr i8, ptr %regs, i32 120
  call void asm sideeffect "std  29, $0", "=*m"(ptr elementtype(i64) %15)
  %16 = getelementptr i8, ptr %regs, i32 128
  call void asm sideeffect "std  30, $0", "=*m"(ptr elementtype(i64) %16)
  %17 = getelementptr i8, ptr %regs, i32 136
  call void asm sideeffect "std  31, $0", "=*m"(ptr elementtype(i64) %17)
  %18 = getelementptr { ptr, ptr }, ptr %fn, i32 0, i32 1
  %.funcptr = load ptr, ptr %18
  %19 = getelementptr { ptr, ptr }, ptr %fn, i32 0, i32 0
  %.ptr = load ptr, ptr %19
  %20 = load ptr, ptr %sp
  call void %.funcptr(ptr %.ptr, ptr %20)
  ret void
}

