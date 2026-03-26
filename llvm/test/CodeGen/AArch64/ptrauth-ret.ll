; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -verify-machineinstrs -disable-post-ra \
; RUN:   -global-isel=0 -o - %s | FileCheck %s
; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -verify-machineinstrs -disable-post-ra \
; RUN:   -global-isel=1 -global-isel-abort=1 -o - %s | FileCheck %s

define i32 @test() #0 {
; CHECK-LABEL: test:
  call void asm sideeffect "", "~{x19}"()
  ret i32 0
}

define i32 @test_alloca() #0 {
; CHECK-LABEL: test_alloca:
  %p = alloca i8, i32 32
  call void asm sideeffect "", "r"(ptr %p)
  ret i32 0
}

define i32 @test_realign_alloca() #0 {
; CHECK-LABEL: test_realign_alloca:
  %p = alloca i8, i32 32, align 128
  call void asm sideeffect "", "r"(ptr %p)
  ret i32 0
}

define i32 @test_big_alloca() #0 {
; CHECK-LABEL: test_big_alloca:
  %p = alloca i8, i32 1024
  call void asm sideeffect "", "r"(ptr %p)
  ret i32 0
}

define i32 @test_var_alloca(i32 %s) #0 {
  %p = alloca i8, i32 %s
  call void asm sideeffect "", "r"(ptr %p)
  ret i32 0
}

define i32 @test_noframe_saved(ptr %p) #0 {
; CHECK-LABEL: test_noframe_saved:
  %v = load i32, ptr %p
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28}"()
  ret i32 %v
}

define void @test_noframe() #0 {
; CHECK-LABEL: test_noframe:
  ret void
}

; FIXME: Inefficient lowering of @llvm.returnaddress
define ptr @test_returnaddress_0() #0 {
; CHECK-LABEL: test_returnaddress_0:
  %r = call ptr @llvm.returnaddress(i32 0)
  ret ptr %r
}

define ptr @test_returnaddress_1() #0 {
; CHECK-LABEL: test_returnaddress_1:
  %r = call ptr @llvm.returnaddress(i32 1)
  ret ptr %r
}

define void @test_noframe_alloca() #0 {
; CHECK-LABEL: test_noframe_alloca:
  %p = alloca i8, i32 1
  call void asm sideeffect "", "r"(ptr %p)
  ret void
}

define void @test_call() #0 {
; CHECK-LABEL: test_call:
  call i32 @bar()
  ret void
}

define void @test_call_alloca() #0 {
; CHECK-LABEL: test_call_alloca:
  alloca i8
  call i32 @bar()
  ret void
}

define void @test_call_shrinkwrapping(i1 %c) #0 {
; CHECK-LABEL: test_call_shrinkwrapping:
  br i1 %c, label %tbb, label %fbb
tbb:
  call i32 @bar()
  br label %fbb
fbb:
  ret void
}

define i32 @test_tailcall() #0 {
; CHECK-LABEL: test_tailcall:
  call i32 @bar()
  %c = tail call i32 @bar()
  ret i32 %c
}

define i32 @test_tailcall_noframe() #0 {
; CHECK-LABEL: test_tailcall_noframe:
  %c = tail call i32 @bar()
  ret i32 %c
}

declare i32 @bar()

declare ptr @llvm.returnaddress(i32)

attributes #0 = { nounwind "ptrauth-returns" }
