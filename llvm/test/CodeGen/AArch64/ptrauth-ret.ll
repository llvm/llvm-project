; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -verify-machineinstrs -disable-post-ra \
; RUN:   -global-isel=0 -o - %s | FileCheck %s
; RUN: llc < %s -mtriple aarch64-linux-gnu -mattr=+pauth -verify-machineinstrs -disable-post-ra \
; RUN:   -global-isel=1 -global-isel-abort=1 -o - %s | FileCheck %s

define i32 @test() #0 {
; CHECK-LABEL: test:
; CHECK:       %bb.0:
; CHECK-NEXT:    str x19, [sp, #-16]!
; CHECK-NEXT:    mov w0, wzr
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    ldr x19, [sp], #16
; CHECK-NEXT:    ret
  call void asm sideeffect "", "~{x19}"()
  ret i32 0
}

define i32 @test_alloca() #0 {
; CHECK-LABEL: test_alloca:
; CHECK:       %bb.0:
; CHECK-NEXT:    sub sp, sp, #32
; CHECK-NEXT:    mov x8, sp
; CHECK-NEXT:    mov w0, wzr
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    add sp, sp, #32
; CHECK-NEXT:    ret
  %p = alloca i8, i32 32
  call void asm sideeffect "", "r"(ptr %p)
  ret i32 0
}

define i32 @test_realign_alloca() #0 {
; CHECK-LABEL: test_realign_alloca:
; CHECK:       %bb.0:
; CHECK-NEXT:    pacibsp
; CHECK-NEXT:    stp x29, x30, [sp, #-16]!
; CHECK-NEXT:    mov x29, sp
; CHECK-NEXT:    sub x9, sp, #112
; CHECK-NEXT:    and sp, x9, #0xffffffffffffff80
; CHECK-NEXT:    mov x8, sp
; CHECK-NEXT:    mov w0, wzr
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    mov sp, x29
; CHECK-NEXT:    ldp x29, x30, [sp], #16
; CHECK-NEXT:    retab
  %p = alloca i8, i32 32, align 128
  call void asm sideeffect "", "r"(ptr %p)
  ret i32 0
}

define i32 @test_big_alloca() #0 {
; CHECK-LABEL: test_big_alloca:
; CHECK:       %bb.0:
; CHECK-NEXT:    str x29, [sp, #-16]!
; CHECK-NEXT:    sub sp, sp, #1024
; CHECK-NEXT:    mov x8, sp
; CHECK-NEXT:    mov w0, wzr
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    add sp, sp, #1024
; CHECK-NEXT:    ldr x29, [sp], #16
; CHECK-NEXT:    ret
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
; CHECK:       %bb.0:


; CHECK-NEXT:  str     x29, [sp, #-96]!
; CHECK-NEXT:  stp     x28, x27, [sp, #16]
; CHECK-NEXT:  stp     x26, x25, [sp, #32]
; CHECK-NEXT:  stp     x24, x23, [sp, #48]
; CHECK-NEXT:  stp     x22, x21, [sp, #64]
; CHECK-NEXT:  stp     x20, x19, [sp, #80]
; CHECK-NEXT:  ldr     w29, [x0]
; CHECK-NEXT:  //APP
; CHECK-NEXT:  //NO_APP
; CHECK-NEXT:  mov     w0, w29
; CHECK-NEXT:  ldp     x20, x19, [sp, #80]
; CHECK-NEXT:  ldp     x22, x21, [sp, #64]
; CHECK-NEXT:  ldp     x24, x23, [sp, #48]
; CHECK-NEXT:  ldp     x26, x25, [sp, #32]
; CHECK-NEXT:  ldp     x28, x27, [sp, #16]
; CHECK-NEXT:  ldr     x29, [sp], #96
; CHECK-NEXT:  ret
  %v = load i32, ptr %p
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28}"()
  ret i32 %v
}

define void @test_noframe() #0 {
; CHECK-LABEL: test_noframe:
; CHECK:       %bb.0:
; CHECK-NEXT:    ret
  ret void
}

; FIXME: Inefficient lowering of @llvm.returnaddress
define ptr @test_returnaddress_0() #0 {
; CHECK-LABEL: test_returnaddress_0:
; CHECK:       %bb.0:
; CHECK-NEXT:    pacibsp
; CHECK-NEXT:    str x30, [sp, #-16]!
; CHECK-NEXT:    xpaci x30
; CHECK-NEXT:    mov x0, x30
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    retab
  %r = call ptr @llvm.returnaddress(i32 0)
  ret ptr %r
}

define ptr @test_returnaddress_1() #0 {
; CHECK-LABEL: test_returnaddress_1:
; CHECK:       %bb.0:
; CHECK-NEXT:    pacibsp
; CHECK-NEXT:    stp x29, x30, [sp, #-16]!
; CHECK-NEXT:    mov x29, sp
; CHECK-NEXT:    ldr x8, [x29]
; CHECK-NEXT:    ldr x0, [x8, #8]
; CHECK-NEXT:    xpaci x0
; CHECK-NEXT:    ldp x29, x30, [sp], #16
; CHECK-NEXT:    retab
  %r = call ptr @llvm.returnaddress(i32 1)
  ret ptr %r
}

define void @test_noframe_alloca() #0 {
; CHECK-LABEL: test_noframe_alloca:
; CHECK:       %bb.0:
; CHECK-NEXT:    sub sp, sp, #16
; CHECK-NEXT:    add x8, sp, #12
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    add sp, sp, #16
; CHECK-NEXT:    ret
  %p = alloca i8, i32 1
  call void asm sideeffect "", "r"(ptr %p)
  ret void
}

define void @test_call() #0 {
; CHECK-LABEL: test_call:
; CHECK:       %bb.0:
; CHECK-NEXT:    pacibsp
; CHECK-NEXT:    str x30, [sp, #-16]!
; CHECK-NEXT:    bl bar
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    retab
  call i32 @bar()
  ret void
}

define void @test_call_alloca() #0 {
; CHECK-LABEL: test_call_alloca:
; CHECK:       %bb.0:
; CHECK-NEXT:    pacibsp
; CHECK-NEXT:    str x30, [sp, #-16]
; CHECK-NEXT:    bl bar
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    retab
  alloca i8
  call i32 @bar()
  ret void
}

define void @test_call_shrinkwrapping(i1 %c) #0 {
; CHECK-LABEL: test_call_shrinkwrapping:
; CHECK:       %bb.0:
; CHECK-NEXT:    tbz w0, #0, .LBB12_2
; CHECK-NEXT:  %bb.1:
; CHECK-NEXT:    pacibsp
; CHECK-NEXT:    str x30, [sp, #-16]!
; CHECK-NEXT:    bl bar
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    autibsp
; CHECK-NEXT:  LBB12_2:
; CHECK-NEXT:    ret
  br i1 %c, label %tbb, label %fbb
tbb:
  call i32 @bar()
  br label %fbb
fbb:
  ret void
}

define i32 @test_tailcall() #0 {
; CHECK-LABEL: test_tailcall:
; CHECK:       %bb.0:
; CHECK-NEXT:    pacibsp
; CHECK-NEXT:    str x30, [sp, #-16]!
; CHECK-NEXT:    bl bar
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    autibsp
; CHECK-NEXT:    b bar
  call i32 @bar()
  %c = tail call i32 @bar()
  ret i32 %c
}

define i32 @test_tailcall_noframe() #0 {
; CHECK-LABEL: test_tailcall_noframe:
; CHECK:       %bb.0:
; CHECK-NEXT:    b bar
  %c = tail call i32 @bar()
  ret i32 %c
}

declare i32 @bar()

declare ptr @llvm.returnaddress(i32)

attributes #0 = { nounwind "ptrauth-returns" }
