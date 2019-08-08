; RUN: llc -mtriple arm64e-apple-darwin -asm-verbose=false -disable-post-ra -o - %s | FileCheck %s

; CHECK-LABEL: _test:
; CHECK-NEXT:  stp x20, x19, [sp, #-16]!
; CHECK-NEXT:  ; InlineAsm Start
; CHECK-NEXT:  ; InlineAsm End
; CHECK-NEXT:  mov  w0, #0
; CHECK-NEXT:  ldp x20, x19, [sp], #16
; CHECK-NEXT:  ret
define i32 @test() #0 {
  call void asm sideeffect "", "~{x19}"()
  ret i32 0
}

; CHECK-LABEL: _test_alloca:
; CHECK-NEXT:  sub sp, sp, #32
; CHECK-NEXT:  mov  x8, sp
; CHECK-NEXT:  ; InlineAsm Start
; CHECK-NEXT:  ; InlineAsm End
; CHECK-NEXT:  mov  w0, #0
; CHECK-NEXT:  add sp, sp, #32
; CHECK-NEXT:  ret
define i32 @test_alloca() #0 {
  %p = alloca i8, i32 32
  call void asm sideeffect "", "r"(i8* %p)
  ret i32 0
}

; CHECK-LABEL: _test_realign_alloca:
; CHECK-NEXT:   pacibsp
; CHECK-NEXT:   stp x29, x30, [sp, #-16]!
; CHECK-NEXT:   mov  x29, sp
; CHECK-NEXT:   sub x9, sp, #112
; CHECK-NEXT:   and sp, x9, #0xffffffffffffff80
; CHECK-NEXT:   mov  x8, sp
; CHECK-NEXT:   ; InlineAsm Start
; CHECK-NEXT:   ; InlineAsm End
; CHECK-NEXT:   mov  w0, #0
; CHECK-NEXT:   mov  sp, x29
; CHECK-NEXT:   ldp x29, x30, [sp], #16
; CHECK-NEXT:   retab
define i32 @test_realign_alloca() #0 {
  %p = alloca i8, i32 32, align 128
  call void asm sideeffect "", "r"(i8* %p)
  ret i32 0
}

; CHECK-LABEL: _test_big_alloca:
; CHECK-NEXT:  stp x28, x27, [sp, #-16]!
; CHECK-NEXT:  sub sp, sp, #1024
; CHECK-NEXT:  mov  x8, sp
; CHECK-NEXT:  ; InlineAsm Start
; CHECK-NEXT:  ; InlineAsm End
; CHECK-NEXT:  mov  w0, #0
; CHECK-NEXT:  add sp, sp, #1024
; CHECK-NEXT:  ldp x28, x27, [sp], #16
; CHECK-NEXT:  ret
define i32 @test_big_alloca() #0 {
  %p = alloca i8, i32 1024
  call void asm sideeffect "", "r"(i8* %p)
  ret i32 0
}

; CHECK-LABEL: _test_var_alloca:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  mov  x29, sp
; CHECK-NEXT:  mov  x8, sp
; CHECK-NEXT:  mov  w9, w0
; CHECK-NEXT:  add x9, x9, #15
; CHECK-NEXT:  and x9, x9, #0x1fffffff0
; CHECK-NEXT:  sub  x8, x8, x9
; CHECK-NEXT:  mov  sp, x8
; CHECK-NEXT:  ; InlineAsm Start
; CHECK-NEXT:  ; InlineAsm End
; CHECK-NEXT:  mov  w0, #0
; CHECK-NEXT:  mov  sp, x29
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_var_alloca(i32 %s) #0 {
  %p = alloca i8, i32 %s
  call void asm sideeffect "", "r"(i8* %p)
  ret i32 0
}

; CHECK-LABEL: _test_noframe_saved:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x28, x27, [sp, #-96]!
; CHECK-NEXT:  stp x26, x25, [sp, #16]
; CHECK-NEXT:  stp x24, x23, [sp, #32]
; CHECK-NEXT:  stp x22, x21, [sp, #48]
; CHECK-NEXT:  stp x20, x19, [sp, #64]
; CHECK-NEXT:  stp x29, x30, [sp, #80]
; CHECK-NEXT:  ldr w30, [x0]
; CHECK-NEXT:  ; InlineAsm Start
; CHECK-NEXT:  ; InlineAsm End
; CHECK-NEXT:  mov x0, x30
; CHECK-NEXT:  ldp x29, x30, [sp, #80]
; CHECK-NEXT:  ldp x20, x19, [sp, #64]
; CHECK-NEXT:  ldp x22, x21, [sp, #48]
; CHECK-NEXT:  ldp x24, x23, [sp, #32]
; CHECK-NEXT:  ldp x26, x25, [sp, #16]
; CHECK-NEXT:  ldp x28, x27, [sp], #96
; CHECK-NEXT:  retab
define i32 @test_noframe_saved(i32* %p) #0 {
  %v = load i32, i32* %p
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28}"()
  ret i32 %v
}

; CHECK-LABEL: _test_noframe:
; CHECK-NEXT:  ret
define void @test_noframe() #0 {
  ret void
}

; CHECK-LABEL: _test_returnaddress_0:
; CHECK-NEXT:  mov x0, x30
; CHECK-NEXT:  xpaci x0
; CHECK-NEXT:  ret
define i8* @test_returnaddress_0() #0 {
  %r = call i8* @llvm.returnaddress(i32 0)
  ret i8* %r
}

; CHECK-LABEL: _test_returnaddress_1:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  mov  x29, sp
; CHECK-NEXT:  ldr x8, [x29]
; CHECK-NEXT:  ldr x8, [x8, #8]
; CHECK-NEXT:  mov x0, x8
; CHECK-NEXT:  xpaci x0
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i8* @test_returnaddress_1() #0 {
  %r = call i8* @llvm.returnaddress(i32 1)
  ret i8* %r
}

; CHECK-LABEL: _test_noframe_alloca:
; CHECK-NEXT:  sub sp, sp, #16
; CHECK-NEXT:  add x8, sp, #15
; CHECK-NEXT:  ; InlineAsm Start
; CHECK-NEXT:  ; InlineAsm End
; CHECK-NEXT:  add sp, sp, #16
; CHECK-NEXT:  ret
define void @test_noframe_alloca() #0 {
  %p = alloca i8, i32 1
  call void asm sideeffect "", "r"(i8* %p)
  ret void
}

; CHECK-LABEL: _test_call:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  bl _bar
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define void @test_call() #0 {
  call i32 @bar()
  ret void
}

; CHECK-LABEL: _test_call_alloca:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  sub sp, sp, #32
; CHECK-NEXT:  stp x29, x30, [sp, #16]
; CHECK-NEXT:  bl _bar
; CHECK-NEXT:  ldp x29, x30, [sp, #16]
; CHECK-NEXT:  add sp, sp, #32
; CHECK-NEXT:  retab
define void @test_call_alloca() #0 {
  alloca i8
  call i32 @bar()
  ret void
}

; CHECK-LABEL: _test_call_shrinkwrapping:
; CHECK-NEXT:  tbz w0, #0, [[RETBB:LBB[0-9_]+]]
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  bl _bar
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  autibsp
; CHECK-NEXT: [[RETBB]]:
; CHECK-NEXT:  ret
define void @test_call_shrinkwrapping(i1 %c) #0 {
  br i1 %c, label %tbb, label %fbb
tbb:
  call i32 @bar()
  br label %fbb
fbb:
  ret void
}

; CHECK-LABEL: _test_tailcall:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  bl _bar
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  autibsp
; CHECK-NEXT:  b _bar
define i32 @test_tailcall() #0 {
  call i32 @bar()
  %c = tail call i32 @bar()
  ret i32 %c
}

; CHECK-LABEL: _test_tailcall_noframe:
; CHECK-NEXT:  b _bar
define i32 @test_tailcall_noframe() #0 {
  %c = tail call i32 @bar()
  ret i32 %c
}

declare i32 @bar()

declare i8* @llvm.returnaddress(i32)

attributes #0 = { nounwind "ptrauth-returns" }
