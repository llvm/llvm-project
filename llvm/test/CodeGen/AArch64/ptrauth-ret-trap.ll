; RUN: llc -mtriple aarch64-linux-gnu -mattr=+pauth -asm-verbose=false -disable-post-ra -o - %s | FileCheck %s

; CHECK-LABEL:  test_tailcall:
; CHECK-NEXT:   pacibsp
; CHECK-NEXT:   str x30, [sp, #-16]!
; CHECK-NEXT:   bl bar
; CHECK-NEXT:   ldr x30, [sp], #16
; CHECK-NEXT:   autibsp
; CHECK-NEXT:   eor x16, x30, x30, lsl #1
; CHECK-NEXT:   tbnz x16, #62, [[BAD:.L.*]]
; CHECK-NEXT:   b bar
; CHECK-NEXT:   [[BAD]]:
; CHECK-NEXT:   brk #0xc471
define i32 @test_tailcall() #0 {
  call i32 @bar()
  %c = tail call i32 @bar()
  ret i32 %c
}

; CHECK-LABEL: test_tailcall_noframe:
; CHECK-NEXT:  b bar
define i32 @test_tailcall_noframe() #0 {
  %c = tail call i32 @bar()
  ret i32 %c
}

; CHECK-LABEL: test_tailcall_indirect:
; CHECK:         autibsp
; CHECK:         eor     x16, x30, x30, lsl #1
; CHECK:         tbnz    x16, #62, [[BAD:.L.*]]
; CHECK:         br      x0
; CHECK: [[BAD]]:
; CHECK:         brk     #0xc471
define void @test_tailcall_indirect(ptr %fptr) #0 {
  call i32 @test_tailcall()
  tail call void %fptr()
  ret void
}

; CHECK-LABEL: test_tailcall_indirect_in_x9:
; CHECK:         autibsp
; CHECK:         eor     x16, x30, x30, lsl #1
; CHECK:         tbnz    x16, #62, [[BAD:.L.*]]
; CHECK:         br      x9
; CHECK: [[BAD]]:
; CHECK:         brk     #0xc471
define void @test_tailcall_indirect_in_x9(ptr sret(i64) %ret, [8 x i64] %in, ptr %fptr) #0 {
  %ptr = alloca i8, i32 16
  call i32 @test_tailcall()
  tail call void %fptr(ptr sret(i64) %ret, [8 x i64] %in)
  ret void
}

; CHECK-LABEL: test_auth_tailcall_indirect:
; CHECK:         autibsp
; CHECK:         eor     x16, x30, x30, lsl #1
; CHECK:         tbnz    x16, #62, [[BAD:.L.*]]
; CHECK:         mov x16, #42
; CHECK:         braa      x0, x16
; CHECK: [[BAD]]:
; CHECK:         brk     #0xc471
define void @test_auth_tailcall_indirect(ptr %fptr) #0 {
  call i32 @test_tailcall()
  tail call void %fptr() [ "ptrauth"(i32 0, i64 42) ]
  ret void
}

; CHECK-LABEL: test_auth_tailcall_indirect_in_x9:
; CHECK:         autibsp
; CHECK:         eor     x16, x30, x30, lsl #1
; CHECK:         tbnz    x16, #62, [[BAD:.L.*]]
; CHECK:         brabz      x9
; CHECK: [[BAD]]:
; CHECK:         brk     #0xc471
define void @test_auth_tailcall_indirect_in_x9(ptr sret(i64) %ret, [8 x i64] %in, ptr %fptr) #0 {
  %ptr = alloca i8, i32 16
  call i32 @test_tailcall()
  tail call void %fptr(ptr sret(i64) %ret, [8 x i64] %in) [ "ptrauth"(i32 1, i64 0) ]
  ret void
}

; CHECK-LABEL: test_auth_tailcall_indirect_bti:
; CHECK:         autibsp
; CHECK:         eor     x17, x30, x30, lsl #1
; CHECK:         tbnz    x17, #62, [[BAD:.L.*]]
; CHECK:         brabz      x16
; CHECK: [[BAD]]:
; CHECK:         brk     #0xc471
define void @test_auth_tailcall_indirect_bti(ptr sret(i64) %ret, [8 x i64] %in, ptr %fptr) #0 "branch-target-enforcement"="true" {
  %ptr = alloca i8, i32 16
  call i32 @test_tailcall()
  tail call void %fptr(ptr sret(i64) %ret, [8 x i64] %in) [ "ptrauth"(i32 1, i64 0) ]
  ret void
}

declare i32 @bar()

attributes #0 = { nounwind "ptrauth-returns" "ptrauth-auth-traps" }
