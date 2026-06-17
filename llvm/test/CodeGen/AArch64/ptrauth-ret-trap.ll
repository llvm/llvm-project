; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: cat common.ll authflag-aa.ll > common-aa.ll

; RUN: llc -mtriple aarch64-linux-gnu -mattr=+pauth -asm-verbose=false -disable-post-ra -o - common.ll | FileCheck --check-prefixes=CHECK,CHECKB %s
; RUN: llc -mtriple aarch64-linux-gnu -mattr=+pauth -asm-verbose=false -disable-post-ra -o - common-aa.ll | FileCheck --check-prefixes=CHECK,CHECKA %s

;--- common.ll

; CHECK-LABEL:  test_tailcall:
; CHECKB-NEXT:  pacibsp
; CHECKA-NEXT:  paciasp
; CHECK-NEXT:   str x30, [sp, #-16]!
; CHECK-NEXT:   bl bar
; CHECK-NEXT:   ldr x30, [sp], #16
; CHECKB-NEXT:  autibsp
; CHECKA-NEXT:  autiasp
; CHECK-NEXT:   eor x16, x30, x30, lsl #1
; CHECK-NEXT:   tbz x16, #62, [[GOOD:.L.*]]
; CHECKB-NEXT:  brk #0xc471
; CHECKA-NEXT:  brk #0xc470
; CHECK-NEXT:   [[GOOD]]:
; CHECK-NEXT:   b bar
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
; CHECKB:        autibsp
; CHECKA:        autiasp
; CHECK:         eor     x16, x30, x30, lsl #1
; CHECK:         tbz     x16, #62, [[GOOD:.L.*]]
; CHECKB:        brk     #0xc471
; CHECKA:        brk     #0xc470
; CHECK: [[GOOD]]:
; CHECK:         br      x0
define void @test_tailcall_indirect(ptr %fptr) #0 {
  call i32 @test_tailcall()
  tail call void %fptr()
  ret void
}

; CHECK-LABEL: test_tailcall_indirect_in_x9:
; CHECKB:        autibsp
; CHECKA:        autiasp
; CHECK:         eor     x16, x30, x30, lsl #1
; CHECK:         tbz     x16, #62, [[GOOD:.L.*]]
; CHECKB:        brk     #0xc471
; CHECKA:        brk     #0xc470
; CHECK: [[GOOD]]:
; CHECK:         br      x9
define void @test_tailcall_indirect_in_x9(ptr sret(i64) %ret, [8 x i64] %in, ptr %fptr) #0 {
  %ptr = alloca i8, i32 16
  call i32 @test_tailcall()
  tail call void %fptr(ptr sret(i64) %ret, [8 x i64] %in)
  ret void
}

; CHECK-LABEL: test_auth_tailcall_indirect:
; CHECKB:        autibsp
; CHECKA:        autiasp
; CHECK:         eor     x16, x30, x30, lsl #1
; CHECK:         tbz     x16, #62, [[GOOD:.L.*]]
; CHECKB:        brk     #0xc471
; CHECKA:        brk     #0xc470
; CHECK: [[GOOD]]:
; CHECK:         mov x16, #42
; CHECK:         braa      x0, x16
define void @test_auth_tailcall_indirect(ptr %fptr) #0 {
  call i32 @test_tailcall()
  tail call void %fptr() [ "ptrauth"(i32 0, i64 42) ]
  ret void
}

; CHECK-LABEL: test_auth_tailcall_indirect_in_x9:
; CHECKB:        autibsp
; CHECKA:        autiasp
; CHECK:         eor     x16, x30, x30, lsl #1
; CHECK:         tbz     x16, #62, [[GOOD:.L.*]]
; CHECKB:        brk     #0xc471
; CHECKA:        brk     #0xc470
; CHECK: [[GOOD]]:
; CHECK:         brabz      x9
define void @test_auth_tailcall_indirect_in_x9(ptr sret(i64) %ret, [8 x i64] %in, ptr %fptr) #0 {
  %ptr = alloca i8, i32 16
  call i32 @test_tailcall()
  tail call void %fptr(ptr sret(i64) %ret, [8 x i64] %in) [ "ptrauth"(i32 1, i64 0) ]
  ret void
}

; CHECK-LABEL: test_auth_tailcall_indirect_bti:
; CHECKB:        autibsp
; CHECKA:        autiasp
; CHECK:         eor     x17, x30, x30, lsl #1
; CHECK:         tbz     x17, #62, [[GOOD:.L.*]]
; CHECKB:        brk     #0xc471
; CHECKA:        brk     #0xc470
; CHECK: [[GOOD]]:
; CHECK:         brabz      x16
define void @test_auth_tailcall_indirect_bti(ptr sret(i64) %ret, [8 x i64] %in, ptr %fptr) #0 "branch-target-enforcement"="true" {
  %ptr = alloca i8, i32 16
  call i32 @test_tailcall()
  tail call void %fptr(ptr sret(i64) %ret, [8 x i64] %in) [ "ptrauth"(i32 1, i64 0) ]
  ret void
}

declare i32 @bar()

attributes #0 = { nounwind "ptrauth-returns" "ptrauth-auth-traps" }

;--- authflag-aa.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ptrauth-keys", !"iadaiada"}
