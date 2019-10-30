; RUN: llc -mtriple arm64e-apple-darwin            -asm-verbose=false -o - %s | FileCheck %s
; RUN: llc -mtriple arm64e-apple-darwin -fast-isel -asm-verbose=false -o - %s | FileCheck %s

; CHECK-LABEL: _test_call_ia_0:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  blraaz x0
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_call_ia_0(i32 ()* %arg0) #0 {
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 0, i64 0) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_call_ib_0:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  blrabz x0
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_call_ib_0(i32 ()* %arg0) #0 {
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 1, i64 0) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ia_0:
; CHECK-NEXT:  braaz x0
define i32 @test_tailcall_ia_0(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 0) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ib_0:
; CHECK-NEXT:  brabz x0
define i32 @test_tailcall_ib_0(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 0) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_call_ia_imm:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  mov w8, #42
; CHECK-NEXT:  blraa x0, x8
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_call_ia_imm(i32 ()* %arg0) #0 {
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 0, i64 42) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_call_ib_imm:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  mov w8, #42
; CHECK-NEXT:  blrab x0, x8
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_call_ib_imm(i32 ()* %arg0) #0 {
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 1, i64 42) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ia_imm:
; CHECK-NEXT:  mov w1, #42
; CHECK-NEXT:  braa x0, x1
define i32 @test_tailcall_ia_imm(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 42) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ib_imm:
; CHECK-NEXT:  mov w1, #42
; CHECK-NEXT:  brab x0, x1
define i32 @test_tailcall_ib_imm(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 42) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_call_ia_var:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  ldr x8, [x1]
; CHECK-NEXT:  blraa x0, x8
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_call_ia_var(i32 ()* %arg0, i64* %arg1) #0 {
  %tmp0 = load i64, i64* %arg1
  %tmp1 = call i32 %arg0() [ "ptrauth"(i32 0, i64 %tmp0) ]
  ret i32 %tmp1
}

; CHECK-LABEL: _test_call_ib_var:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  ldr x8, [x1]
; CHECK-NEXT:  blrab x0, x8
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_call_ib_var(i32 ()* %arg0, i64* %arg1) #0 {
  %tmp0 = load i64, i64* %arg1
  %tmp1 = call i32 %arg0() [ "ptrauth"(i32 1, i64 %tmp0) ]
  ret i32 %tmp1
}

; CHECK-LABEL: _test_tailcall_ia_var:
; CHECK-NEXT:  ldr x1, [x1]
; CHECK-NEXT:  braa x0, x1
define i32 @test_tailcall_ia_var(i32 ()* %arg0, i64* %arg1) #0 {
  %tmp0 = load i64, i64* %arg1
  %tmp1 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 %tmp0) ]
  ret i32 %tmp1
}

; CHECK-LABEL: _test_tailcall_ib_var:
; CHECK-NEXT:  ldr x1, [x1]
; CHECK-NEXT:  brab x0, x1
define i32 @test_tailcall_ib_var(i32 ()* %arg0, i64* %arg1) #0 {
  %tmp0 = load i64, i64* %arg1
  %tmp1 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 %tmp0) ]
  ret i32 %tmp1
}

; CHECK-LABEL: _test_call_ia_arg:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  blraa x0, x1
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_call_ia_arg(i32 ()* %arg0, i64 %arg1) #0 {
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_call_ib_arg:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  blrab x0, x1
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_call_ib_arg(i32 ()* %arg0, i64 %arg1) #0 {
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ia_arg:
; CHECK-NEXT:  braa x0, x1
define i32 @test_tailcall_ia_arg(i32 ()* %arg0, i64 %arg1) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ib_arg:
; CHECK-NEXT:  brab x0, x1
define i32 @test_tailcall_ib_arg(i32 ()* %arg0, i64 %arg1) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_call_ia_arg_ind:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  ldr x8, [x0]
; CHECK-NEXT:  blraa x8, x1
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_call_ia_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = call i32 %tmp0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp1
}

; CHECK-LABEL: _test_call_ib_arg_ind:
; CHECK-NEXT:  pacibsp
; CHECK-NEXT:  stp x29, x30, [sp, #-16]!
; CHECK-NEXT:  ldr x8, [x0]
; CHECK-NEXT:  blrab x8, x1
; CHECK-NEXT:  ldp x29, x30, [sp], #16
; CHECK-NEXT:  retab
define i32 @test_call_ib_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = call i32 %tmp0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp1
}

; CHECK-LABEL: _test_tailcall_ia_arg_ind:
; CHECK-NEXT:  ldr x0, [x0]
; CHECK-NEXT:  braa x0, x1
define i32 @test_tailcall_ia_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = tail call i32 %tmp0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp1
}

; CHECK-LABEL: _test_tailcall_ib_arg_ind:
; CHECK-NEXT:  ldr x0, [x0]
; CHECK-NEXT:  brab x0, x1
define i32 @test_tailcall_ib_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = tail call i32 %tmp0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp1
}

attributes #0 = { nounwind "ptrauth-returns" }
