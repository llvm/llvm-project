; RUN: llc -mtriple arm64e-apple-darwin -mattr=+bti              -asm-verbose=false -o - %s | FileCheck %s
; RUN: llc -mtriple arm64e-apple-darwin -mattr=+bti -global-isel -asm-verbose=false -o - %s | FileCheck %s
; RUN: llc -mtriple arm64e-apple-darwin -mattr=+bti -fast-isel   -asm-verbose=false -o - %s | FileCheck %s

; ptrauth tail-calls can only use x16/x17 with BTI.

; CHECK-LABEL: _test_tailcall_ia_0:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  braaz x16
define i32 @test_tailcall_ia_0(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 0) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ib_0:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  brabz x16
define i32 @test_tailcall_ib_0(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 0) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ia_imm:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  mov x17, #42
; CHECK-NEXT:  braa x16, x17
define i32 @test_tailcall_ia_imm(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 42) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ib_imm:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  mov x17, #42
; CHECK-NEXT:  brab x16, x17
define i32 @test_tailcall_ib_imm(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 42) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ia_var:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  ldr x0, [x1]
; CHECK-NEXT:  braa x16, x0
define i32 @test_tailcall_ia_var(i32 ()* %arg0, i64* %arg1) #0 {
  %tmp0 = load i64, i64* %arg1
  %tmp1 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 %tmp0) ]
  ret i32 %tmp1
}

; CHECK-LABEL: _test_tailcall_ib_var:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  ldr x0, [x1]
; CHECK-NEXT:  brab x16, x0
define i32 @test_tailcall_ib_var(i32 ()* %arg0, i64* %arg1) #0 {
  %tmp0 = load i64, i64* %arg1
  %tmp1 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 %tmp0) ]
  ret i32 %tmp1
}

; CHECK-LABEL: _test_tailcall_ia_arg:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  braa x16, x1
define i32 @test_tailcall_ia_arg(i32 ()* %arg0, i64 %arg1) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ib_arg:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  brab x16, x1
define i32 @test_tailcall_ib_arg(i32 ()* %arg0, i64 %arg1) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp0
}

; CHECK-LABEL: _test_tailcall_ia_arg_ind:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  ldr x16, [x0]
; CHECK-NEXT:  braa x16, x1
define i32 @test_tailcall_ia_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = tail call i32 %tmp0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp1
}

; CHECK-LABEL: _test_tailcall_ib_arg_ind:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  ldr x16, [x0]
; CHECK-NEXT:  brab x16, x1
define i32 @test_tailcall_ib_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = tail call i32 %tmp0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp1
}

attributes #0 = { nounwind "branch-target-enforcement"="true" }
