; RUN: llc -mtriple arm64e-apple-darwin -mattr=+bti              -asm-verbose=false \
; RUN:   -o - %s | FileCheck %s --check-prefixes=DARWIN,CHECK
; RUN: llc -mtriple arm64e-apple-darwin -mattr=+bti -global-isel -asm-verbose=false \
; RUN:   -o - %s | FileCheck %s --check-prefixes=DARWIN,CHECK
; RUN: llc -mtriple arm64e-apple-darwin -mattr=+bti -fast-isel   -asm-verbose=false \
; RUN:   -o - %s | FileCheck %s --check-prefixes=DARWIN,CHECK
; RUN: llc -mtriple aarch64-linux-gnu   -mattr=+bti -mattr=+pauth              -asm-verbose=false \
; RUN:   -o - %s | FileCheck %s --check-prefixes=ELF,CHECK
; RUN: llc -mtriple aarch64-linux-gnu   -mattr=+bti -mattr=+pauth -global-isel -asm-verbose=false \
; RUN:   -o - %s | FileCheck %s --check-prefixes=ELF,CHECK
; RUN: llc -mtriple aarch64-linux-gnu   -mattr=+bti -mattr=+pauth -fast-isel   -asm-verbose=false \
; RUN:   -o - %s | FileCheck %s --check-prefixes=ELF,CHECK

; ptrauth tail-calls can only use x16/x17 with BTI.

; CHECK-LABEL: test_tailcall_ia_0:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  braaz x16
define i32 @test_tailcall_ia_0(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 0) ]
  ret i32 %tmp0
}

; CHECK-LABEL: test_tailcall_ib_0:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  brabz x16
define i32 @test_tailcall_ib_0(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 0) ]
  ret i32 %tmp0
}

; CHECK-LABEL: test_tailcall_ia_imm:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  mov x17, #42
; CHECK-NEXT:  braa x16, x17
define i32 @test_tailcall_ia_imm(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 42) ]
  ret i32 %tmp0
}

; CHECK-LABEL: test_tailcall_ib_imm:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  mov x17, #42
; CHECK-NEXT:  brab x16, x17
define i32 @test_tailcall_ib_imm(i32 ()* %arg0) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 42) ]
  ret i32 %tmp0
}

; CHECK-LABEL: test_tailcall_ia_var:
; DARWIN-NEXT: bti c
; DARWIN-NEXT: mov x16, x0
; DARWIN-NEXT: ldr x0, [x1]
; DARWIN-NEXT: braa x16, x0
; ELF-NEXT:    bti c
; ELF-NEXT:    ldr x1, [x1]
; ELF-NEXT:    mov x16, x0
; ELF-NEXT:    braa x16, x1
define i32 @test_tailcall_ia_var(i32 ()* %arg0, i64* %arg1) #0 {
  %tmp0 = load i64, i64* %arg1
  %tmp1 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 %tmp0) ]
  ret i32 %tmp1
}

; CHECK-LABEL: test_tailcall_ib_var:
; DARWIN-NEXT: bti c
; DARWIN-NEXT: mov x16, x0
; DARWIN-NEXT: ldr x0, [x1]
; DARWIN-NEXT: brab x16, x0
; ELF-NEXT:    bti c
; ELF-NEXT:    ldr x1, [x1]
; ELF-NEXT:    mov x16, x0
; ELF-NEXT:    brab x16, x1
define i32 @test_tailcall_ib_var(i32 ()* %arg0, i64* %arg1) #0 {
  %tmp0 = load i64, i64* %arg1
  %tmp1 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 %tmp0) ]
  ret i32 %tmp1
}

; CHECK-LABEL: test_tailcall_ia_arg:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  braa x16, x1
define i32 @test_tailcall_ia_arg(i32 ()* %arg0, i64 %arg1) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp0
}

; CHECK-LABEL: test_tailcall_ib_arg:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  mov x16, x0
; CHECK-NEXT:  brab x16, x1
define i32 @test_tailcall_ib_arg(i32 ()* %arg0, i64 %arg1) #0 {
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp0
}

; CHECK-LABEL: test_tailcall_ia_arg_ind:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  ldr x16, [x0]
; CHECK-NEXT:  braa x16, x1
define i32 @test_tailcall_ia_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = tail call i32 %tmp0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp1
}

; CHECK-LABEL: test_tailcall_ib_arg_ind:
; CHECK-NEXT:  bti c
; CHECK-NEXT:  ldr x16, [x0]
; CHECK-NEXT:  brab x16, x1
define i32 @test_tailcall_ib_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = tail call i32 %tmp0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp1
}

attributes #0 = { nounwind "branch-target-enforcement"="true" }
