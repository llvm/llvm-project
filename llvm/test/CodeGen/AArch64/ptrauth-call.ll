; RUN: llc -mtriple arm64e-apple-darwin   -o - %s -asm-verbose=0 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DARWIN

; RUN: llc -mtriple aarch64 -mattr=+pauth -o - %s -asm-verbose=0 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,ELF

; RUN: llc -mtriple arm64e-apple-darwin   -o - %s -asm-verbose=0 \
; RUN:   -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:  | FileCheck %s --check-prefixes=CHECK,DARWIN

; RUN: llc -mtriple aarch64 -mattr=+pauth -o - %s -asm-verbose=0 \
; RUN:   -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:  | FileCheck %s --check-prefixes=CHECK,ELF


define i32 @test_call_ia_0(i32 ()* %arg0) #0 {
; DARWIN-LABEL: test_call_ia_0:
; DARWIN-NEXT:    stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:    blraaz x0
; DARWIN-NEXT:    ldp x29, x30, [sp], #16
; DARWIN-NEXT:    ret
;
; ELF-LABEL: test_call_ia_0:
; ELF-NEXT:    str x30, [sp, #-16]!
; ELF-NEXT:    blraaz x0
; ELF-NEXT:    ldr x30, [sp], #16
; ELF-NEXT:    ret
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 0, i64 0) ]
  ret i32 %tmp0
}

define i32 @test_call_ib_0(i32 ()* %arg0) #0 {
; DARWIN-LABEL: test_call_ib_0:
; DARWIN-NEXT:    stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:    blrabz x0
; DARWIN-NEXT:    ldp x29, x30, [sp], #16
; DARWIN-NEXT:    ret
;
; ELF-LABEL: test_call_ib_0:
; ELF-NEXT:    str x30, [sp, #-16]!
; ELF-NEXT:    blrabz x0
; ELF-NEXT:    ldr x30, [sp], #16
; ELF-NEXT:    ret
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 1, i64 0) ]
  ret i32 %tmp0
}

define i32 @test_tailcall_ia_0(i32 ()* %arg0) #0 {
; CHECK-LABEL: test_tailcall_ia_0:
; CHECK:    braaz x0
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 0) ]
  ret i32 %tmp0
}

define i32 @test_tailcall_ib_0(i32 ()* %arg0) #0 {
; CHECK-LABEL: test_tailcall_ib_0:
; CHECK:    brabz x0
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 0) ]
  ret i32 %tmp0
}

define i32 @test_call_ia_imm(i32 ()* %arg0) #0 {
; DARWIN-LABEL: test_call_ia_imm:
; DARWIN-NEXT:    stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:    mov x17, #42
; DARWIN-NEXT:    blraa x0, x17
; DARWIN-NEXT:    ldp x29, x30, [sp], #16
; DARWIN-NEXT:    ret
;
; ELF-LABEL: test_call_ia_imm:
; ELF-NEXT:    str x30, [sp, #-16]!
; ELF-NEXT:    mov x17, #42
; ELF-NEXT:    blraa x0, x17
; ELF-NEXT:    ldr x30, [sp], #16
; ELF-NEXT:    ret
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 0, i64 42) ]
  ret i32 %tmp0
}

define i32 @test_call_ib_imm(i32 ()* %arg0) #0 {
; DARWIN-LABEL: test_call_ib_imm:
; DARWIN-NEXT:    stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:    mov x17, #42
; DARWIN-NEXT:    blrab x0, x17
; DARWIN-NEXT:    ldp x29, x30, [sp], #16
; DARWIN-NEXT:    ret
;
; ELF-LABEL: test_call_ib_imm:
; ELF-NEXT:    str x30, [sp, #-16]!
; ELF-NEXT:    mov x17, #42
; ELF-NEXT:    blrab x0, x17
; ELF-NEXT:    ldr x30, [sp], #16
; ELF-NEXT:    ret
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 1, i64 42) ]
  ret i32 %tmp0
}

define i32 @test_tailcall_ia_imm(i32 ()* %arg0) #0 {
; CHECK-LABEL: test_tailcall_ia_imm:
; CHECK-NEXT:    mov x16, #42
; CHECK-NEXT:    braa x0, x16
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 42) ]
  ret i32 %tmp0
}

define i32 @test_tailcall_ib_imm(i32 ()* %arg0) #0 {
; CHECK-LABEL: test_tailcall_ib_imm:
; CHECK-NEXT:    mov x16, #42
; CHECK-NEXT:    brab x0, x16
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 42) ]
  ret i32 %tmp0
}

define i32 @test_call_ia_var(i32 ()* %arg0, i64* %arg1) #0 {
; DARWIN-LABEL: test_call_ia_var:
; DARWIN-NEXT:    stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:    ldr x8, [x1]
; DARWIN-NEXT:    blraa x0, x8
; DARWIN-NEXT:    ldp x29, x30, [sp], #16
; DARWIN-NEXT:    ret
;
; ELF-LABEL: test_call_ia_var:
; ELF-NEXT:    str x30, [sp, #-16]!
; ELF-NEXT:    ldr x8, [x1]
; ELF-NEXT:    blraa x0, x8
; ELF-NEXT:    ldr x30, [sp], #16
; ELF-NEXT:    ret
  %tmp0 = load i64, i64* %arg1
  %tmp1 = call i32 %arg0() [ "ptrauth"(i32 0, i64 %tmp0) ]
  ret i32 %tmp1
}

define i32 @test_call_ib_var(i32 ()* %arg0, i64* %arg1) #0 {
; DARWIN-LABEL: test_call_ib_var:
; DARWIN-NEXT:    stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:    ldr x8, [x1]
; DARWIN-NEXT:    blrab x0, x8
; DARWIN-NEXT:    ldp x29, x30, [sp], #16
; DARWIN-NEXT:    ret
;
; ELF-LABEL: test_call_ib_var:
; ELF-NEXT:    str x30, [sp, #-16]!
; ELF-NEXT:    ldr x8, [x1]
; ELF-NEXT:    blrab x0, x8
; ELF-NEXT:    ldr x30, [sp], #16
; ELF-NEXT:    ret
  %tmp0 = load i64, i64* %arg1
  %tmp1 = call i32 %arg0() [ "ptrauth"(i32 1, i64 %tmp0) ]
  ret i32 %tmp1
}

define i32 @test_tailcall_ia_var(i32 ()* %arg0, i64* %arg1) #0 {
; CHECK-LABEL: test_tailcall_ia_var:
; CHECK:    ldr x1, [x1]
; CHECK:    braa x0, x1
  %tmp0 = load i64, i64* %arg1
  %tmp1 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 %tmp0) ]
  ret i32 %tmp1
}

define i32 @test_tailcall_ib_var(i32 ()* %arg0, i64* %arg1) #0 {
; CHECK-LABEL: test_tailcall_ib_var:
; CHECK:    ldr x1, [x1]
; CHECK:    brab x0, x1
  %tmp0 = load i64, i64* %arg1
  %tmp1 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 %tmp0) ]
  ret i32 %tmp1
}

define i32 @test_call_ia_arg(i32 ()* %arg0, i64 %arg1) #0 {
; DARWIN-LABEL: test_call_ia_arg:
; DARWIN-NEXT:    stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:    blraa x0, x1
; DARWIN-NEXT:    ldp x29, x30, [sp], #16
; DARWIN-NEXT:    ret
;
; ELF-LABEL: test_call_ia_arg:
; ELF-NEXT:    str x30, [sp, #-16]!
; ELF-NEXT:    blraa x0, x1
; ELF-NEXT:    ldr x30, [sp], #16
; ELF-NEXT:    ret
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp0
}

define i32 @test_call_ib_arg(i32 ()* %arg0, i64 %arg1) #0 {
; DARWIN-LABEL: test_call_ib_arg:
; DARWIN-NEXT:    stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:    blrab x0, x1
; DARWIN-NEXT:    ldp x29, x30, [sp], #16
; DARWIN-NEXT:    ret
;
; ELF-LABEL: test_call_ib_arg:
; ELF-NEXT:    str x30, [sp, #-16]!
; ELF-NEXT:    blrab x0, x1
; ELF-NEXT:    ldr x30, [sp], #16
; ELF-NEXT:    ret
  %tmp0 = call i32 %arg0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp0
}

define i32 @test_tailcall_ia_arg(i32 ()* %arg0, i64 %arg1) #0 {
; CHECK-LABEL: test_tailcall_ia_arg:
; CHECK:    braa x0, x1
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp0
}

define i32 @test_tailcall_ib_arg(i32 ()* %arg0, i64 %arg1) #0 {
; CHECK-LABEL: test_tailcall_ib_arg:
; CHECK:    brab x0, x1
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp0
}

define i32 @test_call_ia_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
; DARWIN-LABEL: test_call_ia_arg_ind:
; DARWIN-NEXT:    stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:    ldr x8, [x0]
; DARWIN-NEXT:    blraa x8, x1
; DARWIN-NEXT:    ldp x29, x30, [sp], #16
; DARWIN-NEXT:    ret
;
; ELF-LABEL: test_call_ia_arg_ind:
; ELF-NEXT:    str x30, [sp, #-16]!
; ELF-NEXT:    ldr x8, [x0]
; ELF-NEXT:    blraa x8, x1
; ELF-NEXT:    ldr x30, [sp], #16
; ELF-NEXT:    ret
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = call i32 %tmp0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp1
}

define i32 @test_call_ib_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
; DARWIN-LABEL: test_call_ib_arg_ind:
; DARWIN-NEXT:    stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:    ldr x8, [x0]
; DARWIN-NEXT:    blrab x8, x1
; DARWIN-NEXT:    ldp x29, x30, [sp], #16
; DARWIN-NEXT:    ret
;
; ELF-LABEL: test_call_ib_arg_ind:
; ELF-NEXT:    str x30, [sp, #-16]!
; ELF-NEXT:    ldr x8, [x0]
; ELF-NEXT:    blrab x8, x1
; ELF-NEXT:    ldr x30, [sp], #16
; ELF-NEXT:    ret
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = call i32 %tmp0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp1
}

define i32 @test_tailcall_ia_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
; CHECK-LABEL: test_tailcall_ia_arg_ind:
; CHECK:    ldr x0, [x0]
; CHECK:    braa x0, x1
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = tail call i32 %tmp0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp1
}

define i32 @test_tailcall_ib_arg_ind(i32 ()** %arg0, i64 %arg1) #0 {
; CHECK-LABEL: test_tailcall_ib_arg_ind:
; CHECK:    ldr x0, [x0]
; CHECK:    brab x0, x1
  %tmp0 = load i32 ()*, i32 ()** %arg0
  %tmp1 = tail call i32 %tmp0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp1
}

attributes #0 = { nounwind }
