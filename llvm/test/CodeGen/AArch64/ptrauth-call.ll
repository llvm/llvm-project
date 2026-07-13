; RUN: llc -mtriple arm64e-apple-darwin   -o - %s -asm-verbose=0 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DARWIN

; RUN: llc -mtriple aarch64 -mattr=+pauth -o - %s -asm-verbose=0 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,ELF

; RUN: llc -mtriple arm64e-apple-darwin   -o - %s -asm-verbose=0 \
; RUN:   -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DARWIN

; RUN: llc -mtriple aarch64 -mattr=+pauth -o - %s -asm-verbose=0 \
; RUN:   -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:   | FileCheck %s --check-prefixes=CHECK,ELF


define i32 @test_call_ia_0(ptr %arg0) #0 {
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

define i32 @test_call_ib_0(ptr %arg0) #0 {
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

define i32 @test_tailcall_ia_0(ptr %arg0) #0 {
; CHECK-LABEL: test_tailcall_ia_0:
; CHECK-NEXT:    braaz x0
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 0) ]
  ret i32 %tmp0
}

define i32 @test_tailcall_ib_0(ptr %arg0) #0 {
; CHECK-LABEL: test_tailcall_ib_0:
; CHECK-NEXT:   brabz x0
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 0) ]
  ret i32 %tmp0
}

define i32 @test_call_ia_imm(ptr %arg0) #0 {
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

define i32 @test_call_ib_imm(ptr %arg0) #0 {
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

define i32 @test_tailcall_ia_imm(ptr %arg0) #0 {
; CHECK-LABEL: test_tailcall_ia_imm:
; CHECK-NEXT:    mov x16, #42
; CHECK-NEXT:    braa x0, x16
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 42) ]
  ret i32 %tmp0
}

define i32 @test_tailcall_ib_imm(ptr %arg0) #0 {
; CHECK-LABEL: test_tailcall_ib_imm:
; CHECK-NEXT:    mov x16, #42
; CHECK-NEXT:    brab x0, x16
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 42) ]
  ret i32 %tmp0
}

define i32 @test_call_ia_var(ptr %arg0, ptr %arg1) #0 {
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
  %tmp0 = load i64, ptr %arg1
  %tmp1 = call i32 %arg0() [ "ptrauth"(i32 0, i64 %tmp0) ]
  ret i32 %tmp1
}

define i32 @test_call_ib_var(ptr %arg0, ptr %arg1) #0 {
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
  %tmp0 = load i64, ptr %arg1
  %tmp1 = call i32 %arg0() [ "ptrauth"(i32 1, i64 %tmp0) ]
  ret i32 %tmp1
}

define i32 @test_tailcall_ia_var(ptr %arg0, ptr %arg1) #0 {
; CHECK-LABEL: test_tailcall_ia_var:
; CHECK:    ldr x1, [x1]
; CHECK:    braa x0, x1
  %tmp0 = load i64, ptr %arg1
  %tmp1 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 %tmp0) ]
  ret i32 %tmp1
}

define i32 @test_tailcall_ib_var(ptr %arg0, ptr %arg1) #0 {
; CHECK-LABEL: test_tailcall_ib_var:
; CHECK:    ldr x1, [x1]
; CHECK:    brab x0, x1
  %tmp0 = load i64, ptr %arg1
  %tmp1 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 %tmp0) ]
  ret i32 %tmp1
}

define void @test_tailcall_omit_mov_x16_x16(ptr %objptr) #0 {
; CHECK-LABEL: test_tailcall_omit_mov_x16_x16:
; DARWIN-NEXT:    ldr     x16, [x0]
; DARWIN-NEXT:    mov     x17, x0
; DARWIN-NEXT:    movk    x17, #6503, lsl #48
; DARWIN-NEXT:    autda   x16, x17
; DARWIN-NEXT:    ldr     x1, [x16]
; DARWIN-NEXT:    movk    x16, #54167, lsl #48
; DARWIN-NEXT:    braa    x1, x16
; ELF-NEXT:       ldr     x1, [x0]
; ELF-NEXT:       mov     x8, x0
; ELF-NEXT:       movk    x8, #6503, lsl #48
; ELF-NEXT:       autda   x1, x8
; ELF-NEXT:       ldr     x2, [x1]
; ELF-NEXT:       mov     x16, x1
; ELF-NEXT:       movk    x16, #54167, lsl #48
; ELF-NEXT:       braa    x2, x16
  %vtable.signed = load ptr, ptr %objptr, align 8
  %objptr.int = ptrtoint ptr %objptr to i64
  %vtable.discr = tail call i64 @llvm.ptrauth.blend(i64 %objptr.int, i64 6503)
  %vtable.signed.int = ptrtoint ptr %vtable.signed to i64
  %vtable.unsigned.int = tail call i64 @llvm.ptrauth.auth(i64 %vtable.signed.int, i32 2, i64 %vtable.discr)
  %vtable.unsigned = inttoptr i64 %vtable.unsigned.int to ptr
  %virt.func.signed = load ptr, ptr %vtable.unsigned, align 8
  %virt.func.discr = tail call i64 @llvm.ptrauth.blend(i64 %vtable.unsigned.int, i64 54167)
  tail call void %virt.func.signed(ptr %objptr) [ "ptrauth"(i32 0, i64 %virt.func.discr) ]
  ret void
}

define i32 @test_call_omit_extra_moves(ptr %objptr) #0 {
; CHECK-LABEL: test_call_omit_extra_moves:
; DARWIN-NEXT:   stp     x29, x30, [sp, #-16]!
; DARWIN-NEXT:   ldr     x16, [x0]
; DARWIN-NEXT:   mov     x17, x0
; DARWIN-NEXT:   movk    x17, #6503, lsl #48
; DARWIN-NEXT:   autda   x16, x17
; DARWIN-NEXT:   ldr     x8, [x16]
; DARWIN-NEXT:   movk    x16, #34646, lsl #48
; DARWIN-NEXT:   blraa   x8, x16
; DARWIN-NEXT:   mov     w0, #42
; DARWIN-NEXT:   ldp     x29, x30, [sp], #16
; ELF-NEXT:      str     x30, [sp, #-16]!
; ELF-NEXT:      ldr     x8, [x0]
; ELF-NEXT:      mov     x9, x0
; ELF-NEXT:      movk    x9, #6503, lsl #48
; ELF-NEXT:      autda   x8, x9
; ELF-NEXT:      ldr     x9, [x8]
; ELF-NEXT:      mov     x17, x8
; ELF-NEXT:      movk    x17, #34646, lsl #48
; ELF-NEXT:      blraa   x9, x17
; ELF-NEXT:      mov     w0, #42
; ELF-NEXT:      ldr     x30, [sp], #16
; CHECK-NEXT:    ret
  %vtable.signed = load ptr, ptr %objptr
  %objptr.int = ptrtoint ptr %objptr to i64
  %vtable.discr = tail call i64 @llvm.ptrauth.blend(i64 %objptr.int, i64 6503)
  %vtable.signed.int = ptrtoint ptr %vtable.signed to i64
  %vtable.int = tail call i64 @llvm.ptrauth.auth(i64 %vtable.signed.int, i32 2, i64 %vtable.discr)
  %vtable = inttoptr i64 %vtable.int to ptr
  %callee.signed = load ptr, ptr %vtable
  %callee.discr = tail call i64 @llvm.ptrauth.blend(i64 %vtable.int, i64 34646)
  %call.result = tail call i32 %callee.signed(ptr %objptr) [ "ptrauth"(i32 0, i64 %callee.discr) ]
  ret i32 42
}

; The second BLRA instruction should not reuse its AddrDisc operand as a scratch register (returned later).
define i64 @test_call_discr_csr_live(ptr %fnptr, i64 %addr.discr) #0 {
; ELF-LABEL: test_call_discr_csr_live:
; ELF-NEXT:    str     x30, [sp, #-32]!
; ELF-NEXT:    stp     x20, x19, [sp, #16]
; ELF-DAG:     mov     x[[FNPTR:[0-9]+]], x0
; ELF-DAG:     mov     x[[ADDR_DISC:[0-9]+]], x1
; ELF-DAG:     mov     x17, x1
; ELF-NEXT:    movk    x17, #6503, lsl #48
; ELF-NEXT:    blraa   x0, x17
; ELF-NEXT:    mov     x17, x[[ADDR_DISC]]
; ELF-NEXT:    movk    x17, #6503, lsl #48
; ELF-NEXT:    blraa   x[[FNPTR]], x17
; ELF-NEXT:    mov     x0, x[[ADDR_DISC]]
; ELF-NEXT:    ldp     x20, x19, [sp, #16]
; ELF-NEXT:    ldr     x30, [sp], #32
; ELF-NEXT:    ret
  %discr = tail call i64 @llvm.ptrauth.blend(i64 %addr.discr, i64 6503)
  tail call void %fnptr() [ "ptrauth"(i32 0, i64 %discr) ]
  tail call void %fnptr() [ "ptrauth"(i32 0, i64 %discr) ]
  ret i64 %addr.discr
}

; The second BLRA instruction may reuse its AddrDisc operand as a scratch register.
define i64 @test_call_discr_csr_killed(ptr %fnptr, i64 %addr.discr) #0 {
; ELF-LABEL: test_call_discr_csr_killed:
; ELF-NEXT:    str     x30, [sp, #-32]!
; ELF-NEXT:    stp     x20, x19, [sp, #16]
; ELF-DAG:     mov     x[[FNPTR:[0-9]+]], x0
; ELF-DAG:     mov     x[[ADDR_DISC:[0-9]+]], x1
; ELF-DAG:     mov     x17, x1
; ELF-NEXT:    movk    x17, #6503, lsl #48
; ELF-NEXT:    blraa   x0, x17
; ELF-DAG:     mov     x17, x[[ADDR_DISC]]
; ELF-NEXT:    movk    x17, #6503, lsl #48
; ELF-NEXT:    blraa   x[[FNPTR]], x17
; ELF-NEXT:    ldp     x20, x19, [sp, #16]
; ELF-NEXT:    mov     w0, #42
; ELF-NEXT:    ldr     x30, [sp], #32
; ELF-NEXT:    ret
  %discr = tail call i64 @llvm.ptrauth.blend(i64 %addr.discr, i64 6503)
  tail call void %fnptr() [ "ptrauth"(i32 0, i64 %discr) ]
  tail call void %fnptr() [ "ptrauth"(i32 0, i64 %discr) ]
  ret i64 42
}

; BLRA instruction should not reuse its AddrDisc operand as a scratch register (function argument).
define i64 @test_call_discr_arg(ptr %fnptr, i64 %addr.discr) #0 {
; ELF-LABEL: test_call_discr_arg:
; ELF-NEXT:    str     x30, [sp, #-16]!
; ELF-NEXT:    mov     x8, x0
; ELF-NEXT:    mov     x0, xzr
; ELF-NEXT:    mov     x17, x1
; ELF-NEXT:    movk    x17, #6503, lsl #48
; ELF-NEXT:    blraa   x8, x17
; ELF-NEXT:    mov     w0, #42
; ELF-NEXT:    ldr     x30, [sp], #16
; ELF-NEXT:    ret
  %discr = tail call i64 @llvm.ptrauth.blend(i64 %addr.discr, i64 6503)
  tail call void %fnptr(ptr null, i64 %addr.discr) [ "ptrauth"(i32 0, i64 %discr) ]
  ret i64 42
}

; BLRA instruction may reuse its AddrDisc operand as a scratch register.
define i64 @test_call_discr_non_arg(ptr %fnptr, i64 %addr.discr) #0 {
; ELF-LABEL: test_call_discr_non_arg:
; ELF-NEXT:    str     x30, [sp, #-16]!
; ELF-NEXT:    mov     x17, x1
; ELF-NEXT:    movk    x17, #6503, lsl #48
; ELF-NEXT:    blraa   x0, x17
; ELF-NEXT:    mov     w0, #42
; ELF-NEXT:    ldr     x30, [sp], #16
; ELF-NEXT:    ret
  %discr = tail call i64 @llvm.ptrauth.blend(i64 %addr.discr, i64 6503)
  tail call void %fnptr() [ "ptrauth"(i32 0, i64 %discr) ]
  ret i64 42
}

; AUTH_TCRETURN instruction should not reuse its AddrDisc operand as a scratch register (function argument).
define i64 @test_tailcall_discr_arg(ptr %fnptr, i64 %addr.discr) #0 {
; ELF-LABEL: test_tailcall_discr_arg:
; ELF-NEXT:    mov     x2, x0
; ELF-NEXT:    mov     x0, xzr
; ELF-NEXT:    mov     x16, x1
; ELF-NEXT:    movk    x16, #6503, lsl #48
; ELF-NEXT:    braa    x2, x16
  %discr = tail call i64 @llvm.ptrauth.blend(i64 %addr.discr, i64 6503)
  %result = tail call i64 %fnptr(ptr null, i64 %addr.discr) [ "ptrauth"(i32 0, i64 %discr) ]
  ret i64 %result
}

define i32 @test_call_ia_arg(ptr %arg0, i64 %arg1) #0 {
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

define i32 @test_call_ib_arg(ptr %arg0, i64 %arg1) #0 {
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

define i32 @test_tailcall_ia_arg(ptr %arg0, i64 %arg1) #0 {
; CHECK-LABEL: test_tailcall_ia_arg:
; CHECK:    braa x0, x1
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp0
}

define i32 @test_tailcall_ib_arg(ptr %arg0, i64 %arg1) #0 {
; CHECK-LABEL: test_tailcall_ib_arg:
; CHECK:    brab x0, x1
  %tmp0 = tail call i32 %arg0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp0
}

define i32 @test_call_ia_arg_ind(ptr %arg0, i64 %arg1) #0 {
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
  %tmp0 = load ptr, ptr %arg0
  %tmp1 = call i32 %tmp0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp1
}

define i32 @test_call_ib_arg_ind(ptr %arg0, i64 %arg1) #0 {
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
  %tmp0 = load ptr, ptr %arg0
  %tmp1 = call i32 %tmp0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp1
}

define i32 @test_tailcall_ia_arg_ind(ptr %arg0, i64 %arg1) #0 {
; CHECK-LABEL: test_tailcall_ia_arg_ind:
; CHECK:    ldr x0, [x0]
; CHECK:    braa x0, x1
  %tmp0 = load ptr, ptr %arg0
  %tmp1 = tail call i32 %tmp0() [ "ptrauth"(i32 0, i64 %arg1) ]
  ret i32 %tmp1
}

define i32 @test_tailcall_ib_arg_ind(ptr %arg0, i64 %arg1) #0 {
; CHECK-LABEL: test_tailcall_ib_arg_ind:
; CHECK:    ldr x0, [x0]
; CHECK:    brab x0, x1
  %tmp0 = load ptr, ptr %arg0
  %tmp1 = tail call i32 %tmp0() [ "ptrauth"(i32 1, i64 %arg1) ]
  ret i32 %tmp1
}

; Test direct calls

define i32 @test_direct_call() #0 {
; DARWIN-LABEL: test_direct_call:
; DARWIN-NEXT:   stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:   bl _f
; DARWIN-NEXT:   ldp x29, x30, [sp], #16
; DARWIN-NEXT:   ret
;
; ELF-LABEL: test_direct_call:
; ELF-NEXT:   str x30, [sp, #-16]!
; ELF-NEXT:   bl f
; ELF-NEXT:   ldr x30, [sp], #16
; ELF-NEXT:   ret
  %tmp0 = call i32 ptrauth(ptr @f, i32 0, i64 42)() [ "ptrauth"(i32 0, i64 42) ]
  ret i32 %tmp0
}

define i32 @test_direct_tailcall(ptr %arg0) #0 {
; DARWIN-LABEL: test_direct_tailcall:
; DARWIN:    b _f
;
; ELF-LABEL: test_direct_tailcall:
; ELF-NEXT:   b f
  %tmp0 = tail call i32 ptrauth(ptr @f, i32 0, i64 42)() [ "ptrauth"(i32 0, i64 42) ]
  ret i32 %tmp0
}

define i32 @test_direct_call_mismatch() #0 {
; DARWIN-LABEL: test_direct_call_mismatch:
; DARWIN-NEXT:   stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:   adrp x16, _f@GOTPAGE
; DARWIN-NEXT:   ldr x16, [x16, _f@GOTPAGEOFF]
; DARWIN-NEXT:   mov x17, #42
; DARWIN-NEXT:   pacia x16, x17
; DARWIN-NEXT:   mov x8, x16
; DARWIN-NEXT:   mov x17, #42
; DARWIN-NEXT:   blrab x8, x17
; DARWIN-NEXT:   ldp x29, x30, [sp], #16
; DARWIN-NEXT:   ret
;
; ELF-LABEL: test_direct_call_mismatch:
; ELF-NEXT:   str x30, [sp, #-16]!
; ELF-NEXT:   adrp x16, :got:f
; ELF-NEXT:   ldr x16, [x16, :got_lo12:f]
; ELF-NEXT:   mov x17, #42
; ELF-NEXT:   pacia x16, x17
; ELF-NEXT:   mov x8, x16
; ELF-NEXT:   mov x17, #42
; ELF-NEXT:   blrab x8, x17
; ELF-NEXT:   ldr x30, [sp], #16
; ELF-NEXT:   ret
  %tmp0 = call i32 ptrauth(ptr @f, i32 0, i64 42)() [ "ptrauth"(i32 1, i64 42) ]
  ret i32 %tmp0
}

define i32 @test_direct_call_addr() #0 {
; DARWIN-LABEL: test_direct_call_addr:
; DARWIN-NEXT:   stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:   bl _f
; DARWIN-NEXT:   ldp x29, x30, [sp], #16
; DARWIN-NEXT:   ret
;
; ELF-LABEL: test_direct_call_addr:
; ELF-NEXT:   str x30, [sp, #-16]!
; ELF-NEXT:   bl f
; ELF-NEXT:   ldr x30, [sp], #16
; ELF-NEXT:   ret
  %tmp0 = call i32 ptrauth(ptr @f, i32 1, i64 0, ptr @f.ref.ib.0.addr)() [ "ptrauth"(i32 1, i64 ptrtoint (ptr @f.ref.ib.0.addr to i64)) ]
  ret i32 %tmp0
}

define i32 @test_direct_call_addr_blend() #0 {
; DARWIN-LABEL: test_direct_call_addr_blend:
; DARWIN-NEXT:   stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:   bl _f
; DARWIN-NEXT:   ldp x29, x30, [sp], #16
; DARWIN-NEXT:   ret
;
; ELF-LABEL: test_direct_call_addr_blend:
; ELF-NEXT:   str x30, [sp, #-16]!
; ELF-NEXT:   bl f
; ELF-NEXT:   ldr x30, [sp], #16
; ELF-NEXT:   ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @f.ref.ib.42.addr to i64), i64 42)
  %tmp1 = call i32 ptrauth(ptr @f, i32 1, i64 42, ptr @f.ref.ib.42.addr)() [ "ptrauth"(i32 1, i64 %tmp0) ]
  ret i32 %tmp1
}

define i32 @test_direct_call_addr_gep_different_index_types() #0 {
; DARWIN-LABEL: test_direct_call_addr_gep_different_index_types:
; DARWIN-NEXT:   stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:   bl _f
; DARWIN-NEXT:   ldp x29, x30, [sp], #16
; DARWIN-NEXT:   ret
;
; ELF-LABEL: test_direct_call_addr_gep_different_index_types:
; ELF-NEXT:   str x30, [sp, #-16]!
; ELF-NEXT:   bl f
; ELF-NEXT:   ldr x30, [sp], #16
; ELF-NEXT:   ret
  %tmp0 = call i32 ptrauth(ptr @f, i32 1, i64 0, ptr getelementptr ({ ptr }, ptr @f_struct.ref.ib.0.addr, i64 0, i32 0))() [ "ptrauth"(i32 1, i64 ptrtoint (ptr getelementptr ({ ptr }, ptr @f_struct.ref.ib.0.addr, i32 0, i32 0) to i64)) ]
  ret i32 %tmp0
}

define i32 @test_direct_call_addr_blend_gep_different_index_types() #0 {
; DARWIN-LABEL: test_direct_call_addr_blend_gep_different_index_types:
; DARWIN-NEXT:   stp x29, x30, [sp, #-16]!
; DARWIN-NEXT:   bl _f
; DARWIN-NEXT:   ldp x29, x30, [sp], #16
; DARWIN-NEXT:   ret
;
; ELF-LABEL: test_direct_call_addr_blend_gep_different_index_types:
; ELF-NEXT:   str x30, [sp, #-16]!
; ELF-NEXT:   bl f
; ELF-NEXT:   ldr x30, [sp], #16
; ELF-NEXT:   ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr getelementptr ({ ptr }, ptr @f_struct.ref.ib.123.addr, i32 0, i32 0) to i64), i64 123)
  %tmp1 = call i32 ptrauth(ptr @f, i32 1, i64 123, ptr getelementptr ({ ptr }, ptr @f_struct.ref.ib.123.addr, i64 0, i32 0))() [ "ptrauth"(i32 1, i64 %tmp0) ]
  ret i32 %tmp1
}

@f.ref.ib.42.addr = external global ptr
@f.ref.ib.0.addr = external global ptr
@f_struct.ref.ib.0.addr = external global ptr
@f_struct.ref.ib.123.addr = external global ptr

declare void @f()

declare i64 @llvm.ptrauth.auth(i64, i32, i64)
declare i64 @llvm.ptrauth.blend(i64, i64)

attributes #0 = { nounwind }
