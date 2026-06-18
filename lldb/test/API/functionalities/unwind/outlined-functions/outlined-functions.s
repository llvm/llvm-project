  .text
// ---------------------------
// A function with its prologue outlined
// ---------------------------

  .globl _function_prologue_outlined
  .p2align 2
_function_prologue_outlined:
  .cfi_startproc

  ;; Do a non-ABI call where I put the
  ;; return address in x0 and then jump
  ;; to an outlined function.
  adrp x0, Lfunc_return0@PAGE
  add x0, x0, Lfunc_return0@PAGEOFF
  b _OUTLINED_FUNCTION_1
Lfunc_return0:
  bl _foo_prologue

  nop  ;; working around a bug where debugserver used to
       ;; migrate past a builtin_debugtrap brk, it would
       ;; do it here and now show us hitting it as we
       ;; private step out of foo_prologue.  Insert a
       ;; nop to avoid that.

  //brk	#0xf000  ;; __builtin_debugtrap "middle" of function

  ldp	x29, x30, [sp, #16]
  add	sp, sp, #32
  ret
  .cfi_endproc

  .globl _OUTLINED_FUNCTION_1
_OUTLINED_FUNCTION_1:
  .cfi_startproc
  sub	sp, sp, #32
  stp	x29, x30, [sp, #16]
  br x0
  .cfi_endproc

  .globl _foo_prologue
_foo_prologue:
  ret


// ---------------------------
// A function with part of its body outlined
// ---------------------------


  .globl _function_body_outlined
  .p2align 2
_function_body_outlined:
  .cfi_startproc
  sub	sp, sp, #32
  stp	x29, x30, [sp, #16]
  .cfi_def_cfa_offset 32
  .cfi_offset w30, -8
  .cfi_offset w29, -16

  bl _OUTLINED_FUNCTION_2

  //brk	#0xf000  ;; __builtin_debugtrap "middle" of function

  ldp	x29, x30, [sp, #16]
  add	sp, sp, #32
  ret
  .cfi_endproc

  ;; This OUTLINED_FUNCTION creates a normal
  ;; stack frame, but doesn't include unwind
  ;; instructions to that effect in the DWARF.
  ;;
  ;; We intend for the debug instructions to be wrong,
  ;; but an instruction analysis unwind plan would do the
  ;; right thing here.
  .globl _OUTLINED_FUNCTION_2
_OUTLINED_FUNCTION_2:
  .cfi_startproc
  sub	sp, sp, #32
  stp	x29, x30, [sp, #16]
  bl _foo_midfunction
  ldp	x29, x30, [sp, #16]
  add	sp, sp, #32
  ret
  .cfi_endproc

  .globl _foo_midfunction
_foo_midfunction:
  ret

// ---------------------------
// A function with its epilogue outlined
// ---------------------------

  .globl _function_epilogue_outlined
  .p2align 2
_function_epilogue_outlined:
  .cfi_startproc
  sub	sp, sp, #32
  stp	x29, x30, [sp, #16]
  .cfi_def_cfa_offset 32
  .cfi_offset w30, -8
  .cfi_offset w29, -16

  //brk	#0xf000  ;; __builtin_debugtrap "middle" of function
  b _OUTLINED_FUNCTION_3
  .cfi_endproc

  .globl _OUTLINED_FUNCTION_3
_OUTLINED_FUNCTION_3:
  .cfi_startproc
  bl _foo_epilogue
  ldp	x29, x30, [sp, #16]
  add	sp, sp, #32
  ret
  .cfi_endproc

  .globl _foo_epilogue
_foo_epilogue:
  ret
