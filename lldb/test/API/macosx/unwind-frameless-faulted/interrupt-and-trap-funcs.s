#define DW_CFA_register 0x9
#define ehframe_x22  22
#define ehframe_x23  23
#define ehframe_pc 32

  .section  __TEXT,__text,regular,pure_instructions

//--------------------------------------
// to_be_interrupted() a frameless function that does a non-ABI
// function call ("is interrupted/traps" simulated) to trap().
// Before it branches to trap(), it puts its return address in
// x23.  trap() knows to branch back to $x23 when it has finished.
//--------------------------------------
  .globl  _to_be_interrupted
  .p2align  2
_to_be_interrupted:
  .cfi_startproc

  // This is a garbage entry to ensure that eh_frame is emitted,
  // it isn't used for anything.  If there's no eh_frame, lldb
  // can do an assembly emulation scan and add a rule for $lr
  // which won't expose the issue at hand.
  .cfi_escape DW_CFA_register, ehframe_x22, ehframe_x23
  mov x24, x0
  add x24, x24, #1

  adrp x23, L_.return@PAGE        ; put return address in x4
  add x23, x23, L_.return@PAGEOFF

  b _trap                     ; branch to trap handler, fake async interrupt

L_.return:
  mov x0, x24
  ret
  .cfi_endproc
  


//--------------------------------------
// trap() trap handler function, sets up stack frame
// with special unwind rule for the pc value of the
// "interrupted" stack frame (it's in x23), then calls
// break_to_debugger().
//--------------------------------------
  .globl  _trap
  .p2align  2
_trap:                                  
  .cfi_startproc
  .cfi_signal_frame

  // The pc value when we were interrupted is in x23
  .cfi_escape DW_CFA_register, ehframe_pc, ehframe_x23

  // standard prologue save of fp & lr so we can call 
  // break_to_debugger()
  sub sp, sp, #32
  stp x29, x30, [sp, #16]
  add x29, sp, #16
  .cfi_def_cfa w29, 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16

  bl _break_to_debugger

  ldp x29, x30, [sp, #16]
  add sp, sp, #32

  // jump back to $x23 to resume execution of to_be_interrupted
  br x23
  .cfi_endproc

//--------------------------------------
// break_to_debugger() executes a BRK instruction
//--------------------------------------
  .globl _break_to_debugger
  .p2align  2
_break_to_debugger:                                  
  .cfi_startproc

  brk #0xf000   ;; __builtin_debugtrap aarch64 instruction

  ret
  .cfi_endproc
