// RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj %s -o - | llvm-readobj --sections - | FileCheck %s

// We were trying to generate compact unwind info for assembly like this.
// The .cfi_def_cfa directive, however, throws a wrench into that and was
// causing an llvm_unreachable() failure. Make sure the assembler can handle
// the input. The actual eh_frames created using these directives are checked
// elsewhere. This test is a simpler "does the code assemble" check.

// rdar://15406518

.macro SaveRegisters

 push %rbp
 .cfi_def_cfa_offset 16
 .cfi_offset rbp, -16

 mov %rsp, %rbp
 .cfi_def_cfa_register rbp

 sub $$0x80+8, %rsp

 movdqa %xmm0, -0x80(%rbp)
 push %rax
 movdqa %xmm1, -0x70(%rbp)
 push %rdi
 movdqa %xmm2, -0x60(%rbp)
 push %rsi
 movdqa %xmm3, -0x50(%rbp)
 push %rdx
 movdqa %xmm4, -0x40(%rbp)
 push %rcx
 movdqa %xmm5, -0x30(%rbp)
 push %r8
 movdqa %xmm6, -0x20(%rbp)
 push %r9
 movdqa %xmm7, -0x10(%rbp)

.endmacro
.macro RestoreRegisters

 movdqa -0x80(%rbp), %xmm0
 pop %r9
 movdqa -0x70(%rbp), %xmm1
 pop %r8
 movdqa -0x60(%rbp), %xmm2
 pop %rcx
 movdqa -0x50(%rbp), %xmm3
 pop %rdx
 movdqa -0x40(%rbp), %xmm4
 pop %rsi
 movdqa -0x30(%rbp), %xmm5
 pop %rdi
 movdqa -0x20(%rbp), %xmm6
 pop %rax
 movdqa -0x10(%rbp), %xmm7

 leave
 .cfi_def_cfa rsp, 8
 .cfi_same_value rbp

.endmacro

_foo:
.cfi_startproc
  SaveRegisters

  RestoreRegisters
  ret
 .cfi_endproc

// CHECK: Section {
// CHECK:   Index: 1
// CHECK:   Name: __eh_frame (5F 5F 65 68 5F 66 72 61 6D 65 00 00 00 00 00 00)
// CHECK:   Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK:   Address: 0x70
// CHECK:   Size: 0x40
// CHECK:   Offset: 480
// CHECK:   Alignment: 3
// CHECK:   RelocationOffset: 0x0
// CHECK:   RelocationCount: 0
// CHECK:   Type: Coalesced (0xB)
// CHECK:   Attributes [ (0x680000)
// CHECK:     LiveSupport (0x80000)
// CHECK:     NoTOC (0x400000)
// CHECK:     StripStaticSyms (0x200000)
// CHECK:   ]
// CHECK:   Reserved1: 0x0
// CHECK:   Reserved2: 0x0
// CHECK: }
