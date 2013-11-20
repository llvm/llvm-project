; This tests ELF EFLAGS setting with direct object.
; When the assembler is ready a .s file for it will
; be created.

; Non-shared (static) is the absence of pic and or cpic.

; EF_MIPS_NOREORDER (0x00000001) is always on by default currently
; EF_MIPS_PIC (0x00000002)
; EF_MIPS_CPIC (0x00000004) - See note below
; EF_MIPS_ABI2 (0x00000020) - n32 not tested yet
; EF_MIPS_ARCH_32 (0x50000000)
; EF_MIPS_ARCH_64 (0x60000000)
; EF_MIPS_ARCH_32R2 (0x70000000)
; EF_MIPS_ARCH_64R2 (0x80000000)

; Note that EF_MIPS_CPIC is set by -mabicalls which is the default on Linux
; TODO need to support -mno-abicalls

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32 -relocation-model=static %s -print-hack-directives -o - | FileCheck -check-prefix=CHECK-BE32 %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32 -print-hack-directives %s -o - | FileCheck -check-prefix=CHECK-BE32_PIC %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -relocation-model=static %s -print-hack-directives -o - | FileCheck -check-prefix=CHECK-BE32R2 %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -print-hack-directives %s -o - | FileCheck -check-prefix=CHECK-BE32R2_PIC %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+micromips -relocation-model=static -print-hack-directives %s -o - | FileCheck -check-prefix=CHECK-BE32R2-MICROMIPS %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+micromips -print-hack-directives %s -o - | FileCheck -check-prefix=CHECK-BE32R2-MICROMIPS_PIC %s

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64 -relocation-model=static %s -print-hack-directives -o - | FileCheck -check-prefix=CHECK-BE64 %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64 %s -print-hack-directives -o - | FileCheck -check-prefix=CHECK-BE64_PIC %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64r2 -relocation-model=static -print-hack-directives %s -o - | FileCheck -check-prefix=CHECK-BE64R2 %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64r2 -print-hack-directives %s -o - | FileCheck -check-prefix=CHECK-BE64R2_PIC %s

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+mips16 -relocation-model=pic -print-hack-directives %s -o - | FileCheck -check-prefix=CHECK-LE32R2-MIPS16 %s
 
; 32(R1) bit with NO_REORDER and static
; CHECK-BE32: .mips_hack_elf_flags 0x50001005
;
; 32(R1) bit with NO_REORDER and PIC
; CHECK-BE32_PIC: .mips_hack_elf_flags 0x50001007
;
; 32R2 bit with NO_REORDER and static
; CHECK-BE32R2: .mips_hack_elf_flags 0x70001005
;
; 32R2 bit with NO_REORDER and PIC
; CHECK-BE32R2_PIC: .mips_hack_elf_flags 0x70001007
;
; 32R2 bit MICROMIPS with NO_REORDER and static
; CHECK-BE32R2-MICROMIPS: .mips_hack_elf_flags 0x72001005
;
; 32R2 bit MICROMIPS with NO_REORDER and PIC
; CHECK-BE32R2-MICROMIPS_PIC: .mips_hack_elf_flags 0x72001007
;
; 64(R1) bit with NO_REORDER and static
; CHECK-BE64: .mips_hack_elf_flags 0x60000005
;
; 64(R1) bit with NO_REORDER and PIC
; CHECK-BE64_PIC: .mips_hack_elf_flags 0x60000007
;
; 64R2 bit with NO_REORDER and static
; CHECK-BE64R2: .mips_hack_elf_flags 0x80000005
;
; 64R2 bit with NO_REORDER and PIC
; CHECK-BE64R2_PIC: .mips_hack_elf_flags 0x80000007
;
; 32R2 bit MIPS16 with PIC
; CHECK-LE32R2-MIPS16: .mips_hack_elf_flags 0x74001006

define i32 @main() nounwind {
entry:
  ret i32 0
}
