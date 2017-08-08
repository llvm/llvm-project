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

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32 -relocation-model=static %s -o - | FileCheck -check-prefix=CHECK-LE32 %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32 %s -o - | FileCheck -check-prefix=CHECK-LE32_PIC %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -relocation-model=static %s -o - | FileCheck -check-prefix=CHECK-LE32R2 %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 %s -o - | FileCheck -check-prefix=CHECK-LE32R2_PIC %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+micromips -relocation-model=static %s -o - | FileCheck -check-prefix=CHECK-LE32R2-MICROMIPS %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+micromips %s -o - | FileCheck -check-prefix=CHECK-LE32R2-MICROMIPS_PIC %s

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips4 -target-abi n64 -relocation-model=static %s -o - | FileCheck -check-prefix=CHECK-LE64 %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips4 -target-abi n64 -relocation-model=pic %s -o - | FileCheck -check-prefix=CHECK-LE64_PIC %s

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64 -target-abi n64 -relocation-model=static %s -o - | FileCheck -check-prefix=CHECK-LE64 %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64 -target-abi n64 -relocation-model=pic %s -o - | FileCheck -check-prefix=CHECK-LE64_PIC %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64r2 -target-abi n64 -relocation-model=static %s -o - | FileCheck -check-prefix=CHECK-LE64R2 %s
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips64r2 -target-abi n64 -relocation-model=pic %s -o - | FileCheck -check-prefix=CHECK-LE64R2_PIC %s

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+mips16 -relocation-model=pic %s -o - | FileCheck -check-prefix=CHECK-LE32R2-MIPS16 %s

; 32(R1) bit with NO_REORDER and static
; CHECK-LE32: .abicalls
; CHECK-LE32: .option	pic0
; CHECK-LE32: .set	noreorder
;
; 32(R1) bit with NO_REORDER and PIC
; CHECK-LE32_PIC: .abicalls
; CHECK-LE32_PIC: .set	noreorder
;
; 32R2 bit with NO_REORDER and static
; CHECK-LE32R2: .abicalls
; CHECK-LE32R2: .option pic0
; CHECK-LE32R2: .set noreorder
;
; 32R2 bit with NO_REORDER and PIC
; CHECK-LE32R2_PIC: .abicalls
; CHECK-LE32R2_PIC: .set noreorder
;
; 32R2 bit MICROMIPS with NO_REORDER and static
; CHECK-LE32R2-MICROMIPS: .abicalls
; CHECK-LE32R2-MICROMIPS: .option pic0
; CHECK-LE32R2-MICROMIPS: .set	micromips
;
; 32R2 bit MICROMIPS with NO_REORDER and PIC
; CHECK-LE32R2-MICROMIPS_PIC: .abicalls
; CHECK-LE32R2-MICROMIPS_PIC: .set micromips
;
; 64(R1) bit with NO_REORDER and static
; CHECK-LE64: .set noreorder
;
; 64(R1) bit with NO_REORDER and PIC
; CHECK-LE64_PIC: .abicalls
; CHECK-LE64_PIC: .set noreorder
;
; 64R2 bit with NO_REORDER and static
; CHECK-LE64R2: .set noreorder
;
; 64R2 bit with NO_REORDER and PIC
; CHECK-LE64R2_PIC: .abicalls
; CHECK-LE64R2_PIC: .set noreorder
;
; 32R2 bit MIPS16 with PIC
; CHECK-LE32R2-MIPS16: .abicalls
; CHECK-LE32R2-MIPS16: .set mips16

define i32 @main() nounwind {
entry:
  ret i32 0
}
