; RUN: llc -march=bpfel -filetype=obj -o - %s \
; RUN:     | llvm-objdump --no-print-imm-hex --arch=bpfel --section=foo -d - \
; RUN:     | FileCheck %s

; This test was added because "isPseudo" flag was missing in FI_ri
; instruction definition and certain byte sequence caused an assertion
; in llvm-objdump tool.

; The value "G" is byte by byte little endian representation of the FI_ri instruction,
; as declared in the BPFInstrInfo.td.
; The first byte encodes an opcode: BPF_IMM(0x00) | BPF_DW(0x18) | BPF_LD(0x00)
; The second byte encodes source and destination registers: 2 and 0 respectively.
; The rest of the bytes are zeroes to comply with the specification.
; An additional 8 bytes follow the instruction as an immediate 64 bit argument,
; (because of the BPF_IMM flag).

; This is a pseudo instruction, meaning that it's not possible to
; write it in assembly directly. Thus it is coded as a byte array.

; Note the "bpfel" flags in the RUN command.

@G = constant [16 x i8]
              [i8 u0x18, i8 u0x20, i8 u0x00, i8 u0x00, i8 u0x00, i8 u0x00, i8 u0x00, i8 u0x00,
               i8 u0x00, i8 u0x00, i8 u0x00, i8 u0x00, i8 u0x00, i8 u0x00, i8 u0x00, i8 u0x00],
              section "foo", align 8

; CHECK-LABEL: G
; CHECK:       0: 18 20 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ld_pseudo r0, 2, 0
