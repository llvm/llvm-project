; RUN: llc -mtriple=riscv64-unknown-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,EXTRA-FEATURES
; RUN: llc -mtriple=riscv64-unknown-linux-gnu -mattr=+d < %s | FileCheck %s --check-prefixes=CHECK,SAME-FEATURES
; RUN: llc -mtriple=riscv64-unknown-linux-gnu -filetype=obj < %s | llvm-objdump -d --show-all-symbols --no-show-raw-insn - | FileCheck %s --check-prefix=OBJ

; This should work fine, because the module asm specifies the necessary
; target features

; SAME-FEATURES-NOT: .option arch
; EXTRA-FEATURES:      .option push
; EXTRA-FEATURES-NEXT: .option arch, +d, +f, +zicsr{{$}}
; CHECK:        .globl func
; CHECK-NEXT: func:
; CHECK-NEXT:   fld ft0, 0(sp)
; CHECK-NEXT:   ret
; EXTRA-FEATURES-NEXT: .option pop

; OBJ-LABEL: Disassembly of section .text:
; OBJ-EMPTY:
; OBJ-NEXT:  0000000000000000 <$xrv64i2p1_f2p2_d2p2_zicsr2p0>:
; OBJ-NEXT:  0000000000000000 <func>:
; OBJ-NEXT:         0:      	fld	ft0, 0x0(sp)
; OBJ-NEXT:         4:      	ret
; OBJ-NOT:   {{.}}

module asm(target_features: "+d")
    ".globl func"
    "func:"
    "fld f0, 0(sp)"
    "ret"
