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

;; TODO: emitTargetFeaturePush does not call setArchString(), so the mapping
;; symbol does not record +d/+f/+zicsr when assembling directly to an object
;; file, causing llvm-objdump to fail to disassemble `fld`.
; OBJ-LABEL: Disassembly of section .text:
; OBJ-EMPTY:
; OBJ-NEXT:  0000000000000000 <$xrv64i2p1>:
; OBJ-NEXT:  0000000000000000 <func>:
; OBJ-NEXT:         0:      	<unknown>
; OBJ-NEXT:         4:      	ret
; OBJ-NOT:   {{.}}

module asm(target_features: "+d")
    ".globl func"
    "func:"
    "fld f0, 0(sp)"
    "ret"
