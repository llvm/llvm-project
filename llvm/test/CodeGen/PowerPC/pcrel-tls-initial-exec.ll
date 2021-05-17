; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names --filetype=obj -o %t.o < %s
; RUN: llvm-objdump --mcpu=pwr10 -dr %t.o | FileCheck %s --check-prefix=CHECK-O
; RUN: llvm-readelf -s %t.o | FileCheck %s --check-prefix=CHECK-SYM

; These test cases are to ensure that when using pc relative memory operations
; ABI correct code and relocations are produced for Initial Exec TLS Model.
; Note that with R_PPC64_TLS relocations, to distinguish PC relative
; TLS the relocation has a field value displaced by one byte from the
; beginning of the instruction.

@x = external thread_local global i32, align 4

define i32* @InitialExecAddressLoad() {
; CHECK-S-LABEL: InitialExecAddressLoad:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    pld r3, x@got@tprel@pcrel(0), 1
; CHECK-S-NEXT:    add r3, r3, x@tls@pcrel
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <InitialExecAddressLoad>:
; CHECK-O:         00 00 10 04 00 00 60 e4      	pld 3, 0(0), 1
; CHECK-O-NEXT:    0000000000000000:  R_PPC64_GOT_TPREL_PCREL34	x
; CHECK-O-NEXT:    14 6a 63 7c                  	add 3, 3, 13
; CHECK-O-NEXT:    0000000000000009:  R_PPC64_TLS	x
; CHECK-O-NEXT:    20 00 80 4e                  	blr
entry:
  ret i32* @x
}

define i32 @InitialExecValueLoad() {
; CHECK-S-LABEL: InitialExecValueLoad:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    pld r3, x@got@tprel@pcrel(0), 1
; CHECK-S-NEXT:    lwzx r3, r3, x@tls@pcrel
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <InitialExecValueLoad>:
; CHECK-O:         00 00 10 04 00 00 60 e4      	pld 3, 0(0), 1
; CHECK-O-NEXT:    0000000000000020:  R_PPC64_GOT_TPREL_PCREL34	x
; CHECK-O-NEXT:    2e 68 63 7c                  	lwzx 3, 3, 13
; CHECK-O-NEXT:    0000000000000029:  R_PPC64_TLS	x
; CHECK-O-NEXT:    20 00 80 4e                  	blr

; CHECK-SYM-LABEL: Symbol table '.symtab' contains 6 entries
; CHECK-SYM:         0000000000000000     0 TLS     GLOBAL DEFAULT  UND x
entry:
  %0 = load i32, i32* @x, align 4
  ret i32 %0
}
