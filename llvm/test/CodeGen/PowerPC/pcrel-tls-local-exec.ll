; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -enable-ppc-pcrel-tls -mcpu=pwr10 -ppc-asm-full-reg-names \
; RUN:   < %s | FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -enable-ppc-pcrel-tls -mcpu=pwr10 -ppc-asm-full-reg-names \
; RUN:   --filetype=obj < %s | llvm-objdump --no-show-raw-insn --mcpu=pwr10 -dr - \
; RUN:   | FileCheck %s --check-prefix=CHECK-O

; These test cases are to ensure that when using pc relative memory operations
; ABI correct code and relocations are produced for the Local Exec TLS Model.

@x = thread_local global i32 0, align 4
@y = thread_local global [5 x i32] [i32 0, i32 0, i32 0, i32 0, i32 0], align 4

define i32* @LocalExecAddressLoad() {
; CHECK-S-LABEL: LocalExecAddressLoad:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    paddi r3, r13, x@TPREL, 0
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <LocalExecAddressLoad>:
; CHECK-O:         0: paddi 3, 13, 0, 0
; CHECK-O-NEXT:    0000000000000000:  R_PPC64_TPREL34 x
; CHECK-O-NEXT:    8: blr
entry:
  ret i32* @x
}

define i32 @LocalExecValueLoad() {
; CHECK-S-LABEL: LocalExecValueLoad:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    paddi r3, r13, x@TPREL, 0
; CHECK-S-NEXT:    lwz r3, 0(r3)
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <LocalExecValueLoad>:
; CHECK-O:         20: paddi 3, 13, 0, 0
; CHECK-O-NEXT:    0000000000000020:  R_PPC64_TPREL34 x
; CHECK-O-NEXT:    28: lwz 3, 0(3)
; CHECK-O-NEXT:    2c: blr
entry:
  %0 = load i32, i32* @x, align 4
  ret i32 %0
}

define i32 @LocalExecValueLoadOffset() {
; CHECK-S-LABEL: LocalExecValueLoadOffset:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    paddi r3, r13, y@TPREL, 0
; CHECK-S-NEXT:    lwz r3, 12(r3)
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <LocalExecValueLoadOffset>:
; CHECK-O:         40: paddi 3, 13, 0, 0
; CHECK-O-NEXT:    0000000000000040:  R_PPC64_TPREL34 y
; CHECK-O-NEXT:    48: lwz 3, 12(3)
; CHECK-O-NEXT:    4c: blr
entry:
  %0 = load i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @y, i64 0, i64 3), align 4
  ret i32 %0
}


define i32* @LocalExecValueLoadOffsetNoLoad() {
; CHECK-S-LABEL: LocalExecValueLoadOffsetNoLoad:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    paddi r3, r13, y@TPREL, 0
; CHECK-S-NEXT:    addi r3, r3, 12
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <LocalExecValueLoadOffsetNoLoad>:
; CHECK-O:         60: paddi 3, 13, 0, 0
; CHECK-O-NEXT:    0000000000000060:  R_PPC64_TPREL34 y
; CHECK-O-NEXT:    68: addi 3, 3, 12
; CHECK-O-NEXT:    6c: blr
entry:
  ret i32* getelementptr inbounds ([5 x i32], [5 x i32]* @y, i64 0, i64 3)
}
