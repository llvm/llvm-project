; RUN: llc -mtriple=mips -debug-only=isel -mips-tail-calls=1 \
; RUN:   -relocation-model=pic < %s 2>&1 | FileCheck --check-prefix=PIC %s
; RUN: llc -mtriple=mips -debug-only=isel -mips-tail-calls=1  \
; RUN:   -relocation-model=static < %s 2>&1 | FileCheck --check-prefix=STATIC %s
; RUN: llc -mtriple=mips64 -debug-only=isel -mips-tail-calls=1 \
; RUN:   -relocation-model=pic < %s 2>&1 | FileCheck --check-prefix=PIC64 %s
; RUN: llc -mtriple=mips64 -debug-only=isel -mips-tail-calls=1  \
; RUN:   -relocation-model=static < %s 2>&1 | FileCheck --check-prefix=STATIC64 %s
; RUN: llc -mtriple=mips -debug-only=isel -mips-tail-calls=1 \
; RUN:   -relocation-model=pic -mattr=+micromips < %s 2>&1 | FileCheck --check-prefix=PIC %s
; RUN: llc -mtriple=mips -debug-only=isel -mips-tail-calls=1  \
; RUN:   -relocation-model=static -mattr=+micromips < %s 2>&1 | FileCheck --check-prefix=STATIC-MM %s
; RUN: llc -mtriple=mips -mcpu=mips32r6 -debug-only=isel -mips-tail-calls=1 \
; RUN:   -relocation-model=pic -mattr=+micromips < %s 2>&1 | FileCheck --check-prefix=PIC %s
; RUN: llc -mtriple=mips -mcpu=mips32r6 -debug-only=isel -mips-tail-calls=1  \
; RUN:   -relocation-model=static -mattr=+micromips < %s 2>&1 | FileCheck --check-prefix=STATIC-MM %s
; RUN: llc -mtriple=mips -debug-only=isel -mips-tail-calls=1  \
; RUN:   -relocation-model=pic    -mattr=+mips16 < %s 2>&1 | FileCheck --check-prefix=MIPS16 %s
; RUN: llc -mtriple=mips -debug-only=isel -mips-tail-calls=1  \
; RUN:   -relocation-model=static -mattr=+mips16 < %s 2>&1 | FileCheck --check-prefix=MIPS16 %s

; REQUIRES: asserts

; Test that the correct pseudo instructions are generated for indirect
; branches and tail calls. Previously, the order of the DAG matcher table
; determined if the correct instruction was selected for mips16.

declare protected void @a()

define void @test1(i32 %a) {
entry:
  %0 = trunc i32 %a to i1
  %1 = select i1 %0,
              ptr blockaddress(@test1, %bb),
              ptr blockaddress(@test1, %bb6)
  indirectbr ptr %1, [label %bb, label %bb6]

; STATIC:     PseudoIndirectBranch
; STATIC-MM:  PseudoIndirectBranch
; STATIC-NOT: PseudoIndirectBranch64
; STATIC64:   PseudoIndirectBranch64
; PIC:        PseudoIndirectBranch
; PIC-NOT:    PseudoIndirectBranch64
; PIC64:      PseudoIndirectBranch64
; MIPS16:     JrcRx16
bb:
  ret void

bb6:
  tail call void @a()

; STATIC:     TAILCALL
; STATIC-NOT: TAILCALL_MM
; STATIC-MM:  TAILCALL_MM
; PIC:        TAILCALLREG
; PIC-NOT:    TAILCALLREG64
; PIC64:      TAILCALLREG64
; MIPS16:     RetRA16
  ret void
}
