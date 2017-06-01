; RUN: llc -march=avr -print-after=expand-isel-pseudos -cgp-freq-ratio-to-skip-merge=10 < %s 2>&1 | FileCheck %s

; Because `switch` seems to trigger Machine Basic Blocks to be ordered
; in a different order than they were constructed, this exposes an
; error in the `expand-isel-pseudos` pass. Specifically, it thought we
; could always fallthrough to a newly-constructed MBB. However,
; there's no guarantee that either of the constructed MBBs need to
; occur immediately after the currently-focused one!
;
; This issue manifests in a CFG that looks something like this:
;
; BB#2: derived from LLVM BB %finish
;     Predecessors according to CFG: BB#0 BB#1
;         %vreg0<def> = PHI %vreg3, <BB#0>, %vreg5, <BB#1>
;         %vreg7<def> = LDIRdK 2
;         %vreg8<def> = LDIRdK 1
;         CPRdRr %vreg2, %vreg0, %SREG<imp-def>
;         BREQk <BB#6>, %SREG<imp-use>
;     Successors according to CFG: BB#5(?%) BB#6(?%)
;
; The code assumes it the fallthrough block after this is BB#5, but
; it's actually BB#3! To be proper, there should be an unconditional
; jump tying this block to BB#5.

define i8 @select_must_add_unconditional_jump(i8 %arg0, i8 %arg1) unnamed_addr {
entry-block:
  switch i8 %arg0, label %dead [
    i8 0, label %zero
    i8 1, label %one
  ]

zero:
  br label %finish

one:
  br label %finish

finish:
  %predicate = phi i8 [ 50, %zero ], [ 100, %one ]
  %is_eq = icmp eq i8 %arg1, %predicate
  %result = select i1 %is_eq, i8 1, i8 2
  ret i8 %result

dead:
  ret i8 0
}

; This check may be a bit brittle, but the important thing is that the
; basic block containing `select` needs to contain explicit jumps to
; both successors.

; CHECK: BB#2: derived from LLVM BB %finish
; CHECK: BREQk <[[BRANCHED:BB#[0-9]+]]>
; CHECK: RJMPk <[[DIRECT:BB#[0-9]+]]>
; CHECK: Successors according to CFG
; CHECK-SAME-DAG: {{.*}}[[BRANCHED]]
; CHECK-SAME-DAG: {{.*}}[[DIRECT]]
; CHECK: BB#3: derived from LLVM BB
