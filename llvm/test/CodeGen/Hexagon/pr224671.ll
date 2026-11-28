; RUN: llc -mtriple=hexagon-unknown-linux-musl -mcpu=hexagonv68 \
; RUN:     -mattr=+memops,+hvxv68,+hvx-length128b,+hvx-qfloat \
; RUN:     -relocation-model=pic -tail-folding-policy=must-fold-tail \
; RUN:     -force-tail-folding-style=data-without-lane-mask \
; RUN:     -verify-machineinstrs < %s -o /dev/null

; This testcase previously tripped the machine verifier with "Using an
; undefined physical register" after the If Converter. HexagonOptAddrMode's
; RDF-based Liveness recomputation left stale (over-approximate) live-in lists
; on some blocks, and If Converter then inserted an incorrect implicit-use
; (old-value) operand on a predicated load whose register was undefined on the
; loop back-edge.

target triple = "hexagon-unknown-linux-musl"

define fastcc { ptr, i32 } @_ZL19checkVectorFunctionIyxEvNSt3__18functionIFT_PT0_S4_jEEES6_PKc(ptr %Patterns, <16 x i1> %0, i1 %1, ptr %2, i64 %3, i1 %4, ptr %5, i64 %6, ptr %7, i64 %8, ptr %9, i64 %10, ptr %11, ptr %12, ptr %13, i64 %14) personality ptr null {
entry:
  %arrayinit.element = getelementptr i8, ptr %Patterns, i32 32
  %arrayinit.element105 = getelementptr i8, ptr %Patterns, i32 64
  invoke fastcc void @_ZZL19checkVectorFunctionIyxEvNSt3__18functionIFT_PT0_S4_jEEES6_PKcENKUlS8_E_clES8_(ptr null, ptr null)
          to label %for.cond62.preheader unwind label %lpad6

common.ret:                                       ; preds = %for.cond.cleanup78, %pred.store.if1663, %for.cond62.preheader, %lpad6
  %common.ret.op = phi { ptr, i32 } [ zeroinitializer, %for.cond.cleanup78 ], [ zeroinitializer, %lpad6 ], [ zeroinitializer, %pred.store.if1663 ], [ zeroinitializer, %for.cond62.preheader ]
  ret { ptr, i32 } %common.ret.op

lpad6:                                            ; preds = %entry
  %15 = landingpad { ptr, i32 }
          cleanup
  br label %common.ret

for.cond62.preheader:                             ; preds = %entry
  br i1 %1, label %common.ret, label %pred.store.continue1632

pred.store.continue1632:                          ; preds = %pred.store.if1665, %for.cond62.preheader
  br i1 %4, label %pred.store.if1633, label %pred.store.continue1634

pred.store.if1633:                                ; preds = %pred.store.continue1632
  %16 = load ptr, ptr null, align 4
  store i64 0, ptr %16, align 8
  br label %pred.store.continue1634

pred.store.continue1634:                          ; preds = %pred.store.if1633, %pred.store.continue1632
  store i64 %14, ptr %2, align 8
  store i64 0, ptr %5, align 8
  store i64 %6, ptr %7, align 8
  store i64 %3, ptr %11, align 8
  store i64 %8, ptr %Patterns, align 8
  %17 = select <16 x i1> %0, <16 x i64> splat (i64 1), <16 x i64> zeroinitializer
  br i1 %4, label %pred.store.if1663, label %pred.store.if1665

pred.store.if1663:                                ; preds = %pred.store.continue1634
  store i64 %10, ptr %Patterns, align 8
  br label %common.ret

pred.store.if1665:                                ; preds = %pred.store.continue1634
  %18 = extractelement <16 x i64> %17, i64 0
  store i64 %18, ptr %12, align 8
  store i64 1, ptr %13, align 8
  br i1 %1, label %for.cond.cleanup78, label %pred.store.continue1632

for.cond.cleanup78:                               ; preds = %pred.store.if1665
  store ptr %arrayinit.element, ptr %9, align 8
  store ptr %arrayinit.element105, ptr %Patterns, align 8
  br label %common.ret
}

declare fastcc void @_ZZL19checkVectorFunctionIyxEvNSt3__18functionIFT_PT0_S4_jEEES6_PKcENKUlS8_E_clES8_(ptr, ptr)
