; RUN: llc -verify-machineinstrs -mtriple=armv7-linux-gnu -O0 %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=thumbv8-linux-gnu -O0 %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=thumbv6m-none-eabi -O0 %s -o - | FileCheck %s --check-prefix=CHECK-T1

; CHECK-T1-NOT: ldrex
; CHECK-T1-NOT: strex

define { i8, i1 } @test_cmpxchg_8(ptr %addr, i8 %desired, i8 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_8:
; CHECK-DAG: mov [[ADDR:r[0-9]+]], r0
; CHECK-DAG: mov [[NEW:r[0-9]+]], r2
; CHECK:     dmb ish
; CHECK:     uxtb [[DESIRED:r[0-9]+]], [[DESIRED]]
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldrexb [[OLD:[lr0-9]+]], [[[ADDR]]]
; CHECK:     cmp [[OLD]], [[DESIRED]]
; CHECK:     bne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     strexb [[STATUS:r[0-9]+]], [[NEW]], [[[ADDR]]]
; CHECK:     cmp{{(\.w)?}} [[STATUS]], #0
; CHECK:     bne [[RETRY]]
; CHECK: [[DONE]]:
; Materialisation of a boolean is done with sub/clz/lsr
; CHECK:     uxtb [[CMP1:r[0-9]+]], [[DESIRED]]
; CHECK:     sub{{(\.w|s)?}} [[CMP1]], [[OLD]], [[CMP1]]
; CHECK:     clz [[CMP2:r[0-9]+]], [[CMP1]]
; CHECK:     lsr{{(s)?}} {{r[0-9]+}}, [[CMP2]], #5
; CHECK:     dmb ish
  %res = cmpxchg ptr %addr, i8 %desired, i8 %new seq_cst monotonic
  ret { i8, i1 } %res
}

define { i16, i1 } @test_cmpxchg_16(ptr %addr, i16 %desired, i16 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_16:
; CHECK-DAG: mov [[ADDR:r[0-9]+]], r0
; CHECK-DAG: mov [[NEW:r[0-9]+]], r2
; CHECK:     dmb ish
; CHECK:     uxth [[DESIRED:r[0-9]+]], [[DESIRED]]
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldrexh [[OLD:[lr0-9]+]], [[[ADDR]]]
; CHECK:     cmp [[OLD]], [[DESIRED]]
; CHECK:     bne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     strexh [[STATUS:r[0-9]+]], [[NEW]], [[[ADDR]]]
; CHECK:     cmp{{(\.w)?}} [[STATUS]], #0
; CHECK:     bne [[RETRY]]
; CHECK: [[DONE]]:
; Materialisation of a boolean is done with sub/clz/lsr
; CHECK:     uxth [[CMP1:r[0-9]+]], [[DESIRED]]
; CHECK:     sub{{(\.w|s)?}} [[CMP1]], [[OLD]], [[CMP1]]
; CHECK:     clz [[CMP2:r[0-9]+]], [[CMP1]]
; CHECK:     lsr{{(s)?}} {{r[0-9]+}}, [[CMP2]], #5
; CHECK:     dmb ish
  %res = cmpxchg ptr %addr, i16 %desired, i16 %new seq_cst monotonic
  ret { i16, i1 } %res
}

define { i32, i1 } @test_cmpxchg_32(ptr %addr, i32 %desired, i32 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_32:
; CHECK-DAG: mov [[ADDR:r[0-9]+]], r0
; CHECK-DAG: mov [[NEW:r[0-9]+]], r2
; CHECK:     dmb ish
; CHECK-NOT:     uxt
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldrex [[OLD:r[0-9]+]], [[[ADDR]]]
; CHECK:     cmp [[OLD]], [[DESIRED]]
; CHECK:     bne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     strex [[STATUS:r[0-9]+]], [[NEW]], [[[ADDR]]]
; CHECK:     cmp{{(\.w)?}} [[STATUS]], #0
; CHECK:     bne [[RETRY]]
; CHECK: [[DONE]]:
; Materialisation of a boolean is done with sub/clz/lsr
; CHECK:     sub{{(s)?}} [[CMP1:r[0-9]+]], [[OLD]], [[DESIRED]]
; CHECK:     clz [[CMP2:r[0-9]+]], [[CMP1]]
; CHECK:     lsr{{(s)?}} {{r[0-9]+}}, [[CMP2]], #5
; CHECK:     dmb ish
  %res = cmpxchg ptr %addr, i32 %desired, i32 %new seq_cst monotonic
  ret { i32, i1 } %res
}

define { i64, i1 } @test_cmpxchg_64(ptr %addr, i64 %desired, i64 %new) nounwind {
; CHECK-LABEL: test_cmpxchg_64:
; CHECK:     dmb ish
; CHECK-NOT: uxt
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldrexd [[OLDLO:r[0-9]+]], [[OLDHI:r[0-9]+]], [r0]
; CHECK:     cmp [[OLDLO]], r6
; CHECK:     cmpeq [[OLDHI]], r7
; CHECK:     bne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     strexd [[STATUS:[lr0-9]+]], r8, r9, [r0]
; CHECK:     cmp{{(\.w)?}} [[STATUS]], #0
; CHECK:     bne [[RETRY]]
; CHECK: [[DONE]]:
; CHECK:     dmb ish
  %res = cmpxchg ptr %addr, i64 %desired, i64 %new seq_cst monotonic
  ret { i64, i1 } %res
}

define { i64, i1 } @test_nontrivial_args(ptr %addr, i64 %desired, i64 %new) {
; CHECK-LABEL: test_nontrivial_args:
; CHECK:     mov [[ADDR:r[0-9]+]], r0
; CHECK:     dmb ish
; CHECK-NOT: uxt
; CHECK: [[RETRY:.LBB[0-9]+_[0-9]+]]:
; CHECK:     ldrexd [[OLDLO:r[0-9]+]], [[OLDHI:r[0-9]+]], [[[ADDR]]]
; CHECK:     cmp [[OLDLO]], {{r[0-9]+}}
; CHECK:     cmpeq [[OLDHI]], {{r[0-9]+}}
; CHECK:     bne [[DONE:.LBB[0-9]+_[0-9]+]]
; CHECK:     strexd [[STATUS:r[0-9]+]], {{r[0-9]+}}, {{r[0-9]+}}, [[[ADDR]]]
; CHECK:     cmp{{(\.w)?}} [[STATUS]], #0
; CHECK:     bne [[RETRY]]
; CHECK: [[DONE]]:
; CHECK:     dmb ish

  %desired1 = add i64 %desired, 1
  %new1 = add i64 %new, 1
  %res = cmpxchg ptr %addr, i64 %desired1, i64 %new1 seq_cst seq_cst
  ret { i64, i1 } %res
}

; The following used to trigger an assertion when creating a spill on thumb2
; for a physreg with RC==GPRPairRegClass.
; CHECK-LABEL: test_cmpxchg_spillbug:
; CHECK: ldrexd
; CHECK: strexd
; CHECK: bne
define void @test_cmpxchg_spillbug() {
  %v = cmpxchg ptr undef, i64 undef, i64 undef seq_cst seq_cst
  ret void
}
