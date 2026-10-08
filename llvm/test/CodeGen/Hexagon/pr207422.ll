; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 \
; RUN:     -mattr=+hvxv68,+hvx-length128b,+hvx-qfloat,+memops,-long-calls \
; RUN:     -hexagon-autohvx -machine-sink-split=0 -verify-machineinstrs -O3 \
; RUN:     < %s -o /dev/null

; This testcase previously tripped the machine verifier
; after the If Converter. HexagonRDFOpt's Liveness recomputation left
; stale (over-approximate) live-in lists on some blocks, and If Converter
; then inserted incorrect implicit-use operands on predicated loads.

define fastcc void @_ZL19checkVectorFunctionIxxEvNSt3__18functionIFT_PT0_S4_jEEES6_PKc(ptr %Src1, i32 %index.next2179, i32 %index2113, <16 x i32> %vec.ind2114, <16 x i1> %0, i1 %1, ptr %2, ptr %3, i1 %4, ptr %5, i64 %6, ptr %7, i64 %8, ptr %9, i64 %10, ptr %11, i64 %12, ptr %13, ptr %14) {
entry:
  br label %vector.body2112

vector.body2112:                                  ; preds = %pred.store.continue2122, %entry
  %15 = icmp eq <16 x i32> %vec.ind2114, zeroinitializer
  %16 = select <16 x i1> %0, <16 x i64> splat (i64 -9223372036854775808), <16 x i64> zeroinitializer
  br i1 %1, label %pred.store.if2115, label %pred.store.continue2116

pred.store.if2115:                                ; preds = %vector.body2112
  store i64 0, ptr %3, align 8
  br label %pred.store.continue2116

pred.store.continue2116:                          ; preds = %pred.store.if2115, %vector.body2112
  br i1 %4, label %pred.store.if2117, label %pred.store.continue2122

pred.store.if2117:                                ; preds = %pred.store.continue2116
  store i64 0, ptr %2, align 8
  br label %pred.store.continue2122

pred.store.continue2122:                          ; preds = %pred.store.if2117, %pred.store.continue2116
  %17 = extractelement <16 x i64> %16, i64 8
  store i64 %17, ptr %5, align 8
  store i64 %6, ptr %7, align 8
  %18 = extractelement <16 x i64> %16, i64 10
  store i64 %18, ptr %Src1, align 8
  store i64 %8, ptr %9, align 8
  store i64 %10, ptr %Src1, align 8
  %19 = select <16 x i1> %15, <16 x i64> splat (i64 9223372036854775807), <16 x i64> zeroinitializer
  %20 = extractelement <16 x i64> %19, i64 0
  store i64 %20, ptr %11, align 8
  store i64 %12, ptr %13, align 8
  store i64 0, ptr %14, align 8
  br label %vector.body2112
}
