; Test that IR `select` on HLX vector types is lowered to LX_mux_h.
; The lowering chain is:
; For a single vector:
; select i1 %cond, <1024 x i16>, <1024 x i16>
;   -> VSELECT SPLAT_VECTOR(%cond), <1024 x i16>, <1024 x i16>
;   -> LX_mux_h  (isel pattern match)
;
; For a vector pair:
; select i1 %cond, <1024 x i32>, <1024 x i32>
;   -> two VSELECT node (lo and hi)
;   -> two LX_mux_h + COMBINE to rebuild the pair
;
; The condition scalar is broadcast into an HLX register via:
;   C2_cmpgti / C2_not -> C2_muxii -> LX_assign_r
; The VSELECT is then emited which ulitmately maps to mux operation.

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -start-before=hexagon-isel -stop-after=hexagon-isel -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=CHECK-ISEL

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048-v16384:16384:16384-v32768:16384:16384"
target triple = "hexagon"

@vtcm_base = dso_local local_unnamed_addr global ptr null, align 4

; ---------------------------------------------------------------------------
; select_single: select on a single HLX_Vector (v1024i16)
;
; IR:  %cond1 = select i1 (cond < 8), <1024 x i16> %a, <1024 x i16> %b
;
; Expected isel output:
;   %p:predregs  = C2_cmpgti %cond_val, 7      ; cond > 7
;   %np:predregs = C2_not %p                   ; invert for mux polarity
;   %sc:intregs  = C2_muxii %np, -1, 0         ; broadcast pred to scalar mask
;   %mk:hlxlr    = LX_assign_r %sc             ; scalar mask -> HLX register
;   %res:hlxlr   = LX_mux_h %b, %a, %mk        ; select: if mask==0xff..ff pick %a else %b
; ---------------------------------------------------------------------------

; CHECK-ISEL-LABEL: name: select_single
; CHECK-ISEL: [[TRUE0:%[0-9]+]]:hlxlr = LX_assign_s 1
; CHECK-ISEL: [[FALSE0:%[0-9]+]]:hlxlr = LX_assign_s 2
; CHECK-ISEL: [[PRED:%[0-9]+]]:predregs = C2_cmpgti {{.*}}, 7
; CHECK-ISEL: [[NPRED:%[0-9]+]]:predregs = C2_not [[PRED]]
; CHECK-ISEL: [[COND:%[0-9]+]]:intregs = C2_muxii [[NPRED]], -1, 0
; CHECK-ISEL: [[MASK:%[0-9]+]]:hlxlr = LX_assign_r [[COND]]
; CHECK-ISEL: [[RES:%[0-9]+]]:hlxlr = LX_mux_h [[FALSE0]], [[TRUE0]], [[MASK]]

define dso_local inreg <1024 x i16> @select_single(i32 %cond) local_unnamed_addr #0 {
entry:
    %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.s(i32 1)
    %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.s(i32 2)
    %cmp = icmp slt i32 %cond, 8
    %cond1 = select i1 %cmp, <1024 x i16> %0, <1024 x i16> %1
    ret <1024 x i16> %cond1
}

; ---------------------------------------------------------------------------
; select_pair: select on an HLX_VectorPair (v1024i32 = hi:lo pair of v1024i16)
;
; IR:  %cond10 = select i1 (cond < 8), <1024 x i32> %p0, <1024 x i32> %p1
;
; Expected isel output:
;   The pair select is split into two independent half-selects:
;     LX_mux_h for the lo half  (vsub_lo of each pair)
;     LX_mux_h for the hi half  (vsub_hi of each pair)
;   Both use the same scalar mask register (LX_assign_r of C2_muxii).
;   The two result halves are reassembled with COMBINE.
; ---------------------------------------------------------------------------

; CHECK-ISEL-LABEL: name: select_pair
; CHECK-ISEL: [[TRUEHI:%[0-9]+]]:hlxlr  = LX_assign_s 3
; CHECK-ISEL: [[TRUELO:%[0-9]+]]:hlxlr  = LX_assign_s 4
; CHECK-ISEL: [[PAIR0:%[0-9]+]]:hlxllr = LX_combine_w [[TRUEHI]], [[TRUELO]]
; CHECK-ISEL: [[FALSEHI:%[0-9]+]]:hlxlr  = LX_assign_s 5
; CHECK-ISEL: [[FALSELO:%[0-9]+]]:hlxlr  = LX_assign_s 6
; CHECK-ISEL: [[PAIR1:%[0-9]+]]:hlxllr = LX_combine_w [[FALSEHI]], [[FALSELO]]
; CHECK-ISEL: [[PRED2:%[0-9]+]]:predregs = C2_cmpgti {{.*}}, 7
; CHECK-ISEL: [[NPRED2:%[0-9]+]]:predregs = C2_not [[PRED2]]
; CHECK-ISEL-DAG: [[P0LO:%[0-9]+]]:hlxlr = COPY [[PAIR0]].vsub_lo
; CHECK-ISEL-DAG: [[P1LO:%[0-9]+]]:hlxlr = COPY [[PAIR1]].vsub_lo
; CHECK-ISEL: [[SC2:%[0-9]+]]:intregs = C2_muxii [[NPRED2]], -1, 0
; CHECK-ISEL: [[MASK2:%[0-9]+]]:hlxlr = LX_assign_r [[SC2]]
; CHECK-ISEL: [[RESLO:%[0-9]+]]:hlxlr = LX_mux_h [[P1LO]], [[P0LO]], [[MASK2]]
; CHECK-ISEL-DAG: [[P0HI:%[0-9]+]]:hlxlr = COPY [[PAIR0]].vsub_hi
; CHECK-ISEL-DAG: [[P1HI:%[0-9]+]]:hlxlr = COPY [[PAIR1]].vsub_hi
; CHECK-ISEL: [[RESHI:%[0-9]+]]:hlxlr = LX_mux_h [[P1HI]], [[P0HI]], [[MASK2]]
; CHECK-ISEL: {{%[0-9]+}}:hlxllr = REG_SEQUENCE [[RESHI]], %subreg.vsub_hi, [[RESLO]], %subreg.vsub_lo

define dso_local inreg <1024 x i32> @select_pair(i32 %cond) local_unnamed_addr #0 {
entry:
    %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.s(i32 3)
    %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.s(i32 4)
    %2 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %0, <1024 x i16> %1)
    %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.s(i32 5)
    %4 = tail call <1024 x i16> @llvm.hexagon.LX.assign.s(i32 6)
    %5 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %3, <1024 x i16> %4)
    %cmp = icmp slt i32 %cond, 8
    %cond1 = select i1 %cmp, <1024 x i32> %2, <1024 x i32> %5
    ret <1024 x i32> %cond1
}

declare <1024 x i16> @llvm.hexagon.LX.assign.s(i32 immarg) #1
declare <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16>, <1024 x i16>) #1

attributes #0 = { nofree nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv85" "target-features"="+hlx,+hlx-length2048b,+hvx-length128b,+hvxv85,+v85,-long-calls" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"frame-pointer", i32 2}
