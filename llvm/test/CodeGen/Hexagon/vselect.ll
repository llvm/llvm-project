; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -mattr=+hvx-length128b < %s | FileCheck %s

; Add patterns to lower vselect q, q, q.

; CHECK-LABEL: vselect_v32i1
; CHECK-DAG: q[[Q1:[0-9]+]] = and(q{{[0-9]+}},!q{{[0-9]+}})
; CHECK-DAG: q[[Q2:[0-9]+]] = and(q{{[0-9]+}},q{{[0-9]+}})
; CHECK: q{{[0-9]+}} = or(q[[Q2]],q[[Q1]])
define <32 x i1> @vselect_v32i1(<32 x i1> %f, <32 x i1> %cond, <32 x i1> %t) #0 {
entry:
  %sel = select <32 x i1> %cond, <32 x i1> %t, <32 x i1> %f
  ret <32 x i1> %sel
}

; CHECK-LABEL: vselect_v64i1
; CHECK-DAG: q[[Q3:[0-9]+]] = and(q{{[0-9]+}},!q{{[0-9]+}})
; CHECK-DAG: q[[Q4:[0-9]+]] = and(q{{[0-9]+}},q{{[0-9]+}})
; CHECK: q{{[0-9]+}} = or(q[[Q4]],q[[Q3]])
define <64 x i1> @vselect_v64i1(<64 x i1> %f, <64 x i1> %cond, <64 x i1> %t) #0 {
entry:
  %sel = select <64 x i1> %cond, <64 x i1> %t, <64 x i1> %f
  ret <64 x i1> %sel
}

attributes #0 = { "target-cpu"="hexagonv68" "target-features"="+hvx-length128b" }
