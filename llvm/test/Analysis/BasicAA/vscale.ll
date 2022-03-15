; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; getelementptr

; CHECK-LABEL: gep_alloca_const_offset_1
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep2
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_alloca_const_offset_1() {
  %alloc = alloca <vscale x 4 x i32>
  %gep1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc, i64 0
  %gep2 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc, i64 1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep2
  ret void
}

; CHECK-LABEL: gep_alloca_const_offset_2
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep2
; TODO: AliasResult for gep1,gep2 can be improved as MustAlias
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_alloca_const_offset_2() {
  %alloc = alloca <vscale x 4 x i32>
  %gep1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc, i64 1
  %gep2 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc, i64 1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep2
  ret void
}

; CHECK-LABEL: gep_alloca_const_offset_3
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, i32* %gep2
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, i32* %gep2
define void @gep_alloca_const_offset_3() {
  %alloc = alloca <vscale x 4 x i32>
  %gep1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc, i64 0
  %gep2 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc, i64 0, i64 1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK-LABEL: gep_alloca_const_offset_4
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep1
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %alloc, i32* %gep2
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %gep1, i32* %gep2
define void @gep_alloca_const_offset_4() {
  %alloc = alloca <vscale x 4 x i32>
  %gep1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc, i64 0
  %gep2 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc, i64 0, i64 0
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK-LABEL: gep_alloca_symbolic_offset
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep2
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_alloca_symbolic_offset(i64 %idx1, i64 %idx2) {
  %alloc = alloca <vscale x 4 x i32>
  %gep1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc, i64 %idx1
  %gep2 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc, i64 %idx2
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %alloc
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep2
  ret void
}

; CHECK-LABEL: gep_same_base_const_offset
; CHECK-DAG:  MayAlias:     i32* %gep1, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep2, <vscale x 4 x i32>* %p
; TODO: AliasResult for gep1,gep2 can be improved as NoAlias
; CHECK-DAG:  MayAlias:     i32* %gep1, i32* %gep2
define void @gep_same_base_const_offset(<vscale x 4 x i32>* %p) {
  %gep1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p, i64 1, i64 0
  %gep2 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p, i64 1, i64 1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %p
  load i32, i32* %gep1
  load i32, i32* %gep2
  ret void
}

; CHECK-LABEL: gep_same_base_symbolic_offset
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep2, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_same_base_symbolic_offset(<vscale x 4 x i32>* %p, i64 %idx1, i64 %idx2) {
  %gep1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p, i64 %idx1
  %gep2 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p, i64 %idx2
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %p
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep2
  ret void
}

; CHECK-LABEL: gep_different_base_const_offset
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %p1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep2, <vscale x 4 x i32>* %p2
; CHECK-DAG:  NoAlias:      <vscale x 4 x i32>* %p1, <vscale x 4 x i32>* %p2
; CHECK-DAG:  NoAlias:      <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %p2
; CHECK-DAG:  NoAlias:      <vscale x 4 x i32>* %gep2, <vscale x 4 x i32>* %p1
; CHECK-DAG:  NoAlias:      <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_different_base_const_offset(<vscale x 4 x i32>* noalias %p1, <vscale x 4 x i32>* noalias %p2) {
  %gep1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p1, i64 1
  %gep2 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p2, i64 1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %p1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %p2
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %gep2
  ret void
}

; getelementptr + bitcast

; CHECK-LABEL: gep_bitcast_1
; CHECK-DAG:   MustAlias:    <vscale x 4 x i32>* %p, i32* %p2
; CHECK-DAG:   MayAlias:     i32* %gep1, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     i32* %gep1, i32* %p2
; CHECK-DAG:   MayAlias:     i32* %gep2, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     i32* %gep1, i32* %gep2
; CHECK-DAG:   NoAlias:      i32* %gep2, i32* %p2
define void @gep_bitcast_1(<vscale x 4 x i32>* %p) {
  %gep1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p, i64 1, i64 0
  %p2 = bitcast <vscale x 4 x i32>* %p to i32*
  %gep2 = getelementptr i32, i32* %p2, i64 4
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %p
  load i32, i32* %gep1
  load i32, i32* %gep2
  load i32, i32* %p2
  ret void
}

; CHECK-LABEL: gep_bitcast_2
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %p, <vscale x 4 x float>* %p2
; CHECK-DAG:  MayAlias:     i32* %gep1, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep1, <vscale x 4 x float>* %p2
; CHECK-DAG:  MayAlias:     float* %gep2, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep1, float* %gep2
; CHECK-DAG:  MayAlias:     float* %gep2, <vscale x 4 x float>* %p2
define void @gep_bitcast_2(<vscale x 4 x i32>* %p) {
  %gep1 = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p, i64 1, i64 0
  %p2 = bitcast <vscale x 4 x i32>* %p to <vscale x 4 x float>*
  %gep2 = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %p2, i64 1, i64 0
  load i32, i32* %gep1
  load float, float* %gep2
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %p
  load <vscale x 4 x float>, <vscale x 4 x float>* %p2
  ret void
}

; getelementptr recursion

; CHECK-LABEL: gep_recursion_level_1
; CHECK-DAG:  MayAlias:     i32* %a, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %a, i32* %gep
; CHECK-DAG:  MayAlias:     i32* %a, i32* %gep_rec_1
; CHECK-DAG:  MayAlias:     i32* %gep, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep_rec_1, <vscale x 4 x i32>* %p
; CHECK-DAG:  NoAlias:      i32* %gep, i32* %gep_rec_1
define void @gep_recursion_level_1(i32* %a, <vscale x 4 x i32>* %p) {
  %gep = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p, i64 1, i64 2
  %gep_rec_1 = getelementptr i32, i32* %gep, i64 1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %p
  load i32, i32* %a
  load i32, i32* %gep
  load i32, i32* %gep_rec_1
  ret void
}

; CHECK-LABEL: gep_recursion_level_1_bitcast
; CHECK-DAG:  MustAlias:    i32* %a, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %a, i32* %gep
; CHECK-DAG:  MayAlias:     i32* %a, i32* %gep_rec_1
; CHECK-DAG:  MayAlias:     i32* %gep, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep_rec_1, <vscale x 4 x i32>* %p
; CHECK-DAG:  NoAlias:      i32* %gep, i32* %gep_rec_1
define void @gep_recursion_level_1_bitcast(i32* %a) {
  %p = bitcast i32* %a to <vscale x 4 x i32>*
  %gep = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p, i64 1, i64 2
  %gep_rec_1 = getelementptr i32, i32* %gep, i64 1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %p
  load i32, i32* %a
  load i32, i32* %gep
  load i32, i32* %gep_rec_1
  ret void
}

; CHECK-LABEL: gep_recursion_level_2
; CHECK-DAG:  MayAlias:     i32* %a, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %a, i32* %gep
; CHECK-DAG:  MayAlias:     i32* %a, i32* %gep_rec_1
; CHECK-DAG:  MayAlias:     i32* %a, i32* %gep_rec_2
; CHECK-DAG:  MayAlias:     i32* %gep, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep_rec_1, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep_rec_2, <vscale x 4 x i32>* %p
; CHECK-DAG:  NoAlias:      i32* %gep, i32* %gep_rec_1
; CHECK-DAG:  NoAlias:      i32* %gep, i32* %gep_rec_2
; CHECK-DAG:  NoAlias:      i32* %gep_rec_1, i32* %gep_rec_2
define void @gep_recursion_level_2(i32* %a, <vscale x 4 x i32>* %p) {
  %gep = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p, i64 1, i64 2
  %gep_rec_1 = getelementptr i32, i32* %gep, i64 1
  %gep_rec_2 = getelementptr i32, i32* %gep_rec_1, i64 1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %p
  load i32, i32* %a
  load i32, i32* %gep
  load i32, i32* %gep_rec_1
  load i32, i32* %gep_rec_2
  ret void
}

; CHECK-LABEL: gep_recursion_max_lookup_depth_reached
; CHECK-DAG: MayAlias:     i32* %a, <vscale x 4 x i32>* %p
; CHECK-DAG: MayAlias:     i32* %a, i32* %gep
; CHECK-DAG: MayAlias:     i32* %a, i32* %gep_rec_1
; CHECK-DAG: MayAlias:     i32* %a, i32* %gep_rec_2
; CHECK-DAG: MayAlias:     i32* %a, i32* %gep_rec_3
; CHECK-DAG: MayAlias:     i32* %a, i32* %gep_rec_4
; CHECK-DAG: MayAlias:     i32* %a, i32* %gep_rec_5
; CHECK-DAG: MayAlias:     i32* %a, i32* %gep_rec_6
; CHECK-DAG: MayAlias:     i32* %gep, <vscale x 4 x i32>* %p
; CHECK-DAG: MayAlias:     i32* %gep_rec_1, <vscale x 4 x i32>* %p
; CHECK-DAG: MayAlias:     i32* %gep_rec_2, <vscale x 4 x i32>* %p
; CHECK-DAG: MayAlias:     i32* %gep_rec_3, <vscale x 4 x i32>* %p
; CHECK-DAG: MayAlias:     i32* %gep_rec_4, <vscale x 4 x i32>* %p
; CHECK-DAG: MayAlias:     i32* %gep_rec_5, <vscale x 4 x i32>* %p
; CHECK-DAG: MayAlias:     i32* %gep_rec_6, <vscale x 4 x i32>* %p
; CHECK-DAG: NoAlias:      i32* %gep, i32* %gep_rec_1
; CHECK-DAG: NoAlias:      i32* %gep, i32* %gep_rec_2
; CHECK-DAG: NoAlias:      i32* %gep, i32* %gep_rec_3
; CHECK-DAG: NoAlias:      i32* %gep, i32* %gep_rec_4
; CHECK-DAG: NoAlias:      i32* %gep, i32* %gep_rec_5
; CHECK-DAG: NoAlias:      i32* %gep, i32* %gep_rec_6
; CHECK-DAG: NoAlias:      i32* %gep_rec_1, i32* %gep_rec_2
; CHECK-DAG: NoAlias:      i32* %gep_rec_1, i32* %gep_rec_3
; CHECK-DAG: NoAlias:      i32* %gep_rec_1, i32* %gep_rec_4
; CHECK-DAG: NoAlias:      i32* %gep_rec_1, i32* %gep_rec_5
; CHECK-DAG: NoAlias:      i32* %gep_rec_1, i32* %gep_rec_6
; CHECK-DAG: NoAlias:      i32* %gep_rec_2, i32* %gep_rec_3
; CHECK-DAG: NoAlias:      i32* %gep_rec_2, i32* %gep_rec_4
; CHECK-DAG: NoAlias:      i32* %gep_rec_2, i32* %gep_rec_5
; CHECK-DAG: NoAlias:      i32* %gep_rec_2, i32* %gep_rec_6
; CHECK-DAG: NoAlias:      i32* %gep_rec_3, i32* %gep_rec_4
; CHECK-DAG: NoAlias:      i32* %gep_rec_3, i32* %gep_rec_5
; CHECK-DAG: NoAlias:      i32* %gep_rec_3, i32* %gep_rec_6
; CHECK-DAG: NoAlias:      i32* %gep_rec_4, i32* %gep_rec_5
; CHECK-DAG: NoAlias:      i32* %gep_rec_4, i32* %gep_rec_6
; CHECK-DAG: NoAlias:      i32* %gep_rec_5, i32* %gep_rec_6
; GEP max lookup depth was set to 6.
define void @gep_recursion_max_lookup_depth_reached(i32* %a, <vscale x 4 x i32>* %p) {
  %gep = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %p, i64 1, i64 2
  %gep_rec_1 = getelementptr i32, i32* %gep, i64 1
  %gep_rec_2 = getelementptr i32, i32* %gep_rec_1, i64 1
  %gep_rec_3 = getelementptr i32, i32* %gep_rec_2, i64 1
  %gep_rec_4 = getelementptr i32, i32* %gep_rec_3, i64 1
  %gep_rec_5 = getelementptr i32, i32* %gep_rec_4, i64 1
  %gep_rec_6 = getelementptr i32, i32* %gep_rec_5, i64 1
  load <vscale x 4 x i32>, <vscale x 4 x i32>* %p
  load i32, i32* %a
  load i32, i32* %gep
  load i32, i32* %gep_rec_1
  load i32, i32* %gep_rec_2
  load i32, i32* %gep_rec_3
  load i32, i32* %gep_rec_4
  load i32, i32* %gep_rec_5
  load i32, i32* %gep_rec_6
  ret void
}
