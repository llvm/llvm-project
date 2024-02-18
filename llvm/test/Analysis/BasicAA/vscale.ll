; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; getelementptr

; CHECK-LABEL: gep_alloca_const_offset_1
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep2
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_alloca_const_offset_1() {
  %alloc = alloca <vscale x 4 x i32>
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %alloc, i64 0
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %alloc, i64 1
  load <vscale x 4 x i32>, ptr %alloc
  load <vscale x 4 x i32>, ptr %gep1
  load <vscale x 4 x i32>, ptr %gep2
  ret void
}

; CHECK-LABEL: gep_alloca_const_offset_2
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep2
; TODO: AliasResult for gep1,gep2 can be improved as MustAlias
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_alloca_const_offset_2() {
  %alloc = alloca <vscale x 4 x i32>
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %alloc, i64 1
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %alloc, i64 1
  load <vscale x 4 x i32>, ptr %alloc
  load <vscale x 4 x i32>, ptr %gep1
  load <vscale x 4 x i32>, ptr %gep2
  ret void
}

; CHECK-LABEL: gep_alloca_const_offset_3
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, i32* %gep2
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, i32* %gep2
define void @gep_alloca_const_offset_3() {
  %alloc = alloca <vscale x 4 x i32>
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %alloc, i64 0
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %alloc, i64 0, i64 1
  load <vscale x 4 x i32>, ptr %alloc
  load <vscale x 4 x i32>, ptr %gep1
  load i32, ptr %gep2
  ret void
}

; CHECK-LABEL: gep_alloca_const_offset_4
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep1
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %alloc, i32* %gep2
; CHECK-DAG:  MustAlias:    <vscale x 4 x i32>* %gep1, i32* %gep2
define void @gep_alloca_const_offset_4() {
  %alloc = alloca <vscale x 4 x i32>
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %alloc, i64 0
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %alloc, i64 0, i64 0
  load <vscale x 4 x i32>, ptr %alloc
  load <vscale x 4 x i32>, ptr %gep1
  load i32, ptr %gep2
  ret void
}

; CHECK-LABEL: gep_alloca_symbolic_offset
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %alloc, <vscale x 4 x i32>* %gep2
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_alloca_symbolic_offset(i64 %idx1, i64 %idx2) {
  %alloc = alloca <vscale x 4 x i32>
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %alloc, i64 %idx1
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %alloc, i64 %idx2
  load <vscale x 4 x i32>, ptr %alloc
  load <vscale x 4 x i32>, ptr %gep1
  load <vscale x 4 x i32>, ptr %gep2
  ret void
}

; CHECK-LABEL: gep_same_base_const_offset
; CHECK-DAG:  MayAlias:     i32* %gep1, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep2, <vscale x 4 x i32>* %p
; TODO: AliasResult for gep1,gep2 can be improved as NoAlias
; CHECK-DAG:  MayAlias:     i32* %gep1, i32* %gep2
define void @gep_same_base_const_offset(ptr %p) {
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 0
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 1
  load <vscale x 4 x i32>, ptr %p
  load i32, ptr %gep1
  load i32, ptr %gep2
  ret void
}

; CHECK-LABEL: gep_same_base_symbolic_offset
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep2, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_same_base_symbolic_offset(ptr %p, i64 %idx1, i64 %idx2) {
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 %idx1
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %p, i64 %idx2
  load <vscale x 4 x i32>, ptr %p
  load <vscale x 4 x i32>, ptr %gep1
  load <vscale x 4 x i32>, ptr %gep2
  ret void
}

; CHECK-LABEL: gep_different_base_const_offset
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %p1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %gep2, <vscale x 4 x i32>* %p2
; CHECK-DAG:  NoAlias:      <vscale x 4 x i32>* %p1, <vscale x 4 x i32>* %p2
; CHECK-DAG:  NoAlias:      <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %p2
; CHECK-DAG:  NoAlias:      <vscale x 4 x i32>* %gep2, <vscale x 4 x i32>* %p1
; CHECK-DAG:  NoAlias:      <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_different_base_const_offset(ptr noalias %p1, ptr noalias %p2) {
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p1, i64 1
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %p2, i64 1
  load <vscale x 4 x i32>, ptr %p1
  load <vscale x 4 x i32>, ptr %p2
  load <vscale x 4 x i32>, ptr %gep1
  load <vscale x 4 x i32>, ptr %gep2
  ret void
}

; getelementptr @llvm.vscale tests
; CHECK-LABEL: gep_llvm_vscale_no_alias
; CHECK-DAG: MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
; CHECK-DAG: MayAlias:     <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep3
; CHECK-DAG: MayAlias:     <vscale x 4 x i32>* %gep2, <vscale x 4 x i32>* %gep3
define void @gep_llvm_vscale_no_alias(ptr %p) {
  %t1 = tail call i64 @llvm.vscale.i64()
  %t2 = shl nuw nsw i64 %t1, 3
  %gep1 = getelementptr i32, ptr %p, i64 %t2
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1
  %gep3 = getelementptr <vscale x 4 x i32>, ptr %p, i64 2
  load <vscale x 4 x i32>, ptr %gep1
  load <vscale x 4 x i32>, ptr %gep2
  load <vscale x 4 x i32>, ptr %gep3
  ret void
}

declare i64 @llvm.vscale.i64()

; CHECK-LABEL: gep_llvm_vscale_squared_may_alias
; CHECK-DAG: MayAlias:      <vscale x 4 x i32>* %gep1, <vscale x 4 x i32>* %gep2
define void @gep_llvm_vscale_squared_may_alias(ptr %p) {
  %t1 = tail call i64 @llvm.vscale.i64()
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 %t1
  %gep2 = getelementptr i32, ptr %p, i64 1
  load <vscale x 4 x i32>, ptr %gep1
  load <vscale x 4 x i32>, ptr %gep2
  ret void
}

; getelementptr + bitcast

; CHECK-LABEL: gep_bitcast_1
; CHECK-DAG:   MustAlias:    i32* %p, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     i32* %gep1, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     i32* %gep1, i32* %p
; CHECK-DAG:   MayAlias:     i32* %gep2, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     i32* %gep1, i32* %gep2
; CHECK-DAG:   NoAlias:      i32* %gep2, i32* %p
define void @gep_bitcast_1(ptr %p) {
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 0
  %gep2 = getelementptr i32, ptr %p, i64 4
  load <vscale x 4 x i32>, ptr %p
  load i32, ptr %gep1
  load i32, ptr %gep2
  load i32, ptr %p
  ret void
}

; CHECK-LABEL: gep_bitcast_2
; CHECK-DAG:  MustAlias:    <vscale x 4 x float>* %p, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep1, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep1, <vscale x 4 x float>* %p
; CHECK-DAG:  MayAlias:     float* %gep2, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     i32* %gep1, float* %gep2
; CHECK-DAG:  MayAlias:     float* %gep2, <vscale x 4 x float>* %p
define void @gep_bitcast_2(ptr %p) {
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 0
  %gep2 = getelementptr <vscale x 4 x float>, ptr %p, i64 1, i64 0
  load i32, ptr %gep1
  load float, ptr %gep2
  load <vscale x 4 x i32>, ptr %p
  load <vscale x 4 x float>, ptr %p
  ret void
}

; negative offset tests

; CHECK-LABEL: gep_neg_notscalable
; CHECK-DAG:   MayAlias:    <4 x i32>* %p, <4 x i32>* %vm16
; CHECK-DAG:   NoAlias:     <4 x i32>* %m16, <4 x i32>* %p
; CHECK-DAG:   MayAlias:    <4 x i32>* %m16, <4 x i32>* %vm16
; CHECK-DAG:   MayAlias:    <4 x i32>* %p, <4 x i32>* %vm16m16
; CHECK-DAG:   NoAlias:     <4 x i32>* %vm16, <4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:    <4 x i32>* %m16, <4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:    <4 x i32>* %m16pv16, <4 x i32>* %p
; CHECK-DAG:   MayAlias:    <4 x i32>* %m16pv16, <4 x i32>* %vm16
; CHECK-DAG:   MayAlias:    <4 x i32>* %m16, <4 x i32>* %m16pv16
; CHECK-DAG:   MayAlias:    <4 x i32>* %m16pv16, <4 x i32>* %vm16m16
define void @gep_neg_notscalable(ptr %p) vscale_range(1,16) {
  %vm16 = getelementptr <vscale x 4 x i32>, ptr %p, i64 -1
  %m16 = getelementptr <4 x i32>, ptr %p, i64 -1
  %vm16m16 = getelementptr <4 x i32>, ptr %vm16, i64 -1
  %m16pv16 = getelementptr <vscale x 4 x i32>, ptr %m16, i64 1
  load <4 x i32>, ptr %p
  load <4 x i32>, ptr %vm16
  load <4 x i32>, ptr %m16
  load <4 x i32>, ptr %vm16m16
  load <4 x i32>, ptr %m16pv16
  ret void
}

; CHECK-LABEL: gep_neg_scalable
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %vm16, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %m16pv16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %vm16m16
define void @gep_neg_scalable(ptr %p) vscale_range(1,16) {
  %vm16 = getelementptr <vscale x 4 x i32>, ptr %p, i64 -1
  %m16 = getelementptr <4 x i32>, ptr %p, i64 -1
  %vm16m16 = getelementptr <4 x i32>, ptr %vm16, i64 -1
  %m16pv16 = getelementptr <vscale x 4 x i32>, ptr %vm16, i64 1
  load <vscale x 4 x i32>, ptr %p
  load <vscale x 4 x i32>, ptr %vm16
  load <vscale x 4 x i32>, ptr %m16
  load <vscale x 4 x i32>, ptr %vm16m16
  load <vscale x 4 x i32>, ptr %m16pv16
  ret void
}

; CHECK-LABEL: gep_pos_notscalable
; CHECK-DAG:   MayAlias:     <4 x i32>* %p, <4 x i32>* %vm16
; CHECK-DAG:   NoAlias:      <4 x i32>* %m16, <4 x i32>* %p
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16, <4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <4 x i32>* %p, <4 x i32>* %vm16m16
; CHECK-DAG:   NoAlias:      <4 x i32>* %vm16, <4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16, <4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16pv16, <4 x i32>* %p
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16pv16, <4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16, <4 x i32>* %m16pv16
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16pv16, <4 x i32>* %vm16m16
define void @gep_pos_notscalable(ptr %p) vscale_range(1,16) {
  %vm16 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1
  %m16 = getelementptr <4 x i32>, ptr %p, i64 1
  %vm16m16 = getelementptr <4 x i32>, ptr %vm16, i64 1
  %m16pv16 = getelementptr <vscale x 4 x i32>, ptr %vm16, i64 -1
  load <4 x i32>, ptr %p
  load <4 x i32>, ptr %vm16
  load <4 x i32>, ptr %m16
  load <4 x i32>, ptr %vm16m16
  load <4 x i32>, ptr %m16pv16
  ret void
}

; CHECK-LABEL: gep_pos_scalable
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %vm16, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %m16pv16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %vm16m16
define void @gep_pos_scalable(ptr %p) vscale_range(1,16) {
  %vm16 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1
  %m16 = getelementptr <4 x i32>, ptr %p, i64 1
  %vm16m16 = getelementptr <4 x i32>, ptr %vm16, i64 1
  %m16pv16 = getelementptr <vscale x 4 x i32>, ptr %vm16, i64 -1
  load <vscale x 4 x i32>, ptr %p
  load <vscale x 4 x i32>, ptr %vm16
  load <vscale x 4 x i32>, ptr %m16
  load <vscale x 4 x i32>, ptr %vm16m16
  load <vscale x 4 x i32>, ptr %m16pv16
  ret void
}

; CHECK-LABEL: v1v2types
; CHECK-DAG:  MustAlias:    <4 x i32>* %p, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16
; CHECK-DAG:  MayAlias:     <4 x i32>* %p, <vscale x 4 x i32>* %vm16
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %p, <4 x i32>* %vm16
; CHECK-DAG:  MayAlias:     <4 x i32>* %p, <4 x i32>* %vm16
; CHECK-DAG:  MustAlias:    <4 x i32>* %vm16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %p
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %m16, <4 x i32>* %p
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %m16, <4 x i32>* %vm16
; CHECK-DAG:  NoAlias:      <4 x i32>* %m16, <vscale x 4 x i32>* %p
; CHECK-DAG:  NoAlias:      <4 x i32>* %m16, <4 x i32>* %p
; CHECK-DAG:  MayAlias:     <4 x i32>* %m16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:  MayAlias:     <4 x i32>* %m16, <4 x i32>* %vm16
; CHECK-DAG:  MustAlias:    <4 x i32>* %m16, <vscale x 4 x i32>* %m16
define void @v1v2types(ptr %p) vscale_range(1,16) {
  %vm16 = getelementptr <vscale x 4 x i32>, ptr %p, i64 -1
  %m16 = getelementptr <4 x i32>, ptr %p, i64 -1
  load <vscale x 4 x i32>, ptr %p
  load <4 x i32>, ptr %p
  load <vscale x 4 x i32>, ptr %vm16
  load <4 x i32>, ptr %vm16
  load <vscale x 4 x i32>, ptr %m16
  load <4 x i32>, ptr %m16
  ret void
}

; VScale intrinsic offset tests

; CHECK-LABEL: vscale_neg_notscalable
; CHECK-DAG:   NoAlias:     <4 x i32>* %p, <4 x i32>* %vm16
; CHECK-DAG:   NoAlias:     <4 x i32>* %m16, <4 x i32>* %p
; CHECK-DAG:   MayAlias:    <4 x i32>* %m16, <4 x i32>* %vm16
; CHECK-DAG:   MayAlias:    <4 x i32>* %p, <4 x i32>* %vm16m16
; CHECK-DAG:   NoAlias:     <4 x i32>* %vm16, <4 x i32>* %vm16m16
; CHECK-DAG:   NoAlias:     <4 x i32>* %m16, <4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:    <4 x i32>* %m16pv16, <4 x i32>* %p
; CHECK-DAG:   NoAlias:     <4 x i32>* %m16pv16, <4 x i32>* %vm16
; CHECK-DAG:   NoAlias:     <4 x i32>* %m16, <4 x i32>* %m16pv16
; CHECK-DAG:   MayAlias:    <4 x i32>* %m16pv16, <4 x i32>* %vm16m16
define void @vscale_neg_notscalable(ptr %p) {
  %v = call i64 @llvm.vscale.i64()
  %vp = mul nsw i64 %v, 16
  %vm = mul nsw i64 %v, -16
  %vm16 = getelementptr i8, ptr %p, i64 %vm
  %m16 = getelementptr <4 x i32>, ptr %p, i64 -1
  %vm16m16 = getelementptr <4 x i32>, ptr %vm16, i64 -1
  %m16pv16 = getelementptr i8, ptr %m16, i64 %vp
  load <4 x i32>, ptr %p
  load <4 x i32>, ptr %vm16
  load <4 x i32>, ptr %m16
  load <4 x i32>, ptr %vm16m16
  load <4 x i32>, ptr %m16pv16
  ret void
}

; CHECK-LABEL: vscale_neg_scalable
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %vm16, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %m16pv16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %vm16m16
define void @vscale_neg_scalable(ptr %p) {
  %v = call i64 @llvm.vscale.i64()
  %vp = mul nsw i64 %v, 16
  %vm = mul nsw i64 %v, -16
  %vm16 = getelementptr i8, ptr %p, i64 %vm
  %m16 = getelementptr <4 x i32>, ptr %p, i64 -1
  %vm16m16 = getelementptr <4 x i32>, ptr %vm16, i64 -1
  %m16pv16 = getelementptr i8, ptr %m16, i64 %vp
  load <vscale x 4 x i32>, ptr %p
  load <vscale x 4 x i32>, ptr %vm16
  load <vscale x 4 x i32>, ptr %m16
  load <vscale x 4 x i32>, ptr %vm16m16
  load <vscale x 4 x i32>, ptr %m16pv16
  ret void
}

; CHECK-LABEL: vscale_pos_notscalable
; CHECK-DAG:   NoAlias:      <4 x i32>* %p, <4 x i32>* %vm16
; CHECK-DAG:   NoAlias:      <4 x i32>* %m16, <4 x i32>* %p
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16, <4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <4 x i32>* %p, <4 x i32>* %vm16m16
; CHECK-DAG:   NoAlias:      <4 x i32>* %vm16, <4 x i32>* %vm16m16
; CHECK-DAG:   NoAlias:      <4 x i32>* %m16, <4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16pv16, <4 x i32>* %p
; CHECK-DAG:   NoAlias:      <4 x i32>* %m16pv16, <4 x i32>* %vm16
; CHECK-DAG:   NoAlias:      <4 x i32>* %m16, <4 x i32>* %m16pv16
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16pv16, <4 x i32>* %vm16m16
define void @vscale_pos_notscalable(ptr %p) {
  %v = call i64 @llvm.vscale.i64()
  %vp = mul nsw i64 %v, 16
  %vm = mul nsw i64 %v, -16
  %vm16 = getelementptr i8, ptr %p, i64 %vp
  %m16 = getelementptr <4 x i32>, ptr %p, i64 1
  %vm16m16 = getelementptr <4 x i32>, ptr %vm16, i64 1
  %m16pv16 = getelementptr i8, ptr %m16, i64 %vm
  load <4 x i32>, ptr %p
  load <4 x i32>, ptr %vm16
  load <4 x i32>, ptr %m16
  load <4 x i32>, ptr %vm16m16
  load <4 x i32>, ptr %m16pv16
  ret void
}

; CHECK-LABEL: vscale_pos_scalable
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %vm16, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vm16m16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %m16pv16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16pv16, <vscale x 4 x i32>* %vm16m16
define void @vscale_pos_scalable(ptr %p) {
  %v = call i64 @llvm.vscale.i64()
  %vp = mul nsw i64 %v, 16
  %vm = mul nsw i64 %v, -16
  %vm16 = getelementptr i8, ptr %p, i64 %vp
  %m16 = getelementptr <4 x i32>, ptr %p, i64 1
  %vm16m16 = getelementptr <4 x i32>, ptr %vm16, i64 1
  %m16pv16 = getelementptr i8, ptr %m16, i64 %vm
  load <vscale x 4 x i32>, ptr %p
  load <vscale x 4 x i32>, ptr %vm16
  load <vscale x 4 x i32>, ptr %m16
  load <vscale x 4 x i32>, ptr %vm16m16
  load <vscale x 4 x i32>, ptr %m16pv16
  ret void
}

; CHECK-LABEL: vscale_v1v2types
; CHECK-DAG:   MustAlias:    <4 x i32>* %p, <vscale x 4 x i32>* %p
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   NoAlias:      <4 x i32>* %p, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %p, <4 x i32>* %vm16
; CHECK-DAG:   NoAlias:      <4 x i32>* %p, <4 x i32>* %vm16
; CHECK-DAG:   MustAlias:    <4 x i32>* %vm16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <4 x i32>* %vm16
; CHECK-DAG:   NoAlias:      <4 x i32>* %m16, <vscale x 4 x i32>* %p
; CHECK-DAG:   NoAlias:      <4 x i32>* %m16, <4 x i32>* %p
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16, <vscale x 4 x i32>* %vm16
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16, <4 x i32>* %vm16
; CHECK-DAG:   MustAlias:    <4 x i32>* %m16, <vscale x 4 x i32>* %m16
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vp16
; CHECK-DAG:   NoAlias:      <4 x i32>* %p, <vscale x 4 x i32>* %vp16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %vm16, <vscale x 4 x i32>* %vp16
; CHECK-DAG:   MayAlias:     <4 x i32>* %vm16, <vscale x 4 x i32>* %vp16
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %m16, <vscale x 4 x i32>* %vp16
; CHECK-DAG:   MayAlias:     <4 x i32>* %m16, <vscale x 4 x i32>* %vp16
define void @vscale_v1v2types(ptr %p) {
  %v = call i64 @llvm.vscale.i64()
  %vp = mul nsw i64 %v, 16
  %vm = mul nsw i64 %v, -16
  %vp16 = getelementptr i8, ptr %p, i64 %vp
  %vm16 = getelementptr i8, ptr %p, i64 %vm
  %m16 = getelementptr <4 x i32>, ptr %p, i64 -1
  load <vscale x 4 x i32>, ptr %p
  load <4 x i32>, ptr %p
  load <vscale x 4 x i32>, ptr %vm16
  load <4 x i32>, ptr %vm16
  load <vscale x 4 x i32>, ptr %m16
  load <4 x i32>, ptr %m16
  load <vscale x 4 x i32>, ptr %vp16
  ret void
}

; CHECK-LABEL: vscale_negativescale
; CHECK-DAG:   MayAlias:    <vscale x 4 x i32>* %p, <vscale x 4 x i32>* %vm16
define void @vscale_negativescale(ptr %p) vscale_range(1,16) {
  %v = call i64 @llvm.vscale.i64()
  %vm = mul nsw i64 %v, -15
  %vm16 = getelementptr i8, ptr %p, i64 %vm
  load <vscale x 4 x i32>, ptr %vm16
  load <vscale x 4 x i32>, ptr %p
  ret void
}

; CHECK-LABEL: onevscale
; CHECK-DAG:   MustAlias:    <vscale x 4 x i32>* %vp161, <vscale x 4 x i32>* %vp162
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %vp161, <vscale x 4 x i32>* %vp161b
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %vp161b, <vscale x 4 x i32>* %vp162
define void @onevscale(ptr %p) vscale_range(1,16) {
  %v1 = call i64 @llvm.vscale.i64()
  %vp1 = mul nsw i64 %v1, 16
  %vp2 = mul nsw i64 %v1, 16
  %vp3 = mul nsw i64 %v1, 17
  %vp161 = getelementptr i8, ptr %p, i64 %vp1
  %vp162 = getelementptr i8, ptr %p, i64 %vp2
  %vp161b = getelementptr i8, ptr %vp161, i64 %vp3
  load <vscale x 4 x i32>, ptr %vp161
  load <vscale x 4 x i32>, ptr %vp162
  load <vscale x 4 x i32>, ptr %vp161b
  ret void
}

; CHECK-LABEL: twovscales
; CHECK-DAG:   MustAlias:    <vscale x 4 x i32>* %vp161, <vscale x 4 x i32>* %vp162
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %vp161, <vscale x 4 x i32>* %vp161b
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %vp161b, <vscale x 4 x i32>* %vp162
define void @twovscales(ptr %p) vscale_range(1,16) {
  %v1 = call i64 @llvm.vscale.i64()
  %v2 = call i64 @llvm.vscale.i64()
  %vp1 = mul nsw i64 %v1, 16
  %vp2 = mul nsw i64 %v2, 16
  %vp3 = mul nsw i64 %v1, 17
  %vp161 = getelementptr i8, ptr %p, i64 %vp1
  %vp162 = getelementptr i8, ptr %p, i64 %vp2
  %vp161b = getelementptr i8, ptr %vp161, i64 %vp3
  load <vscale x 4 x i32>, ptr %vp161
  load <vscale x 4 x i32>, ptr %vp162
  load <vscale x 4 x i32>, ptr %vp161b
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
define void @gep_recursion_level_1(ptr %a, ptr %p) {
  %gep = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 2
  %gep_rec_1 = getelementptr i32, ptr %gep, i64 1
  load <vscale x 4 x i32>, ptr %p
  load i32, ptr %a
  load i32, ptr %gep
  load i32, ptr %gep_rec_1
  ret void
}

; CHECK-LABEL: gep_recursion_level_1_bitcast
; CHECK-DAG:  MustAlias:    i32* %a, <vscale x 4 x i32>* %a
; CHECK-DAG:  MayAlias:     i32* %a, i32* %gep
; CHECK-DAG:  MayAlias:     i32* %a, i32* %gep_rec_1
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %a, i32* %gep
; CHECK-DAG:  MayAlias:     <vscale x 4 x i32>* %a, i32* %gep_rec_1
; CHECK-DAG:  NoAlias:      i32* %gep, i32* %gep_rec_1
define void @gep_recursion_level_1_bitcast(ptr %a) {
  %gep = getelementptr <vscale x 4 x i32>, ptr %a, i64 1, i64 2
  %gep_rec_1 = getelementptr i32, ptr %gep, i64 1
  load <vscale x 4 x i32>, ptr %a
  load i32, ptr %a
  load i32, ptr %gep
  load i32, ptr %gep_rec_1
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
define void @gep_recursion_level_2(ptr %a, ptr %p) {
  %gep = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 2
  %gep_rec_1 = getelementptr i32, ptr %gep, i64 1
  %gep_rec_2 = getelementptr i32, ptr %gep_rec_1, i64 1
  load <vscale x 4 x i32>, ptr %p
  load i32, ptr %a
  load i32, ptr %gep
  load i32, ptr %gep_rec_1
  load i32, ptr %gep_rec_2
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
define void @gep_recursion_max_lookup_depth_reached(ptr %a, ptr %p) {
  %gep = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 2
  %gep_rec_1 = getelementptr i32, ptr %gep, i64 1
  %gep_rec_2 = getelementptr i32, ptr %gep_rec_1, i64 1
  %gep_rec_3 = getelementptr i32, ptr %gep_rec_2, i64 1
  %gep_rec_4 = getelementptr i32, ptr %gep_rec_3, i64 1
  %gep_rec_5 = getelementptr i32, ptr %gep_rec_4, i64 1
  %gep_rec_6 = getelementptr i32, ptr %gep_rec_5, i64 1
  load <vscale x 4 x i32>, ptr %p
  load i32, ptr %a
  load i32, ptr %gep
  load i32, ptr %gep_rec_1
  load i32, ptr %gep_rec_2
  load i32, ptr %gep_rec_3
  load i32, ptr %gep_rec_4
  load i32, ptr %gep_rec_5
  load i32, ptr %gep_rec_6
  ret void
}

; CHECK-LABEL: gep_2048
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %off255, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %noff255, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %noff255, <vscale x 4 x i32>* %off255
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %off256, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %off255, <vscale x 4 x i32>* %off256
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %noff255, <vscale x 4 x i32>* %off256
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %noff256, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %noff256, <vscale x 4 x i32>* %off255
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %noff255, <vscale x 4 x i32>* %noff256
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %noff256, <vscale x 4 x i32>* %off256
define void @gep_2048(ptr %p) {
  %off255 = getelementptr i8, ptr %p, i64 255
  %noff255 = getelementptr i8, ptr %p, i64 -255
  %off256 = getelementptr i8, ptr %p, i64 256
  %noff256 = getelementptr i8, ptr %p, i64 -256
  load <vscale x 4 x i32>, ptr %p
  load <vscale x 4 x i32>, ptr %off255
  load <vscale x 4 x i32>, ptr %noff255
  load <vscale x 4 x i32>, ptr %off256
  load <vscale x 4 x i32>, ptr %noff256
  ret void
}

; CHECK-LABEL: gep_2048_vscalerange
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %off255, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %noff255, <vscale x 4 x i32>* %p
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %noff255, <vscale x 4 x i32>* %off255
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %off256, <vscale x 4 x i32>* %p
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %off255, <vscale x 4 x i32>* %off256
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %noff255, <vscale x 4 x i32>* %off256
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %noff256, <vscale x 4 x i32>* %p
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %noff256, <vscale x 4 x i32>* %off255
; CHECK-DAG:   MayAlias:     <vscale x 4 x i32>* %noff255, <vscale x 4 x i32>* %noff256
; CHECK-DAG:   NoAlias:      <vscale x 4 x i32>* %noff256, <vscale x 4 x i32>* %off256
define void @gep_2048_vscalerange(ptr %p) vscale_range(1,16) {
  %off255 = getelementptr i8, ptr %p, i64 255
  %noff255 = getelementptr i8, ptr %p, i64 -255
  %off256 = getelementptr i8, ptr %p, i64 256
  %noff256 = getelementptr i8, ptr %p, i64 -256
  load <vscale x 4 x i32>, ptr %p
  load <vscale x 4 x i32>, ptr %off255
  load <vscale x 4 x i32>, ptr %noff255
  load <vscale x 4 x i32>, ptr %off256
  load <vscale x 4 x i32>, ptr %noff256
  ret void
}
