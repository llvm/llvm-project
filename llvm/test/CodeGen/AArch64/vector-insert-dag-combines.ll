; RUN: llc -debug-only=isel -o /dev/null < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; NOTE: Due to their nature the expected inserts and extracts often emit no
; instructions and so these tests verify the output of DAGCombiner directly.

target triple = "aarch64-unknown-linux-gnu"

; CHECK: Initial selection DAG: %bb.0 'insert_small_fixed_into_big_fixed:'
; CHECK: SelectionDAG has 10 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:         t2: v8i8,ch = CopyFromReg t0, Register:v8i8 %0
; CHECK:       t4: v4i8 = extract_subvector t2, Constant:i64<0>
; CHECK:     t6: v16i8 = insert_subvector undef:v16i8, t4, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:v16i8 $q0, t6
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:v16i8 $q0, t8:1

; CHECK: Optimized lowered selection DAG: %bb.0 'insert_small_fixed_into_big_fixed:'
; CHECK: SelectionDAG has 9 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:       t2: v8i8,ch = CopyFromReg t0, Register:v8i8 %0
; CHECK:     t10: v16i8 = insert_subvector undef:v16i8, t2, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:v16i8 $q0, t10
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:v16i8 $q0, t8:1

define <16 x i8> @insert_small_fixed_into_big_fixed(<8 x i8> %a) #0 {
  %extract = call <4 x i8> @llvm.vector.extract(<8 x i8> %a, i64 0)
  %insert = call <16 x i8> @llvm.vector.insert(<16 x i8> undef, <4 x i8> %extract, i64 0)
  ret <16 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'insert_small_fixed_into_big_scalable:'
; CHECK: SelectionDAG has 10 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:         t2: v8i8,ch = CopyFromReg t0, Register:v8i8 %0
; CHECK:       t4: v4i8 = extract_subvector t2, Constant:i64<0>
; CHECK:     t6: nxv16i8 = insert_subvector undef:nxv16i8, t4, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:nxv16i8 $z0, t6
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:nxv16i8 $z0, t8:1

; CHECK: Optimized lowered selection DAG: %bb.0 'insert_small_fixed_into_big_scalable:'
; CHECK: SelectionDAG has 9 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:       t2: v8i8,ch = CopyFromReg t0, Register:v8i8 %0
; CHECK:     t10: nxv16i8 = insert_subvector undef:nxv16i8, t2, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:nxv16i8 $z0, t10
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:nxv16i8 $z0, t8:1

define <vscale x 16 x i8> @insert_small_fixed_into_big_scalable(<8 x i8> %a) #0 {
  %extract = call <4 x i8> @llvm.vector.extract(<8 x i8> %a, i64 0)
  %insert = call <vscale x 16 x i8> @llvm.vector.insert(<vscale x 16 x i8> undef, <4 x i8> %extract, i64 0)
  ret <vscale x 16 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'insert_small_scalable_into_big_fixed:'
; CHECK: SelectionDAG has 11 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:           t2: nxv8i16,ch = CopyFromReg t0, Register:nxv8i16 %0
; CHECK:         t3: nxv8i8 = truncate t2
; CHECK:       t5: v4i8 = extract_subvector t3, Constant:i64<0>
; CHECK:     t7: v16i8 = insert_subvector undef:v16i8, t5, Constant:i64<0>
; CHECK:   t9: ch,glue = CopyToReg t0, Register:v16i8 $q0, t7
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:v16i8 $q0, t9:1

; CHECK: Optimized lowered selection DAG: %bb.0 'insert_small_scalable_into_big_fixed:'
; CHECK: SelectionDAG has 11 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:           t2: nxv8i16,ch = CopyFromReg t0, Register:nxv8i16 %0
; CHECK:         t3: nxv8i8 = truncate t2
; CHECK:       t5: v4i8 = extract_subvector t3, Constant:i64<0>
; CHECK:     t7: v16i8 = insert_subvector undef:v16i8, t5, Constant:i64<0>
; CHECK:   t9: ch,glue = CopyToReg t0, Register:v16i8 $q0, t7
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:v16i8 $q0, t9:1

; Resulting insert would not be legal, so there's no transformation.
define <16 x i8> @insert_small_scalable_into_big_fixed(<vscale x 8 x i8> %a) #0 {
  %extract = call <4 x i8> @llvm.vector.extract(<vscale x 8 x i8> %a, i64 0)
  %insert = call <16 x i8> @llvm.vector.insert(<16 x i8> undef, <4 x i8> %extract, i64 0)
  ret <16 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'insert_small_scalable_into_big_scalable_1:'
; CHECK: SelectionDAG has 11 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:           t2: nxv8i16,ch = CopyFromReg t0, Register:nxv8i16 %0
; CHECK:         t3: nxv8i8 = truncate t2
; CHECK:       t5: v4i8 = extract_subvector t3, Constant:i64<0>
; CHECK:     t7: nxv16i8 = insert_subvector undef:nxv16i8, t5, Constant:i64<0>
; CHECK:   t9: ch,glue = CopyToReg t0, Register:nxv16i8 $z0, t7
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:nxv16i8 $z0, t9:1

; CHECK: Optimized lowered selection DAG: %bb.0 'insert_small_scalable_into_big_scalable_1:'
; CHECK: SelectionDAG has 10 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:         t2: nxv8i16,ch = CopyFromReg t0, Register:nxv8i16 %0
; CHECK:       t3: nxv8i8 = truncate t2
; CHECK:     t11: nxv16i8 = insert_subvector undef:nxv16i8, t3, Constant:i64<0>
; CHECK:   t9: ch,glue = CopyToReg t0, Register:nxv16i8 $z0, t11
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:nxv16i8 $z0, t9:1

define <vscale x 16 x i8> @insert_small_scalable_into_big_scalable_1(<vscale x 8 x i8> %a) #0 {
  %extract = call <4 x i8> @llvm.vector.extract(<vscale x 8 x i8> %a, i64 0)
  %insert = call <vscale x 16 x i8> @llvm.vector.insert(<vscale x 16 x i8> undef, <4 x i8> %extract, i64 0)
  ret <vscale x 16 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'insert_small_scalable_into_big_scalable_2:'
; CHECK: SelectionDAG has 11 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:           t2: nxv8i16,ch = CopyFromReg t0, Register:nxv8i16 %0
; CHECK:         t3: nxv8i8 = truncate t2
; CHECK:       t5: nxv4i8 = extract_subvector t3, Constant:i64<0>
; CHECK:     t7: nxv16i8 = insert_subvector undef:nxv16i8, t5, Constant:i64<0>
; CHECK:   t9: ch,glue = CopyToReg t0, Register:nxv16i8 $z0, t7
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:nxv16i8 $z0, t9:1

; CHECK: Optimized lowered selection DAG: %bb.0 'insert_small_scalable_into_big_scalable_2:'
; CHECK: SelectionDAG has 10 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:         t2: nxv8i16,ch = CopyFromReg t0, Register:nxv8i16 %0
; CHECK:       t3: nxv8i8 = truncate t2
; CHECK:     t11: nxv16i8 = insert_subvector undef:nxv16i8, t3, Constant:i64<0>
; CHECK:   t9: ch,glue = CopyToReg t0, Register:nxv16i8 $z0, t11
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:nxv16i8 $z0, t9:1

define <vscale x 16 x i8> @insert_small_scalable_into_big_scalable_2(<vscale x 8 x i8> %a) #0 {
  %extract = call <vscale x 4 x i8> @llvm.vector.extract(<vscale x 8 x i8> %a, i64 0)
  %insert = call <vscale x 16 x i8> @llvm.vector.insert(<vscale x 16 x i8> undef, <vscale x 4 x i8> %extract, i64 0)
  ret <vscale x 16 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'extract_small_fixed_from_big_fixed:'
; CHECK: SelectionDAG has 10 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:         t2: v16i8,ch = CopyFromReg t0, Register:v16i8 %0
; CHECK:       t4: v4i8 = extract_subvector t2, Constant:i64<0>
; CHECK:     t6: v8i8 = insert_subvector undef:v8i8, t4, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:v8i8 $d0, t6
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:v8i8 $d0, t8:1

; CHECK: Optimized lowered selection DAG: %bb.0 'extract_small_fixed_from_big_fixed:'
; CHECK: SelectionDAG has 8 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:       t2: v16i8,ch = CopyFromReg t0, Register:v16i8 %0
; CHECK:     t10: v8i8 = extract_subvector t2, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:v8i8 $d0, t10
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:v8i8 $d0, t8:1

define <8 x i8> @extract_small_fixed_from_big_fixed(<16 x i8> %a) #0 {
  %extract = call <4 x i8> @llvm.vector.extract(<16 x i8> %a, i64 0)
  %insert = call <8 x i8> @llvm.vector.insert(<8 x i8> undef, <4 x i8> %extract, i64 0)
  ret <8 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'extract_small_scalable_from_big_fixed:'
; CHECK: SelectionDAG has 11 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:           t2: v16i8,ch = CopyFromReg t0, Register:v16i8 %0
; CHECK:         t4: v4i8 = extract_subvector t2, Constant:i64<0>
; CHECK:       t6: nxv8i8 = insert_subvector undef:nxv8i8, t4, Constant:i64<0>
; CHECK:     t7: nxv8i16 = any_extend t6
; CHECK:   t9: ch,glue = CopyToReg t0, Register:nxv8i16 $z0, t7
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:nxv8i16 $z0, t9:1

; CHECK: Optimized lowered selection DAG: %bb.0 'extract_small_scalable_from_big_fixed:'
; CHECK: SelectionDAG has 11 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:           t2: v16i8,ch = CopyFromReg t0, Register:v16i8 %0
; CHECK:         t4: v4i8 = extract_subvector t2, Constant:i64<0>
; CHECK:       t6: nxv8i8 = insert_subvector undef:nxv8i8, t4, Constant:i64<0>
; CHECK:     t7: nxv8i16 = any_extend t6
; CHECK:   t9: ch,glue = CopyToReg t0, Register:nxv8i16 $z0, t7
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:nxv8i16 $z0, t9:1

; Resulting insert would not be legal, so there's no transformation.
define <vscale x 8 x i8> @extract_small_scalable_from_big_fixed(<16 x i8> %a) #0 {
  %extract = call <4 x i8> @llvm.vector.extract(<16 x i8> %a, i64 0)
  %insert = call <vscale x 8 x i8> @llvm.vector.insert(<vscale x 8 x i8> undef, <4 x i8> %extract, i64 0)
  ret <vscale x 8 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'extract_small_fixed_from_big_scalable:'
; CHECK: SelectionDAG has 10 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:         t2: nxv16i8,ch = CopyFromReg t0, Register:nxv16i8 %0
; CHECK:       t4: v4i8 = extract_subvector t2, Constant:i64<0>
; CHECK:     t6: v8i8 = insert_subvector undef:v8i8, t4, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:v8i8 $d0, t6
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:v8i8 $d0, t8:1

; CHECK: Optimized lowered selection DAG: %bb.0 'extract_small_fixed_from_big_scalable:'
; CHECK: SelectionDAG has 8 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:       t2: nxv16i8,ch = CopyFromReg t0, Register:nxv16i8 %0
; CHECK:     t10: v8i8 = extract_subvector t2, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:v8i8 $d0, t10
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:v8i8 $d0, t8:1

define <8 x i8> @extract_small_fixed_from_big_scalable(<vscale x 16 x i8> %a) #0 {
  %extract = call <4 x i8> @llvm.vector.extract(<vscale x 16 x i8> %a, i64 0)
  %insert = call <8 x i8> @llvm.vector.insert(<8 x i8> undef, <4 x i8> %extract, i64 0)
  ret <8 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'extract_small_scalable_from_big_scalable_1:'
; CHECK: SelectionDAG has 11 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:           t2: nxv16i8,ch = CopyFromReg t0, Register:nxv16i8 %0
; CHECK:         t4: v4i8 = extract_subvector t2, Constant:i64<0>
; CHECK:       t6: nxv8i8 = insert_subvector undef:nxv8i8, t4, Constant:i64<0>
; CHECK:     t7: nxv8i16 = any_extend t6
; CHECK:   t9: ch,glue = CopyToReg t0, Register:nxv8i16 $z0, t7
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:nxv8i16 $z0, t9:1

; CHECK: Optimized lowered selection DAG: %bb.0 'extract_small_scalable_from_big_scalable_1:'
; CHECK: SelectionDAG has 9 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:         t2: nxv16i8,ch = CopyFromReg t0, Register:nxv16i8 %0
; CHECK:       t11: nxv8i8 = extract_subvector t2, Constant:i64<0>
; CHECK:     t7: nxv8i16 = any_extend t11
; CHECK:   t9: ch,glue = CopyToReg t0, Register:nxv8i16 $z0, t7
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:nxv8i16 $z0, t9:1

define <vscale x 8 x i8> @extract_small_scalable_from_big_scalable_1(<vscale x 16 x i8> %a) #0 {
  %extract = call <4 x i8> @llvm.vector.extract(<vscale x 16 x i8> %a, i64 0)
  %insert = call <vscale x 8 x i8> @llvm.vector.insert(<vscale x 8 x i8> undef, <4 x i8> %extract, i64 0)
  ret <vscale x 8 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'extract_small_scalable_from_big_scalable_2:'
; CHECK: SelectionDAG has 11 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:           t2: nxv16i8,ch = CopyFromReg t0, Register:nxv16i8 %0
; CHECK:         t4: nxv4i8 = extract_subvector t2, Constant:i64<0>
; CHECK:       t6: nxv8i8 = insert_subvector undef:nxv8i8, t4, Constant:i64<0>
; CHECK:     t7: nxv8i16 = any_extend t6
; CHECK:   t9: ch,glue = CopyToReg t0, Register:nxv8i16 $z0, t7
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:nxv8i16 $z0, t9:1

; CHECK: Optimized lowered selection DAG: %bb.0 'extract_small_scalable_from_big_scalable_2:'
; CHECK: SelectionDAG has 9 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:         t2: nxv16i8,ch = CopyFromReg t0, Register:nxv16i8 %0
; CHECK:       t11: nxv8i8 = extract_subvector t2, Constant:i64<0>
; CHECK:     t7: nxv8i16 = any_extend t11
; CHECK:   t9: ch,glue = CopyToReg t0, Register:nxv8i16 $z0, t7
; CHECK:   t10: ch = AArch64ISD::RET_GLUE t9, Register:nxv8i16 $z0, t9:1

define <vscale x 8 x i8> @extract_small_scalable_from_big_scalable_2(<vscale x 16 x i8> %a) #0 {
  %extract = call <vscale x 4 x i8> @llvm.vector.extract(<vscale x 16 x i8> %a, i64 0)
  %insert = call <vscale x 8 x i8> @llvm.vector.insert(<vscale x 8 x i8> undef, <vscale x 4 x i8> %extract, i64 0)
  ret <vscale x 8 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'extract_fixed_from_scalable:'
; CHECK: SelectionDAG has 10 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:         t2: nxv16i8,ch = CopyFromReg t0, Register:nxv16i8 %0
; CHECK:       t4: v4i8 = extract_subvector t2, Constant:i64<0>
; CHECK:     t6: v16i8 = insert_subvector undef:v16i8, t4, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:v16i8 $q0, t6
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:v16i8 $q0, t8:1

; CHECK: Optimized lowered selection DAG: %bb.0 'extract_fixed_from_scalable:'
; CHECK: SelectionDAG has 8 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:       t2: nxv16i8,ch = CopyFromReg t0, Register:nxv16i8 %0
; CHECK:     t10: v16i8 = extract_subvector t2, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:v16i8 $q0, t10
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:v16i8 $q0, t8:1

; A variant of insert_small_scalable_into_big_fixed whose vector types prevent
; the expected transformation because the resulting insert would not be legal.
; In this instance their matching minimum vector lengths allow us to perform the
; opposite transformation and emit an extract instead.
define <16 x i8> @extract_fixed_from_scalable(<vscale x 16 x i8> %a) #0 {
  %extract = call <4 x i8> @llvm.vector.extract(<vscale x 16 x i8> %a, i64 0)
  %insert = call <16 x i8> @llvm.vector.insert(<16 x i8> undef, <4 x i8> %extract, i64 0)
  ret <16 x i8> %insert
}

; CHECK: Initial selection DAG: %bb.0 'insert_fixed_into_scalable:'
; CHECK: SelectionDAG has 10 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:         t2: v16i8,ch = CopyFromReg t0, Register:v16i8 %0
; CHECK:       t4: v4i8 = extract_subvector t2, Constant:i64<0>
; CHECK:     t6: nxv16i8 = insert_subvector undef:nxv16i8, t4, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:nxv16i8 $z0, t6
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:nxv16i8 $z0, t8:1

; CHECK: Optimized lowered selection DAG: %bb.0 'insert_fixed_into_scalable:'
; CHECK: SelectionDAG has 9 nodes:
; CHECK:   t0: ch,glue = EntryToken
; CHECK:       t2: v16i8,ch = CopyFromReg t0, Register:v16i8 %0
; CHECK:     t10: nxv16i8 = insert_subvector undef:nxv16i8, t2, Constant:i64<0>
; CHECK:   t8: ch,glue = CopyToReg t0, Register:nxv16i8 $z0, t10
; CHECK:   t9: ch = AArch64ISD::RET_GLUE t8, Register:nxv16i8 $z0, t8:1

; A variant of extract_small_scalable_from_big_fixed whose vector types prevent
; the expected transformation because the resulting extract would not be legal.
; In this instance their matching minimum vector lengths allow us to perform the
; opposite transformation and emit an insert instead.
define <vscale x 16 x i8> @insert_fixed_into_scalable(<16 x i8> %a) #0 {
  %extract = call <4 x i8> @llvm.vector.extract(<16 x i8> %a, i64 0)
  %insert = call <vscale x 16 x i8> @llvm.vector.insert(<vscale x 16 x i8> undef, <4 x i8> %extract, i64 0)
  ret <vscale x 16 x i8> %insert
}

attributes #0 = { "target-features"="+sve" }
