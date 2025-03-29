// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -gicombiner-debug-typeinfer -combiners=MyCombiner %s 2>&1 >/dev/null | \
// RUN: FileCheck %s

// Checks reasoning of the inference rules.

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// This also checks that the type of a def is preferred when inferring the type
// of an immediate.
// CHECK:      Rule Operand Type Equivalence Classes for inference_mul_by_neg_one:
// CHECK-NEXT:         Groups for __inference_mul_by_neg_one_match_0:             [dst, x]
// CHECK-NEXT:         Groups for __inference_mul_by_neg_one_apply_0:             [dst, x]
// CHECK-NEXT: Final Type Equivalence Classes: [dst, x]
// CHECK-NEXT: INFER: imm 0 -> GITypeOf<$dst>
// CHECK-NEXT: Apply patterns for rule inference_mul_by_neg_one after inference:
// CHECK-NEXT:   (CodeGenInstructionPattern name:__inference_mul_by_neg_one_apply_0 G_SUB operands:[<def>$dst, (GITypeOf<$dst> 0), $x])
def inference_mul_by_neg_one: GICombineRule <
  (defs root:$dst),
  (match (G_MUL $dst, $x, -1)),
  (apply (G_SUB $dst, 0, $x))
>;

// CHECK:      Rule Operand Type Equivalence Classes for infer_complex_tempreg:
// CHECK-NEXT:         Groups for __infer_complex_tempreg_match_0:                [dst]   [x, y, z]
// CHECK-NEXT:         Groups for __infer_complex_tempreg_apply_0:                [tmp2]  [x, y]
// CHECK-NEXT:         Groups for __infer_complex_tempreg_apply_1:                [tmp, tmp2]
// CHECK-NEXT:         Groups for __infer_complex_tempreg_apply_2:                [dst, tmp]
// CHECK-NEXT: Final Type Equivalence Classes: [dst, tmp, tmp2] [x, y, z]
// CHECK-NEXT: INFER: MachineOperand $tmp2 -> GITypeOf<$dst>
// CHECK-NEXT: INFER: MachineOperand $tmp -> GITypeOf<$dst>
// CHECK-NEXT: Apply patterns for rule infer_complex_tempreg after inference:
// CHECK-NEXT:   (CodeGenInstructionPattern name:__infer_complex_tempreg_apply_0 G_BUILD_VECTOR operands:[<def>GITypeOf<$dst>:$tmp2, $x, $y])
// CHECK-NEXT:   (CodeGenInstructionPattern name:__infer_complex_tempreg_apply_1 G_FNEG operands:[<def>GITypeOf<$dst>:$tmp, GITypeOf<$dst>:$tmp2])
// CHECK-NEXT:   (CodeGenInstructionPattern name:__infer_complex_tempreg_apply_2 G_FNEG operands:[<def>$dst, GITypeOf<$dst>:$tmp])
def infer_complex_tempreg: GICombineRule <
  (defs root:$dst),
  (match (G_MERGE_VALUES $dst, $x, $y, $z)),
  (apply (G_BUILD_VECTOR $tmp2, $x, $y),
         (G_FNEG $tmp, $tmp2),
         (G_FNEG $dst, $tmp))
>;

// CHECK:      Rule Operand Type Equivalence Classes for infer_variadic_outs:
// CHECK-NEXT:         Groups for  __infer_variadic_outs_match_0:          [x, y]  [vec]
// CHECK-NEXT:         Groups for  __infer_variadic_outs_match_1:          [dst, x]
// CHECK-NEXT:         Groups for  __infer_variadic_outs_apply_0:          [tmp, y]
// CHECK-NEXT:         Groups for  __infer_variadic_outs_apply_1:
// CHECK-NEXT: Final Type Equivalence Classes: [tmp, dst, x, y]  [vec]
// CHECK-NEXT: INFER: MachineOperand $tmp -> GITypeOf<$dst>
// CHECK-NEXT: Apply patterns for rule infer_variadic_outs after inference:
// CHECK-NEXT:   (CodeGenInstructionPattern name:__infer_variadic_outs_apply_0 G_FNEG operands:[<def>GITypeOf<$dst>:$tmp, $y])
// CHECK-NEXT:   (CodeGenInstructionPattern name:__infer_variadic_outs_apply_1 COPY operands:[<def>$dst, GITypeOf<$dst>:$tmp])
def infer_variadic_outs: GICombineRule <
  (defs root:$dst),
  (match  (G_UNMERGE_VALUES $x, $y, $vec),
          (G_FNEG $dst, $x)),
  (apply (G_FNEG $tmp, $y),
         (COPY $dst, $tmp))
>;

// Check that the type of an immediate is inferred when there is just one
// element in the corresponding equivalence class.
// CHECK:      Rule Operand Type Equivalence Classes for infer_imm_0:
// CHECK-NEXT:   Groups for __infer_imm_0_match_0: [dst]
// CHECK-NEXT:   Groups for __infer_imm_0_apply_0: [dst]
// CHECK-NEXT: Final Type Equivalence Classes: [dst]
// CHECK-NEXT: INFER: imm 1 -> GITypeOf<$dst>
// CHECK-NEXT: INFER: imm 2 -> GITypeOf<$dst>
// CHECK-NEXT: Apply patterns for rule infer_imm_0 after inference:
// CHECK-NEXT:   (CodeGenInstructionPattern name:__infer_imm_0_apply_0 G_ADD operands:[<def>$dst, (GITypeOf<$dst> 1), (GITypeOf<$dst> 2)])
def infer_imm_0 : GICombineRule<
  (defs root:$dst),
  (match (G_ADD $dst, 0, 3)),
  (apply (G_ADD $dst, 1, 2))
>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  inference_mul_by_neg_one,
  infer_complex_tempreg,
  infer_variadic_outs,
  infer_imm_0,
]>;
