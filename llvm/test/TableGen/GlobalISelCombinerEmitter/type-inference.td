// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -gicombiner-debug-typeinfer -combiners=MyCombiner %s 2>&1 | \
// RUN: FileCheck %s

// Checks reasoning of the inference rules.

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// CHECK:      Rule Operand Type Equivalence Classes for inference_mul_by_neg_one:
// CHECK-NEXT:         __inference_mul_by_neg_one_match_0:             [dst, x]
// CHECK-NEXT:         __inference_mul_by_neg_one_apply_0:             [dst, x]
// CHECK-NEXT: (merging [dst, x] | [dst, x])
// CHECK-NEXT: Result: [dst, x]
// CHECK-NEXT: INFER: imm 0 -> GITypeOf<$x>
// CHECK-NEXT: Apply patterns for rule inference_mul_by_neg_one after inference:
// CHECK-NEXT: (CodeGenInstructionPattern name:__inference_mul_by_neg_one_apply_0 G_SUB operands:[<def>$dst, (GITypeOf<$x> 0), $x])
def inference_mul_by_neg_one: GICombineRule <
  (defs root:$dst),
  (match (G_MUL $dst, $x, -1)),
  (apply (G_SUB $dst, 0, $x))
>;

// CHECK:      Rule Operand Type Equivalence Classes for infer_complex_tempreg:
// CHECK-NEXT:         __infer_complex_tempreg_match_0:                [dst]   [x, y, z]
// CHECK-NEXT:         __infer_complex_tempreg_apply_0:                [tmp2]  [x, y]
// CHECK-NEXT:         __infer_complex_tempreg_apply_1:                [tmp, tmp2]
// CHECK-NEXT:         __infer_complex_tempreg_apply_2:                [dst, tmp]
// CHECK-NEXT: (merging [dst] | [dst, tmp])
// CHECK-NEXT: (merging [x, y, z] | [x, y])
// CHECK-NEXT: (merging [tmp2] | [tmp, tmp2])
// CHECK-NEXT: (merging [dst, tmp] | [tmp2, tmp])
// CHECK-NEXT: Result: [dst, tmp, tmp2] [x, y, z]
// CHECK-NEXT: INFER: MachineOperand $tmp2 -> GITypeOf<$dst>
// CHECK-NEXT: INFER: MachineOperand $tmp -> GITypeOf<$dst>
// CHECK-NEXT: Apply patterns for rule infer_complex_tempreg after inference:
// CHECK-NEXT: (CodeGenInstructionPattern name:__infer_complex_tempreg_apply_0 G_BUILD_VECTOR operands:[<def>GITypeOf<$dst>:$tmp2, $x, $y])
// CHECK-NEXT: (CodeGenInstructionPattern name:__infer_complex_tempreg_apply_1 G_FNEG operands:[<def>GITypeOf<$dst>:$tmp, GITypeOf<$dst>:$tmp2])
// CHECK-NEXT: (CodeGenInstructionPattern name:__infer_complex_tempreg_apply_2 G_FNEG operands:[<def>$dst, GITypeOf<$dst>:$tmp])
def infer_complex_tempreg: GICombineRule <
  (defs root:$dst),
  (match (G_MERGE_VALUES $dst, $x, $y, $z)),
  (apply (G_BUILD_VECTOR $tmp2, $x, $y),
         (G_FNEG $tmp, $tmp2),
         (G_FNEG $dst, $tmp))
>;

// CHECK:      Rule Operand Type Equivalence Classes for infer_variadic_outs:
// CHECK-NEXT:         __infer_variadic_outs_match_0:          [x, y]  [vec]
// CHECK-NEXT:         __infer_variadic_outs_match_1:          [dst, x]
// CHECK-NEXT:         __infer_variadic_outs_apply_0:          [tmp, y]
// CHECK-NEXT:         __infer_variadic_outs_apply_1:  (empty)
// CHECK-NEXT: (merging [x, y] | [dst, x])
// CHECK-NEXT: (merging [x, y, dst] | [tmp, y])
// CHECK-NEXT: Result: [x, y, dst, tmp] [vec]
// CHECK-NEXT: INFER: MachineOperand $tmp -> GITypeOf<$x>
// CHECK-NEXT: Apply patterns for rule infer_variadic_outs after inference:
// CHECK-NEXT: (CodeGenInstructionPattern name:__infer_variadic_outs_apply_0 G_FNEG operands:[<def>GITypeOf<$x>:$tmp, $y])
// CHECK-NEXT: (CodeGenInstructionPattern name:__infer_variadic_outs_apply_1 COPY operands:[<def>$dst, GITypeOf<$x>:$tmp])
def infer_variadic_outs: GICombineRule <
  (defs root:$dst),
  (match  (G_UNMERGE_VALUES $x, $y, $vec),
          (G_FNEG $dst, $x)),
  (apply (G_FNEG $tmp, $y),
         (COPY $dst, $tmp))
>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  inference_mul_by_neg_one,
  infer_complex_tempreg,
  infer_variadic_outs
]>;
