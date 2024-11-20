// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -gicombiner-stop-after-parse -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// CHECK:      (CombineRule name:InstTest0 id:0 root:x
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     __InstTest0_match_0:(CodeGenInstructionPattern G_BUILD_VECTOR operands:[<def>i64:$z, i32:$y, i16:$w])
// CHECK-NEXT:     <match_root>__InstTest0_match_1:(CodeGenInstructionPattern G_BUILD_VECTOR operands:[<def>i64:$x, i32:$y, i64:$z])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__InstTest0_apply_0:(CodeGenInstructionPattern G_MUL operands:[<def>i64:$x, i32:$y, i16:$w])
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     w -> <live-in>
// CHECK-NEXT:     x -> __InstTest0_match_1
// CHECK-NEXT:     y -> <live-in>
// CHECK-NEXT:     z -> __InstTest0_match_0
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     w -> <live-in>
// CHECK-NEXT:     x -> __InstTest0_apply_0
// CHECK-NEXT:     y -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT: )
def InstTest0 : GICombineRule<
  (defs root:$x),
  (match
      (G_BUILD_VECTOR i64:$z, $y, $w),
      (G_BUILD_VECTOR $x, i32:$y, $z)),
  (apply (G_MUL i64:$x, $y, i16:$w))>;

// CHECK:      (CombineRule name:PatFragTest0 id:1 root:dst
// CHECK-NEXT:   (PatFrags
// CHECK-NEXT:     (PatFrag name:FooPF
// CHECK-NEXT:       (outs [cst:root])
// CHECK-NEXT:       (alternatives [
// CHECK-NEXT:         [
// CHECK-NEXT:           (CodeGenInstructionPattern name:__FooPF_alt0_pattern_0 G_BUILD_VECTOR operands:[<def>i64:$cst, i8:$y, $x]),
// CHECK-NEXT:           (CodeGenInstructionPattern name:__FooPF_alt0_pattern_1 G_BUILD_VECTOR operands:[<def>$x, i8:$y, $w]),
// CHECK-NEXT:         ],
// CHECK-NEXT:         [
// CHECK-NEXT:           (CodeGenInstructionPattern name:__FooPF_alt1_pattern_0 G_BUILD_VECTOR operands:[<def>i32:$cst, i32:$y, i16:$x]),
// CHECK-NEXT:           (CodeGenInstructionPattern name:__FooPF_alt1_pattern_1 G_BUILD_VECTOR operands:[<def>i16:$x, i32:$y, $w]),
// CHECK-NEXT:         ],
// CHECK-NEXT:       ])
// CHECK-NEXT:     )
// CHECK-NEXT:   )
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__PatFragTest0_match_0:(PatFragPattern FooPF operands:[<def>$dst])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__PatFragTest0_apply_0:(CodeGenInstructionPattern COPY operands:[<def>$dst, (i32 0)])
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     dst -> __PatFragTest0_match_0
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     dst -> __PatFragTest0_apply_0
// CHECK-NEXT:   )
// CHECK-NEXT:   (PermutationsToEmit
// CHECK-NEXT:     [__PatFragTest0_match_0[0]],
// CHECK-NEXT:     [__PatFragTest0_match_0[1]],
// CHECK-NEXT:   )
// CHECK-NEXT: )
def FooPF: GICombinePatFrag<
    (outs root:$cst),(ins),
    [
      (pattern (G_BUILD_VECTOR i64:$cst, $y, $x), (G_BUILD_VECTOR $x, i8:$y, $w)),
      (pattern (G_BUILD_VECTOR i32:$cst, $y, i16:$x), (G_BUILD_VECTOR $x, i32:$y, $w)),
    ]>;
def PatFragTest0 : GICombineRule<
  (defs root:$dst),
  (match (FooPF $dst)),
  (apply (COPY $dst, (i32 0)))>;


// CHECK:      (CombineRule name:TypeOfProp id:2 root:x
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__TypeOfProp_match_0:(CodeGenInstructionPattern G_ZEXT operands:[<def>$x, $y])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__TypeOfProp_apply_0:(CodeGenInstructionPattern G_ANYEXT operands:[<def>$x, GITypeOf<$y>:$tmp])
// CHECK-NEXT:     __TypeOfProp_apply_1:(CodeGenInstructionPattern G_ANYEXT operands:[<def>GITypeOf<$y>:$tmp, $y])
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     x -> __TypeOfProp_match_0
// CHECK-NEXT:     y -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     tmp -> __TypeOfProp_apply_1
// CHECK-NEXT:     x -> __TypeOfProp_apply_0
// CHECK-NEXT:     y -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT: )
def TypeOfProp : GICombineRule<
  (defs root:$x),
  (match (G_ZEXT $x, $y)),
  (apply (G_ANYEXT $x, GITypeOf<"$y">:$tmp),
         (G_ANYEXT $tmp, $y))>;

// CHECK:      (CombineRule name:VariadicTypeTest id:3 root:a
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__VariadicTypeTest_match_0:(CodeGenInstructionPattern G_UNMERGE_VALUES operands:[<def>$a, <def>$b, GIVariadic<1,0>:$z])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__VariadicTypeTest_apply_0:(CodeGenInstructionPattern G_UNMERGE_VALUES operands:[<def>$a, <def>$b, GIVariadic<1,0>:$z])
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     a -> __VariadicTypeTest_match_0
// CHECK-NEXT:     b -> __VariadicTypeTest_match_0
// CHECK-NEXT:     z -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     a -> __VariadicTypeTest_apply_0
// CHECK-NEXT:     b -> __VariadicTypeTest_apply_0
// CHECK-NEXT:     z -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT: )
def VariadicTypeTest: GICombineRule<
  (defs root:$a),
  (match (G_UNMERGE_VALUES $a, $b, GIVariadic<>:$z)),
  (apply (G_UNMERGE_VALUES $a, $b, $z))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  InstTest0,
  PatFragTest0,
  TypeOfProp,
  VariadicTypeTest,
]>;
