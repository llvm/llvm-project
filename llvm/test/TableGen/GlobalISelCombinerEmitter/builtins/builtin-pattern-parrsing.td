// RUN: llvm-tblgen -I %p/../../../../include -gen-global-isel-combiner \
// RUN:     -gicombiner-stop-after-parse -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// CHECK:      (CombineRule name:BuiltinTest0 id:0 root:a
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__BuiltinTest0_match_0:(CodeGenInstructionPattern G_TRUNC operands:[<def>$a, $b])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__BuiltinTest0_apply_0:(BuiltinPattern GIReplaceReg operands:[<def>$a, $b])
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     a -> __BuiltinTest0_match_0
// CHECK-NEXT:     b -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     a -> __BuiltinTest0_apply_0
// CHECK-NEXT:     b -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT: )
def BuiltinTest0 : GICombineRule<
  (defs root:$a),
  (match (G_TRUNC $a, $b)),
  (apply (GIReplaceReg $a, $b))
>;

// CHECK:      (CombineRule name:BuiltinTest1 id:1 root:mi
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>mi:(CodeGenInstructionPattern G_STORE operands:[$a, $b])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     __BuiltinTest1_apply_0:(BuiltinPattern GIEraseRoot operands:[])
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     a -> <live-in>
// CHECK-NEXT:     b -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats <empty>)
// CHECK-NEXT: )
def BuiltinTest1 : GICombineRule<
  (defs root:$mi),
  (match (G_STORE $a, $b):$mi),
  (apply (GIEraseRoot))
>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  BuiltinTest0,
  BuiltinTest1
]>;
