// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner-matchtable \
// RUN:     -gicombiner-stop-after-parse -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def dummy;

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}
def MOV : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def TRUNC : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def ZEXT : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def SEXT : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;

def HasAnswerToEverything : Predicate<"Subtarget->getAnswerToUniverse() == 42 && Subtarget->getAnswerToLife() == 42">;
def reg_matchinfo : GIDefMatchData<"Register">;

// CHECK:      (CombineRule name:WipOpcodeTest0 id:0 root:d
// CHECK-NEXT:   (MatchDatas <empty>)
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <root>d:(AnyOpcodePattern [TRUNC])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     __anon_pat_apply_0_0:(CXXPattern apply code:"APPLY")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable <empty>)
// CHECK-NEXT: )
def WipOpcodeTest0 : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode TRUNC):$d),
  (apply [{ APPLY }])>;

// CHECK:     (CombineRule name:WipOpcodeTest1 id:1 root:d
// CHECK-NEXT:   (MatchDatas <empty>)
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <root>d:(AnyOpcodePattern [TRUNC, SEXT])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     __anon_pat_apply_1_0:(CXXPattern apply code:"APPLY")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable <empty>)
// CHECK-NEXT: )
def WipOpcodeTest1 : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode TRUNC, SEXT):$d),
  (apply [{ APPLY }])>;

// CHECK:     (CombineRule name:InstTest0 id:2 root:d
// CHECK-NEXT:   (MatchDatas <empty>)
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <root>d:(InstructionPattern inst:MOV operands:[<def>a, b])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     __anon_pat_apply_2_0:(CXXPattern apply code:"APPLY")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable
// CHECK-NEXT:     [a match_pat:d]
// CHECK-NEXT:     [b live-in]
// CHECK-NEXT:   )
// CHECK-NEXT: )
def InstTest0 : GICombineRule<
  (defs root:$d),
  (match (MOV $a, $b):$d),
  (apply [{ APPLY }])>;

// CHECK:      (CombineRule name:InstTest1 id:3 root:d
// CHECK-NEXT:   (MatchDatas
// CHECK-NEXT:      (MatchDataInfo pattern_symbol:r0 type:'Register' var_name:MDInfo0)
// CHECK-NEXT:   )
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <root>d:(InstructionPattern inst:MOV operands:[<def>a, b])
// CHECK-NEXT:     __anon_pat_match_3_0:(InstructionPattern inst:ZEXT operands:[<def>x, a])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     __anon_pat_apply_3_1:(CXXPattern apply code:"APPLY")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable
// CHECK-NEXT:     [a match_pat:d]
// CHECK-NEXT:     [b live-in]
// CHECK-NEXT:     [x match_pat:__anon_pat_match_3_0]
// CHECK-NEXT:   )
// CHECK-NEXT: )
let Predicates = [HasAnswerToEverything] in
def InstTest1 : GICombineRule<
  (defs root:$d, reg_matchinfo:$r0),
  (match (MOV $a, $b):$d,
         (ZEXT $x, $a)),
  (apply [{ APPLY }])>;

def MyCombiner: GICombinerHelper<"GenMyCombiner", [
  WipOpcodeTest0,
  WipOpcodeTest1,
  InstTest0,
  InstTest1
]>;
