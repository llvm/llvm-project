// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -gicombiner-stop-after-parse -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def dummy;

def HasAnswerToEverything : Predicate<"Subtarget->getAnswerToUniverse() == 42 && Subtarget->getAnswerToLife() == 42">;
def reg_matchinfo : GIDefMatchData<"Register">;

// CHECK:      (CombineRule name:WipOpcodeTest0 id:0 root:d
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>d:(AnyOpcodePattern [G_TRUNC])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     __WipOpcodeTest0_apply_0:(CXXPattern apply code:"APPLY")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats <empty>)
// CHECK-NEXT:   (OperandTable ApplyPats <empty>)
// CHECK-NEXT: )
def WipOpcodeTest0 : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode G_TRUNC):$d),
  (apply [{ APPLY }])>;

// CHECK:     (CombineRule name:WipOpcodeTest1 id:1 root:d
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>d:(AnyOpcodePattern [G_TRUNC, G_SEXT])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     __WipOpcodeTest1_apply_0:(CXXPattern apply code:"APPLY")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats <empty>)
// CHECK-NEXT:   (OperandTable ApplyPats <empty>)
// CHECK-NEXT: )
def WipOpcodeTest1 : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode G_TRUNC, G_SEXT):$d),
  (apply [{ APPLY }])>;

// CHECK:     (CombineRule name:InstTest0 id:2 root:d
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>d:(CodeGenInstructionPattern COPY operands:[<def>$a, $b])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     __InstTest0_apply_0:(CXXPattern apply code:"APPLY")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     a -> d
// CHECK-NEXT:     b -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats <empty>)
// CHECK-NEXT: )
def InstTest0 : GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b):$d),
  (apply [{ APPLY }])>;

// CHECK:      (CombineRule name:InstTest1 id:3 root:d
// CHECK-NEXT:   (MatchDatas
// CHECK-NEXT:      (MatchDataInfo pattern_symbol:r0 type:'Register' var_name:MDInfo0)
// CHECK-NEXT:   )
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>d:(CodeGenInstructionPattern COPY operands:[<def>$a, i32:$b])
// CHECK-NEXT:     __InstTest1_match_1:(CodeGenInstructionPattern G_ZEXT operands:[<def>$x, 0])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     __InstTest1_apply_0:(CXXPattern apply code:"APPLY")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     a -> d
// CHECK-NEXT:     b -> <live-in>
// CHECK-NEXT:     x -> __InstTest1_match_1
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats <empty>)
// CHECK-NEXT: )
let Predicates = [HasAnswerToEverything] in
def InstTest1 : GICombineRule<
  (defs root:$d, reg_matchinfo:$r0),
  (match (COPY $a, i32:$b):$d,
         (G_ZEXT $x, 0)),
  (apply [{ APPLY }])>;

// CHECK:      (CombineRule name:InstTest2 id:4 root:d
// CHECK-NEXT:   (MatchDatas
// CHECK-NEXT:     (MatchDataInfo pattern_symbol:r0 type:'Register' var_name:MDInfo0)
// CHECK-NEXT:   )
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__InstTest2_match_0:(CodeGenInstructionPattern COPY operands:[<def>$d, (i32 0):$x])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     __InstTest2_apply_0:(CXXPattern apply code:"APPLY")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     d -> __InstTest2_match_0
// CHECK-NEXT:     x -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats <empty>)
// CHECK-NEXT: )
def InstTest2 : GICombineRule<
  (defs root:$d, reg_matchinfo:$r0),
  (match (COPY $d, (i32 0):$x)),
  (apply [{ APPLY }])>;

// CHECK:      (CombineRule name:InOutInstTest0 id:5 root:dst
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__InOutInstTest0_match_0:(CodeGenInstructionPattern COPY operands:[<def>$dst, $tmp])
// CHECK-NEXT:     __InOutInstTest0_match_1:(CodeGenInstructionPattern G_ZEXT operands:[<def>$tmp, $src])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__InOutInstTest0_apply_0:(CodeGenInstructionPattern G_TRUNC operands:[<def>$dst, $src])
// CHECK-NEXT:     __InOutInstTest0_apply_1:(CXXPattern apply code:"APPLY ${src}")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     dst -> __InOutInstTest0_match_0
// CHECK-NEXT:     src -> <live-in>
// CHECK-NEXT:     tmp -> __InOutInstTest0_match_1
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     dst -> __InOutInstTest0_apply_0
// CHECK-NEXT:     src -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT: )
def InOutInstTest0 : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $tmp),
         (G_ZEXT $tmp, $src)),
  (apply (G_TRUNC $dst, $src), "APPLY ${src}")>;

def MatchICst: GICombinePatFrag<
    (outs),
    (ins gi_mo:$foo, gi_imm:$cst),
    [(pattern "return matchIConstant(${foo}, ${cst})")]>;

// CHECK:      (CombineRule name:PatFragTest0 id:6 root:dst
// CHECK-NEXT:   (PatFrags
// CHECK-NEXT:     (PatFrag name:MatchICst
// CHECK-NEXT:       (ins [foo:machine_operand, cst:imm])
// CHECK-NEXT:       (alternatives [
// CHECK-NEXT:         [
// CHECK-NEXT:           (CXXPattern name:__MatchICst_alt0_pattern_0 match code:"return matchIConstant(${foo}, ${cst})"),
// CHECK-NEXT:         ],
// CHECK-NEXT:       ])
// CHECK-NEXT:     )
// CHECK-NEXT:   )
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__PatFragTest0_match_0:(CodeGenInstructionPattern G_ZEXT operands:[<def>$dst, $cst])
// CHECK-NEXT:     __PatFragTest0_match_1:(PatFragPattern MatchICst operands:[$cst, (i32 0)])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__PatFragTest0_apply_0:(CodeGenInstructionPattern COPY operands:[<def>$dst, (i32 0)])
// CHECK-NEXT:     __PatFragTest0_apply_1:(CXXPattern apply code:"APPLY ${src}")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     cst -> <live-in>
// CHECK-NEXT:     dst -> __PatFragTest0_match_0
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     dst -> __PatFragTest0_apply_0
// CHECK-NEXT:   )
// CHECK-NEXT: )
def PatFragTest0 : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $cst), (MatchICst $cst, (i32 0))),
  (apply (COPY $dst, (i32 0)), "APPLY ${src}")>;

def MatchFooPerms: GICombinePatFrag<
    (outs),
    (ins gi_mo:$foo, gi_imm:$cst),
    [
      (pattern "return foo(${foo}, ${cst})"),
      (pattern "return bar(${foo}, ${cst})"),
      (pattern "return bux(${foo}, ${cst})"),
    ]>;

// CHECK:      (CombineRule name:PatFragTest1 id:7 root:dst
// CHECK-NEXT:   (PatFrags
// CHECK-NEXT:     (PatFrag name:MatchFooPerms
// CHECK-NEXT:       (ins [foo:machine_operand, cst:imm])
// CHECK-NEXT:       (alternatives [
// CHECK-NEXT:         [
// CHECK-NEXT:           (CXXPattern name:__MatchFooPerms_alt0_pattern_0 match code:"return foo(${foo}, ${cst})"),
// CHECK-NEXT:         ],
// CHECK-NEXT:         [
// CHECK-NEXT:           (CXXPattern name:__MatchFooPerms_alt1_pattern_0 match code:"return bar(${foo}, ${cst})"),
// CHECK-NEXT:         ],
// CHECK-NEXT:         [
// CHECK-NEXT:           (CXXPattern name:__MatchFooPerms_alt2_pattern_0 match code:"return bux(${foo}, ${cst})"),
// CHECK-NEXT:         ],
// CHECK-NEXT:       ])
// CHECK-NEXT:     )
// CHECK-NEXT:   )
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__PatFragTest1_match_0:(CodeGenInstructionPattern G_ZEXT operands:[<def>$dst, $cst])
// CHECK-NEXT:     a:(PatFragPattern MatchFooPerms operands:[$cst, (i32 0)])
// CHECK-NEXT:     b:(PatFragPattern MatchFooPerms operands:[$cst, (i32 0)])
// CHECK-NEXT:     c:(PatFragPattern MatchFooPerms operands:[$cst, (i32 0)])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__PatFragTest1_apply_0:(CodeGenInstructionPattern COPY operands:[<def>$dst, (i32 0)])
// CHECK-NEXT:     __PatFragTest1_apply_1:(CXXPattern apply code:"APPLY ${src}")
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     cst -> <live-in>
// CHECK-NEXT:     dst -> __PatFragTest1_match_0
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     dst -> __PatFragTest1_apply_0
// CHECK-NEXT:   )
// CHECK-NEXT:   (PermutationsToEmit
// CHECK-NEXT:     [a[0], b[0], c[0]],
// CHECK-NEXT:     [a[0], b[0], c[1]],
// CHECK-NEXT:     [a[0], b[0], c[2]],
// CHECK-NEXT:     [a[0], b[1], c[0]],
// CHECK-NEXT:     [a[0], b[1], c[1]],
// CHECK-NEXT:     [a[0], b[1], c[2]],
// CHECK-NEXT:     [a[0], b[2], c[0]],
// CHECK-NEXT:     [a[0], b[2], c[1]],
// CHECK-NEXT:     [a[0], b[2], c[2]],
// CHECK-NEXT:     [a[1], b[0], c[0]],
// CHECK-NEXT:     [a[1], b[0], c[1]],
// CHECK-NEXT:     [a[1], b[0], c[2]],
// CHECK-NEXT:     [a[1], b[1], c[0]],
// CHECK-NEXT:     [a[1], b[1], c[1]],
// CHECK-NEXT:     [a[1], b[1], c[2]],
// CHECK-NEXT:     [a[1], b[2], c[0]],
// CHECK-NEXT:     [a[1], b[2], c[1]],
// CHECK-NEXT:     [a[1], b[2], c[2]],
// CHECK-NEXT:     [a[2], b[0], c[0]],
// CHECK-NEXT:     [a[2], b[0], c[1]],
// CHECK-NEXT:     [a[2], b[0], c[2]],
// CHECK-NEXT:     [a[2], b[1], c[0]],
// CHECK-NEXT:     [a[2], b[1], c[1]],
// CHECK-NEXT:     [a[2], b[1], c[2]],
// CHECK-NEXT:     [a[2], b[2], c[0]],
// CHECK-NEXT:     [a[2], b[2], c[1]],
// CHECK-NEXT:     [a[2], b[2], c[2]],
// CHECK-NEXT:   )
// CHECK-NEXT: )
let MaxPermutations = -1 in
def PatFragTest1 : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $cst),
         (MatchFooPerms $cst, (i32 0)):$a,
         (MatchFooPerms $cst, (i32 0)):$b,
         (MatchFooPerms $cst, (i32 0)):$c
  ),
  (apply (COPY $dst, (i32 0)), "APPLY ${src}")>;

// CHECK:      (CombineRule name:VariadicsInTest id:8 root:dst
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__VariadicsInTest_match_0:(CodeGenInstructionPattern G_BUILD_VECTOR operands:[<def>$dst, $a, $b])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__VariadicsInTest_apply_0:(CodeGenInstructionPattern COPY operands:[<def>$dst, (i32 0)])
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     a -> <live-in>
// CHECK-NEXT:     b -> <live-in>
// CHECK-NEXT:     dst -> __VariadicsInTest_match_0
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     dst -> __VariadicsInTest_apply_0
// CHECK-NEXT:   )
// CHECK-NEXT: )
def VariadicsInTest : GICombineRule<
  (defs root:$dst),
  (match (G_BUILD_VECTOR $dst, $a, $b)),
  (apply (COPY $dst, (i32 0)))>;

// CHECK:      (CombineRule name:VariadicsOutTest id:9 root:a
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__VariadicsOutTest_match_0:(CodeGenInstructionPattern G_UNMERGE_VALUES operands:[<def>$a, <def>$b, $src])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__VariadicsOutTest_apply_0:(CodeGenInstructionPattern COPY operands:[<def>$a, (i32 0)])
// CHECK-NEXT:     <apply_root>__VariadicsOutTest_apply_1:(CodeGenInstructionPattern COPY operands:[<def>$b, (i32 0)])
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     a -> __VariadicsOutTest_match_0
// CHECK-NEXT:     b -> __VariadicsOutTest_match_0
// CHECK-NEXT:     src -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     a -> __VariadicsOutTest_apply_0
// CHECK-NEXT:     b -> __VariadicsOutTest_apply_1
// CHECK-NEXT:   )
// CHECK-NEXT: )
def VariadicsOutTest : GICombineRule<
  (defs root:$a),
  (match (G_UNMERGE_VALUES $a, $b, $src)),
  (apply (COPY $a, (i32 0)),
         (COPY $b, (i32 0)))>;


def MyCombiner: GICombiner<"GenMyCombiner", [
  WipOpcodeTest0,
  WipOpcodeTest1,
  InstTest0,
  InstTest1,
  InstTest2,
  InOutInstTest0,
  PatFragTest0,
  PatFragTest1,
  VariadicsInTest,
  VariadicsOutTest
]>;
