// RUN: llvm-tblgen -I %S/Inputs -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -gicombiner-stop-after-parse -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

include "test-intrinsics.td"

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
// CHECK-NEXT:      (MatchDataDef symbol:r0 type:Register)
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
// CHECK-NEXT:     (MatchDataDef symbol:r0 type:Register)
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
  (apply (G_TRUNC $dst, $src))>;

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
  (apply (COPY $dst, (i32 0)))>;

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
  (apply (COPY $dst, (i32 0)))>;

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

// CHECK:      (CombineRule name:TypeOfTest id:10 root:dst
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__TypeOfTest_match_0:(CodeGenInstructionPattern COPY operands:[<def>$dst, $tmp])
// CHECK-NEXT:     __TypeOfTest_match_1:(CodeGenInstructionPattern G_ZEXT operands:[<def>$tmp, $src])
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__TypeOfTest_apply_0:(CodeGenInstructionPattern G_MUL operands:[<def>$dst, (GITypeOf<$src> 0), (GITypeOf<$dst> -1)])
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     dst -> __TypeOfTest_match_0
// CHECK-NEXT:     src -> <live-in>
// CHECK-NEXT:     tmp -> __TypeOfTest_match_1
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     dst -> __TypeOfTest_apply_0
// CHECK-NEXT:   )
// CHECK-NEXT: )
def TypeOfTest : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $tmp),
         (G_ZEXT $tmp, $src)),
  (apply (G_MUL $dst, (GITypeOf<"$src"> 0), (GITypeOf<"$dst"> -1)))>;


// CHECK:      (CombineRule name:MIFlagsTest id:11 root:dst
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>mi:(CodeGenInstructionPattern G_ZEXT operands:[<def>$dst, $src] (MIFlags (set MachineInstr::FmReassoc) (unset MachineInstr::FmNoNans, MachineInstr::FmArcp)))
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__MIFlagsTest_apply_0:(CodeGenInstructionPattern G_MUL operands:[<def>$dst, $src, $src] (MIFlags (set MachineInstr::FmReassoc) (unset MachineInstr::FmNsz, MachineInstr::FmArcp) (copy mi)))
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     dst -> mi
// CHECK-NEXT:     src -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     dst -> __MIFlagsTest_apply_0
// CHECK-NEXT:     src -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT: )
def MIFlagsTest : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $src, (MIFlags FmReassoc, (not FmNoNans, FmArcp))):$mi),
  (apply (G_MUL $dst, $src, $src, (MIFlags $mi, FmReassoc, (not FmNsz, FmArcp))))>;

// CHECK-NEXT: (CombineRule name:IntrinTest0 id:12 root:a
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__IntrinTest0_match_0:(CodeGenInstructionPattern G_INTRINSIC operands:[<def>$a, $b] intrinsic(@llvm.1in.1out))
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__IntrinTest0_apply_0:(CodeGenInstructionPattern G_INTRINSIC_W_SIDE_EFFECTS operands:[<def>$a, $b] intrinsic(@llvm.sideeffects.1in.1out))
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     a -> __IntrinTest0_match_0
// CHECK-NEXT:     b -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     a -> __IntrinTest0_apply_0
// CHECK-NEXT:     b -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT: )
def IntrinTest0 : GICombineRule<
  (defs root:$a),
  (match (int_1in_1out $a, $b)),
  (apply (int_sideeffects_1in_1out $a, $b))>;

// CHECK:      (CombineRule name:IntrinTest1 id:13 root:a
// CHECK-NEXT:   (MatchPats
// CHECK-NEXT:     <match_root>__IntrinTest1_match_0:(CodeGenInstructionPattern G_INTRINSIC_CONVERGENT operands:[<def>$a, $b] intrinsic(@llvm.convergent.1in.1out))
// CHECK-NEXT:   )
// CHECK-NEXT:   (ApplyPats
// CHECK-NEXT:     <apply_root>__IntrinTest1_apply_0:(CodeGenInstructionPattern G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS operands:[<def>$a, $b] intrinsic(@llvm.convergent.sideeffects.1in.1out))
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable MatchPats
// CHECK-NEXT:     a -> __IntrinTest1_match_0
// CHECK-NEXT:     b -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT:   (OperandTable ApplyPats
// CHECK-NEXT:     a -> __IntrinTest1_apply_0
// CHECK-NEXT:     b -> <live-in>
// CHECK-NEXT:   )
// CHECK-NEXT: )
def IntrinTest1 : GICombineRule<
  (defs root:$a),
  (match (int_convergent_1in_1out $a, $b)),
  (apply (int_convergent_sideeffects_1in_1out $a, $b))>;

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
  VariadicsOutTest,
  TypeOfTest,
  MIFlagsTest,
  IntrinTest0,
  IntrinTest1
]>;
