// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombinerHelper -gicombiner-stop-after-build %s \
// RUN:     -o %t.inc | FileCheck %s
//
// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombinerHelper %s | \
// RUN: FileCheck --check-prefix=CODE %s

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
def ADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def SUB : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def MUL : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def TRUNC : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def SEXT : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def ZEXT : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def ICMP : I<(outs GPR32:$dst), (ins GPR32:$tst, GPR32:$src1, GPR32:$src2), []>;

def HasFoo : Predicate<"Subtarget->hasFoo()">;
def HasAnswerToEverything : Predicate<"Subtarget->getAnswerToUniverse() == 42 && Subtarget->getAnswerToLife() == 42">;

def Rule0 : GICombineRule<
  (defs root:$d),
  (match (MUL $t, $s1, $s2),
         (SUB $d, $t, $s3)),
  (apply [{ APPLY }])>;

def Rule1 : GICombineRule<
  (defs root:$d),
  (match (MOV $s1, $s2),
         (MOV $d, $s1)),
  (apply [{ APPLY }])>;

def Rule2 : GICombineRule<
  (defs root:$d),
  (match (MOV $d, $s)),
  (apply [{ APPLY }])>;

def Rule3 : GICombineRule<
  (defs root:$d),
  (match (MUL $t, $s1, $s2),
         (ADD $d, $t, $s3), [{ A }]),
  (apply [{ APPLY }])>;

def Rule4 : GICombineRule<
  (defs root:$d),
  (match (ADD $d, $s1, $s2)),
  (apply [{ APPLY }])>;

let Predicates = [HasFoo] in
def Rule5 : GICombineRule<
  (defs root:$d),
  (match (SUB $d, $s1, $s2)),
  (apply [{ APPLY }])>;

let Predicates = [HasFoo, HasAnswerToEverything] in
def Rule6 : GICombineRule<
  (defs root:$d),
  (match (SEXT $t, $s1),
         (TRUNC $d, $t)),
  (apply [{ APPLY }])>;

def Rule7 : GICombineRule<
  (defs root:$d),
  (match (ZEXT $t, $s1),
         (TRUNC $d, $t)),
  (apply [{ APPLY }])>;

// Rules 8&9 check that the partitions are formed correctly if
// - there is an edge different from Operand(1) -> Operand(0)
// - more than one leaf is ignored because the leaf does not
//   care about the instruction
// - a single instruction has more operands than all others
// These conditions triggered a crash when emitting the
// resulting source code.
def Rule8 : GICombineRule<
  (defs root:$d),
  (match (ICMP $ic, $cc, $s2, $s3),
         (ZEXT $z, $ic),
         (MUL $d, $t, $z),
         [{ MATCH }]),
  (apply [{ APPLY }])>;

def Rule9 : GICombineRule<
  (defs root:$d),
  (match (MUL $d, $t, $z)),
  (apply [{ APPLY }])>;

def MyCombinerHelper: GICombinerHelper<"GenMyCombinerHelper", [
  Rule0,
  Rule1,
  Rule2,
  Rule3,
  Rule4,
  Rule5,
  Rule6,
  Rule7,
  Rule8,
  Rule9
]>;

// CHECK-LABEL: digraph "matchtree" {
// CHECK-DAG:   Node[[N0:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[0].getOpcode()|5 partitions|Rule0,Rule1,Rule2,Rule3,Rule4,Rule5,Rule6,Rule7,Rule8,Rule9}"]
// CHECK-DAG:   Node[[N1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1] = getVRegDef(MI[0].getOperand(1))|2 partitions|Rule0,Rule5}"]
// CHECK-DAG:   Node[[N2:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1].getOpcode()|2 partitions|Rule0,Rule5}"]
// CHECK-DAG:   Node[[N3:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule0}"]
// CHECK-DAG:   Node[[N4:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule5}"]
// CHECK-DAG:   Node[[N5:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule5}"]
// CHECK-DAG:   Node[[N6:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1] = getVRegDef(MI[0].getOperand(1))|2 partitions|Rule1,Rule2}"]
// CHECK-DAG:   Node[[N7:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1].getOpcode()|2 partitions|Rule1,Rule2}"]
// CHECK-DAG:   Node[[N8:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule1}"]
// CHECK-DAG:   Node[[N9:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule2}"]
// CHECK-DAG:   Node[[N10:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule2}"]
// CHECK-DAG:   Node[[N11:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1] = getVRegDef(MI[0].getOperand(1))|2 partitions|Rule3,Rule4}"]
// CHECK-DAG:   Node[[N12:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1].getOpcode()|2 partitions|Rule3,Rule4}"]
// CHECK-DAG:   Node[[N13:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule3,Rule4}",color=red]
// CHECK-DAG:   Node[[N14:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule4}"]
// CHECK-DAG:   Node[[N15:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule4}"]
// CHECK-DAG:   Node[[N16:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1] = getVRegDef(MI[0].getOperand(1))|1 partitions|Rule6,Rule7}"]
// CHECK-DAG:   Node[[N17:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1].getOpcode()|2 partitions|Rule6,Rule7}"]
// CHECK-DAG:   Node[[N18:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule6}"]
// CHECK-DAG:   Node[[N19:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule7}"]
// CHECK-DAG:   Node[[N20:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1] = getVRegDef(MI[0].getOperand(2))|2 partitions|Rule8,Rule9}"]
// CHECK-DAG:   Node[[N21:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1].getOpcode()|2 partitions|Rule8,Rule9}"]
// CHECK-DAG:   Node[[N22:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[2] = getVRegDef(MI[1].getOperand(1))|1 partitions|Rule8,Rule9}"]
// CHECK-DAG:   Node[[N23:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[2].getOpcode()|2 partitions|Rule8,Rule9}"]
// CHECK-DAG:   Node[[N24:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule8,Rule9}",color=red]
// CHECK-DAG:   Node[[N25:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule9}"]
// CHECK-DAG:   Node[[N26:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule9}"]
// CHECK-DAG:   Node[[N27:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule9}"]

// The most important partitioner is on the first opcode:
// CHECK-DAG:   Node[[N0]] -> Node[[N1]] [label="#0 MyTarget::SUB"]
// CHECK-DAG:   Node[[N0]] -> Node[[N6]] [label="#1 MyTarget::MOV"]
// CHECK-DAG:   Node[[N0]] -> Node[[N11]] [label="#2 MyTarget::ADD"]
// CHECK-DAG:   Node[[N0]] -> Node[[N16]] [label="#3 MyTarget::TRUNC"]
// CHECK-DAG:   Node[[N0]] -> Node[[N20]] [label="#4 MyTarget::MUL"]

// For, MI[0].getOpcode() == SUB, then has to determine whether it has a reg
// operand and follow that link. If it can't then Rule5 is the only choice as
// that rule is not constrained to a reg.
// CHECK-DAG:   Node[[N1]] -> Node[[N2]] [label="#0 true"]
// CHECK-DAG:   Node[[N1]] -> Node[[N5]] [label="#1 false"]

// For, MI[0].getOpcode() == SUB && MI[0].getOperand(1).isReg(), if MI[1] is a
// MUL then it must be either Rule0 or Rule5. Rule0 is fully tested so Rule5 is
// unreachable. If it's not MUL then it must be Rule5.
// CHECK-DAG:   Node[[N2]] -> Node[[N3]] [label="#0 MyTarget::MUL"]
// CHECK-DAG:   Node[[N2]] -> Node[[N4]] [label="#1 * or nullptr"]

// CHECK-DAG:   Node[[N6]] -> Node[[N7]] [label="#0 true"]
// CHECK-DAG:   Node[[N6]] -> Node[[N10]] [label="#1 false"]

// CHECK-DAG:   Node[[N7]] -> Node[[N8]] [label="#0 MyTarget::MOV"]
// CHECK-DAG:   Node[[N7]] -> Node[[N9]] [label="#1 * or nullptr"]

// CHECK-DAG:   Node[[N11]] -> Node[[N12]] [label="#0 true"]
// CHECK-DAG:   Node[[N11]] -> Node[[N15]] [label="#1 false"]

// CHECK-DAG:   Node[[N12]] -> Node[[N13]] [label="#0 MyTarget::MUL"]
// CHECK-DAG:   Node[[N12]] -> Node[[N14]] [label="#1 * or nullptr"]

// CHECK-DAG:   Node[[N16]] -> Node[[N17]] [label="#0 true"]

// CHECK-DAG:   Node[[N17]] -> Node[[N18]] [label="#0 MyTarget::SEXT"]
// CHECK-DAG:   Node[[N17]] -> Node[[N19]] [label="#1 MyTarget::ZEXT"]

// Follow the links for MI[0].getOpcode() == MUL.
// CHECK-DAG:   Node[[N20]] -> Node[[N21]] [label="#0 true"]
// CHECK-DAG:   Node[[N20]] -> Node[[N27]] [label="#1 false"]

// CHECK-DAG:   Node[[N21]] -> Node[[N22]] [label="#0 MyTarget::ZEXT"]
// CHECK-DAG:   Node[[N21]] -> Node[[N26]] [label="#1 * or nullptr"]

// CHECK-DAG:   Node[[N22]] -> Node[[N23]] [label="#0 true"]

// CHECK-DAG:   Node[[N23]] -> Node[[N24]] [label="#0 MyTarget::ICMP"]
// CHECK-DAG:   Node[[N23]] -> Node[[N25]] [label="#1 * or nullptr"]


// CHECK-LABEL: {{^}$}}


// Check the generated source code.

// CODE-LABEL: GenMyCombinerHelper::tryCombineAll

// Check the first partition. The numbers correspond to the labels above.
// CODE:      switch (MIs[0]->getOpcode()) {
// CODE-NEXT:   case MyTarget::SUB: Partition = 0; break;
// CODE-NEXT:   case MyTarget::MOV: Partition = 1; break;
// CODE-NEXT:   case MyTarget::ADD: Partition = 2; break;
// CODE-NEXT:   case MyTarget::TRUNC: Partition = 3; break;
// CODE-NEXT:   case MyTarget::MUL: Partition = 4; break;
// CODE-NEXT: }

// Check that the correct partition is choosen if operand 1 is a register.

// CODE:       if (Partition == 0 /* MyTarget::SUB */) {
// CODE-NEXT:    Partition = -1;
// CODE-NEXT:    if (MIs.size() <= 1) MIs.resize(2);
// CODE-NEXT:    MIs[1] = nullptr;
// CODE-NEXT:    if (MIs[0]->getOperand(1).isReg())
// CODE-NEXT:      MIs[1] = MRI.getVRegDef(MIs[0]->getOperand(1).getReg());
// CODE-NEXT:    if (MIs[1] == nullptr) Partition = 1;
// CODE-NEXT:    if (MIs[1] != nullptr) Partition = 0;


// Check that the MUL opcode is tested.

// CODE:         if (Partition == 0 /* true */) {
// CODE-NEXT:      Partition = -1;
// CODE-NEXT:      switch (MIs[1]->getOpcode()) {
// CODE-NEXT:      case MyTarget::MUL: Partition = 0; break;
// CODE-NEXT:      default: Partition = 1; break;
// CODE-NEXT:      }

// Check that action for MUL is executed.

// CODE:          if (Partition == 0 /* MyTarget::MUL */) {
// CODE-NEXT:       // Leaf name: Rule0
// CODE-NEXT:        // Rule: Rule0
// CODE-NEXT:        if (!RuleConfig->isRuleDisabled(0)) {
// CODE-NEXT:          if (1
// CODE-NEXT:) {
// CODE-NEXT:            LLVM_DEBUG(dbgs() << "Applying rule 'Rule0'\n");
// CODE-NEXT:            APPLY
// CODE-NEXT:            return true;
// CODE-NEXT:          }
// CODE-NEXT:        }
// CODE-NEXT:        llvm_unreachable("Combine rule elision was incorrect");
// CODE-NEXT:        return false;
// CODE-NEXT:     }

// Check that the other rule involving SUB (Rule5) is run otherwise.

// CODE-NEXT:      if (Partition == 1 /* * or nullptr */) {
// CODE-NEXT:        // Leaf name: Rule5
// CODE-NEXT:        // Rule: Rule5
// CODE-NEXT:        if (!RuleConfig->isRuleDisabled(5)) {
// CODE-NEXT:          if (1
// CODE-NEXT:               && (
// CODE-NEXT:               // Predicate: HasFoo
// CODE-NEXT:               Subtarget->hasFoo()
// CODE-NEXT:               )
// CODE-NEXT:) {
// CODE-NEXT:            LLVM_DEBUG(dbgs() << "Applying rule 'Rule5'\n");
// CODE-NEXT:            APPLY
// CODE-NEXT:            return true;
// CODE-NEXT:          }
// CODE-NEXT:        }
// CODE-NEXT:        llvm_unreachable("Combine rule elision was incorrect");
// CODE-NEXT:        return false;
// CODE-NEXT:      }
// CODE-NEXT:    }

// Check that Rule5 is run if operand 1 is not MUL.

// CODE-NEXT:    if (Partition == 1 /* false */) {
// CODE-NEXT:      // Leaf name: Rule5
// CODE-NEXT:      // Rule: Rule5
// CODE-NEXT:      if (!RuleConfig->isRuleDisabled(5)) {
// CODE-NEXT:        if (1
// CODE-NEXT:             && (
// CODE-NEXT:             // Predicate: HasFoo
// CODE-NEXT:             Subtarget->hasFoo()
// CODE-NEXT:             )
// CODE-NEXT:      ) {
// CODE-NEXT:          LLVM_DEBUG(dbgs() << "Applying rule 'Rule5'\n");
// CODE-NEXT:          APPLY
// CODE-NEXT:          return true;
// CODE-NEXT:        }
// CODE-NEXT:      }
// CODE-NEXT:      llvm_unreachable("Combine rule elision was incorrect");
// CODE-NEXT:      return false;
// CODE-NEXT:    }
// CODE-NEXT:  }


// Check multiple predicates are correctly emitted

// CODE:      // Leaf name: Rule6
// CODE-NEXT: // Rule: Rule6
// CODE-NEXT: if (!RuleConfig->isRuleDisabled(6)) {
// CODE-NEXT:   if (1
// CODE-NEXT:       && (
// CODE-NEXT:            // Predicate: HasFoo
// CODE-NEXT:            Subtarget->hasFoo()
// CODE-NEXT:          )
// CODE-NEXT:       && (
// CODE-NEXT:            // Predicate: HasAnswerToEverything
// CODE-NEXT:            Subtarget->getAnswerToUniverse() == 42 && Subtarget->getAnswerToLife() == 42
// CODE-NEXT:          )
// CODE-NEXT:      ) {
// CODE-NEXT:     LLVM_DEBUG(dbgs() << "Applying rule 'Rule6'\n");
// CODE-NEXT:     APPLY
// CODE-NEXT:     return true;
// CODE-NEXT:   }
// CODE-NEXT: }
