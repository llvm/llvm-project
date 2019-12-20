// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner -gicombiner-stop-after-parse %s \
// RUN:     -o /dev/null -debug 2>&1 | FileCheck %s
// REQUIRES: asserts

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
def MOV2 : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;

def trivial : GICombineRule<
  (defs root:$d),
  (match (MOV $d, $s)),
  (apply [{ APPLY }])>;

// CHECK-LABEL: Parsed rule defs/match for 'trivial'

// The matchdag block is a fairly direct dump of the information that was read.
// It's oriented towards the data structures within tablegen.
// CHECK-NEXT: matchdag {
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon0_0 // $d=getOperand(0), $s=getOperand(1)
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred0_1
// CHECK-NEXT:     __anon0_0 ==> __anonpred0_1[mi]
// CHECK-NEXT: {{^}$}}

// The digraph block is a human-oriented dump of the information that was read.
// Run it through graphviz to produce a nice DAG showing the matcher behaviour.
// CHECK-NEXT: digraph "trivial" {
// CHECK-NEXT:   rankdir="BT"
// CHECK-NEXT:   Node[[N1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon0_0|MOV|Match starts here|$d=getOperand(0), $s=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}",color=red]
// CHECK-NEXT:   Pred[[P1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred0_1|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Node[[N1]]:e -> Pred[[P1]]:d1:s [style=dotted]
// CHECK-NEXT: {{^}$}}

def simple : GICombineRule<
  (defs root:$d),
  (match (MOV $t, $s),
         (MOV $d, $t)),
  (apply [{ APPLY }])>;

// CHECK-LABEL: Parsed rule defs/match for 'simple'

// CHECK-NEXT: matchdag {
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon1_0 // $t=getOperand(0), $s=getOperand(1)
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon1_2 // $d=getOperand(0), $t=getOperand(1)
// CHECK-NEXT:     __anon1_2[src1] --[t]--> __anon1_0[dst]
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred1_1
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred1_3
// CHECK-NEXT:     __anon1_0 ==> __anonpred1_1[mi]
// CHECK-NEXT:     __anon1_2 ==> __anonpred1_3[mi]
// CHECK-NEXT: {{^}$}}

// CHECK-NEXT: digraph "simple" {
// CHECK-NEXT:   rankdir="BT"
// CHECK-NEXT:   Node[[N1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon1_0|MOV|$t=getOperand(0), $s=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}"]
// CHECK-NEXT:   Node[[N2:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon1_2|MOV|Match starts here|$d=getOperand(0), $t=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}",color=red]
// CHECK-NEXT:   Node[[N2]]:s1:n -> Node[[N1]]:d0:s [label="$t"]
// CHECK-NEXT:   Pred[[P1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred1_1|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Pred[[P2:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred1_3|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Node[[N1]]:e -> Pred[[P1]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N2]]:e -> Pred[[P2]]:d1:s [style=dotted]
// CHECK-NEXT: {{^}$}}

def multiroot : GICombineRule<
  (defs root:$d1, root:$d2),
  (match (MOV $s, $s2),
         (MOV $d1, $s),
         (MOV $d2, $s)),
  (apply [{ APPLY }])>;

// CHECK-LABEL: Parsed rule defs/match for 'multiroot'

// CHECK-NEXT: matchdag {
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon2_0 // $s=getOperand(0), $s2=getOperand(1)
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon2_2 // $d1=getOperand(0), $s=getOperand(1)
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon2_4 // $d2=getOperand(0), $s=getOperand(1)
// CHECK-NEXT:     __anon2_2[src1] --[s]--> __anon2_0[dst]
// CHECK-NEXT:     __anon2_4[src1] --[s]--> __anon2_0[dst]
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred2_1
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred2_3
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred2_5
// CHECK-NEXT:     <<$mi0 == $mi1>>:$__anonpred2_6
// CHECK-NEXT:     __anon2_0 ==> __anonpred2_1[mi]
// CHECK-NEXT:     __anon2_2 ==> __anonpred2_3[mi]
// CHECK-NEXT:     __anon2_4 ==> __anonpred2_5[mi]
// CHECK-NEXT:     __anon2_2[src1] ==> __anonpred2_6[mi0]
// CHECK-NEXT:     __anon2_4[src1] ==> __anonpred2_6[mi1]
// CHECK-NEXT: {{^}$}}

// CHECK-NEXT: digraph "multiroot" {
// CHECK-NEXT:   rankdir="BT"
// CHECK-NEXT:   Node[[N1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon2_0|MOV|$s=getOperand(0), $s2=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}"]
// CHECK-NEXT:   Node[[N2:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon2_2|MOV|Match starts here|$d1=getOperand(0), $s=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}",color=red]
// CHECK-NEXT:   Node[[N3:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon2_4|MOV|Match starts here|$d2=getOperand(0), $s=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}",color=red]
// CHECK-NEXT:   Node[[N2]]:s1:n -> Node[[N1]]:d0:s [label="$s"]
// CHECK-NEXT:   Node[[N3]]:s1:n -> Node[[N1]]:d0:s [label="$s"]
// CHECK-NEXT:   Pred[[P1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred2_1|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Pred[[P2:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred2_3|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Pred[[P3:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred2_5|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Pred[[P4:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi0|<s2>#2 $mi1}|__anonpred2_6|$mi0 == $mi1|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi0|<d2>#2 $mi1}}",style=dotted]
// CHECK-NEXT:   Node[[N1]]:e -> Pred[[P1]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N2]]:e -> Pred[[P2]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N3]]:e -> Pred[[P3]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N2]]:s1:n -> Pred[[P4]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N3]]:s1:n -> Pred[[P4]]:d2:s [style=dotted]
// CHECK-NEXT: {{^}$}}

def nonstandardroot : GICombineRule<
  (defs root:$s),
  (match (MOV $s, $s2),
         (MOV $d1, $s),
         (MOV $d2, $s)),
  (apply [{ APPLY }])>;

// CHECK-LABEL: Parsed rule defs/match for 'nonstandardroot'

// CHECK-NEXT: matchdag {
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon3_0 // $s=getOperand(0), $s2=getOperand(1)
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon3_2 // $d1=getOperand(0), $s=getOperand(1)
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon3_4 // $d2=getOperand(0), $s=getOperand(1)
// CHECK-NEXT:     __anon3_0[dst] --[s]--> __anon3_2[src1]
// CHECK-NEXT:     __anon3_0[dst] --[s]--> __anon3_4[src1]
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred3_1
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred3_3
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred3_5
// CHECK-NEXT:     <<$mi0 == $mi1>>:$__anonpred3_6
// CHECK-NEXT:     __anon3_0 ==> __anonpred3_1[mi]
// CHECK-NEXT:     __anon3_2 ==> __anonpred3_3[mi]
// CHECK-NEXT:     __anon3_4 ==> __anonpred3_5[mi]
// CHECK-NEXT:     __anon3_2[src1] ==> __anonpred3_6[mi0]
// CHECK-NEXT:     __anon3_4[src1] ==> __anonpred3_6[mi1]
// CHECK-NEXT: {{^}$}}

// CHECK-NEXT: digraph "nonstandardroot" {
// CHECK-NEXT:   rankdir="BT"
// CHECK-NEXT:   Node[[N1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon3_0|MOV|Match starts here|$s=getOperand(0), $s2=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}",color=red]
// CHECK-NEXT:   Node[[N2:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon3_2|MOV|$d1=getOperand(0), $s=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}"]
// CHECK-NEXT:   Node[[N3:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon3_4|MOV|$d2=getOperand(0), $s=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}"]
// CHECK-NEXT:   Node[[N2]]:s1:n -> Node[[N1]]:d0:s [label="$s",dir=back,arrowtail=crow]
// CHECK-NEXT:   Node[[N3]]:s1:n -> Node[[N1]]:d0:s [label="$s",dir=back,arrowtail=crow]
// CHECK-NEXT:   Pred[[P1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred3_1|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Pred[[P2:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred3_3|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Pred[[P3:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred3_5|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Pred[[P4:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi0|<s2>#2 $mi1}|__anonpred3_6|$mi0 == $mi1|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi0|<d2>#2 $mi1}}",style=dotted]
// CHECK-NEXT:   Node[[N1]]:e -> Pred[[P1]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N2]]:e -> Pred[[P2]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N3]]:e -> Pred[[P3]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N2]]:s1:n -> Pred[[P4]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N3]]:s1:n -> Pred[[P4]]:d2:s [style=dotted]
// CHECK-NEXT: {{^}$}}

def multiref_use : GICombineRule<
  (defs root:$d1, root:$d2),
  (match (MOV $d1, $s),
         (MOV $d2, $s)),
  (apply [{ APPLY }])>;

// CHECK-LABEL: Parsed rule defs/match for 'multiref_use'

// CHECK-NEXT: matchdag {
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon4_0 // $d1=getOperand(0), $s=getOperand(1)
// CHECK-NEXT:     (MOV 0:dst<def>, 1:src1):$__anon4_2 // $d2=getOperand(0), $s=getOperand(1)
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred4_1
// CHECK-NEXT:     <<$mi.getOpcode() == MOV>>:$__anonpred4_3
// CHECK-NEXT:     <<$mi0 == $mi1>>:$__anonpred4_4
// CHECK-NEXT:     __anon4_0 ==> __anonpred4_1[mi]
// CHECK-NEXT:     __anon4_2 ==> __anonpred4_3[mi]
// CHECK-NEXT:     __anon4_0[src1] ==> __anonpred4_4[mi0]
// CHECK-NEXT:     __anon4_2[src1] ==> __anonpred4_4[mi1]
// CHECK-NEXT: {{^}$}}

// CHECK-NEXT: digraph "multiref_use" {
// CHECK-NEXT:   rankdir="BT"
// CHECK-NEXT:   Node[[N1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon4_0|MOV|Match starts here|$d1=getOperand(0), $s=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}",color=red]
// CHECK-NEXT:   Node[[N2:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $dst|<s1>#1 $src1}|__anon4_2|MOV|Match starts here|$d2=getOperand(0), $s=getOperand(1)|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $dst|<d1>#1 $src1}}",color=red]
// CHECK-NEXT:   Pred[[P1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred4_1|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Pred[[P2:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi}|__anonpred4_3|$mi.getOpcode() == MOV|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi}}",style=dotted]
// CHECK-NEXT:   Pred[[P3:(0x)?[0-9a-fA-F]+]] [shape=record,label="{{[{]{}}<s0>#0 $$|<s1>#1 $mi0|<s2>#2 $mi1}|__anonpred4_4|$mi0 == $mi1|{{(0x)?[0-9a-fA-F]+}}|{<d0>#0 $$|<d1>#1 $mi0|<d2>#2 $mi1}}",style=dotted]
// CHECK-NEXT:   Node[[N1]]:e -> Pred[[P1]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N2]]:e -> Pred[[P2]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N1]]:s1:n -> Pred[[P3]]:d1:s [style=dotted]
// CHECK-NEXT:   Node[[N2]]:s1:n -> Pred[[P3]]:d2:s [style=dotted]
// CHECK-NEXT: {{^}$}}

def MyCombiner: GICombinerHelper<"GenMyCombiner", [
  trivial,
  simple,
  multiroot,
  nonstandardroot,
  multiref_use
]>;

// Verify we're sharing operand lists correctly
// CHECK-LABEL: GIMatchDagOperandListContext {
// CHECK-NEXT:    OperandLists {
// CHECK-NEXT:      0:dst<def>, 1:src1{{$}}
// CHECK-NEXT:      0:$<def>, 1:mi{{$}}
// CHECK-NEXT:      0:$<def>, 1:mi0, 2:mi1{{$}}
// CHECK-NEXT:    }
// CHECK-NEXT:  }
