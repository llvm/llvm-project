// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s

/* We test the following poset of user-defined asm operand classes:
 *
 *                          Top
 *                         /   \
 *                        /   Right1
 *                       /      |
 *                     Left   Right2
 *                       \      |
 *                        \   Right3
 *                         \   /
 *                        Bottom
 *
 * With this poset, Bottom should be sorted before Right3, but its left-biased
 * depth is only 2 while the left-biased depth of Right3 is 3. Thus with a
 * careless use of left-biased depth ordering, we might end up sorting Bottom
 * after Right3, which will fail EXPENSIVE_CHECKS.
 */

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo;

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def DummyReg : Register<"dummy">;
def DummyRC : RegisterClass<"Arch", [i16], 16, (add DummyReg)>;

class AOC<string N, list<AsmOperandClass> supers = []> : AsmOperandClass {
  let Name = N;
  let SuperClasses = supers;
}

def Top : AOC<"Top">;
def Left : AOC<"Left", [Top]>;
def Right1 : AOC<"Right1", [Top]>;
def Right2 : AOC<"Right2", [Right1]>;
def Right3 : AOC<"Right3", [Right2]>;
def Bottom : AOC<"Bottom", [Left, Right3]>;

// The MatchClassKind enum is emitted in the topsort order, so it's a good proxy:
// CHECK:     enum MatchClassKind {
// CHECK-NOT: };
// CHECK:     MCK_Bottom,
// CHECK-NOT: };
// CHECK:     MCK_Right3,
