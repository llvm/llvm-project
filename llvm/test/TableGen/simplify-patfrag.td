// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s 2>&1 | FileCheck %s
// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include -DIGNORE %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

// Minimal Target definition
def DemoInstrInfo : InstrInfo;
def Demo : Target {
  let InstructionSet = DemoInstrInfo;
}

// Some registers which can hold ints or floats
foreach i = 0...7 in
  def "R" # i: Register<"r" # i>;
def GPR : RegisterClass<"Demo", [i32, f32], 32, (sequence "R%u", 0, 7)>;

// Instruction to convert an int to a float
def i2f : Instruction {
  let Size = 2;
  let OutOperandList = (outs GPR:$dst);
  let InOperandList = (ins GPR:$src);
  let AsmString = "i2f $dst, $src";
}

// Some kind of special type-conversion node supported by this target
def specialconvert : SDNode<"TEST_TARGET_ISD::SPECIAL_CONVERT", SDTUnaryOp>;

// A PatFrags that matches either bitconvert or the special version
def anyconvert : PatFrags<(ops node:$src),
                          [(bitconvert node:$src),
                           (specialconvert node:$src)]>;

#ifdef IGNORE
// Ensure ShouldIgnore does not disable records in dag isel emitter
let GISelShouldIgnore = 1 in
#endif
// And a rule that matches that PatFrag and turns it into i2f
def : Pat<(f32 (anyconvert (i32 GPR:$val))), (i2f GPR:$val)>;

// CHECK: SwitchOpcode{{.*}}ISD::BITCAST
// CHECK: MorphNodeTo1{{.*}}i2f
// CHECK: SwitchOpcode{{.*}}TEST_TARGET_ISD::SPECIAL_CONVERT
// CHECK: MorphNodeTo1{{.*}}i2f
