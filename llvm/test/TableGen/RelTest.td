// RUN: not llvm-tblgen -gen-instr-info -I %p/../../include %s 2>&1 | FileCheck %s

// This test verifies that TableGen is displaying an error when mapped instruction
// does not contain a field listed under RowFields.

include "llvm/Target/Target.td"

class SimpleReg<string n> : Register<n> {
  let Namespace = "Simple";
}
def R0 : SimpleReg<"r0">;
def SimpleRegClass : RegisterClass<"Simple",[i32],0,(add R0)>;
def SimpleInstrInfo : InstrInfo;

def SimpleTarget : Target {
  let InstructionSet = SimpleInstrInfo;
}

class SimpleRel;

def REL_DEF : InstrMapping {
  let FilterClass = "SimpleRel";
  let RowFields = ["BaseName"];
  let ColFields = ["Col"];
  let KeyCol = ["KeyCol"];
  let ValueCols = [["ValCol"]];
}

class INSTR_DEF : Instruction {
  let Namespace = "Simple";
  let OutOperandList = (outs);
  let InOperandList = (ins);
  string Basename = "";
  string Col = "";
}

def SimpleInstr : SimpleRel, INSTR_DEF;

// CHECK: error: No value "BaseName" found in "SimpleInstr" instruction description.
// CHECK: def SimpleInstr : SimpleRel, INSTR_DEF;
