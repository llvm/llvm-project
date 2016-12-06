// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s | FileCheck %s

// CHECK: ABCForm_A
// CHECK-NOT: ABCForm_A

//
// include Target.td for InstrMapping class and define minimally required objects
//

include "llvm/Target/Target.td"

class DFVReg<string n> : Register<n> {
  let Namespace = "DFV";
}

def R0 : DFVReg<"r0">;
def DFVRegClass : RegisterClass<"DFV",[i32],0,(add R0)>;
def DFVInstrInfo : InstrInfo;

def DFVTest : Target {
  let InstructionSet = DFVInstrInfo;
}

//
// Define a number of a InstrMappings with repeated ValueCol fields
//

class ABCRel;

def getAFormFromBForm : InstrMapping {
  let FilterClass = "ABCRel";
  let RowFields = ["BaseName"];
  let ColFields = ["ABCForm"];
  let KeyCol = ["B"];
  let ValueCols = [["A"]];
}

def getAFormFromCForm : InstrMapping {
  let FilterClass = "ABCRel";
  let RowFields = ["BaseName"];
  let ColFields = ["ABCForm"];
  let KeyCol = ["C"];
  let ValueCols = [["A"]];
}

def getAFormFromDForm : InstrMapping {
  let FilterClass = "ABCRel";
  let RowFields = ["BaseName"];
  let ColFields = ["ABCForm"];
  let KeyCol = ["D"];
  let ValueCols = [["A"]];
}

def getAFormFromEForm : InstrMapping {
  let FilterClass = "ABCRel";
  let RowFields = ["BaseName"];
  let ColFields = ["ABCForm"];
  let KeyCol = ["E"];
  let ValueCols = [["A"]];
}

class I : Instruction {
  let Namespace = "DFV";
  let OutOperandList = (outs);
  let InOperandList = (ins);

  string BaseName = "";
  string ABCForm = "";
}

class isAForm { string ABCForm = "A"; }
class isBForm { string ABCForm = "B"; }
class isCForm { string ABCForm = "C"; }
class isDForm { string ABCForm = "D"; }
class isEForm { string ABCForm = "E"; }

let BaseName = "0" in {
  def A0 : I, ABCRel, isAForm;
  def B0 : I, ABCRel, isBForm;
  def C0 : I, ABCRel, isCForm;
  def D0 : I, ABCRel, isDForm;
  def E0 : I, ABCRel, isEForm;
}

