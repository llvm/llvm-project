// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=warning:

include "llvm/Target/Target.td"

def MyTarget : Target;

def WriteA : SchedWrite;
def WriteB : SchedWrite;
def WriteC : SchedWrite;
def ReadB : SchedRead;

let OutOperandList = (outs), InOperandList = (ins) in {
  def Inst_Same : Instruction { let SchedRW = [WriteA]; }
  def Inst_Equal : Instruction { let SchedRW = [WriteA]; }
  def Inst_Slower : Instruction { let SchedRW = [WriteA]; }
  def Inst_Fwd : Instruction { let SchedRW = [WriteB, ReadB]; }
  def Inst_AllA : Instruction { let SchedRW = [WriteA]; }
  def Inst_AllC : Instruction { let SchedRW = [WriteC]; }
  def Inst_PartA : Instruction { let SchedRW = [WriteA]; }
  def Inst_PartB : Instruction { let SchedRW = [WriteB]; }
  def Inst_Off : Instruction { let SchedRW = [WriteA]; }
}

def ModelOn : SchedMachineModel { let CompleteModel = 0; }
def ModelOff : SchedMachineModel {
  let CompleteModel = 0;
  let WarnRedundantInstRW = 0;
}

def UnitOn : ProcResource<1> { let SchedModel = ModelOn; }
def UnitOff : ProcResource<1> { let SchedModel = ModelOff; }

let SchedModel = ModelOn in {
  def : WriteRes<WriteA, [UnitOn]> { let Latency = 3; }
  def : WriteRes<WriteB, [UnitOn]> { let Latency = 3; }
  def : WriteRes<WriteC, [UnitOn]> { let Latency = 3; }
  def : ReadAdvance<ReadB, 1, [WriteB]>;

  def OnEqual : SchedWriteRes<[UnitOn]> { let Latency = 3; }
  def OnSlower : SchedWriteRes<[UnitOn]> { let Latency = 4; }
  def OnFwd : SchedWriteRes<[UnitOn]> { let Latency = 3; }

  def : InstRW<[WriteA], (instrs Inst_Same)>;
// CHECK: [[@LINE-1]]:3: warning: InstRW on ModelOn matches the base SchedRW of all its instructions
  def : InstRW<[OnEqual], (instrs Inst_Equal)>;
// CHECK: [[@LINE-1]]:3: warning: InstRW on ModelOn matches the base SchedRW of all its instructions
  def : InstRW<[OnSlower], (instrs Inst_Slower)>;
  def : InstRW<[OnFwd, ReadB], (instrs Inst_Fwd)>;
  def : InstRW<[OnEqual], (instrs Inst_AllA, Inst_AllC)>;
// CHECK: [[@LINE-1]]:3: warning: InstRW on ModelOn matches the base SchedRW of all its instructions
  def : InstRW<[OnEqual], (instrs Inst_PartA, Inst_PartB)>;
}

let SchedModel = ModelOff in {
  def : WriteRes<WriteA, [UnitOff]>;
  def : WriteRes<WriteB, [UnitOff]>;
  def : WriteRes<WriteC, [UnitOff]>;
  def : ReadAdvance<ReadB, 0>;
  def : InstRW<[WriteA], (instrs Inst_Off)>;
}

def ProcOn : ProcessorModel<"proc-on", ModelOn, []>;
def ProcOff : ProcessorModel<"proc-off", ModelOff, []>;
