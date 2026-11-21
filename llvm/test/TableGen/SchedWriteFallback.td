// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s | FileCheck %s
// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include -DCYCLE %s 2>&1 | FileCheck %s --check-prefix=CYCLE
// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include -DWRITERES %s 2>&1 | FileCheck %s --check-prefix=WRITERES
// RUN: llvm-tblgen -gen-subtarget -I %p/../../include -DWARN %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:

include "llvm/Target/Target.td"

def MyTarget : Target;

def WriteBase : SchedWrite;
def WriteK : SchedWriteFallback<WriteBase>;
def WriteKZ : SchedWriteFallback<WriteK>;
def WriteZ : SchedWriteFallback<UnsupportedWrite>;
def WriteVarBase : SchedWrite;
def WriteVarK : SchedWriteFallback<WriteVarBase>;

let OutOperandList = (outs), InOperandList = (ins) in {
  def Inst_Base : Instruction { let SchedRW = [WriteBase]; }
  def Inst_K : Instruction { let SchedRW = [WriteK]; }
  def Inst_KZ : Instruction { let SchedRW = [WriteKZ]; }
  def Inst_Z : Instruction { let SchedRW = [WriteZ]; }
  def Inst_VarK : Instruction { let SchedRW = [WriteVarK]; }
}

def ModelA : SchedMachineModel { let CompleteModel = 0; }
def ModelB : SchedMachineModel { let CompleteModel = 0; }
def ModelC : SchedMachineModel { let CompleteModel = 0; }

def UnitA : ProcResource<1> { let SchedModel = ModelA; }
def UnitB : ProcResource<1> { let SchedModel = ModelB; }
def UnitC : ProcResource<1> { let SchedModel = ModelC; }

let SchedModel = ModelA in {
  def : WriteRes<WriteBase, [UnitA]> { let Latency = 2; }
  def : WriteRes<WriteVarBase, [UnitA]> { let Latency = 2; }
}

let SchedModel = ModelB in {
  def : WriteRes<WriteBase, [UnitB]> { let Latency = 2; }
  def : WriteRes<WriteK, [UnitB]> { let Latency = 5; }
  def : WriteRes<WriteZ, [UnitB]> { let Latency = 7; }
  def : WriteRes<WriteVarBase, [UnitB]> { let Latency = 2; }
}

def CPred : MCSchedPredicate<TruePred>;

let SchedModel = ModelC in {
  def : WriteRes<WriteBase, [UnitC]> { let Latency = 2; }
  def CFast : SchedWriteRes<[UnitC]> { let Latency = 1; }
  def CSlow : SchedWriteRes<[UnitC]> { let Latency = 9; }
  def CVar : SchedWriteVariant<[SchedVar<CPred, [CFast]>,
                                SchedVar<NoSchedPred, [CSlow]>]>;
  def : SchedAlias<WriteVarBase, CVar>;
}

def ProcA : ProcessorModel<"proc-a", ModelA, []>;
def ProcB : ProcessorModel<"proc-b", ModelB, []>;
def ProcC : ProcessorModel<"proc-c", ModelC, []>;

// CHECK-LABEL: MyTargetWriteLatencyTable[] = {
// CHECK-NEXT: { 0, 0}, // Invalid
// CHECK-NEXT: { 2, 0}, // #1 WriteBase_WriteVarBase
// CHECK-NEXT: { 5, 0}, // #2 WriteK
// CHECK-NEXT: { 7, 0}, // #3 WriteZ
// CHECK-NEXT: { 1, 0}, // #4 CFast
// CHECK-NEXT: { 9, 0} // #5 CSlow

// CHECK-LABEL: ModelASchedClasses[] = {
// CHECK: /*WriteBase*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 1, 0, 1, 1}, // #1
// CHECK-NEXT: /*WriteK*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 1, 0, 1, 1}, // #2
// CHECK-NEXT: /*WriteKZ*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 1, 0, 1, 1}, // #3
// CHECK-NEXT: /*WriteVarK*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 1, 0, 1, 1}, // #4
// CHECK-NEXT: /*WriteZ*/ {{[0-9]+}}) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #5

// CHECK-LABEL: ModelBSchedClasses[] = {
// CHECK: /*WriteBase*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 1, 0, 1, 1}, // #1
// CHECK-NEXT: /*WriteK*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 2, 0, 1, 1}, // #2
// CHECK-NEXT: /*WriteKZ*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 2, 0, 1, 1}, // #3
// CHECK-NEXT: /*WriteVarK*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 1, 0, 1, 1}, // #4
// CHECK-NEXT: /*WriteZ*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 3, 0, 1, 1}, // #5

// CHECK-LABEL: ModelCSchedClasses[] = {
// CHECK: /*WriteBase*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 1, 0, 1, 1}, // #1
// CHECK-NEXT: /*WriteK*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 1, 0, 1, 1}, // #2
// CHECK-NEXT: /*WriteKZ*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 1, 0, 1, 1}, // #3
// CHECK-NEXT: /*WriteVarK*/ {{[0-9]+}}) 8190, false, false, false, 0, 0, 0, 0, 0, 0}, // #4
// CHECK-NEXT: /*WriteZ*/ {{[0-9]+}}) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #5
// CHECK-NEXT: /*CFast*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 4, 0, 1, 1}, // #6
// CHECK-NEXT: /*CSlow*/ {{[0-9]+}}) 1, false, false, false, 0, 1, 5, 0, 1, 1}, // #7

// CHECK: case 4: // WriteVarK
// CHECK-NEXT: if (CPUID == 3) { // ModelC

#ifdef CYCLE
def WriteLoop : SchedWriteFallback<NoWrite> {
  let Fallback = !cast<SchedWrite>("WriteLoop");
}
let OutOperandList = (outs), InOperandList = (ins) in
def Inst_Loop : Instruction { let SchedRW = [WriteLoop]; }
// CYCLE: [[@LINE-5]]:5: error: Fallback cycle through WriteLoop
#endif

#ifdef WARN
def WriteKF : SchedWriteFallback<WriteBase>;
def ReadF : SchedRead;
let OutOperandList = (outs), InOperandList = (ins) in
def Inst_KF : Instruction { let SchedRW = [WriteKF, ReadF]; }
foreach M = [ModelA, ModelB, ModelC] in
let SchedModel = M in def : ReadAdvance<ReadF, 0>;

def ModelD : SchedMachineModel { let CompleteModel = 0; }
def ModelE : SchedMachineModel {
  let CompleteModel = 0;
  let WarnRedundantWriteRes = 0;
}

def UnitD : ProcResource<1> { let SchedModel = ModelD; }
def UnitE : ProcResource<1> { let SchedModel = ModelE; }

let SchedModel = ModelD in {
  def : WriteRes<WriteBase, [UnitD]> { let Latency = 2; }
  def : WriteRes<WriteK, [UnitD]> { let Latency = 2; }
// WARN: [[@LINE-1]]:3: warning: WriteRes for WriteK on ModelD matches its fallback
  def : WriteRes<WriteKZ, [UnitD]> { let Latency = 3; }
  def : WriteRes<WriteKF, [UnitD]> { let Latency = 2; }
  def : ReadAdvance<ReadF, 1, [WriteKF]>;
  def : WriteRes<WriteVarK, [UnitD]> { let Latency = 2; }
  let Unsupported = true in
  def : WriteRes<WriteZ, []>;
// WARN: [[@LINE-1]]:3: warning: WriteRes for WriteZ on ModelD matches its fallback
}

let SchedModel = ModelE in {
  def : WriteRes<WriteBase, [UnitE]> { let Latency = 2; }
  def : WriteRes<WriteK, [UnitE]> { let Latency = 2; }
  def : WriteRes<WriteVarBase, [UnitE]> { let Latency = 2; }
  def : ReadAdvance<ReadF, 0>;
}

def ProcD : ProcessorModel<"proc-d", ModelD, []>;
def ProcE : ProcessorModel<"proc-e", ModelE, []>;
#endif

#ifdef WRITERES
def BadRes : SchedWriteRes<[]>, SchedWriteFallback<WriteBase>;
// WRITERES: [[@LINE-1]]:5: error: SchedWriteRes, SchedWriteVariant and WriteSequence cannot have a fallback
let OutOperandList = (outs), InOperandList = (ins) in
def Inst_BadRes : Instruction { let SchedRW = [BadRes]; }
#endif
