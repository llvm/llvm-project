// RUN: llvm-tblgen -gen-dfa-packetizer -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestTarget : Target;

def TestSchedModel : SchedMachineModel {
  let CompleteModel = 0;
}

def TestProcessor1 : ProcessorModel<"testprocessor1", TestSchedModel, []>;

def FU0 : FuncUnit;
def FU1 : FuncUnit;

def OP0 : InstrItinClass;
def OP1 : InstrItinClass;

def Itin {
  list<InstrItinData> ItinList = [
    InstrItinData<OP0, [InstrStage<1, [FU0]>]>,
    InstrItinData<OP1, [InstrStage<1, [FU1]>]>,
  ];
}

// CHECK: std::map<unsigned, unsigned> TestTargetProcIdToResourceIndexStartMapping = {
// CHECK-NEXT:   { 2,  1 }, // TestItinerariesModel
// CHECK-NEXT: };

def TestItineraries: ProcessorItineraries<[], [], Itin.ItinList>;
def TestProcessor2 : Processor<"testprocessor2", TestItineraries, []>;
