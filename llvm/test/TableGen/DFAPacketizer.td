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

// CHECK:      int TestTargetGetResourceIndex(unsigned ProcID) {
// CHECK-NEXT:   static const unsigned TestTargetProcIdToProcResourceIdxTable[][2] = {
// CHECK-NEXT:     { 2,  1 }, // TestItinerariesModel
// CHECK-NEXT:   };
// CHECK-NEXT:   auto It = llvm::lower_bound(TestTargetProcIdToProcResourceIdxTable, ProcID,
// CHECK-NEXT:       [](const unsigned LHS[], unsigned Val) { return LHS[0] < Val; });
// CHECK-NEXT:   assert(*It[0] == ProcID);
// CHECK-NEXT:   return (*It)[1];
// CHECK-NEXT: }

// CHECK:  unsigned Index = TestTargetGetResourceIndex(IID->SchedModel.ProcID);

def TestItineraries: ProcessorItineraries<[], [], Itin.ItinList>;
def TestProcessor2 : Processor<"testprocessor2", TestItineraries, []>;
