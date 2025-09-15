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
// CHECK-NEXT:   unsigned Mid;
// CHECK-NEXT:   unsigned Start = 0;
// CHECK-NEXT:   unsigned End = 1;
// CHECK-NEXT:   while (Start < End) {
// CHECK-NEXT:   Mid = Start + (End - Start) / 2;
// CHECK-NEXT:   if (ProcID == TestTargetProcIdToProcResourceIdxTable[Mid][0]) {
// CHECK-NEXT:     break;
// CHECK-NEXT:   }
// CHECK-NEXT:   if (ProcID < TestTargetProcIdToProcResourceIdxTable[Mid][0])
// CHECK-NEXT:     End = Mid;
// CHECK-NEXT:   else
// CHECK-NEXT:     Start = Mid + 1;
// CHECK-NEXT:   }
// CHECK-NEXT:   if (Start == End)
// CHECK-NEXT:     return -1; // Didn't find
// CHECK-NEXT:   return TestTargetProcIdToProcResourceIdxTable[Mid][1];
// CHECK-NEXT: }

// CHECK:  unsigned Index = TestTargetGetResourceIndex(IID->SchedModel.ProcID);

def TestItineraries: ProcessorItineraries<[], [], Itin.ItinList>;
def TestProcessor2 : Processor<"testprocessor2", TestItineraries, []>;
