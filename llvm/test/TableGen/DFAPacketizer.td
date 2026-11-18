// RUN: llvm-tblgen -gen-dfa-packetizer -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def TestSchedModel : SchedMachineModel {
  let CompleteModel = 0;
}

def TestProcessor1 : ProcessorModel<"testprocessor1", TestSchedModel, []>;

def FU0 : FuncUnit;
def FU1 : FuncUnit;

def OP0 : InstrItinClass;
def OP1 : InstrItinClass;
def UniqueFUClass : InstrItinClass;
def SharedFUClass : InstrItinClass;

class TestInst<InstrItinClass Itin> : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins);
  let AsmString = "";
  let Itinerary = Itin;
}

def INST_UNIQUE : TestInst<UniqueFUClass>;
def INST_SHARED : TestInst<SharedFUClass>;

def Itin {
  list<InstrItinData> ItinList = [
    InstrItinData<OP0, [InstrStage<1, [FU0]>]>,
    InstrItinData<OP1, [InstrStage<1, [FU1]>]>,
  ];
}

def FU_A : FuncUnit;
def FU_B : FuncUnit;
def FU_C : FuncUnit;
def FU_D : FuncUnit;
def UNIQUE_FU : FuncUnit;
def SHARED_FU : FuncUnit;

def WideFUList {
  list<InstrItinData> ItinList = [
    InstrItinData<UniqueFUClass, [InstrStage<1, [UNIQUE_FU]>]>,
    InstrItinData<SharedFUClass, [InstrStage<1, [SHARED_FU]>]>
  ];
}

def NarrowFUList {
  list<InstrItinData> ItinList = [
    InstrItinData<UniqueFUClass, [InstrStage<1, [SHARED_FU]>]>,
    InstrItinData<SharedFUClass, [InstrStage<1, [SHARED_FU]>]>
  ];
}

// CHECK:      int TestTargetGetResourceIndex(unsigned ProcID) {
// CHECK-NEXT:   static const unsigned TestTargetProcIdToProcResourceIdxTable[][2] = {
// CHECK-NEXT:     { {{[0-9]+}},  1 }, // TestNarrowItinerariesModel
// CHECK-NEXT:     { {{[0-9]+}},  2 }, // TestWideItinerariesModel
// CHECK-NEXT:     { {{[0-9]+}},  3 }, // TestItinerariesModel
// CHECK-NEXT:   };
// CHECK-NEXT:   auto It = llvm::lower_bound(TestTargetProcIdToProcResourceIdxTable, ProcID,
// CHECK-NEXT:       [](const unsigned LHS[], unsigned Val) { return LHS[0] < Val; });
// CHECK-NEXT:   assert(*It[0] == ProcID);
// CHECK-NEXT:   return (*It)[1];
// CHECK-NEXT: }

// Check that duplicate functional unit names keep their first assigned bit.
// If `SHARED_FU` was reassigned while processing `TestNarrowItineraries`, it
// would collide with `UNIQUE_FU` and the generated automaton would not contain
// the distinct 0x10, 0x20, and 0x30 resource masks below.
// CHECK: const std::array<NfaStatePair, 8> TestTargetTransitionInfo
// CHECK:   /* 0 */ {0, 16}, {0, 0},
// CHECK-NEXT:   /* 2 */ {0, 32}, {0, 0},
// CHECK:   /* 4 */ {16, 48}, {0, 0},
// CHECK:   /* 6 */ {32, 48}, {0, 0},

// CHECK:  unsigned Index = TestTargetGetResourceIndex(IID->SchedModel.ProcID);

def TestItineraries: ProcessorItineraries<[], [], Itin.ItinList>;
def TestProcessor2 : Processor<"testprocessor2", TestItineraries, []>;

def TestWideItineraries :
    ProcessorItineraries<[FU_A, FU_B, FU_C, FU_D, UNIQUE_FU, SHARED_FU],
                         [], WideFUList.ItinList>;
def TestWideProcessor : Processor<"test-wide", TestWideItineraries, []>;

def TestNarrowItineraries :
    ProcessorItineraries<[FU_A, FU_B, FU_C, FU_D, SHARED_FU],
                         [], NarrowFUList.ItinList>;
def TestNarrowProcessor : Processor<"test-narrow", TestNarrowItineraries, []>;
