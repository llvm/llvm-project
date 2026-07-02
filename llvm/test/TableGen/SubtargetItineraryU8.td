// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s -o - | FileCheck %s
// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %s -DNEGATIVE_CYCLE -o - 2>&1 | FileCheck %s --check-prefix=INVALID-CYCLE
// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %s -DLARGE_CYCLE -o - 2>&1 | FileCheck %s --check-prefix=INVALID-CYCLE
// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %s -DTOO_MANY_BYPASSES -o - 2>&1 | FileCheck %s --check-prefix=TOO-MANY-BYPASSES
// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %s -DTOO_MANY_BYPASSES_WITHOUT_FU -o - 2>&1 | FileCheck %s --check-prefix=TOO-MANY-BYPASSES

include "llvm/Target/Target.td"

def TestTarget : Target;

def TestFU : FuncUnit;
def TestItinClass : InstrItinClass;
def TestInst : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins);
  let Itinerary = TestItinClass;
}

def TestBP0 : Bypass;
def TestBP1 : Bypass;
def TestBP2 : Bypass;
def TestBP3 : Bypass;
def TestBP4 : Bypass;
def TestBP5 : Bypass;
def TestBP6 : Bypass;
def TestBP7 : Bypass;

#ifdef TOO_MANY_BYPASSES
def TestBP8 : Bypass;
def TestItinData
    : InstrItinData<TestItinClass, [InstrStage<1, [TestFU]>], [1],
                    [TestBP8]>;
def TestItineraries
    : ProcessorItineraries<[TestFU],
                           [TestBP0, TestBP1, TestBP2, TestBP3, TestBP4,
                            TestBP5, TestBP6, TestBP7, TestBP8],
                           [TestItinData]>;
#else
#ifdef TOO_MANY_BYPASSES_WITHOUT_FU
def TestBP8 : Bypass;
def TestItinData
    : InstrItinData<TestItinClass, [InstrStage<1, [TestFU]>], [1],
                    [TestBP8]>;
def TestItineraries
    : ProcessorItineraries<[],
                           [TestBP0, TestBP1, TestBP2, TestBP3, TestBP4,
                            TestBP5, TestBP6, TestBP7, TestBP8],
                           [TestItinData]>;
#else
#ifdef NEGATIVE_CYCLE
def TestItinData
    : InstrItinData<TestItinClass, [InstrStage<1, [TestFU]>], [-1],
                    [TestBP7]>;
#else
#ifdef LARGE_CYCLE
def TestItinData
    : InstrItinData<TestItinClass, [InstrStage<1, [TestFU]>], [4294967296],
                    [TestBP7]>;
#else
def TestItinData
    : InstrItinData<TestItinClass, [InstrStage<1, [TestFU]>], [255],
                    [TestBP7]>;
#endif
#endif
def TestItineraries
    : ProcessorItineraries<[TestFU],
                           [TestBP0, TestBP1, TestBP2, TestBP3, TestBP4,
                            TestBP5, TestBP6, TestBP7],
                           [TestItinData]>;
#endif
#endif

def TestProcessor : Processor<"test", TestItineraries, []>;

// CHECK: const uint8_t TestBP7 = 1 << 7;
// CHECK: extern const uint8_t TestTargetOperandCycles[] = {
// CHECK: 255,
// CHECK: static_assert(sizeof(TestTargetOperandCycles) == 3);
// CHECK: static_assert(255 <= static_cast<uint8_t>(-1));
// CHECK: extern const uint8_t TestTargetForwardingPaths[] = {
// CHECK: TestItinerariesBypass::TestBP7,
// CHECK: static_assert(sizeof(TestTargetForwardingPaths) == 3);
// CHECK: static_assert(sizeof(TestTargetForwardingPaths) == sizeof(TestTargetOperandCycles));
// CHECK: static_assert((1U << 7) <= static_cast<uint8_t>(-1));
// CHECK: const uint8_t *OC, const uint8_t *FP)
// CHECK: extern const uint8_t TestTargetOperandCycles[];
// CHECK: extern const uint8_t TestTargetForwardingPaths[];

// INVALID-CYCLE: error: operand cycle must fit in a uint8_t
// TOO-MANY-BYPASSES: error: at most eight itinerary bypasses are supported
