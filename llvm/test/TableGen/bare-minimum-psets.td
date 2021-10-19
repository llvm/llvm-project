// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s

// Test to check the bare minimum pressure sets.
// At least one register pressure set is required for the target.
// Allowed the pset only for D_32 regclass and ignored it for all
// other classes including the tuples.
include "llvm/Target/Target.td"

class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"MyTarget", types, size, registers> {
  let Size = size;
}

def sub0 : SubRegIndex<32>;
def sub1 : SubRegIndex<32, 32>;

let Namespace = "MyTarget" in {
  def D : Register<"d">;
  foreach Index = 0-7 in {
    def S#Index : Register <"s"#Index>;
  }
}

// Should generate psets for D_32
def D_32 : MyClass<32,  [i32], (add D)>;

let GeneratePressureSet = 0 in {
  def S_32 : MyClass<32,  [i32], (sequence "S%u", 0, 7)>;
  def SD_32 : MyClass<32,  [i32], (add S_32, D_32)>;
}

def S_64 : RegisterTuples<[sub0, sub1],
                           [(decimate (shl S_32, 0), 1),
                            (decimate (shl S_32, 1), 1)
                           ]>;

def SReg_64 : MyClass<64, [i64], (add S_64)> {
  let GeneratePressureSet = 0;
}

def MyTarget : Target;

// CHECK-LABEL:    // Register pressure sets enum.
// CHECK-NEXT: namespace MyTarget {
// CHECK-NEXT: enum RegisterPressureSets {
// CHECK-NEXT: D_32 = 0,
// CHECK-NEXT:  };
// NAMESPACE-NEXT: } // end namespace TestNamespace

// CHECK-LABEL: getRegPressureSetName(unsigned Idx) const {
// CHECK-NEXT:    static const char *const PressureNameTable[] = {
// CHECK-NEXT:      "D_32",
// CHECK-NEXT:    };
// CHECK-NEXT:    return PressureNameTable[Idx];
// CHECK-NEXT:  }

// CHECK:      unsigned MyTargetGenRegisterInfo::
// CHECK-NEXT: getRegPressureSetLimit(const MachineFunction &MF, unsigned Idx) const {
// CHECK-NEXT:  static const uint8_t PressureLimitTable[] = {
// CHECK-NEXT:    {{[0-9]+}},        // 0: D_32
// CHECK-NEXT:  };
// CHECK-NEXT:  return PressureLimitTable[Idx];
// CHECK-NEXT:}

// CHECK:      static const int RCSetsTable[] = {
// CHECK-NEXT:   /* 0 */ 0, -1,
// CHECK-NEXT: };
