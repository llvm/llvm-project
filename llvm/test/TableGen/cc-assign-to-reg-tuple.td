// RUN: llvm-tblgen --gen-callingconv -I %p/../../include -I %p/Common %s 2>&1 | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 --gen-callingconv -I %p/../../include -I %p/Common %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 --gen-callingconv -I %p/../../include -I %p/Common %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR2 %s

include "reg-with-subregs-common.td"

def CC_ABI1 : CallingConv<[
  // Use singleton definitions directly.
  CCIfType<[i32, f32],
      CCAssignToReg<[R8, R9, R10, R11, R12, R13, R14, R15]>>,

  // Use tuple definitions indirectly as strings.
  CCIfType<[i64, f64],
      CCAssignToRegTuple<["R8_R9", "R10_R11", "R12_R13", "R14_R15"]>>,

  CCIfType<[i128],
      CCAssignToRegTuple<["R8_R9_R10_R11", "R12_R13_R14_R15"]>>,

  CCIfType<[v8i32],
      CCAssignToRegTuple<["R8_R9_R10_R11_R12_R13_R14_R15"]>>,
]>;

// CHECK: if (LocVT == MVT::i32 ||
// CHECK:      LocVT == MVT::f32) {
// CHECK:    static const MCPhysReg RegList1[] = {
// CHECK:      R8, R9, R10, R11, R12, R13, R14, R15
// CHECK:    };
// CHECK:    if (MCRegister Reg = State.AllocateReg(RegList1)) {
// CHECK:      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
// CHECK:      return false;
// CHECK:    }
// CHECK:  }

// CHECK:  if (LocVT == MVT::i64 ||
// CHECK:      LocVT == MVT::f64) {
// CHECK:    static const MCPhysReg RegList2[] = {
// CHECK:      R8_R9, R10_R11, R12_R13, R14_R15
// CHECK:    };
// CHECK:    if (MCRegister Reg = State.AllocateReg(RegList2)) {
// CHECK:      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
// CHECK:      return false;
// CHECK:    }
// CHECK:  }

// CHECK:  if (LocVT == MVT::i128) {
// CHECK:    static const MCPhysReg RegList3[] = {
// CHECK:      R8_R9_R10_R11, R12_R13_R14_R15
// CHECK:    };
// CHECK:    if (MCRegister Reg = State.AllocateReg(RegList3)) {
// CHECK:      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
// CHECK:      return false;
// CHECK:    }
// CHECK:  }

// CHECK:  if (LocVT == MVT::v8i32) {
// CHECK:    if (MCRegister Reg = State.AllocateReg(R8_R9_R10_R11_R12_R13_R14_R15)) {
// CHECK:      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
// CHECK:      return false;
// CHECK:    }
// CHECK:  }

#ifdef ERROR1
def CC_ABI2 : CallingConv<[
  // Test that referencing an undefined tuple is diagnosed as an error.
  // CHECK-ERROR1: error: register not defined: "R89_R33"
  CCIfType<[i64, f64],
      CCAssignToRegTuple<["R89_R33", "R12_R13", "R14_R15"]>>,
]>;
#endif

#ifdef ERROR2
def CC_ABI3 : CallingConv<[
  // Currently an error: Use tuple definitions directly.
  // CHECK-ERROR2: error: Variable not defined: 'R8_R9_R10_R11'
  CCIfType<[i128],
      CCAssignToRegTuple<[R8_R9_R10_R11, R12_R13_R14_R15]>>,
]>;
#endif
