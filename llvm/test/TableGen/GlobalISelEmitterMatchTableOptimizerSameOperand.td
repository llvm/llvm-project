// RUN: llvm-tblgen %s -gen-global-isel -optimize-match-table=true -I %p/../../include -I %p/Common -o - | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def InstTwoOperands : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def InstThreeOperands : I<(outs GPR32:$dst), (ins GPR32:$cond, GPR32:$src,GPR32:$src2), []>;

// Make sure the GIM_CheckIsSameOperand check is not hoisted into the common header group

// CHECK:       GIM_Try, /*On fail goto*//*Label 1*/
// CHECK-NEXT:  GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NOT:   GIM_CheckIsSameOperand
// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 2*/
// CHECK:       GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/3, /*OtherMI*/2, /*OtherOpIdx*/1,
// CHECK:       // Label 2
// CHECK-NEXT:  GIM_Try, /*On fail goto*//*Label 3*/
// CHECK:       GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/3, /*OtherMI*/2, /*OtherOpIdx*/2,
// CHECK:       // Label 1
def : Pat<(i32 (select GPR32:$cond, GPR32:$src1, GPR32:$src2)),
          (InstThreeOperands GPR32:$cond, GPR32:$src1, GPR32:$src2)>;

def : Pat<(i32 (select (i32 (setcc GPR32:$cond, (i32 0), (OtherVT SETEQ))),
                      (i32 (add GPR32:$src1, GPR32:$const)),
                       GPR32:$src1)),
          (InstThreeOperands GPR32:$cond, GPR32:$src1, GPR32:$const)>;
