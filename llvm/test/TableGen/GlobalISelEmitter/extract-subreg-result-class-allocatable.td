// RUN: llvm-tblgen %s -gen-global-isel -I %p/../../../include -I %p/../Common | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def sub_128 : SubRegIndex<128>;

class MyReg<string n, list<Register> subregs = []> : Register<n> {
  let Namespace = "MyTarget";
  let SubRegs = subregs;
}

class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"MyTarget", types, size, registers> {
  let Size = size;
}

def X0 : MyReg<"x0">;
def X1 : MyReg<"x1">;
let SubRegIndices = [sub_128] in {
def Z0 : MyReg<"z0", [X0]>;
def Z1 : MyReg<"z1", [X1]>;
}

// ARegs and VRegs cover the same registers and types. ARegs sorts first
// (same size and member count, name breaks the tie) but is not allocatable,
// so the extracted result must be constrained to VRegs.
def ARegs : MyClass<128, [v4i32, v2i64], (sequence "X%u", 0, 1)> {
  let isAllocatable = 0;
}
def VRegs : MyClass<128, [v4i32, v2i64], (sequence "X%u", 0, 1)>;
def ZRegs : MyClass<512, [v16i32], (sequence "Z%u", 0, 1)>;

def VOP : RegisterOperand<VRegs>;
def ZOP : RegisterOperand<ZRegs>;

def WIDEN : I<(outs ZRegs:$dst), (ins ZOP:$src),
              [(set ZRegs:$dst, (ctpop ZOP:$src))]>;
def : Pat<(v4i32 (ctpop (v4i32 VOP:$src1))),
          (EXTRACT_SUBREG
            (WIDEN (INSERT_SUBREG (v16i32 (IMPLICIT_DEF)), VOP:$src1, sub_128)),
            sub_128)>;

// CHECK: GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, GIMT_Encode2(MyTarget::VRegsRegClassID)
