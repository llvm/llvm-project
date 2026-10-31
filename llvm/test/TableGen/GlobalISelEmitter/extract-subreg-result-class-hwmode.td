// RUN: llvm-tblgen %s -gen-global-isel -I %p/../../../include -I %p/../Common | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def Has64 : Predicate<"Subtarget->has64()">;
def Mode64 : HwMode<[Has64]>;

def ModeVT : ValueTypeByHwMode<[DefaultMode, Mode64], [i32, i64]>;
def ModeSuperVT : ValueTypeByHwMode<[DefaultMode, Mode64], [v2i32, v2i64]>;
def ModeRI : RegInfoByHwMode<[DefaultMode, Mode64],
                             [RegInfo<32, 32, 32>, RegInfo<64, 64, 64>]>;
def ModeSuperRI : RegInfoByHwMode<[DefaultMode, Mode64],
                                  [RegInfo<64, 64, 64>, RegInfo<128, 128, 128>]>;

def sub_32 : SubRegIndex<32> {
  let SubRegRanges = SubRegRangeByHwMode<[DefaultMode, Mode64],
                                         [SubRegRange<32, 0>, SubRegRange<64, 0>]>;
}

class MyReg<string n, list<Register> subregs = []> : Register<n> {
  let Namespace = "MyTarget";
  let SubRegs = subregs;
}

class MyClass<list<ValueType> types, int size, dag registers>
  : RegisterClass<"MyTarget", types, size, registers> {
  let RegInfos = ModeRI;
}

class MySuperClass<list<ValueType> types, dag registers>
  : RegisterClass<"MyTarget", types, 64, registers> {
  let RegInfos = ModeSuperRI;
}

def X0 : MyReg<"x0">;
def X1 : MyReg<"x1">;
let SubRegIndices = [sub_32] in {
def Z0 : MyReg<"z0", [X0]>;
def Z1 : MyReg<"z1", [X1]>;
}

// ARegs only holds i32, while the extracted value is i32 in the default mode
// and i64 in Mode64. ARegs sorts first (same size and member count, name
// breaks the tie), but its types must not be accepted for the multi-mode
// value: VRegs holds both i32 and i64.
def ARegs : MyClass<[i32], 32, (sequence "X%u", 0, 1)>;
def VRegs : MyClass<[i32, i64], 32, (sequence "X%u", 0, 1)>;
def ZRegs : MySuperClass<[ModeSuperVT], (sequence "Z%u", 0, 1)>;

def VOP : RegisterOperand<VRegs>;
def ZOP : RegisterOperand<ZRegs>;

def WIDEN : I<(outs ZRegs:$dst), (ins ZOP:$src),
              [(set ZRegs:$dst, (ctpop ZOP:$src))]>;
def : Pat<(ModeVT (ctpop (ModeVT VOP:$src1))),
          (EXTRACT_SUBREG
            (WIDEN (INSERT_SUBREG (ModeSuperVT (IMPLICIT_DEF)), VOP:$src1,
                    sub_32)),
            sub_32)>;

// CHECK: GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, GIMT_Encode2(MyTarget::VRegsRegClassID)
