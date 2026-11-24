// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s -o - | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;
def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def HasFeat : Predicate<"Subtarget->hasFeat()">;
def TestMode : HwMode<[HasFeat]>;

def TestVT : ValueTypeByHwMode<[DefaultMode, TestMode], [i16, i32]>;

def X0 : Register<"x0">;
def XRegs : RegisterClass<"TestTarget", [f32, TestVT], 32, (add X0)>;

def FNEG : Instruction {
  let Namespace = "TestTarget";
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins XRegs:$src);
}

def : Pat<(f32 (fneg XRegs:$src)), (FNEG XRegs:$src)>;

// CHECK: // Src: (fneg:{ *:[f32] m1:[f32] } XRegs:{ *:[f32] m1:[f32] }:$src)
