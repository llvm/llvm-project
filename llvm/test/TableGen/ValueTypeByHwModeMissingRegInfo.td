// RUN: not llvm-tblgen -gen-asm-matcher -I %p/../../include %s -o - 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def Is32Bit : Predicate<"!Subtarget->is64Bit()">;
def Is64Bit : Predicate<"Subtarget->is64Bit()">;
defvar Ptr32 = DefaultMode;
def Ptr64 : HwMode<[Is64Bit]>;

class MyReg<string n> : Register<n> {
  let Namespace = "MyTarget";
}

def X0 : MyReg<"x0">;
def X1 : MyReg<"x1">;
def X2 : MyReg<"x2">;
def X3 : MyReg<"x3">;

def XLenVT : ValueTypeByHwMode<[Ptr32, Ptr64], [i32, i64]>;
def XLenRI : RegInfoByHwMode<[Ptr32,             Ptr64],
                             [RegInfo<32,32,32>, RegInfo<64,64,64>]>;

def XRegs : RegisterClass<"MyTarget", [XLenVT], 32, (add X0, X1, X2, X3)> {
  // Note: Would need this to determine size, otherwise we get an error.
  // let RegInfos = XLenRI;
}
// CHECK: [[#@LINE-4]]:5: error: Impossible to determine register size

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }
