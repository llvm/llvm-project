// RUN: llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %S %s -o - | FileCheck %s

include "Common/RegClassByHwModeCommon.td"

// Define features
def Feature64Bit : SubtargetFeature<"64bit", "Is64Bit", "true", "64-bit mode">;
def FeatureCap : SubtargetFeature<"cap", "IsCap", "true", "Capability mode">;

// Define predicates
def Is64Bit : Predicate<"Subtarget->is64Bit()">,
              AssemblerPredicate<(all_of Feature64Bit)>;
def Not64Bit : Predicate<"!Subtarget->is64Bit()">,
               AssemblerPredicate<(all_of (not Feature64Bit))>;
def IsCap : Predicate<"Subtarget->isCap()">,
            AssemblerPredicate<(all_of FeatureCap)>;
def NotCap : Predicate<"!Subtarget->isCap()">,
             AssemblerPredicate<(all_of (not FeatureCap))>;

// Define 4 modes (similar to RISC-V RV32I, RV64I, RV32Y, RV64Y)
def Mode32I : HwMode<[Not64Bit, NotCap]>;
def Mode64I : HwMode<[Is64Bit, NotCap]>;
def Mode32C : HwMode<[Not64Bit, IsCap]>;
def Mode64C : HwMode<[Is64Bit, IsCap]>;

// Define mode-dependent register class (SP-like)
// Resolves to XRegs in Integer modes, and YRegs in Capability modes
def PtrRC : RegClassByHwMode<[Mode32I, Mode64I, Mode32C, Mode64C],
                             [XRegs,   XRegs,   YRegs,   YRegs]>;

def X_MOV : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins XRegs:$src);
  let AsmString = "x_mov $dst, $src";
  let opcode = 0;
}

def Y_MOV : TestInstruction {
  let OutOperandList = (outs YRegs:$dst);
  let InOperandList = (ins YRegs:$src);
  let AsmString = "y_mov $dst, $src";
  let opcode = 5;
}

def PTR_MOV_SMALL : TestInstruction {
  let OutOperandList = (outs PtrRC:$dst);
  let InOperandList = (ins PtrRC:$src);
  let AsmString = "ptr_mov.small $dst, $src";
  let opcode = 4;
  let Size = 1;
}

// Test partial resolution with 4 modes.
// NotCap only resolves the 'Type' dimension (to Integer), leaving 32-bit and 64-bit modes active.
// Both active modes (Mode32I, Mode64I) resolve PtrRC to XRegs, so it should resolve successfully.
let Predicates = [NotCap] in
def : CompressPat<(X_MOV XRegs:$dst, XRegs:$src),
                  (PTR_MOV_SMALL XRegs:$dst, XRegs:$src)>;

// IsCap only resolves the 'Type' dimension (to Capability), leaving 32-bit and 64-bit modes active.
// Both active modes (Mode32C, Mode64C) resolve PtrRC to YRegs, so it should resolve successfully.
let Predicates = [IsCap] in
def : CompressPat<(Y_MOV YRegs:$dst, YRegs:$src),
                  (PTR_MOV_SMALL YRegs:$dst, YRegs:$src)>;

// CHECK:     static bool compressInst(MCInst &OutInst,
// CHECK-NEXT:                          const MCInst &MI,
// CHECK-NEXT:                          const MCSubtargetInfo &STI) {
// CHECK-NEXT:   {{\[\[}}maybe_unused]] unsigned HwModeId = STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo);
// CHECK-NEXT:   switch (MI.getOpcode()) {
// CHECK-NEXT:   default: return false;
// CHECK-NEXT:   case MyTarget::X_MOV: {
// CHECK-NEXT:     if (!STI.getFeatureBits()[MyTarget::FeatureCap] &&
// CHECK-NEXT:         MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // ptr_mov.small $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::PTR_MOV_SMALL);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case X_MOV
// CHECK-NEXT:   case MyTarget::Y_MOV: {
// CHECK-NEXT:     if (STI.getFeatureBits()[MyTarget::FeatureCap] &&
// CHECK-NEXT:         MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::YRegsRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::YRegsRegClassID].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // ptr_mov.small $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::PTR_MOV_SMALL);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case Y_MOV
// CHECK-NEXT:   } // switch
// CHECK-NEXT:   return false;
// CHECK-NEXT: }

def MyTargetISA : InstrInfo;
def MyTargetAsmWriter : AsmWriter {
  let PassSubtarget = 1;
}
def MyTarget : Target {
  let InstructionSet = MyTargetISA;
  let AssemblyWriters = [MyTargetAsmWriter];
}
