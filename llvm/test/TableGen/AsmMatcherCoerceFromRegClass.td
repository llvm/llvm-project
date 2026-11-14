// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s
// RUN: not llvm-tblgen -gen-asm-matcher -I %p/../../include %s -DERR_DUPLICATE_NAME 2>&1 | FileCheck %s --check-prefix=ERR-DUPLICATE-NAME

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo;
def ArchAsmParser : AsmParser {
  let AllowDuplicateRegisterNames = 1;
}
def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyParsers = [ArchAsmParser];
}

let Namespace = "Arch" in {
  def R0 : Register<"r0">;
  def R1 : Register<"r1">;
  def R0_Y : Register<"r0">;
  def R1_Y : Register<"r1">;

  def V0 : Register<"v0">;
  def V1 : Register<"v1">;
  def V2 : Register<"v2">;
  def V3 : Register<"v3">;
  def V0M2 : Register<"v0">;
  def V2M2 : Register<"v2">;
}

def GPR : RegisterClass<"Arch", [i32], 32, (add R0, R1)>;
/// Contains the exact same {R0_Y, R1_Y, R0, R1} register set that YGPR matches
/// via CoerceFromRegClass, so TableGen deduplicates their match class to
/// MCK_AnyGPR. This tests that InstAnyGPR still uses plain CVT_95_Reg while
/// InstYGPR uses CVT_95_convertToYGPR.
def AnyGPR : RegisterClass<"Arch", [i64], 64, (add R0_Y, R1_Y, R0, R1)>;
def YGPR : RegisterClass<"Arch", [i64], 64, (add R0_Y, R1_Y)> {
  let CoerceFromRegClass = GPR;
}

/// Only the even registers in VR have a counterpart in VRM2.
def VR : RegisterClass<"Arch", [v2i32], 64, (add V0, V1, V2, V3)>;
def VRM2 : RegisterClass<"Arch", [v4i32], 128, (add V0M2, V2M2)> {
  let CoerceFromRegClass = VR;
}

let Namespace = "Arch" in {
def InstAnyGPR : Instruction {
  let OutOperandList = (outs AnyGPR:$rd);
  let InOperandList = (ins GPR:$rs);
  let AsmString = "anyinst $rd, $rs";
}

def InstYGPR : Instruction {
  let OutOperandList = (outs YGPR:$rd);
  let InOperandList = (ins GPR:$rs);
  let AsmString = "yinst $rd, $rs";
}

def InstVRM2 : Instruction {
  let OutOperandList = (outs VRM2:$vd);
  let InOperandList = (ins VR:$vs);
  let AsmString = "vinst $vd, $vs";
}
}

#ifdef ERR_DUPLICATE_NAME
def BadRC : RegisterClass<"Arch", [i64], 64, (add R0_Y, R0)> {
  let CoerceFromRegClass = GPR;
}
// ERR-DUPLICATE-NAME: error: RegisterClass 'BadRC' with CoerceFromRegClass contains multiple registers named 'r0'
#endif

// CHECK-LABEL: static MCRegister convertToVRM2(MCRegister Reg) {
// CHECK-NEXT:    switch (Reg.id()) {
// CHECK-NEXT:    case Arch::V0: return Arch::V0M2;
// CHECK-NEXT:    case Arch::V2: return Arch::V2M2;
// CHECK-NEXT:    default: llvm_unreachable("unexpected register");
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-LABEL: static MCRegister convertToYGPR(MCRegister Reg) {
// CHECK-NEXT:    switch (Reg.id()) {
// CHECK-NEXT:    case Arch::R0: return Arch::R0_Y;
// CHECK-NEXT:    case Arch::R1: return Arch::R1_Y;
// CHECK-NEXT:    default: llvm_unreachable("unexpected register");
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-LABEL: convertToMCInst(
// CHECK:       case CVT_95_Reg:
// CHECK-NEXT:    static_cast<ArchOperand &>(*Operands[OpIdx]).addRegOperands(Inst, 1);
// CHECK-NEXT:    break;
// CHECK-NEXT:  case CVT_95_convertToVRM2: {
// CHECK-NEXT:    MCRegister Reg = Operands[OpIdx]->getReg();
// CHECK-NEXT:    if (!getArchMCRegisterClass(Arch::VRM2RegClassID).contains(Reg))
// CHECK-NEXT:      Reg = convertToVRM2(Reg);
// CHECK-NEXT:    Inst.addOperand(MCOperand::createReg(Reg));
// CHECK-NEXT:    break;
// CHECK-NEXT:  }
// CHECK-NEXT:  case CVT_95_convertToYGPR: {
// CHECK-NEXT:    MCRegister Reg = Operands[OpIdx]->getReg();
// CHECK-NEXT:    if (!getArchMCRegisterClass(Arch::YGPRRegClassID).contains(Reg))
// CHECK-NEXT:      Reg = convertToYGPR(Reg);
// CHECK-NEXT:    Inst.addOperand(MCOperand::createReg(Reg));
// CHECK-NEXT:    break;
// CHECK-NEXT:  }

/// Coerced YGPR and AnyGPR share MCK_AnyGPR because they match the same set of
/// registers, whereas VRM2 gets its own MCK_VRM2 ({V0M2, V2M2, V0, V2}).
// CHECK-LABEL: enum MatchClassKind {
// CHECK:         MCK_GPR, // register class 'GPR'
// CHECK-NEXT:    MCK_AnyGPR, // register class 'AnyGPR,YGPR'
// CHECK-NEXT:    MCK_VR, // register class 'VR'
// CHECK-NEXT:    MCK_VRM2, // register class 'VRM2'

// CHECK-LABEL: MatchEntry MatchTable0[] = {
// CHECK-NEXT:    /* anyinst */, Arch::InstAnyGPR, Convert__Reg1_0__Reg1_1, AMFBS_None, { MCK_AnyGPR, MCK_GPR },
// CHECK-NEXT:    /* vinst */, Arch::InstVRM2, Convert__VRM21_0__Reg1_1, AMFBS_None, { MCK_VRM2, MCK_VR },
// CHECK-NEXT:    /* yinst */, Arch::InstYGPR, Convert__YGPR1_0__Reg1_1, AMFBS_None, { MCK_AnyGPR, MCK_GPR },
