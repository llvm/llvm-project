// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s
// RUN: not llvm-tblgen -gen-asm-matcher -I %p/../../include %s -DERR_DUPLICATE_NAME 2>&1 | FileCheck %s --check-prefix=ERR-DUPLICATE-NAME
// RUN: not llvm-tblgen -gen-asm-matcher -I %p/../../include %s -DERR_SHADOWED 2>&1 | FileCheck %s --check-prefix=ERR-SHADOWED

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
  def V4 : Register<"v4">;
  def V5 : Register<"v5">;
  def V6 : Register<"v6">;
  def V7 : Register<"v7">;
  // Like RISC-V, use PositionOrder to keep each LMUL group contiguous in enum
  // order.
  let PositionOrder = 1 in {
    def V0M2 : Register<"v0">;
    def V2M2 : Register<"v2">;
    def V4M2 : Register<"v4">;
    def V6M2 : Register<"v6">;
  }
  let PositionOrder = 2 in {
    def V0M4 : Register<"v0">;
    def V4M4 : Register<"v4">;
  }

  def W0 : Register<"w0">;
  def W1 : Register<"w1">;
  def W2 : Register<"w2">;
  // WP2 is not part of WPR, so the WPR registers are not evenly spaced.
  let PositionOrder = 3 in {
    def WP0 : Register<"w0">;
    def WP1 : Register<"w1">;
    def WP2 : Register<"wp2">;
    def WP3 : Register<"w2">;
  }

  def U0 : Register<"u0">;
  def U1 : Register<"u1">;
  def U2 : Register<"u2">;
  def U3 : Register<"u3">;
  def U4 : Register<"u4">;
  // Only every third register is part of UPR.
  let PositionOrder = 4 in {
    def UP0 : Register<"u0">;
    def UP1 : Register<"up1">;
    def UP2 : Register<"up2">;
    def UP3 : Register<"u2">;
    def UP4 : Register<"up4">;
    def UP5 : Register<"up5">;
    def UP6 : Register<"u4">;
  }
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

/// Only the even/4-aligned registers in VR have counterparts in VRM2/VRM4, so
/// the conversions use strided arithmetic.
def VR : RegisterClass<"Arch", [v2i32], 64,
                       (add V0, V1, V2, V3, V4, V5, V6, V7)>;
def VRM2 : RegisterClass<"Arch", [v4i32], 128, (add V0M2, V2M2, V4M2, V6M2)> {
  let CoerceFromRegClass = VR;
}
def VRM4 : RegisterClass<"Arch", [v8i32], 256, (add V0M4, V4M4)> {
  let CoerceFromRegClass = VR;
}

/// The WPR registers are not evenly spaced in enum order, so the conversion
/// falls back to a switch.
def WR : RegisterClass<"Arch", [i32], 32, (add W0, W1, W2)>;
def WPR : RegisterClass<"Arch", [i64], 64, (add WP0, WP1, WP3)> {
  let CoerceFromRegClass = WR;
}

/// Only the even UR registers have counterparts in UPR, which are three apart
/// in enum order, so the conversion has to both divide and multiply.
def UR : RegisterClass<"Arch", [i32], 32, (add U0, U1, U2, U3, U4)>;
def UPR : RegisterClass<"Arch", [i64], 64, (add UP0, UP3, UP6)> {
  let CoerceFromRegClass = UR;
}

/// Registers are only converted in the HwModes that select a class with
/// CoerceFromRegClass.
def IsCapMode : Predicate<"Subtarget->hasCapMode()">;
def CapMode : HwMode<[IsCapMode]>;
def PtrRC : RegClassByHwMode<[DefaultMode, CapMode], [GPR, YGPR]>;

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

def InstVRM4 : Instruction {
  let OutOperandList = (outs VRM4:$vd);
  let InOperandList = (ins VR:$vs);
  let AsmString = "v4inst $vd, $vs";
}

def InstWPR : Instruction {
  let OutOperandList = (outs WPR:$wd);
  let InOperandList = (ins WR:$ws);
  let AsmString = "winst $wd, $ws";
}

def InstUPR : Instruction {
  let OutOperandList = (outs UPR:$ud);
  let InOperandList = (ins UR:$us);
  let AsmString = "uinst $ud, $us";
}

def InstPtr : Instruction {
  let OutOperandList = (outs GPR:$rd);
  let InOperandList = (ins PtrRC:$ptr);
  let AsmString = "zinst $rd, $ptr";
}
}

#ifdef ERR_DUPLICATE_NAME
def BadRC : RegisterClass<"Arch", [i64], 64, (add R0_Y, R0)> {
  let CoerceFromRegClass = GPR;
}
// ERR-DUPLICATE-NAME: error: RegisterClass 'BadRC' with CoerceFromRegClass contains multiple registers named 'r0'
#endif

/// MatchRegisterName returns R0 rather than R0_Y for "r0", so R0_Y is never
/// parsed and can't be coerced from.
#ifdef ERR_SHADOWED
def BadRC : RegisterClass<"Arch", [i32], 32, (add R0, R1)> {
  let CoerceFromRegClass = YGPR;
}
// ERR-SHADOWED: error: RegisterClass 'BadRC' cannot coerce from 'R0_Y' since MatchRegisterName returns 'R0' for 'r0'
#endif

// CHECK-LABEL: static MCRegister convertToUPR(MCRegister Reg) {
// CHECK-NEXT:    assert(Reg.id() >= Arch::U0 && Reg.id() <= Arch::U4 &&
// CHECK-NEXT:           (Reg.id() - Arch::U0) % 2 == 0 && "unexpected register");
// CHECK-NEXT:    return Arch::UP0 + (Reg.id() - Arch::U0) / 2 * 3;
// CHECK-NEXT:  }

// CHECK-LABEL: static MCRegister convertToVRM4(MCRegister Reg) {
// CHECK-NEXT:    assert(Reg.id() >= Arch::V0 && Reg.id() <= Arch::V4 &&
// CHECK-NEXT:           (Reg.id() - Arch::V0) % 4 == 0 && "unexpected register");
// CHECK-NEXT:    return Arch::V0M4 + (Reg.id() - Arch::V0) / 4;
// CHECK-NEXT:  }

// CHECK-LABEL: static MCRegister convertToVRM2(MCRegister Reg) {
// CHECK-NEXT:    assert(Reg.id() >= Arch::V0 && Reg.id() <= Arch::V6 &&
// CHECK-NEXT:           (Reg.id() - Arch::V0) % 2 == 0 && "unexpected register");
// CHECK-NEXT:    return Arch::V0M2 + (Reg.id() - Arch::V0) / 2;
// CHECK-NEXT:  }

// CHECK-LABEL: static MCRegister convertToWPR(MCRegister Reg) {
// CHECK-NEXT:    switch (Reg.id()) {
// CHECK-NEXT:    case Arch::W0: return Arch::WP0;
// CHECK-NEXT:    case Arch::W1: return Arch::WP1;
// CHECK-NEXT:    case Arch::W2: return Arch::WP3;
// CHECK-NEXT:    default: llvm_unreachable("unexpected register");
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-LABEL: static MCRegister convertToYGPR(MCRegister Reg) {
// CHECK-NEXT:    assert(Reg.id() >= Arch::R0 && Reg.id() <= Arch::R1 && "unexpected register");
// CHECK-NEXT:    return Arch::R0_Y + (Reg.id() - Arch::R0);
// CHECK-NEXT:  }

// CHECK-LABEL: convertToMCInst(
// CHECK:       case CVT_95_Reg:
// CHECK-NEXT:    static_cast<ArchOperand &>(*Operands[OpIdx]).addRegOperands(Inst, 1);
// CHECK-NEXT:    break;
// CHECK-NEXT:  case CVT_95_convertToUPR: {
// CHECK-NEXT:    MCRegister Reg = Operands[OpIdx]->getReg();
// CHECK-NEXT:    if (!getArchMCRegisterClass(Arch::UPRRegClassID).contains(Reg))
// CHECK-NEXT:      Reg = convertToUPR(Reg);
// CHECK-NEXT:    Inst.addOperand(MCOperand::createReg(Reg));
// CHECK-NEXT:    break;
// CHECK-NEXT:  }
// CHECK-NEXT:  case CVT_95_convertToVRM4: {
// CHECK-NEXT:    MCRegister Reg = Operands[OpIdx]->getReg();
// CHECK-NEXT:    if (!getArchMCRegisterClass(Arch::VRM4RegClassID).contains(Reg))
// CHECK-NEXT:      Reg = convertToVRM4(Reg);
// CHECK-NEXT:    Inst.addOperand(MCOperand::createReg(Reg));
// CHECK-NEXT:    break;
// CHECK-NEXT:  }
// CHECK-NEXT:  case CVT_95_convertToVRM2: {
// CHECK-NEXT:    MCRegister Reg = Operands[OpIdx]->getReg();
// CHECK-NEXT:    if (!getArchMCRegisterClass(Arch::VRM2RegClassID).contains(Reg))
// CHECK-NEXT:      Reg = convertToVRM2(Reg);
// CHECK-NEXT:    Inst.addOperand(MCOperand::createReg(Reg));
// CHECK-NEXT:    break;
// CHECK-NEXT:  }
// CHECK-NEXT:  case CVT_95_convertToWPR: {
// CHECK-NEXT:    MCRegister Reg = Operands[OpIdx]->getReg();
// CHECK-NEXT:    if (!getArchMCRegisterClass(Arch::WPRRegClassID).contains(Reg))
// CHECK-NEXT:      Reg = convertToWPR(Reg);
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
// CHECK-NEXT:  case CVT_95_convertToRegByHwMode_95_PtrRC:
// CHECK-NEXT:    switch (STI->getHwMode(MCSubtargetInfo::HwMode_RegInfo)) {
// CHECK-NEXT:    case 1: { // CapMode
// CHECK-NEXT:      MCRegister Reg = Operands[OpIdx]->getReg();
// CHECK-NEXT:      if (!getArchMCRegisterClass(Arch::YGPRRegClassID).contains(Reg))
// CHECK-NEXT:        Reg = convertToYGPR(Reg);
// CHECK-NEXT:      Inst.addOperand(MCOperand::createReg(Reg));
// CHECK-NEXT:      break;
// CHECK-NEXT:    }
// CHECK-NEXT:    default:
// CHECK-NEXT:      static_cast<ArchOperand &>(*Operands[OpIdx]).addRegOperands(Inst, 1);
// CHECK-NEXT:      break;
// CHECK-NEXT:    }
// CHECK-NEXT:    break;

/// Coerced YGPR and AnyGPR share MCK_AnyGPR because they match the same set of
/// registers, whereas VRM2, VRM4, WPR and UPR get their own match classes.
// CHECK-LABEL: enum MatchClassKind {
// CHECK:         MCK_GPR, // register class 'GPR'
// CHECK-NEXT:    MCK_Reg10, // derived register class
// CHECK-NEXT:    MCK_WR, // register class 'WR'
// CHECK-NEXT:    MCK_Reg3, // derived register class
// CHECK-NEXT:    MCK_AnyGPR, // register class 'AnyGPR,YGPR'
// CHECK-NEXT:    MCK_VRM4, // register class 'VRM4'
// CHECK-NEXT:    MCK_UR, // register class 'UR'
// CHECK-NEXT:    MCK_UPR, // register class 'UPR'
// CHECK-NEXT:    MCK_WPR, // register class 'WPR'
// CHECK-NEXT:    MCK_VR, // register class 'VR'
// CHECK-NEXT:    MCK_VRM2, // register class 'VRM2'

// CHECK-LABEL: MatchEntry MatchTable0[] = {
// CHECK-NEXT:    /* anyinst */, Arch::InstAnyGPR, Convert__Reg1_0__Reg1_1, AMFBS_None, { MCK_AnyGPR, MCK_GPR },
// CHECK-NEXT:    /* uinst */, Arch::InstUPR, Convert__UPR1_0__Reg1_1, AMFBS_None, { MCK_UPR, MCK_UR },
// CHECK-NEXT:    /* v4inst */, Arch::InstVRM4, Convert__VRM41_0__Reg1_1, AMFBS_None, { MCK_VRM4, MCK_VR },
// CHECK-NEXT:    /* vinst */, Arch::InstVRM2, Convert__VRM21_0__Reg1_1, AMFBS_None, { MCK_VRM2, MCK_VR },
// CHECK-NEXT:    /* winst */, Arch::InstWPR, Convert__WPR1_0__Reg1_1, AMFBS_None, { MCK_WPR, MCK_WR },
// CHECK-NEXT:    /* yinst */, Arch::InstYGPR, Convert__YGPR1_0__Reg1_1, AMFBS_None, { MCK_AnyGPR, MCK_GPR },
