//===-- SystemZELFAsmPrinter.cpp - SystemZ ELF asm printer ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemZELFAsmPrinter class.
//
//===----------------------------------------------------------------------===//

#include "SystemZELFAsmPrinter.h"
#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "SystemZConstantPoolValue.h"
#include "SystemZMCInstLower.h"
#include "SystemZSubtarget.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

// Emit the largest nop instruction smaller than or equal to NumBytes
// bytes.  Return the size of nop emitted.
static unsigned EmitNop(MCContext &OutContext, MCStreamer &OutStreamer,
                        unsigned NumBytes, const MCSubtargetInfo &STI) {
  if (NumBytes < 2) {
    llvm_unreachable("Zero nops?");
    return 0;
  } else if (NumBytes < 4) {
    OutStreamer.emitInstruction(
        MCInstBuilder(SystemZ::BCRAsm).addImm(0).addReg(SystemZ::R0D), STI);
    return 2;
  } else if (NumBytes < 6) {
    OutStreamer.emitInstruction(
        MCInstBuilder(SystemZ::BCAsm).addImm(0).addReg(0).addImm(0).addReg(0),
        STI);
    return 4;
  } else {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    const MCSymbolRefExpr *Dot = MCSymbolRefExpr::create(DotSym, OutContext);
    OutStreamer.emitLabel(DotSym);
    OutStreamer.emitInstruction(
        MCInstBuilder(SystemZ::BRCLAsm).addImm(0).addExpr(Dot), STI);
    return 6;
  }
}

static const MCSymbolRefExpr *getTLSGetOffset(MCContext &Context) {
  StringRef Name = "__tls_get_offset";
  return MCSymbolRefExpr::create(Context.getOrCreateSymbol(Name),
                                 SystemZ::S_PLT, Context);
}

static const MCSymbolRefExpr *getGlobalOffsetTable(MCContext &Context) {
  StringRef Name = "_GLOBAL_OFFSET_TABLE_";
  return MCSymbolRefExpr::create(Context.getOrCreateSymbol(Name), Context);
}

SystemZELFAsmPrinter::SystemZELFAsmPrinter(TargetMachine &TM,
                                           std::unique_ptr<MCStreamer> Streamer)
    : SystemZAsmPrinter(TM, std::move(Streamer)) {}

void SystemZELFAsmPrinter::emitInstruction(const MachineInstr *MI) {
  SystemZMCInstLower Lower(MF->getContext(), *this);

  switch (MI->getOpcode()) {
  case TargetOpcode::FENTRY_CALL:
    LowerFENTRY_CALL(*MI, Lower);
    return;
  case TargetOpcode::STACKMAP:
    LowerSTACKMAP(*MI);
    return;
  case TargetOpcode::PATCHPOINT:
    LowerPATCHPOINT(*MI, Lower);
    return;
  case TargetOpcode::PATCHABLE_FUNCTION_ENTER:
    LowerPATCHABLE_FUNCTION_ENTER(*MI, Lower);
    return;
  case TargetOpcode::PATCHABLE_RET:
    LowerPATCHABLE_RET(*MI, Lower);
    return;
  case SystemZ::CallBASR:
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BASR)
                                     .addReg(SystemZ::R14D)
                                     .addReg(MI->getOperand(0).getReg()));
    return;
  case SystemZ::CallBRASL:
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BRASL)
                                     .addReg(SystemZ::R14D)
                                     .addExpr(Lower.getExpr(MI->getOperand(0),
                                                            SystemZ::S_PLT)));
    return;
  case SystemZ::CallJG:
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::JG)
                                     .addExpr(Lower.getExpr(MI->getOperand(0),
                                                            SystemZ::S_PLT)));
    return;
  case SystemZ::CallBRCL:
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BRCL)
                                     .addImm(MI->getOperand(0).getImm())
                                     .addImm(MI->getOperand(1).getImm())
                                     .addExpr(Lower.getExpr(MI->getOperand(2),
                                                            SystemZ::S_PLT)));
    return;
  case SystemZ::TLS_GDCALL:
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BRASL)
                                     .addReg(SystemZ::R14D)
                                     .addExpr(getTLSGetOffset(MF->getContext()))
                                     .addExpr(Lower.getExpr(MI->getOperand(0),
                                                            SystemZ::S_TLSGD)));
    return;
  case SystemZ::TLS_LDCALL:
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(SystemZ::BRASL)
            .addReg(SystemZ::R14D)
            .addExpr(getTLSGetOffset(MF->getContext()))
            .addExpr(Lower.getExpr(MI->getOperand(0), SystemZ::S_TLSLDM)));
    return;
  case SystemZ::GOT:
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(SystemZ::LARL)
                       .addReg(MI->getOperand(0).getReg())
                       .addExpr(getGlobalOffsetTable(MF->getContext())));
    return;
  case SystemZ::LOAD_TLS_BLOCK_ADDR:
    lowerLOAD_TLS_BLOCK_ADDR(*MI, Lower);
    return;
  case SystemZ::LOAD_GLOBAL_STACKGUARD_ADDR:
    lowerLOAD_GLOBAL_STACKGUARD_ADDR(*MI, Lower);
    return;
  default:
    SystemZAsmPrinter::emitInstruction(MI);
    return;
  }
}

void SystemZELFAsmPrinter::LowerFENTRY_CALL(const MachineInstr &MI,
                                            SystemZMCInstLower &Lower) {
  MCContext &Ctx = MF->getContext();
  if (MF->getFunction().hasFnAttribute("mrecord-mcount")) {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    OutStreamer->pushSection();
    OutStreamer->switchSection(
        Ctx.getELFSection("__mcount_loc", ELF::SHT_PROGBITS, ELF::SHF_ALLOC));
    OutStreamer->emitSymbolValue(DotSym, 8);
    OutStreamer->popSection();
    OutStreamer->emitLabel(DotSym);
  }

  if (MF->getFunction().hasFnAttribute("mnop-mcount")) {
    EmitNop(Ctx, *OutStreamer, 6, getSubtargetInfo());
    return;
  }

  MCSymbol *fentry = Ctx.getOrCreateSymbol("__fentry__");
  const MCSymbolRefExpr *Op =
      MCSymbolRefExpr::create(fentry, SystemZ::S_PLT, Ctx);
  OutStreamer->emitInstruction(
      MCInstBuilder(SystemZ::BRASL).addReg(SystemZ::R0D).addExpr(Op),
      getSubtargetInfo());
}

void SystemZELFAsmPrinter::LowerSTACKMAP(const MachineInstr &MI) {
  auto *TII = MF->getSubtarget<SystemZSubtarget>().getInstrInfo();

  unsigned NumNOPBytes = MI.getOperand(1).getImm();

  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);

  SM.recordStackMap(*MILabel, MI);
  assert(NumNOPBytes % 2 == 0 && "Invalid number of NOP bytes requested!");

  // Scan ahead to trim the shadow.
  unsigned ShadowBytes = 0;
  const MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::const_iterator MII(MI);
  ++MII;
  while (ShadowBytes < NumNOPBytes) {
    if (MII == MBB.end() || MII->getOpcode() == TargetOpcode::PATCHPOINT ||
        MII->getOpcode() == TargetOpcode::STACKMAP)
      break;
    ShadowBytes += TII->getInstSizeInBytes(*MII);
    if (MII->isCall())
      break;
    ++MII;
  }

  // Emit nops.
  while (ShadowBytes < NumNOPBytes)
    ShadowBytes += EmitNop(OutContext, *OutStreamer, NumNOPBytes - ShadowBytes,
                           getSubtargetInfo());
}

// Lower a patchpoint of the form:
// [<def>], <id>, <numBytes>, <target>, <numArgs>
void SystemZELFAsmPrinter::LowerPATCHPOINT(const MachineInstr &MI,
                                           SystemZMCInstLower &Lower) {
  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);

  SM.recordPatchPoint(*MILabel, MI);
  PatchPointOpers Opers(&MI);

  unsigned EncodedBytes = 0;
  const MachineOperand &CalleeMO = Opers.getCallTarget();

  if (CalleeMO.isImm()) {
    uint64_t CallTarget = CalleeMO.getImm();
    if (CallTarget) {
      unsigned ScratchIdx = -1;
      unsigned ScratchReg = 0;
      do {
        ScratchIdx = Opers.getNextScratchIdx(ScratchIdx + 1);
        ScratchReg = MI.getOperand(ScratchIdx).getReg();
      } while (ScratchReg == SystemZ::R0D);

      // Materialize the call target address
      EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::LLILF)
                                       .addReg(ScratchReg)
                                       .addImm(CallTarget & 0xFFFFFFFF));
      EncodedBytes += 6;
      if (CallTarget >> 32) {
        EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::IIHF)
                                         .addReg(ScratchReg)
                                         .addImm(CallTarget >> 32));
        EncodedBytes += 6;
      }

      EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BASR)
                                       .addReg(SystemZ::R14D)
                                       .addReg(ScratchReg));
      EncodedBytes += 2;
    }
  } else if (CalleeMO.isGlobal()) {
    const MCExpr *Expr = Lower.getExpr(CalleeMO, SystemZ::S_PLT);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(SystemZ::BRASL).addReg(SystemZ::R14D).addExpr(Expr));
    EncodedBytes += 6;
  }

  // Emit padding.
  unsigned NumBytes = Opers.getNumPatchBytes();
  assert(NumBytes >= EncodedBytes &&
         "Patchpoint can't request size less than the length of a call.");
  assert((NumBytes - EncodedBytes) % 2 == 0 &&
         "Invalid number of NOP bytes requested!");
  while (EncodedBytes < NumBytes)
    EncodedBytes += EmitNop(OutContext, *OutStreamer, NumBytes - EncodedBytes,
                            getSubtargetInfo());
}

void SystemZELFAsmPrinter::LowerPATCHABLE_FUNCTION_ENTER(
    const MachineInstr &MI, SystemZMCInstLower &Lower) {
  const MachineFunction &MF = *(MI.getParent()->getParent());
  const Function &F = MF.getFunction();

  // If patchable-function-entry is set, emit in-function nops here.
  if (F.hasFnAttribute("patchable-function-entry")) {
    unsigned Num = F.getFnAttributeAsParsedInteger("patchable-function-entry");
    for (unsigned I = 0; I < Num; ++I)
      EmitToStreamer(*OutStreamer, MF.getSubtarget().getInstrInfo()->getNop());
    return;
  }
  // Otherwise, emit xray sled.
  // .begin:
  //   j .end    # -> stmg    %r2, %r15, 16(%r15)
  //   nop
  //   llilf   %2, FuncID
  //   brasl   %r14, __xray_FunctionEntry@GOT
  // .end:
  //
  // Update compiler-rt/lib/xray/xray_s390x.cpp accordingly when number
  // of instructions change.
  bool HasVectorFeature =
      TM.getMCSubtargetInfo().hasFeature(SystemZ::FeatureVector) &&
      !TM.getMCSubtargetInfo().hasFeature(SystemZ::FeatureSoftFloat);
  MCSymbol *FuncEntry = OutContext.getOrCreateSymbol(
      HasVectorFeature ? "__xray_FunctionEntryVec" : "__xray_FunctionEntry");
  MCSymbol *BeginOfSled = OutContext.createTempSymbol("xray_sled_", true);
  MCSymbol *EndOfSled = OutContext.createTempSymbol();
  OutStreamer->emitLabel(BeginOfSled);
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::J)
                     .addExpr(MCSymbolRefExpr::create(EndOfSled, OutContext)));
  EmitNop(OutContext, *OutStreamer, 2, getSubtargetInfo());
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::LLILF).addReg(SystemZ::R2D).addImm(0));
  EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BRASL)
                                   .addReg(SystemZ::R14D)
                                   .addExpr(MCSymbolRefExpr::create(
                                       FuncEntry, SystemZ::S_PLT, OutContext)));
  OutStreamer->emitLabel(EndOfSled);
  recordSled(BeginOfSled, MI, SledKind::FUNCTION_ENTER, 2);
}

void SystemZELFAsmPrinter::LowerPATCHABLE_RET(const MachineInstr &MI,
                                              SystemZMCInstLower &Lower) {
  unsigned OpCode = MI.getOperand(0).getImm();
  MCSymbol *FallthroughLabel = nullptr;
  if (OpCode == SystemZ::CondReturn) {
    FallthroughLabel = OutContext.createTempSymbol();
    int64_t Cond0 = MI.getOperand(1).getImm();
    int64_t Cond1 = MI.getOperand(2).getImm();
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BRC)
                                     .addImm(Cond0)
                                     .addImm(Cond1 ^ Cond0)
                                     .addExpr(MCSymbolRefExpr::create(
                                         FallthroughLabel, OutContext)));
  }
  // .begin:
  //   br %r14    # -> stmg    %r2, %r15, 24(%r15)
  //   nop
  //   nop
  //   llilf   %2,FuncID
  //   j       __xray_FunctionExit@GOT
  //
  // Update compiler-rt/lib/xray/xray_s390x.cpp accordingly when number
  // of instructions change.
  bool HasVectorFeature =
      TM.getMCSubtargetInfo().hasFeature(SystemZ::FeatureVector) &&
      !TM.getMCSubtargetInfo().hasFeature(SystemZ::FeatureSoftFloat);
  MCSymbol *FuncExit = OutContext.getOrCreateSymbol(
      HasVectorFeature ? "__xray_FunctionExitVec" : "__xray_FunctionExit");
  MCSymbol *BeginOfSled = OutContext.createTempSymbol("xray_sled_", true);
  OutStreamer->emitLabel(BeginOfSled);
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::BR).addReg(SystemZ::R14D));
  EmitNop(OutContext, *OutStreamer, 4, getSubtargetInfo());
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::LLILF).addReg(SystemZ::R2D).addImm(0));
  EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::J)
                                   .addExpr(MCSymbolRefExpr::create(
                                       FuncExit, SystemZ::S_PLT, OutContext)));
  if (FallthroughLabel)
    OutStreamer->emitLabel(FallthroughLabel);
  recordSled(BeginOfSled, MI, SledKind::FUNCTION_EXIT, 2);
}

void SystemZELFAsmPrinter::lowerLOAD_TLS_BLOCK_ADDR(const MachineInstr &MI,
                                                    SystemZMCInstLower &Lower) {
  Register AddrReg = MI.getOperand(0).getReg();
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();

  // EAR can only load the low subregister so use a shift for %a0 to produce
  // the GR containing %a0 and %a1.
  const Register Reg32 =
      MRI.getTargetRegisterInfo()->getSubReg(AddrReg, SystemZ::subreg_l32);

  // ear <reg>, %a0
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::EAR).addReg(Reg32).addReg(SystemZ::A0));

  // sllg <reg>, <reg>, 32
  EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::SLLG)
                                   .addReg(AddrReg)
                                   .addReg(AddrReg)
                                   .addReg(0)
                                   .addImm(32));

  // ear <reg>, %a1
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::EAR).addReg(Reg32).addReg(SystemZ::A1));
}

void SystemZELFAsmPrinter::lowerLOAD_GLOBAL_STACKGUARD_ADDR(
    const MachineInstr &MI, SystemZMCInstLower &Lower) {
  Register AddrReg = MI.getOperand(0).getReg();
  const MachineFunction &MF = *(MI.getParent()->getParent());
  const Module *M = MF.getFunction().getParent();
  const TargetLowering *TLI = MF.getSubtarget().getTargetLowering();

  // Obtain the global value (assert if stack guard variable can't be found).
  const GlobalVariable *GV = cast<GlobalVariable>(
      TLI->getSDagStackGuard(*M, TLI->getLibcallLoweringInfo()));

  // If configured, emit the `__stack_protector_loc` entry
  if (M->hasStackProtectorGuardRecord()) {
    MCSymbol *Sym = OutContext.createTempSymbol();
    OutStreamer->pushSection();
    OutStreamer->switchSection(OutContext.getELFSection(
        "__stack_protector_loc", ELF::SHT_PROGBITS, ELF::SHF_ALLOC));
    OutStreamer->emitSymbolValue(Sym, getDataLayout().getPointerSize());
    OutStreamer->popSection();
    OutStreamer->emitLabel(Sym);
  }
  // Emit the address load.
  if (M->getPICLevel() == PICLevel::NotPIC) {
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::LARL)
                                     .addReg(AddrReg)
                                     .addExpr(MCSymbolRefExpr::create(
                                         getSymbol(GV), OutContext)));
  } else {
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(SystemZ::LGRL)
                       .addReg(AddrReg)
                       .addExpr(MCSymbolRefExpr::create(
                           getSymbol(GV), SystemZ::S_GOTENT, OutContext)));
  }
}

// The *alignment* of 128-bit vector types is different between the software
// and hardware vector ABIs. If there is an externally visible use of a
// vector type in the module it should be annotated with an attribute.
void SystemZELFAsmPrinter::emitAttributes(Module &M) {
  if (M.getModuleFlag("s390x-visible-vector-ABI")) {
    bool HasVectorFeature =
        TM.getMCSubtargetInfo().hasFeature(SystemZ::FeatureVector);
    OutStreamer->emitGNUAttribute(8, HasVectorFeature ? 2 : 1);
  }
}

// Convert a SystemZ-specific constant pool modifier into the associated
// specifier.
static uint8_t getSpecifierFromModifier(SystemZCP::SystemZCPModifier Modifier) {
  switch (Modifier) {
  case SystemZCP::TLSGD:
    return SystemZ::S_TLSGD;
  case SystemZCP::TLSLDM:
    return SystemZ::S_TLSLDM;
  case SystemZCP::DTPOFF:
    return SystemZ::S_DTPOFF;
  case SystemZCP::NTPOFF:
    return SystemZ::S_NTPOFF;
  }
  llvm_unreachable("Invalid SystemCPModifier!");
}

void SystemZELFAsmPrinter::emitMachineConstantPoolValue(
    MachineConstantPoolValue *MCPV) {
  auto *ZCPV = static_cast<SystemZConstantPoolValue *>(MCPV);

  const MCExpr *Expr = MCSymbolRefExpr::create(
      getSymbol(ZCPV->getGlobalValue()),
      getSpecifierFromModifier(ZCPV->getModifier()), OutContext);
  uint64_t Size = getDataLayout().getTypeAllocSize(ZCPV->getType());

  OutStreamer->emitValue(Expr, Size);
}

void SystemZELFAsmPrinter::emitEndOfAsmFile(Module &M) { emitAttributes(M); }
