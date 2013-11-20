//===---------------------------- StackMaps.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "stackmaps"

#include "llvm/CodeGen/StackMaps.h"

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOpcodes.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

#include <iterator>

using namespace llvm;

PatchPointOpers::PatchPointOpers(const MachineInstr *MI):
  MI(MI),
  HasDef(MI->getOperand(0).isReg() && MI->getOperand(0).isDef() &&
         !MI->getOperand(0).isImplicit()),
  IsAnyReg(MI->getOperand(getMetaIdx(CCPos)).getImm() == CallingConv::AnyReg) {

#ifndef NDEBUG
  {
  unsigned CheckStartIdx = 0, e = MI->getNumOperands();
  while (CheckStartIdx < e && MI->getOperand(CheckStartIdx).isReg() &&
         MI->getOperand(CheckStartIdx).isDef() &&
         !MI->getOperand(CheckStartIdx).isImplicit())
    ++CheckStartIdx;

  assert(getMetaIdx() == CheckStartIdx &&
         "Unexpected additonal definition in Patchpoint intrinsic.");
  }
#endif
}

unsigned PatchPointOpers::getNextScratchIdx(unsigned StartIdx) const {
  if (!StartIdx)
    StartIdx = getVarIdx();

  // Find the next scratch register (implicit def and early clobber)
  unsigned ScratchIdx = StartIdx, e = MI->getNumOperands();
  while (ScratchIdx < e &&
         !(MI->getOperand(ScratchIdx).isReg() &&
           MI->getOperand(ScratchIdx).isDef() &&
           MI->getOperand(ScratchIdx).isImplicit() &&
           MI->getOperand(ScratchIdx).isEarlyClobber()))
    ++ScratchIdx;

  assert(ScratchIdx != e && "No scratch register available");
  return ScratchIdx;
}

void StackMaps::recordStackMapOpers(const MachineInstr &MI, uint32_t ID,
                                    MachineInstr::const_mop_iterator MOI,
                                    MachineInstr::const_mop_iterator MOE,
                                    bool recordResult) {

  MCContext &OutContext = AP.OutStreamer.getContext();
  MCSymbol *MILabel = OutContext.CreateTempSymbol();
  AP.OutStreamer.EmitLabel(MILabel);

  LocationVec CallsiteLocs;

  if (recordResult) {
    std::pair<Location, MachineInstr::const_mop_iterator> ParseResult =
      OpParser(MI.operands_begin(), llvm::next(MI.operands_begin()), AP.TM);

    Location &Loc = ParseResult.first;
    assert(Loc.LocType == Location::Register &&
           "Stackmap return location must be a register.");
    CallsiteLocs.push_back(Loc);
  }

  while (MOI != MOE) {
    std::pair<Location, MachineInstr::const_mop_iterator> ParseResult =
      OpParser(MOI, MOE, AP.TM);

    Location &Loc = ParseResult.first;

    // Move large constants into the constant pool.
    if (Loc.LocType == Location::Constant && (Loc.Offset & ~0xFFFFFFFFULL)) {
      Loc.LocType = Location::ConstantIndex;
      Loc.Offset = ConstPool.getConstantIndex(Loc.Offset);
    }

    CallsiteLocs.push_back(Loc);
    MOI = ParseResult.second;
  }

  const MCExpr *CSOffsetExpr = MCBinaryExpr::CreateSub(
    MCSymbolRefExpr::Create(MILabel, OutContext),
    MCSymbolRefExpr::Create(AP.CurrentFnSym, OutContext),
    OutContext);

  CSInfos.push_back(CallsiteInfo(CSOffsetExpr, ID, CallsiteLocs));
}

static MachineInstr::const_mop_iterator
getStackMapEndMOP(MachineInstr::const_mop_iterator MOI,
                  MachineInstr::const_mop_iterator MOE) {
  for (; MOI != MOE; ++MOI)
    if (MOI->isRegMask() || (MOI->isReg() && MOI->isImplicit()))
      break;

  return MOI;
}

void StackMaps::recordStackMap(const MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::STACKMAP && "exected stackmap");

  int64_t ID = MI.getOperand(0).getImm();
  assert((int32_t)ID == ID && "Stack maps hold 32-bit IDs");
  recordStackMapOpers(MI, ID, llvm::next(MI.operands_begin(), 2),
                      getStackMapEndMOP(MI.operands_begin(),
                                        MI.operands_end()));
}

void StackMaps::recordPatchPoint(const MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::PATCHPOINT && "exected stackmap");

  PatchPointOpers opers(&MI);
  int64_t ID = opers.getMetaOper(PatchPointOpers::IDPos).getImm();
  assert((int32_t)ID == ID && "Stack maps hold 32-bit IDs");
  MachineInstr::const_mop_iterator MOI =
    llvm::next(MI.operands_begin(), opers.getStackMapStartIdx());
  recordStackMapOpers(MI, ID, MOI, getStackMapEndMOP(MOI, MI.operands_end()),
                      opers.isAnyReg() && opers.hasDef());

#ifndef NDEBUG
  // verify anyregcc
  LocationVec &Locations = CSInfos.back().Locations;
  if (opers.isAnyReg()) {
    unsigned NArgs = opers.getMetaOper(PatchPointOpers::NArgPos).getImm();
    for (unsigned i = 0, e = (opers.hasDef() ? NArgs+1 : NArgs); i != e; ++i)
      assert(Locations[i].LocType == Location::Register &&
             "anyreg arg must be in reg.");
  }
#endif
}

/// serializeToStackMapSection conceptually populates the following fields:
///
/// uint32 : Reserved (header)
/// uint32 : NumConstants
/// int64  : Constants[NumConstants]
/// uint32 : NumRecords
/// StkMapRecord[NumRecords] {
///   uint32 : PatchPoint ID
///   uint32 : Instruction Offset
///   uint16 : Reserved (record flags)
///   uint16 : NumLocations
///   Location[NumLocations] {
///     uint8  : Register | Direct | Indirect | Constant | ConstantIndex
///     uint8  : Size in Bytes
///     uint16 : Dwarf RegNum
///     int32  : Offset
///   }
/// }
///
/// Location Encoding, Type, Value:
///   0x1, Register, Reg                 (value in register)
///   0x2, Direct, Reg + Offset          (frame index)
///   0x3, Indirect, [Reg + Offset]      (spilled value)
///   0x4, Constant, Offset              (small constant)
///   0x5, ConstIndex, Constants[Offset] (large constant)
///
void StackMaps::serializeToStackMapSection() {
  // Bail out if there's no stack map data.
  if (CSInfos.empty())
    return;

  MCContext &OutContext = AP.OutStreamer.getContext();
  const TargetRegisterInfo *TRI = AP.TM.getRegisterInfo();

  // Create the section.
  const MCSection *StackMapSection =
    OutContext.getObjectFileInfo()->getStackMapSection();
  AP.OutStreamer.SwitchSection(StackMapSection);

  // Emit a dummy symbol to force section inclusion.
  AP.OutStreamer.EmitLabel(
    OutContext.GetOrCreateSymbol(Twine("__LLVM_StackMaps")));

  // Serialize data.
  const char *WSMP = "Stack Maps: ";
  (void)WSMP;
  const MCRegisterInfo &MCRI = *OutContext.getRegisterInfo();

  DEBUG(dbgs() << "********** Stack Map Output **********\n");

  // Header.
  AP.OutStreamer.EmitIntValue(0, 4);

  // Num constants.
  AP.OutStreamer.EmitIntValue(ConstPool.getNumConstants(), 4);

  // Constant pool entries.
  for (unsigned i = 0; i < ConstPool.getNumConstants(); ++i)
    AP.OutStreamer.EmitIntValue(ConstPool.getConstant(i), 8);

  DEBUG(dbgs() << WSMP << "#callsites = " << CSInfos.size() << "\n");
  AP.OutStreamer.EmitIntValue(CSInfos.size(), 4);

  for (CallsiteInfoList::const_iterator CSII = CSInfos.begin(),
                                        CSIE = CSInfos.end();
       CSII != CSIE; ++CSII) {

    unsigned CallsiteID = CSII->ID;
    const LocationVec &CSLocs = CSII->Locations;

    DEBUG(dbgs() << WSMP << "callsite " << CallsiteID << "\n");

    // Verify stack map entry. It's better to communicate a problem to the
    // runtime than crash in case of in-process compilation. Currently, we do
    // simple overflow checks, but we may eventually communicate other
    // compilation errors this way.
    if (CSLocs.size() > UINT16_MAX) {
      AP.OutStreamer.EmitIntValue(UINT32_MAX, 4); // Invalid ID.
      AP.OutStreamer.EmitValue(CSII->CSOffsetExpr, 4);
      AP.OutStreamer.EmitIntValue(0, 2); // Reserved.
      AP.OutStreamer.EmitIntValue(0, 2); // 0 locations.
      continue;
    }

    AP.OutStreamer.EmitIntValue(CallsiteID, 4);
    AP.OutStreamer.EmitValue(CSII->CSOffsetExpr, 4);

    // Reserved for flags.
    AP.OutStreamer.EmitIntValue(0, 2);

    DEBUG(dbgs() << WSMP << "  has " << CSLocs.size() << " locations\n");

    AP.OutStreamer.EmitIntValue(CSLocs.size(), 2);

    unsigned operIdx = 0;
    for (LocationVec::const_iterator LocI = CSLocs.begin(), LocE = CSLocs.end();
         LocI != LocE; ++LocI, ++operIdx) {
      const Location &Loc = *LocI;
      DEBUG(
        dbgs() << WSMP << "  Loc " << operIdx << ": ";
        switch (Loc.LocType) {
        case Location::Unprocessed:
          dbgs() << "<Unprocessed operand>";
          break;
        case Location::Register:
          dbgs() << "Register " << MCRI.getName(Loc.Reg);
          break;
        case Location::Direct:
          dbgs() << "Direct " << MCRI.getName(Loc.Reg);
          if (Loc.Offset)
            dbgs() << " + " << Loc.Offset;
          break;
        case Location::Indirect:
          dbgs() << "Indirect " << MCRI.getName(Loc.Reg)
                 << " + " << Loc.Offset;
          break;
        case Location::Constant:
          dbgs() << "Constant " << Loc.Offset;
          break;
        case Location::ConstantIndex:
          dbgs() << "Constant Index " << Loc.Offset;
          break;
        }
        dbgs() << "\n";
      );

      unsigned RegNo = 0;
      int Offset = Loc.Offset;
      if(Loc.Reg) {
        RegNo = MCRI.getDwarfRegNum(Loc.Reg, false);
        for (MCSuperRegIterator SR(Loc.Reg, TRI);
             SR.isValid() && (int)RegNo < 0; ++SR) {
          RegNo = TRI->getDwarfRegNum(*SR, false);
        }
        // If this is a register location, put the subregister byte offset in
        // the location offset.
        if (Loc.LocType == Location::Register) {
          assert(!Loc.Offset && "Register location should have zero offset");
          unsigned LLVMRegNo = MCRI.getLLVMRegNum(RegNo, false);
          unsigned SubRegIdx = MCRI.getSubRegIndex(LLVMRegNo, Loc.Reg);
          if (SubRegIdx)
            Offset = MCRI.getSubRegIdxOffset(SubRegIdx);
        }
      }
      else {
        assert(Loc.LocType != Location::Register &&
               "Missing location register");
      }
      AP.OutStreamer.EmitIntValue(Loc.LocType, 1);
      AP.OutStreamer.EmitIntValue(Loc.Size, 1);
      AP.OutStreamer.EmitIntValue(RegNo, 2);
      AP.OutStreamer.EmitIntValue(Offset, 4);
    }
  }

  AP.OutStreamer.AddBlankLine();

  CSInfos.clear();
}
