//=== HexagonMCELFStreamer.cpp - Hexagon subclass of MCELFStreamer -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a stub that parses a MCInst bundle and passes the
// instructions on to the real streamer.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "hexagonmcelfstreamer"

#include "Hexagon.h"
#include "HexagonMCELFStreamer.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCShuffler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<unsigned>
    GPSize("gpsize", cl::NotHidden,
           cl::desc("Global Pointer Addressing Size.  The default size is 8."),
           cl::Prefix, cl::init(8));

void HexagonMCELFStreamer::EmitInstruction(const MCInst &MCK,
                                           const MCSubtargetInfo &STI) {
  MCInst HMI;
  HMI.setOpcode(Hexagon::BUNDLE);
  HMI.addOperand(MCOperand::createImm(0));
  MCInst *MCB;

  if (MCK.getOpcode() != Hexagon::BUNDLE) {
    HMI.addOperand(MCOperand::createInst(&MCK));
    MCB = &HMI;
  } else
    MCB = const_cast<MCInst *>(&MCK);

  // Examines packet and pad the packet, if needed, when an
  // end-loop is in the bundle.
  HexagonMCInstrInfo::padEndloop(*MCB);
  HexagonMCShuffle(*MCII, STI, *MCB);

  assert(HexagonMCInstrInfo::bundleSize(*MCB) <= HEXAGON_PACKET_SIZE);
  bool Extended = false;
  for (auto &I : HexagonMCInstrInfo::bundleInstructions(*MCB)) {
    MCInst *MCI = const_cast<MCInst *>(I.getInst());
    if (Extended) {
      if (HexagonMCInstrInfo::isDuplex(*MCII, *MCI)) {
        MCInst *SubInst = const_cast<MCInst *>(MCI->getOperand(1).getInst());
        HexagonMCInstrInfo::clampExtended(*MCII, *SubInst);
      } else {
        HexagonMCInstrInfo::clampExtended(*MCII, *MCI);
      }
      Extended = false;
    } else {
      Extended = HexagonMCInstrInfo::isImmext(*MCI);
    }
  }

  // At this point, MCB is a bundle
  // Iterate through the bundle and assign addends for the instructions
  for (auto const &I : HexagonMCInstrInfo::bundleInstructions(*MCB)) {
    MCInst *MCI = const_cast<MCInst *>(I.getInst());
    EmitSymbol(*MCI);
  }
  MCObjectStreamer::EmitInstruction(*MCB, STI);
}

void HexagonMCELFStreamer::EmitSymbol(const MCInst &Inst) {
  // Scan for values.
  for (unsigned i = Inst.getNumOperands(); i--;)
    if (Inst.getOperand(i).isExpr())
      visitUsedExpr(*Inst.getOperand(i).getExpr());
}

// EmitCommonSymbol and EmitLocalCommonSymbol are extended versions of the
// functions found in MCELFStreamer.cpp taking AccessSize as an additional
// parameter.
void HexagonMCELFStreamer::HexagonMCEmitCommonSymbol(MCSymbol *Symbol,
                                                     uint64_t Size,
                                                     unsigned ByteAlignment,
                                                     unsigned AccessSize) {
  getAssembler().registerSymbol(*Symbol);
  StringRef sbss[4] = {".sbss.1", ".sbss.2", ".sbss.4", ".sbss.8"};

  auto ELFSymbol = cast<MCSymbolELF>(Symbol);
  if (!ELFSymbol->isBindingSet()) {
    ELFSymbol->setBinding(ELF::STB_GLOBAL);
    ELFSymbol->setExternal(true);
  }

  ELFSymbol->setType(ELF::STT_OBJECT);

  if (ELFSymbol->getBinding() == ELF::STB_LOCAL) {
    StringRef SectionName =
        ((AccessSize == 0) || (Size == 0) || (Size > GPSize))
            ? ".bss"
            : sbss[(Log2_64(AccessSize))];

    MCSection *CrntSection = getCurrentSection().first;
    MCSection *Section = getAssembler().getContext().getELFSection(
        SectionName, ELF::SHT_NOBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC);
    SwitchSection(Section);
    AssignFragment(Symbol, getCurrentFragment());

    MCELFStreamer::EmitCommonSymbol(Symbol, Size, ByteAlignment);
    SwitchSection(CrntSection);
  } else {
    if (ELFSymbol->declareCommon(Size, ByteAlignment))
      report_fatal_error("Symbol: " + Symbol->getName() +
                         " redeclared as different type");
    if ((AccessSize) && (Size <= GPSize)) {
      uint64_t SectionIndex =
          (AccessSize <= GPSize)
              ? ELF::SHN_HEXAGON_SCOMMON + (Log2_64(AccessSize) + 1)
              : (unsigned)ELF::SHN_HEXAGON_SCOMMON;
      ELFSymbol->setIndex(SectionIndex);
    }
  }

  ELFSymbol->setSize(MCConstantExpr::create(Size, getContext()));
}

void HexagonMCELFStreamer::HexagonMCEmitLocalCommonSymbol(
    MCSymbol *Symbol, uint64_t Size, unsigned ByteAlignment,
    unsigned AccessSize) {
  getAssembler().registerSymbol(*Symbol);
  auto ELFSymbol = cast<MCSymbolELF>(Symbol);
  ELFSymbol->setBinding(ELF::STB_LOCAL);
  ELFSymbol->setExternal(false);
  HexagonMCEmitCommonSymbol(Symbol, Size, ByteAlignment, AccessSize);
}

namespace llvm {
MCStreamer *createHexagonELFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                     raw_pwrite_stream &OS, MCCodeEmitter *CE) {
  return new HexagonMCELFStreamer(Context, MAB, OS, CE);
}
}
