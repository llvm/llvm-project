//===- lib/MC/MCWasmStreamer.cpp - Wasm Object Output ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits Wasm .o object files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCWasmStreamer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

MCWasmStreamer::~MCWasmStreamer() = default; // anchor.

void MCWasmStreamer::mergeFragment(MCDataFragment *DF, MCDataFragment *EF) {
  flushPendingLabels(DF, DF->getContents().size());

  for (unsigned I = 0, E = EF->getFixups().size(); I != E; ++I) {
    EF->getFixups()[I].setOffset(EF->getFixups()[I].getOffset() +
                                 DF->getContents().size());
    DF->getFixups().push_back(EF->getFixups()[I]);
  }
  if (DF->getSubtargetInfo() == nullptr && EF->getSubtargetInfo())
    DF->setHasInstructions(*EF->getSubtargetInfo());
  DF->getContents().append(EF->getContents().begin(), EF->getContents().end());
}

void MCWasmStreamer::emitAssemblerFlag(MCAssemblerFlag Flag) {
  // Let the target do whatever target specific stuff it needs to do.
  getAssembler().getBackend().handleAssemblerFlag(Flag);

  // Do any generic stuff we need to do.
  llvm_unreachable("invalid assembler flag!");
}

void MCWasmStreamer::changeSection(MCSection *Section,
                                   const MCExpr *Subsection) {
  MCAssembler &Asm = getAssembler();
  auto *SectionWasm = cast<MCSectionWasm>(Section);
  const MCSymbol *Grp = SectionWasm->getGroup();
  if (Grp)
    Asm.registerSymbol(*Grp);

  this->MCObjectStreamer::changeSection(Section, Subsection);
  Asm.registerSymbol(*Section->getBeginSymbol());
}

void MCWasmStreamer::emitWeakReference(MCSymbol *Alias,
                                       const MCSymbol *Symbol) {
  getAssembler().registerSymbol(*Symbol);
  const MCExpr *Value = MCSymbolRefExpr::create(
      Symbol, MCSymbolRefExpr::VK_WEAKREF, getContext());
  Alias->setVariableValue(Value);
}

bool MCWasmStreamer::emitSymbolAttribute(MCSymbol *S, MCSymbolAttr Attribute) {
  assert(Attribute != MCSA_IndirectSymbol && "indirect symbols not supported");

  auto *Symbol = cast<MCSymbolWasm>(S);

  // Adding a symbol attribute always introduces the symbol; note that an
  // important side effect of calling registerSymbol here is to register the
  // symbol with the assembler.
  getAssembler().registerSymbol(*Symbol);

  switch (Attribute) {
  case MCSA_LazyReference:
  case MCSA_Reference:
  case MCSA_SymbolResolver:
  case MCSA_PrivateExtern:
  case MCSA_WeakDefinition:
  case MCSA_WeakDefAutoPrivate:
  case MCSA_Invalid:
  case MCSA_IndirectSymbol:
  case MCSA_Protected:
    return false;

  case MCSA_Hidden:
    Symbol->setHidden(true);
    break;

  case MCSA_Weak:
  case MCSA_WeakReference:
    Symbol->setWeak(true);
    Symbol->setExternal(true);
    break;

  case MCSA_Global:
    Symbol->setExternal(true);
    break;

  case MCSA_ELF_TypeFunction:
    Symbol->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);
    break;

  case MCSA_ELF_TypeObject:
  case MCSA_Cold:
    break;

  case MCSA_NoDeadStrip:
    Symbol->setNoStrip();
    break;

  default:
    // unrecognized directive
    llvm_unreachable("unexpected MCSymbolAttr");
    return false;
  }

  return true;
}

void MCWasmStreamer::emitCommonSymbol(MCSymbol *S, uint64_t Size,
                                      unsigned ByteAlignment) {
  llvm_unreachable("Common symbols are not yet implemented for Wasm");
}

void MCWasmStreamer::emitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
  cast<MCSymbolWasm>(Symbol)->setSize(Value);
}

void MCWasmStreamer::emitLocalCommonSymbol(MCSymbol *S, uint64_t Size,
                                           unsigned ByteAlignment) {
  llvm_unreachable("Local common symbols are not yet implemented for Wasm");
}

void MCWasmStreamer::emitIdent(StringRef IdentString) {
  // TODO(sbc): Add the ident section once we support mergable strings
  // sections in the object format
}

void MCWasmStreamer::emitInstToFragment(const MCInst &Inst,
                                        const MCSubtargetInfo &STI) {
  this->MCObjectStreamer::emitInstToFragment(Inst, STI);
}

void MCWasmStreamer::emitInstToData(const MCInst &Inst,
                                    const MCSubtargetInfo &STI) {
  MCAssembler &Assembler = getAssembler();
  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  Assembler.getEmitter().encodeInstruction(Inst, VecOS, Fixups, STI);

  // Append the encoded instruction to the current data fragment (or create a
  // new such fragment if the current fragment is not a data fragment).
  MCDataFragment *DF = getOrCreateDataFragment();

  // Add the fixups and data.
  for (unsigned I = 0, E = Fixups.size(); I != E; ++I) {
    Fixups[I].setOffset(Fixups[I].getOffset() + DF->getContents().size());
    DF->getFixups().push_back(Fixups[I]);
  }
  DF->setHasInstructions(STI);
  DF->getContents().append(Code.begin(), Code.end());
}

void MCWasmStreamer::finishImpl() {
  emitFrames(nullptr);

  this->MCObjectStreamer::finishImpl();
}

MCStreamer *llvm::createWasmStreamer(MCContext &Context,
                                     std::unique_ptr<MCAsmBackend> &&MAB,
                                     std::unique_ptr<MCObjectWriter> &&OW,
                                     std::unique_ptr<MCCodeEmitter> &&CE,
                                     bool RelaxAll) {
  MCWasmStreamer *S =
      new MCWasmStreamer(Context, std::move(MAB), std::move(OW), std::move(CE));
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}

void MCWasmStreamer::emitThumbFunc(MCSymbol *Func) {
  llvm_unreachable("Generic Wasm doesn't support this directive");
}

void MCWasmStreamer::emitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  llvm_unreachable("Wasm doesn't support this directive");
}

void MCWasmStreamer::emitZerofill(MCSection *Section, MCSymbol *Symbol,
                                  uint64_t Size, unsigned ByteAlignment,
                                  SMLoc Loc) {
  llvm_unreachable("Wasm doesn't support this directive");
}

void MCWasmStreamer::emitTBSSSymbol(MCSection *Section, MCSymbol *Symbol,
                                    uint64_t Size, unsigned ByteAlignment) {
  llvm_unreachable("Wasm doesn't support this directive");
}
