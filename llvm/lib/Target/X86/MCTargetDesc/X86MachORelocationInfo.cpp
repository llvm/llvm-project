//===-- X86MachORelocationInfo.cpp ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRelocationInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/MachO.h"

using namespace llvm;
using namespace object;
using namespace MachO;

namespace {
class X86_64MachORelocationInfo : public MCRelocationInfo {
public:
  X86_64MachORelocationInfo(MCContext &Ctx) : MCRelocationInfo(Ctx) {}

  const MCExpr *createExprForRelocation(RelocationRef Rel) override {
    const MachOObjectFile *Obj = cast<MachOObjectFile>(Rel.getObject());

    uint64_t RelType = Rel.getType();
    symbol_iterator SymI = Rel.getSymbol();

    ErrorOr<StringRef> SymNameOrErr = SymI->getName();
    if (std::error_code EC = SymNameOrErr.getError())
      report_fatal_error(EC.message());
    StringRef SymName = *SymNameOrErr;
    uint64_t SymAddr = SymI->getValue();

    any_relocation_info RE = Obj->getRelocation(Rel.getRawDataRefImpl());
    bool isPCRel = Obj->getAnyRelocationPCRel(RE);

    MCSymbol *Sym = Ctx.getOrCreateSymbol(SymName);
    // FIXME: check that the value is actually the same.
    if (!Sym->isVariable())
      Sym->setVariableValue(MCConstantExpr::create(SymAddr, Ctx));
    const MCExpr *Expr = nullptr;

    switch(RelType) {
    case X86_64_RELOC_TLV:
      Expr = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_TLVP, Ctx);
      break;
    case X86_64_RELOC_SIGNED_4:
      Expr = MCBinaryExpr::createAdd(MCSymbolRefExpr::create(Sym, Ctx),
                                     MCConstantExpr::create(4, Ctx),
                                     Ctx);
      break;
    case X86_64_RELOC_SIGNED_2:
      Expr = MCBinaryExpr::createAdd(MCSymbolRefExpr::create(Sym, Ctx),
                                     MCConstantExpr::create(2, Ctx),
                                     Ctx);
      break;
    case X86_64_RELOC_SIGNED_1:
      Expr = MCBinaryExpr::createAdd(MCSymbolRefExpr::create(Sym, Ctx),
                                     MCConstantExpr::create(1, Ctx),
                                     Ctx);
      break;
    case X86_64_RELOC_GOT_LOAD:
      Expr = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_GOTPCREL, Ctx);
      break;
    case X86_64_RELOC_GOT:
      Expr = MCSymbolRefExpr::create(Sym, isPCRel ?
                                     MCSymbolRefExpr::VK_GOTPCREL :
                                     MCSymbolRefExpr::VK_GOT,
                                     Ctx);
      break;
    case X86_64_RELOC_SUBTRACTOR:
      {
        Rel.moveNext();
        any_relocation_info RENext =
            Obj->getRelocation(Rel.getRawDataRefImpl());

        // X86_64_SUBTRACTOR must be followed by a relocation of type
        // X86_64_RELOC_UNSIGNED.
        // NOTE: Scattered relocations don't exist on x86_64.
        unsigned RType = Obj->getAnyRelocationType(RENext);
        if (RType != X86_64_RELOC_UNSIGNED)
          report_fatal_error("Expected X86_64_RELOC_UNSIGNED after "
                             "X86_64_RELOC_SUBTRACTOR.");

        const MCExpr *LHS = MCSymbolRefExpr::create(Sym, Ctx);

        symbol_iterator RSymI = Rel.getSymbol();
        uint64_t RSymAddr = RSymI->getValue();
        ErrorOr<StringRef> RSymName = RSymI->getName();
        if (std::error_code EC = RSymName.getError())
          report_fatal_error(EC.message());

        MCSymbol *RSym = Ctx.getOrCreateSymbol(*RSymName);
        if (!RSym->isVariable())
          RSym->setVariableValue(MCConstantExpr::create(RSymAddr, Ctx));

        const MCExpr *RHS = MCSymbolRefExpr::create(RSym, Ctx);

        Expr = MCBinaryExpr::createSub(LHS, RHS, Ctx);
        break;
      }
    default:
      Expr = MCSymbolRefExpr::create(Sym, Ctx);
      break;
    }
    return Expr;
  }
};
} // End unnamed namespace

/// createX86_64MachORelocationInfo - Construct an X86-64 Mach-O RelocationInfo.
MCRelocationInfo *llvm::createX86_64MachORelocationInfo(MCContext &Ctx) {
  return new X86_64MachORelocationInfo(Ctx);
}
