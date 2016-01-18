//===-- X86ELFRelocationInfo.cpp ----------------------------------------===//
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
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/ELF.h"

using namespace llvm;
using namespace object;
using namespace ELF;

namespace {
class X86_64ELFRelocationInfo : public MCRelocationInfo {
public:
  X86_64ELFRelocationInfo(MCContext &Ctx) : MCRelocationInfo(Ctx) {}

  const MCExpr *createExprForRelocation(RelocationRef Rel) override {
    uint64_t RelType = Rel.getType();
    elf_symbol_iterator SymI = Rel.getSymbol();

    ErrorOr<StringRef> SymNameOrErr = SymI->getName();
    if (std::error_code EC = SymNameOrErr.getError())
      report_fatal_error(EC.message());
    StringRef SymName = *SymNameOrErr;

    ErrorOr<uint64_t> SymAddr = SymI->getAddress();
    if (std::error_code EC = SymAddr.getError())
      report_fatal_error(EC.message());
    uint64_t SymSize = SymI->getSize();
    int64_t Addend = *ELFRelocationRef(Rel).getAddend();

    MCSymbol *Sym = Ctx.getOrCreateSymbol(SymName);
    // FIXME: check that the value is actually the same.
    if (!Sym->isVariable())
      Sym->setVariableValue(MCConstantExpr::create(*SymAddr, Ctx));

    const MCExpr *Expr = nullptr;
    // If hasAddend is true, then we need to add Addend (r_addend) to Expr.
    bool hasAddend = false;

    // The AMD64 SysV ABI says:
    // A: the addend used to compute the value of the relocatable field.
    // B: the base address at which a shared object has been loaded into memory
    //    during execution. Generally, a shared object is built with a 0 base
    //    virtual address, but the execution address will be different.
    // G: the offset into the global offset table at which the relocation
    //    entry's symbol will reside during execution.
    // GOT: the address of the global offset table.
    // L: the place (section offset or address) of the Procedure Linkage Table
    //    entry for a symbol.
    // P: the place (section offset or address) of the storage unit being
    //    relocated (computed using r_offset).
    // S: the value of the symbol whose index resides in the relocation entry.
    // Z: the size of the symbol whose index resides in the relocation entry.

    switch(RelType) {
    case R_X86_64_NONE:
    case R_X86_64_COPY:
      // none
      break;
    case R_X86_64_64:
    case R_X86_64_16:
    case R_X86_64_8:
      // S + A
    case R_X86_64_32:
    case R_X86_64_32S:
      // S + A (We don't care about the result not fitting in 32 bits.)
    case R_X86_64_PC32:
    case R_X86_64_PC16:
    case R_X86_64_PC8:
    case R_X86_64_PC64:
      // S + A - P (P/pcrel is implicit)
      hasAddend = true;
      Expr = MCSymbolRefExpr::create(Sym, Ctx);
      break;
    case R_X86_64_GOT32:
    case R_X86_64_GOT64:
    case R_X86_64_GOTPC32:
    case R_X86_64_GOTPC64:
    case R_X86_64_GOTPLT64:
      // G + A
      hasAddend = true;
      Expr = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_GOT, Ctx);
      break;
    case R_X86_64_PLT32:
      // L + A - P -> S@PLT + A
      hasAddend = true;
      Expr = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_PLT, Ctx);
      break;
    case R_X86_64_GLOB_DAT:
    case R_X86_64_JUMP_SLOT:
      // S
      Expr = MCSymbolRefExpr::create(Sym, Ctx);
      break;
    case R_X86_64_GOTPCREL:
    case R_X86_64_GOTPCREL64:
      // G + GOT + A - P -> S@GOTPCREL + A
      hasAddend = true;
      Expr = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_GOTPCREL, Ctx);
      break;
    case R_X86_64_GOTOFF64:
      // S + A - GOT
      Expr = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_GOTOFF, Ctx);
      break;
    case R_X86_64_PLTOFF64:
      // L + A - GOT
      break;
    case R_X86_64_SIZE32:
    case R_X86_64_SIZE64:
      // Z + A
      Expr = MCConstantExpr::create(SymSize, Ctx);
      break;
    default:
      Expr = MCSymbolRefExpr::create(Sym, Ctx);
      break;
    }
    if (Expr && hasAddend && Addend != 0)
      Expr = MCBinaryExpr::createAdd(Expr,
                                     MCConstantExpr::create(Addend, Ctx),
                                     Ctx);
    return Expr;
  }
};
} // End unnamed namespace

/// createX86ELFRelocationInfo - Construct an X86 Mach-O RelocationInfo.
MCRelocationInfo *llvm::createX86_64ELFRelocationInfo(MCContext &Ctx) {
  // We only handle x86-64 for now.
  return new X86_64ELFRelocationInfo(Ctx);
}
