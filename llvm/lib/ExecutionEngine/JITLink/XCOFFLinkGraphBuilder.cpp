// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic XCOFF LinkGraph building code.
//
//===----------------------------------------------------------------------===//

#include "XCOFFLinkGraphBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/ppc64.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/MemoryFlags.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

XCOFFLinkGraphBuilder::XCOFFLinkGraphBuilder(
    const object::XCOFFObjectFile &Obj,
    std::shared_ptr<orc::SymbolStringPool> SSP, Triple TT,
    SubtargetFeatures Features,
    LinkGraph::GetEdgeKindNameFunction GetEdgeKindName)
    : Obj(Obj),
      G(std::make_unique<LinkGraph>(
          std::string(Obj.getFileName()), std::move(SSP), std::move(TT),
          std::move(Features), std::move(GetEdgeKindName))) {}

#ifndef NDEBUG
static llvm::StringRef getStorageClassString(XCOFF::StorageClass SC) {
  switch (SC) {
  case XCOFF::StorageClass::C_FILE:
    return "C_FILE (File name)";
  case XCOFF::StorageClass::C_BINCL:
    return "C_BINCL (Beginning of include file)";
  case XCOFF::StorageClass::C_EINCL:
    return "C_EINCL (Ending of include file)";
  case XCOFF::StorageClass::C_GSYM:
    return "C_GSYM (Global variable)";
  case XCOFF::StorageClass::C_STSYM:
    return "C_STSYM (Statically allocated symbol)";
  case XCOFF::StorageClass::C_BCOMM:
    return "C_BCOMM (Beginning of common block)";
  case XCOFF::StorageClass::C_ECOMM:
    return "C_ECOMM (End of common block)";
  case XCOFF::StorageClass::C_ENTRY:
    return "C_ENTRY (Alternate entry)";
  case XCOFF::StorageClass::C_BSTAT:
    return "C_BSTAT (Beginning of static block)";
  case XCOFF::StorageClass::C_ESTAT:
    return "C_ESTAT (End of static block)";
  case XCOFF::StorageClass::C_GTLS:
    return "C_GTLS (Global thread-local variable)";
  case XCOFF::StorageClass::C_STTLS:
    return "C_STTLS (Static thread-local variable)";
  case XCOFF::StorageClass::C_DWARF:
    return "C_DWARF (DWARF section symbol)";
  case XCOFF::StorageClass::C_LSYM:
    return "C_LSYM (Automatic variable allocated on stack)";
  case XCOFF::StorageClass::C_PSYM:
    return "C_PSYM (Argument to subroutine allocated on stack)";
  case XCOFF::StorageClass::C_RSYM:
    return "C_RSYM (Register variable)";
  case XCOFF::StorageClass::C_RPSYM:
    return "C_RPSYM (Argument to function stored in register)";
  case XCOFF::StorageClass::C_ECOML:
    return "C_ECOML (Local member of common block)";
  case XCOFF::StorageClass::C_FUN:
    return "C_FUN (Function or procedure)";
  case XCOFF::StorageClass::C_EXT:
    return "C_EXT (External symbol)";
  case XCOFF::StorageClass::C_WEAKEXT:
    return "C_WEAKEXT (Weak external symbol)";
  case XCOFF::StorageClass::C_NULL:
    return "C_NULL";
  case XCOFF::StorageClass::C_STAT:
    return "C_STAT (Static)";
  case XCOFF::StorageClass::C_BLOCK:
    return "C_BLOCK (\".bb\" or \".eb\")";
  case XCOFF::StorageClass::C_FCN:
    return "C_FCN (\".bf\" or \".ef\")";
  case XCOFF::StorageClass::C_HIDEXT:
    return "C_HIDEXT (Un-named external symbol)";
  case XCOFF::StorageClass::C_INFO:
    return "C_INFO (Comment string in .info section)";
  case XCOFF::StorageClass::C_DECL:
    return "C_DECL (Declaration of object)";
  case XCOFF::StorageClass::C_AUTO:
    return "C_AUTO (Automatic variable)";
  case XCOFF::StorageClass::C_REG:
    return "C_REG (Register variable)";
  case XCOFF::StorageClass::C_EXTDEF:
    return "C_EXTDEF (External definition)";
  case XCOFF::StorageClass::C_LABEL:
    return "C_LABEL (Label)";
  case XCOFF::StorageClass::C_ULABEL:
    return "C_ULABEL (Undefined label)";
  case XCOFF::StorageClass::C_MOS:
    return "C_MOS (Member of structure)";
  case XCOFF::StorageClass::C_ARG:
    return "C_ARG (Function argument)";
  case XCOFF::StorageClass::C_STRTAG:
    return "C_STRTAG (Structure tag)";
  case XCOFF::StorageClass::C_MOU:
    return "C_MOU (Member of union)";
  case XCOFF::StorageClass::C_UNTAG:
    return "C_UNTAG (Union tag)";
  case XCOFF::StorageClass::C_TPDEF:
    return "C_TPDEF (Type definition)";
  case XCOFF::StorageClass::C_USTATIC:
    return "C_USTATIC (Undefined static)";
  case XCOFF::StorageClass::C_ENTAG:
    return "C_ENTAG (Enumeration tag)";
  case XCOFF::StorageClass::C_MOE:
    return "C_MOE (Member of enumeration)";
  case XCOFF::StorageClass::C_REGPARM:
    return "C_REGPARM (Register parameter)";
  case XCOFF::StorageClass::C_FIELD:
    return "C_FIELD (Bit field)";
  case XCOFF::StorageClass::C_EOS:
    return "C_EOS (End of structure)";
  case XCOFF::StorageClass::C_LINE:
    return "C_LINE";
  case XCOFF::StorageClass::C_ALIAS:
    return "C_ALIAS (Duplicate tag)";
  case XCOFF::StorageClass::C_HIDDEN:
    return "C_HIDDEN (Special storage class for external)";
  case XCOFF::StorageClass::C_EFCN:
    return "C_EFCN (Physical end of function)";
  case XCOFF::StorageClass::C_TCSYM:
    return "C_TCSYM (Reserved)";
  }
  llvm_unreachable("Unknown XCOFF::StorageClass enum");
}
#endif

Error XCOFFLinkGraphBuilder::processSections() {
  LLVM_DEBUG(dbgs() << "  Creating graph sections...\n");

  UndefSection = &G->createSection("*UND*", orc::MemProt::None);

  for (object::SectionRef Section : Obj.sections()) {
    auto SectionName = Section.getName();
    if (!SectionName)
      return SectionName.takeError();

    LLVM_DEBUG({
      dbgs() << "    section = " << *SectionName
             << ", idx = " << Section.getIndex()
             << ", size = " << format_hex_no_prefix(Section.getSize(), 8)
             << ", vma = " << format_hex(Section.getAddress(), 16) << "\n";
    });

    // We can skip debug (including dawrf) and pad sections
    if (Section.isDebugSection() || *SectionName == "pad")
      continue;
    LLVM_DEBUG(dbgs() << "        creating graph section\n");

    orc::MemProt Prot = orc::MemProt::Read;
    if (Section.isText())
      Prot |= orc::MemProt::Exec;
    if (Section.isData() || Section.isBSS())
      Prot |= orc::MemProt::Write;

    jitlink::Section *GraphSec = &G->createSection(*SectionName, Prot);
    // TODO: Check for no_alloc for certain sections

    assert(!SectionTable.contains(Section.getIndex()) &&
           "Section with same index already exists");
    SectionTable[Section.getIndex()] = {GraphSec, Section};
  }

  return Error::success();
}

static std::optional<object::XCOFFSymbolRef>
getXCOFFSymbolContainingSymbolRef(const object::XCOFFObjectFile &Obj,
                                  const object::SymbolRef &Sym) {
  const object::XCOFFSymbolRef SymRef =
      Obj.toSymbolRef(Sym.getRawDataRefImpl());
  if (!SymRef.isCsectSymbol())
    return std::nullopt;

  Expected<object::XCOFFCsectAuxRef> CsectAuxEntOrErr =
      SymRef.getXCOFFCsectAuxRef();
  if (!CsectAuxEntOrErr || !CsectAuxEntOrErr.get().isLabel())
    return std::nullopt;
  uint32_t Idx =
      static_cast<uint32_t>(CsectAuxEntOrErr.get().getSectionOrLength());
  object::DataRefImpl DRI;
  DRI.p = Obj.getSymbolByIndex(Idx);
  return object::XCOFFSymbolRef(DRI, &Obj);
}

#ifndef NDEBUG
static void printSymbolEntry(raw_ostream &OS,
                             const object::XCOFFObjectFile &Obj,
                             const object::XCOFFSymbolRef &Sym) {
  OS << "    " << format_hex(cantFail(Sym.getAddress()), 16);
  OS << " " << left_justify(cantFail(Sym.getName()), 10);
  if (Sym.isCsectSymbol()) {
    auto CsectAuxEntry = cantFail(Sym.getXCOFFCsectAuxRef());
    if (!CsectAuxEntry.isLabel()) {
      std::string MCStr =
          "[" +
          XCOFF::getMappingClassString(CsectAuxEntry.getStorageMappingClass())
              .str() +
          "]";
      OS << left_justify(MCStr, 3);
    }
  }
  OS << " " << format_hex(Sym.getSize(), 8);
  OS << " " << Sym.getSectionNumber();
  OS << " " << getStorageClassString(Sym.getStorageClass());
  OS << " (idx: " << Obj.getSymbolIndex(Sym.getRawDataRefImpl().p) << ")";
  if (Sym.isCsectSymbol()) {
    if (auto ParentSym = getXCOFFSymbolContainingSymbolRef(Obj, Sym)) {
      OS << " (csect idx: "
         << Obj.getSymbolIndex(ParentSym->getRawDataRefImpl().p) << ")";
    }
  }
  OS << "\n";
}
#endif

Error XCOFFLinkGraphBuilder::processCsectsAndSymbols() {
  LLVM_DEBUG(dbgs() << "  Creating graph blocks and symbols...\n");

  for ([[maybe_unused]] auto [K, V] : SectionTable) {
    LLVM_DEBUG(dbgs() << "    section entry(idx: " << K
                      << " section: " << V.Section->getName() << ")\n");
  }

  for (object::XCOFFSymbolRef Symbol : Obj.symbols()) {
    LLVM_DEBUG({ printSymbolEntry(dbgs(), Obj, Symbol); });

    auto Flags = Symbol.getFlags();
    if (!Flags)
      return Flags.takeError();

    bool External = *Flags & object::SymbolRef::SF_Undefined;
    bool Weak = *Flags & object::SymbolRef::SF_Weak;
    bool Global = *Flags & object::SymbolRef::SF_Global;

    auto SymbolIndex = Obj.getSymbolIndex(Symbol.getEntryAddress());
    auto SymbolName = Symbol.getName();
    if (!SymbolName)
      return SymbolName.takeError();

    if (External) {
      LLVM_DEBUG(dbgs() << "      created external symbol\n");
      SymbolIndexTable[SymbolIndex] =
          &G->addExternalSymbol(*SymbolName, Symbol.getSize(), Weak);
      continue;
    }

    if (!Symbol.isCsectSymbol()) {
      LLVM_DEBUG(dbgs() << "      skipped: not a csect symbol\n");
      continue;
    }

    auto ParentSym = getXCOFFSymbolContainingSymbolRef(Obj, Symbol);
    object::XCOFFSymbolRef CsectSymbol = ParentSym ? *ParentSym : Symbol;

    auto CsectSymbolIndex = Obj.getSymbolIndex(CsectSymbol.getEntryAddress());
    auto ParentSectionNumber = CsectSymbol.getSectionNumber();

    bool IsUndefinedSection = !SectionTable.contains(ParentSectionNumber);
    Section *ParentSection = !IsUndefinedSection
                                 ? SectionTable[ParentSectionNumber].Section
                                 : UndefSection;
    Block *B = nullptr;

    // TODO: Clean up the logic for handling undefined symbols
    if (!CsectTable.contains(CsectSymbolIndex) && !IsUndefinedSection) {
      object::SectionRef &SectionRef =
          SectionTable[ParentSectionNumber].SectionData;
      auto Data = SectionRef.getContents();
      if (!Data)
        return Data.takeError();
      auto CsectSymbolAddr = CsectSymbol.getAddress();
      if (!CsectSymbolAddr)
        return CsectSymbolAddr.takeError();

      ArrayRef<char> SectionBuffer{Data->data(), Data->size()};
      auto Offset = *CsectSymbolAddr - SectionRef.getAddress();

      LLVM_DEBUG(dbgs() << "      symbol entry: offset = " << Offset
                        << ", size = " << CsectSymbol.getSize()
                        << ", storage class = "
                        << getStorageClassString(CsectSymbol.getStorageClass())
                        << "\n");

      B = &G->createContentBlock(
          *ParentSection, SectionBuffer.slice(Offset, CsectSymbol.getSize()),
          orc::ExecutorAddr(*CsectSymbolAddr), CsectSymbol.getAlignment(), 0);

      CsectTable[CsectSymbolIndex] = B;
    } else {
      B = CsectTable[CsectSymbolIndex];
    }

    Scope S{Scope::Local};
    if (Symbol.getSymbolType() & XCOFF::SYM_V_HIDDEN ||
        Symbol.getSymbolType() & XCOFF::SYM_V_INTERNAL)
      S = Scope::Hidden;
    else if (Global)
      S = Scope::Default;
    // TODO: map all symbols for c++ static initialization to SideEffectOnly

    Linkage L = Weak ? Linkage::Weak : Linkage::Strong;
    auto SymbolAddr = Symbol.getAddress();
    if (!SymbolAddr)
      return SymbolAddr.takeError();
    auto IsCallableOrErr = Symbol.isFunction();
    if (!IsCallableOrErr)
      return IsCallableOrErr.takeError();

    auto BlockOffset = *SymbolAddr - B->getAddress().getValue();

    LLVM_DEBUG(dbgs() << "      creating with linkage = " << getLinkageName(L)
                      << ", scope = " << getScopeName(S) << ", B = "
                      << format_hex(B->getAddress().getValue(), 16) << "\n");

    SymbolIndexTable[SymbolIndex] =
        &G->addDefinedSymbol(*B, BlockOffset, *SymbolName, Symbol.getSize(), L,
                             S, *IsCallableOrErr, true);
  }

  return Error::success();
}

Error XCOFFLinkGraphBuilder::processRelocations() {
  LLVM_DEBUG(dbgs() << "  Creating relocations...\n");

  for (object::SectionRef Section : Obj.sections()) {
    auto SectionName = Section.getName();
    if (!SectionName)
      return SectionName.takeError();

    LLVM_DEBUG(dbgs() << "    Relocations for section " << *SectionName
                      << ":\n");

    for (object::RelocationRef Relocation : Section.relocations()) {
      SmallString<16> RelocName;
      Relocation.getTypeName(RelocName);
      object::SymbolRef Symbol = *Relocation.getSymbol();

      auto TargetSymbol = Symbol.getName();
      if (!TargetSymbol)
        return TargetSymbol.takeError();

      auto SymbolIndex = Obj.getSymbolIndex(Symbol.getRawDataRefImpl().p);

      LLVM_DEBUG(dbgs() << "      " << format_hex(Relocation.getOffset(), 16)
                        << " (idx: " << SymbolIndex << ")"
                        << " " << RelocName << " " << *TargetSymbol << "\n";);

      assert(SymbolIndexTable.contains(SymbolIndex) &&
             "Relocation needs a record in the symbol table");
      auto *S = SymbolIndexTable[SymbolIndex];
      auto It = find_if(G->blocks(),
                        [Target = orc::ExecutorAddr(Section.getAddress() +
                                                    Relocation.getOffset())](
                            const Block *B) -> bool {
                          return B->getRange().contains(Target);
                        });
      assert(It != G->blocks().end() &&
             "Cannot find the target relocation block");
      Block *B = *It;

      auto TargetBlockOffset = Section.getAddress() + Relocation.getOffset() -
                               B->getAddress().getValue();
      switch (Relocation.getType()) {
      case XCOFF::R_POS:
        B->addEdge(ppc64::EdgeKind_ppc64::Pointer64, TargetBlockOffset, *S, 0);
        break;
      default:
        SmallString<16> RelocType;
        Relocation.getTypeName(RelocType);
        return make_error<StringError>(
            "Unsupported Relocation Type: " + RelocType, std::error_code());
      }
    }
  }

  return Error::success();
}

Expected<std::unique_ptr<LinkGraph>> XCOFFLinkGraphBuilder::buildGraph() {
  LLVM_DEBUG(dbgs() << "Building XCOFFLinkGraph...\n");

  // FIXME: Check to make sure the object is relocatable

  if (auto Err = processSections())
    return Err;
  if (auto Err = processCsectsAndSymbols())
    return Err;
  if (auto Err = processRelocations())
    return Err;

  return std::move(G);
}

} // namespace jitlink
} // namespace llvm
