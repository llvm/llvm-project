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

static llvm::raw_ostream &debugStorageClass(llvm::raw_ostream &OS,
                                            XCOFF::StorageClass SC) {
  switch (SC) {
    // Debug symbols
  case XCOFF::StorageClass::C_FILE:
    OS << "C_FILE (File name)";
    break;
  case XCOFF::StorageClass::C_BINCL:
    OS << "C_BINCL (Beginning of include file)";
    break;
  case XCOFF::StorageClass::C_EINCL:
    OS << "C_EINCL (Ending of include file)";
    break;
  case XCOFF::StorageClass::C_GSYM:
    OS << "C_GSYM (Global variable)";
    break;
  case XCOFF::StorageClass::C_STSYM:
    OS << "C_STSYM (Statically allocated symbol)";
    break;
  case XCOFF::StorageClass::C_BCOMM:
    OS << "C_BCOMM (Beginning of common block)";
    break;
  case XCOFF::StorageClass::C_ECOMM:
    OS << "C_ECOMM (End of common block)";
    break;
  case XCOFF::StorageClass::C_ENTRY:
    OS << "C_ENTRY (Alternate entry)";
    break;
  case XCOFF::StorageClass::C_BSTAT:
    OS << "C_BSTAT (Beginning of static block)";
    break;
  case XCOFF::StorageClass::C_ESTAT:
    OS << "C_ESTAT (End of static block)";
    break;
  case XCOFF::StorageClass::C_GTLS:
    OS << "C_GTLS (Global thread-local variable)";
    break;
  case XCOFF::StorageClass::C_STTLS:
    OS << "C_STTLS (Static thread-local variable)";
    break;

    // DWARF symbols
  case XCOFF::StorageClass::C_DWARF:
    OS << "C_DWARF (DWARF section symbol)";
    break;

    // Absolute symbols
  case XCOFF::StorageClass::C_LSYM:
    OS << "C_LSYM (Automatic variable allocated on stack)";
    break;
  case XCOFF::StorageClass::C_PSYM:
    OS << "C_PSYM (Argument to subroutine allocated on stack)";
    break;
  case XCOFF::StorageClass::C_RSYM:
    OS << "C_RSYM (Register variable)";
    break;
  case XCOFF::StorageClass::C_RPSYM:
    OS << "C_RPSYM (Argument to function stored in register)";
    break;
  case XCOFF::StorageClass::C_ECOML:
    OS << "C_ECOML (Local member of common block)";
    break;
  case XCOFF::StorageClass::C_FUN:
    OS << "C_FUN (Function or procedure)";
    break;

    // External symbols
  case XCOFF::StorageClass::C_EXT:
    OS << "C_EXT (External symbol)";
    break;
  case XCOFF::StorageClass::C_WEAKEXT:
    OS << "C_WEAKEXT (Weak external symbol)";
    break;

    // General sections
  case XCOFF::StorageClass::C_NULL:
    OS << "C_NULL";
    break;
  case XCOFF::StorageClass::C_STAT:
    OS << "C_STAT (Static)";
    break;
  case XCOFF::StorageClass::C_BLOCK:
    OS << "C_BLOCK (\".bb\" or \".eb\")";
    break;
  case XCOFF::StorageClass::C_FCN:
    OS << "C_FCN (\".bf\" or \".ef\")";
    break;
  case XCOFF::StorageClass::C_HIDEXT:
    OS << "C_HIDEXT (Un-named external symbol)";
    break;
  case XCOFF::StorageClass::C_INFO:
    OS << "C_INFO (Comment string in .info section)";
    break;
  case XCOFF::StorageClass::C_DECL:
    OS << "C_DECL (Declaration of object)";
    break;

    // Obsolete/Undocumented
  case XCOFF::StorageClass::C_AUTO:
    OS << "C_AUTO (Automatic variable)";
    break;
  case XCOFF::StorageClass::C_REG:
    OS << "C_REG (Register variable)";
    break;
  case XCOFF::StorageClass::C_EXTDEF:
    OS << "C_EXTDEF (External definition)";
    break;
  case XCOFF::StorageClass::C_LABEL:
    OS << "C_LABEL (Label)";
    break;
  case XCOFF::StorageClass::C_ULABEL:
    OS << "C_ULABEL (Undefined label)";
    break;
  case XCOFF::StorageClass::C_MOS:
    OS << "C_MOS (Member of structure)";
    break;
  case XCOFF::StorageClass::C_ARG:
    OS << "C_ARG (Function argument)";
    break;
  case XCOFF::StorageClass::C_STRTAG:
    OS << "C_STRTAG (Structure tag)";
    break;
  case XCOFF::StorageClass::C_MOU:
    OS << "C_MOU (Member of union)";
    break;
  case XCOFF::StorageClass::C_UNTAG:
    OS << "C_UNTAG (Union tag)";
    break;
  case XCOFF::StorageClass::C_TPDEF:
    OS << "C_TPDEF (Type definition)";
    break;
  case XCOFF::StorageClass::C_USTATIC:
    OS << "C_USTATIC (Undefined static)";
    break;
  case XCOFF::StorageClass::C_ENTAG:
    OS << "C_ENTAG (Enumeration tag)";
    break;
  case XCOFF::StorageClass::C_MOE:
    OS << "C_MOE (Member of enumeration)";
    break;
  case XCOFF::StorageClass::C_REGPARM:
    OS << "C_REGPARM (Register parameter)";
    break;
  case XCOFF::StorageClass::C_FIELD:
    OS << "C_FIELD (Bit field)";
    break;
  case XCOFF::StorageClass::C_EOS:
    OS << "C_EOS (End of structure)";
    break;
  case XCOFF::StorageClass::C_LINE:
    OS << "C_LINE";
    break;
  case XCOFF::StorageClass::C_ALIAS:
    OS << "C_ALIAS (Duplicate tag)";
    break;
  case XCOFF::StorageClass::C_HIDDEN:
    OS << "C_HIDDEN (Special storage class for external)";
    break;
  case XCOFF::StorageClass::C_EFCN:
    OS << "C_EFCN (Physical end of function)";
    break;

    // Reserved
  case XCOFF::StorageClass::C_TCSYM:
    OS << "C_TCSYM (Reserved)";
    break;
  }
  return OS;
}

Error XCOFFLinkGraphBuilder::processSections() {
  LLVM_DEBUG(dbgs() << "  Creating graph sections...\n");

  // Create undefined section to contain all external symbols
  UndefSection = &G->createSection("*UND*", orc::MemProt::None);

  for (auto Section : Obj.sections()) {
    LLVM_DEBUG({
      dbgs() << "    section = " << cantFail(Section.getName())
             << ", idx = " << Section.getIndex()
             << ", size = " << format_hex_no_prefix(Section.getSize(), 8)
             << ", vma = " << format_hex(Section.getAddress(), 16) << "\n";
    });

    // We can skip debug (including dawrf) and pad sections
    if (Section.isDebugSection() || cantFail(Section.getName()) == "pad") {
      LLVM_DEBUG(dbgs() << "      skipping...\n");
      continue;
    }

    auto SectionName = cantFail(Section.getName());
    LLVM_DEBUG(dbgs() << "        creating graph section...\n");

    orc::MemProt Prot = orc::MemProt::Read;
    if (Section.isText())
      Prot |= orc::MemProt::Exec;
    if (Section.isData() || Section.isBSS())
      Prot |= orc::MemProt::Write;

    auto *GraphSec = &G->createSection(SectionName, Prot);
    // TODO(HJ): Check for memory lifetime no alloc for certain sections

    SectionDataMap[Section.getIndex()] = Section;
    SectionMap[Section.getIndex()] = GraphSec;
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
  OS << " ";
  debugStorageClass(dbgs(), Sym.getStorageClass());
  OS << " (idx: " << Obj.getSymbolIndex(Sym.getRawDataRefImpl().p) << ")";
  if (Sym.isCsectSymbol()) {
    if (auto ParentSym = getXCOFFSymbolContainingSymbolRef(Obj, Sym)) {
      OS << " (csect idx: "
         << Obj.getSymbolIndex(ParentSym->getRawDataRefImpl().p) << ")";
    }
  }
  OS << "\n";
}

Error XCOFFLinkGraphBuilder::processCsectsAndSymbols() {
  LLVM_DEBUG(dbgs() << "  Creating graph blocks and symbols...\n");

  for (auto [K, V] : SectionMap) {
    LLVM_DEBUG(dbgs() << "    section entry(idx: " << K
                      << " section: " << V->getName() << ")\n");
  }

  for (object::XCOFFSymbolRef Symbol : Obj.symbols()) {
    LLVM_DEBUG({ printSymbolEntry(dbgs(), Obj, Symbol); });

    auto Flags = cantFail(Symbol.getFlags());
    bool External = Flags & object::SymbolRef::SF_Undefined;
    bool Weak = Flags & object::SymbolRef::SF_Weak;
    bool Hidden = Flags & object::SymbolRef::SF_Hidden;
    bool Exported = Flags & object::SymbolRef::SF_Exported;
    bool Absolute = Flags & object::SymbolRef::SF_Absolute;
    bool Global = Flags & object::SymbolRef::SF_Global;

    auto SymbolIndex = Obj.getSymbolIndex(Symbol.getEntryAddress());

    if (External) {
      LLVM_DEBUG(dbgs() << "      created external symbol\n");
      SymbolIdxMap[SymbolIndex] = &G->addExternalSymbol(
          cantFail(Symbol.getName()), Symbol.getSize(), Weak);
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

    bool IsUndefinedSection = !SectionMap.contains(ParentSectionNumber);
    Section *ParentSection =
        !IsUndefinedSection ? SectionMap[ParentSectionNumber] : UndefSection;
    Block *B = nullptr;

    if (!CsectMap.contains(CsectSymbolIndex) && !IsUndefinedSection) {
      auto SectionRef = SectionDataMap[ParentSectionNumber];
      auto Data = SectionRef.getContents();
      if (!Data)
        return Data.takeError();
      ArrayRef<char> SectionBuffer{Data->data(), Data->size()};
      auto Offset =
          cantFail(CsectSymbol.getAddress()) - SectionRef.getAddress();

      LLVM_DEBUG({
        dbgs() << "      symbol entry: offset = " << Offset
               << ", size = " << CsectSymbol.getSize() << ", storage class = ";
        debugStorageClass(dbgs(), CsectSymbol.getStorageClass()) << "\n";
      });

      B = &G->createContentBlock(
          *ParentSection, SectionBuffer.slice(Offset, CsectSymbol.getSize()),
          orc::ExecutorAddr(cantFail(CsectSymbol.getAddress())),
          CsectSymbol.getAlignment(), 0);

      CsectMap[CsectSymbolIndex] = B;
    } else {
      B = CsectMap[CsectSymbolIndex];
    }

    Scope S{Scope::Default};
    if (Hidden)
      S = Scope::Hidden;
    // TODO(HJ): Got this from llvm-objdump.cpp:2938 not sure if its correct
    if (!Weak) {
      if (Global)
        S = Scope::Default;
      else
        S = Scope::Local;
    }

    // TODO(HJ): not sure what is Scope::SideEffectsOnly
    Linkage L = Weak ? Linkage::Weak : Linkage::Strong;
    auto BlockOffset =
        cantFail(Symbol.getAddress()) - B->getAddress().getValue();

    LLVM_DEBUG(dbgs() << "      creating with linkage = " << getLinkageName(L)
                      << ", scope = " << getScopeName(S) << ", B = "
                      << format_hex(B->getAddress().getValue(), 16) << "\n");

    SymbolIdxMap[SymbolIndex] = &G->addDefinedSymbol(
        *B, BlockOffset, cantFail(Symbol.getName()), Symbol.getSize(), L, S,
        cantFail(Symbol.isFunction()), true);
  }

  return Error::success();
}

Error XCOFFLinkGraphBuilder::processRelocations() {
  LLVM_DEBUG(dbgs() << "  Creating relocations...\n");

  for (object::SectionRef Section : Obj.sections()) {

    LLVM_DEBUG(dbgs() << "    Relocations for section "
                      << cantFail(Section.getName()) << ":\n");
    for (object::RelocationRef Relocation : Section.relocations()) {
      SmallString<16> RelocName;
      Relocation.getTypeName(RelocName);
      object::SymbolRef Symbol = *Relocation.getSymbol();
      StringRef TargetSymbol = cantFail(Symbol.getName());
      auto SymbolIndex = Obj.getSymbolIndex(Symbol.getRawDataRefImpl().p);

      LLVM_DEBUG(dbgs() << "      " << format_hex(Relocation.getOffset(), 16)
                        << " (idx: " << SymbolIndex << ")"
                        << " " << RelocName << " " << TargetSymbol << "\n";);

      assert(SymbolIdxMap.contains(SymbolIndex) &&
             "Relocation needs a record in the symbol table");
      auto *S = SymbolIdxMap[SymbolIndex];
      auto It = find_if(G->blocks(),
                        [Target = orc::ExecutorAddr(Section.getAddress() +
                                                    Relocation.getOffset())](
                            const Block *B) -> bool {
                          return B->getRange().contains(Target);
                        });
      assert(It != G->blocks().end() &&
             "Cannot find the target relocation block");
      Block *B = *It;
      LLVM_DEBUG(dbgs() << "        found target relocation block: "
                        << format_hex(B->getAddress().getValue(), 16) << "\n");

      // TODO(HJ): correctly map edge kind and figure out if we need addend
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

  // TODO(HJ): Check for for relocation
  // if (Obj.isRelocatableObject())
  //   return make_error<JITLinkError>("XCOFF object is not relocatable");

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
