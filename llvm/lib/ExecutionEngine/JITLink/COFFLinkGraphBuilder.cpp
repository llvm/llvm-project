//=--------- COFFLinkGraphBuilder.cpp - COFF LinkGraph builder ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic COFF LinkGraph buliding code.
//
//===----------------------------------------------------------------------===//
#include "COFFLinkGraphBuilder.h"

#define DEBUG_TYPE "jitlink"

static const char *CommonSectionName = "__common";

namespace llvm {
namespace jitlink {

COFFLinkGraphBuilder::COFFLinkGraphBuilder(
    const object::COFFObjectFile &Obj, Triple TT,
    LinkGraph::GetEdgeKindNameFunction GetEdgeKindName)
    : Obj(Obj),
      G(std::make_unique<LinkGraph>(
          Obj.getFileName().str(), Triple(std::move(TT)), getPointerSize(Obj),
          getEndianness(Obj), std::move(GetEdgeKindName))) {
  LLVM_DEBUG({
    dbgs() << "Created COFFLinkGraphBuilder for \"" << Obj.getFileName()
           << "\"\n";
  });
}

COFFLinkGraphBuilder::~COFFLinkGraphBuilder() = default;

unsigned
COFFLinkGraphBuilder::getPointerSize(const object::COFFObjectFile &Obj) {
  return Obj.getBytesInAddress();
}

support::endianness
COFFLinkGraphBuilder::getEndianness(const object::COFFObjectFile &Obj) {
  return Obj.isLittleEndian() ? support::little : support::big;
}

uint64_t COFFLinkGraphBuilder::getSectionSize(const object::COFFObjectFile &Obj,
                                              const object::coff_section *Sec) {
  // Consider the difference between executable form and object form.
  // More information is inside COFFObjectFile::getSectionSize
  if (Obj.getDOSHeader())
    return std::min(Sec->VirtualSize, Sec->SizeOfRawData);
  return Sec->SizeOfRawData;
}

uint64_t
COFFLinkGraphBuilder::getSectionAddress(const object::COFFObjectFile &Obj,
                                        const object::coff_section *Section) {
  return Section->VirtualAddress + Obj.getImageBase();
}

bool COFFLinkGraphBuilder::isComdatSection(
    const object::coff_section *Section) {
  return Section->Characteristics & COFF::IMAGE_SCN_LNK_COMDAT;
}

Section &COFFLinkGraphBuilder::getCommonSection() {
  if (!CommonSection)
    CommonSection =
        &G->createSection(CommonSectionName, MemProt::Read | MemProt::Write);
  return *CommonSection;
}

Expected<std::unique_ptr<LinkGraph>> COFFLinkGraphBuilder::buildGraph() {
  if (!Obj.isRelocatableObject())
    return make_error<JITLinkError>("Object is not a relocatable COFF file");

  if (auto Err = graphifySections())
    return std::move(Err);

  if (auto Err = graphifySymbols())
    return std::move(Err);

  if (auto Err = addRelocations())
    return std::move(Err);

  return std::move(G);
}

StringRef
COFFLinkGraphBuilder::getCOFFSectionName(COFFSectionIndex SectionIndex,
                                         const object::coff_section *Sec,
                                         object::COFFSymbolRef Sym) {
  switch (SectionIndex) {
  case COFF::IMAGE_SYM_UNDEFINED: {
    if (Sym.getValue())
      return "(common)";
    else
      return "(external)";
  }
  case COFF::IMAGE_SYM_ABSOLUTE:
    return "(absolute)";
  case COFF::IMAGE_SYM_DEBUG: {
    // Used with .file symbol
    return "(debug)";
  }
  default: {
    // Non reserved regular section numbers
    if (Expected<StringRef> SecNameOrErr = Obj.getSectionName(Sec))
      return *SecNameOrErr;
  }
  }
  return "";
}

Error COFFLinkGraphBuilder::graphifySections() {
  LLVM_DEBUG(dbgs() << "  Creating graph sections...\n");

  GraphBlocks.resize(Obj.getNumberOfSections() + 1);
  // For each section...
  for (COFFSectionIndex SecIndex = 1;
       SecIndex <= static_cast<COFFSectionIndex>(Obj.getNumberOfSections());
       SecIndex++) {
    Expected<const object::coff_section *> Sec = Obj.getSection(SecIndex);
    if (!Sec)
      return Sec.takeError();

    StringRef SectionName;
    if (Expected<StringRef> SecNameOrErr = Obj.getSectionName(*Sec))
      SectionName = *SecNameOrErr;

    bool IsDiscardable =
        (*Sec)->Characteristics &
        (COFF::IMAGE_SCN_MEM_DISCARDABLE | COFF::IMAGE_SCN_LNK_INFO);
    if (IsDiscardable) {
      LLVM_DEBUG(dbgs() << "    " << SecIndex << ": \"" << SectionName
                        << "\" is discardable: "
                           "No graph section will be created.\n");
      continue;
    }

    // FIXME: Skip debug info sections

    LLVM_DEBUG({
      dbgs() << "    "
             << "Creating section for \"" << SectionName << "\"\n";
    });

    // Get the section's memory protection flags.
    MemProt Prot = MemProt::None;
    if ((*Sec)->Characteristics & COFF::IMAGE_SCN_MEM_EXECUTE)
      Prot |= MemProt::Exec;
    if ((*Sec)->Characteristics & COFF::IMAGE_SCN_MEM_READ)
      Prot |= MemProt::Read;
    if ((*Sec)->Characteristics & COFF::IMAGE_SCN_MEM_WRITE)
      Prot |= MemProt::Write;

    // Look for existing sections first.
    auto *GraphSec = G->findSectionByName(SectionName);
    if (!GraphSec)
      GraphSec = &G->createSection(SectionName, Prot);
    if (GraphSec->getMemProt() != Prot)
      return make_error<JITLinkError>("MemProt should match");

    Block *B = nullptr;
    if ((*Sec)->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
      B = &G->createZeroFillBlock(
          *GraphSec, getSectionSize(Obj, *Sec),
          orc::ExecutorAddr(getSectionAddress(Obj, *Sec)),
          (*Sec)->getAlignment(), 0);
    else {
      ArrayRef<uint8_t> Data;
      if (auto Err = Obj.getSectionContents(*Sec, Data))
        return Err;

      B = &G->createContentBlock(
          *GraphSec,
          ArrayRef<char>(reinterpret_cast<const char *>(Data.data()),
                         Data.size()),
          orc::ExecutorAddr(getSectionAddress(Obj, *Sec)),
          (*Sec)->getAlignment(), 0);
    }

    setGraphBlock(SecIndex, B);
  }

  return Error::success();
}

Error COFFLinkGraphBuilder::graphifySymbols() {
  LLVM_DEBUG(dbgs() << "  Creating graph symbols...\n");

  SymbolSets.resize(Obj.getNumberOfSections() + 1);
  GraphSymbols.resize(Obj.getNumberOfSymbols());

  for (COFFSymbolIndex SymIndex = 0;
       SymIndex < static_cast<COFFSymbolIndex>(Obj.getNumberOfSymbols());
       SymIndex++) {
    Expected<object::COFFSymbolRef> Sym = Obj.getSymbol(SymIndex);
    if (!Sym)
      return Sym.takeError();

    StringRef SymbolName;
    if (Expected<StringRef> SymNameOrErr = Obj.getSymbolName(*Sym))
      SymbolName = *SymNameOrErr;

    COFFSectionIndex SectionIndex = Sym->getSectionNumber();
    const object::coff_section *Sec = nullptr;

    if (!COFF::isReservedSectionNumber(SectionIndex)) {
      auto SecOrErr = Obj.getSection(SectionIndex);
      if (!SecOrErr)
        return make_error<JITLinkError>(
            "Invalid COFF section number:" + formatv("{0:d}: ", SectionIndex) +
            " (" + toString(SecOrErr.takeError()) + ")");
      Sec = *SecOrErr;
    }

    // Create jitlink symbol
    jitlink::Symbol *GSym = nullptr;
    if (Sym->isFileRecord())
      LLVM_DEBUG({
        dbgs() << "    " << SymIndex << ": Skipping FileRecord symbol \""
               << SymbolName << "\" in "
               << getCOFFSectionName(SectionIndex, Sec, *Sym)
               << " (index: " << SectionIndex << ") \n";
      });
    else if (Sym->isUndefined()) {
      LLVM_DEBUG({
        dbgs() << "    " << SymIndex
               << ": Creating external graph symbol for COFF symbol \""
               << SymbolName << "\" in "
               << getCOFFSectionName(SectionIndex, Sec, *Sym)
               << " (index: " << SectionIndex << ") \n";
      });
      GSym =
          &G->addExternalSymbol(SymbolName, Sym->getValue(), Linkage::Strong);
    } else if (Sym->isWeakExternal()) {
      COFFSymbolIndex TagIndex =
          Sym->getAux<object::coff_aux_weak_external>()->TagIndex;
      assert(Sym->getAux<object::coff_aux_weak_external>()->Characteristics !=
                 COFF::IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY &&
             "IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY is not supported.");
      assert(Sym->getAux<object::coff_aux_weak_external>()->Characteristics !=
                 COFF::IMAGE_WEAK_EXTERN_SEARCH_LIBRARY &&
             "IMAGE_WEAK_EXTERN_SEARCH_LIBRARY is not supported.");
      WeakAliasRequests.push_back({SymIndex, TagIndex, SymbolName});
    } else {
      Expected<jitlink::Symbol *> NewGSym =
          createDefinedSymbol(SymIndex, SymbolName, *Sym, Sec);
      if (!NewGSym)
        return NewGSym.takeError();
      GSym = *NewGSym;
      if (GSym) {
        LLVM_DEBUG({
          dbgs() << "    " << SymIndex
                 << ": Creating defined graph symbol for COFF symbol \""
                 << SymbolName << "\" in "
                 << getCOFFSectionName(SectionIndex, Sec, *Sym)
                 << " (index: " << SectionIndex << ") \n";
          dbgs() << "      " << *GSym << "\n";
        });
      }
    }

    // Register the symbol
    if (GSym)
      setGraphSymbol(SectionIndex, SymIndex, *GSym);
    SymIndex += Sym->getNumberOfAuxSymbols();
  }

  if (auto Err = flushWeakAliasRequests())
    return Err;

  if (auto Err = calculateImplicitSizeOfSymbols())
    return Err;

  return Error::success();
}

Error COFFLinkGraphBuilder::flushWeakAliasRequests() {
  // Export the weak external symbols and alias it
  for (auto &WeakAlias : WeakAliasRequests) {
    if (auto *Target = getGraphSymbol(WeakAlias.Target)) {
      Expected<object::COFFSymbolRef> AliasSymbol =
          Obj.getSymbol(WeakAlias.Alias);
      if (!AliasSymbol)
        return AliasSymbol.takeError();

      // FIXME: Support this when there's a way to handle this.
      if (!Target->isDefined())
        return make_error<JITLinkError>("Weak external symbol with external "
                                        "symbol as alternative not supported.");

      jitlink::Symbol *NewSymbol = &G->addDefinedSymbol(
          Target->getBlock(), Target->getOffset(), WeakAlias.SymbolName,
          Target->getSize(), Linkage::Weak, Scope::Default,
          Target->isCallable(), false);
      setGraphSymbol(AliasSymbol->getSectionNumber(), WeakAlias.Alias,
                     *NewSymbol);
      LLVM_DEBUG({
        dbgs() << "    " << WeakAlias.Alias
               << ": Creating weak external symbol for COFF symbol \""
               << WeakAlias.SymbolName << "\" in section "
               << AliasSymbol->getSectionNumber() << "\n";
        dbgs() << "      " << *NewSymbol << "\n";
      });
    } else
      return make_error<JITLinkError>("Weak symbol alias requested but actual "
                                      "symbol not found for symbol " +
                                      formatv("{0:d}", WeakAlias.Alias));
  }
  return Error::success();
}

// In COFF, most of the defined symbols don't contain the size information.
// Hence, we calculate the "implicit" size of symbol by taking the delta of
// offsets of consecutive symbols within a block. We maintain a balanced tree
// set of symbols sorted by offset per each block in order to achieve
// logarithmic time complexity of sorted symbol insertion. Symbol is inserted to
// the set once it's processed in graphifySymbols. In this function, we iterate
// each collected symbol in sorted order and calculate the implicit size.
Error COFFLinkGraphBuilder::calculateImplicitSizeOfSymbols() {
  for (COFFSectionIndex SecIndex = 1;
       SecIndex <= static_cast<COFFSectionIndex>(Obj.getNumberOfSections());
       SecIndex++) {
    auto &SymbolSet = SymbolSets[SecIndex];
    jitlink::Block *B = getGraphBlock(SecIndex);
    orc::ExecutorAddrDiff LastOffset = B->getSize();
    orc::ExecutorAddrDiff LastDifferentOffset = B->getSize();
    orc::ExecutorAddrDiff LastSize = 0;
    for (auto It = SymbolSet.rbegin(); It != SymbolSet.rend(); It++) {
      orc::ExecutorAddrDiff Offset = It->first;
      jitlink::Symbol *Symbol = It->second;
      orc::ExecutorAddrDiff CandSize;
      // Last offset can be same when aliasing happened
      if (Symbol->getOffset() == LastOffset)
        CandSize = LastSize;
      else
        CandSize = LastOffset - Offset;

      LLVM_DEBUG({
        if (Offset + Symbol->getSize() > LastDifferentOffset)
          dbgs() << "  Overlapping symbol range generated for the following "
                    "symbol:"
                 << "\n"
                 << "    " << *Symbol << "\n";
      });
      (void)LastDifferentOffset;
      if (LastOffset != Offset)
        LastDifferentOffset = Offset;
      LastSize = CandSize;
      LastOffset = Offset;
      if (Symbol->getSize()) {
        // Non empty symbol can happen in COMDAT symbol.
        // We don't consider the possibility of overlapping symbol range that
        // could be introduced by disparity between inferred symbol size and
        // defined symbol size because symbol size information is currently only
        // used by jitlink-check where we have control to not make overlapping
        // ranges.
        continue;
      }

      LLVM_DEBUG({
        if (!CandSize)
          dbgs() << "  Empty implicit symbol size generated for the following "
                    "symbol:"
                 << "\n"
                 << "    " << *Symbol << "\n";
      });

      Symbol->setSize(CandSize);
    }
  }
  return Error::success();
}

Expected<Symbol *> COFFLinkGraphBuilder::createDefinedSymbol(
    COFFSymbolIndex SymIndex, StringRef SymbolName,
    object::COFFSymbolRef Symbol, const object::coff_section *Section) {
  if (Symbol.isCommon()) {
    // FIXME: correct alignment
    return &G->addCommonSymbol(SymbolName, Scope::Default, getCommonSection(),
                               orc::ExecutorAddr(), Symbol.getValue(),
                               Symbol.getValue(), false);
  }
  if (Symbol.isAbsolute())
    return &G->addAbsoluteSymbol(SymbolName,
                                 orc::ExecutorAddr(Symbol.getValue()), 0,
                                 Linkage::Strong, Scope::Local, false);

  if (llvm::COFF::isReservedSectionNumber(Symbol.getSectionNumber()))
    return make_error<JITLinkError>(
        "Reserved section number used in regular symbol " +
        formatv("{0:d}", SymIndex));

  Block *B = getGraphBlock(Symbol.getSectionNumber());
  if (Symbol.isExternal()) {
    // This is not a comdat sequence, export the symbol as it is
    if (!isComdatSection(Section))
      return &G->addDefinedSymbol(
          *B, Symbol.getValue(), SymbolName, 0, Linkage::Strong, Scope::Default,
          Symbol.getComplexType() == COFF::IMAGE_SYM_DTYPE_FUNCTION, false);
    else {
      if (!PendingComdatExport)
        return make_error<JITLinkError>("No pending COMDAT export for symbol " +
                                        formatv("{0:d}", SymIndex));
      if (PendingComdatExport->SectionIndex != Symbol.getSectionNumber())
        return make_error<JITLinkError>(
            "COMDAT export section number mismatch for symbol " +
            formatv("{0:d}", SymIndex));
      return exportCOMDATSymbol(SymIndex, SymbolName, Symbol);
    }
  }

  if (Symbol.getStorageClass() == COFF::IMAGE_SYM_CLASS_STATIC) {
    const object::coff_aux_section_definition *Definition =
        Symbol.getSectionDefinition();
    if (!Definition || !isComdatSection(Section)) {
      // Handle typical static symbol
      return &G->addDefinedSymbol(
          *B, Symbol.getValue(), SymbolName, 0, Linkage::Strong, Scope::Local,
          Symbol.getComplexType() == COFF::IMAGE_SYM_DTYPE_FUNCTION, false);
    }
    if (Definition->Selection == COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE) {
      auto Target = Definition->getNumber(Symbol.isBigObj());
      auto GSym = &G->addDefinedSymbol(
          *B, Symbol.getValue(), SymbolName, 0, Linkage::Strong, Scope::Local,
          Symbol.getComplexType() == COFF::IMAGE_SYM_DTYPE_FUNCTION, false);
      getGraphBlock(Target)->addEdge(Edge::KeepAlive, 0, *GSym, 0);
      return GSym;
    }
    if (PendingComdatExport)
      return make_error<JITLinkError>(
          "COMDAT export request already exists before symbol " +
          formatv("{0:d}", SymIndex));
    return createCOMDATExportRequest(SymIndex, Symbol, Definition);
  }
  return make_error<JITLinkError>("Unsupported storage class " +
                                  formatv("{0:d}", Symbol.getStorageClass()) +
                                  " in symbol " + formatv("{0:d}", SymIndex));
}

// COMDAT handling:
// When IMAGE_SCN_LNK_COMDAT flag is set in the flags of a section,
// the section is called a COMDAT section. It contains two symbols
// in a sequence that specifes the behavior. First symbol is the section
// symbol which contains the size and name of the section. It also contains
// selection type that specifies how duplicate of the symbol is handled.
// Second symbol is COMDAT symbol which usually defines the external name and
// data type.
//
// Since two symbols always come in a specific order, we initiate pending COMDAT
// export request when we encounter the first symbol and actually exports it
// when we process the second symbol.
//
// Process the first symbol of COMDAT sequence.
Expected<Symbol *> COFFLinkGraphBuilder::createCOMDATExportRequest(
    COFFSymbolIndex SymIndex, object::COFFSymbolRef Symbol,
    const object::coff_aux_section_definition *Definition) {
  Block *B = getGraphBlock(Symbol.getSectionNumber());
  Linkage L = Linkage::Strong;
  switch (Definition->Selection) {
  case COFF::IMAGE_COMDAT_SELECT_NODUPLICATES: {
    L = Linkage::Strong;
    break;
  }
  case COFF::IMAGE_COMDAT_SELECT_ANY: {
    L = Linkage::Weak;
    break;
  }
  case COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH:
  case COFF::IMAGE_COMDAT_SELECT_SAME_SIZE: {
    // FIXME: Implement size/content validation when LinkGraph is able to
    // handle this.
    L = Linkage::Weak;
    break;
  }
  case COFF::IMAGE_COMDAT_SELECT_LARGEST: {
    // FIXME: Support IMAGE_COMDAT_SELECT_LARGEST when LinkGraph is able to
    // handle this.
    return make_error<JITLinkError>(
        "IMAGE_COMDAT_SELECT_LARGEST is not supported.");
  }
  case COFF::IMAGE_COMDAT_SELECT_NEWEST: {
    // Even link.exe doesn't support this selection properly.
    return make_error<JITLinkError>(
        "IMAGE_COMDAT_SELECT_NEWEST is not supported.");
  }
  default: {
    return make_error<JITLinkError>("Invalid comdat selection type: " +
                                    formatv("{0:d}", Definition->Selection));
  }
  }
  PendingComdatExport = {SymIndex, Symbol.getSectionNumber(), L};
  return &G->addAnonymousSymbol(*B, Symbol.getValue(), Definition->Length,
                                false, false);
}

// Process the second symbol of COMDAT sequence.
Expected<Symbol *>
COFFLinkGraphBuilder::exportCOMDATSymbol(COFFSymbolIndex SymIndex,
                                         StringRef SymbolName,
                                         object::COFFSymbolRef Symbol) {
  COFFSymbolIndex TargetIndex = PendingComdatExport->SymbolIndex;
  Linkage L = PendingComdatExport->Linkage;
  jitlink::Symbol *Target = getGraphSymbol(TargetIndex);
  assert(Target && "COMDAT leaader is invalid.");
  assert((llvm::count_if(G->defined_symbols(),
                         [&](const jitlink::Symbol *Sym) {
                           return Sym->getName() == SymbolName;
                         }) == 0) &&
         "Duplicate defined symbol");
  Target->setName(SymbolName);
  Target->setLinkage(L);
  Target->setCallable(Symbol.getComplexType() ==
                      COFF::IMAGE_SYM_DTYPE_FUNCTION);
  Target->setScope(Scope::Default);
  LLVM_DEBUG({
    dbgs() << "    " << SymIndex
           << ": Exporting COMDAT graph symbol for COFF symbol \"" << SymbolName
           << "\" in section " << Symbol.getSectionNumber() << "\n";
    dbgs() << "      " << *Target << "\n";
  });
  PendingComdatExport = None;
  return Target;
}

} // namespace jitlink
} // namespace llvm
