//===-- ProfiledBinary.cpp - Binary decoder ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProfiledBinary.h"
#include "ErrorHandling.h"
#include "ProfileGenerator.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/TargetSelect.h"

#define DEBUG_TYPE "load-binary"

using namespace llvm;
using namespace sampleprof;

cl::opt<bool> ShowDisassemblyOnly("show-disassembly-only", cl::init(false),
                                  cl::ZeroOrMore,
                                  cl::desc("Print disassembled code."));

cl::opt<bool> ShowSourceLocations("show-source-locations", cl::init(false),
                                  cl::ZeroOrMore,
                                  cl::desc("Print source locations."));

static cl::opt<bool>
    ShowCanonicalFnName("show-canonical-fname", cl::init(false), cl::ZeroOrMore,
                        cl::desc("Print canonical function name."));

static cl::opt<bool> ShowPseudoProbe(
    "show-pseudo-probe", cl::init(false), cl::ZeroOrMore,
    cl::desc("Print pseudo probe section and disassembled info."));

static cl::opt<bool> UseDwarfCorrelation(
    "use-dwarf-correlation", cl::init(false), cl::ZeroOrMore,
    cl::desc("Use dwarf for profile correlation even when binary contains "
             "pseudo probe."));

static cl::list<std::string> DisassembleFunctions(
    "disassemble-functions", cl::CommaSeparated,
    cl::desc("List of functions to print disassembly for. Accept demangled "
             "names only. Only work with show-disassembly-only"));

namespace llvm {
namespace sampleprof {

static const Target *getTarget(const ObjectFile *Obj) {
  Triple TheTriple = Obj->makeTriple();
  std::string Error;
  std::string ArchName;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(ArchName, TheTriple, Error);
  if (!TheTarget)
    exitWithError(Error, Obj->getFileName());
  return TheTarget;
}

void BinarySizeContextTracker::addInstructionForContext(
    const SampleContextFrameVector &Context, uint32_t InstrSize) {
  ContextTrieNode *CurNode = &RootContext;
  bool IsLeaf = true;
  for (const auto &Callsite : reverse(Context)) {
    StringRef CallerName = Callsite.FuncName;
    LineLocation CallsiteLoc = IsLeaf ? LineLocation(0, 0) : Callsite.Location;
    CurNode = CurNode->getOrCreateChildContext(CallsiteLoc, CallerName);
    IsLeaf = false;
  }

  CurNode->addFunctionSize(InstrSize);
}

uint32_t
BinarySizeContextTracker::getFuncSizeForContext(const SampleContext &Context) {
  ContextTrieNode *CurrNode = &RootContext;
  ContextTrieNode *PrevNode = nullptr;
  SampleContextFrames Frames = Context.getContextFrames();
  int32_t I = Frames.size() - 1;
  Optional<uint32_t> Size;

  // Start from top-level context-less function, traverse down the reverse
  // context trie to find the best/longest match for given context, then
  // retrieve the size.

  while (CurrNode && I >= 0) {
    // Process from leaf function to callers (added to context).
    const auto &ChildFrame = Frames[I--];
    PrevNode = CurrNode;
    CurrNode =
        CurrNode->getChildContext(ChildFrame.Location, ChildFrame.FuncName);
    if (CurrNode && CurrNode->getFunctionSize().hasValue())
      Size = CurrNode->getFunctionSize().getValue();
  }

  // If we traversed all nodes along the path of the context and haven't
  // found a size yet, pivot to look for size from sibling nodes, i.e size
  // of inlinee under different context.
  if (!Size.hasValue()) {
    if (!CurrNode)
      CurrNode = PrevNode;
    while (!Size.hasValue() && CurrNode &&
           !CurrNode->getAllChildContext().empty()) {
      CurrNode = &CurrNode->getAllChildContext().begin()->second;
      if (CurrNode->getFunctionSize().hasValue())
        Size = CurrNode->getFunctionSize().getValue();
    }
  }

  assert(Size.hasValue() && "We should at least find one context size.");
  return Size.getValue();
}

void BinarySizeContextTracker::trackInlineesOptimizedAway(
    MCPseudoProbeDecoder &ProbeDecoder) {
  ProbeFrameStack ProbeContext;
  for (const auto &Child : ProbeDecoder.getDummyInlineRoot().getChildren())
    trackInlineesOptimizedAway(ProbeDecoder, *Child.second.get(), ProbeContext);
}

void BinarySizeContextTracker::trackInlineesOptimizedAway(
    MCPseudoProbeDecoder &ProbeDecoder,
    MCDecodedPseudoProbeInlineTree &ProbeNode, ProbeFrameStack &ProbeContext) {
  StringRef FuncName =
      ProbeDecoder.getFuncDescForGUID(ProbeNode.Guid)->FuncName;
  ProbeContext.emplace_back(FuncName, 0);

  // This ProbeContext has a probe, so it has code before inlining and
  // optimization. Make sure we mark its size as known.
  if (!ProbeNode.getProbes().empty()) {
    ContextTrieNode *SizeContext = &RootContext;
    for (auto &ProbeFrame : reverse(ProbeContext)) {
      StringRef CallerName = ProbeFrame.first;
      LineLocation CallsiteLoc(ProbeFrame.second, 0);
      SizeContext =
          SizeContext->getOrCreateChildContext(CallsiteLoc, CallerName);
    }
    // Add 0 size to make known.
    SizeContext->addFunctionSize(0);
  }

  // DFS down the probe inline tree
  for (const auto &ChildNode : ProbeNode.getChildren()) {
    InlineSite Location = ChildNode.first;
    ProbeContext.back().second = std::get<1>(Location);
    trackInlineesOptimizedAway(ProbeDecoder, *ChildNode.second.get(), ProbeContext);
  }

  ProbeContext.pop_back();
}

void ProfiledBinary::load() {
  // Attempt to open the binary.
  OwningBinary<Binary> OBinary = unwrapOrError(createBinary(Path), Path);
  Binary &Binary = *OBinary.getBinary();

  auto *Obj = dyn_cast<ELFObjectFileBase>(&Binary);
  if (!Obj)
    exitWithError("not a valid Elf image", Path);

  TheTriple = Obj->makeTriple();
  // Current only support X86
  if (!TheTriple.isX86())
    exitWithError("unsupported target", TheTriple.getTriple());
  LLVM_DEBUG(dbgs() << "Loading " << Path << "\n");

  // Find the preferred load address for text sections.
  setPreferredTextSegmentAddresses(Obj);

  // Decode pseudo probe related section
  decodePseudoProbe(Obj);

  // Load debug info of subprograms from DWARF section.
  loadSymbolsFromDWARF(*dyn_cast<ObjectFile>(&Binary));

  // Disassemble the text sections.
  disassemble(Obj);

  // Track size for optimized inlinees when probe is available
  if (UsePseudoProbes && TrackFuncContextSize)
    FuncSizeTracker.trackInlineesOptimizedAway(ProbeDecoder);

  // Use function start and return address to infer prolog and epilog
  ProEpilogTracker.inferPrologOffsets(StartOffset2FuncRangeMap);
  ProEpilogTracker.inferEpilogOffsets(RetAddrs);

  // TODO: decode other sections.
}

bool ProfiledBinary::inlineContextEqual(uint64_t Address1, uint64_t Address2) {
  uint64_t Offset1 = virtualAddrToOffset(Address1);
  uint64_t Offset2 = virtualAddrToOffset(Address2);
  const SampleContextFrameVector &Context1 = getFrameLocationStack(Offset1);
  const SampleContextFrameVector &Context2 = getFrameLocationStack(Offset2);
  if (Context1.size() != Context2.size())
    return false;
  if (Context1.empty())
    return false;
  // The leaf frame contains location within the leaf, and it
  // needs to be remove that as it's not part of the calling context
  return std::equal(Context1.begin(), Context1.begin() + Context1.size() - 1,
                    Context2.begin(), Context2.begin() + Context2.size() - 1);
}

SampleContextFrameVector
ProfiledBinary::getExpandedContext(const SmallVectorImpl<uint64_t> &Stack,
                                   bool &WasLeafInlined) {
  SampleContextFrameVector ContextVec;
  // Process from frame root to leaf
  for (auto Address : Stack) {
    uint64_t Offset = virtualAddrToOffset(Address);
    const SampleContextFrameVector &ExpandedContext =
        getFrameLocationStack(Offset);
    // An instruction without a valid debug line will be ignored by sample
    // processing
    if (ExpandedContext.empty())
      return SampleContextFrameVector();
    // Set WasLeafInlined to the size of inlined frame count for the last
    // address which is leaf
    WasLeafInlined = (ExpandedContext.size() > 1);
    ContextVec.append(ExpandedContext);
  }

  // Replace with decoded base discriminator
  for (auto &Frame : ContextVec) {
    Frame.Location.Discriminator = ProfileGeneratorBase::getBaseDiscriminator(
        Frame.Location.Discriminator);
  }

  assert(ContextVec.size() && "Context length should be at least 1");

  // Compress the context string except for the leaf frame
  auto LeafFrame = ContextVec.back();
  LeafFrame.Location = LineLocation(0, 0);
  ContextVec.pop_back();
  CSProfileGenerator::compressRecursionContext(ContextVec);
  CSProfileGenerator::trimContext(ContextVec);
  ContextVec.push_back(LeafFrame);
  return ContextVec;
}

template <class ELFT>
void ProfiledBinary::setPreferredTextSegmentAddresses(const ELFFile<ELFT> &Obj, StringRef FileName) {
  const auto &PhdrRange = unwrapOrError(Obj.program_headers(), FileName);
  // FIXME: This should be the page size of the system running profiling.
  // However such info isn't available at post-processing time, assuming
  // 4K page now. Note that we don't use EXEC_PAGESIZE from <linux/param.h>
  // because we may build the tools on non-linux.
  uint32_t PageSize = 0x1000;
  for (const typename ELFT::Phdr &Phdr : PhdrRange) {
    if ((Phdr.p_type == ELF::PT_LOAD) && (Phdr.p_flags & ELF::PF_X)) {
        // Segments will always be loaded at a page boundary.
        PreferredTextSegmentAddresses.push_back(Phdr.p_vaddr &
                                                ~(PageSize - 1U));
        TextSegmentOffsets.push_back(Phdr.p_offset & ~(PageSize - 1U));
      }
  }

  if (PreferredTextSegmentAddresses.empty())
    exitWithError("no executable segment found", FileName);
}

void ProfiledBinary::setPreferredTextSegmentAddresses(const ELFObjectFileBase *Obj) {
  if (const auto *ELFObj = dyn_cast<ELF32LEObjectFile>(Obj))
    setPreferredTextSegmentAddresses(ELFObj->getELFFile(), Obj->getFileName());
  else if (const auto *ELFObj = dyn_cast<ELF32BEObjectFile>(Obj))
    setPreferredTextSegmentAddresses(ELFObj->getELFFile(), Obj->getFileName());
  else if (const auto *ELFObj = dyn_cast<ELF64LEObjectFile>(Obj))
    setPreferredTextSegmentAddresses(ELFObj->getELFFile(), Obj->getFileName());
  else if (const auto *ELFObj = cast<ELF64BEObjectFile>(Obj))
    setPreferredTextSegmentAddresses(ELFObj->getELFFile(), Obj->getFileName());
  else
    llvm_unreachable("invalid ELF object format");
}

void ProfiledBinary::decodePseudoProbe(const ELFObjectFileBase *Obj) {
  if (UseDwarfCorrelation)
    return;

  StringRef FileName = Obj->getFileName();
  for (section_iterator SI = Obj->section_begin(), SE = Obj->section_end();
       SI != SE; ++SI) {
    const SectionRef &Section = *SI;
    StringRef SectionName = unwrapOrError(Section.getName(), FileName);

    if (SectionName == ".pseudo_probe_desc") {
      StringRef Contents = unwrapOrError(Section.getContents(), FileName);
      if (!ProbeDecoder.buildGUID2FuncDescMap(
              reinterpret_cast<const uint8_t *>(Contents.data()),
              Contents.size()))
        exitWithError("Pseudo Probe decoder fail in .pseudo_probe_desc section");
    } else if (SectionName == ".pseudo_probe") {
      StringRef Contents = unwrapOrError(Section.getContents(), FileName);
      if (!ProbeDecoder.buildAddress2ProbeMap(
              reinterpret_cast<const uint8_t *>(Contents.data()),
              Contents.size()))
        exitWithError("Pseudo Probe decoder fail in .pseudo_probe section");
      // set UsePseudoProbes flag, used for PerfReader
      UsePseudoProbes = true;
    }
  }

  if (ShowPseudoProbe)
    ProbeDecoder.printGUID2FuncDescMap(outs());
}

void ProfiledBinary::setIsFuncEntry(uint64_t Offset, StringRef RangeSymName) {
  // Note that the start offset of each ELF section can be a non-function
  // symbol, we need to binary search for the start of a real function range.
  auto *FuncRange = findFuncRangeForOffset(Offset);
  // Skip external function symbol.
  if (!FuncRange)
    return;

  // Set IsFuncEntry to ture if the RangeSymName from ELF is equal to its
  // DWARF-based function name.
  if (!FuncRange->IsFuncEntry && FuncRange->getFuncName() == RangeSymName)
    FuncRange->IsFuncEntry = true;
}

bool ProfiledBinary::dissassembleSymbol(std::size_t SI, ArrayRef<uint8_t> Bytes,
                                        SectionSymbolsTy &Symbols,
                                        const SectionRef &Section) {
  std::size_t SE = Symbols.size();
  uint64_t SectionOffset = Section.getAddress() - getPreferredBaseAddress();
  uint64_t SectSize = Section.getSize();
  uint64_t StartOffset = Symbols[SI].Addr - getPreferredBaseAddress();
  uint64_t NextStartOffset =
      (SI + 1 < SE) ? Symbols[SI + 1].Addr - getPreferredBaseAddress()
                    : SectionOffset + SectSize;
  if (StartOffset > NextStartOffset)
    return true;

  StringRef SymbolName =
      ShowCanonicalFnName
          ? FunctionSamples::getCanonicalFnName(Symbols[SI].Name)
          : Symbols[SI].Name;
  bool ShowDisassembly =
      ShowDisassemblyOnly && (DisassembleFunctionSet.empty() ||
                              DisassembleFunctionSet.count(SymbolName));
  if (ShowDisassembly)
    outs() << '<' << SymbolName << ">:\n";

  auto WarnInvalidInsts = [](uint64_t Start, uint64_t End) {
    WithColor::warning() << "Invalid instructions at "
                         << format("%8" PRIx64, Start) << " - "
                         << format("%8" PRIx64, End) << "\n";
  };

  uint64_t Offset = StartOffset;
  // Size of a consecutive invalid instruction range starting from Offset -1
  // backwards.
  uint64_t InvalidInstLength = 0;
  while (Offset < NextStartOffset) {
    MCInst Inst;
    uint64_t Size;
    // Disassemble an instruction.
    bool Disassembled =
        DisAsm->getInstruction(Inst, Size, Bytes.slice(Offset - SectionOffset),
                               Offset + getPreferredBaseAddress(), nulls());
    if (Size == 0)
      Size = 1;

    if (ShowDisassembly) {
      if (ShowPseudoProbe) {
        ProbeDecoder.printProbeForAddress(outs(),
                                          Offset + getPreferredBaseAddress());
      }
      outs() << format("%8" PRIx64 ":", Offset + getPreferredBaseAddress());
      size_t Start = outs().tell();
      if (Disassembled)
        IPrinter->printInst(&Inst, Offset + Size, "", *STI.get(), outs());
      else
        outs() << "\t<unknown>";
      if (ShowSourceLocations) {
        unsigned Cur = outs().tell() - Start;
        if (Cur < 40)
          outs().indent(40 - Cur);
        InstructionPointer IP(this, Offset);
        outs() << getReversedLocWithContext(
            symbolize(IP, ShowCanonicalFnName, ShowPseudoProbe));
      }
      outs() << "\n";
    }

    if (Disassembled) {
      const MCInstrDesc &MCDesc = MII->get(Inst.getOpcode());

      // Record instruction size.
      Offset2InstSizeMap[Offset] = Size;

      // Populate address maps.
      CodeAddrOffsets.push_back(Offset);
      if (MCDesc.isCall())
        CallAddrs.insert(Offset);
      else if (MCDesc.isReturn())
        RetAddrs.insert(Offset);

      if (InvalidInstLength) {
        WarnInvalidInsts(Offset - InvalidInstLength, Offset - 1);
        InvalidInstLength = 0;
      }
    } else {
      InvalidInstLength += Size;
    }

    Offset += Size;
  }

  if (InvalidInstLength)
    WarnInvalidInsts(Offset - InvalidInstLength, Offset - 1);

  if (ShowDisassembly)
    outs() << "\n";

  setIsFuncEntry(StartOffset, Symbols[SI].Name);

  return true;
}

void ProfiledBinary::setUpDisassembler(const ELFObjectFileBase *Obj) {
  const Target *TheTarget = getTarget(Obj);
  std::string TripleName = TheTriple.getTriple();
  StringRef FileName = Obj->getFileName();

  MRI.reset(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    exitWithError("no register info for target " + TripleName, FileName);

  MCTargetOptions MCOptions;
  AsmInfo.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!AsmInfo)
    exitWithError("no assembly info for target " + TripleName, FileName);

  SubtargetFeatures Features = Obj->getFeatures();
  STI.reset(
      TheTarget->createMCSubtargetInfo(TripleName, "", Features.getString()));
  if (!STI)
    exitWithError("no subtarget info for target " + TripleName, FileName);

  MII.reset(TheTarget->createMCInstrInfo());
  if (!MII)
    exitWithError("no instruction info for target " + TripleName, FileName);

  MCContext Ctx(Triple(TripleName), AsmInfo.get(), MRI.get(), STI.get());
  std::unique_ptr<MCObjectFileInfo> MOFI(
      TheTarget->createMCObjectFileInfo(Ctx, /*PIC=*/false));
  Ctx.setObjectFileInfo(MOFI.get());
  DisAsm.reset(TheTarget->createMCDisassembler(*STI, Ctx));
  if (!DisAsm)
    exitWithError("no disassembler for target " + TripleName, FileName);

  MIA.reset(TheTarget->createMCInstrAnalysis(MII.get()));

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  IPrinter.reset(TheTarget->createMCInstPrinter(
      Triple(TripleName), AsmPrinterVariant, *AsmInfo, *MII, *MRI));
  IPrinter->setPrintBranchImmAsAddress(true);
}

void ProfiledBinary::disassemble(const ELFObjectFileBase *Obj) {
  // Set up disassembler and related components.
  setUpDisassembler(Obj);

  // Create a mapping from virtual address to symbol name. The symbols in text
  // sections are the candidates to dissassemble.
  std::map<SectionRef, SectionSymbolsTy> AllSymbols;
  StringRef FileName = Obj->getFileName();
  for (const SymbolRef &Symbol : Obj->symbols()) {
    const uint64_t Addr = unwrapOrError(Symbol.getAddress(), FileName);
    const StringRef Name = unwrapOrError(Symbol.getName(), FileName);
    section_iterator SecI = unwrapOrError(Symbol.getSection(), FileName);
    if (SecI != Obj->section_end())
      AllSymbols[*SecI].push_back(SymbolInfoTy(Addr, Name, ELF::STT_NOTYPE));
  }

  // Sort all the symbols. Use a stable sort to stabilize the output.
  for (std::pair<const SectionRef, SectionSymbolsTy> &SecSyms : AllSymbols)
    stable_sort(SecSyms.second);

  DisassembleFunctionSet.insert(DisassembleFunctions.begin(),
                                DisassembleFunctions.end());
  assert((DisassembleFunctionSet.empty() || ShowDisassemblyOnly) &&
         "Functions to disassemble should be only specified together with "
         "--show-disassembly-only");

  if (ShowDisassemblyOnly)
    outs() << "\nDisassembly of " << FileName << ":\n";

  // Dissassemble a text section.
  for (section_iterator SI = Obj->section_begin(), SE = Obj->section_end();
       SI != SE; ++SI) {
    const SectionRef &Section = *SI;
    if (!Section.isText())
      continue;

    uint64_t ImageLoadAddr = getPreferredBaseAddress();
    uint64_t SectionOffset = Section.getAddress() - ImageLoadAddr;
    uint64_t SectSize = Section.getSize();
    if (!SectSize)
      continue;

    // Register the text section.
    TextSections.insert({SectionOffset, SectSize});

    if (ShowDisassemblyOnly) {
      StringRef SectionName = unwrapOrError(Section.getName(), FileName);
      outs() << "\nDisassembly of section " << SectionName;
      outs() << " [" << format("0x%" PRIx64, Section.getAddress()) << ", "
             << format("0x%" PRIx64, Section.getAddress() + SectSize)
             << "]:\n\n";
    }

    // Get the section data.
    ArrayRef<uint8_t> Bytes =
        arrayRefFromStringRef(unwrapOrError(Section.getContents(), FileName));

    // Get the list of all the symbols in this section.
    SectionSymbolsTy &Symbols = AllSymbols[Section];

    // Disassemble symbol by symbol.
    for (std::size_t SI = 0, SE = Symbols.size(); SI != SE; ++SI) {
      if (!dissassembleSymbol(SI, Bytes, Symbols, Section))
        exitWithError("disassembling error", FileName);
    }
  }
}

void ProfiledBinary::loadSymbolsFromDWARF(ObjectFile &Obj) {
  auto DebugContext = llvm::DWARFContext::create(Obj);
  if (!DebugContext)
    exitWithError("Misssing debug info.", Path);

  for (const auto &CompilationUnit : DebugContext->compile_units()) {
    for (const auto &DieInfo : CompilationUnit->dies()) {
      llvm::DWARFDie Die(CompilationUnit.get(), &DieInfo);

      if (!Die.isSubprogramDIE())
        continue;
      auto Name = Die.getName(llvm::DINameKind::LinkageName);
      if (!Name)
        Name = Die.getName(llvm::DINameKind::ShortName);
      if (!Name)
        continue;

      auto RangesOrError = Die.getAddressRanges();
      if (!RangesOrError)
        continue;
      const DWARFAddressRangesVector &Ranges = RangesOrError.get();

      if (Ranges.empty())
        continue;

      // Different DWARF symbols can have same function name, search or create
      // BinaryFunction indexed by the name.
      auto Ret = BinaryFunctions.emplace(Name, BinaryFunction());
      auto &Func = Ret.first->second;
      if (Ret.second)
        Func.FuncName = Ret.first->first;

      for (const auto &Range : Ranges) {
        uint64_t FuncStart = Range.LowPC;
        uint64_t FuncSize = Range.HighPC - FuncStart;

        if (FuncSize == 0 || FuncStart < getPreferredBaseAddress())
          continue;

        uint64_t StartOffset = FuncStart - getPreferredBaseAddress();
        uint64_t EndOffset = Range.HighPC - getPreferredBaseAddress();

        // We may want to know all ranges for one function. Here group the
        // ranges and store them into BinaryFunction.
        Func.Ranges.emplace_back(StartOffset, EndOffset);

        auto R = StartOffset2FuncRangeMap.emplace(StartOffset, FuncRange());
        if (R.second) {
          FuncRange &FRange = R.first->second;
          FRange.Func = &Func;
          FRange.StartOffset = StartOffset;
          FRange.EndOffset = EndOffset;
        } else {
          WithColor::warning()
              << "Duplicated symbol start address at "
              << format("%8" PRIx64, StartOffset + getPreferredBaseAddress())
              << " " << R.first->second.getFuncName() << " and " << Name
              << "\n";
        }
      }
    }
  }
  assert(!StartOffset2FuncRangeMap.empty() && "Misssing debug info.");
}

void ProfiledBinary::populateSymbolListFromDWARF(
    ProfileSymbolList &SymbolList) {
  for (auto &I : StartOffset2FuncRangeMap)
    SymbolList.add(I.second.getFuncName());
}

void ProfiledBinary::setupSymbolizer() {
  symbolize::LLVMSymbolizer::Options SymbolizerOpts;
  SymbolizerOpts.PrintFunctions =
      DILineInfoSpecifier::FunctionNameKind::LinkageName;
  SymbolizerOpts.Demangle = false;
  SymbolizerOpts.DefaultArch = TheTriple.getArchName().str();
  SymbolizerOpts.UseSymbolTable = false;
  SymbolizerOpts.RelativeAddresses = false;
  Symbolizer = std::make_unique<symbolize::LLVMSymbolizer>(SymbolizerOpts);
}

SampleContextFrameVector ProfiledBinary::symbolize(const InstructionPointer &IP,
                                                   bool UseCanonicalFnName,
                                                   bool UseProbeDiscriminator) {
  assert(this == IP.Binary &&
         "Binary should only symbolize its own instruction");
  auto Addr = object::SectionedAddress{IP.Offset + getPreferredBaseAddress(),
                                       object::SectionedAddress::UndefSection};
  DIInliningInfo InlineStack =
      unwrapOrError(Symbolizer->symbolizeInlinedCode(Path, Addr), getName());

  SampleContextFrameVector CallStack;
  for (int32_t I = InlineStack.getNumberOfFrames() - 1; I >= 0; I--) {
    const auto &CallerFrame = InlineStack.getFrame(I);
    if (CallerFrame.FunctionName == "<invalid>")
      break;

    StringRef FunctionName(CallerFrame.FunctionName);
    if (UseCanonicalFnName)
      FunctionName = FunctionSamples::getCanonicalFnName(FunctionName);

    uint32_t Discriminator = CallerFrame.Discriminator;
    uint32_t LineOffset = CallerFrame.Line - CallerFrame.StartLine;
    if (UseProbeDiscriminator) {
      LineOffset =
          PseudoProbeDwarfDiscriminator::extractProbeIndex(Discriminator);
      Discriminator = 0;
    } else {
      // Filter out invalid negative(int type) lineOffset
      if (LineOffset & 0xffff0000)
        return SampleContextFrameVector();
    }

    LineLocation Line(LineOffset, Discriminator);
    auto It = NameStrings.insert(FunctionName.str());
    CallStack.emplace_back(*It.first, Line);
  }

  return CallStack;
}

void ProfiledBinary::computeInlinedContextSizeForRange(uint64_t StartOffset,
                                                       uint64_t EndOffset) {
  uint32_t Index = getIndexForOffset(StartOffset);
  if (CodeAddrOffsets[Index] != StartOffset)
    WithColor::warning() << "Invalid start instruction at "
                         << format("%8" PRIx64, StartOffset) << "\n";

  uint64_t Offset = CodeAddrOffsets[Index];
  while (Offset < EndOffset) {
    const SampleContextFrameVector &SymbolizedCallStack =
        getFrameLocationStack(Offset, UsePseudoProbes);
    uint64_t Size = Offset2InstSizeMap[Offset];

    // Record instruction size for the corresponding context
    FuncSizeTracker.addInstructionForContext(SymbolizedCallStack, Size);

    Offset = CodeAddrOffsets[++Index];
  }
}

InstructionPointer::InstructionPointer(const ProfiledBinary *Binary,
                                       uint64_t Address, bool RoundToNext)
    : Binary(Binary), Address(Address) {
  Index = Binary->getIndexForAddr(Address);
  if (RoundToNext) {
    // we might get address which is not the code
    // it should round to the next valid address
    this->Address = Binary->getAddressforIndex(Index);
  }
}

void InstructionPointer::advance() {
  Index++;
  Address = Binary->getAddressforIndex(Index);
}

void InstructionPointer::backward() {
  Index--;
  Address = Binary->getAddressforIndex(Index);
}

void InstructionPointer::update(uint64_t Addr) {
  Address = Addr;
  Index = Binary->getIndexForAddr(Address);
}

} // end namespace sampleprof
} // end namespace llvm
