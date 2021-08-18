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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#define DEBUG_TYPE "load-binary"

using namespace llvm;
using namespace sampleprof;

cl::opt<bool> ShowDisassemblyOnly("show-disassembly-only", cl::ReallyHidden,
                                  cl::init(false), cl::ZeroOrMore,
                                  cl::desc("Print disassembled code."));

cl::opt<bool> ShowSourceLocations("show-source-locations", cl::ReallyHidden,
                                  cl::init(false), cl::ZeroOrMore,
                                  cl::desc("Print source locations."));

cl::opt<bool> ShowCanonicalFnName("show-canonical-fname", cl::ReallyHidden,
                                  cl::init(false), cl::ZeroOrMore,
                                  cl::desc("Print canonical function name."));

cl::opt<bool> ShowPseudoProbe(
    "show-pseudo-probe", cl::ReallyHidden, cl::init(false), cl::ZeroOrMore,
    cl::desc("Print pseudo probe section and disassembled info."));

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

  // Disassemble the text sections.
  disassemble(Obj);

  // Use function start and return address to infer prolog and epilog
  ProEpilogTracker.inferPrologOffsets(FuncStartAddrMap);
  ProEpilogTracker.inferEpilogOffsets(RetAddrs);

  // TODO: decode other sections.
}

bool ProfiledBinary::inlineContextEqual(uint64_t Address1,
                                        uint64_t Address2) const {
  uint64_t Offset1 = virtualAddrToOffset(Address1);
  uint64_t Offset2 = virtualAddrToOffset(Address2);
  const FrameLocationStack &Context1 = getFrameLocationStack(Offset1);
  const FrameLocationStack &Context2 = getFrameLocationStack(Offset2);
  if (Context1.size() != Context2.size())
    return false;
  if (Context1.empty())
    return false;
  // The leaf frame contains location within the leaf, and it
  // needs to be remove that as it's not part of the calling context
  return std::equal(Context1.begin(), Context1.begin() + Context1.size() - 1,
                    Context2.begin(), Context2.begin() + Context2.size() - 1);
}

std::string
ProfiledBinary::getExpandedContextStr(const SmallVectorImpl<uint64_t> &Stack,
                                      bool &WasLeafInlined) const {
  std::string ContextStr;
  SmallVector<std::string, 16> ContextVec;
  // Process from frame root to leaf
  for (auto Address : Stack) {
    uint64_t Offset = virtualAddrToOffset(Address);
    const FrameLocationStack &ExpandedContext = getFrameLocationStack(Offset);
    // An instruction without a valid debug line will be ignored by sample
    // processing
    if (ExpandedContext.empty())
      return std::string();
    // Set WasLeafInlined to the size of inlined frame count for the last
    // address which is leaf
    WasLeafInlined = (ExpandedContext.size() > 1);
    for (const auto &Loc : ExpandedContext) {
      ContextVec.push_back(getCallSite(Loc));
    }
  }

  assert(ContextVec.size() && "Context length should be at least 1");
  // Compress the context string except for the leaf frame
  std::string LeafFrame = ContextVec.back();
  ContextVec.pop_back();
  CSProfileGenerator::compressRecursionContext<std::string>(ContextVec);
  CSProfileGenerator::trimContext<std::string>(ContextVec);

  std::ostringstream OContextStr;
  for (uint32_t I = 0; I < (uint32_t)ContextVec.size(); I++) {
    if (OContextStr.str().size()) {
      OContextStr << " @ ";
    }
    OContextStr << ContextVec[I];
  }
  // Only keep the function name for the leaf frame
  if (OContextStr.str().size())
    OContextStr << " @ ";
  OContextStr << StringRef(LeafFrame).split(":").first.str();
  return OContextStr.str();
}

template <class ELFT>
void ProfiledBinary::setPreferredTextSegmentAddresses(const ELFFile<ELFT> &Obj, StringRef FileName) {
  const auto &PhdrRange = unwrapOrError(Obj.program_headers(), FileName);
  for (const typename ELFT::Phdr &Phdr : PhdrRange) {
    if ((Phdr.p_type == ELF::PT_LOAD) && (Phdr.p_flags & ELF::PF_X)) {
        // Segments will always be loaded at a page boundary.
        PreferredTextSegmentAddresses.push_back(Phdr.p_vaddr & ~(Phdr.p_align - 1U));
        TextSegmentOffsets.push_back(Phdr.p_offset & ~(Phdr.p_align - 1U));
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

bool ProfiledBinary::dissassembleSymbol(std::size_t SI, ArrayRef<uint8_t> Bytes,
                                        SectionSymbolsTy &Symbols,
                                        const SectionRef &Section) {
  std::size_t SE = Symbols.size();
  uint64_t SectionOffset = Section.getAddress() - getPreferredBaseAddress();
  uint64_t SectSize = Section.getSize();
  uint64_t StartOffset = Symbols[SI].Addr - getPreferredBaseAddress();
  uint64_t EndOffset = (SI + 1 < SE)
                           ? Symbols[SI + 1].Addr - getPreferredBaseAddress()
                           : SectionOffset + SectSize;
  if (StartOffset >= EndOffset)
    return true;

  StringRef SymbolName =
      ShowCanonicalFnName
          ? FunctionSamples::getCanonicalFnName(Symbols[SI].Name)
          : Symbols[SI].Name;
  if (ShowDisassemblyOnly)
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
  while (Offset < EndOffset) {
    MCInst Inst;
    uint64_t Size;
    // Disassemble an instruction.
    bool Disassembled =
        DisAsm->getInstruction(Inst, Size, Bytes.slice(Offset - SectionOffset),
                               Offset + getPreferredBaseAddress(), nulls());
    if (Size == 0)
      Size = 1;

    if (ShowDisassemblyOnly) {
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
        outs() << getReversedLocWithContext(symbolize(IP, ShowCanonicalFnName));
      }
      outs() << "\n";
    }

    if (Disassembled) {
      const MCInstrDesc &MCDesc = MII->get(Inst.getOpcode());
      // Populate a vector of the symbolized callsite at this location
      // We don't need symbolized info for probe-based profile, just use an
      // empty stack as an entry to indicate a valid binary offset
      FrameLocationStack SymbolizedCallStack;
      if (!UsePseudoProbes) {
        InstructionPointer IP(this, Offset);
        SymbolizedCallStack = symbolize(IP, true);
      }
      Offset2LocStackMap[Offset] = SymbolizedCallStack;
      // Populate address maps.
      CodeAddrs.push_back(Offset);
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

  if (ShowDisassemblyOnly)
    outs() << "\n";

  FuncStartAddrMap[StartOffset] = Symbols[SI].Name.str();
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

FrameLocationStack ProfiledBinary::symbolize(const InstructionPointer &IP,
                                             bool UseCanonicalFnName) {
  assert(this == IP.Binary &&
         "Binary should only symbolize its own instruction");
  auto Addr = object::SectionedAddress{IP.Offset + getPreferredBaseAddress(),
                                       object::SectionedAddress::UndefSection};
  DIInliningInfo InlineStack =
      unwrapOrError(Symbolizer->symbolizeInlinedCode(Path, Addr), getName());

  FrameLocationStack CallStack;

  for (int32_t I = InlineStack.getNumberOfFrames() - 1; I >= 0; I--) {
    const auto &CallerFrame = InlineStack.getFrame(I);
    if (CallerFrame.FunctionName == "<invalid>")
      break;
    StringRef FunctionName(CallerFrame.FunctionName);
    if (UseCanonicalFnName)
      FunctionName = FunctionSamples::getCanonicalFnName(FunctionName);
    LineLocation Line(CallerFrame.Line - CallerFrame.StartLine,
                      DILocation::getBaseDiscriminatorFromDiscriminator(
                          CallerFrame.Discriminator,
                          /* IsFSDiscriminator */ false));
    FrameLocation Callsite(FunctionName.str(), Line);
    CallStack.push_back(Callsite);
  }

  return CallStack;
}

InstructionPointer::InstructionPointer(ProfiledBinary *Binary, uint64_t Address,
                                       bool RoundToNext)
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
