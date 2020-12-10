//===-- ProfiledBinary.cpp - Binary decoder ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProfiledBinary.h"
#include "ErrorHandling.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#define DEBUG_TYPE "load-binary"

using namespace llvm;
using namespace sampleprof;

static cl::opt<bool> ShowDisassembly("show-disassembly", cl::ReallyHidden,
                                     cl::init(false), cl::ZeroOrMore,
                                     cl::desc("Print disassembled code."));

static cl::opt<bool> ShowSourceLocations("show-source-locations",
                                         cl::ReallyHidden, cl::init(false),
                                         cl::ZeroOrMore,
                                         cl::desc("Print source locations."));

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

template <class ELFT>
static uint64_t getELFImageLMAForSec(const ELFFile<ELFT> &Obj,
                                     const object::ELFSectionRef &Sec,
                                     StringRef FileName) {
  // Search for a PT_LOAD segment containing the requested section. Return this
  // segment's p_addr as the image load address for the section.
  const auto &PhdrRange = unwrapOrError(Obj.program_headers(), FileName);
  for (const typename ELFT::Phdr &Phdr : PhdrRange)
    if ((Phdr.p_type == ELF::PT_LOAD) && (Phdr.p_vaddr <= Sec.getAddress()) &&
        (Phdr.p_vaddr + Phdr.p_memsz > Sec.getAddress()))
      // Segments will always be loaded at a page boundary.
      return Phdr.p_paddr & ~(Phdr.p_align - 1U);
  return 0;
}

// Get the image load address for a specific section. Note that an image is
// loaded by segments (a group of sections) and segments may not be consecutive
// in memory.
static uint64_t getELFImageLMAForSec(const object::ELFSectionRef &Sec) {
  if (const auto *ELFObj = dyn_cast<ELF32LEObjectFile>(Sec.getObject()))
    return getELFImageLMAForSec(ELFObj->getELFFile(), Sec,
                                ELFObj->getFileName());
  else if (const auto *ELFObj = dyn_cast<ELF32BEObjectFile>(Sec.getObject()))
    return getELFImageLMAForSec(ELFObj->getELFFile(), Sec,
                                ELFObj->getFileName());
  else if (const auto *ELFObj = dyn_cast<ELF64LEObjectFile>(Sec.getObject()))
    return getELFImageLMAForSec(ELFObj->getELFFile(), Sec,
                                ELFObj->getFileName());
  const auto *ELFObj = cast<ELF64BEObjectFile>(Sec.getObject());
  return getELFImageLMAForSec(ELFObj->getELFFile(), Sec, ELFObj->getFileName());
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

  // Find the preferred base address for text sections.
  setPreferredBaseAddress(Obj);

  // Disassemble the text sections.
  disassemble(Obj);

  // Use function start and return address to infer prolog and epilog
  ProEpilogTracker.inferPrologOffsets(FuncStartAddrMap);
  ProEpilogTracker.inferEpilogOffsets(RetAddrs);

  // TODO: decode other sections.

  return;
}

bool ProfiledBinary::inlineContextEqual(uint64_t Address1,
                                        uint64_t Address2) const {
  uint64_t Offset1 = virtualAddrToOffset(Address1);
  uint64_t Offset2 = virtualAddrToOffset(Address2);
  const FrameLocationStack &Context1 = getFrameLocationStack(Offset1);
  const FrameLocationStack &Context2 = getFrameLocationStack(Offset2);
  if (Context1.size() != Context2.size())
    return false;

  // The leaf frame contains location within the leaf, and it
  // needs to be remove that as it's not part of the calling context
  return std::equal(Context1.begin(), Context1.begin() + Context1.size() - 1,
                    Context2.begin(), Context2.begin() + Context2.size() - 1);
}

std::string
ProfiledBinary::getExpandedContextStr(const std::list<uint64_t> &Stack) const {
  std::string ContextStr;
  SmallVector<std::string, 8> ContextVec;
  // Process from frame root to leaf
  for (auto Iter = Stack.rbegin(); Iter != Stack.rend(); Iter++) {
    uint64_t Offset = virtualAddrToOffset(*Iter);
    const FrameLocationStack &ExpandedContext = getFrameLocationStack(Offset);
    for (const auto &Loc : ExpandedContext) {
      ContextVec.push_back(getCallSite(Loc));
    }
  }

  assert(ContextVec.size() && "Context length should be at least 1");

  std::ostringstream OContextStr;
  for (uint32_t I = 0; I < (uint32_t)ContextVec.size(); I++) {
    if (OContextStr.str().size()) {
      OContextStr << " @ ";
    }

    if (I == ContextVec.size() - 1) {
      // Only keep the function name for the leaf frame
      StringRef Ref(ContextVec[I]);
      OContextStr << Ref.split(":").first.str();
    } else {
      OContextStr << ContextVec[I];
    }
  }

  return OContextStr.str();
}

void ProfiledBinary::setPreferredBaseAddress(const ELFObjectFileBase *Obj) {
  for (section_iterator SI = Obj->section_begin(), SE = Obj->section_end();
       SI != SE; ++SI) {
    const SectionRef &Section = *SI;
    if (Section.isText()) {
      PreferredBaseAddress = getELFImageLMAForSec(Section);
      return;
    }
  }
  exitWithError("no text section found", Obj->getFileName());
}

bool ProfiledBinary::dissassembleSymbol(std::size_t SI, ArrayRef<uint8_t> Bytes,
                                        SectionSymbolsTy &Symbols,
                                        const SectionRef &Section) {

  std::size_t SE = Symbols.size();
  uint64_t SectionOffset = Section.getAddress() - PreferredBaseAddress;
  uint64_t SectSize = Section.getSize();
  uint64_t StartOffset = Symbols[SI].Addr - PreferredBaseAddress;
  uint64_t EndOffset = (SI + 1 < SE)
                           ? Symbols[SI + 1].Addr - PreferredBaseAddress
                           : SectionOffset + SectSize;
  if (StartOffset >= EndOffset)
    return true;

  std::string &&SymbolName = Symbols[SI].Name.str();
  if (ShowDisassembly)
    outs() << '<' << SymbolName << ">:\n";

  uint64_t Offset = StartOffset;
  while (Offset < EndOffset) {
    MCInst Inst;
    uint64_t Size;
    // Disassemble an instruction.
    if (!DisAsm->getInstruction(Inst, Size, Bytes.slice(Offset - SectionOffset),
                                Offset + PreferredBaseAddress, nulls()))
      return false;

    if (ShowDisassembly) {
      outs() << format("%8" PRIx64 ":", Offset);
      size_t Start = outs().tell();
      IPrinter->printInst(&Inst, Offset + Size, "", *STI.get(), outs());
      if (ShowSourceLocations) {
        unsigned Cur = outs().tell() - Start;
        if (Cur < 40)
          outs().indent(40 - Cur);
        InstructionPointer Inst(this, Offset);
        outs() << getReversedLocWithContext(symbolize(Inst));
      }
      outs() << "\n";
    }

    const MCInstrDesc &MCDesc = MII->get(Inst.getOpcode());

    // Populate a vector of the symbolized callsite at this location
    InstructionPointer IP(this, Offset);
    Offset2LocStackMap[Offset] = symbolize(IP, true);

    // Populate address maps.
    CodeAddrs.push_back(Offset);
    if (MCDesc.isCall())
      CallAddrs.insert(Offset);
    else if (MCDesc.isReturn())
      RetAddrs.insert(Offset);

    Offset += Size;
  }

  if (ShowDisassembly)
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

  MCObjectFileInfo MOFI;
  MCContext Ctx(AsmInfo.get(), MRI.get(), &MOFI);
  MOFI.InitMCObjectFileInfo(Triple(TripleName), false, Ctx);
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

  if (ShowDisassembly)
    outs() << "\nDisassembly of " << FileName << ":\n";

  // Dissassemble a text section.
  for (section_iterator SI = Obj->section_begin(), SE = Obj->section_end();
       SI != SE; ++SI) {
    const SectionRef &Section = *SI;
    if (!Section.isText())
      continue;

    uint64_t ImageLoadAddr = PreferredBaseAddress;
    uint64_t SectionOffset = Section.getAddress() - ImageLoadAddr;
    uint64_t SectSize = Section.getSize();
    if (!SectSize)
      continue;

    // Register the text section.
    TextSections.insert({SectionOffset, SectSize});

    if (ShowDisassembly) {
      StringRef SectionName = unwrapOrError(Section.getName(), FileName);
      outs() << "\nDisassembly of section " << SectionName;
      outs() << " [" << format("0x%" PRIx64, SectionOffset) << ", "
             << format("0x%" PRIx64, SectionOffset + SectSize) << "]:\n\n";
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
  auto Addr = object::SectionedAddress{IP.Offset + PreferredBaseAddress,
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
                      CallerFrame.Discriminator);
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
