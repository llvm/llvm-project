//===-- CodeObjectDisassembler.cpp - Disassembler for HSA Code Object------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// This file contains definition for HSA Code Object Dissassembler
//
//===----------------------------------------------------------------------===//

#include "CodeObjectDisassembler.h"

#include "AMDGPU.h"
#include "Disassembler/CodeObject.h"
#include "Disassembler/AMDGPUDisassembler.h"
#include "MCTargetDesc/AMDGPUInstPrinter.h"
#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "TargetInfo/AMDGPUTargetInfo.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/TargetRegistry.h"

#define DEBUG_TYPE "amdgpu-disassembler"

namespace llvm {
#include "AMDGPUPTNote.h"
}

using namespace llvm;

CodeObjectDisassembler::CodeObjectDisassembler(MCContext *C,
                                               StringRef TN,
                                               MCInstPrinter *IP,
                                               MCTargetStreamer *TS)
  : Ctx(C), TripleName(TN), InstPrinter(IP),
    AsmStreamer(static_cast<AMDGPUTargetStreamer *>(TS)) {}

Expected<CodeObjectDisassembler::SymbolsTy>
CodeObjectDisassembler::CollectSymbols(const HSACodeObject *CodeObject) {
  SymbolsTy Symbols;
  for (const auto &Symbol : CodeObject->symbols()) {
    auto AddressOr = Symbol.getAddress();
    if (!AddressOr)
      return AddressOr.takeError();

    auto NameOr = Symbol.getName();
    if (!NameOr)
      return NameOr.takeError();
    if (NameOr->empty())
      continue;

    uint8_t SymbolType = CodeObject->getSymbol(Symbol.getRawDataRefImpl())->getType();
    Symbols.emplace_back(*AddressOr, *NameOr, SymbolType);
  }
  return Symbols;
}

std::error_code CodeObjectDisassembler::printNotes(const HSACodeObject *CodeObject) {
  for (auto Note : CodeObject->notes()) {
    if (!Note)
      return errorToErrorCode(Note.takeError());

    switch (Note->type) {
    case AMDGPU::ElfNote::NT_AMDGPU_HSA_CODE_OBJECT_VERSION: {
      auto VersionOr = Note->as<amdgpu_hsa_code_object_version>();
      if (!VersionOr)
        return errorToErrorCode(VersionOr.takeError());

      auto *Version = *VersionOr;
      AsmStreamer->EmitDirectiveHSACodeObjectVersion(
        Version->major_version,
        Version->minor_version);
      AsmStreamer->getStreamer().EmitRawText("");
      break;
    }

    case AMDGPU::ElfNote::NT_AMDGPU_HSA_ISA: {
      auto IsaOr = Note->as<amdgpu_hsa_isa>();
      if (!IsaOr)
        return errorToErrorCode(IsaOr.takeError());

      auto *Isa = *IsaOr;
      AsmStreamer->EmitDirectiveHSACodeObjectISA(
        Isa->major,
        Isa->minor,
        Isa->stepping,
        Isa->getVendorName(),
        Isa->getArchitectureName());
      AsmStreamer->getStreamer().EmitRawText("");
      break;
    }
    }
  }
  return std::error_code();
}

static std::string getCPUName(const HSACodeObject *CodeObject) {
  for (auto Note : CodeObject->notes()) {
    if (!Note)
      return "";

    if (Note->type == AMDGPU::ElfNote::NT_AMDGPU_HSA_ISA) {
      auto IsaOr = Note->as<amdgpu_hsa_isa>();
      if (!IsaOr)
        return "";
      auto *Isa = *IsaOr;

      SmallString<6> OutStr;
      raw_svector_ostream OS(OutStr);
      OS << "gfx" << Isa->major << Isa->minor << Isa->stepping;
      return OS.str();
    }
  }
  return "";
}

std::error_code
CodeObjectDisassembler::printFunctions(const HSACodeObject *CodeObject,
                                       raw_ostream &ES) {
  AsmStreamer->getStreamer().InitSections(true);

  // setup disassembler
  auto SymbolsOr = CollectSymbols(CodeObject);
  if (!SymbolsOr)
    return errorToErrorCode(SymbolsOr.takeError());
  
  const auto &Target = getTheGCNTarget();
  std::unique_ptr<MCSubtargetInfo> STI(
    Target.createMCSubtargetInfo(TripleName, getCPUName(CodeObject), ""));
  if (!STI)
    return object::object_error::parse_failed;

  std::unique_ptr<MCDisassembler> InstDisasm(
    Target.createMCDisassembler(*STI, *Ctx));
  if (!InstDisasm)
    return object::object_error::parse_failed;

  std::unique_ptr<MCRelocationInfo> RelInfo(
    Target.createMCRelocationInfo(TripleName, *Ctx));
  if (RelInfo) {
    std::unique_ptr<MCSymbolizer> Symbolizer(
      Target.createMCSymbolizer(
        TripleName, nullptr, nullptr, &(*SymbolsOr), Ctx, std::move(RelInfo)));
    InstDisasm->setSymbolizer(std::move(Symbolizer));
  }

  // print functions
  for (const auto &Sym : CodeObject->functions()) {
    auto ExpectedFunction = FunctionSym::asFunctionSym(
        CodeObject->getSymbol(Sym.getRawDataRefImpl()));
    if (!ExpectedFunction)
      return errorToErrorCode(ExpectedFunction.takeError());
    auto Function = ExpectedFunction.get();

    auto NameEr = Sym.getName();
    if (!NameEr)
      return object::object_error::parse_failed;

    auto AddressOr = Function->getAddress(CodeObject);
    if (!AddressOr)
      return errorToErrorCode(AddressOr.takeError());
    uint64_t Address = *AddressOr;

    auto ExpectedKernel = KernelSym::asKernelSym(Function);
    if (ExpectedKernel) {
      auto Kernel = ExpectedKernel.get();
      Function = Kernel;

      auto KernelCodeTOr = Kernel->getAmdKernelCodeT(CodeObject);
      if (!KernelCodeTOr)
        return errorToErrorCode(KernelCodeTOr.takeError());

      AsmStreamer->EmitAMDGPUSymbolType(*NameEr, Kernel->getType());
      AsmStreamer->getStreamer().EmitRawText("");

      AsmStreamer->getStreamer().EmitRawText(*NameEr + ":");

      AsmStreamer->EmitAMDKernelCodeT(*(*KernelCodeTOr));
      AsmStreamer->getStreamer().EmitRawText("");

      Address += (*KernelCodeTOr)->kernel_code_entry_byte_offset;
    } else {
      consumeError(ExpectedKernel.takeError());
      auto Name = *NameEr;
      MCSymbolELF *Symbol = cast<MCSymbolELF>(
          AsmStreamer->getStreamer().getContext().getOrCreateSymbol(Name));
      Symbol->setType(ELF::STT_FUNC);
      AsmStreamer->getStreamer().EmitSymbolAttribute(Symbol,
                                                     MCSA_ELF_TypeFunction);
      AsmStreamer->getStreamer().EmitLabel(Symbol);
    }

    auto CodeOr = CodeObject->getCode(Function);
    if (!CodeOr)
      return errorToErrorCode(CodeOr.takeError());

    printFunctionCode(*InstDisasm, *CodeOr, Address, *SymbolsOr, ES);
  }
  return std::error_code();
}

template <typename T>
static ArrayRef<T> trimTrailingZeroes(ArrayRef<T> A, size_t Limit) {
  const auto SizeLimit = (Limit < A.size()) ? (A.size() - Limit) : 0;
  while (A.size() > SizeLimit && !A.back())
    A = A.drop_back();
  return A;
}

void CodeObjectDisassembler::printFunctionCode(const MCDisassembler &InstDisasm,
                                               ArrayRef<uint8_t> Bytes,
                                               uint64_t Address,
                                               const SymbolsTy &Symbols,
                                               raw_ostream &ES) {
#ifdef NDEBUG
  const bool DebugFlag = false;
#endif

  Bytes = trimTrailingZeroes(Bytes, 256);

  AsmStreamer->getStreamer().EmitRawText("// Disassembly:");
  SmallString<40> InstStr, CommentStr, OutStr;
  for (uint64_t Index = 0; Index < Bytes.size();) {
    ArrayRef<uint8_t> Code = Bytes.slice(Index);

    InstStr.clear();
    raw_svector_ostream IS(InstStr);
    CommentStr.clear();
    raw_svector_ostream CS(CommentStr);
    OutStr.clear();
    raw_svector_ostream OS(OutStr);

    // check for labels
    for (const auto &Sym: Symbols) {
      if (std::get<0>(Sym) == Address && std::get<2>(Sym) == ELF::STT_NOTYPE) {
        OS << std::get<1>(Sym) << ":\n";
      }
    }

    MCInst Inst;
    uint64_t EatenBytesNum = 0;
    if (InstDisasm.getInstruction(Inst, EatenBytesNum,
                                  Code,
                                  Address,
                                  DebugFlag ? dbgs() : nulls(),
                                  CS)) {
      InstPrinter->printInst(&Inst, IS, "", InstDisasm.getSubtargetInfo());
    } else {
      IS << "\t// unrecognized instruction ";
      if (EatenBytesNum == 0)
        EatenBytesNum = 4;
    }
    assert(0 == EatenBytesNum % 4);

    OS << left_justify(IS.str(), 60) << format("// %012X:", Address);
    typedef support::ulittle32_t U32;
    for (auto D : makeArrayRef(reinterpret_cast<const U32 *>(Code.data()),
                               EatenBytesNum / sizeof(U32)))
      // D should be explicitly casted to uint32_t here as it is passed
      // by format to snprintf as vararg.
      OS << format(" %08" PRIX32, static_cast<uint32_t>(D));

    if (!CS.str().empty())
      OS << " // " << CS.str();

    AsmStreamer->getStreamer().EmitRawText(OS.str());

    Address += EatenBytesNum;
    Index += EatenBytesNum;
  }
  AsmStreamer->getStreamer().EmitRawText("");
}

std::error_code CodeObjectDisassembler::Disassemble(MemoryBufferRef Buffer,
                                                    raw_ostream &ES) {
  using namespace object;
  
  // Create ELF 64-bit low-endian object file
  Expected<std::unique_ptr<HSACodeObject>> CodeObjectOrError =
      HSACodeObject::create(Buffer);
  if (Error E = CodeObjectOrError.takeError())
    return errorToErrorCode(std::move(E));

  std::unique_ptr<HSACodeObject> CodeObject = std::move(*CodeObjectOrError);

  std::error_code EC = printNotes(CodeObject.get());
  if (EC)
    return EC;

  EC = printFunctions(CodeObject.get(), ES);
  if (EC)
    return EC;

  return std::error_code();
}
