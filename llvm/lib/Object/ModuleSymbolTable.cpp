//===- ModuleSymbolTable.cpp - symbol table for in-memory IR --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class represents a symbol table built from in-memory IR. It provides
// access to GlobalValues and should only be used if such access is required
// (e.g. in the LTO implementation).
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ModuleSymbolTable.h"
#include "RecordStreamer.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>

using namespace llvm;
using namespace object;

void ModuleSymbolTable::addModule(Module *M) {
  if (FirstMod)
    assert(FirstMod->getTargetTriple() == M->getTargetTriple());
  else
    FirstMod = M;

  for (GlobalValue &GV : M->global_values())
    SymTab.push_back(&GV);

  CollectAsmSymbols(
      *M,
      [this](StringRef Name, BasicSymbolRef::Flags Flags) {
        SymTab.push_back(new (AsmSymbols.Allocate())
                             AsmSymbol(std::string(Name), Flags));
      },
      [this](const Function &F) { AsmDefiningFunctions.insert(&F); });
}

bool ModuleSymbolTable::inlineAsmMayDefineSymbols(StringRef Asm) {
  return Asm.contains(".glob") || Asm.contains(".weak");
}

// Rewrite the asm string of an inline asm call into text the assembler can
// parse. Operands are not known, so each becomes 0. The syntax is that of
// AsmPrinter::emitInlineAsm.
static std::string expandInlineAsm(StringRef Asm,
                                   InlineAsm::AsmDialect Dialect) {
  std::string Out;
  // Position in a $( ... $| ... $) group of dialect alternatives.
  bool InAlternatives = false;
  unsigned Alternative = 0;
  unsigned Wanted = Dialect == InlineAsm::AD_Intel ? 1 : 0;
  auto Emit = [&](StringRef Text) {
    if (!InAlternatives || Alternative == Wanted)
      Out += Text;
  };
  for (size_t I = 0, E = Asm.size(); I != E; ++I) {
    char C = Asm[I];
    if (C != '$' || I + 1 == E) {
      Emit(StringRef(&C, 1));
      continue;
    }
    char N = Asm[++I];
    switch (N) {
    case '$':
      Emit("$");
      break;
    case '(':
      InAlternatives = true;
      Alternative = 0;
      break;
    case '|':
      ++Alternative;
      break;
    case ')':
      InAlternatives = false;
      break;
    case '{': {
      size_t Close = Asm.find('}', I);
      if (Close == StringRef::npos)
        Close = E - 1;
      // ${N:modifier} is an operand, ${:name} a built in string.
      Emit("0");
      I = Close;
      break;
    }
    default:
      if (isDigit(N)) {
        while (I + 1 != E && isDigit(Asm[I + 1]))
          ++I;
        Emit("0");
      } else {
        Emit(StringRef(&C, 1));
        Emit(StringRef(&N, 1));
      }
    }
  }
  return Out;
}

namespace {
// Module level asm, or the asm of an inline asm call in the body of Origin.
struct AsmFragment {
  std::string Asm;
  std::string CPU;
  std::string Features;
  InlineAsm::AsmDialect Dialect = InlineAsm::AD_ATT;
  const Function *Origin = nullptr;
};
} // namespace

static void initializeRecordStreamer(
    const Module &M,
    function_ref<void(RecordStreamer &, const Function *)> Init,
    bool IncludeFunctions = true) {
  // This function may be called twice, once for ModuleSummaryIndexAnalysis and
  // the other when writing the IR symbol table. If parsing inline assembly has
  // caused errors in the first run, suppress the second run.
  if (M.getContext().getDiagHandlerPtr()->HasErrors)
    return;

  SmallVector<AsmFragment, 1> Fragments;
  for (const Module::GlobalAsmFragment &Frag : M.getModuleInlineAsm()) {
    AsmFragment F;
    F.Asm = Frag.Asm;
    F.CPU = Frag.Props.TargetCPU;
    F.Features = Frag.Props.TargetFeatures;
    Fragments.push_back(std::move(F));
  }
  if (IncludeFunctions) {
    // Inline asm in a function can define symbols the linker has to know about.
    SmallPtrSet<const InlineAsm *, 4> Seen;
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;
      Seen.clear();
      for (const BasicBlock &BB : F)
        for (const Instruction &I : BB) {
          const auto *CB = dyn_cast<CallBase>(&I);
          if (!CB)
            continue;
          const auto *IA = dyn_cast<InlineAsm>(CB->getCalledOperand());
          if (!IA ||
              !ModuleSymbolTable::inlineAsmMayDefineSymbols(
                  IA->getAsmString()) ||
              !Seen.insert(IA).second)
            continue;
          AsmFragment Frag;
          Frag.Asm = expandInlineAsm(IA->getAsmString(), IA->getDialect());
          Frag.CPU = F.getFnAttribute("target-cpu").getValueAsString();
          Frag.Features =
              F.getFnAttribute("target-features").getValueAsString();
          Frag.Dialect = IA->getDialect();
          Frag.Origin = &F;
          Fragments.push_back(std::move(Frag));
        }
    }
  }
  if (Fragments.empty())
    return;

  std::string Err;
  const Triple TT(M.getTargetTriple());
  const Target *T = TargetRegistry::lookupTarget(TT, Err);
  if (!T || !T->hasMCAsmParser()) {
    assert(!M.hasModuleInlineAsm() && "inline asm without an assembler parser");
    return;
  }

  std::unique_ptr<MCRegisterInfo> MRI(T->createMCRegInfo(TT));
  if (!MRI)
    return;

  MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(T->createMCAsmInfo(*MRI, TT, MCOptions));
  if (!MAI)
    return;

  std::unique_ptr<MCInstrInfo> MCII(T->createMCInstrInfo());
  if (!MCII)
    return;

  for (const AsmFragment &Frag : Fragments) {
    std::unique_ptr<MCSubtargetInfo> STI(
        T->createMCSubtargetInfo(TT, Frag.CPU, Frag.Features));
    if (!STI)
      return;

    std::unique_ptr<MemoryBuffer> Buffer(
        MemoryBuffer::getMemBuffer(Frag.Asm, "<inline asm>"));
    SourceMgr SrcMgr;
    SrcMgr.AddNewSourceBuffer(std::move(Buffer), SMLoc());

    MCContext MCCtx(TT, *MAI, *MRI, *STI, &SrcMgr);
    std::unique_ptr<MCObjectFileInfo> MOFI(
        T->createMCObjectFileInfo(MCCtx, /*PIC=*/false));
    MCCtx.setObjectFileInfo(MOFI.get());
    RecordStreamer Streamer(MCCtx, M);
    T->createNullTargetStreamer(Streamer);

    std::unique_ptr<MCAsmParser> Parser(
        createMCAsmParser(SrcMgr, MCCtx, Streamer, *MAI));

    std::unique_ptr<MCTargetAsmParser> TAP(
        T->createMCAsmParser(*STI, *Parser, *MCII));
    if (!TAP)
      return;

    // The operands are guessed, so the asm may not parse. Errors are reported
    // when the function is assembled.
    if (Frag.Origin)
      MCCtx.setDiagnosticHandler([](const SMDiagnostic &, bool,
                                    const SourceMgr &,
                                    std::vector<const MDNode *> &) {});
    else
      MCCtx.setDiagnosticHandler([&](const SMDiagnostic &SMD, bool IsInlineAsm,
                                     const SourceMgr &SrcMgr,
                                     std::vector<const MDNode *> &LocInfos) {
        M.getContext().diagnose(DiagnosticInfoSrcMgr(SMD, M.getName(),
                                                     IsInlineAsm,
                                                     /*LocCookie=*/0));
      });

    // Module-level inline asm is assumed to use At&t syntax (see
    // AsmPrinter::doInitialization()).
    Parser->setAssemblerDialect(Frag.Dialect);

    Parser->setSymbolScanningMode(true);

    Parser->setTargetParser(*TAP);
    if (Parser->Run(false)) {
      if (Frag.Origin)
        continue;
      return;
    }

    Init(Streamer, Frag.Origin);
  }
}

void ModuleSymbolTable::CollectAsmSymbols(
    const Module &M,
    function_ref<void(StringRef, BasicSymbolRef::Flags)> AsmSymbol,
    function_ref<void(const Function &)> AsmDefiningFunction) {
  // Asm in a function can name values with local linkage, which are not
  // symbols the linker sees.
  StringSet<> LocalNames;
  bool LocalNamesBuilt = false;
  auto IsLocalName = [&](StringRef Name) {
    if (!LocalNamesBuilt) {
      LocalNamesBuilt = true;
      Mangler Mang;
      SmallString<64> MangledName;
      for (const GlobalValue &GV : M.global_values()) {
        if (!GV.hasName() || !GV.hasLocalLinkage())
          continue;
        MangledName.clear();
        Mang.getNameWithPrefix(MangledName, &GV,
                               /*CannotUsePrivateLabel=*/false);
        LocalNames.insert(MangledName);
      }
    }
    return LocalNames.contains(Name);
  };

  initializeRecordStreamer(
      M, [&](RecordStreamer &Streamer, const Function *Origin) {
        Streamer.flushSymverDirectives();

        for (auto &KV : Streamer) {
          StringRef Key = KV.first();
          RecordStreamer::State Value = KV.second;
          // Labels the asm defines for itself stay inside the function.
          if (Origin && Value == RecordStreamer::Defined)
            continue;
          if (Origin && Value == RecordStreamer::Used && IsLocalName(Key))
            continue;
          if (Origin && AsmDefiningFunction &&
              (Value == RecordStreamer::DefinedGlobal ||
               Value == RecordStreamer::DefinedWeak))
            AsmDefiningFunction(*Origin);
          // FIXME: For now we just assume that all asm symbols are executable.
          uint32_t Res = BasicSymbolRef::SF_Executable;
          switch (Value) {
          case RecordStreamer::NeverSeen:
            llvm_unreachable("NeverSeen should have been replaced earlier");
          case RecordStreamer::DefinedGlobal:
            Res |= BasicSymbolRef::SF_Global;
            break;
          case RecordStreamer::Defined:
            break;
          case RecordStreamer::Global:
          case RecordStreamer::Used:
            Res |= BasicSymbolRef::SF_Undefined;
            Res |= BasicSymbolRef::SF_Global;
            break;
          case RecordStreamer::DefinedWeak:
            Res |= BasicSymbolRef::SF_Weak;
            Res |= BasicSymbolRef::SF_Global;
            break;
          case RecordStreamer::UndefinedWeak:
            Res |= BasicSymbolRef::SF_Weak;
            Res |= BasicSymbolRef::SF_Undefined;
          }
          AsmSymbol(Key, BasicSymbolRef::Flags(Res));
        }
      });

  // In ELF, object code generated for x86-32 and some code models of x86-64 may
  // reference the special symbol _GLOBAL_OFFSET_TABLE_ that is not used in the
  // IR. Record it like inline asm symbols.
  Triple TT(M.getTargetTriple());
  if (!TT.isOSBinFormatELF() || !TT.isX86())
    return;
  auto CM = M.getCodeModel();
  if (TT.getArch() == Triple::x86 || CM == CodeModel::Medium ||
      CM == CodeModel::Large) {
    AsmSymbol("_GLOBAL_OFFSET_TABLE_",
              BasicSymbolRef::Flags(BasicSymbolRef::SF_Undefined |
                                    BasicSymbolRef::SF_Global));
  }
}

void ModuleSymbolTable::CollectAsmSymvers(
    const Module &M, function_ref<void(StringRef, StringRef)> AsmSymver) {
  initializeRecordStreamer(
      M,
      [&](RecordStreamer &Streamer, const Function *) {
        for (auto &KV : Streamer.symverAliases())
          for (auto &Alias : KV.second)
            AsmSymver(KV.first->getName(), Alias);
      },
      /*IncludeFunctions=*/false);
}

void ModuleSymbolTable::printSymbolName(raw_ostream &OS, Symbol S) const {
  if (isa<AsmSymbol *>(S)) {
    OS << cast<AsmSymbol *>(S)->first;
    return;
  }

  auto *GV = cast<GlobalValue *>(S);
  if (GV->hasDLLImportStorageClass())
    OS << "__imp_";

  Mang.getNameWithPrefix(OS, GV, false);
}

uint32_t ModuleSymbolTable::getSymbolFlags(Symbol S) const {
  if (isa<AsmSymbol *>(S))
    return cast<AsmSymbol *>(S)->second;

  auto *GV = cast<GlobalValue *>(S);

  uint32_t Res = BasicSymbolRef::SF_None;
  if (GV->isDeclarationForLinker())
    Res |= BasicSymbolRef::SF_Undefined;
  else if (GV->hasHiddenVisibility() && !GV->hasLocalLinkage())
    Res |= BasicSymbolRef::SF_Hidden;
  if (const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV)) {
    if (GVar->isConstant())
      Res |= BasicSymbolRef::SF_Const;
  }
  if (const GlobalObject *GO = GV->getAliaseeObject())
    if (isa<Function>(GO) || isa<GlobalIFunc>(GO))
      Res |= BasicSymbolRef::SF_Executable;
  if (isa<GlobalAlias>(GV))
    Res |= BasicSymbolRef::SF_Indirect;
  if (GV->hasPrivateLinkage())
    Res |= BasicSymbolRef::SF_FormatSpecific;
  if (!GV->hasLocalLinkage())
    Res |= BasicSymbolRef::SF_Global;
  if (GV->hasCommonLinkage())
    Res |= BasicSymbolRef::SF_Common;
  if (GV->hasLinkOnceLinkage() || GV->hasWeakLinkage() ||
      GV->hasExternalWeakLinkage())
    Res |= BasicSymbolRef::SF_Weak;

  if (GV->getName().starts_with("llvm."))
    Res |= BasicSymbolRef::SF_FormatSpecific;
  else if (auto *Var = dyn_cast<GlobalVariable>(GV)) {
    if (Var->getSection() == "llvm.metadata")
      Res |= BasicSymbolRef::SF_FormatSpecific;
  }

  return Res;
}
