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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InlineAsm.h"
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
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>

using namespace llvm;
using namespace object;

static void addSpecialSymbols(
    const Module &M,
    function_ref<void(StringRef, BasicSymbolRef::Flags)> AsmSymbol) {
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

void ModuleSymbolTable::addModule(Module *M) {
  if (FirstMod)
    assert(FirstMod->getTargetTriple() == M->getTargetTriple());
  else
    FirstMod = M;

  for (GlobalValue &GV : M->global_values())
    SymTab.push_back(&GV);

  auto AddSymbols = [this](StringRef Name, BasicSymbolRef::Flags Flags) {
    SymTab.push_back(new (AsmSymbols.Allocate())
                         AsmSymbol(std::string(Name), Flags));
  };

  if (M->getModuleInlineAsm().empty()) {
    addSpecialSymbols(*M, AddSymbols);
    return;
  }

  // Make sure that global-asm-symbols is materialized. Otherwise
  // CollectAsmSymbols falls back to parsing.
  consumeError(M->materializeMetadata());

  CollectAsmSymbols(*M, AddSymbols);
}

static void initializeRecordStreamer(
    const Module &M, StringRef CPU, StringRef Features,
    function_ref<void(RecordStreamer &)> Init,
    function_ref<void(const DiagnosticInfo &DI)> DiagHandler) {
  // This function may be called twice, once for ModuleSummaryIndexAnalysis and
  // the other when writing the IR symbol table. If parsing inline assembly has
  // caused errors in the first run, suppress the second run.
  if (M.getContext().getDiagHandlerPtr()->HasErrors)
    return;
  StringRef InlineAsm = M.getModuleInlineAsm();
  if (InlineAsm.empty())
    return;

  std::string Err;
  const Triple TT(M.getTargetTriple());
  const Target *T = TargetRegistry::lookupTarget(TT, Err);
  assert(T && T->hasMCAsmParser());

  std::unique_ptr<MCRegisterInfo> MRI(T->createMCRegInfo(TT));
  if (!MRI)
    return;

  MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(T->createMCAsmInfo(*MRI, TT, MCOptions));
  if (!MAI)
    return;

  std::unique_ptr<MCSubtargetInfo> STI(
      T->createMCSubtargetInfo(TT, CPU, Features));
  if (!STI)
    return;

  std::unique_ptr<MCInstrInfo> MCII(T->createMCInstrInfo());
  if (!MCII)
    return;

  std::unique_ptr<MemoryBuffer> Buffer(
      MemoryBuffer::getMemBuffer(InlineAsm, "<inline asm>"));
  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(std::move(Buffer), SMLoc());

  MCContext MCCtx(TT, MAI.get(), MRI.get(), STI.get(), &SrcMgr);
  std::unique_ptr<MCObjectFileInfo> MOFI(
      T->createMCObjectFileInfo(MCCtx, /*PIC=*/false));
  MCCtx.setObjectFileInfo(MOFI.get());
  RecordStreamer Streamer(MCCtx, M);
  T->createNullTargetStreamer(Streamer);

  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(SrcMgr, MCCtx, Streamer, *MAI));

  std::unique_ptr<MCTargetAsmParser> TAP(
      T->createMCAsmParser(*STI, *Parser, *MCII, MCOptions));
  if (!TAP)
    return;

  MCCtx.setDiagnosticHandler([&](const SMDiagnostic &SMD, bool IsInlineAsm,
                                 const SourceMgr &SrcMgr,
                                 std::vector<const MDNode *> &LocInfos) {
    DiagnosticInfoSrcMgr Diag(SMD, M.getName(), IsInlineAsm, /*LocCookie=*/0);
    if (DiagHandler) {
      DiagHandler(Diag);
      return;
    }
    M.getContext().diagnose(Diag);
  });

  // Module-level inline asm is assumed to use At&t syntax (see
  // AsmPrinter::doInitialization()).
  Parser->setAssemblerDialect(InlineAsm::AD_ATT);

  Parser->setTargetParser(*TAP);
  if (Parser->Run(false))
    return;

  Init(Streamer);
}

static void
addSymbols(RecordStreamer &Streamer,
           function_ref<void(StringRef, BasicSymbolRef::Flags)> AsmSymbol) {
  Streamer.flushSymverDirectives();

  for (const auto &[Name, State] : Streamer) {
    // FIXME: For now we just assume that all asm symbols are executable.
    uint32_t Res = BasicSymbolRef::SF_Executable;
    switch (State) {
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
    AsmSymbol(Name, BasicSymbolRef::Flags(Res));
  }
}

static void collectAsmSymbolsFromMetadata(
    const Module &M, MDTuple *SymbolsMD, MDTuple *SymversMD,
    function_ref<void(StringRef, BasicSymbolRef::Flags)> AsmSymbol) {

  // Extract all symbols from global-asm-symbols module flag. With AppendUnique
  // ModFlagBehavior there may only be symbols with unique Name and Flags
  // combination.
  MapVector<StringRef, SmallVector<BasicSymbolRef::Flags, 2>> Symbols;
  for (const Metadata *MD : SymbolsMD->operands()) {
    const MDTuple *SymMD = cast<MDTuple>(MD);
    const MDString *Name = cast<MDString>(SymMD->getOperand(0));
    const ConstantInt *Flags =
        mdconst::extract<ConstantInt>(SymMD->getOperand(1));

    // Collect symbols with the same name, but different flags.
    Symbols[Name->getString()].push_back(
        static_cast<BasicSymbolRef::Flags>(Flags->getZExtValue()));
  }

  // If the same symbol is duplicated with different flags, assume that it can
  // be redefined and select the most accurate set of flags. If it actually
  // cannot be redefined, AsmParser will diagnose this later.
  auto LessFn = [](const BasicSymbolRef::Flags &LHS,
                   const BasicSymbolRef::Flags &RHS) {
    // Select defined symbols instead of undefined ones.
    if (LHS & BasicSymbolRef::SF_Undefined)
      return false;

    // Return true if LHS is "better" (more defined) than RHS.
    return true;
  };
  // Find the "best" set of flags and put it to the front. Ignore the rest.
  for (auto &[Name, Flags] : Symbols) {
    auto FlagsIt = std::min_element(Flags.begin(), Flags.end(), LessFn);
    Flags[0] = *FlagsIt;
  }

  // Promote symver symbols from SF_Undefined when the corresponding aliasee
  // symbol is defined either in assembly, or in IR.
  MapVector<StringRef, StringRef> SymverToSym;
  for (const Metadata *MD : SymversMD->operands()) {
    const MDTuple *SymverMD = cast<MDTuple>(MD);

    StringRef AliaseeName =
        cast<MDString>(SymverMD->getOperand(0))->getString();
    auto AliaseeIt = Symbols.find(AliaseeName);

    // Aliasee symbols should normally be emitted along with its symver. Bail
    // out if it was dropped for some reason.
    if (AliaseeIt == Symbols.end())
      continue;

    BasicSymbolRef::Flags AliaseeFlag = AliaseeIt->second[0];

    // Iterate over symvers (aliases) of the aliasee.
    for (size_t Idx = 1, End = SymverMD->getNumOperands(); Idx < End; ++Idx) {
      StringRef SymverName =
          cast<MDString>(SymverMD->getOperand(Idx))->getString();

      auto SymverIt = Symbols.find(SymverName);
      if (SymverIt == Symbols.end())
        continue;

      BasicSymbolRef::Flags SymverFlag = SymverIt->second[0];
      BasicSymbolRef::Flags SymverFlagDefined =
          static_cast<BasicSymbolRef::Flags>(SymverFlag &
                                             (~BasicSymbolRef::SF_Undefined));

      if (SymverFlag == SymverFlagDefined)
        continue;

      // If the aliasee is defined - define the symver.
      if (!(AliaseeFlag & BasicSymbolRef::SF_Undefined)) {
        SymverIt->second[0] = SymverFlagDefined;
        continue;
      }

      // If aliasee is defined in IR - define the symver.
      if (const GlobalValue *GV = M.getNamedValue(AliaseeName)) {
        if (!GV->isDeclarationForLinker())
          SymverIt->second[0] = SymverFlagDefined;
      }
    }
  }

  for (auto &[Name, Flags] : Symbols)
    AsmSymbol(Name, Flags[0]);
}

void ModuleSymbolTable::CollectAsmSymbols(
    const Module &M,
    function_ref<void(StringRef, BasicSymbolRef::Flags)> AsmSymbol) {

  MDTuple *SymbolsMD =
      dyn_cast_if_present<MDTuple>(M.getModuleFlag("global-asm-symbols"));

  if (SymbolsMD) {
    MDTuple *SymversMD = cast<MDTuple>(M.getModuleFlag("global-asm-symvers"));
    collectAsmSymbolsFromMetadata(M, SymbolsMD, SymversMD, AsmSymbol);
    addSpecialSymbols(M, AsmSymbol);
    return;
  }

  initializeRecordStreamer(
      M, /*CPU=*/"", /*Features=*/"",
      [&](RecordStreamer &Streamer) { addSymbols(Streamer, AsmSymbol); },
      /*DiagHandler=*/nullptr);

  addSpecialSymbols(M, AsmSymbol);
}

static void addSymvers(RecordStreamer &Streamer,
                       function_ref<void(StringRef, StringRef)> AsmSymver) {
  for (const auto &[Name, Aliases] : Streamer.symverAliases())
    for (StringRef Alias : Aliases)
      AsmSymver(Name->getName(), Alias);
}

static void collectAsmSymversFromMetadata(
    MDTuple *SymversMD, function_ref<void(StringRef, StringRef)> AsmSymver) {

  // Extract all symvers from global-asm-symvers module flag. These are stored
  // as lists of [symbol, symver1, ..., symverN], so they may not be fully
  // uniqued by AppendUnique ModFlagBehavior.
  MapVector<StringRef, SmallSet<StringRef, 4>> Symvers;
  for (const Metadata *MD : SymversMD->operands()) {
    const MDTuple *SymverMD = cast<MDTuple>(MD);
    StringRef Name = cast<MDString>(SymverMD->getOperand(0))->getString();
    for (size_t Idx = 1, End = SymverMD->getNumOperands(); Idx < End; ++Idx) {
      Symvers[Name].insert(
          cast<MDString>(SymverMD->getOperand(Idx))->getString());
    }
  }

  for (const auto &[Name, Aliases] : Symvers) {
    for (StringRef Alias : Aliases)
      AsmSymver(Name, Alias);
  }
}

void ModuleSymbolTable::CollectAsmSymvers(
    const Module &M, function_ref<void(StringRef, StringRef)> AsmSymver) {

  MDTuple *SymversMD =
      dyn_cast_if_present<MDTuple>(M.getModuleFlag("global-asm-symvers"));

  if (SymversMD) {
    collectAsmSymversFromMetadata(SymversMD, AsmSymver);
    return;
  }

  initializeRecordStreamer(
      M, /*CPU=*/"", /*Features=*/"",
      [&](RecordStreamer &Streamer) { addSymvers(Streamer, AsmSymver); },
      /*DiagHandler=*/nullptr);
}

bool ModuleSymbolTable::EmitModuleFlags(Module &M, StringRef CPU,
                                        StringRef Features) {
  if (M.getModuleInlineAsm().empty())
    return false;

  llvm::LLVMContext &Ctx = M.getContext();

  bool HaveErrors = false;
  auto DiagHandler = [&](const llvm::DiagnosticInfo &DI) {
    // Ignore diagnostics from the assembly parser.
    //
    // Errors in assembly mean that we cannot build a symbol table
    // from it. However, we do not diagnose them here, because we
    // don't know if the Module is ever going to actually reach
    // CodeGen where this would matter.
    if (DI.getSeverity() == llvm::DS_Error)
      HaveErrors = true;
  };

  // Build global-asm-symbols as a list of pairs (name, flags bitmask).
  SmallVector<llvm::Metadata *, 16> Symbols;

  auto AsmSymbol = [&](StringRef Name,
                       llvm::object::BasicSymbolRef::Flags Flags) {
    Symbols.push_back(llvm::MDNode::get(
        Ctx, {llvm::MDString::get(Ctx, Name),
              llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                  llvm::Type::getInt32Ty(Ctx), Flags))}));
  };

  // Build global-asm-symvers as a list of lists (name, followed by all
  // aliases).
  llvm::MapVector<StringRef, SmallVector<llvm::Metadata *, 2>> SymversMap;

  auto AsmSymver = [&](StringRef Name, StringRef Alias) {
    auto ItNew = SymversMap.try_emplace(Name);
    SmallVector<llvm::Metadata *, 2> &Aliases = ItNew.first->second;

    // If it is a new list, insert the primary name at the front.
    if (ItNew.second)
      Aliases.push_back(llvm::MDString::get(Ctx, Name));

    Aliases.push_back(llvm::MDString::get(Ctx, Alias));
  };

  // Parse global inline assembly and collect all symbols and symvers.
  initializeRecordStreamer(
      M, CPU, Features,
      [&](RecordStreamer &Streamer) {
        addSymvers(Streamer, AsmSymver);
        addSymbols(Streamer, AsmSymbol);
      },
      DiagHandler);

  if (HaveErrors)
    return false;

  // Emit a symbol table as module flags, so they can be traversed
  // later with CollectAsmSymbols and CollectAsmSymvers.
  M.addModuleFlag(llvm::Module::AppendUnique, "global-asm-symbols",
                  llvm::MDNode::get(Ctx, Symbols));

  SmallVector<llvm::Metadata *, 16> Symvers;
  Symvers.reserve(SymversMap.size());
  for (const auto &KV : SymversMap)
    Symvers.push_back(llvm::MDNode::get(Ctx, KV.second));

  M.addModuleFlag(llvm::Module::AppendUnique, "global-asm-symvers",
                  llvm::MDNode::get(Ctx, Symvers));

  return true;
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
