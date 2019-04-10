//===-- WebAssemblyAsmPrinter.cpp - WebAssembly LLVM assembly writer ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains a printer that converts from our internal
/// representation of machine-dependent LLVM code to the WebAssembly assembly
/// language.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyAsmPrinter.h"
#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "MCTargetDesc/WebAssemblyTargetStreamer.h"
#include "WebAssembly.h"
#include "WebAssemblyMCInstLower.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyRegisterInfo.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

extern cl::opt<bool> WasmKeepRegisters;

//===----------------------------------------------------------------------===//
// Helpers.
//===----------------------------------------------------------------------===//

MVT WebAssemblyAsmPrinter::getRegType(unsigned RegNo) const {
  const TargetRegisterInfo *TRI = Subtarget->getRegisterInfo();
  const TargetRegisterClass *TRC = MRI->getRegClass(RegNo);
  for (MVT T : {MVT::i32, MVT::i64, MVT::f32, MVT::f64, MVT::v16i8, MVT::v8i16,
                MVT::v4i32, MVT::v2i64, MVT::v4f32, MVT::v2f64})
    if (TRI->isTypeLegalForClass(*TRC, T))
      return T;
  LLVM_DEBUG(errs() << "Unknown type for register number: " << RegNo);
  llvm_unreachable("Unknown register type");
  return MVT::Other;
}

std::string WebAssemblyAsmPrinter::regToString(const MachineOperand &MO) {
  unsigned RegNo = MO.getReg();
  assert(TargetRegisterInfo::isVirtualRegister(RegNo) &&
         "Unlowered physical register encountered during assembly printing");
  assert(!MFI->isVRegStackified(RegNo));
  unsigned WAReg = MFI->getWAReg(RegNo);
  assert(WAReg != WebAssemblyFunctionInfo::UnusedReg);
  return '$' + utostr(WAReg);
}

WebAssemblyTargetStreamer *WebAssemblyAsmPrinter::getTargetStreamer() {
  MCTargetStreamer *TS = OutStreamer->getTargetStreamer();
  return static_cast<WebAssemblyTargetStreamer *>(TS);
}

//===----------------------------------------------------------------------===//
// WebAssemblyAsmPrinter Implementation.
//===----------------------------------------------------------------------===//

void WebAssemblyAsmPrinter::EmitEndOfAsmFile(Module &M) {
  for (auto &It : OutContext.getSymbols()) {
    // Emit a .globaltype and .eventtype declaration.
    auto Sym = cast<MCSymbolWasm>(It.getValue());
    if (Sym->getType() == wasm::WASM_SYMBOL_TYPE_GLOBAL)
      getTargetStreamer()->emitGlobalType(Sym);
    else if (Sym->getType() == wasm::WASM_SYMBOL_TYPE_EVENT)
      getTargetStreamer()->emitEventType(Sym);
  }

  for (const auto &F : M) {
    // Emit function type info for all undefined functions
    if (F.isDeclarationForLinker() && !F.isIntrinsic()) {
      SmallVector<MVT, 4> Results;
      SmallVector<MVT, 4> Params;
      computeSignatureVTs(F.getFunctionType(), F, TM, Params, Results);
      auto *Sym = cast<MCSymbolWasm>(getSymbol(&F));
      Sym->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);
      if (!Sym->getSignature()) {
        auto Signature = signatureFromMVTs(Results, Params);
        Sym->setSignature(Signature.get());
        addSignature(std::move(Signature));
      }
      // FIXME: this was originally intended for post-linking and was only used
      // for imports that were only called indirectly (i.e. s2wasm could not
      // infer the type from a call). With object files it applies to all
      // imports. so fix the names and the tests, or rethink how import
      // delcarations work in asm files.
      getTargetStreamer()->emitFunctionType(Sym);

      if (TM.getTargetTriple().isOSBinFormatWasm() &&
          F.hasFnAttribute("wasm-import-module")) {
        StringRef Name =
            F.getFnAttribute("wasm-import-module").getValueAsString();
        Sym->setImportModule(Name);
        getTargetStreamer()->emitImportModule(Sym, Name);
      }
      if (TM.getTargetTriple().isOSBinFormatWasm() &&
          F.hasFnAttribute("wasm-import-name")) {
        StringRef Name =
            F.getFnAttribute("wasm-import-name").getValueAsString();
        Sym->setImportName(Name);
        getTargetStreamer()->emitImportName(Sym, Name);
      }
    }
  }

  for (const auto &G : M.globals()) {
    if (!G.hasInitializer() && G.hasExternalLinkage()) {
      if (G.getValueType()->isSized()) {
        uint16_t Size = M.getDataLayout().getTypeAllocSize(G.getValueType());
        OutStreamer->emitELFSize(getSymbol(&G),
                                 MCConstantExpr::create(Size, OutContext));
      }
    }
  }

  if (const NamedMDNode *Named = M.getNamedMetadata("wasm.custom_sections")) {
    for (const Metadata *MD : Named->operands()) {
      const auto *Tuple = dyn_cast<MDTuple>(MD);
      if (!Tuple || Tuple->getNumOperands() != 2)
        continue;
      const MDString *Name = dyn_cast<MDString>(Tuple->getOperand(0));
      const MDString *Contents = dyn_cast<MDString>(Tuple->getOperand(1));
      if (!Name || !Contents)
        continue;

      OutStreamer->PushSection();
      std::string SectionName = (".custom_section." + Name->getString()).str();
      MCSectionWasm *MySection =
          OutContext.getWasmSection(SectionName, SectionKind::getMetadata());
      OutStreamer->SwitchSection(MySection);
      OutStreamer->EmitBytes(Contents->getString());
      OutStreamer->PopSection();
    }
  }

  EmitProducerInfo(M);
  EmitTargetFeatures(M);
}

void WebAssemblyAsmPrinter::EmitProducerInfo(Module &M) {
  llvm::SmallVector<std::pair<std::string, std::string>, 4> Languages;
  if (const NamedMDNode *Debug = M.getNamedMetadata("llvm.dbg.cu")) {
    llvm::SmallSet<StringRef, 4> SeenLanguages;
    for (size_t I = 0, E = Debug->getNumOperands(); I < E; ++I) {
      const auto *CU = cast<DICompileUnit>(Debug->getOperand(I));
      StringRef Language = dwarf::LanguageString(CU->getSourceLanguage());
      Language.consume_front("DW_LANG_");
      if (SeenLanguages.insert(Language).second)
        Languages.emplace_back(Language.str(), "");
    }
  }

  llvm::SmallVector<std::pair<std::string, std::string>, 4> Tools;
  if (const NamedMDNode *Ident = M.getNamedMetadata("llvm.ident")) {
    llvm::SmallSet<StringRef, 4> SeenTools;
    for (size_t I = 0, E = Ident->getNumOperands(); I < E; ++I) {
      const auto *S = cast<MDString>(Ident->getOperand(I)->getOperand(0));
      std::pair<StringRef, StringRef> Field = S->getString().split("version");
      StringRef Name = Field.first.trim();
      StringRef Version = Field.second.trim();
      if (SeenTools.insert(Name).second)
        Tools.emplace_back(Name.str(), Version.str());
    }
  }

  int FieldCount = int(!Languages.empty()) + int(!Tools.empty());
  if (FieldCount != 0) {
    MCSectionWasm *Producers = OutContext.getWasmSection(
        ".custom_section.producers", SectionKind::getMetadata());
    OutStreamer->PushSection();
    OutStreamer->SwitchSection(Producers);
    OutStreamer->EmitULEB128IntValue(FieldCount);
    for (auto &Producers : {std::make_pair("language", &Languages),
            std::make_pair("processed-by", &Tools)}) {
      if (Producers.second->empty())
        continue;
      OutStreamer->EmitULEB128IntValue(strlen(Producers.first));
      OutStreamer->EmitBytes(Producers.first);
      OutStreamer->EmitULEB128IntValue(Producers.second->size());
      for (auto &Producer : *Producers.second) {
        OutStreamer->EmitULEB128IntValue(Producer.first.size());
        OutStreamer->EmitBytes(Producer.first);
        OutStreamer->EmitULEB128IntValue(Producer.second.size());
        OutStreamer->EmitBytes(Producer.second);
      }
    }
    OutStreamer->PopSection();
  }
}

void WebAssemblyAsmPrinter::EmitTargetFeatures(Module &M) {
  struct FeatureEntry {
    uint8_t Prefix;
    StringRef Name;
  };

  // Read target features and linkage policies from module metadata
  SmallVector<FeatureEntry, 4> EmittedFeatures;
  for (const SubtargetFeatureKV &KV : WebAssemblyFeatureKV) {
    std::string MDKey = (StringRef("wasm-feature-") + KV.Key).str();
    Metadata *Policy = M.getModuleFlag(MDKey);
    if (Policy == nullptr)
      continue;

    FeatureEntry Entry;
    Entry.Prefix = 0;
    Entry.Name = KV.Key;

    if (auto *MD = cast<ConstantAsMetadata>(Policy))
      if (auto *I = cast<ConstantInt>(MD->getValue()))
        Entry.Prefix = I->getZExtValue();

    // Silently ignore invalid metadata
    if (Entry.Prefix != wasm::WASM_FEATURE_PREFIX_USED &&
        Entry.Prefix != wasm::WASM_FEATURE_PREFIX_REQUIRED &&
        Entry.Prefix != wasm::WASM_FEATURE_PREFIX_DISALLOWED)
      continue;

    EmittedFeatures.push_back(Entry);
  }

  if (EmittedFeatures.size() == 0)
    return;

  // Emit features and linkage policies into the "target_features" section
  MCSectionWasm *FeaturesSection = OutContext.getWasmSection(
      ".custom_section.target_features", SectionKind::getMetadata());
  OutStreamer->PushSection();
  OutStreamer->SwitchSection(FeaturesSection);

  OutStreamer->EmitULEB128IntValue(EmittedFeatures.size());
  for (auto &F : EmittedFeatures) {
    OutStreamer->EmitIntValue(F.Prefix, 1);
    OutStreamer->EmitULEB128IntValue(F.Name.size());
    OutStreamer->EmitBytes(F.Name);
  }

  OutStreamer->PopSection();
}

void WebAssemblyAsmPrinter::EmitConstantPool() {
  assert(MF->getConstantPool()->getConstants().empty() &&
         "WebAssembly disables constant pools");
}

void WebAssemblyAsmPrinter::EmitJumpTableInfo() {
  // Nothing to do; jump tables are incorporated into the instruction stream.
}

void WebAssemblyAsmPrinter::EmitFunctionBodyStart() {
  const Function &F = MF->getFunction();
  SmallVector<MVT, 1> ResultVTs;
  SmallVector<MVT, 4> ParamVTs;
  computeSignatureVTs(F.getFunctionType(), F, TM, ParamVTs, ResultVTs);
  auto Signature = signatureFromMVTs(ResultVTs, ParamVTs);
  auto *WasmSym = cast<MCSymbolWasm>(CurrentFnSym);
  WasmSym->setSignature(Signature.get());
  addSignature(std::move(Signature));
  WasmSym->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);

  // FIXME: clean up how params and results are emitted (use signatures)
  getTargetStreamer()->emitFunctionType(WasmSym);

  // Emit the function index.
  if (MDNode *Idx = F.getMetadata("wasm.index")) {
    assert(Idx->getNumOperands() == 1);

    getTargetStreamer()->emitIndIdx(AsmPrinter::lowerConstant(
        cast<ConstantAsMetadata>(Idx->getOperand(0))->getValue()));
  }

  SmallVector<wasm::ValType, 16> Locals;
  valTypesFromMVTs(MFI->getLocals(), Locals);
  getTargetStreamer()->emitLocal(Locals);

  AsmPrinter::EmitFunctionBodyStart();
}

void WebAssemblyAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  LLVM_DEBUG(dbgs() << "EmitInstruction: " << *MI << '\n');

  switch (MI->getOpcode()) {
  case WebAssembly::ARGUMENT_i32:
  case WebAssembly::ARGUMENT_i32_S:
  case WebAssembly::ARGUMENT_i64:
  case WebAssembly::ARGUMENT_i64_S:
  case WebAssembly::ARGUMENT_f32:
  case WebAssembly::ARGUMENT_f32_S:
  case WebAssembly::ARGUMENT_f64:
  case WebAssembly::ARGUMENT_f64_S:
  case WebAssembly::ARGUMENT_v16i8:
  case WebAssembly::ARGUMENT_v16i8_S:
  case WebAssembly::ARGUMENT_v8i16:
  case WebAssembly::ARGUMENT_v8i16_S:
  case WebAssembly::ARGUMENT_v4i32:
  case WebAssembly::ARGUMENT_v4i32_S:
  case WebAssembly::ARGUMENT_v2i64:
  case WebAssembly::ARGUMENT_v2i64_S:
  case WebAssembly::ARGUMENT_v4f32:
  case WebAssembly::ARGUMENT_v4f32_S:
  case WebAssembly::ARGUMENT_v2f64:
  case WebAssembly::ARGUMENT_v2f64_S:
    // These represent values which are live into the function entry, so there's
    // no instruction to emit.
    break;
  case WebAssembly::FALLTHROUGH_RETURN_I32:
  case WebAssembly::FALLTHROUGH_RETURN_I32_S:
  case WebAssembly::FALLTHROUGH_RETURN_I64:
  case WebAssembly::FALLTHROUGH_RETURN_I64_S:
  case WebAssembly::FALLTHROUGH_RETURN_F32:
  case WebAssembly::FALLTHROUGH_RETURN_F32_S:
  case WebAssembly::FALLTHROUGH_RETURN_F64:
  case WebAssembly::FALLTHROUGH_RETURN_F64_S:
  case WebAssembly::FALLTHROUGH_RETURN_v16i8:
  case WebAssembly::FALLTHROUGH_RETURN_v16i8_S:
  case WebAssembly::FALLTHROUGH_RETURN_v8i16:
  case WebAssembly::FALLTHROUGH_RETURN_v8i16_S:
  case WebAssembly::FALLTHROUGH_RETURN_v4i32:
  case WebAssembly::FALLTHROUGH_RETURN_v4i32_S:
  case WebAssembly::FALLTHROUGH_RETURN_v2i64:
  case WebAssembly::FALLTHROUGH_RETURN_v2i64_S:
  case WebAssembly::FALLTHROUGH_RETURN_v4f32:
  case WebAssembly::FALLTHROUGH_RETURN_v4f32_S:
  case WebAssembly::FALLTHROUGH_RETURN_v2f64:
  case WebAssembly::FALLTHROUGH_RETURN_v2f64_S: {
    // These instructions represent the implicit return at the end of a
    // function body. Always pops one value off the stack.
    if (isVerbose()) {
      OutStreamer->AddComment("fallthrough-return-value");
      OutStreamer->AddBlankLine();
    }
    break;
  }
  case WebAssembly::FALLTHROUGH_RETURN_VOID:
  case WebAssembly::FALLTHROUGH_RETURN_VOID_S:
    // This instruction represents the implicit return at the end of a
    // function body with no return value.
    if (isVerbose()) {
      OutStreamer->AddComment("fallthrough-return-void");
      OutStreamer->AddBlankLine();
    }
    break;
  case WebAssembly::EXTRACT_EXCEPTION_I32:
  case WebAssembly::EXTRACT_EXCEPTION_I32_S:
    // These are pseudo instructions that simulates popping values from stack.
    // We print these only when we have -wasm-keep-registers on for assembly
    // readability.
    if (!WasmKeepRegisters)
      break;
    LLVM_FALLTHROUGH;
  default: {
    WebAssemblyMCInstLower MCInstLowering(OutContext, *this);
    MCInst TmpInst;
    MCInstLowering.lower(MI, TmpInst);
    EmitToStreamer(*OutStreamer, TmpInst);
    break;
  }
  }
}

bool WebAssemblyAsmPrinter::PrintAsmOperand(const MachineInstr *MI,
                                            unsigned OpNo,
                                            const char *ExtraCode,
                                            raw_ostream &OS) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS))
    return false;

  if (!ExtraCode) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    switch (MO.getType()) {
    case MachineOperand::MO_Immediate:
      OS << MO.getImm();
      return false;
    case MachineOperand::MO_Register:
      // FIXME: only opcode that still contains registers, as required by
      // MachineInstr::getDebugVariable().
      assert(MI->getOpcode() == WebAssembly::INLINEASM);
      OS << regToString(MO);
      return false;
    case MachineOperand::MO_GlobalAddress:
      getSymbol(MO.getGlobal())->print(OS, MAI);
      printOffset(MO.getOffset(), OS);
      return false;
    case MachineOperand::MO_ExternalSymbol:
      GetExternalSymbolSymbol(MO.getSymbolName())->print(OS, MAI);
      printOffset(MO.getOffset(), OS);
      return false;
    case MachineOperand::MO_MachineBasicBlock:
      MO.getMBB()->getSymbol()->print(OS, MAI);
      return false;
    default:
      break;
    }
  }

  return true;
}

bool WebAssemblyAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                                  unsigned OpNo,
                                                  const char *ExtraCode,
                                                  raw_ostream &OS) {
  // The current approach to inline asm is that "r" constraints are expressed
  // as local indices, rather than values on the operand stack. This simplifies
  // using "r" as it eliminates the need to push and pop the values in a
  // particular order, however it also makes it impossible to have an "m"
  // constraint. So we don't support it.

  return AsmPrinter::PrintAsmMemoryOperand(MI, OpNo, ExtraCode, OS);
}

// Force static initialization.
extern "C" void LLVMInitializeWebAssemblyAsmPrinter() {
  RegisterAsmPrinter<WebAssemblyAsmPrinter> X(getTheWebAssemblyTarget32());
  RegisterAsmPrinter<WebAssemblyAsmPrinter> Y(getTheWebAssemblyTarget64());
}
