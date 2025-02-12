//===- DXILRootSignature.cpp - DXIL Root Signature helper objects -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects and APIs for working with DXIL
///       Root Signatures.
///
//===----------------------------------------------------------------------===//
#include "DXILRootSignature.h"
#include "DirectX.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <optional>
#include <utility>

using namespace llvm;
using namespace llvm::dxil;

static bool reportError(LLVMContext *Ctx, Twine Message,
                        DiagnosticSeverity Severity = DS_Error) {
  Ctx->diagnose(DiagnosticInfoGeneric(Message, Severity));
  return true;
}

static bool parseRootFlags(LLVMContext *Ctx, ModuleRootSignature &MRS,
                           MDNode *RootFlagNode) {

  if (RootFlagNode->getNumOperands() != 2)
    return reportError(Ctx, "Invalid format for RootFlag Element");

  auto *Flag = mdconst::extract<ConstantInt>(RootFlagNode->getOperand(1));
  MRS.Flags = Flag->getZExtValue();

  return false;
}

static bool parseRootSignatureElement(LLVMContext *Ctx,
                                      ModuleRootSignature &MRS,
                                      MDNode *Element) {
  MDString *ElementText = cast<MDString>(Element->getOperand(0));
  if (ElementText == nullptr)
    return reportError(Ctx, "Invalid format for Root Element");

  RootSignatureElementKind ElementKind =
      StringSwitch<RootSignatureElementKind>(ElementText->getString())
          .Case("RootFlags", RootSignatureElementKind::RootFlags)
          .Default(RootSignatureElementKind::None);

  switch (ElementKind) {

  case RootSignatureElementKind::RootFlags:
    return parseRootFlags(Ctx, MRS, Element);
  case RootSignatureElementKind::None:
    return reportError(Ctx, "Invalid Root Signature Element: " +
                                ElementText->getString());
  }

  llvm_unreachable("Root signature element kind not expected.");
}

static bool parse(LLVMContext *Ctx, ModuleRootSignature &MRS, MDNode *Node) {
  bool HasError = false;

  // Loop through the Root Elements of the root signature.
  for (const auto &Operand : Node->operands()) {
    MDNode *Element = dyn_cast<MDNode>(Operand);
    if (Element == nullptr)
      return reportError(Ctx, "Missing Root Element Metadata Node.");

    HasError = HasError || parseRootSignatureElement(Ctx, MRS, Element);
  }

  return HasError;
}

static bool validate(LLVMContext *Ctx, const ModuleRootSignature &MRS) {
  if (!dxbc::RootSignatureValidations::isValidRootFlag(MRS.Flags)) {
    return reportError(Ctx, "Invalid Root Signature flag value");
  }
  return false;
}

static SmallDenseMap<const Function *, ModuleRootSignature>
analyzeModule(Module &M) {

  /** Root Signature are specified as following in the metadata:

    !dx.rootsignatures = !{!2} ; list of function/root signature pairs
    !2 = !{ ptr @main, !3 } ; function, root signature
    !3 = !{ !4, !5, !6, !7 } ; list of root signature elements

    So for each MDNode inside dx.rootsignatures NamedMDNode
    (the Root parameter of this function), the parsing process needs
    to loop through each of its operands and process the function,
    signature pair.
 */

  LLVMContext *Ctx = &M.getContext();

  SmallDenseMap<const Function *, ModuleRootSignature> MRSMap;

  NamedMDNode *RootSignatureNode = M.getNamedMetadata("dx.rootsignatures");
  if (RootSignatureNode == nullptr)
    return MRSMap;

  for (const auto &RSDefNode : RootSignatureNode->operands()) {
    if (RSDefNode->getNumOperands() != 2) {
      reportError(Ctx, "Invalid format for Root Signature Definition. Pairs "
                       "of function, root signature expected.");
      continue;
    }

    // Function was pruned during compilation.
    const MDOperand &FunctionPointerMdNode = RSDefNode->getOperand(0);
    if (FunctionPointerMdNode == nullptr) {
      continue;
    }

    ValueAsMetadata *VAM =
        llvm::dyn_cast<ValueAsMetadata>(FunctionPointerMdNode.get());
    if (VAM == nullptr) {
      reportError(Ctx, "First element of root signature is not a Value");
      continue;
    }

    Function *F = dyn_cast<Function>(VAM->getValue());
    if (F == nullptr) {
      reportError(Ctx, "First element of root signature is not a Function");
      continue;
    }

    MDNode *RootElementListNode =
        dyn_cast<MDNode>(RSDefNode->getOperand(1).get());

    if (RootElementListNode == nullptr) {
      reportError(Ctx, "Missing Root Element List Metadata node.");
    }

    ModuleRootSignature MRS;

    if (parse(Ctx, MRS, RootElementListNode) || validate(Ctx, MRS)) {
      return MRSMap;
    }

    MRSMap.insert(std::make_pair(F, MRS));
  }

  return MRSMap;
}

AnalysisKey RootSignatureAnalysis::Key;

SmallDenseMap<const Function *, ModuleRootSignature>
RootSignatureAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  return analyzeModule(M);
}

//===----------------------------------------------------------------------===//

static void printSpaces(raw_ostream &Stream, unsigned int Count) {
  for (unsigned int I = 0; I < Count; ++I) {
    Stream << ' ';
  }
}

PreservedAnalyses RootSignatureAnalysisPrinter::run(Module &M,
                                                    ModuleAnalysisManager &AM) {

  SmallDenseMap<const Function *, ModuleRootSignature> &MRSMap =
      AM.getResult<RootSignatureAnalysis>(M);
  OS << "Root Signature Definitions"
     << "\n";
  uint8_t Space = 0;
  for (const auto &P : MRSMap) {
    const auto &[Function, MRS] = P;
    OS << "Definition for '" << Function->getName() << "':\n";

    // start root signature header
    Space++;
    printSpaces(OS, Space);
    OS << "Flags: " << format_hex(MRS.Flags, 8) << ":\n";
    Space--;
    // end root signature header
  }

  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
bool RootSignatureAnalysisWrapper::runOnModule(Module &M) {
  MRS = analyzeModule(M);
  return false;
}

void RootSignatureAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DXILMetadataAnalysisWrapperPass>();
}

char RootSignatureAnalysisWrapper::ID = 0;

INITIALIZE_PASS_BEGIN(RootSignatureAnalysisWrapper,
                      "dxil-root-signature-analysis",
                      "DXIL Root Signature Analysis", true, true)
INITIALIZE_PASS_END(RootSignatureAnalysisWrapper,
                    "dxil-root-signature-analysis",
                    "DXIL Root Signature Analysis", true, true)
