//===- DXILSignatureAnalysis.cpp - Semantic signatures
//---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILSignatureAnalysis.h"
#include "DirectX.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Frontend/HLSL/SemanticSignaturePacking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace llvm;
using namespace llvm::dxil;
using namespace llvm::hlsl;

namespace {

Error signatureError(const Twine &Message) {
  return createStringError(inconvertibleErrorCode(), Message);
}

bool isSignatureAccess(const IntrinsicInst &I) {
  return I.getIntrinsicID() == Intrinsic::dx_load_input ||
         I.getIntrinsicID() == Intrinsic::dx_store_output;
}

// Validate widths before fromMetadata extracts integer values or narrows them.
Error checkElementNode(const MDNode *Node) {
  if (!Node || Node->getNumOperands() != 13)
    return signatureError("expected a 13-operand signature element");
  for (unsigned I = 0; I != 13; ++I) {
    if (I == 1 || I == 4)
      continue;
    auto *CI = mdconst::dyn_extract_or_null<ConstantInt>(Node->getOperand(I));
    unsigned Width = (I == 7 || (I >= 9 && I <= 11)) ? 8 : 32;
    if (!CI || CI->getBitWidth() != Width)
      return signatureError("expected i" + Twine(Width) + " at operand " +
                            Twine(I));
  }
  if (!isa_and_nonnull<MDString>(Node->getOperand(1)))
    return signatureError("expected semantic name string");
  auto *Indices = dyn_cast_or_null<MDNode>(Node->getOperand(4));
  if (!Indices)
    return signatureError("expected semantic indices node");
  for (const MDOperand &Index : Indices->operands()) {
    auto *CI = mdconst::dyn_extract_or_null<ConstantInt>(Index);
    if (!CI || CI->getBitWidth() != 32)
      return signatureError("expected i32 semantic index");
  }
  return Error::success();
}

bool isIntegerComponent(ElementType Type) {
  return Type == ElementType::I1 || Type == ElementType::I16 ||
         Type == ElementType::U16 || Type == ElementType::I32 ||
         Type == ElementType::U32;
}

Error parseSignature(Metadata *MD,
                     SmallVectorImpl<SemanticSignatureElement> &Out,
                     ModuleSignatureInfo &Info, Triple::EnvironmentType Stage,
                     IOType IO) {
  if (!MD)
    return Error::success();
  auto *List = dyn_cast<MDNode>(MD);
  if (!List)
    return signatureError("expected a signature element list or null");
  if (List->getNumOperands() > 128)
    return signatureError("too many signature elements");

  std::set<std::pair<std::string, uint32_t>> Semantics;
  for (const MDOperand &Op : List->operands()) {
    auto *Node = dyn_cast_or_null<MDNode>(Op);
    if (Error Err = checkElementNode(Node))
      return Err;
    auto Parsed = SemanticSignatureElement::fromMetadata(Node);
    if (!Parsed)
      return Parsed.takeError();
    SemanticSignatureElement E = std::move(*Parsed);
    if (E.SigId != Out.size())
      return signatureError("signature IDs must be dense and in list order");
    if (Stage != Triple::Vertex && Stage != Triple::Pixel)
      return signatureError("nonempty signatures are currently supported only "
                            "for vertex and pixel entries");
    if (!E.Rows || E.Rows > MaxSignatureRows)
      return signatureError("signature row count must be within 1-32");
    if (E.SemanticName.empty() || E.SemanticName.contains('\0'))
      return signatureError("invalid semantic name");
    if (getSemanticKind(E.SemanticName) != E.SemanticKind)
      return signatureError("semantic name and kind disagree");
    if (E.CompType != ElementType::F16 && E.CompType != ElementType::F32 &&
        !isIntegerComponent(E.CompType))
      return signatureError("unsupported signature component type");
    if (E.GSStream != 0)
      return signatureError("nonzero stream requires a geometry output");
    auto Interpretation = getInterpretationKind(E.SemanticKind, Stage, IO);
    if (Interpretation == SemanticInterpretation::Invalid)
      return signatureError("invalid or unsupported semantic for this stage "
                            "and signature category");
    if (Interpretation == SemanticInterpretation::NotAllocated)
      return signatureError("semantic accessed by a dedicated intrinsic must "
                            "not appear in an input/output signature");
    if (E.isAllocated() &&
        (E.StartRow >= MaxSignatureRows ||
         E.Rows > MaxSignatureRows - E.StartRow || E.StartCol + E.Cols > 4))
      return signatureError(
          "allocated signature element exceeds register bounds");
    if (E.InterpMode == dxbc::PSV::InterpolationMode::Invalid)
      return signatureError("invalid interpolation mode");

    for (uint32_t Index : E.SemanticIndices)
      if (!Semantics.emplace(E.SemanticName.upper(), Index).second)
        return signatureError("duplicate semantic name and index");
    if (Interpretation == SemanticInterpretation::Target &&
        (E.Rows != 1 || E.SemanticIndices.front() >= 8))
      return signatureError(
          "pixel targets require one row and an index within 0-7");
    if (Interpretation == SemanticInterpretation::SV &&
        E.SemanticKind == dxbc::PSV::SemanticKind::Position &&
        (E.Rows != 1 || E.Cols != 4 || E.CompType != ElementType::F32))
      return signatureError(
          "SV_Position requires one row of four f32 components");
    if (E.SemanticKind == dxbc::PSV::SemanticKind::VertexID &&
        (E.Rows != 1 || E.Cols != 1 ||
         (E.CompType != ElementType::I32 && E.CompType != ElementType::U32)))
      return signatureError("SV_VertexID requires a scalar 32-bit integer");

    // SV names can be interpreted as arbitrary semantics at a signature point.
    if (Interpretation == SemanticInterpretation::Arbitrary)
      E.SemanticKind = dxbc::PSV::SemanticKind::Arbitrary;
    if (Stage == Triple::Pixel && IO == IOType::In) {
      using Mode = dxbc::PSV::InterpolationMode;
      if (E.InterpMode == Mode::Undefined)
        E.InterpMode = isIntegerComponent(E.CompType) ? Mode::Constant
                       : E.SemanticKind == dxbc::PSV::SemanticKind::Position
                           ? Mode::LinearNoperspective
                           : Mode::Linear;
      if (isIntegerComponent(E.CompType) && E.InterpMode != Mode::Constant)
        return signatureError(
            "integer pixel inputs require constant interpolation");
    }
    E.SemanticName = Info.Names.insert(E.SemanticName).first->getKey();
    // Recompute masks from the surviving signature accesses, not stale
    // metadata.
    E.UsageMask = E.DynIndexMask = 0;
    Out.push_back(std::move(E));
  }
  return Error::success();
}

Error packSignature(SmallVectorImpl<SemanticSignatureElement> &Elements,
                    Triple::EnvironmentType Stage, IOType IO, bool Native16,
                    unsigned &Extent) {
  if (Elements.empty())
    return Error::success();
  SmallVector<std::pair<uint32_t, uint8_t>> Locations;
  bool AnyAllocated = false, AllAllocated = true;
  for (auto &E : Elements) {
    AnyAllocated |= E.isAllocated();
    AllAllocated &= E.isAllocated();
    Locations.emplace_back(E.StartRow, E.StartCol);
    E.StartRow = UnallocatedRow;
    E.StartCol = UnallocatedCol;
  }
  if (AnyAllocated && !AllAllocated)
    return signatureError("partially allocated signatures are not supported");
  Expected<unsigned> Packed =
      Stage == Triple::Vertex && IO == IOType::In
          ? packSignatureStacked(Elements, Stage, IO)
      : Stage == Triple::Pixel && IO == IOType::Out
          ? packSignatureIndexed(Elements, Stage, IO)
          : packSignaturePrefixStable(Elements, Stage, IO, Native16);
  if (!Packed)
    return Packed.takeError();
  Extent = *Packed;
  for (auto [E, Loc] : zip(Elements, Locations)) {
    if (AllAllocated && Loc != std::make_pair(E.StartRow, E.StartCol))
      return signatureError("preallocated signature does not match the "
                            "stage's packing layout");
    E.UsageMask <<= E.StartCol;
  }
  return Error::success();
}

Error analyzeAccess(const IntrinsicInst &I, EntrySignature &Sig) {
  bool Input = I.getIntrinsicID() == Intrinsic::dx_load_input;
  auto &Elements = Input ? Sig.Inputs : Sig.Outputs;
  auto *ID = dyn_cast<ConstantInt>(I.getArgOperand(0));
  auto *Col = dyn_cast<ConstantInt>(I.getArgOperand(2));
  if (!ID || ID->getZExtValue() >= Elements.size())
    return signatureError("signature access has an invalid element ID");
  auto &E = Elements[ID->getZExtValue()];
  Type *Ty = Input ? I.getType() : I.getArgOperand(3)->getType();
  unsigned Width = 1;
  if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
    Width = VT->getNumElements();
    Ty = VT->getElementType();
  }
  if (!Col || Col->getZExtValue() >= E.Cols ||
      Width > E.Cols - Col->getZExtValue())
    return signatureError("signature access has an invalid component index");
  auto *Row = dyn_cast<ConstantInt>(I.getArgOperand(1));
  if (Row && Row->getZExtValue() >= E.Rows)
    return signatureError("signature access has an invalid row index");
  if (!Row && E.Rows == 1)
    return signatureError(
        "dynamic indexing requires a multi-row signature element");
  bool CorrectType = false;
  switch (E.CompType) {
  case ElementType::F16:
    CorrectType = Ty->isHalfTy();
    break;
  case ElementType::F32:
    CorrectType = Ty->isFloatTy();
    break;
  case ElementType::I16:
  case ElementType::U16:
    CorrectType = Ty->isIntegerTy(16);
    break;
  case ElementType::I1:
    // Boolean signature elements use the i32 load/store overload.
    CorrectType = Ty->isIntegerTy(32);
    break;
  case ElementType::I32:
  case ElementType::U32:
    CorrectType = Ty->isIntegerTy(32);
    break;
  default:
    llvm_unreachable("validated signature component type");
  }
  if (!CorrectType)
    return signatureError("signature access type disagrees with its element");
  uint8_t Mask = ((1U << Width) - 1) << Col->getZExtValue();
  // As in DXC, the input usage mask includes conditional reads as well.
  E.UsageMask |= Mask;
  if (!Row)
    E.DynIndexMask |= Mask;
  return Error::success();
}

void buildDependencyMap(EntrySignature &Sig) {
  unsigned Words = (Sig.OutputVectors + 7) / 8;
  Sig.InputOutputMap.assign(Sig.InputVectors * 4 * Words, 0);
  // Conservatively connect every accessed input component to every written
  // output component. This includes control, memory, call, and dynamic-index
  // dependencies without incorrectly claiming that unknown dependencies are
  // absent. Unused components and padding remain zero. Both DXIL and PSV use
  // this same map; a more precise dependency analysis can refine it later.
  for (const auto &Input : Sig.Inputs)
    for (unsigned IR = 0; IR != Input.Rows; ++IR)
      for (unsigned IC = 0; IC != 4; ++IC) {
        if (!(Input.UsageMask & (1U << IC)))
          continue;
        unsigned Base = ((Input.StartRow + IR) * 4 + IC) * Words;
        for (const auto &Output : Sig.Outputs)
          for (unsigned OR = 0; OR != Output.Rows; ++OR)
            for (unsigned OC = 0; OC != 4; ++OC) {
              if (!(Output.UsageMask & (1U << OC)))
                continue;
              unsigned Bit = (Output.StartRow + OR) * 4 + OC;
              Sig.InputOutputMap[Base + Bit / 32] |= 1U << (Bit % 32);
            }
      }
}

Expected<ModuleSignatureInfo> analyzeModule(Module &M,
                                            const ModuleMetadataInfo &MMI) {
  ModuleSignatureInfo Info;
  if (NamedMDNode *Table = M.getNamedMetadata("dx.semantic.signatures")) {
    for (const MDNode *Record : Table->operands()) {
      if (Record->getNumOperands() != 3)
        return signatureError(
            "expected an entry/input/output signature triple");
      auto *VAM = dyn_cast_or_null<ValueAsMetadata>(Record->getOperand(0));
      auto *F = VAM ? dyn_cast<Function>(VAM->getValue()) : nullptr;
      if (!F || F->getParent() != &M || F->isDeclaration())
        return signatureError(
            "signature entry must be a defined function in the module");
      auto EP =
          llvm::find_if(MMI.EntryPropertyVec,
                        [F](const EntryProperties &E) { return E.Entry == F; });
      if (EP == MMI.EntryPropertyVec.end())
        return signatureError("signature function is not a shader entry");
      if (Info.Entries.contains(F))
        return signatureError("duplicate signature record for entry '" +
                              F->getName() + "'");
      EntrySignature Sig;
      Sig.Stage = EP->ShaderStage;
      if (auto *Native = mdconst::dyn_extract_or_null<ConstantInt>(
              M.getModuleFlag("dx.nativelowprec")))
        Sig.UseNative16Bit = !Native->isZero();
      if (Sig.UseNative16Bit && MMI.ShaderModelVersion < VersionTuple(6, 2))
        return signatureError(
            "native 16-bit signatures require shader model 6.2");
      auto AddContext = [&](Error Err, StringRef Category) -> Error {
        return signatureError("entry '" + F->getName() + "' " + Category +
                              " signature: " + toString(std::move(Err)));
      };
      if (Error Err = parseSignature(Record->getOperand(1), Sig.Inputs, Info,
                                     Sig.Stage, IOType::In))
        return AddContext(std::move(Err), "input");
      if (Error Err = parseSignature(Record->getOperand(2), Sig.Outputs, Info,
                                     Sig.Stage, IOType::Out))
        return AddContext(std::move(Err), "output");
      for (const Instruction &I : instructions(F))
        if (auto *II = dyn_cast<IntrinsicInst>(&I);
            II && isSignatureAccess(*II))
          if (Error Err = analyzeAccess(*II, Sig))
            return signatureError("entry '" + F->getName() +
                                  "': " + toString(std::move(Err)));
      if (Error Err = packSignature(Sig.Inputs, Sig.Stage, IOType::In,
                                    Sig.UseNative16Bit, Sig.InputVectors))
        return AddContext(std::move(Err), "input");
      if (Error Err = packSignature(Sig.Outputs, Sig.Stage, IOType::Out,
                                    Sig.UseNative16Bit, Sig.OutputVectors))
        return AddContext(std::move(Err), "output");
      buildDependencyMap(Sig);
      Info.Entries.try_emplace(F, std::move(Sig));
    }
  }
  // Signature IDs are local to an entry. Do not guess which entry a helper's
  // accesses belong to, even if that helper happens to have a single caller.
  for (const Function &F : M)
    if (!Info.Entries.contains(&F))
      for (const Instruction &I : instructions(F))
        if (auto *II = dyn_cast<IntrinsicInst>(&I);
            II && isSignatureAccess(*II))
          return signatureError("signature access in function '" + F.getName() +
                                "' has no entry signature metadata");
  return std::move(Info);
}

ModuleSignatureInfo collectSignatures(Module &M,
                                      const ModuleMetadataInfo &MMI) {
  auto Result = analyzeModule(M, MMI);
  if (!Result) {
    M.getContext().emitError("Invalid semantic signature: " +
                             toString(Result.takeError()));
    return ModuleSignatureInfo();
  }
  return std::move(*Result);
}

MDNode *emitSignature(LLVMContext &Ctx,
                      ArrayRef<SemanticSignatureElement> Elements,
                      VersionTuple ValidatorVersion) {
  if (Elements.empty())
    return nullptr;
  auto I32 = [&](uint32_t V) {
    return ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), V));
  };
  auto I8 = [&](uint8_t V) {
    return ConstantAsMetadata::get(ConstantInt::get(Type::getInt8Ty(Ctx), V));
  };
  SmallVector<Metadata *> Nodes;
  for (const auto &E : Elements) {
    SmallVector<Metadata *> Indices, Props;
    for (uint32_t Index : E.SemanticIndices)
      Indices.push_back(I32(Index));
    if (E.GSStream)
      Props.append({I32(0), I32(E.GSStream)});
    if (E.DynIndexMask)
      Props.append({I32(2), I32(E.DynIndexMask)});
    if (E.UsageMask &&
        (ValidatorVersion.empty() || ValidatorVersion == VersionTuple(0, 0) ||
         ValidatorVersion >= VersionTuple(1, 5)))
      Props.append({I32(3), I32(E.UsageMask >> E.StartCol)});
    Nodes.push_back(MDNode::get(
        Ctx, {I32(E.SigId), MDString::get(Ctx, E.SemanticName),
              I8(static_cast<uint8_t>(E.CompType)),
              I8(static_cast<uint8_t>(E.SemanticKind)),
              MDNode::get(Ctx, Indices), I8(static_cast<uint8_t>(E.InterpMode)),
              I32(E.Rows), I8(E.Cols), I32(E.StartRow), I8(E.StartCol),
              Props.empty() ? nullptr : MDNode::get(Ctx, Props)}));
  }
  return MDNode::get(Ctx, Nodes);
}

} // namespace

MDTuple *EntrySignature::getAsMetadata(LLVMContext &Ctx,
                                       VersionTuple ValidatorVersion) const {
  if (Inputs.empty() && Outputs.empty())
    return nullptr;
  return MDNode::get(Ctx,
                     {emitSignature(Ctx, Inputs, ValidatorVersion),
                      emitSignature(Ctx, Outputs, ValidatorVersion), nullptr});
}

SmallVector<uint32_t> EntrySignature::getDependencyState() const {
  SmallVector<uint32_t> State = {InputVectors * 4, OutputVectors * 4};
  llvm::append_range(State, InputOutputMap);
  return State;
}

void EntrySignature::print(raw_ostream &OS) const {
  auto Print = [&](StringRef Name, ArrayRef<SemanticSignatureElement> Elements,
                   unsigned Vectors) {
    OS << "  " << Name << ": " << Elements.size() << " elements, " << Vectors
       << " vectors\n";
    for (const auto &E : Elements)
      OS << "    " << E.SigId << ": " << E.SemanticName << " rows=" << E.Rows
         << " cols=" << unsigned(E.Cols) << " at " << E.StartRow << ":"
         << unsigned(E.StartCol) << " usage=" << unsigned(E.UsageMask)
         << " dynamic=" << unsigned(E.DynIndexMask) << '\n';
  };
  Print("Inputs", Inputs, InputVectors);
  Print("Outputs", Outputs, OutputVectors);
}

void ModuleSignatureInfo::print(raw_ostream &OS, const Module &M) const {
  for (const Function &F : M)
    if (const auto *Sig = get(&F)) {
      OS << "Semantic signatures for '" << F.getName() << "':\n";
      Sig->print(OS);
    }
}

AnalysisKey SignatureAnalysis::Key;
ModuleSignatureInfo SignatureAnalysis::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  return collectSignatures(M, AM.getResult<DXILMetadataAnalysis>(M));
}

PreservedAnalyses SignatureAnalysisPrinter::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  AM.getResult<SignatureAnalysis>(M).print(OS, M);
  return PreservedAnalyses::all();
}

bool SignatureAnalysisWrapper::runOnModule(Module &M) {
  Info = std::make_unique<ModuleSignatureInfo>(collectSignatures(
      M, getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata()));
  return false;
}

void SignatureAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DXILMetadataAnalysisWrapperPass>();
}

void SignatureAnalysisWrapper::print(raw_ostream &OS, const Module *M) const {
  if (Info)
    Info->print(OS, *M);
}

char SignatureAnalysisWrapper::ID;
INITIALIZE_PASS_BEGIN(SignatureAnalysisWrapper, "dxil-signature-analysis",
                      "DXIL Semantic Signature Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_END(SignatureAnalysisWrapper, "dxil-signature-analysis",
                    "DXIL Semantic Signature Analysis", false, true)
