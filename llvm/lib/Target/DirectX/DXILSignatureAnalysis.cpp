//===- DXILSignatureAnalysis.cpp - Semantic signatures --------------------===//
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
#include "llvm/MC/DXContainerPSVInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <set>

using namespace llvm;
using namespace llvm::dxil;
using namespace llvm::hlsl;

static Error signatureError(const Twine &Message) {
  return createStringError(inconvertibleErrorCode(), Message);
}

static bool isSignatureAccess(const IntrinsicInst &I) {
  return I.getIntrinsicID() == Intrinsic::dx_load_input ||
         I.getIntrinsicID() == Intrinsic::dx_store_output;
}

// Validate widths before fromMetadata extracts integer values or narrows them.
static Error checkElementNode(const MDNode *Node) {
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

static bool isIntegerComponent(ElementType Type) {
  return Type == ElementType::I1 || Type == ElementType::I16 ||
         Type == ElementType::U16 || Type == ElementType::I32 ||
         Type == ElementType::U32;
}

static Error diagnoseSemanticType(const SemanticSignatureElement &E) {
  switch (E.SemanticKind) {
  case dxbc::PSV::SemanticKind::Target:
    if (E.CompType != ElementType::F16 && E.CompType != ElementType::F32)
      return signatureError(
          "SV_Target requires 16- or 32-bit floating-point components");
    break;
  case dxbc::PSV::SemanticKind::Position:
    if (E.Rows != 1 || E.Cols != 4 ||
        (E.CompType != ElementType::F16 && E.CompType != ElementType::F32))
      return signatureError(
          "SV_Position requires one row of four floating-point components");
    break;
  case dxbc::PSV::SemanticKind::VertexID:
    if (E.Rows != 1 || E.Cols != 1 ||
        (E.CompType != ElementType::I16 && E.CompType != ElementType::U16 &&
         E.CompType != ElementType::I32 && E.CompType != ElementType::U32))
      return signatureError(
          "SV_VertexID requires a scalar 16- or 32-bit integer");
    break;
  case dxbc::PSV::SemanticKind::InstanceID:
    if (E.Rows != 1 || E.Cols != 1 ||
        (E.CompType != ElementType::U16 && E.CompType != ElementType::U32))
      return signatureError(
          "SV_InstanceID requires a scalar 16- or 32-bit unsigned integer");
    break;
  case dxbc::PSV::SemanticKind::ShadingRate:
    if (E.Rows != 1 || E.Cols != 1 ||
        (E.CompType != ElementType::U16 && E.CompType != ElementType::U32))
      return signatureError(
          "SV_ShadingRate requires a scalar 16- or 32-bit unsigned integer");
    break;
  case dxbc::PSV::SemanticKind::SampleIndex:
    if (E.Rows != 1 || E.Cols != 1 ||
        (E.CompType != ElementType::U16 && E.CompType != ElementType::U32))
      return signatureError(
          "SV_SampleIndex requires a scalar 16- or 32-bit unsigned integer");
    break;
  case dxbc::PSV::SemanticKind::PrimitiveID:
    if (E.Rows != 1 || E.Cols != 1 ||
        (E.CompType != ElementType::U16 && E.CompType != ElementType::U32))
      return signatureError(
          "SV_PrimitiveID requires a scalar 16- or 32-bit unsigned integer");
    break;
  case dxbc::PSV::SemanticKind::StencilRef:
    if (E.Rows != 1 || E.Cols != 1 ||
        (E.CompType != ElementType::U16 && E.CompType != ElementType::U32))
      return signatureError(
          "SV_StencilRef requires a scalar 16- or 32-bit unsigned integer");
    break;
  case dxbc::PSV::SemanticKind::ViewPortArrayIndex:
    if (E.Rows != 1 || E.Cols != 1 ||
        (E.CompType != ElementType::U16 && E.CompType != ElementType::U32))
      return signatureError("SV_ViewportArrayIndex requires a scalar 16- or "
                            "32-bit unsigned integer");
    break;
  case dxbc::PSV::SemanticKind::RenderTargetArrayIndex:
    if (E.Rows != 1 || E.Cols != 1 ||
        (E.CompType != ElementType::U16 && E.CompType != ElementType::U32))
      return signatureError("SV_RenderTargetArrayIndex requires a scalar 16- "
                            "or 32-bit unsigned integer");
    break;
  case dxbc::PSV::SemanticKind::Barycentrics:
    if (E.Rows != 1 || E.Cols != 3 ||
        (E.CompType != ElementType::F16 && E.CompType != ElementType::F32))
      return signatureError("SV_Barycentrics requires one row of three 16- or "
                            "32-bit floating-point components");
    break;
  default:
    break;
  }
  return Error::success();
}

static Error diagnoseSemanticIndex(const SemanticSignatureElement &E) {
  switch (E.SemanticKind) {
  case dxbc::PSV::SemanticKind::Target:
    if (E.Rows != 1 || E.SemanticIndices.front() >= 8)
      return signatureError(
          "pixel targets require one row and an index within 0-7");
    break;
  case dxbc::PSV::SemanticKind::VertexID:
    if (E.SemanticIndices.front() != 0)
      return signatureError("SV_VertexID requires semantic index 0");
    break;
  case dxbc::PSV::SemanticKind::Barycentrics:
    if (E.SemanticIndices.front() != 0)
      return signatureError("SV_Barycentrics requires semantic index 0");
    break;
  case dxbc::PSV::SemanticKind::SampleIndex:
    if (E.SemanticIndices.front() != 0)
      return signatureError("SV_SampleIndex requires semantic index 0");
    break;
  case dxbc::PSV::SemanticKind::PrimitiveID:
    if (E.SemanticIndices.front() != 0)
      return signatureError("SV_PrimitiveID requires semantic index 0");
    break;
  case dxbc::PSV::SemanticKind::StencilRef:
    if (E.SemanticIndices.front() != 0)
      return signatureError("SV_StencilRef requires semantic index 0");
    break;
  case dxbc::PSV::SemanticKind::ViewPortArrayIndex:
    if (E.SemanticIndices.front() != 0)
      return signatureError("SV_ViewportArrayIndex requires semantic index 0");
    break;
  case dxbc::PSV::SemanticKind::RenderTargetArrayIndex:
    if (E.SemanticIndices.front() != 0)
      return signatureError(
          "SV_RenderTargetArrayIndex requires semantic index 0");
    break;
  default:
    break;
  }
  return Error::success();
}

static Error
diagnoseSemanticShaderModel(const SemanticSignatureElement &E,
                            const VersionTuple &ShaderModelVersion) {
  switch (E.SemanticKind) {
  case dxbc::PSV::SemanticKind::Barycentrics:
    if (ShaderModelVersion < VersionTuple(6, 1))
      return signatureError("SV_Barycentrics requires shader model 6.1");
    break;
  default:
    break;
  }
  return Error::success();
}

static Error parseSignature(Metadata *MD,
                            SmallVectorImpl<SemanticSignatureElement> &Out,
                            ModuleSignatureInfo &Info,
                            Triple::EnvironmentType Stage, IOType IO,
                            const VersionTuple &ShaderModelVersion,
                            bool Native16, unsigned &Extent) {
  if (!MD)
    return Error::success();
  auto *List = dyn_cast<MDNode>(MD);
  if (!List)
    return signatureError("expected a signature element list or null");

  std::set<std::pair<std::string, uint32_t>> Semantics;
  constexpr unsigned MaxAllocatedElements = MaxSignatureRows * 4;
  unsigned NumAllocated = 0;
  std::array<uint8_t, MaxSignatureRows> AllocatedColumns = {};
  std::array<dxbc::PSV::InterpolationMode, MaxSignatureRows> RowModes = {};
  std::array<bool, MaxSignatureRows> RowNative16 = {};
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
    if (E.CompType != ElementType::F16 && E.CompType != ElementType::F32 &&
        !isIntegerComponent(E.CompType))
      return signatureError("unsupported signature component type");
    if (E.GSStream != 0)
      return signatureError("nonzero stream requires a geometry output");
    auto NamedKind = getSemanticKind(E.SemanticName);
    auto Interpretation = getInterpretationKind(NamedKind, Stage, IO);
    if (Interpretation == SemanticInterpretation::Invalid)
      return signatureError("invalid or unsupported semantic for this stage "
                            "and signature category");
    if (Interpretation == SemanticInterpretation::NotAllocated)
      return signatureError("semantic accessed by a dedicated intrinsic must "
                            "not appear in an input/output signature");
    auto ExpectedKind = Interpretation == SemanticInterpretation::Arbitrary
                            ? dxbc::PSV::SemanticKind::Arbitrary
                            : NamedKind;
    if (E.SemanticKind != ExpectedKind)
      return signatureError("semantic name and kind disagree at this signature "
                            "point");
    if (Interpretation == SemanticInterpretation::NotPacked) {
      if (E.isAllocated())
        return signatureError(E.SemanticName +
                              " must not have an allocated location");
    } else {
      if (!E.isAllocated())
        return signatureError("signature element " + Twine(E.SigId) + " ('" +
                              E.SemanticName + "') has no allocated location");
      if (E.StartRow >= MaxSignatureRows ||
          E.Rows > MaxSignatureRows - E.StartRow || E.StartCol + E.Cols > 4)
        return signatureError(
            "allocated signature element exceeds register bounds");
    }
    if (E.InterpMode == dxbc::PSV::InterpolationMode::Invalid)
      return signatureError("invalid interpolation mode");

    for (uint32_t Index : E.SemanticIndices)
      if (!Semantics.emplace(E.SemanticName.upper(), Index).second)
        return signatureError("duplicate semantic name and index");
    if (Error Err = diagnoseSemanticType(E))
      return Err;
    if (Error Err = diagnoseSemanticIndex(E))
      return Err;
    if (Error Err = diagnoseSemanticShaderModel(E, ShaderModelVersion))
      return Err;
    if (E.SemanticKind == dxbc::PSV::SemanticKind::Barycentrics) {
      using Mode = dxbc::PSV::InterpolationMode;
      if (E.InterpMode == Mode::Undefined || E.InterpMode == Mode::Constant)
        return signatureError("SV_Barycentrics requires perspective or "
                              "noperspective interpolation");
    }

    if (Stage == Triple::Vertex && IO == IOType::In &&
        (E.StartRow != Extent || E.StartCol != 0))
      return signatureError("vertex inputs require stacked locations in "
                            "signature order (element " +
                            Twine(E.SigId) + ")");
    if (Interpretation == SemanticInterpretation::Target &&
        (E.StartRow != E.SemanticIndices.front() || E.StartCol != 0))
      return signatureError("pixel target location must match its semantic "
                            "index at column 0");
    if (Stage == Triple::Pixel && IO == IOType::In &&
        isIntegerComponent(E.CompType) &&
        E.InterpMode != dxbc::PSV::InterpolationMode::Constant)
      return signatureError(
          "integer pixel inputs require constant interpolation");

    if (E.isAllocated()) {
      if (++NumAllocated > MaxAllocatedElements)
        return signatureError("too many signature elements");
      // Co-packed elements must obey row constraints, even when their component
      // masks do not overlap. Min-precision types occupy 32-bit components.
      bool IsNative16 = Native16 && (E.CompType == ElementType::F16 ||
                                     E.CompType == ElementType::I16 ||
                                     E.CompType == ElementType::U16);
      uint8_t Mask = E.getDeclaredMask();
      for (unsigned Row = E.StartRow; Row != E.StartRow + E.Rows; ++Row) {
        if (AllocatedColumns[Row] & Mask)
          return signatureError("signature elements overlap at row " +
                                Twine(Row) + " (element " + Twine(E.SigId) +
                                ")");
        if (AllocatedColumns[Row] && RowModes[Row] != E.InterpMode)
          return signatureError(
              "signature elements have different interpolation "
              "modes at row " +
              Twine(Row) + " (element " + Twine(E.SigId) + ")");
        if (AllocatedColumns[Row] && RowNative16[Row] != IsNative16)
          return signatureError("signature elements have different component "
                                "widths at row " +
                                Twine(Row) + " (element " + Twine(E.SigId) +
                                ")");
        AllocatedColumns[Row] |= Mask;
        RowModes[Row] = E.InterpMode;
        RowNative16[Row] = IsNative16;
      }
      Extent = std::max(Extent, E.StartRow + E.Rows);
    }
    E.SemanticName = Info.Names.insert(E.SemanticName).first->getKey();

    // Access masks will be recomputed so reset to 0 here.
    E.UsageMask = 0;
    E.DynIndexMask = 0;
    Out.push_back(std::move(E));
  }
  return Error::success();
}

static Error analyzeAccess(const IntrinsicInst &I, EntrySignature &Sig) {
  bool Input = I.getIntrinsicID() == Intrinsic::dx_load_input;
  auto &Elements = Input ? Sig.Inputs : Sig.Outputs;
  auto *ID = dyn_cast<ConstantInt>(I.getArgOperand(0));
  auto *Col = dyn_cast<ConstantInt>(I.getArgOperand(2));
  StringRef Category = Input ? "input" : "output";
  assert(ID && "signature ID is marked as ImmArg and must be constant");
  if (ID->getZExtValue() >= Elements.size())
    return signatureError(Category +
                          " signature: signature access has an invalid element "
                          "ID " +
                          Twine(ID->getZExtValue()));
  auto &E = Elements[ID->getZExtValue()];
  auto AccessError = [&](const Twine &Message) {
    return signatureError(Category + " signature element " + Twine(E.SigId) +
                          " ('" + E.SemanticName + "'): " + Message);
  };
  if (E.SemanticKind == dxbc::PSV::SemanticKind::SampleIndex)
    return AccessError("must not be read with dx.op.load/store");
  Type *Ty = Input ? I.getType() : I.getArgOperand(3)->getType();
  unsigned Width = 1;
  if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
    Width = VT->getNumElements();
    Ty = VT->getElementType();
  }
  if (!Col || Col->getZExtValue() >= E.Cols ||
      Width > E.Cols - Col->getZExtValue())
    return AccessError("signature access has an invalid component index");
  auto *Row = dyn_cast<ConstantInt>(I.getArgOperand(1));
  if (Row && Row->getZExtValue() >= E.Rows)
    return AccessError("signature access has an invalid row index");
  if (!Row && E.Rows == 1)
    return AccessError(
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
    llvm_unreachable("invalid signature component type");
  }
  if (!CorrectType)
    return AccessError("signature access type disagrees with its element");
  uint8_t Mask = ((1U << Width) - 1) << Col->getZExtValue();
  E.UsageMask |= Mask << E.getStartColOrZero();
  if (!Row)
    E.DynIndexMask |= Mask;
  return Error::success();
}

static void buildDependencyMap(EntrySignature &Sig) {
  // Each output vector needs four component bits, so one 32-bit word holds
  // eight vectors. Adding seven rounds up to include a partially filled word.
  unsigned Words = (Sig.OutputVectors + 7) / 8;
  Sig.InputOutputMap.assign(Sig.InputVectors * 4 * Words, 0);
  // Conservatively connect every accessed input component to every written
  // output component. This includes control, memory, call, and dynamic-index
  // dependencies without incorrectly claiming that unknown dependencies are
  // absent. Unused components and padding remain zero. Both DXIL and PSV use
  // this same map; a more precise dependency analysis can refine it later.
  for (const auto &Input : Sig.Inputs) {
    if (!Input.isAllocated())
      continue;
    for (unsigned IR = 0; IR != Input.Rows; ++IR)
      for (unsigned IC = 0; IC != 4; ++IC) {
        if (!(Input.UsageMask & (1U << IC)))
          continue;
        unsigned Base = ((Input.StartRow + IR) * 4 + IC) * Words;
        for (const auto &Output : Sig.Outputs) {
          // Unallocated outputs, such as SV_StencilRef, have no row in the
          // dependency table.
          if (!Output.isAllocated())
            continue;
          for (unsigned OR = 0; OR != Output.Rows; ++OR)
            for (unsigned OC = 0; OC != 4; ++OC) {
              if (!(Output.UsageMask & (1U << OC)))
                continue;
              unsigned Bit = (Output.StartRow + OR) * 4 + OC;
              Sig.InputOutputMap[Base + Bit / 32] |= 1U << (Bit % 32);
            }
        }
      }
  }
}

static Expected<ModuleSignatureInfo>
analyzeModule(Module &M, const ModuleMetadataInfo &MMI) {
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
      if (Error Err = parseSignature(
              Record->getOperand(1), Sig.Inputs, Info, Sig.Stage, IOType::In,
              MMI.ShaderModelVersion, Sig.UseNative16Bit, Sig.InputVectors))
        return AddContext(std::move(Err), "input");
      if (Error Err = parseSignature(
              Record->getOperand(2), Sig.Outputs, Info, Sig.Stage, IOType::Out,
              MMI.ShaderModelVersion, Sig.UseNative16Bit, Sig.OutputVectors))
        return AddContext(std::move(Err), "output");
      for (const Instruction &I : instructions(F))
        if (auto *II = dyn_cast<IntrinsicInst>(&I);
            II && isSignatureAccess(*II))
          if (Error Err = analyzeAccess(*II, Sig))
            return signatureError("entry '" + F->getName() +
                                  "': " + toString(std::move(Err)));
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

static ModuleSignatureInfo collectSignatures(Module &M,
                                             const ModuleMetadataInfo &MMI) {
  auto Result = analyzeModule(M, MMI);
  if (!Result) {
    M.getContext().emitError("Invalid semantic signature: " +
                             toString(Result.takeError()));
    return ModuleSignatureInfo();
  }
  return std::move(*Result);
}

static MDNode *emitSignature(LLVMContext &Ctx,
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
      Props.append({I32(3), I32(E.UsageMask >> E.getStartColOrZero())});
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

static dxbc::SigComponentType getComponentType(ElementType Type, bool Legacy) {
  switch (Type) {
  case ElementType::I1:
    return Legacy ? dxbc::SigComponentType::Unknown
                  : dxbc::SigComponentType::UInt32;
  case ElementType::I16:
    return dxbc::SigComponentType::SInt16;
  case ElementType::U16:
    return dxbc::SigComponentType::UInt16;
  case ElementType::I32:
    return dxbc::SigComponentType::SInt32;
  case ElementType::U32:
    return dxbc::SigComponentType::UInt32;
  case ElementType::F16:
    return dxbc::SigComponentType::Float16;
  case ElementType::F32:
    return dxbc::SigComponentType::Float32;
  default:
    llvm_unreachable("invalid signature component type");
  }
}

static dxbc::D3DSystemValue getSystemValue(dxbc::PSV::SemanticKind Kind) {
  using KindTy = dxbc::PSV::SemanticKind;
  switch (Kind) {
  case KindTy::Arbitrary:
    return dxbc::D3DSystemValue::Undefined;
  case KindTy::VertexID:
    return dxbc::D3DSystemValue::VertexID;
  case KindTy::InstanceID:
    return dxbc::D3DSystemValue::InstanceID;
  case KindTy::Position:
    return dxbc::D3DSystemValue::Position;
  case KindTy::IsFrontFace:
    return dxbc::D3DSystemValue::IsFrontFace;
  case KindTy::ClipDistance:
    return dxbc::D3DSystemValue::ClipDistance;
  case KindTy::CullDistance:
    return dxbc::D3DSystemValue::CullDistance;
  case KindTy::Target:
    return dxbc::D3DSystemValue::Target;
  case KindTy::ShadingRate:
    return dxbc::D3DSystemValue::ShadingRate;
  case KindTy::Barycentrics:
    return dxbc::D3DSystemValue::Barycentrics;
  case KindTy::SampleIndex:
    return dxbc::D3DSystemValue::SampleIndex;
  case KindTy::PrimitiveID:
    return dxbc::D3DSystemValue::PrimitiveID;
  case KindTy::StencilRef:
    return dxbc::D3DSystemValue::StencilRef;
  case KindTy::ViewPortArrayIndex:
    return dxbc::D3DSystemValue::ViewPortArrayIndex;
  case KindTy::RenderTargetArrayIndex:
    return dxbc::D3DSystemValue::RenderTargetArrayIndex;
  default:
    llvm_unreachable("validated vertex/pixel signature semantic");
  }
}

// The DXIL reader normalizes system-value semantic names when loading module
// metadata. The validator then recreates ISG1/OSG1 from those normalized names
// and compares the parts byte-for-byte, so the container must use the same
// spelling even if HLSL (or the source metadata) used a different casing.
static StringRef getContainerSemanticName(const SemanticSignatureElement &E) {
  using Kind = dxbc::PSV::SemanticKind;
  switch (E.SemanticKind) {
  case Kind::Arbitrary:
    return E.SemanticName;
  case Kind::VertexID:
    return "SV_VertexID";
  case Kind::InstanceID:
    return "SV_InstanceID";
  case Kind::Position:
    return "SV_Position";
  case Kind::IsFrontFace:
    return "SV_IsFrontFace";
  case Kind::ClipDistance:
    return "SV_ClipDistance";
  case Kind::CullDistance:
    return "SV_CullDistance";
  case Kind::Target:
    return "SV_Target";
  case Kind::ShadingRate:
    return "SV_ShadingRate";
  case Kind::Barycentrics:
    return "SV_Barycentrics";
  case Kind::SampleIndex:
    return "SV_SampleIndex";
  case Kind::PrimitiveID:
    return "SV_PrimitiveID";
  case Kind::StencilRef:
    return "SV_StencilRef";
  case Kind::ViewPortArrayIndex:
    return "SV_ViewportArrayIndex";
  case Kind::RenderTargetArrayIndex:
    return "SV_RenderTargetArrayIndex";
  default:
    llvm_unreachable("validated vertex/pixel signature semantic");
  }
}

static void buildSignature(mcdxbc::Signature &Sig,
                           ArrayRef<SemanticSignatureElement> Elements,
                           bool Input, bool Native16, bool Legacy) {
  for (const auto &E : Elements) {
    uint8_t Mask = E.getDeclaredMask();
    uint8_t ExclusiveMask =
        Input ? E.getAlwaysReadsMask() : E.getNeverWritesMask();
    if (Legacy)
      ExclusiveMask = Input ? 0 : static_cast<uint8_t>(~Mask);
    for (auto [Row, Index] : enumerate(E.SemanticIndices))
      Sig.addParam(E.GSStream, getContainerSemanticName(E), Index,
                   getSystemValue(E.SemanticKind),
                   getComponentType(E.CompType, Legacy),
                   E.isAllocated() ? E.StartRow + Row : UnallocatedRow, Mask,
                   ExclusiveMask, E.getMinPrecision(!Native16));
  }
}

static void buildPSVElements(SmallVectorImpl<mcdxbc::PSVSignatureElement> &Dest,
                             ArrayRef<SemanticSignatureElement> Elements,
                             bool Legacy) {
  for (const auto &E : Elements) {
    auto Type = getComponentType(E.CompType, Legacy);
    // SigComponentType and PSV ComponentType share COMPONENT_TYPE definitions.
    Dest.push_back(
        {E.SemanticKind == dxbc::PSV::SemanticKind::Arbitrary ? E.SemanticName
                                                              : StringRef(),
         E.SemanticIndices,
         E.isAllocated() ? static_cast<uint8_t>(E.StartRow) : uint8_t(0),
         E.Cols, E.getStartColOrZero(), E.isAllocated(), E.SemanticKind,
         static_cast<dxbc::PSV::ComponentType>(Type), E.InterpMode,
         E.DynIndexMask, static_cast<uint8_t>(E.GSStream)});
  }
}

MDTuple *EntrySignature::getAsMetadata(LLVMContext &Ctx,
                                       VersionTuple ValidatorVersion) const {
  if (Inputs.empty() && Outputs.empty())
    return nullptr;
  return MDNode::get(Ctx,
                     {emitSignature(Ctx, Inputs, ValidatorVersion),
                      emitSignature(Ctx, Outputs, ValidatorVersion), nullptr});
}

void EntrySignature::buildSignatures(mcdxbc::Signature &Input,
                                     mcdxbc::Signature &Output,
                                     VersionTuple ValidatorVersion) const {
  bool Legacy = !ValidatorVersion.empty() &&
                ValidatorVersion != VersionTuple(0, 0) &&
                ValidatorVersion < VersionTuple(1, 5);
  buildSignature(Input, Inputs, true, UseNative16Bit, Legacy);
  buildSignature(Output, Outputs, false, UseNative16Bit, Legacy);
}

SmallVector<uint32_t> EntrySignature::getViewIDOutputMask() const {
  // Each output vector needs four component bits, so one 32-bit word holds
  // eight vectors. Conservatively mark every written output component as
  // dependent on the view ID, matching the conservative input/output map.
  SmallVector<uint32_t> Mask((OutputVectors + 7) / 8, 0);
  for (const auto &Output : Outputs) {
    if (!Output.isAllocated())
      continue;
    for (unsigned R = 0; R != Output.Rows; ++R)
      for (unsigned C = 0; C != 4; ++C) {
        if (!(Output.UsageMask & (1U << C)))
          continue;
        unsigned Bit = (Output.StartRow + R) * 4 + C;
        Mask[Bit / 32] |= 1U << (Bit % 32);
      }
  }
  return Mask;
}

void EntrySignature::updatePSV(mcdxbc::PSVRuntimeInfo &PSV,
                               VersionTuple ValidatorVersion,
                               bool UsesViewID) const {
  bool Legacy = !ValidatorVersion.empty() &&
                ValidatorVersion != VersionTuple(0, 0) &&
                ValidatorVersion < VersionTuple(1, 5);
  buildPSVElements(PSV.InputElements, Inputs, Legacy);
  buildPSVElements(PSV.OutputElements, Outputs, Legacy);
  PSV.BaseData.SigInputVectors = InputVectors;
  PSV.BaseData.SigOutputVectors[0] = OutputVectors;
  PSV.InputOutputMap[0] = InputOutputMap;
  if (UsesViewID)
    PSV.OutputVectorMasks[0] = getViewIDOutputMask();
  if (Stage == Triple::Vertex)
    PSV.BaseData.StageInfo.VS.OutputPositionPresent =
        llvm::any_of(Outputs, [](const auto &E) {
          return E.SemanticKind == dxbc::PSV::SemanticKind::Position;
        });
  if (Stage == Triple::Pixel)
    PSV.BaseData.StageInfo.PS.SampleFrequency =
        llvm::any_of(Inputs, [](const auto &E) {
          return E.InterpMode == dxbc::PSV::InterpolationMode::LinearSample ||
                 E.InterpMode ==
                     dxbc::PSV::InterpolationMode::LinearNoperspectiveSample ||
                 E.SemanticKind == dxbc::PSV::SemanticKind::SampleIndex;
        });
}

SmallVector<uint32_t>
EntrySignature::getDependencyState(bool UsesViewID) const {
  SmallVector<uint32_t> State = {InputVectors * 4, OutputVectors * 4};
  if (UsesViewID)
    llvm::append_range(State, getViewIDOutputMask());
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
