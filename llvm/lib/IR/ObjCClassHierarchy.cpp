//===- ObjCClassHierarchy.cpp - ObjC class layout facts -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for collecting and resolving Objective-C class layout facts used by
// ivar offset constification.
//===----------------------------------------------------------------------===//

#include "llvm/IR/ObjCClassHierarchy.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

static constexpr uint32_t DarwinNonFragileNSObjectInstanceSize = 8;
static constexpr StringLiteral NonFragileNSObjectClassName =
    "OBJC_CLASS_$_NSObject";

// IR parsing helpers.

static std::optional<uint32_t> getStableRootSize(GlobalValue::GUID GUID) {
  if (GUID ==
      GlobalValue::getGUIDAssumingExternalLinkage(NonFragileNSObjectClassName))
    return DarwinNonFragileNSObjectInstanceSize;
  return std::nullopt;
}

static const GlobalVariable *stripToGlobal(const Constant *C) {
  return C ? dyn_cast<GlobalVariable>(C->stripPointerCasts()) : nullptr;
}

template <typename ConstantT = ConstantStruct>
static const ConstantT *getInitializerStruct(const GlobalVariable *GV) {
  return (GV && GV->hasInitializer())
             ? dyn_cast<ConstantT>(GV->getInitializer())
             : nullptr;
}

// Walk over every ObjC class registered in the module's __objc_classlist and
// __objc_nlclslist sections
template <typename CallbackT>
static void forEachClassListEntry(const Module &M, CallbackT &&Callback) {
  for (const auto &GV : M.globals()) {
    if (GV.isDeclaration())
      continue;

    StringRef Section = GV.getSection();
    if (!Section.contains("__objc_classlist") &&
        !Section.contains("__objc_nlclslist"))
      continue;

    const auto *ClassList = getInitializerStruct<ConstantArray>(&GV);
    if (!ClassList)
      continue;

    for (const auto &Class : ClassList->operands()) {
      const GlobalVariable *ClassGV =
          stripToGlobal(cast<Constant>(Class.get()));
      if (ClassGV)
        Callback(*ClassGV);
    }
  }
}

namespace {

// Intermediate raw class layout parsed from module IR, before pre-sliding.
struct ParsedClassInfo {
  ObjCClassInfo Layout = {0, 0, 0, 1};
  GlobalVariable *ROGV = nullptr;
};

} // end anonymous namespace

// Extract raw layout info from a _class_t global by walking the ObjC ABI v2 struct chain.
// Return nullopt if _class_t is malformed.
static std::optional<ParsedClassInfo>
parseClassInfo(const GlobalVariable *ClassGV) {
  //   _class_t: { isa, superclass, cache, vtable, ro_data }
  auto ClassStruct = getInitializerStruct(ClassGV);
  if (!ClassStruct || ClassStruct->getNumOperands() < 5)
    return std::nullopt;

  //   _class_ro_t: { flags, instanceStart, instanceSize, ..., ivarList, ... }
  GlobalVariable *ROGV =
      const_cast<GlobalVariable *>(stripToGlobal(ClassStruct->getOperand(4)));
  auto ROStruct = getInitializerStruct(ROGV);
  if (!ROStruct || ROStruct->getNumOperands() < 8)
    return std::nullopt;

  const auto *Start = dyn_cast<ConstantInt>(ROStruct->getOperand(1));
  const auto *Size = dyn_cast<ConstantInt>(ROStruct->getOperand(2));
  if (!Start || !Size)
    return std::nullopt;

  //   _ivar_t: { offset, name, type, alignment_raw, size }
  ParsedClassInfo Info;
  Info.ROGV = ROGV;
  if (const GlobalVariable *SuperclassGV =
          stripToGlobal(ClassStruct->getOperand(1)))
    Info.Layout.SuperclassGUID = SuperclassGV->getGUID();
  Info.Layout.InstanceStart = static_cast<uint32_t>(Start->getZExtValue());
  Info.Layout.InstanceSize = static_cast<uint32_t>(Size->getZExtValue());

  const GlobalVariable *IvarListGV = stripToGlobal(ROStruct->getOperand(7));
  if (!IvarListGV || !IvarListGV->hasInitializer())
    return Info;

  const auto *IvarList = getInitializerStruct(IvarListGV);

  //   _ivar_list_t: { entsize, count, ivars }
  if (!IvarList || IvarList->getNumOperands() < 3)
    return Info;

  const auto *Ivars = dyn_cast<ConstantArray>(IvarList->getOperand(2));
  if (!Ivars)
    return Info;

  for (const auto &Ivar : Ivars->operands()) {
    const auto *IvarStruct = dyn_cast<ConstantStruct>(Ivar.get());
    if (!IvarStruct || IvarStruct->getNumOperands() < 5)
      continue;

    const auto *AlignRaw = dyn_cast<ConstantInt>(IvarStruct->getOperand(3));
    if (!AlignRaw)
      continue;
    uint32_t Alignment = 1U << static_cast<uint32_t>(AlignRaw->getZExtValue());
    if (Info.Layout.MaxIvarAlignment < Alignment)
      Info.Layout.MaxIvarAlignment = Alignment;
  }

  return Info;
}

// Pre-slide the hierarchy graph materialized in this map; unreachable classes are erased.
void ObjCClassHierarchy::resolveHierarchy(
    DenseMap<GlobalValue::GUID, ObjCClassInfo> &Classes) {
  if (Classes.empty())
    return;

  // Hierarchy graph materialization.
  DenseMap<GlobalValue::GUID, SmallVector<GlobalValue::GUID, 4>> Children;
  for (const auto &[GUID, Info] : Classes)
    if (Info.SuperclassGUID != 0)
      Children[Info.SuperclassGUID].push_back(GUID);

  // Mimic runtime moveIvars(): slide InstanceStart/Size by the aligned delta.
  auto Slide = [](ObjCClassInfo &Info, uint32_t SuperSize) {
    auto AlignIvarSlide = [](uint32_t Diff, uint32_t MaxIvarAlignment) {
      if (Diff == 0 || MaxIvarAlignment <= 1)
        return Diff;
      uint32_t Mask = MaxIvarAlignment - 1;
      return (Diff + Mask) & ~Mask;
    };
    uint32_t Diff =
        AlignIvarSlide(SuperSize - Info.InstanceStart, Info.MaxIvarAlignment);
    Info.InstanceStart += Diff;
    Info.InstanceSize += Diff;
  };

  // Seed worklist with roots and direct subclasses of known stable roots.
  DenseMap<GlobalValue::GUID, uint32_t> ResolvedSize;
  SmallVector<GlobalValue::GUID> Worklist;
  for (auto &[GUID, Info] : Classes) {
    if (Info.SuperclassGUID == 0) {
      ResolvedSize[GUID] = Info.InstanceSize;
      Worklist.push_back(GUID);
      continue;
    }

    std::optional<uint32_t> SuperSize = getStableRootSize(Info.SuperclassGUID);
    if (!SuperSize || Classes.contains(Info.SuperclassGUID))
      continue;

    Slide(Info, *SuperSize);
    ResolvedSize[GUID] = Info.InstanceSize;
    Worklist.push_back(GUID);
  }

  // Top-down pre-sliding.
  while (!Worklist.empty()) {
    GlobalValue::GUID ParentGUID = Worklist.pop_back_val();
    uint32_t ParentSize = ResolvedSize[ParentGUID];
    for (auto ChildGUID : Children[ParentGUID]) {
      if (!Classes.contains(ChildGUID) || ResolvedSize.contains(ChildGUID))
        continue;
      auto Child = Classes.find(ChildGUID);
      Slide(Child->second, ParentSize);
      ResolvedSize[ChildGUID] = Child->second.InstanceSize;
      Worklist.push_back(ChildGUID);
    }
  }

  // Erase classes that were never reached -- their hierarchy is incomplete.
  SmallVector<GlobalValue::GUID, 8> UnresolvedClasses;
  for (const auto &[GUID, Info] : Classes)
    if (!ResolvedSize.contains(GUID))
      UnresolvedClasses.push_back(GUID);
  for (auto GUID : UnresolvedClasses)
    Classes.erase(GUID);
}

// Build the list of classes eligible for constification.
// ThinLTO: pair module IR handles with pre-resolved summary layout.
// FullLTO: parse all classes locally, resolve hierarchy, then collect results.
ObjCClassHierarchy::ObjCClassHierarchy(
    const Module &M, const ModuleSummaryIndex *ImportSummary) {
  // ThinLTO path: summary already contains resolved sizes from the thin link.
  if (ImportSummary) {
    const auto &Classes = ImportSummary->getObjCClasses();
    forEachClassListEntry(M, [&](const GlobalVariable &ClassGV) {
      auto SummaryIt = Classes.find(ClassGV.getGUID());
      if (SummaryIt == Classes.end())
        return;

      // Only need the ROGV handle; layout comes from the summary.
      if (std::optional<ParsedClassInfo> Info = parseClassInfo(&ClassGV)) {
        Resolved.push_back({Info->ROGV, SummaryIt->second.InstanceStart,
                            SummaryIt->second.InstanceSize});
      }
    });
    return;
  }

  // FullLTO path: collect all classes and their RO handles.
  DenseMap<GlobalValue::GUID, ObjCClassInfo> Classes;
  DenseMap<GlobalValue::GUID, GlobalVariable *> GUIDToROGV;

  forEachClassListEntry(M, [&](const GlobalVariable &ClassGV) {
    if (std::optional<ParsedClassInfo> Info = parseClassInfo(&ClassGV)) {
      Classes[ClassGV.getGUID()] = Info->Layout;
      GUIDToROGV[ClassGV.getGUID()] = Info->ROGV;
    }
  });

  // Resolve hierarchy locally.
  resolveHierarchy(Classes);

  // Collect surviving classes into the final resolved list.
  for (const auto &[GUID, ResolvedInfo] : Classes) {
    auto It = GUIDToROGV.find(GUID);
    if (It == GUIDToROGV.end())
      continue;
    Resolved.push_back(
        {It->second, ResolvedInfo.InstanceStart, ResolvedInfo.InstanceSize});
  }
}

// Export raw class layout facts into the summary index (producer side).
void ObjCClassHierarchy::exportToSummary(const Module &M,
                                         ModuleSummaryIndex &Index) {
  forEachClassListEntry(M, [&](const GlobalVariable &ClassGV) {
    std::optional<ParsedClassInfo> Info = parseClassInfo(&ClassGV);
    if (!Info)
      return;

    Index.getObjCClasses()[ClassGV.getGUID()] = Info->Layout;
  });
}
