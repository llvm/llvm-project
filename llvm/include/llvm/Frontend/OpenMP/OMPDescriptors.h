//===-- OMPDescriptors.h - OpenMP descriptors --------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions and declarations of OpenMP elements.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OPENMP_OMPDESCRIPTORS_H
#define LLVM_FRONTEND_OPENMP_OMPDESCRIPTORS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/OpenMP/OMP.h"

namespace llvm::omp {
// The enumerations will be auto-generated.

enum class Property {
  AllDataEnvironments,
  First_ = AllDataEnvironments,
  AllPrivatizing,
  Complex,
  Constant,
  DataCopying,
  DataEnvironmentAttribute,
  DataMappingAttribute,
  DataMotionAttribute,
  DataSharingAttribute,
  DefaultAttribute,
  DeviceAssociated,
  DeviceGlobalRequirement,
  DispatchCompliant,
  DispatchRequirement,
  EndClause,
  Exclusive,
  IcvDefaulted,
  IcvModifying,
  InnermostLeaf,
  MapTypeModifying,
  OnceForAllConstituents,
  Optional,
  OriginalListItemUpdating,
  OutermostLeaf,
  Positive,
  PostModified,
  Privatization,
  ReductionParticipating,
  ReductionScoping,
  RegionInvariant,
  Repeatable,
  Required,
  ScheduleSpecification,
  SpaceConfiguring,
  TargetConsistent,
  TaskInherited,
  TaskSynchronizing,
  TaskgraphAltering,
  Ultimate,
  Unique,
  Last_ = Unique,
};

static constexpr size_t Property_enumSize =
    llvm::to_underlying(Property::Last_) -
    llvm::to_underlying(Property::First_) + 1;

enum class Modifier {
  AccessGroup,
  First_ = AccessGroup,
  AdjustOp,
  AlignModifier,
  Alignment,
  AllocatorComplexModifier,
  AllocatorSimpleModifier,
  AlwaysModifier,
  AttachModifier,
  AutomapModifier,
  ChunkModifier,
  CloseModifier,
  ContextSelector,
  DefaultModifier,
  DeleteModifier,
  DependenceType,
  DepinfoModifier,
  DepobjModifier,
  DeviceModifier,
  DimsModifier,
  DirectiveNameModifier,
  Expectation,
  Fallback,
  FallbackModifier,
  InductionIdentifier,
  InductionModifier,
  InscanModifier,
  InteropType,
  Iterator,
  LastprivateModifier,
  LinearModifier,
  LinearStep,
  LocModifier,
  LoopModifier,
  LowerBound,
  MapType,
  MapTypeModifier,
  Mapper,
  MemSpace,
  Monotonic,
  NeedDeviceAddr,
  NeedDevicePtr,
  NoncommutativeModifier,
  Nonmonotonic,
  Nothing,
  OmpxHoldModifier,
  OptionalModifier,
  OrderModifier,
  OrderingModifier,
  OriginalSharingModifier,
  PerThreadModifier,
  PreferType,
  Prescriptiveness,
  PrescriptivenessModifier,
  PresentModifier,
  ReductionIdentifier,
  ReductionModifier,
  RefModifier,
  Saved,
  ScaledModifier,
  SelfModifier,
  Simd,
  SizesSelector,
  StepComplexModifier,
  StepModifier,
  StepSimpleModifier,
  TargetType,
  TargetsyncType,
  TaskDependenceType,
  TaskModifier,
  TraitsArray,
  VariableCategory,
  Last_ = VariableCategory,
};

static constexpr size_t Modifier_enumSize =
    llvm::to_underlying(Modifier::Last_) -
    llvm::to_underlying(Modifier::First_) + 1;

enum class ModifierSet {
  DependModifierSet1,
  First_ = DependModifierSet1,
  NumTeamsModifierSet1,
  ReductionModifierSet1,
  Last_ = ReductionModifierSet1,
};

static constexpr size_t ModifierSet_enumSize =
    llvm::to_underlying(ModifierSet::Last_) -
    llvm::to_underlying(ModifierSet::First_) + 1;

enum class ModifierGroup {
  AdjustOp,
  First_ = AdjustOp,
  DependListType,
  InteropType,
  MapTypeModifying,
  ReductionType,
  VariableSelector,
  Last_ = VariableSelector,
};

static constexpr size_t ModifierGroup_enumSize =
    llvm::to_underlying(ModifierGroup::Last_) -
    llvm::to_underlying(ModifierGroup::First_) + 1;

using Properties = EnumSet<Property, Property_enumSize>;
using Modifiers = EnumSet<Modifier, Modifier_enumSize>;
using ModifierSets = EnumSet<ModifierSet, ModifierSet_enumSize>;
using ModifierGroups = EnumSet<ModifierGroup, ModifierGroup_enumSize>;

using Clauses = llvm::omp::ClauseSet;
using Directives = llvm::omp::DirectiveSet;

namespace desc {
struct BaseDetails {
  Properties Props;
};

// struct DirectiveDetails : public BaseDetails {
//   Association Assoc;
//   Category Cat;
//   Clauses Cls;
//   SourceLanguage Langs;
//   StringRef Spelling;
// };

struct ClauseDetails : public BaseDetails {
  Directives Dirs;
  //  SourceLanguage Langs;
  Modifiers Mods;
  ModifierSets ModSets;
  ModifierGroups ModGroups;
  //  StringRef Spelling;
};

struct ModifierDetails : public BaseDetails {
  Clauses Cls;
};

struct ModifierSetDetails : public BaseDetails {
  Modifiers Mods;
};

struct ModifierGroupDetails : public ModifierSetDetails {
  Clauses Cls;
};

template <typename DetailsTy> using DetailsMap = DenseMap<unsigned, DetailsTy>;
} // namespace desc

template <typename DetailsTy> struct Descriptor {
  //  Descriptor(StringRef N, desc::DetailsMap<DetailsTy> &&D) : Name(N),
  //  Details(std::move(D)) {}
  StringRef getName() const { return Name; }
  SmallVector<unsigned> getVersions() const {
    SmallVector<unsigned> Vs;
    for (unsigned V : llvm::omp::getOpenMPVersions()) {
      if (auto F = Details.find(V); F != Details.end())
        Vs.push_back(V);
    }
    return Vs;
  }
  // private:
  StringRef Name;
  // protected:
  desc::DetailsMap<DetailsTy> Details;
};

template <typename Enum, typename DescriptorTy>
using DescriptorMap = DenseMap<Enum, DescriptorTy>;

struct ClauseDesc : public Descriptor<desc::ClauseDetails> {
  Properties getProperties(unsigned V) const;
  Directives getDirectives(unsigned V) const;
  Modifiers getModifiers(unsigned V) const;
  ModifierSets getModifierSets(unsigned V) const;
  ModifierGroups getModifierGroups(unsigned V) const;
};

struct ModifierDesc : public Descriptor<desc::ModifierDetails> {
  Properties getProperties(unsigned V) const;
  Clauses getClauses(unsigned V) const;
};

struct ModifierSetDesc : public Descriptor<desc::ModifierSetDetails> {
  Properties getProperties(unsigned V) const;
  Modifiers getModifiers(unsigned V) const;
};

struct ModifierGroupDesc : public Descriptor<desc::ModifierGroupDetails> {
  Properties getProperties(unsigned V) const;
  Modifiers getModifiers(unsigned V) const;
  Clauses getClauses(unsigned V) const;
};

const ClauseDesc &getDescriptor(llvm::omp::Clause C);
const ModifierDesc &getDescriptor(llvm::omp::Modifier M);
const ModifierSetDesc &getDescriptor(llvm::omp::ModifierSet S);
const ModifierGroupDesc &getDescriptor(llvm::omp::ModifierGroup G);

Properties getProperties(Clause C, unsigned Version);
} // namespace llvm::omp

#endif // LLVM_FRONTEND_OPENMP_OMPDESCRIPTORS_H
