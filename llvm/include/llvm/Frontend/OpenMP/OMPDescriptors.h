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

#define Unassociated None

namespace llvm::omp {
enum class Property {
#define GEN_OMP_PROPERTY_ENUMS
#include "llvm/Frontend/OpenMP/OMPDescriptors.h.inc"
#undef GEN_OMP_PROPERTY_ENUMS
};

enum class ClauseSet {
#define GEN_OMP_CLAUSE_GROUP_ENUMS
#define First_ FirstGroup_
#define Last_ LastGroup_
#include "llvm/Frontend/OpenMP/OMPDescriptors.h.inc"
#undef Last_
#undef First_
#undef GEN_OMP_CLAUSE_GROUP_ENUMS

#define GEN_OMP_CLAUSE_SET_ENUMS
#define First_ FirstSet_
#define Last_ LastSet_
#include "llvm/Frontend/OpenMP/OMPDescriptors.h.inc"
#undef Last_
#undef First_
#undef GEN_OMP_CLAUSE_SET_ENUMS
  First_ = FirstGroup_,
  Last_ = LastSet_,
};

constexpr inline bool isClauseGroup(ClauseSet S) {
  return //
      llvm::to_underlying(ClauseSet::FirstGroup_) <= llvm::to_underlying(S) &&
      llvm::to_underlying(S) <= llvm::to_underlying(ClauseSet::LastGroup_);
}

enum class Modifier {
#define GEN_OMP_MODIFIER_ENUMS
#include "llvm/Frontend/OpenMP/OMPDescriptors.h.inc"
#undef GEN_OMP_MODIFIER_ENUMS
};

enum class ModifierSet {
#define GEN_OMP_MODIFIER_GROUP_ENUMS
#define First_ FirstGroup_
#define Last_ LastGroup_
#include "llvm/Frontend/OpenMP/OMPDescriptors.h.inc"
#undef Last_
#undef First_
#undef GEN_OMP_MODIFIER_GROUP_ENUMS

#define GEN_OMP_MODIFIER_SET_ENUMS
#define First_ FirstSet_
#define Last_ LastSet_
#include "llvm/Frontend/OpenMP/OMPDescriptors.h.inc"
#undef Last_
#undef First_
#undef GEN_OMP_MODIFIER_SET_ENUMS
  First_ = FirstGroup_,
  Last_ = LastSet_,
};

constexpr inline bool isModifierGroup(ModifierSet S) {
  return //
      llvm::to_underlying(ModifierSet::FirstGroup_) <= llvm::to_underlying(S) &&
      llvm::to_underlying(S) <= llvm::to_underlying(ModifierSet::LastGroup_);
}

using Properties = EnumSet<Property>;
using ClauseSets = EnumSet<ClauseSet>;
using Modifiers = EnumSet<Modifier>;
using ModifierSets = EnumSet<ModifierSet>;

namespace descriptor {
namespace details {
struct Base {
  Properties Props;
};

struct Clause : public Base {
  StringRef Spelling;
  Directives Dirs;
  SourceLanguage Langs;
  Modifiers Mods;
  ModifierSets ModSets;
};

struct ClauseSet : public Base {
  Clauses Cls;
  Directives Dirs;
};

struct Directive : public Base {
  StringRef Spelling;
  Association Assoc;
  Category Cat;
  Clauses Cls;
  ClauseSets ClsSets;
};

struct Modifier : public Base {
  Clauses Cls;
};

struct ModifierSet : public Base {
  Modifiers Mods;
  Clauses Cls;
};
} // namespace details

template <typename DetailsTy> using DetailsMap = DenseMap<Version, DetailsTy>;

template <typename DetailsTy> struct Descriptor {
  Descriptor(const Descriptor &) = default;
  Descriptor(Descriptor &&) = default;
  Descriptor(StringRef N, DetailsMap<DetailsTy> &&D)
      : Name(N), Details(std::move(D)) {}

  StringRef getName() const { return Name; }
  const DetailsMap<DetailsTy> &getDetails() const { return Details; }

  SmallVector<Version> getVersions() const {
    SmallVector<Version> Vs;
    for (Version V : getOpenMPVersions()) {
      if (auto F = Details.find(V); F != Details.end())
        Vs.push_back(V);
    }
    return Vs;
  }

private:
  StringRef Name;

protected:
  DetailsMap<DetailsTy> Details;
};

struct Clause : public Descriptor<details::Clause> {
  using Base = Descriptor<details::Clause>;
  using Base::Base;
  LLVM_ABI Properties getProperties(Version V) const;
  LLVM_ABI Directives getDirectives(Version V) const;
  LLVM_ABI Modifiers getModifiers(Version V) const;
  LLVM_ABI ModifierSets getModifierSets(Version V) const;
};

struct ClauseSet : public Descriptor<details::ClauseSet> {
  using Base = Descriptor<details::ClauseSet>;
  using Base::Base;
  LLVM_ABI Properties getProperties(Version V) const;
  LLVM_ABI Clauses getClauses(Version V) const;
  LLVM_ABI Directives getDirectives(Version V) const;
};

struct Directive : public Descriptor<details::Directive> {
  using Base = Descriptor<details::Directive>;
  using Base::Base;
  LLVM_ABI Properties getProperties(Version V) const;
  LLVM_ABI Association getAssociation(Version V) const;
  LLVM_ABI Category getCategory(Version V) const;
  LLVM_ABI Clauses getClauses(Version V) const;
  LLVM_ABI ClauseSets getClauseSets(Version V) const;
};

struct Modifier : public Descriptor<details::Modifier> {
  using Base = Descriptor<details::Modifier>;
  using Base::Base;
  LLVM_ABI Properties getProperties(Version V) const;
  LLVM_ABI Clauses getClauses(Version V) const;
};

struct ModifierSet : public Descriptor<details::ModifierSet> {
  using Base = Descriptor<details::ModifierSet>;
  using Base::Base;
  LLVM_ABI Properties getProperties(Version V) const;
  LLVM_ABI Modifiers getModifiers(Version V) const;
  LLVM_ABI Clauses getClauses(Version V) const;
};
} // namespace descriptor

template <typename Enum, typename DescriptorTy>
using DescriptorMap = DenseMap<Enum, DescriptorTy>;

LLVM_ABI const descriptor::Clause &getDescriptor(Clause C);
LLVM_ABI const descriptor::ClauseSet &getDescriptor(ClauseSet S);
LLVM_ABI const descriptor::Directive &getDescriptor(Directive D);
LLVM_ABI const descriptor::Modifier &getDescriptor(Modifier M);
LLVM_ABI const descriptor::ModifierSet &getDescriptor(ModifierSet S);

LLVM_ABI Properties getProperties(Clause C, Version V);
} // namespace llvm::omp

#undef Unassociated
#endif // LLVM_FRONTEND_OPENMP_OMPDESCRIPTORS_H
