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
enum class Property {
#define GEN_OMP_PROPERTY_ENUMS
#include "llvm/Frontend/OpenMP/OMPDescriptors.h.inc"
#undef GEN_OMP_PROPERTY_ENUMS
};

static constexpr size_t Property_enumSize =
    llvm::to_underlying(Property::Last_) -
    llvm::to_underlying(Property::First_) + 1;

enum class Modifier {
#define GEN_OMP_MODIFIER_ENUMS
#include "llvm/Frontend/OpenMP/OMPDescriptors.h.inc"
#undef GEN_OMP_MODIFIER_ENUMS
};

static constexpr size_t Modifier_enumSize =
    llvm::to_underlying(Modifier::Last_) -
    llvm::to_underlying(Modifier::First_) + 1;

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

static constexpr size_t ModifierSet_enumSize =
    llvm::to_underlying(ModifierSet::Last_) -
    llvm::to_underlying(ModifierSet::First_) + 1;

constexpr inline bool isModifierGroup(ModifierSet S) {
  return //
      llvm::to_underlying(ModifierSet::FirstGroup_) <= llvm::to_underlying(S) &&
      llvm::to_underlying(S) <= llvm::to_underlying(ModifierSet::LastGroup_);
}

using Properties = EnumSet<Property, Property_enumSize>;
using Modifiers = EnumSet<Modifier, Modifier_enumSize>;
using ModifierSets = EnumSet<ModifierSet, ModifierSet_enumSize>;

using Clauses = llvm::omp::ClauseSet;
using Directives = llvm::omp::DirectiveSet;

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
    for (Version V : llvm::omp::getOpenMPVersions()) {
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

LLVM_ABI const descriptor::Clause &getDescriptor(llvm::omp::Clause C);
LLVM_ABI const descriptor::Modifier &getDescriptor(llvm::omp::Modifier M);
LLVM_ABI const descriptor::ModifierSet &getDescriptor(llvm::omp::ModifierSet S);

LLVM_ABI Properties getProperties(Clause C, Version V);
} // namespace llvm::omp

#endif // LLVM_FRONTEND_OPENMP_OMPDESCRIPTORS_H
