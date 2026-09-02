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

using Properties = EnumSet<Property, Property_enumSize>;
using Modifiers = EnumSet<Modifier, Modifier_enumSize>;

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
};

struct Modifier : public Base {
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
};

struct Modifier : public Descriptor<details::Modifier> {
  using Base = Descriptor<details::Modifier>;
  using Base::Base;
  LLVM_ABI Properties getProperties(Version V) const;
  LLVM_ABI Clauses getClauses(Version V) const;
};
} // namespace descriptor

template <typename Enum, typename DescriptorTy>
using DescriptorMap = DenseMap<Enum, DescriptorTy>;

LLVM_ABI const descriptor::Clause &getDescriptor(llvm::omp::Clause C);
LLVM_ABI const descriptor::Modifier &getDescriptor(llvm::omp::Modifier M);

LLVM_ABI Properties getProperties(Clause C, Version V);
} // namespace llvm::omp

#endif // LLVM_FRONTEND_OPENMP_OMPDESCRIPTORS_H
