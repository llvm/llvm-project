//===-- OMPDescriptors.cpp - OpenMP descriptors ------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains descriptors of OpenMP elements.
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPDescriptors.h"

namespace llvm::omp {
const DescriptorMap<Clause, descriptor::Clause> &getClauseMap() {
  static const DescriptorMap<Clause, descriptor::Clause> Map{
#define GEN_OMP_CLAUSE_DESCRIPTORS
#include "OMPDescriptors.inc"
#undef GEN_OMP_CLAUSE_DESCRIPTORS
  };
  return Map;
}

const DescriptorMap<Modifier, descriptor::Modifier> &getModifierMap() {
  static const DescriptorMap<Modifier, descriptor::Modifier> Map{
#define GEN_OMP_MODIFIER_DESCRIPTORS
#include "OMPDescriptors.inc"
#undef GEN_OMP_MODIFIER_DESCRIPTORS
  };
  return Map;
}

#define GET_THING_OR_EMPTY(Thing, Member)                                      \
  template <typename DetailsTy>                                                \
  static Thing get##Thing##OrEmpty(const DetailsTy &D, Version V) {            \
    V = std::max(V, Version(45));                                              \
    if (auto Found = D.find(V); Found != D.end())                              \
      return Found->second.Member;                                             \
    return Thing{};                                                            \
  }

GET_THING_OR_EMPTY(Clauses, Cls)
GET_THING_OR_EMPTY(Directives, Dirs)
GET_THING_OR_EMPTY(Modifiers, Mods)
GET_THING_OR_EMPTY(Properties, Props)

#undef GET_THING_OR_EMPTY

Properties descriptor::Clause::getProperties(Version V) const {
  return getPropertiesOrEmpty(Details, V);
}
Directives descriptor::Clause::getDirectives(Version V) const {
  return getDirectivesOrEmpty(Details, V);
}
Modifiers descriptor::Clause::getModifiers(Version V) const {
  return getModifiersOrEmpty(Details, V);
}
Properties descriptor::Modifier::getProperties(Version V) const {
  return getPropertiesOrEmpty(Details, V);
}
Clauses descriptor::Modifier::getClauses(Version V) const {
  return getClausesOrEmpty(Details, V);
}

const descriptor::Clause &getDescriptor(llvm::omp::Clause C) {
  return getClauseMap().at(C);
}

const descriptor::Modifier &getDescriptor(llvm::omp::Modifier M) {
  return getModifierMap().at(M);
}

Properties getProperties(Clause C, Version V) {
  return getDescriptor(C).getProperties(std::max(V, Version(45)));
}
} // namespace llvm::omp
