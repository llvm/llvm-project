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

#include <map>

// For unassociated directives the .inc file uses an "Unassociated" enum
// (since it's the name used in the OpenMP spec), while the existing enum
// (in OMP.h.inc) uses "None". Redefine the token "Unassociated" to "None"
// in this file to allow the reuse of the pre-existing enum.
#define Unassociated None

namespace llvm::omp {
const DescriptorMap<Clause, descriptor::Clause> &getClauseMap() {
  static const DescriptorMap<Clause, descriptor::Clause> Map{
#define GEN_OMP_CLAUSE_DESCRIPTORS
#include "OMPDescriptors.inc"
#undef GEN_OMP_CLAUSE_DESCRIPTORS
  };
  return Map;
}

const DescriptorMap<ClauseSet, descriptor::ClauseSet> &getClauseSetMap() {
  static const DescriptorMap<ClauseSet, descriptor::ClauseSet> Map{
#define GEN_OMP_CLAUSE_GROUP_DESCRIPTORS
#include "OMPDescriptors.inc"
#undef GEN_OMP_CLAUSE_GROUP_DESCRIPTORS

#define GEN_OMP_CLAUSE_SET_DESCRIPTORS
#include "OMPDescriptors.inc"
#undef GEN_OMP_CLAUSE_SET_DESCRIPTORS
  };
  return Map;
}

const DescriptorMap<Directive, descriptor::Directive> &getDirectiveMap() {
  static const DescriptorMap<Directive, descriptor::Directive> Map{
#define GEN_OMP_DIRECTIVE_DESCRIPTORS
#include "OMPDescriptors.inc"
#undef GEN_OMP_DIRECTIVE_DESCRIPTORS
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

const DescriptorMap<ModifierSet, descriptor::ModifierSet> &getModifierSetMap() {
  static const DescriptorMap<ModifierSet, descriptor::ModifierSet> Map{
#define GEN_OMP_MODIFIER_GROUP_DESCRIPTORS
#include "OMPDescriptors.inc"
#undef GEN_OMP_MODIFIER_GROUP_DESCRIPTORS

#define GEN_OMP_MODIFIER_SET_DESCRIPTORS
#include "OMPDescriptors.inc"
#undef GEN_OMP_MODIFIER_SET_DESCRIPTORS
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
GET_THING_OR_EMPTY(ClauseSets, ClsSets)
GET_THING_OR_EMPTY(Directives, Dirs)
GET_THING_OR_EMPTY(Modifiers, Mods)
GET_THING_OR_EMPTY(ModifierSets, ModSets)
GET_THING_OR_EMPTY(Properties, Props)

#undef GET_THING_OR_EMPTY

// Clause
Properties descriptor::Clause::getProperties(Version V) const {
  return getPropertiesOrEmpty(Details, V);
}
Directives descriptor::Clause::getDirectives(Version V) const {
  return getDirectivesOrEmpty(Details, V);
}
Modifiers descriptor::Clause::getModifiers(Version V) const {
  return getModifiersOrEmpty(Details, V);
}
ModifierSets descriptor::Clause::getModifierSets(Version V) const {
  return getModifierSetsOrEmpty(Details, V);
}
// ClauseSet
Properties descriptor::ClauseSet::getProperties(Version V) const {
  return getPropertiesOrEmpty(Details, V);
}
Clauses descriptor::ClauseSet::getClauses(Version V) const {
  return getClausesOrEmpty(Details, V);
}
Directives descriptor::ClauseSet::getDirectives(Version V) const {
  return getDirectivesOrEmpty(Details, V);
}
// Directive
Properties descriptor::Directive::getProperties(Version V) const {
  return getPropertiesOrEmpty(Details, V);
}
Association descriptor::Directive::getAssociation(Version V) const {
  return Details.at(std::max(V, Version(45))).Assoc;
}
Category descriptor::Directive::getCategory(Version V) const {
  return Details.at(std::max(V, Version(45))).Cat;
}
Clauses descriptor::Directive::getClauses(Version V) const {
  return getClausesOrEmpty(Details, V);
}
ClauseSets descriptor::Directive::getClauseSets(Version V) const {
  return getClauseSetsOrEmpty(Details, V);
}

// Modifier
Properties descriptor::Modifier::getProperties(Version V) const {
  return getPropertiesOrEmpty(Details, V);
}
Clauses descriptor::Modifier::getClauses(Version V) const {
  return getClausesOrEmpty(Details, V);
}
// ModifierSet
Properties descriptor::ModifierSet::getProperties(Version V) const {
  return getPropertiesOrEmpty(Details, V);
}
Modifiers descriptor::ModifierSet::getModifiers(Version V) const {
  return getModifiersOrEmpty(Details, V);
}
Clauses descriptor::ModifierSet::getClauses(Version V) const {
  return getClausesOrEmpty(Details, V);
}

const descriptor::Clause &getDescriptor(Clause C) {
  return getClauseMap().at(C);
}

const descriptor::ClauseSet &getDescriptor(ClauseSet S) {
  return getClauseSetMap().at(S);
}

const descriptor::Directive &getDescriptor(Directive D) {
  ArrayRef<llvm::omp::Directive> Leafs = llvm::omp::getLeafConstructsOrSelf(D);
  if (Leafs.size() == 1)
    return getDirectiveMap().at(Leafs[0]);

  static std::map<Directive, descriptor::Directive> Compound;
  if (auto F = Compound.find(D); F != Compound.end())
    return F->second;

  // Combine details from all leafs into a single map and create a descriptor
  // from it.
  using Details = descriptor::details::Directive;
  using DetailsMap = descriptor::DetailsMap<Details>;
  DetailsMap Det;
  // The name of the descriptor will be the spelling from the latest supported
  // version.
  llvm::omp::Version MaxV(0);

  for (llvm::omp::Directive L : Leafs) {
    const auto &Desc{getDescriptor(L)};
    for (auto &[V, T] : Desc.getDetails()) {
      MaxV = std::max(MaxV, V); // Used for descriptor's name.
      auto F = Det.insert({V, Details{}}).first;
      F->second.Props |= T.Props;
      F->second.Spelling = getOpenMPDirectiveName(D, V);
      // The category should be the same for all of them (i.e. executable),
      // the association of the last one is the association of the whole
      // directive. Since the leafs are ordered from the outermost to the
      // innermost it is safe to simply overwrite the non-set members each
      // time.
      F->second.Assoc = T.Assoc;
      F->second.Cat = T.Cat;
      F->second.Cls |= T.Cls;
      F->second.ClsSets |= T.ClsSets;
    }
  }

  auto At =
      Compound.try_emplace(D, getOpenMPDirectiveName(D, MaxV), std::move(Det));
  return At.first->second;
}

const descriptor::Modifier &getDescriptor(Modifier M) {
  return getModifierMap().at(M);
}

const descriptor::ModifierSet &getDescriptor(ModifierSet S) {
  return getModifierSetMap().at(S);
}

Properties getProperties(Clause C, Version V) {
  return getDescriptor(C).getProperties(std::max(V, Version(45)));
}
} // namespace llvm::omp

#undef Unassociated
