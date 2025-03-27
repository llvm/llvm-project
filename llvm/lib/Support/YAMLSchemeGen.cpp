//===- lib/Support/YAMLSchemeGen.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/YAMLSchemeGen.h"

using namespace llvm;
using namespace yaml;

//===----------------------------------------------------------------------===//
//  SchemeGen
//===----------------------------------------------------------------------===//

SchemeGen::SchemeGen(raw_ostream &RO, void *Ctxt, int WrapColumn)
    : O(RO, Ctxt, WrapColumn) {}

IOKind SchemeGen::getKind() const { return IOKind::SchemeGenering; }

bool SchemeGen::outputting() const { return false; }

bool SchemeGen::mapTag(StringRef, bool) { return false; }

void SchemeGen::beginMapping() {
  auto *Top = getTopScheme();
  assert(Top);
  auto *Type = getOrCreateProperty<TypeProperty>(*Top);
  Type->setValue("object");
}

void SchemeGen::endMapping() {}

bool SchemeGen::preflightKey(const char *Key, bool Required, bool SameAsDefault,
                             bool &UseDefault, void *&SaveInfo) {
  auto *Top = getTopScheme();
  assert(Top);
  if (Required) {
    auto *Req = getOrCreateProperty<RequiredProperty>(*Top);
    Req->emplace_back(Key);
  }
  auto *S = createScheme();
  auto *Properties = getOrCreateProperty<PropertiesProperty>(*Top);
  Properties->emplace_back(Key, S);
  Schemes.push_back(S);
  return true;
}

void SchemeGen::postflightKey(void *) {
  assert(!Schemes.empty());
  Schemes.pop_back();
}

std::vector<StringRef> SchemeGen::keys() { return {}; }

void SchemeGen::beginFlowMapping() { beginMapping(); }

void SchemeGen::endFlowMapping() { endMapping(); }

unsigned SchemeGen::beginSequence() {
  auto *Top = getTopScheme();
  assert(Top);
  auto *Type = getOrCreateProperty<TypeProperty>(*Top);
  Type->setValue("array");
  getOrCreateProperty<ItemsProperty>(*Top);
  return 1;
}

void SchemeGen::endSequence() {}

bool SchemeGen::preflightElement(unsigned, void *&) {
  auto *Top = getTopScheme();
  assert(Top);
  auto *S = createScheme();
  auto *Items = getOrCreateProperty<ItemsProperty>(*Top);
  Items->emplace_back(S);
  Schemes.push_back(S);
  return true;
}

void SchemeGen::postflightElement(void *) {
  assert(!Schemes.empty());
  Schemes.pop_back();
}

unsigned SchemeGen::beginFlowSequence() { return beginSequence(); }

bool SchemeGen::preflightFlowElement(unsigned Arg1, void *&Arg2) {
  return preflightElement(Arg1, Arg2);
}

void SchemeGen::postflightFlowElement(void *Arg1) { postflightElement(Arg1); }

void SchemeGen::endFlowSequence() { endSequence(); }

void SchemeGen::beginEnumScalar() {
  auto *Top = getTopScheme();
  assert(Top);
  auto *Type = getOrCreateProperty<TypeProperty>(*Top);
  Type->setValue("string");
  getOrCreateProperty<EnumProperty>(*Top);
}

bool SchemeGen::matchEnumScalar(const char *Val, bool) {
  auto *Top = getTopScheme();
  assert(Top);
  auto *Enum = getOrCreateProperty<EnumProperty>(*Top);
  Enum->emplace_back(Val);
  return false;
}

bool SchemeGen::matchEnumFallback() { return false; }

void SchemeGen::endEnumScalar() {}

bool SchemeGen::beginBitSetScalar(bool &) {
  beginEnumScalar();
  return true;
}

bool SchemeGen::bitSetMatch(const char *Val, bool Arg) {
  return matchEnumScalar(Val, Arg);
}

void SchemeGen::endBitSetScalar() { endEnumScalar(); }

void SchemeGen::scalarString(StringRef &Val, QuotingType) {
  auto *Top = getTopScheme();
  assert(Top);
  auto *Type = getOrCreateProperty<TypeProperty>(*Top);
  Type->setValue("string");
}

void SchemeGen::blockScalarString(StringRef &Val) {}

void SchemeGen::scalarTag(std::string &val) {}

NodeKind SchemeGen::getNodeKind() { report_fatal_error("invalid call"); }

void SchemeGen::setError(const Twine &) {}

std::error_code SchemeGen::error() { return {}; }

bool SchemeGen::canElideEmptySequence() { return false; }

// These are only used by operator<<. They could be private
// if that templated operator could be made a friend.
void SchemeGen::beginDocuments() {}

bool SchemeGen::preflightDocument(unsigned) {
  auto *S = createScheme();
  Root = S;
  Schemes.push_back(S);
  return true;
}

void SchemeGen::postflightDocument() {
  assert(!Schemes.empty());
  Schemes.pop_back();
  O << *Root->yamlize(*this);
}

void SchemeGen::endDocuments() {}
