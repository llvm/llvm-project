//===- lib/Support/YAMLGenerateSchema.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/YAMLGenerateSchema.h"

using namespace llvm;
using namespace yaml;

//===----------------------------------------------------------------------===//
//  GenerateSchema
//===----------------------------------------------------------------------===//

GenerateSchema::GenerateSchema(raw_ostream &RO, void *Ctxt, int WrapColumn)
    : O(RO, Ctxt, WrapColumn) {}

IOKind GenerateSchema::getKind() const { return IOKind::GeneratingSchema; }

bool GenerateSchema::outputting() const { return false; }

bool GenerateSchema::mapTag(StringRef, bool) { return false; }

void GenerateSchema::beginMapping() {
  auto *Top = getTopSchema();
  assert(Top);
  auto *Type = getOrCreateProperty<TypeProperty>(*Top);
  Type->setValue("object");
}

void GenerateSchema::endMapping() {}

bool GenerateSchema::preflightKey(const char *Key, bool Required,
                                  bool SameAsDefault, bool &UseDefault,
                                  void *&SaveInfo) {
  auto *Top = getTopSchema();
  assert(Top);
  if (Required) {
    auto *Req = getOrCreateProperty<RequiredProperty>(*Top);
    Req->emplace_back(Key);
  }
  auto *S = createSchema();
  auto *Properties = getOrCreateProperty<PropertiesProperty>(*Top);
  Properties->emplace_back(Key, S);
  Schemas.push_back(S);
  return true;
}

void GenerateSchema::postflightKey(void *) {
  assert(!Schemas.empty());
  Schemas.pop_back();
}

std::vector<StringRef> GenerateSchema::keys() { return {}; }

void GenerateSchema::beginFlowMapping() { beginMapping(); }

void GenerateSchema::endFlowMapping() { endMapping(); }

unsigned GenerateSchema::beginSequence() {
  auto *Top = getTopSchema();
  assert(Top);
  auto *Type = getOrCreateProperty<TypeProperty>(*Top);
  Type->setValue("array");
  getOrCreateProperty<ItemsProperty>(*Top);
  return 1;
}

void GenerateSchema::endSequence() {}

bool GenerateSchema::preflightElement(unsigned, void *&) {
  auto *Top = getTopSchema();
  assert(Top);
  auto *S = createSchema();
  auto *Items = getOrCreateProperty<ItemsProperty>(*Top);
  Items->emplace_back(S);
  Schemas.push_back(S);
  return true;
}

void GenerateSchema::postflightElement(void *) {
  assert(!Schemas.empty());
  Schemas.pop_back();
}

unsigned GenerateSchema::beginFlowSequence() { return beginSequence(); }

bool GenerateSchema::preflightFlowElement(unsigned Arg1, void *&Arg2) {
  return preflightElement(Arg1, Arg2);
}

void GenerateSchema::postflightFlowElement(void *Arg1) {
  postflightElement(Arg1);
}

void GenerateSchema::endFlowSequence() { endSequence(); }

void GenerateSchema::beginEnumScalar() {
  auto *Top = getTopSchema();
  assert(Top);
  auto *Type = getOrCreateProperty<TypeProperty>(*Top);
  Type->setValue("string");
  getOrCreateProperty<EnumProperty>(*Top);
}

bool GenerateSchema::matchEnumScalar(const char *Val, bool) {
  auto *Top = getTopSchema();
  assert(Top);
  auto *Enum = getOrCreateProperty<EnumProperty>(*Top);
  Enum->emplace_back(Val);
  return false;
}

bool GenerateSchema::matchEnumFallback() { return false; }

void GenerateSchema::endEnumScalar() {}

bool GenerateSchema::beginBitSetScalar(bool &) {
  beginEnumScalar();
  return true;
}

bool GenerateSchema::bitSetMatch(const char *Val, bool Arg) {
  return matchEnumScalar(Val, Arg);
}

void GenerateSchema::endBitSetScalar() { endEnumScalar(); }

void GenerateSchema::scalarString(StringRef &Val, QuotingType) {
  auto *Top = getTopSchema();
  assert(Top);
  auto *Type = getOrCreateProperty<TypeProperty>(*Top);
  Type->setValue("string");
}

void GenerateSchema::blockScalarString(StringRef &Val) {}

void GenerateSchema::scalarTag(std::string &val) {}

NodeKind GenerateSchema::getNodeKind() { report_fatal_error("invalid call"); }

void GenerateSchema::setError(const Twine &) {}

std::error_code GenerateSchema::error() { return {}; }

bool GenerateSchema::canElideEmptySequence() { return false; }

// These are only used by operator<<. They could be private
// if that templated operator could be made a friend.
void GenerateSchema::beginDocuments() {}

bool GenerateSchema::preflightDocument(unsigned) {
  auto *S = createSchema();
  Root = S;
  Schemas.push_back(S);
  return true;
}

void GenerateSchema::postflightDocument() {
  assert(!Schemas.empty());
  Schemas.pop_back();
  O << *Root->yamlize(*this);
}

void GenerateSchema::endDocuments() {}
