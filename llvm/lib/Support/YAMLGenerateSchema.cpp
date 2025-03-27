//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/YAMLGenerateSchema.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace yaml;

//===----------------------------------------------------------------------===//
//  GenerateSchema
//===----------------------------------------------------------------------===//

GenerateSchema::GenerateSchema(raw_ostream &RO) : RO(RO) {}

IOKind GenerateSchema::getKind() const { return IOKind::GeneratingSchema; }

bool GenerateSchema::outputting() const { return false; }

bool GenerateSchema::mapTag(StringRef, bool) { return false; }

void GenerateSchema::beginMapping() {
  Schema *Top = getTopSchema();
  assert(Top);
  TypeProperty *Type = createProperty<TypeProperty>("object");
  Top->emplace_back(Type);
  FlowStyleProperty *FlowStyle = createProperty<FlowStyleProperty>();
  Top->emplace_back(FlowStyle);
}

void GenerateSchema::endMapping() {}

bool GenerateSchema::preflightKey(StringRef Key, bool Required,
                                  bool SameAsDefault, bool &UseDefault,
                                  void *&SaveInfo) {
  Schema *Top = getTopSchema();
  assert(Top);
  if (Key == "additionalProperties") {
    AdditionalPropertiesProperty *Additional =
        getOrCreateProperty<AdditionalPropertiesProperty>(*Top);
    Schema *S = createSchema();
    Additional->setSchema(S);
    Schemas.push_back(S);
    return true;
  }

  if (Required) {
    RequiredProperty *Req = getOrCreateProperty<RequiredProperty>(*Top);
    Req->emplace_back(Key);
  } else {
    OptionalProperty *Opt = getOrCreateProperty<OptionalProperty>(*Top);
    Opt->emplace_back(Key);
  }
  PropertiesProperty *Properties =
      getOrCreateProperty<PropertiesProperty>(*Top);
  Schema *S = createSchema();
  UserDefinedProperty *UserDefined =
      createProperty<UserDefinedProperty>(Key, S);
  Properties->emplace_back(UserDefined);
  Schemas.push_back(S);
  return true;
}

void GenerateSchema::postflightKey(void *) {
  assert(!Schemas.empty());
  Schemas.pop_back();
}

std::vector<StringRef> GenerateSchema::keys() { return {}; }

void GenerateSchema::beginFlowMapping() {
  Schema *Top = getTopSchema();
  assert(Top);
  TypeProperty *Type = createProperty<TypeProperty>("object");
  Top->emplace_back(Type);
  FlowStyleProperty *FlowStyle =
      createProperty<FlowStyleProperty>(FlowStyle::Flow);
  Top->emplace_back(FlowStyle);
}

void GenerateSchema::endFlowMapping() {}

unsigned GenerateSchema::beginSequence() {
  Schema *Top = getTopSchema();
  assert(Top);
  TypeProperty *Type = createProperty<TypeProperty>("array");
  Top->emplace_back(Type);
  ItemsProperty *Items = createProperty<ItemsProperty>();
  Top->emplace_back(Items);
  FlowStyleProperty *FlowStyle = createProperty<FlowStyleProperty>();
  Top->emplace_back(FlowStyle);
  return 1;
}

void GenerateSchema::endSequence() {}

bool GenerateSchema::preflightElement(unsigned, void *&) {
  Schema *Top = getTopSchema();
  assert(Top);
  Schema *S = createSchema();
  ItemsProperty *Items = getOrCreateProperty<ItemsProperty>(*Top);
  Items->setSchema(S);
  Schemas.push_back(S);
  return true;
}

void GenerateSchema::postflightElement(void *) {
  assert(!Schemas.empty());
  Schemas.pop_back();
}

unsigned GenerateSchema::beginFlowSequence() {
  Schema *Top = getTopSchema();
  assert(Top);
  TypeProperty *Type = createProperty<TypeProperty>("array");
  Top->emplace_back(Type);
  ItemsProperty *Items = createProperty<ItemsProperty>();
  Top->emplace_back(Items);
  FlowStyleProperty *FlowStyle =
      createProperty<FlowStyleProperty>(FlowStyle::Flow);
  Top->emplace_back(FlowStyle);
  return 1;
}

bool GenerateSchema::preflightFlowElement(unsigned Arg1, void *&Arg2) {
  return preflightElement(Arg1, Arg2);
}

void GenerateSchema::postflightFlowElement(void *Arg1) {
  postflightElement(Arg1);
}

void GenerateSchema::endFlowSequence() { endSequence(); }

void GenerateSchema::beginEnumScalar() {
  Schema *Top = getTopSchema();
  assert(Top);
  TypeProperty *Type = createProperty<TypeProperty>("string");
  Top->emplace_back(Type);
  EnumProperty *Enum = createProperty<EnumProperty>();
  Top->emplace_back(Enum);
}

bool GenerateSchema::matchEnumScalar(StringRef Val, bool) {
  Schema *Top = getTopSchema();
  assert(Top);
  EnumProperty *Enum = getOrCreateProperty<EnumProperty>(*Top);
  Enum->emplace_back(Val);
  return false;
}

bool GenerateSchema::matchEnumFallback() { return false; }

void GenerateSchema::endEnumScalar() {}

bool GenerateSchema::beginBitSetScalar(bool &) {
  beginEnumScalar();
  return true;
}

bool GenerateSchema::bitSetMatch(StringRef Val, bool Arg) {
  return matchEnumScalar(Val, Arg);
}

void GenerateSchema::endBitSetScalar() { endEnumScalar(); }

void GenerateSchema::scalarString(StringRef &Val, QuotingType) {
  Schema *Top = getTopSchema();
  assert(Top);
  TypeProperty *Type = createProperty<TypeProperty>("string");
  Top->emplace_back(Type);
}

void GenerateSchema::blockScalarString(StringRef &S) {
  scalarString(S, QuotingType::None);
}

void GenerateSchema::scalarTag(std::string &val) {}

NodeKind GenerateSchema::getNodeKind() { report_fatal_error("invalid call"); }

void GenerateSchema::setError(const Twine &) {}

std::error_code GenerateSchema::error() { return {}; }

bool GenerateSchema::canElideEmptySequence() { return false; }

// These are only used by operator<<. They could be private
// if that templated operator could be made a friend.

bool GenerateSchema::preflightDocument() {
  Schema *S = createSchema();
  Root = S;
  Schemas.push_back(S);
  return true;
}

void GenerateSchema::postflightDocument() {
  assert(!Schemas.empty());
  Schemas.pop_back();
  json::Value JSONValue = Root->toJSON();
  RO << llvm::formatv("{0:2}\n", JSONValue);
}

json::Value GenerateSchema::UserDefinedProperty::toJSON() const {
  return Value->toJSON();
}

json::Value GenerateSchema::PropertiesProperty::toJSON() const {
  json::Object JSONObject;
  for (UserDefinedProperty *Property : *this) {
    json::Value JSONValue = Property->toJSON();
    JSONObject.try_emplace(Property->getName().data(), std::move(JSONValue));
  }
  return JSONObject;
}

json::Value GenerateSchema::AdditionalPropertiesProperty::toJSON() const {
  return Value->toJSON();
}

json::Value GenerateSchema::RequiredProperty::toJSON() const {
  json::Array JSONArray;
  for (StringRef Value : *this) {
    json::Value JSONValue(Value.data());
    JSONArray.emplace_back(std::move(JSONValue));
  }
  return JSONArray;
}

json::Value GenerateSchema::OptionalProperty::toJSON() const {
  json::Array JSONArray;
  for (StringRef Value : *this) {
    json::Value JSONValue(Value.data());
    JSONArray.emplace_back(std::move(JSONValue));
  }
  return JSONArray;
}

json::Value GenerateSchema::TypeProperty::toJSON() const {
  json::Value JSONValue(Value.data());
  return JSONValue;
}

json::Value GenerateSchema::EnumProperty::toJSON() const {
  json::Array JSONArray;
  for (StringRef Value : *this) {
    json::Value JSONValue(Value.data());
    JSONArray.emplace_back(std::move(JSONValue));
  }
  return JSONArray;
}

json::Value GenerateSchema::ItemsProperty::toJSON() const {
  return Value->toJSON();
}

json::Value GenerateSchema::FlowStyleProperty::toJSON() const {
  StringRef Value;
  switch (Style) {
  case FlowStyle::Block:
    Value = "block";
    break;
  case FlowStyle::Flow:
    Value = "flow";
    break;
  }
  json::Value JSONValue(Value.data());
  return JSONValue;
}

json::Value GenerateSchema::Schema::toJSON() const {
  json::Object JSONObject;
  for (SchemaProperty *Value : *this) {
    json::Value JSONValue = Value->toJSON();
    StringRef Key = Value->getName();
    JSONObject.try_emplace(Key.data(), std::move(JSONValue));
  }
  return JSONObject;
}
