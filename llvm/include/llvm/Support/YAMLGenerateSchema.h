//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_YAMLGENERATE_SCHEMA_H
#define LLVM_SUPPORT_YAMLGENERATE_SCHEMA_H

#include "llvm/Support/Casting.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {

namespace json {
class Value;
}

namespace yaml {

class GenerateSchema : public IO {
public:
  GenerateSchema(raw_ostream &RO);
  ~GenerateSchema() override = default;

  IOKind getKind() const override;
  bool outputting() const override;
  bool mapTag(StringRef, bool) override;
  void beginMapping() override;
  void endMapping() override;
  bool preflightKey(StringRef, bool, bool, bool &, void *&) override;
  void postflightKey(void *) override;
  std::vector<StringRef> keys() override;
  void beginFlowMapping() override;
  void endFlowMapping() override;
  unsigned beginSequence() override;
  void endSequence() override;
  bool preflightElement(unsigned, void *&) override;
  void postflightElement(void *) override;
  unsigned beginFlowSequence() override;
  bool preflightFlowElement(unsigned, void *&) override;
  void postflightFlowElement(void *) override;
  void endFlowSequence() override;
  void beginEnumScalar() override;
  bool matchEnumScalar(StringRef, bool) override;
  bool matchEnumFallback() override;
  void endEnumScalar() override;
  bool beginBitSetScalar(bool &) override;
  bool bitSetMatch(StringRef, bool) override;
  void endBitSetScalar() override;
  void scalarString(StringRef &, QuotingType) override;
  void blockScalarString(StringRef &) override;
  void scalarTag(std::string &) override;
  NodeKind getNodeKind() override;
  void setError(const Twine &message) override;
  std::error_code error() override;
  bool canElideEmptySequence() override;

  bool preflightDocument();
  void postflightDocument();

  class SchemaNode {
  public:
    virtual json::Value toJSON() const = 0;

    virtual ~SchemaNode() = default;
  };

  enum class PropertyKind : uint8_t {
    UserDefined,
    Properties,
    AdditionalProperties,
    Required,
    Optional,
    Type,
    Enum,
    Items,
    FlowStyle,
  };

  class SchemaProperty : public SchemaNode {
    StringRef Name;
    PropertyKind Kind;

  public:
    SchemaProperty(StringRef Name, PropertyKind Kind)
        : Name(Name), Kind(Kind) {}

    PropertyKind getKind() const { return Kind; }

    StringRef getName() const { return Name; }
  };

  class Schema;

  class UserDefinedProperty final : public SchemaProperty {
    Schema *Value;

  public:
    UserDefinedProperty(StringRef Name, Schema *Value)
        : SchemaProperty(Name, PropertyKind::UserDefined), Value(Value) {}

    Schema *getSchema() const { return Value; }

    json::Value toJSON() const override;

    static bool classof(const SchemaProperty *Property) {
      return Property->getKind() == PropertyKind::UserDefined;
    }
  };

  class PropertiesProperty final : public SchemaProperty,
                                   SmallVector<UserDefinedProperty *, 8> {
  public:
    using BaseVector = SmallVector<UserDefinedProperty *, 8>;

    PropertiesProperty()
        : SchemaProperty("properties", PropertyKind::Properties) {}

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;

    json::Value toJSON() const override;

    static bool classof(const SchemaProperty *Property) {
      return Property->getKind() == PropertyKind::Properties;
    }
  };

  class AdditionalPropertiesProperty final : public SchemaProperty {
    Schema *Value;

  public:
    AdditionalPropertiesProperty(Schema *Value = nullptr)
        : SchemaProperty("additionalProperties",
                         PropertyKind::AdditionalProperties),
          Value(Value) {}

    Schema *getSchema() const { return Value; }

    void setSchema(Schema *S) { Value = S; }

    json::Value toJSON() const override;

    static bool classof(const SchemaProperty *Property) {
      return Property->getKind() == PropertyKind::AdditionalProperties;
    }
  };

  class RequiredProperty final : public SchemaProperty,
                                 SmallVector<StringRef, 4> {
  public:
    using BaseVector = SmallVector<StringRef, 4>;

    RequiredProperty() : SchemaProperty("required", PropertyKind::Required) {}

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;

    json::Value toJSON() const override;

    static bool classof(const SchemaProperty *Property) {
      return Property->getKind() == PropertyKind::Required;
    }
  };

  class OptionalProperty final : public SchemaProperty,
                                 SmallVector<StringRef, 4> {
  public:
    using BaseVector = SmallVector<StringRef, 4>;

    OptionalProperty() : SchemaProperty("optional", PropertyKind::Optional) {}

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;

    json::Value toJSON() const override;

    static bool classof(const SchemaProperty *Property) {
      return Property->getKind() == PropertyKind::Optional;
    }
  };

  class TypeProperty final : public SchemaProperty {
    StringRef Value;

  public:
    TypeProperty(StringRef Value)
        : SchemaProperty("type", PropertyKind::Type), Value(Value) {}

    json::Value toJSON() const override;

    static bool classof(const SchemaProperty *Property) {
      return Property->getKind() == PropertyKind::Type;
    }
  };

  class EnumProperty final : public SchemaProperty, SmallVector<StringRef, 4> {
  public:
    using BaseVector = SmallVector<StringRef, 4>;

    EnumProperty() : SchemaProperty("enum", PropertyKind::Enum) {}

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;

    json::Value toJSON() const override;

    static bool classof(const SchemaProperty *Property) {
      return Property->getKind() == PropertyKind::Enum;
    }
  };

  class ItemsProperty final : public SchemaProperty {
    Schema *Value;

  public:
    ItemsProperty(Schema *Value = nullptr)
        : SchemaProperty("items", PropertyKind::Items), Value(Value) {}

    Schema *getSchema() const { return Value; }

    void setSchema(Schema *S) { Value = S; }

    json::Value toJSON() const override;

    static bool classof(const SchemaProperty *Property) {
      return Property->getKind() == PropertyKind::Items;
    }
  };

  enum class FlowStyle : bool {
    Block,
    Flow,
  };

  class FlowStyleProperty final : public SchemaProperty {
    FlowStyle Style;

  public:
    FlowStyleProperty(FlowStyle Style = FlowStyle::Block)
        : SchemaProperty("flowStyle", PropertyKind::FlowStyle), Style(Style) {}

    void setStyle(FlowStyle S) { Style = S; }

    FlowStyle getStyle() const { return Style; }

    json::Value toJSON() const override;

    static bool classof(const SchemaProperty *Property) {
      return Property->getKind() == PropertyKind::FlowStyle;
    }
  };

  class Schema final : public SchemaNode, SmallVector<SchemaProperty *, 8> {
  public:
    using BaseVector = SmallVector<SchemaProperty *, 8>;

    Schema() = default;

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;

    json::Value toJSON() const override;
  };

private:
  std::vector<std::unique_ptr<SchemaNode>> SchemaNodes;
  SmallVector<Schema *, 8> Schemas;
  raw_ostream &RO;
  SchemaNode *Root = nullptr;

  template <typename PropertyType, typename... PropertyArgs>
  PropertyType *createProperty(PropertyArgs &&...Args) {
    std::unique_ptr<PropertyType> UPtr =
        std::make_unique<PropertyType>(std::forward<PropertyArgs>(Args)...);
    PropertyType *Ptr = UPtr.get();
    SchemaNodes.emplace_back(std::move(UPtr));
    return Ptr;
  }

  template <typename PropertyType, typename... PropertyArgs>
  PropertyType *getOrCreateProperty(Schema &S, PropertyArgs... Args) {
    auto Found = std::find_if(S.begin(), S.end(), [](SchemaProperty *Property) {
      return isa<PropertyType>(Property);
    });
    if (Found != S.end()) {
      return cast<PropertyType>(*Found);
    }
    PropertyType *Created =
        createProperty<PropertyType>(std::forward<PropertyArgs>(Args)...);
    S.emplace_back(Created);
    return Created;
  }

  Schema *createSchema() {
    std::unique_ptr<Schema> UPtr = std::make_unique<Schema>();
    Schema *Ptr = UPtr.get();
    SchemaNodes.emplace_back(std::move(UPtr));
    return Ptr;
  }

  Schema *getTopSchema() const {
    return Schemas.empty() ? nullptr : Schemas.back();
  }
};

// Define non-member operator<< so that Output can stream out document list.
template <typename T>
inline std::enable_if_t<has_DocumentListTraits<T>::value, GenerateSchema &>
operator<<(GenerateSchema &Gen, T &DocList) {
  EmptyContext Ctx;
  Gen.preflightDocument();
  yamlize(Gen, DocumentListTraits<T>::element(Gen, DocList, 0), true, Ctx);
  Gen.postflightDocument();
  return Gen;
}

// Define non-member operator<< so that Output can stream out a map.
template <typename T>
inline std::enable_if_t<has_MappingTraits<T, EmptyContext>::value,
                        GenerateSchema &>
operator<<(GenerateSchema &Gen, T &Map) {
  EmptyContext Ctx;
  Gen.preflightDocument();
  yamlize(Gen, Map, true, Ctx);
  Gen.postflightDocument();
  return Gen;
}

// Define non-member operator<< so that Output can stream out a sequence.
template <typename T>
inline std::enable_if_t<has_SequenceTraits<T>::value, GenerateSchema &>
operator<<(GenerateSchema &Gen, T &Seq) {
  EmptyContext Ctx;
  Gen.preflightDocument();
  yamlize(Gen, Seq, true, Ctx);
  Gen.postflightDocument();
  return Gen;
}

// Define non-member operator<< so that Output can stream out a block scalar.
template <typename T>
inline std::enable_if_t<has_BlockScalarTraits<T>::value, GenerateSchema &>
operator<<(GenerateSchema &Gen, T &Val) {
  EmptyContext Ctx;
  Gen.preflightDocument();
  yamlize(Gen, Val, true, Ctx);
  Gen.postflightDocument();
  return Gen;
}

// Define non-member operator<< so that Output can stream out a string map.
template <typename T>
inline std::enable_if_t<has_CustomMappingTraits<T>::value, GenerateSchema &>
operator<<(GenerateSchema &Gen, T &Val) {
  EmptyContext Ctx;
  Gen.preflightDocument();
  yamlize(Gen, Val, true, Ctx);
  Gen.postflightDocument();
  return Gen;
}

// Define non-member operator<< so that Output can stream out a polymorphic
// type.
template <typename T>
inline std::enable_if_t<has_PolymorphicTraits<T>::value, GenerateSchema &>
operator<<(GenerateSchema &Gen, T &Val) {
  EmptyContext Ctx;
  Gen.preflightDocument();
  yamlize(Gen, Val, true, Ctx);
  Gen.postflightDocument();
  return Gen;
}

// Provide better error message about types missing a trait specialization
template <typename T>
inline std::enable_if_t<missingTraits<T, EmptyContext>::value, GenerateSchema &>
operator<<(GenerateSchema &Gen, T &seq) {
  char missing_yaml_trait_for_type[sizeof(MissingTrait<T>)];
  return Gen;
}

} // namespace yaml

} // namespace llvm

#endif // LLVM_SUPPORT_YAMLGENERATE_SCHEMA_H
