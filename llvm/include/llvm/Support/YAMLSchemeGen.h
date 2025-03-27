//===- llvm/Support/YAMLSchemeGen.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_YAMLSCHEMEGEN_H
#define LLVM_SUPPORT_YAMLSCHEMEGEN_H

#include "llvm/Support/YAMLTraits.h"

namespace llvm {

namespace yaml {

class SchemeGen : public IO {
public:
  SchemeGen(raw_ostream &RO, void *Ctxt = nullptr, int WrapColumn = 70);
  ~SchemeGen() override = default;

  IOKind getKind() const override;
  bool outputting() const override;
  bool mapTag(StringRef, bool) override;
  void beginMapping() override;
  void endMapping() override;
  bool preflightKey(const char *key, bool, bool, bool &, void *&) override;
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
  bool matchEnumScalar(const char *, bool) override;
  bool matchEnumFallback() override;
  void endEnumScalar() override;
  bool beginBitSetScalar(bool &) override;
  bool bitSetMatch(const char *, bool) override;
  void endBitSetScalar() override;
  void scalarString(StringRef &, QuotingType) override;
  void blockScalarString(StringRef &) override;
  void scalarTag(std::string &) override;
  NodeKind getNodeKind() override;
  void setError(const Twine &message) override;
  std::error_code error() override;
  bool canElideEmptySequence() override;

  // These are only used by operator<<. They could be private
  // if that templated operator could be made a friend.
  void beginDocuments();
  bool preflightDocument(unsigned);
  void postflightDocument();
  void endDocuments();

  // These are needed to yamlize scheme.
  struct YNode {
  public:
    YNode(NodeKind Kind) : Kind(Kind) {}

    NodeKind getNodeKind() const { return Kind; }

  private:
    NodeKind Kind;
  };

  class ScalarYNode final : public YNode {
    StringRef Str;
    QuotingType QT;

  public:
    ScalarYNode(StringRef Str, QuotingType QT)
        : YNode(NodeKind::Scalar), Str(Str), QT(QT) {}

    StringRef getValue() const { return Str; }

    QuotingType getQuotingType() const { return QT; }
  };

  class MapYNode final : public YNode,
                         SmallVector<std::pair<StringRef, YNode *>, 8> {
  public:
    using BaseVector = SmallVector<std::pair<StringRef, YNode *>, 8>;

    MapYNode() : YNode(NodeKind::Map) {}

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;
  };

  class SequenceYNode final : public YNode, SmallVector<YNode *, 8> {
  public:
    using BaseVector = SmallVector<YNode *, 8>;

    SequenceYNode() : YNode(NodeKind::Sequence) {}

    using BaseVector::operator[];
    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::resize;
    using BaseVector::size;
  };

private:
  template <typename YNodeType, typename... YNodeArgs>
  YNodeType *createYNode(YNodeArgs &&...Args) {
    auto UPtr = std::make_unique<YNodeType>(std::forward<YNodeArgs>(Args)...);
    auto *Ptr = UPtr.get();
    YNodes.emplace_back(std::move(UPtr));
    return Ptr;
  }

public:
  ScalarYNode *createScalarYNode(StringRef Str) {
    return createYNode<ScalarYNode>(Str, QuotingType::None);
  }

  MapYNode *createMapYNode() { return createYNode<MapYNode>(); }

  SequenceYNode *createSequenceYNode() { return createYNode<SequenceYNode>(); }

  std::vector<std::unique_ptr<YNode>> YNodes;

  using StringVector = SmallVector<StringRef, 8>;

  class SchemeNode {
  public:
    virtual YNode *yamlize(SchemeGen &Gen) const = 0;

    virtual ~SchemeNode() = default;
  };

  enum class PropertyKind : uint8_t {
    Properties,
    Required,
    Type,
    Enum,
    Items,
  };

  class Scheme;

  class SchemeProperty : public SchemeNode {
    PropertyKind Kind;
    StringRef Name;

  public:
    SchemeProperty(PropertyKind Kind, StringRef Name)
        : Kind(Kind), Name(Name) {}

    PropertyKind getKind() const { return Kind; }

    StringRef getName() const { return Name; }
  };

  class PropertiesProperty final
      : public SchemeProperty,
        SmallVector<std::pair<StringRef, Scheme *>, 8> {
  public:
    using BaseVector = SmallVector<std::pair<StringRef, Scheme *>, 8>;

    PropertiesProperty()
        : SchemeProperty(PropertyKind::Properties, "properties") {}

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;

    YNode *yamlize(SchemeGen &Gen) const override {
      auto *Map = Gen.createMapYNode();
      for (auto &&[Key, Value] : *this) {
        auto *Y = Value->yamlize(Gen);
        Map->emplace_back(Key, Y);
      }
      return Map;
    }
  };

  class RequiredProperty final : public SchemeProperty, StringVector {
  public:
    using BaseVector = StringVector;

    RequiredProperty() : SchemeProperty(PropertyKind::Required, "required") {}

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;

    YNode *yamlize(SchemeGen &Gen) const override {
      if (size() == 1) {
        auto *Scalar = Gen.createScalarYNode(*begin());
        return Scalar;
      }
      auto *Sequence = Gen.createSequenceYNode();
      for (auto &&Value : *this) {
        auto *Scalar = Gen.createScalarYNode(Value);
        Sequence->emplace_back(Scalar);
      }
      return Sequence;
    }
  };

  class TypeProperty final : public SchemeProperty {
    StringRef Value = "any";

  public:
    TypeProperty() : SchemeProperty(PropertyKind::Type, "type") {}

    StringRef getValue() const { return Value; }

    void setValue(StringRef Val) { Value = Val; }

    YNode *yamlize(SchemeGen &Gen) const override {
      auto *Scalar = Gen.createScalarYNode(Value);
      return Scalar;
    }
  };

  class EnumProperty final : public SchemeProperty, StringVector {
  public:
    using BaseVector = StringVector;

    EnumProperty() : SchemeProperty(PropertyKind::Enum, "enum") {}

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;

    YNode *yamlize(SchemeGen &Gen) const override {
      auto *Sequence = Gen.createSequenceYNode();
      for (auto &&Value : *this) {
        auto *Scalar = Gen.createScalarYNode(Value);
        Sequence->emplace_back(Scalar);
      }
      return Sequence;
    }
  };

  class ItemsProperty final : public SchemeProperty, SmallVector<Scheme *, 8> {
  public:
    using BaseVector = SmallVector<Scheme *, 8>;

    ItemsProperty() : SchemeProperty(PropertyKind::Items, "items") {}

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;

    YNode *yamlize(SchemeGen &Gen) const override {
      auto *Sequence = Gen.createSequenceYNode();
      for (auto &&Value : *this) {
        auto *Y = Value->yamlize(Gen);
        Sequence->emplace_back(Y);
      }
      return Sequence;
    }
  };

  class Scheme final : public SchemeNode, SmallVector<SchemeProperty *, 8> {
  public:
    using BaseVector = SmallVector<SchemeProperty *, 8>;

    Scheme() = default;

    using BaseVector::begin;
    using BaseVector::emplace_back;
    using BaseVector::end;
    using BaseVector::size;

    template <typename PropertyType> PropertyType *findProperty() const {
      auto P = PropertyType{};
      auto Found = std::find_if(begin(), end(), [&](auto &&Prop) {
        return Prop->getKind() == P.getKind();
      });
      return Found != end() ? static_cast<PropertyType *>(*Found) : nullptr;
    }

    template <typename PropertyType> bool hasProperty() const {
      return findProperty<PropertyType>() != nullptr;
    }

    YNode *yamlize(SchemeGen &Gen) const override {
      auto *Map = Gen.createMapYNode();
      for (auto &&Value : *this) {
        auto *Y = Value->yamlize(Gen);
        auto Name = Value->getName();
        Map->emplace_back(Name, Y);
      }
      return Map;
    }
  };

private:
  std::vector<std::unique_ptr<SchemeNode>> SchemeNodes;
  SmallVector<Scheme *, 8> Schemes;

  template <typename PropertyType> PropertyType *createProperty() {
    auto UPtr = std::make_unique<PropertyType>();
    auto *Ptr = UPtr.get();
    SchemeNodes.emplace_back(std::move(UPtr));
    return Ptr;
  }

public:
  template <typename PropertyType>
  PropertyType *getOrCreateProperty(Scheme &S) {
    auto Found = S.findProperty<PropertyType>();
    if (!Found) {
      Found = createProperty<PropertyType>();
      S.emplace_back(Found);
    }
    return Found;
  }

  Scheme *createScheme() {
    auto UPtr = std::make_unique<Scheme>();
    auto *Ptr = UPtr.get();
    SchemeNodes.emplace_back(std::move(UPtr));
    return Ptr;
  }

  Scheme *getTopScheme() const {
    return Schemes.empty() ? nullptr : Schemes.back();
  }

private:
  Output O;
  SchemeNode *Root = nullptr;
};

template <> struct PolymorphicTraits<SchemeGen::YNode> {
  static NodeKind getKind(const SchemeGen::YNode &N) { return N.getNodeKind(); }

  static SchemeGen::ScalarYNode &getAsScalar(SchemeGen::YNode &N) {
    return (SchemeGen::ScalarYNode &)N;
  }

  static SchemeGen::MapYNode &getAsMap(SchemeGen::YNode &N) {
    return (SchemeGen::MapYNode &)N;
  }

  static SchemeGen::SequenceYNode &getAsSequence(SchemeGen::YNode &N) {
    return (SchemeGen::SequenceYNode &)N;
  }
};

template <> struct ScalarTraits<SchemeGen::ScalarYNode> {
  static void output(const SchemeGen::ScalarYNode &N, void *Ctx,
                     llvm::raw_ostream &Out) {
    Out << N.getValue();
  }

  static StringRef input(StringRef Scalar, void *Ctx,
                         SchemeGen::ScalarYNode &N) {
    return {};
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

template <> struct MappingTraits<SchemeGen::MapYNode> {
  static void mapping(IO &io, SchemeGen::MapYNode &N) {
    for (auto &&[Key, Value] : N) {
      io.mapRequired(Key.data(), *Value);
    }
  }
};

template <> struct SequenceTraits<SchemeGen::SequenceYNode> {
  static size_t size(IO &io, SchemeGen::SequenceYNode &N) { return N.size(); }

  static SchemeGen::YNode &element(IO &, SchemeGen::SequenceYNode &N,
                                   size_t Idx) {
    if (Idx >= N.size())
      N.resize(Idx + 1);
    return *(N[Idx]);
  }
};

// Define non-member operator<< so that Output can stream out document list.
template <typename T>
inline std::enable_if_t<has_DocumentListTraits<T>::value, SchemeGen &>
operator<<(SchemeGen &Gen, T &DocList) {
  EmptyContext Ctx;
  Gen.beginDocuments();
  const size_t count = DocumentListTraits<T>::size(Gen, DocList);
  for (size_t i = 0; i < count; ++i) {
    if (Gen.preflightDocument(i)) {
      yamlize(Gen, DocumentListTraits<T>::element(Gen, DocList, i), true, Ctx);
      Gen.postflightDocument();
    }
  }
  Gen.endDocuments();
  return Gen;
}

// Define non-member operator<< so that Output can stream out a map.
template <typename T>
inline std::enable_if_t<has_MappingTraits<T, EmptyContext>::value, SchemeGen &>
operator<<(SchemeGen &Gen, T &Map) {
  EmptyContext Ctx;
  Gen.beginDocuments();
  if (Gen.preflightDocument(0)) {
    yamlize(Gen, Map, true, Ctx);
    Gen.postflightDocument();
  }
  Gen.endDocuments();
  return Gen;
}

// Define non-member operator<< so that Output can stream out a sequence.
template <typename T>
inline std::enable_if_t<has_SequenceTraits<T>::value, SchemeGen &>
operator<<(SchemeGen &Gen, T &Seq) {
  EmptyContext Ctx;
  Gen.beginDocuments();
  if (Gen.preflightDocument(0)) {
    yamlize(Gen, Seq, true, Ctx);
    Gen.postflightDocument();
  }
  Gen.endDocuments();
  return Gen;
}

// Define non-member operator<< so that Output can stream out a block scalar.
template <typename T>
inline std::enable_if_t<has_BlockScalarTraits<T>::value, SchemeGen &>
operator<<(SchemeGen &Gen, T &Val) {
  EmptyContext Ctx;
  Gen.beginDocuments();
  if (Gen.preflightDocument(0)) {
    yamlize(Gen, Val, true, Ctx);
    Gen.postflightDocument();
  }
  Gen.endDocuments();
  return Gen;
}

// Define non-member operator<< so that Output can stream out a string map.
template <typename T>
inline std::enable_if_t<has_CustomMappingTraits<T>::value, SchemeGen &>
operator<<(SchemeGen &Gen, T &Val) {
  EmptyContext Ctx;
  Gen.beginDocuments();
  if (Gen.preflightDocument(0)) {
    yamlize(Gen, Val, true, Ctx);
    Gen.postflightDocument();
  }
  Gen.endDocuments();
  return Gen;
}

// Define non-member operator<< so that Output can stream out a polymorphic
// type.
template <typename T>
inline std::enable_if_t<has_PolymorphicTraits<T>::value, SchemeGen &>
operator<<(SchemeGen &Gen, T &Val) {
  EmptyContext Ctx;
  Gen.beginDocuments();
  if (Gen.preflightDocument(0)) {
    // FIXME: The parser does not support explicit documents terminated with a
    // plain scalar; the end-marker is included as part of the scalar token.
    assert(PolymorphicTraits<T>::getKind(Val) != NodeKind::Scalar &&
           "plain scalar documents are not supported");
    yamlize(Gen, Val, true, Ctx);
    Gen.postflightDocument();
  }
  Gen.endDocuments();
  return Gen;
}

// Provide better error message about types missing a trait specialization
template <typename T>
inline std::enable_if_t<missingTraits<T, EmptyContext>::value, SchemeGen &>
operator<<(SchemeGen &Gen, T &seq) {
  char missing_yaml_trait_for_type[sizeof(MissingTrait<T>)];
  return Gen;
}

} // namespace yaml

} // namespace llvm

#endif // LLVM_SUPPORT_YAMLSCHEMEGEN_H
