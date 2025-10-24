//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_DECODERTREE_H
#define LLVM_UTILS_TABLEGEN_DECODERTREE_H

#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <memory>

namespace llvm {

class InstructionEncoding;

using PredicateSet = SetVector<CachedHashString>;
using DecoderSet = SetVector<CachedHashString>;

/// Context shared across decoder trees.
/// Predicates and decoders are shared across decoder trees to provide more
/// opportunities for uniqueness. If SpecializeDecodersPerBitwidth is enabled,
/// decoders are shared across all trees for a given bitwidth, else they are
/// shared across all trees. Predicates are always shared across all trees.
struct DecoderContext {
  PredicateSet Predicates;
  DecoderSet Decoders;

  unsigned getPredicateIndex(StringRef Predicate) {
    Predicates.insert(CachedHashString(Predicate));
    PredicateSet::const_iterator I = find(Predicates, Predicate);
    return std::distance(Predicates.begin(), I);
  }

  unsigned getDecoderIndex(StringRef Decoder) {
    Decoders.insert(CachedHashString(Decoder));
    DecoderSet::const_iterator I = find(Decoders, Decoder);
    return std::distance(Decoders.begin(), I);
  }
};

class DecoderTreeNode {
public:
  virtual ~DecoderTreeNode();

  enum KindTy {
    CheckAny,
    CheckAll,
    CheckField,
    SwitchField,
    CheckPredicate,
    SoftFail,
    Decode,
  };

  KindTy getKind() const { return Kind; }

protected:
  explicit DecoderTreeNode(KindTy Kind) : Kind(Kind) {}

private:
  KindTy Kind;
};

/// Common base class for nodes with multiple children.
class CheckManyNode : public DecoderTreeNode {
  SmallVector<std::unique_ptr<DecoderTreeNode>, 0> Children;

  static const DecoderTreeNode *
  mapElement(decltype(Children)::const_reference Element) {
    return Element.get();
  }

protected:
  explicit CheckManyNode(KindTy Kind) : DecoderTreeNode(Kind) {}

public:
  void addChild(std::unique_ptr<DecoderTreeNode> Child) {
    Children.push_back(std::move(Child));
  }

  using child_iterator = mapped_iterator<decltype(Children)::const_iterator,
                                         decltype(&mapElement)>;

  child_iterator child_begin() const {
    return child_iterator(Children.begin(), mapElement);
  }

  child_iterator child_end() const {
    return child_iterator(Children.end(), mapElement);
  }

  iterator_range<child_iterator> children() const {
    return make_range(child_begin(), child_end());
  }
};

/// Executes child nodes one by one until one of them succeeds or all fail.
/// The node fails if all child nodes fail. It never succeeds, because if a
/// child node succeeds, it does not return.
class CheckAnyNode : public CheckManyNode {
public:
  CheckAnyNode() : CheckManyNode(CheckAny) {}
};

/// Executes child nodes one by one until one of them fails all all succeed.
/// The node fails if any of the child nodes fails.
class CheckAllNode : public CheckManyNode {
public:
  CheckAllNode() : CheckManyNode(CheckAll) {}
};

/// Checks the value of encoding bits in the specified range.
class CheckFieldNode : public DecoderTreeNode {
  unsigned StartBit;
  unsigned NumBits;
  uint64_t Value;

public:
  CheckFieldNode(unsigned StartBit, unsigned NumBits, uint64_t Value)
      : DecoderTreeNode(CheckField), StartBit(StartBit), NumBits(NumBits),
        Value(Value) {}

  unsigned getStartBit() const { return StartBit; }

  unsigned getNumBits() const { return NumBits; }

  uint64_t getValue() const { return Value; }
};

/// Switch based on the value of encoding bits in the specified range.
/// If the value of the bits in the range doesn't match any of the cases,
/// the node fails. This is semantically equivalent to CheckAny node where
/// every child is a CheckField node, but is faster.
class SwitchFieldNode : public DecoderTreeNode {
  unsigned StartBit;
  unsigned NumBits;
  std::map<uint64_t, std::unique_ptr<DecoderTreeNode>> Cases;

  static std::pair<uint64_t, const DecoderTreeNode *>
  mapElement(decltype(Cases)::const_reference Element) {
    return std::pair(Element.first, Element.second.get());
  }

public:
  SwitchFieldNode(unsigned StartBit, unsigned NumBits)
      : DecoderTreeNode(SwitchField), StartBit(StartBit), NumBits(NumBits) {}

  void addCase(uint64_t Value, std::unique_ptr<DecoderTreeNode> N) {
    Cases.try_emplace(Value, std::move(N));
  }

  unsigned getStartBit() const { return StartBit; }

  unsigned getNumBits() const { return NumBits; }

  using case_iterator =
      mapped_iterator<decltype(Cases)::const_iterator, decltype(&mapElement)>;

  case_iterator case_begin() const {
    return case_iterator(Cases.begin(), mapElement);
  }

  case_iterator case_end() const {
    return case_iterator(Cases.end(), mapElement);
  }

  iterator_range<case_iterator> cases() const {
    return make_range(case_begin(), case_end());
  }
};

/// Checks that the instruction to be decoded has its predicates satisfied.
class CheckPredicateNode : public DecoderTreeNode {
  unsigned PredicateIndex;

public:
  explicit CheckPredicateNode(unsigned PredicateIndex)
      : DecoderTreeNode(CheckPredicate), PredicateIndex(PredicateIndex) {}

  unsigned getPredicateIndex() const { return PredicateIndex; }
};

/// Checks if the encoding bits are correct w.r.t. SoftFail semantics.
/// This is the only node that can never fail.
class SoftFailNode : public DecoderTreeNode {
  uint64_t PositiveMask, NegativeMask;

public:
  SoftFailNode(uint64_t PositiveMask, uint64_t NegativeMask)
      : DecoderTreeNode(SoftFail), PositiveMask(PositiveMask),
        NegativeMask(NegativeMask) {}

  uint64_t getPositiveMask() const { return PositiveMask; }
  uint64_t getNegativeMask() const { return NegativeMask; }
};

/// Attempts to decode the specified encoding as the specified instruction.
class DecodeNode : public DecoderTreeNode {
  StringRef EncodingName;
  unsigned InstOpcode;
  unsigned DecoderIndex;

public:
  DecodeNode(StringRef EncodingName, unsigned InstOpcode, unsigned DecoderIndex)
      : DecoderTreeNode(Decode), EncodingName(EncodingName),
        InstOpcode(InstOpcode), DecoderIndex(DecoderIndex) {}

  StringRef getEncodingName() const { return EncodingName; }

  unsigned getInstOpcode() const { return InstOpcode; }

  unsigned getDecoderIndex() const { return DecoderIndex; }
};

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_DECODERTREE_H
