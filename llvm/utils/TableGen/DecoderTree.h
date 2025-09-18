//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_DECODERTREE_H
#define LLVM_UTILS_TABLEGEN_DECODERTREE_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <memory>

namespace llvm {

class InstructionEncoding;

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

class CheckAnyNode : public CheckManyNode {
public:
  CheckAnyNode() : CheckManyNode(CheckAny) {}
};

class CheckAllNode : public CheckManyNode {
public:
  CheckAllNode() : CheckManyNode(CheckAll) {}
};

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

class CheckPredicateNode : public DecoderTreeNode {
  std::string PredicateString;

public:
  explicit CheckPredicateNode(std::string PredicateString)
      : DecoderTreeNode(CheckPredicate),
        PredicateString(std::move(PredicateString)) {}

  StringRef getPredicateString() const { return PredicateString; }
};

class SoftFailNode : public DecoderTreeNode {
  uint64_t PositiveMask, NegativeMask;

public:
  SoftFailNode(uint64_t PositiveMask, uint64_t NegativeMask)
      : DecoderTreeNode(SoftFail), PositiveMask(PositiveMask),
        NegativeMask(NegativeMask) {}

  uint64_t getPositiveMask() const { return PositiveMask; }
  uint64_t getNegativeMask() const { return NegativeMask; }
};

class DecodeNode : public DecoderTreeNode {
  const InstructionEncoding &Encoding;
  std::string DecoderString;

public:
  DecodeNode(const InstructionEncoding &Encoding, std::string DecoderString)
      : DecoderTreeNode(Decode), Encoding(Encoding),
        DecoderString(std::move(DecoderString)) {}

  const InstructionEncoding &getEncoding() const { return Encoding; }

  StringRef getDecoderString() const { return DecoderString; }
};

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_DECODERTREE_H
