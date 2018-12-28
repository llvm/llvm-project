//===- SPIRVDecorate.h - SPIR-V Decorations ---------------------*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines SPIR-V decorations.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVDECORATE_H
#define SPIRV_LIBSPIRV_SPIRVDECORATE_H

#include "SPIRVEntry.h"
#include "SPIRVStream.h"
#include "SPIRVUtil.h"
#include <string>
#include <utility>
#include <vector>

namespace SPIRV {
class SPIRVDecorationGroup;
class SPIRVDecorateGeneric : public SPIRVAnnotationGeneric {
public:
  // Complete constructor for decorations without literals
  SPIRVDecorateGeneric(Op OC, SPIRVWord WC, Decoration TheDec,
                       SPIRVEntry *TheTarget);
  // Complete constructor for decorations with one word literal
  SPIRVDecorateGeneric(Op OC, SPIRVWord WC, Decoration TheDec,
                       SPIRVEntry *TheTarget, SPIRVWord V);
  // Incomplete constructor
  SPIRVDecorateGeneric(Op OC);

  SPIRVWord getLiteral(size_t) const;
  Decoration getDecorateKind() const;
  size_t getLiteralCount() const;
  /// Compare for kind and literal only.
  struct Comparator {
    bool operator()(const SPIRVDecorateGeneric *A,
                    const SPIRVDecorateGeneric *B) const;
  };
  /// Compare kind, literals and target.
  friend bool operator==(const SPIRVDecorateGeneric &A,
                         const SPIRVDecorateGeneric &B);

  SPIRVDecorationGroup *getOwner() const { return Owner; }

  void setOwner(SPIRVDecorationGroup *Owner) { this->Owner = Owner; }

  SPIRVCapVec getRequiredCapability() const override {
    return getCapability(Dec);
  }

  SPIRVWord getRequiredSPIRVVersion() const override {
    switch (Dec) {
    case DecorationSpecId:
      if (getModule()->hasCapability(CapabilityKernel))
        return SPIRV_1_1;
      else
        return SPIRV_1_0;

    case DecorationMaxByteOffset:
      return SPIRV_1_1;

    default:
      return SPIRV_1_0;
    }
  }

protected:
  Decoration Dec;
  std::vector<SPIRVWord> Literals;
  SPIRVDecorationGroup *Owner; // Owning decorate group
};

class SPIRVDecorateSet
    : public std::multiset<SPIRVDecorateGeneric *,
                           SPIRVDecorateGeneric::Comparator> {
public:
  typedef std::multiset<SPIRVDecorateGeneric *,
                        SPIRVDecorateGeneric::Comparator>
      BaseType;
  iterator insert(value_type &Dec) {
    auto ER = BaseType::equal_range(Dec);
    for (auto I = ER.first, E = ER.second; I != E; ++I) {
      SPIRVDBG(spvdbgs() << "[compare decorate] " << *Dec << " vs " << **I
                         << " : ");
      if (**I == *Dec)
        return I;
      SPIRVDBG(spvdbgs() << " diff\n");
    }
    SPIRVDBG(spvdbgs() << "[add decorate] " << *Dec << '\n');
    return BaseType::insert(Dec);
  }
};

class SPIRVDecorate : public SPIRVDecorateGeneric {
public:
  static const Op OC = OpDecorate;
  static const SPIRVWord FixedWC = 3;
  // Complete constructor for decorations without literals
  SPIRVDecorate(Decoration TheDec, SPIRVEntry *TheTarget)
      : SPIRVDecorateGeneric(OC, 3, TheDec, TheTarget) {}
  // Complete constructor for decorations with one word literal
  SPIRVDecorate(Decoration TheDec, SPIRVEntry *TheTarget, SPIRVWord V)
      : SPIRVDecorateGeneric(OC, 4, TheDec, TheTarget, V) {}
  // Incomplete constructor
  SPIRVDecorate() : SPIRVDecorateGeneric(OC) {}

  _SPIRV_DCL_ENCDEC
  void setWordCount(SPIRVWord) override;
  void validate() const override {
    SPIRVDecorateGeneric::validate();
    assert(WordCount == Literals.size() + FixedWC);
  }
};

class SPIRVDecorateLinkageAttr : public SPIRVDecorate {
public:
  // Complete constructor for LinkageAttributes decorations
  SPIRVDecorateLinkageAttr(SPIRVEntry *TheTarget, const std::string &Name,
                           SPIRVLinkageTypeKind Kind)
      : SPIRVDecorate(DecorationLinkageAttributes, TheTarget) {
    for (auto &I : getVec(Name))
      Literals.push_back(I);
    Literals.push_back(Kind);
    WordCount += Literals.size();
  }
  // Incomplete constructor
  SPIRVDecorateLinkageAttr() : SPIRVDecorate() {}

  std::string getLinkageName() const {
    return getString(Literals.cbegin(), Literals.cend() - 1);
  }
  SPIRVLinkageTypeKind getLinkageType() const {
    return (SPIRVLinkageTypeKind)Literals.back();
  }

  static void encodeLiterals(SPIRVEncoder &Encoder,
                             const std::vector<SPIRVWord> &Literals) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
    if (SPIRVUseTextFormat) {
      Encoder << getString(Literals.cbegin(), Literals.cend() - 1);
      Encoder.OS << " ";
      Encoder << (SPIRVLinkageTypeKind)Literals.back();
    } else
#endif
      Encoder << Literals;
  }

  static void decodeLiterals(SPIRVDecoder &Decoder,
                             std::vector<SPIRVWord> &Literals) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
    if (SPIRVUseTextFormat) {
      std::string Name;
      Decoder >> Name;
      SPIRVLinkageTypeKind Kind;
      Decoder >> Kind;
      std::copy_n(getVec(Name).begin(), Literals.size() - 1, Literals.begin());
      Literals.back() = Kind;
    } else
#endif
      Decoder >> Literals;
  }
};

class SPIRVMemberDecorate : public SPIRVDecorateGeneric {
public:
  static const Op OC = OpMemberDecorate;
  static const SPIRVWord FixedWC = 4;
  // Complete constructor for decorations without literals
  SPIRVMemberDecorate(Decoration TheDec, SPIRVWord Member,
                      SPIRVEntry *TheTarget)
      : SPIRVDecorateGeneric(OC, 4, TheDec, TheTarget), MemberNumber(Member) {}

  // Complete constructor for decorations with one word literal
  SPIRVMemberDecorate(Decoration TheDec, SPIRVWord Member,
                      SPIRVEntry *TheTarget, SPIRVWord V)
      : SPIRVDecorateGeneric(OC, 5, TheDec, TheTarget, V),
        MemberNumber(Member) {}

  // Incomplete constructor
  SPIRVMemberDecorate()
      : SPIRVDecorateGeneric(OC), MemberNumber(SPIRVWORD_MAX) {}

  SPIRVWord getMemberNumber() const { return MemberNumber; }
  std::pair<SPIRVWord, Decoration> getPair() const {
    return std::make_pair(MemberNumber, Dec);
  }

  _SPIRV_DCL_ENCDEC
  void setWordCount(SPIRVWord) override;

  void validate() const override {
    SPIRVDecorateGeneric::validate();
    assert(WordCount == Literals.size() + FixedWC);
  }

protected:
  SPIRVWord MemberNumber;
};

class SPIRVDecorationGroup : public SPIRVEntry {
public:
  static const Op OC = OpDecorationGroup;
  static const SPIRVWord WC = 2;
  // Complete constructor. Does not populate Decorations.
  SPIRVDecorationGroup(SPIRVModule *TheModule, SPIRVId TheId)
      : SPIRVEntry(TheModule, WC, OC, TheId) {
    validate();
  };
  // Incomplete constructor
  SPIRVDecorationGroup() : SPIRVEntry(OC) {}
  void encodeAll(spv_ostream &O) const override;
  _SPIRV_DCL_ENCDEC
  // Move the given decorates to the decoration group
  void takeDecorates(SPIRVDecorateSet &Decs) {
    Decorations = std::move(Decs);
    for (auto &I : Decorations)
      const_cast<SPIRVDecorateGeneric *>(I)->setOwner(this);
    Decs.clear();
  }

  SPIRVDecorateSet &getDecorations() { return Decorations; }

protected:
  SPIRVDecorateSet Decorations;
  void validate() const override {
    assert(OpCode == OC);
    assert(WordCount == WC);
  }
};

class SPIRVGroupDecorateGeneric : public SPIRVEntryNoIdGeneric {
public:
  static const SPIRVWord FixedWC = 2;
  // Complete constructor
  SPIRVGroupDecorateGeneric(Op OC, SPIRVDecorationGroup *TheGroup,
                            const std::vector<SPIRVId> &TheTargets)
      : SPIRVEntryNoIdGeneric(TheGroup->getModule(),
                              FixedWC + TheTargets.size(), OC),
        DecorationGroup(TheGroup), Targets(TheTargets) {}
  // Incomplete constructor
  SPIRVGroupDecorateGeneric(Op OC)
      : SPIRVEntryNoIdGeneric(OC), DecorationGroup(nullptr) {}

  void setWordCount(SPIRVWord WC) override {
    SPIRVEntryNoIdGeneric::setWordCount(WC);
    Targets.resize(WC - FixedWC);
  }
  virtual void decorateTargets() = 0;
  _SPIRV_DCL_ENCDEC
protected:
  SPIRVDecorationGroup *DecorationGroup;
  std::vector<SPIRVId> Targets;
};

class SPIRVGroupDecorate : public SPIRVGroupDecorateGeneric {
public:
  static const Op OC = OpGroupDecorate;
  // Complete constructor
  SPIRVGroupDecorate(SPIRVDecorationGroup *TheGroup,
                     const std::vector<SPIRVId> &TheTargets)
      : SPIRVGroupDecorateGeneric(OC, TheGroup, TheTargets) {}
  // Incomplete constructor
  SPIRVGroupDecorate() : SPIRVGroupDecorateGeneric(OC) {}

  void decorateTargets() override;
};

class SPIRVGroupMemberDecorate : public SPIRVGroupDecorateGeneric {
public:
  static const Op OC = OpGroupMemberDecorate;
  // Complete constructor
  SPIRVGroupMemberDecorate(SPIRVDecorationGroup *TheGroup,
                           const std::vector<SPIRVId> &TheTargets)
      : SPIRVGroupDecorateGeneric(OC, TheGroup, TheTargets) {}
  // Incomplete constructor
  SPIRVGroupMemberDecorate() : SPIRVGroupDecorateGeneric(OC) {}

  void decorateTargets() override;
};

} // namespace SPIRV

#endif // SPIRV_LIBSPIRV_SPIRVDECORATE_H
