//===- MemoryModelRelaxationAnnotations.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

// FIXME: Only needed for canInstructionHaveMMRAs, should it move to another
// file?
#include "llvm/IR/Instructions.h"

using namespace llvm;

static bool isTagMD(const MDNode *MD) {
  return isa<MDTuple>(MD) && MD->getNumOperands() == 2 &&
         isa<MDString>(MD->getOperand(0)) && isa<MDString>(MD->getOperand(1));
}

MMRAMetadata::MMRAMetadata(const Instruction &I)
    : MMRAMetadata(I.getMetadata(LLVMContext::MD_MMRA)) {}

MMRAMetadata::MMRAMetadata(MDNode *MD) {
  if (!MD)
    return;

  // TODO: Split this into a "tryParse" function that can return an err.
  // CTor can use the tryParse & just fatal on err.

  MDTuple *Tuple = dyn_cast<MDTuple>(MD);
  if (!Tuple)
    report_fatal_error("MMRAs should always be MDTuples!");

  const auto HandleTagMD = [this](MDNode *TagMD) {
    addTag(cast<MDString>(TagMD->getOperand(0))->getString(),
           cast<MDString>(TagMD->getOperand(1))->getString());
  };

  if (isTagMD(Tuple)) {
    HandleTagMD(Tuple);
    return;
  }

  for (const MDOperand &Op : Tuple->operands()) {
    MDNode *MDOp = cast<MDNode>(Op.get());
    if (!isTagMD(MDOp)) {
      errs() << "MD Node:\n";
      MD->print(errs());
      errs() << "Operand:\n";
      Op->print(errs());
      report_fatal_error("Invalid MMRA Metadata Structure!");
    }

    HandleTagMD(MDOp);
  }
}

bool MMRAMetadata::isCompatibleWith(const MMRAMetadata &Other) const {
  // Two sets of tags are compatible iff, for every unique tag prefix P
  // present in at least one set:
  //   - the other set contains no tag that with the same prefix, or
  //   - at least one tag with the same prefix is present in both sets.

  // These sets are generally small so we don't bother uniquing
  // the prefixes beforehand. Checking a prefix twice is likely cheaper
  // than building a map.
  for (const auto &[P, S] : Tags) {
    if (!Other.hasTag(P, S) && Other.hasTagWithPrefix(P))
      return false;
  }

  for (const auto &[P, S] : Other) {
    if (!hasTag(P, S) && hasTagWithPrefix(P))
      return false;
  }

  return true;
}

MMRAMetadata MMRAMetadata::combine(const MMRAMetadata &Other) const {
  // Let A and B be two tags set, and U be the prefix-wise union of A and B.
  // For every unique tag prefix P present in A or B:
  // * If either A or B has no tags with prefix P, no tags with prefix
  //   P are added to U.
  // * If both A and B have at least one tag with prefix P, only the tags
  //   common to A and B are added to U.

  StringSet<> Prefixes;
  for (const auto &[P, S] : Tags)
    Prefixes.insert(P);
  for (const auto &[P, S] : Other)
    Prefixes.insert(P);

  MMRAMetadata U;
  for (StringRef P : Prefixes.keys()) {
    auto A = getAllTagsWithPrefix(P);
    auto B = Other.getAllTagsWithPrefix(P);

    if (A.empty() || B.empty())
      continue;

    for (const auto &Tag : A) {
      if (hasTag(Tag) && Other.hasTag(Tag))
        U.addTag(Tag);
    }
    for (const auto &Tag : B) {
      if (hasTag(Tag) && Other.hasTag(Tag))
        U.addTag(Tag);
    }
  }

  return U;
}

MMRAMetadata &MMRAMetadata::addTag(StringRef Prefix, StringRef Suffix) {
  Tags.insert(std::make_pair(Prefix.str(), Suffix.str()));
  return *this;
}

bool MMRAMetadata::hasTag(StringRef Prefix, StringRef Suffix) const {
  return Tags.count(std::make_pair(Prefix.str(), Suffix.str()));
}

std::vector<MMRAMetadata::TagT>
MMRAMetadata::getAllTagsWithPrefix(StringRef Prefix) const {
  std::vector<TagT> Result;
  for (const auto &T : Tags) {
    if (T.first == Prefix)
      Result.push_back(T);
  }
  return Result;
}

bool MMRAMetadata::hasTagWithPrefix(StringRef Prefix) const {
  for (const auto &[P, S] : Tags)
    if (P == Prefix)
      return true;
  return false;
}

MMRAMetadata::const_iterator MMRAMetadata::begin() const {
  return Tags.begin();
}

MMRAMetadata::const_iterator MMRAMetadata::end() const { return Tags.end(); }

bool MMRAMetadata::empty() const { return Tags.empty(); }

unsigned MMRAMetadata::size() const { return Tags.size(); }

static MDTuple *getMDForPair(LLVMContext &Ctx, StringRef P, StringRef S) {
  return MDTuple::get(Ctx, {MDString::get(Ctx, P), MDString::get(Ctx, S)});
}

MDTuple *MMRAMetadata::getAsMD(LLVMContext &Ctx) const {
  if (empty())
    return MDTuple::get(Ctx, {});

  std::vector<Metadata *> TagMDs;
  TagMDs.reserve(Tags.size());

  for (const auto &[P, S] : Tags)
    TagMDs.push_back(getMDForPair(Ctx, P, S));

  if (TagMDs.size() == 1)
    return cast<MDTuple>(TagMDs.front());
  return MDTuple::get(Ctx, TagMDs);
}

void MMRAMetadata::print(raw_ostream &OS) const {
  bool IsFirst = true;
  // TODO: use map_iter + join
  for (const auto &[P, S] : Tags) {
    if (IsFirst)
      IsFirst = false;
    else
      OS << ", ";
    OS << P << ":" << S;
  }
}

LLVM_DUMP_METHOD
void MMRAMetadata::dump() const { print(dbgs()); }

static bool isReadWriteMemCall(const Instruction &I) {
  if (const auto *C = dyn_cast<CallBase>(&I))
    return C->mayReadOrWriteMemory() ||
           !C->getMemoryEffects().doesNotAccessMemory();
  return false;
}

bool llvm::canInstructionHaveMMRAs(const Instruction &I) {
  return isa<LoadInst>(I) || isa<StoreInst>(I) || isa<AtomicCmpXchgInst>(I) ||
         isa<AtomicRMWInst>(I) || isa<FenceInst>(I) || isReadWriteMemCall(I);
}
