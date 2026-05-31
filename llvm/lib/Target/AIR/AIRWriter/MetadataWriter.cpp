//===- MetadataWriter.cpp - AIR metadata-block writer -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetadataWriter.h"
#include "BitcodeEncoding.h"
#include "ValueEnumerator.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace llvm {
namespace metal {

void MetadataEnumerator::collect(Module &M, ValueEnumerator &E) {
  for (auto &NMD : M.named_metadata())
    for (unsigned I = 0; I < NMD.getNumOperands(); I++)
      visitNode(NMD.getOperand(I), E);
  // E2a: pick up MDNodes attached to instructions (alias.scope, noalias,
  // tbaa, etc.). Apple's writer threads these through ValueEnumerator;
  // ours did not. Without this pass METADATA_BLOCK has no IDs for the
  // FUNC_CODE_INST_ATTACHMENT records to reference.
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    for (auto &BB : F) {
      for (auto &I : BB) {
        SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
        I.getAllMetadataOtherThanDebugLoc(MDs);
        for (auto &P : MDs)
          visitNode(P.second, E);
      }
    }
  }
}

void MetadataEnumerator::visitNode(const MDNode *N, ValueEnumerator &E) {
  if (!N || nodeMap.count(N))
    return;
  // For distinct, self-referential nodes (alias-scope domains:
  // `!d = distinct !{!d, ...}`) we MUST reserve the slot before recursing,
  // otherwise we infinite-loop. Non-distinct nodes can keep the original
  // post-order semantics so named-metadata references don't shift.
  if (N->isDistinct()) {
    unsigned MyID = nodes.size();
    nodeMap[N] = MyID;
    nodes.push_back(N);
  }
  for (unsigned I = 0; I < N->getNumOperands(); I++) {
    auto *Op = N->getOperand(I).get();
    if (!Op)
      continue;
    if (auto *S = dyn_cast<MDString>(Op)) {
      if (!stringMap.count(S)) {
        stringMap[S] = strings.size();
        strings.push_back(S->getString());
      }
    } else if (auto *VAM = dyn_cast<ValueAsMetadata>(Op)) {
      if (!valueMap.count(VAM)) {
        ValEntry VE;
        VE.typeIdx = E.typeIdx(VAM->getValue()->getType());
        if (E.globalValueMap.count(VAM->getValue()))
          VE.valueID = E.globalIdx(VAM->getValue());
        else if (auto *C = dyn_cast<Constant>(VAM->getValue()))
          VE.valueID = E.moduleConstIdx(C);
        else
          VE.valueID = 0;
        valueMap[VAM] = values.size();
        values.push_back(VE);
      }
    } else if (auto *Sub = dyn_cast<MDNode>(Op)) {
      visitNode(Sub, E);
    }
  }
  if (!N->isDistinct()) {
    nodeMap[N] = nodes.size();
    nodes.push_back(N);
  }
}

uint64_t MetadataEnumerator::operandID(const Metadata *Op) const {
  if (!Op)
    return 0;
  if (auto *S = dyn_cast<MDString>(Op))
    return stringMap.lookup(S) + 1;
  if (auto *VAM = dyn_cast<ValueAsMetadata>(Op))
    return strings.size() + valueMap.lookup(VAM) + 1;
  if (auto *N = dyn_cast<MDNode>(Op))
    return strings.size() + values.size() + nodeMap.lookup(N) + 1;
  return 0;
}

void emitMetadataKindBlock(BitstreamWriter &W) {
  W.EnterSubblock(bitc::METADATA_KIND_BLOCK_ID, 3);
  static const char *Kinds[] = {
      "dbg",
      "tbaa",
      "prof",
      "fpmath",
      "range",
      "tbaa.struct",
      "invariant.load",
      "alias.scope",
      "noalias",
      "nontemporal",
      "llvm.mem.parallel_loop_access",
      "nonnull",
      "dereferenceable",
      "dereferenceable_or_null",
      "make.implicit",
      "unpredictable",
      "invariant.group",
      "align",
      "llvm.loop",
      "type",
      "section_prefix",
      "absolute_symbol",
      "associated",
      "callees",
      "irr_loop",
      "llvm.access.group",
      "callback",
      "llvm.preserve.access.index",
      "vcall_visibility",
      "noundef",
      "annotation",
      "heapallocsite",
      "air.function_groups",
  };
  for (unsigned I = 0; I < sizeof(Kinds) / sizeof(Kinds[0]); I++) {
    SmallVector<uint64_t, 32> V;
    V.push_back(I);
    for (const char *P = Kinds[I]; *P; P++)
      V.push_back((unsigned char)*P);
    W.EmitRecord(bitc::METADATA_KIND, V);
  }
  W.ExitBlock();
}

void emitMetadataBlock(BitstreamWriter &W, Module &M, ValueEnumerator &E,
                       MetadataEnumerator &MD) {
  if (MD.strings.empty() && MD.values.empty() && MD.nodes.empty())
    return;

  W.EnterSubblock(bitc::METADATA_BLOCK_ID, 4);

  for (auto &S : MD.strings)
    emitString(W, bitc::METADATA_STRING_OLD, S);

  for (auto &V : MD.values) {
    SmallVector<uint64_t, 2> Ops = {V.typeIdx, V.valueID};
    W.EmitRecord(bitc::METADATA_VALUE, Ops);
  }

  for (auto *N : MD.nodes) {
    SmallVector<uint64_t, 8> Ops;
    for (unsigned I = 0; I < N->getNumOperands(); I++)
      Ops.push_back(MD.operandID(N->getOperand(I).get()));
    W.EmitRecord(N->isDistinct() ? bitc::METADATA_DISTINCT_NODE
                                 : bitc::METADATA_NODE,
                 Ops);
  }

  for (auto &NMD : M.named_metadata()) {
    emitString(W, bitc::METADATA_NAME, NMD.getName());
    SmallVector<uint64_t, 4> Ops;
    for (unsigned I = 0; I < NMD.getNumOperands(); I++)
      Ops.push_back(MD.strings.size() + MD.values.size() +
                    MD.nodeMap.lookup(NMD.getOperand(I)));
    W.EmitRecord(bitc::METADATA_NAMED_NODE, Ops);
  }

  W.ExitBlock();
}

void emitOperandBundleTagsBlock(BitstreamWriter &W) {
  W.EnterSubblock(bitc::OPERAND_BUNDLE_TAGS_BLOCK_ID, 3);
  static const char *Tags[] = {
      "deopt",        "funclet", "gc-transition",          "cfguardtarget",
      "preallocated", "gc-live", "clang.arc.attachedcall", "ptrauth",
  };
  for (auto *T : Tags)
    emitString(W, bitc::OPERAND_BUNDLE_TAG, T);
  W.ExitBlock();
}

void emitSinglethreadBlock(BitstreamWriter &W) {
  W.EnterSubblock(26, 2);
  emitString(W, 1, "singlethread");
  W.ExitBlock();
}

} // namespace metal
} // namespace llvm
