//===----- x86_64.cpp - Generic JITLink x86-64 edge kinds, utilities ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing x86-64 objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/x86_64.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {
namespace x86_64 {

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
  case Pointer64:
    return "Pointer64";
  case Pointer32:
    return "Pointer32";
  case Delta64:
    return "Delta64";
  case Delta32:
    return "Delta32";
  case NegDelta64:
    return "NegDelta64";
  case NegDelta32:
    return "NegDelta32";
  case Delta64FromGOT:
    return "Delta64FromGOT";
  case BranchPCRel32:
    return "BranchPCRel32";
  case BranchPCRel32ToPtrJumpStub:
    return "BranchPCRel32ToPtrJumpStub";
  case BranchPCRel32ToPtrJumpStubBypassable:
    return "BranchPCRel32ToPtrJumpStubBypassable";
  case RequestGOTAndTransformToDelta32:
    return "RequestGOTAndTransformToDelta32";
  case RequestGOTAndTransformToDelta64:
    return "RequestGOTAndTransformToDelta64";
  case RequestGOTAndTransformToDelta64FromGOT:
    return "RequestGOTAndTransformToDelta64FromGOT";
  case PCRel32GOTLoadREXRelaxable:
    return "PCRel32GOTLoadREXRelaxable";
  case RequestGOTAndTransformToPCRel32GOTLoadREXRelaxable:
    return "RequestGOTAndTransformToPCRel32GOTLoadREXRelaxable";
  case PCRel32GOTLoadRelaxable:
    return "PCRel32GOTLoadRelaxable";
  case RequestGOTAndTransformToPCRel32GOTLoadRelaxable:
    return "RequestGOTAndTransformToPCRel32GOTLoadRelaxable";
  case PCRel32TLVPLoadREXRelaxable:
    return "PCRel32TLVPLoadREXRelaxable";
  case RequestTLVPAndTransformToPCRel32TLVPLoadREXRelaxable:
    return "RequestTLVPAndTransformToPCRel32TLVPLoadREXRelaxable";
  default:
    return getGenericEdgeKindName(static_cast<Edge::Kind>(K));
  }
}

const char NullPointerContent[PointerSize] = {0x00, 0x00, 0x00, 0x00,
                                              0x00, 0x00, 0x00, 0x00};

const char PointerJumpStubContent[6] = {
    static_cast<char>(0xFFu), 0x25, 0x00, 0x00, 0x00, 0x00};

Error optimize_x86_64_GOTAndStubs(LinkGraph &G) {
  LLVM_DEBUG(dbgs() << "Optimizing GOT entries and stubs:\n");

  for (auto *B : G.blocks())
    for (auto &E : B->edges())
      if (E.getKind() == x86_64::PCRel32GOTLoadREXRelaxable) {
        // Replace GOT load with LEA only for MOVQ instructions.
        assert(E.getOffset() >= 3 && "GOT edge occurs too early in block");

        constexpr uint8_t MOVQRIPRel[] = {0x48, 0x8b};
        if (strncmp(B->getContent().data() + E.getOffset() - 3,
                    reinterpret_cast<const char *>(MOVQRIPRel), 2) != 0)
          continue;

        auto &GOTBlock = E.getTarget().getBlock();
        assert(GOTBlock.getSize() == G.getPointerSize() &&
               "GOT entry block should be pointer sized");
        assert(GOTBlock.edges_size() == 1 &&
               "GOT entry should only have one outgoing edge");

        auto &GOTTarget = GOTBlock.edges().begin()->getTarget();
        JITTargetAddress EdgeAddr = B->getAddress() + E.getOffset();
        JITTargetAddress TargetAddr = GOTTarget.getAddress();

        int64_t Displacement = TargetAddr - EdgeAddr + 4;
        if (isInRangeForImmS32(Displacement)) {
          // Change the edge kind as we don't go through GOT anymore. This is
          // for formal correctness only. Technically, the two relocation kinds
          // are resolved the same way.
          E.setKind(x86_64::Delta32);
          E.setTarget(GOTTarget);
          E.setAddend(E.getAddend() - 4);
          auto *BlockData = reinterpret_cast<uint8_t *>(
              const_cast<char *>(B->getContent().data()));
          BlockData[E.getOffset() - 2] = 0x8d;
          LLVM_DEBUG({
            dbgs() << "  Replaced GOT load wih LEA:\n    ";
            printEdge(dbgs(), *B, E, getEdgeKindName(E.getKind()));
            dbgs() << "\n";
          });
        }
      } else if (E.getKind() == x86_64::BranchPCRel32ToPtrJumpStubBypassable) {
        auto &StubBlock = E.getTarget().getBlock();
        assert(StubBlock.getSize() == sizeof(PointerJumpStubContent) &&
               "Stub block should be stub sized");
        assert(StubBlock.edges_size() == 1 &&
               "Stub block should only have one outgoing edge");

        auto &GOTBlock = StubBlock.edges().begin()->getTarget().getBlock();
        assert(GOTBlock.getSize() == G.getPointerSize() &&
               "GOT block should be pointer sized");
        assert(GOTBlock.edges_size() == 1 &&
               "GOT block should only have one outgoing edge");

        auto &GOTTarget = GOTBlock.edges().begin()->getTarget();
        JITTargetAddress EdgeAddr = B->getAddress() + E.getOffset();
        JITTargetAddress TargetAddr = GOTTarget.getAddress();

        int64_t Displacement = TargetAddr - EdgeAddr + 4;
        if (isInRangeForImmS32(Displacement)) {
          E.setKind(x86_64::BranchPCRel32);
          E.setTarget(GOTTarget);
          LLVM_DEBUG({
            dbgs() << "  Replaced stub branch with direct branch:\n    ";
            printEdge(dbgs(), *B, E, getEdgeKindName(E.getKind()));
            dbgs() << "\n";
          });
        }
      }

  return Error::success();
}

} // end namespace x86_64
} // end namespace jitlink
} // end namespace llvm
