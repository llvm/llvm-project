//===---- lib/CodeGen/GlobalISel/MachineLegalizer.cpp - IRTranslator -------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement an interface to specify and query how an illegal operation on a
// given type should be expanded.
//
// Issues to be resolved:
//   + Make it fast.
//   + Support weird types like i3, <7 x i3>, ...
//   + Operations with more than one type (ICMP, CMPXCHG, intrinsics, ...)
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/GlobalISel/MachineLegalizer.h"
#include "llvm/IR/Type.h"
#include "llvm/Target/TargetOpcodes.h"
using namespace llvm;

MachineLegalizer::MachineLegalizer() : TablesInitialized(false) {
  // FIXME: these two can be legalized to the fundamental load/store Jakob
  // proposed. Once loads & stores are supported.
  DefaultActions[TargetOpcode::G_ANYEXTEND] = Legal;
  DefaultActions[TargetOpcode::G_TRUNC] = Legal;

  DefaultActions[TargetOpcode::G_ADD] = NarrowScalar;
}

void MachineLegalizer::computeTables() {
  for (auto &Op : Actions) {
    LLT Ty = Op.first.second;
    if (!Ty.isVector())
      continue;

    auto &Entry =
        MaxLegalVectorElts[std::make_pair(Op.first.first, Ty.getElementType())];
    Entry = std::max(Entry, Ty.getNumElements());
  }

  TablesInitialized = true;
}

// FIXME: inefficient implementation for now. Without ComputeValueVTs we're
// probably going to need specialized lookup structures for various types before
// we have any hope of doing well with something like <13 x i3>. Even the common
// cases should do better than what we have now.
std::pair<MachineLegalizer::LegalizeAction, LLT>
MachineLegalizer::getAction(unsigned Opcode, LLT Ty) const {
  assert(TablesInitialized && "backend forgot to call computeTables");
  // These *have* to be implemented for now, they're the fundamental basis of
  // how everything else is transformed.

  // FIXME: the long-term plan calls for expansion in terms of load/store (if
  // they're not legal).
  if (Opcode == TargetOpcode::G_SEQUENCE || Opcode == TargetOpcode::G_EXTRACT)
    return std::make_pair(Legal, Ty);

  auto ActionIt = Actions.find(std::make_pair(Opcode, Ty));
  if (ActionIt != Actions.end())
    return findLegalAction(Opcode, Ty, ActionIt->second);

  if (!Ty.isVector()) {
    auto DefaultAction = DefaultActions.find(Opcode);
    if (DefaultAction != DefaultActions.end() && DefaultAction->second == Legal)
      return std::make_pair(Legal, Ty);

    assert(DefaultAction->second == NarrowScalar && "unexpected default");
    return findLegalAction(Opcode, Ty, NarrowScalar);
  }

  LLT EltTy = Ty.getElementType();
  int NumElts = Ty.getNumElements();

  auto ScalarAction = ScalarInVectorActions.find(std::make_pair(Opcode, EltTy));
  if (ScalarAction != ScalarInVectorActions.end() &&
      ScalarAction->second != Legal)
    return findLegalAction(Opcode, EltTy, ScalarAction->second);

  // The element type is legal in principle, but the number of elements is
  // wrong.
  auto MaxLegalElts = MaxLegalVectorElts.lookup(std::make_pair(Opcode, EltTy));
  if (MaxLegalElts > NumElts)
    return findLegalAction(Opcode, Ty, MoreElements);

  if (MaxLegalElts == 0) {
    // Scalarize if there's no legal vector type, which is just a special case
    // of FewerElements.
    return std::make_pair(FewerElements, EltTy);
  }

  return findLegalAction(Opcode, Ty, FewerElements);
}

std::pair<MachineLegalizer::LegalizeAction, LLT>
MachineLegalizer::getAction(const MachineInstr &MI) const {
  return getAction(MI.getOpcode(), MI.getType());
}

bool MachineLegalizer::isLegal(const MachineInstr &MI) const {
  return getAction(MI).first == Legal;
}

LLT MachineLegalizer::findLegalType(unsigned Opcode, LLT Ty,
                                    LegalizeAction Action) const {
  switch(Action) {
  default:
    llvm_unreachable("Cannot find legal type");
  case Legal:
    return Ty;
  case NarrowScalar: {
    return findLegalType(Opcode, Ty,
                         [&](LLT Ty) -> LLT { return Ty.halfScalarSize(); });
  }
  case WidenScalar: {
    return findLegalType(Opcode, Ty, [&](LLT Ty) -> LLT {
      return Ty.getSizeInBits() < 8 ? LLT::scalar(8) : Ty.doubleScalarSize();
    });
  }
  case FewerElements: {
    return findLegalType(Opcode, Ty,
                         [&](LLT Ty) -> LLT { return Ty.halfElements(); });
  }
  case MoreElements: {
    return findLegalType(
        Opcode, Ty, [&](LLT Ty) -> LLT { return Ty.doubleElements(); });
  }
  }
}
