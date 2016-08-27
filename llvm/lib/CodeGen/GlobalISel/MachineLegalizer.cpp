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
  DefaultActions[TargetOpcode::G_ANYEXT] = Legal;
  DefaultActions[TargetOpcode::G_TRUNC] = Legal;

  DefaultActions[TargetOpcode::G_INTRINSIC] = Legal;
  DefaultActions[TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS] = Legal;

  DefaultActions[TargetOpcode::G_ADD] = NarrowScalar;

  DefaultActions[TargetOpcode::G_BRCOND] = WidenScalar;
}

void MachineLegalizer::computeTables() {
  for (auto &Op : Actions) {
    LLT Ty = Op.first.Type;
    if (!Ty.isVector())
      continue;

    auto &Entry = MaxLegalVectorElts[std::make_pair(Op.first.Opcode,
                                                    Ty.getElementType())];
    Entry = std::max(Entry, Ty.getNumElements());
  }

  TablesInitialized = true;
}

// FIXME: inefficient implementation for now. Without ComputeValueVTs we're
// probably going to need specialized lookup structures for various types before
// we have any hope of doing well with something like <13 x i3>. Even the common
// cases should do better than what we have now.
std::pair<MachineLegalizer::LegalizeAction, LLT>
MachineLegalizer::getAction(const InstrAspect &Aspect) const {
  assert(TablesInitialized && "backend forgot to call computeTables");
  // These *have* to be implemented for now, they're the fundamental basis of
  // how everything else is transformed.

  // FIXME: the long-term plan calls for expansion in terms of load/store (if
  // they're not legal).
  if (Aspect.Opcode == TargetOpcode::G_SEQUENCE ||
      Aspect.Opcode == TargetOpcode::G_EXTRACT)
    return std::make_pair(Legal, Aspect.Type);

  auto ActionIt = Actions.find(Aspect);
  if (ActionIt != Actions.end())
    return findLegalAction(Aspect, ActionIt->second);

  unsigned Opcode = Aspect.Opcode;
  LLT Ty = Aspect.Type;
  if (!Ty.isVector()) {
    auto DefaultAction = DefaultActions.find(Aspect.Opcode);
    if (DefaultAction != DefaultActions.end() && DefaultAction->second == Legal)
      return std::make_pair(Legal, Ty);

    assert(DefaultAction->second == NarrowScalar && "unexpected default");
    return findLegalAction(Aspect, NarrowScalar);
  }

  LLT EltTy = Ty.getElementType();
  int NumElts = Ty.getNumElements();

  auto ScalarAction = ScalarInVectorActions.find(std::make_pair(Opcode, EltTy));
  if (ScalarAction != ScalarInVectorActions.end() &&
      ScalarAction->second != Legal)
    return findLegalAction(Aspect, ScalarAction->second);

  // The element type is legal in principle, but the number of elements is
  // wrong.
  auto MaxLegalElts = MaxLegalVectorElts.lookup(std::make_pair(Opcode, EltTy));
  if (MaxLegalElts > NumElts)
    return findLegalAction(Aspect, MoreElements);

  if (MaxLegalElts == 0) {
    // Scalarize if there's no legal vector type, which is just a special case
    // of FewerElements.
    return std::make_pair(FewerElements, EltTy);
  }

  return findLegalAction(Aspect, FewerElements);
}

std::tuple<MachineLegalizer::LegalizeAction, unsigned, LLT>
MachineLegalizer::getAction(const MachineInstr &MI) const {
  for (unsigned i = 0; i < MI.getNumTypes(); ++i) {
    auto Action = getAction({MI.getOpcode(), i, MI.getType(i)});
    if (Action.first != Legal)
      return std::make_tuple(Action.first, i, Action.second);
  }
  return std::make_tuple(Legal, 0, LLT{});
}

bool MachineLegalizer::isLegal(const MachineInstr &MI) const {
  return std::get<0>(getAction(MI)) == Legal;
}

LLT MachineLegalizer::findLegalType(const InstrAspect &Aspect,
                                    LegalizeAction Action) const {
  switch(Action) {
  default:
    llvm_unreachable("Cannot find legal type");
  case Legal:
  case Lower:
    return Aspect.Type;
  case NarrowScalar: {
    return findLegalType(Aspect,
                         [&](LLT Ty) -> LLT { return Ty.halfScalarSize(); });
  }
  case WidenScalar: {
    return findLegalType(Aspect, [&](LLT Ty) -> LLT {
      return Ty.getSizeInBits() < 8 ? LLT::scalar(8) : Ty.doubleScalarSize();
    });
  }
  case FewerElements: {
    return findLegalType(Aspect,
                         [&](LLT Ty) -> LLT { return Ty.halfElements(); });
  }
  case MoreElements: {
    return findLegalType(Aspect,
                         [&](LLT Ty) -> LLT { return Ty.doubleElements(); });
  }
  }
}
