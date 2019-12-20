//===- GIMatchDagOperands.cpp - A shared operand list for nodes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GIMatchDagOperands.h"

#include "../CodeGenInstruction.h"

using namespace llvm;

void GIMatchDagOperand::Profile(FoldingSetNodeID &ID) const {
  Profile(ID, Idx, Name, IsDef);
}

void GIMatchDagOperand::Profile(FoldingSetNodeID &ID, size_t Idx,
                                       StringRef Name, bool IsDef) {
  ID.AddInteger(Idx);
  ID.AddString(Name);
  ID.AddBoolean(IsDef);
}

void GIMatchDagOperandList::add(StringRef Name, unsigned Idx, bool IsDef) {
  assert(Idx == Operands.size() && "Operands added in wrong order");
  Operands.emplace_back(Operands.size(), Name, IsDef);
  OperandsByName.try_emplace(Operands.back().getName(), Operands.size() - 1);
}

void GIMatchDagOperandList::Profile(FoldingSetNodeID &ID) const {
  for (const auto &I : enumerate(Operands))
    GIMatchDagOperand::Profile(ID, I.index(), I.value().getName(),
                               I.value().isDef());
}

void GIMatchDagOperandList::print(raw_ostream &OS) const {
  if (Operands.empty()) {
    OS << "<empty>";
    return;
  }
  StringRef Separator = "";
  for (const auto &I : Operands) {
    OS << Separator << I.getIdx() << ":" << I.getName();
    if (I.isDef())
      OS << "<def>";
    Separator = ", ";
  }
}

const GIMatchDagOperandList::value_type &GIMatchDagOperandList::
operator[](StringRef K) const {
  const auto &I = OperandsByName.find(K);
  assert(I != OperandsByName.end() && "Operand not found by name");
  return Operands[I->second];
}

const GIMatchDagOperandList &
GIMatchDagOperandListContext::makeEmptyOperandList() {
  FoldingSetNodeID ID;

  void *InsertPoint;
  GIMatchDagOperandList *Value =
      OperandLists.FindNodeOrInsertPos(ID, InsertPoint);
  if (Value)
    return *Value;

  std::unique_ptr<GIMatchDagOperandList> NewValue =
      std::make_unique<GIMatchDagOperandList>();
  OperandLists.InsertNode(NewValue.get(), InsertPoint);
  OperandListsOwner.push_back(std::move(NewValue));
  return *OperandListsOwner.back().get();
}

const GIMatchDagOperandList &
GIMatchDagOperandListContext::makeOperandList(const CodeGenInstruction &I) {
  FoldingSetNodeID ID;
  for (unsigned i = 0; i < I.Operands.size(); ++i)
    GIMatchDagOperand::Profile(ID, i, I.Operands[i].Name,
                               i < I.Operands.NumDefs);

  void *InsertPoint;
  GIMatchDagOperandList *Value =
      OperandLists.FindNodeOrInsertPos(ID, InsertPoint);
  if (Value)
    return *Value;

  std::unique_ptr<GIMatchDagOperandList> NewValue =
      std::make_unique<GIMatchDagOperandList>();
  for (unsigned i = 0; i < I.Operands.size(); ++i)
    NewValue->add(I.Operands[i].Name, i, i < I.Operands.NumDefs);
  OperandLists.InsertNode(NewValue.get(), InsertPoint);
  OperandListsOwner.push_back(std::move(NewValue));
  return *OperandListsOwner.back().get();
}

const GIMatchDagOperandList &
GIMatchDagOperandListContext::makeMIPredicateOperandList() {
  FoldingSetNodeID ID;
  GIMatchDagOperand::Profile(ID, 0, "$", true);
  GIMatchDagOperand::Profile(ID, 1, "mi", false);

  void *InsertPoint;
  GIMatchDagOperandList *Value =
      OperandLists.FindNodeOrInsertPos(ID, InsertPoint);
  if (Value)
    return *Value;

  std::unique_ptr<GIMatchDagOperandList> NewValue =
      std::make_unique<GIMatchDagOperandList>();
  NewValue->add("$", 0, true);
  NewValue->add("mi", 1, false);
  OperandLists.InsertNode(NewValue.get(), InsertPoint);
  OperandListsOwner.push_back(std::move(NewValue));
  return *OperandListsOwner.back().get();
}


const GIMatchDagOperandList &
GIMatchDagOperandListContext::makeTwoMOPredicateOperandList() {
  FoldingSetNodeID ID;
  GIMatchDagOperand::Profile(ID, 0, "$", true);
  GIMatchDagOperand::Profile(ID, 1, "mi0", false);
  GIMatchDagOperand::Profile(ID, 2, "mi1", false);

  void *InsertPoint;
  GIMatchDagOperandList *Value =
      OperandLists.FindNodeOrInsertPos(ID, InsertPoint);
  if (Value)
    return *Value;

  std::unique_ptr<GIMatchDagOperandList> NewValue =
      std::make_unique<GIMatchDagOperandList>();
  NewValue->add("$", 0, true);
  NewValue->add("mi0", 1, false);
  NewValue->add("mi1", 2, false);
  OperandLists.InsertNode(NewValue.get(), InsertPoint);
  OperandListsOwner.push_back(std::move(NewValue));
  return *OperandListsOwner.back().get();
}

void GIMatchDagOperandListContext::print(raw_ostream &OS) const {
  OS << "GIMatchDagOperandListContext {\n"
     << "  OperandLists {\n";
  for (const auto &I : OperandListsOwner) {
    OS << "    ";
    I->print(OS);
    OS << "\n";
  }
  OS << "  }\n"
     << "}\n";
}
