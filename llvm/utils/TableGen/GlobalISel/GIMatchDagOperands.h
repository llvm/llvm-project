//===- GIMatchDagOperands.h - Represent a shared operand list for nodes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAGOPERANDS_H
#define LLVM_UTILS_TABLEGEN_GIMATCHDAGOPERANDS_H

#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

namespace llvm {
class CodeGenInstruction;
/// Describes an operand of a MachineInstr w.r.t the DAG Matching. This
/// information is derived from CodeGenInstruction::Operands but is more
/// readily available for context-less access as we don't need to know which
/// instruction it's used with or know how many defs that instruction had.
///
/// There may be multiple GIMatchDagOperand's with the same contents. However,
/// they are uniqued within the set of instructions that have the same overall
/// operand list. For example, given:
///     Inst1 operands ($dst:<def>, $src1, $src2)
///     Inst2 operands ($dst:<def>, $src1, $src2)
///     Inst3 operands ($dst:<def>, $src)
/// $src1 will have a single instance of GIMatchDagOperand shared by Inst1 and
/// Inst2, as will $src2. $dst however, will have two instances one shared
/// between Inst1 and Inst2 and one unique to Inst3. We could potentially
/// fully de-dupe the GIMatchDagOperand instances but the saving is not expected
/// to be worth the overhead.
///
/// The result of this is that the address of the object can be relied upon to
/// trivially identify commonality between two instructions which will be useful
/// when generating the matcher. When the pointers differ, the contents can be
/// inspected instead.
class GIMatchDagOperand {
  unsigned Idx;
  StringRef Name;
  bool IsDef;

public:
  GIMatchDagOperand(unsigned Idx, StringRef Name, bool IsDef)
      : Idx(Idx), Name(Name), IsDef(IsDef) {}

  unsigned getIdx() const { return Idx; }
  StringRef getName() const { return Name; }
  bool isDef() const { return IsDef; }

  /// This object isn't a FoldingSetNode but it's part of one. See FoldingSet
  /// for details on the Profile function.
  void Profile(FoldingSetNodeID &ID) const;

  /// A helper that behaves like Profile() but is also usable without the object.
  /// We use size_t here to match enumerate<...>::index(). If we don't match
  /// that the hashes won't be equal.
  static void Profile(FoldingSetNodeID &ID, size_t Idx, StringRef Name,
                      bool IsDef);
};

/// A list of GIMatchDagOperands for an instruction without any association with
/// a particular instruction.
///
/// An important detail to be aware of with this class is that they are shared
/// with other instructions of a similar 'shape'. For example, all the binary
/// instructions are likely to share a single GIMatchDagOperandList. This is
/// primarily a memory optimization as it's fairly common to have a large number
/// of instructions but only a few 'shapes'.
///
/// See GIMatchDagOperandList::Profile() for the details on how they are folded.
class GIMatchDagOperandList : public FoldingSetNode {
public:
  using value_type = GIMatchDagOperand;

protected:
  using vector_type = SmallVector<GIMatchDagOperand, 3>;

public:
  using iterator = vector_type::iterator;
  using const_iterator = vector_type::const_iterator;

protected:
  vector_type Operands;
  StringMap<unsigned> OperandsByName;

public:
  void add(StringRef Name, unsigned Idx, bool IsDef);

  /// See FoldingSet for details.
  void Profile(FoldingSetNodeID &ID) const;

  iterator begin() { return Operands.begin(); }
  const_iterator begin() const { return Operands.begin(); }
  iterator end() { return Operands.end(); }
  const_iterator end() const { return Operands.end(); }

  const value_type &operator[](unsigned I) const { return Operands[I]; }
  const value_type &operator[](StringRef K) const;

  void print(raw_ostream &OS) const;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(errs()); }
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
};

/// This is the portion of GIMatchDagContext that directly relates to
/// GIMatchDagOperandList and GIMatchDagOperandList.
class GIMatchDagOperandListContext {
  FoldingSet<GIMatchDagOperandList> OperandLists;
  std::vector<std::unique_ptr<GIMatchDagOperandList>> OperandListsOwner;

public:
  const GIMatchDagOperandList &makeEmptyOperandList();
  const GIMatchDagOperandList &makeOperandList(const CodeGenInstruction &I);
  const GIMatchDagOperandList &makeMIPredicateOperandList();
  const GIMatchDagOperandList &makeTwoMOPredicateOperandList();

  void print(raw_ostream &OS) const;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(errs()); }
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
};

} // end namespace llvm
#endif // ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAGOPERANDS_H
