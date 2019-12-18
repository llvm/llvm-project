//===- GIMatchDagInstr.h - Represent a instruction to be matched ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAGINSTR_H
#define LLVM_UTILS_TABLEGEN_GIMATCHDAGINSTR_H

#include "GIMatchDagOperands.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class GIMatchDag;

/// Represents an instruction in the match DAG. This object knows very little
/// about the actual instruction to be matched as the bulk of that is in
/// predicates that are associated with the match DAG. It merely knows the names
/// and indices of any operands that need to be matched in order to allow edges
/// to link to them.
///
/// Instances of this class objects are owned by the GIMatchDag and are not
/// shareable between instances of GIMatchDag. This is because the Name,
/// IsMatchRoot, and OpcodeAnnotation are likely to differ between GIMatchDag
/// instances.
class GIMatchDagInstr {
public:
  using const_user_assigned_operand_names_iterator =
      DenseMap<unsigned, StringRef>::const_iterator;

protected:
  /// The match DAG this instruction belongs to.
  GIMatchDag &Dag;

  /// The name of the instruction in the pattern. For example:
  ///     (FOO $a, $b, $c):$name
  /// will cause name to be assigned to this member. Anonymous instructions will
  /// have a name assigned for debugging purposes.
  StringRef Name;

  /// The name of the instruction in the pattern as assigned by the user. For
  /// example:
  ///     (FOO $a, $b, $c):$name
  /// will cause name to be assigned to this member. If a name is not provided,
  /// this will be empty. This name is used to bind variables from rules to the
  /// matched instruction.
  StringRef UserAssignedName;

  /// The name of each operand (if any) that was assigned by the user. For
  /// example:
  ///     (FOO $a, $b, $c):$name
  /// will cause {0, "a"}, {1, "b"}, {2, "c} to be inserted into this map.
  DenseMap<unsigned, StringRef> UserAssignedNamesForOperands;

  /// The operand list for this instruction. This object may be shared with
  /// other instructions of a similar 'shape'.
  const GIMatchDagOperandList &OperandInfo;

  /// For debugging purposes, it's helpful to have access to a description of
  /// the Opcode. However, this object shouldn't use it for more than debugging
  /// output since predicates are expected to be handled outside the DAG.
  CodeGenInstruction *OpcodeAnnotation = 0;

  /// When true, this instruction will be a starting point for a match attempt.
  bool IsMatchRoot = false;

public:
  GIMatchDagInstr(GIMatchDag &Dag, StringRef Name, StringRef UserAssignedName,
                  const GIMatchDagOperandList &OperandInfo)
      : Dag(Dag), Name(Name), UserAssignedName(UserAssignedName),
        OperandInfo(OperandInfo) {}

  const GIMatchDagOperandList &getOperandInfo() const { return OperandInfo; }
  StringRef getName() const { return Name; }
  void assignNameToOperand(unsigned Idx, StringRef Name) {
    assert(UserAssignedNamesForOperands[Idx].empty() && "Cannot assign twice");
    UserAssignedNamesForOperands[Idx] = Name;
  }

  const_user_assigned_operand_names_iterator
  user_assigned_operand_names_begin() const {
    return UserAssignedNamesForOperands.begin();
  }
  const_user_assigned_operand_names_iterator
  user_assigned_operand_names_end() const {
    return UserAssignedNamesForOperands.end();
  }
  iterator_range<const_user_assigned_operand_names_iterator>
  user_assigned_operand_names() const {
    return make_range(user_assigned_operand_names_begin(),
                      user_assigned_operand_names_end());
  }

  /// Mark this instruction as being a root of the match. This means that the
  /// matcher will start from this node when attempting to match MIR.
  void setMatchRoot();
  bool isMatchRoot() const { return IsMatchRoot; }

  void setOpcodeAnnotation(CodeGenInstruction *I) { OpcodeAnnotation = I; }
  CodeGenInstruction *getOpcodeAnnotation() const { return OpcodeAnnotation; }

  void print(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(errs()); }
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
};

raw_ostream &operator<<(raw_ostream &OS, const GIMatchDagInstr &N);

} // end namespace llvm
#endif // ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAGINSTR_H
