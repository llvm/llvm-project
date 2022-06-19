//===- GIMatchDagPredicate - Represent a predicate to check ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAGPREDICATE_H
#define LLVM_UTILS_TABLEGEN_GIMATCHDAGPREDICATE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
#include "llvm/Support/raw_ostream.h"
#endif

namespace llvm {
class CodeExpansions;
class CodeGenInstruction;
class GIMatchDagOperandList;
class GIMatchDagContext;
class raw_ostream;

/// Represents a predicate on the match DAG. This records the details of the
/// predicate. The dependencies are stored in the GIMatchDag as edges.
///
/// Instances of this class objects are owned by the GIMatchDag and are not
/// shareable between instances of GIMatchDag.
class GIMatchDagPredicate {
public:
  enum GIMatchDagPredicateKind {
    GIMatchDagPredicateKind_Opcode,
    GIMatchDagPredicateKind_OneOfOpcodes,
    GIMatchDagPredicateKind_SameMO,
  };

protected:
  const GIMatchDagPredicateKind Kind;

  /// The name of the predicate. For example:
  ///     (FOO $a:s32, $b, $c)
  /// will cause 's32' to be assigned to this member for the $a predicate.
  /// Similarly, the opcode predicate will cause 'FOO' to be assigned to this
  /// member. Anonymous instructions will have a name assigned for debugging
  /// purposes.
  StringRef Name;

  /// The operand list for this predicate. This object may be shared with
  /// other predicates of a similar 'shape'.
  const GIMatchDagOperandList &OperandInfo;

public:
  GIMatchDagPredicate(GIMatchDagPredicateKind Kind, StringRef Name,
                      const GIMatchDagOperandList &OperandInfo)
      : Kind(Kind), Name(Name), OperandInfo(OperandInfo) {}
  virtual ~GIMatchDagPredicate() {}

  GIMatchDagPredicateKind getKind() const { return Kind; }

  StringRef getName() const { return Name; }
  const GIMatchDagOperandList &getOperandInfo() const { return OperandInfo; }

  // Generate C++ code to check this predicate. If a partitioner has already
  // tested this predicate then this function won't be called. If this function
  // is called, it must emit code and return true to indicate that it did so. If
  // it ever returns false, then the caller will abort due to an untested
  // predicate.
  virtual bool generateCheckCode(raw_ostream &OS, StringRef Indent,
                                 const CodeExpansions &Expansions) const {
    return false;
  }

  virtual void print(raw_ostream &OS) const;
  virtual void printDescription(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  virtual LLVM_DUMP_METHOD void dump() const { print(errs()); }
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
};

class GIMatchDagOpcodePredicate : public GIMatchDagPredicate {
  const CodeGenInstruction &Instr;

public:
  GIMatchDagOpcodePredicate(GIMatchDagContext &Ctx, StringRef Name,
                            const CodeGenInstruction &Instr);

  static bool classof(const GIMatchDagPredicate *P) {
    return P->getKind() == GIMatchDagPredicateKind_Opcode;
  }

  const CodeGenInstruction *getInstr() const { return &Instr; }

  void printDescription(raw_ostream &OS) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  virtual LLVM_DUMP_METHOD void dump() const override { print(errs()); }
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
};

class GIMatchDagOneOfOpcodesPredicate : public GIMatchDagPredicate {
  SmallVector<const CodeGenInstruction *, 4> Instrs;

public:
  GIMatchDagOneOfOpcodesPredicate(GIMatchDagContext &Ctx, StringRef Name);

  void addOpcode(const CodeGenInstruction *Instr) { Instrs.push_back(Instr); }

  static bool classof(const GIMatchDagPredicate *P) {
    return P->getKind() == GIMatchDagPredicateKind_OneOfOpcodes;
  }

  const SmallVectorImpl<const CodeGenInstruction *> &getInstrs() const {
    return Instrs;
  }

  void printDescription(raw_ostream &OS) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  virtual LLVM_DUMP_METHOD void dump() const override { print(errs()); }
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
};

class GIMatchDagSameMOPredicate : public GIMatchDagPredicate {
public:
  GIMatchDagSameMOPredicate(GIMatchDagContext &Ctx, StringRef Name);

  static bool classof(const GIMatchDagPredicate *P) {
    return P->getKind() == GIMatchDagPredicateKind_SameMO;
  }

  void printDescription(raw_ostream &OS) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  virtual LLVM_DUMP_METHOD void dump() const override { print(errs()); }
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
};

raw_ostream &operator<<(raw_ostream &OS, const GIMatchDagPredicate &N);
raw_ostream &operator<<(raw_ostream &OS, const GIMatchDagOpcodePredicate &N);

} // end namespace llvm
#endif // ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAGPREDICATE_H
