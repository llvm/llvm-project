//===- VPlanValue.h - Represent Values in Vectorizer Plan -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the entities induced by Vectorization
/// Plans, e.g. the instructions the VPlan intends to generate if executed.
/// VPlan models the following entities:
/// VPValue
///  |-- VPUser
///  |    |-- VPInstruction
/// These are documented in docs/VectorizationPlan.rst.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLAN_VALUE_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLAN_VALUE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

// Forward declarations.
class VPUser;

// This is the base class of the VPlan Def/Use graph, used for modeling the data
// flow into, within and out of the VPlan. VPValues can stand for live-ins
// coming from the input IR, instructions which VPlan will generate if executed
// and live-outs which the VPlan will need to fix accordingly.
class VPValue {
private:
  const unsigned char SubclassID; ///< Subclass identifier (for isa/dyn_cast).

  SmallVector<VPUser *, 1> Users;

protected:
  VPValue(const unsigned char SC) : SubclassID(SC) {}

public:
  /// An enumeration for keeping track of the concrete subclass of VPValue that
  /// are actually instantiated. Values of this enumeration are kept in the
  /// SubclassID field of the VPValue objects. They are used for concrete
  /// type identification.
  enum { VPValueSC, VPUserSC, VPInstructionSC };

  VPValue() : SubclassID(VPValueSC) {}
  VPValue(const VPValue &) = delete;
  VPValue &operator=(const VPValue &) = delete;

  /// \return an ID for the concrete type of this object.
  /// This is used to implement the classof checks. This should not be used
  /// for any other purpose, as the values may change as LLVM evolves.
  unsigned getVPValueID() const { return SubclassID; }

  void printAsOperand(raw_ostream &OS) const {
    OS << "%vp" << (unsigned short)(unsigned long long)this;
  }

  unsigned getNumUsers() const { return Users.size(); }
  void addUser(VPUser &User) { Users.push_back(&User); }

  typedef SmallVectorImpl<VPUser *>::iterator user_iterator;
  typedef SmallVectorImpl<VPUser *>::const_iterator const_user_iterator;
  typedef iterator_range<user_iterator> user_range;
  typedef iterator_range<const_user_iterator> const_user_range;

  user_iterator user_begin() { return Users.begin(); }
  const_user_iterator user_begin() const { return Users.begin(); }
  user_iterator user_end() { return Users.end(); }
  const_user_iterator user_end() const { return Users.end(); }
  user_range users() { return user_range(user_begin(), user_end()); }
  const_user_range users() const {
    return const_user_range(user_begin(), user_end());
  }
};

typedef DenseMap<Value *, VPValue *> Value2VPValueTy;
typedef DenseMap<VPValue *, Value *> VPValue2ValueTy;

raw_ostream &operator<<(raw_ostream &OS, const VPValue &V);

/// This class augments VPValue with operands which provide the inverse def-use
/// edges from VPValue's users to their defs.
class VPUser : public VPValue {
private:
  SmallVector<VPValue *, 2> Operands;

  void addOperand(VPValue *Operand) {
    Operands.push_back(Operand);
    Operand->addUser(*this);
  }

protected:
  VPUser(const unsigned char SC) : VPValue(SC) {}
  VPUser(const unsigned char SC, ArrayRef<VPValue *> Operands) : VPValue(SC) {
    for (VPValue *Operand : Operands)
      addOperand(Operand);
  }

public:
  VPUser() : VPValue(VPValue::VPUserSC) {}
  VPUser(ArrayRef<VPValue *> Operands) : VPUser(VPValue::VPUserSC, Operands) {}
  VPUser(std::initializer_list<VPValue *> Operands)
      : VPUser(ArrayRef<VPValue *>(Operands)) {}
  VPUser(const VPUser &) = delete;
  VPUser &operator=(const VPUser &) = delete;

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const VPValue *V) {
    return V->getVPValueID() >= VPUserSC &&
           V->getVPValueID() <= VPInstructionSC;
  }

  unsigned getNumOperands() const { return Operands.size(); }
  inline VPValue *getOperand(unsigned N) const {
    assert(N < Operands.size() && "Operand index out of bounds");
    return Operands[N];
  }

  typedef SmallVectorImpl<VPValue *>::iterator operand_iterator;
  typedef SmallVectorImpl<VPValue *>::const_iterator const_operand_iterator;
  typedef iterator_range<operand_iterator> operand_range;
  typedef iterator_range<const_operand_iterator> const_operand_range;

  operand_iterator op_begin() { return Operands.begin(); }
  const_operand_iterator op_begin() const { return Operands.begin(); }
  operand_iterator op_end() { return Operands.end(); }
  const_operand_iterator op_end() const { return Operands.end(); }
  operand_range operands() { return operand_range(op_begin(), op_end()); }
  const_operand_range operands() const {
    return const_operand_range(op_begin(), op_end());
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLAN_VALUE_H
