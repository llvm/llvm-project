//===- VPlanValue.h - Represent Values in Vectorizer Plan -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the entities induced by Vectorization
/// Plans, e.g. the instructions the VPlan intends to generate if executed.
/// VPlan models the following entities:
/// VPValue   VPUser
///    |        |
///   VPInstruction
/// These are documented in docs/VectorizationPlan.rst.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLAN_VALUE_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLAN_VALUE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {

// Forward declarations.
class raw_ostream;
class Value;
class VPSlotTracker;
class VPUser;
class VPRecipeBase;

// This is the base class of the VPlan Def/Use graph, used for modeling the data
// flow into, within and out of the VPlan. VPValues can stand for live-ins
// coming from the input IR, instructions which VPlan will generate if executed
// and live-outs which the VPlan will need to fix accordingly.
class VPValue {
  friend class VPBuilder;
  friend struct VPlanTransforms;
  friend class VPBasicBlock;
  friend class VPInterleavedAccessInfo;
  friend class VPSlotTracker;

  const unsigned char SubclassID; ///< Subclass identifier (for isa/dyn_cast).

  SmallVector<VPUser *, 1> Users;

protected:
  // Hold the underlying Value, if any, attached to this VPValue.
  Value *UnderlyingVal;

  VPValue(const unsigned char SC, Value *UV = nullptr)
      : SubclassID(SC), UnderlyingVal(UV) {}

  // DESIGN PRINCIPLE: Access to the underlying IR must be strictly limited to
  // the front-end and back-end of VPlan so that the middle-end is as
  // independent as possible of the underlying IR. We grant access to the
  // underlying IR using friendship. In that way, we should be able to use VPlan
  // for multiple underlying IRs (Polly?) by providing a new VPlan front-end,
  // back-end and analysis information for the new IR.

  /// Return the underlying Value attached to this VPValue.
  Value *getUnderlyingValue() { return UnderlyingVal; }
  const Value *getUnderlyingValue() const { return UnderlyingVal; }

  // Set \p Val as the underlying Value of this VPValue.
  void setUnderlyingValue(Value *Val) {
    assert(!UnderlyingVal && "Underlying Value is already set.");
    UnderlyingVal = Val;
  }

public:
  /// An enumeration for keeping track of the concrete subclass of VPValue that
  /// are actually instantiated. Values of this enumeration are kept in the
  /// SubclassID field of the VPValue objects. They are used for concrete
  /// type identification.
  enum { VPValueSC, VPInstructionSC };

  VPValue(Value *UV = nullptr) : VPValue(VPValueSC, UV) {}
  VPValue(const VPValue &) = delete;
  VPValue &operator=(const VPValue &) = delete;

  /// \return an ID for the concrete type of this object.
  /// This is used to implement the classof checks. This should not be used
  /// for any other purpose, as the values may change as LLVM evolves.
  unsigned getVPValueID() const { return SubclassID; }

  void printAsOperand(raw_ostream &OS, VPSlotTracker &Tracker) const;
  void print(raw_ostream &OS, VPSlotTracker &Tracker) const;

  /// Dump the value to stderr (for debugging).
  void dump() const;

  unsigned getNumUsers() const { return Users.size(); }
  void addUser(VPUser &User) { Users.push_back(&User); }

  /// Remove a single \p User from the list of users.
  void removeUser(VPUser &User) {
    bool Found = false;
    // The same user can be added multiple times, e.g. because the same VPValue
    // is used twice by the same VPUser. Remove a single one.
    erase_if(Users, [&User, &Found](VPUser *Other) {
      if (Found)
        return false;
      if (Other == &User) {
        Found = true;
        return true;
      }
      return false;
    });
  }

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

  /// Returns true if the value has more than one unique user.
  bool hasMoreThanOneUniqueUser() {
    if (getNumUsers() == 0)
      return false;

    // Check if all users match the first user.
    auto Current = std::next(user_begin());
    while (Current != user_end() && *user_begin() == *Current)
      Current++;
    return Current != user_end();
  }

  void replaceAllUsesWith(VPValue *New);
};

typedef DenseMap<Value *, VPValue *> Value2VPValueTy;
typedef DenseMap<VPValue *, Value *> VPValue2ValueTy;

raw_ostream &operator<<(raw_ostream &OS, const VPValue &V);

/// This class augments VPValue with operands which provide the inverse def-use
/// edges from VPValue's users to their defs.
class VPUser {
  SmallVector<VPValue *, 2> Operands;

public:
  VPUser() {}
  VPUser(ArrayRef<VPValue *> Operands) {
    for (VPValue *Operand : Operands)
      addOperand(Operand);
  }

  VPUser(std::initializer_list<VPValue *> Operands)
      : VPUser(ArrayRef<VPValue *>(Operands)) {}
  template <typename IterT> VPUser(iterator_range<IterT> Operands) {
    for (VPValue *Operand : Operands)
      addOperand(Operand);
  }

  VPUser(const VPUser &) = delete;
  VPUser &operator=(const VPUser &) = delete;
  virtual ~VPUser() {
    for (VPValue *Op : operands())
      Op->removeUser(*this);
  }

  void addOperand(VPValue *Operand) {
    Operands.push_back(Operand);
    Operand->addUser(*this);
  }

  unsigned getNumOperands() const { return Operands.size(); }
  inline VPValue *getOperand(unsigned N) const {
    assert(N < Operands.size() && "Operand index out of bounds");
    return Operands[N];
  }

  void setOperand(unsigned I, VPValue *New) {
    Operands[I]->removeUser(*this);
    Operands[I] = New;
    New->addUser(*this);
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

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const VPRecipeBase *Recipe);
};
class VPlan;
class VPBasicBlock;
class VPRegionBlock;

/// This class can be used to assign consecutive numbers to all VPValues in a
/// VPlan and allows querying the numbering for printing, similar to the
/// ModuleSlotTracker for IR values.
class VPSlotTracker {
  DenseMap<const VPValue *, unsigned> Slots;
  unsigned NextSlot = 0;

  void assignSlots(const VPBlockBase *VPBB);
  void assignSlots(const VPRegionBlock *Region);
  void assignSlots(const VPBasicBlock *VPBB);
  void assignSlot(const VPValue *V);

  void assignSlots(const VPlan &Plan);

public:
  VPSlotTracker(const VPlan *Plan) {
    if (Plan)
      assignSlots(*Plan);
  }

  unsigned getSlot(const VPValue *V) const {
    auto I = Slots.find(V);
    if (I == Slots.end())
      return -1;
    return I->second;
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLAN_VALUE_H
