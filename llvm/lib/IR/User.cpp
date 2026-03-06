//===-- User.cpp - Implement the User class -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/User.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IntrinsicInst.h"

using namespace llvm;

namespace llvm {
class BasicBlock;
}

//===----------------------------------------------------------------------===//
//                                 User Class
//===----------------------------------------------------------------------===//

bool User::replaceUsesOfWith(Value *From, Value *To) {
  bool Changed = false;
  if (From == To) return Changed;   // Duh what?

  assert((!isa<Constant>(this) || isa<GlobalValue>(this)) &&
         "Cannot call User::replaceUsesOfWith on a constant!");

  for (unsigned i = 0, E = getNumOperands(); i != E; ++i)
    if (getOperand(i) == From) {  // Is This operand is pointing to oldval?
      // The side effects of this setOperand call include linking to
      // "To", adding "this" to the uses list of To, and
      // most importantly, removing "this" from the use list of "From".
      setOperand(i, To);
      Changed = true;
    }
  if (auto DVI = dyn_cast_or_null<DbgVariableIntrinsic>(this)) {
    if (is_contained(DVI->location_ops(), From)) {
      DVI->replaceVariableLocationOp(From, To);
      Changed = true;
    }
  }

  return Changed;
}

//===----------------------------------------------------------------------===//
//                         User allocHungoffUses Implementation
//===----------------------------------------------------------------------===//

void User::allocHungoffUses(unsigned N, bool WithExtraValues) {
  assert(HasHungOffUses && "alloc must have hung off uses");

  static_assert(alignof(Use) >= alignof(Value *),
                "Alignment is insufficient for 'hung-off-uses' pieces");

  // Allocate the array of Uses
  size_t size = N * sizeof(Use);
  if (WithExtraValues)
    size += N * sizeof(Value *);
  Use *Begin = static_cast<Use*>(::operator new(size));
  Use *End = Begin + N;
  setOperandList(Begin);
  for (; Begin != End; Begin++)
    new (Begin) Use(this);
}

void User::growHungoffUses(unsigned NewNumUses, bool WithExtraValues) {
  assert(HasHungOffUses && "realloc must have hung off uses");

  unsigned OldNumUses = getNumOperands();

  // We don't support shrinking the number of uses.  We wouldn't have enough
  // space to copy the old uses in to the new space.
  assert(NewNumUses > OldNumUses && "realloc must grow num uses");

  Use *OldOps = getOperandList();
  allocHungoffUses(NewNumUses, WithExtraValues);
  Use *NewOps = getOperandList();

  // Now copy from the old operands list to the new one.
  std::copy(OldOps, OldOps + OldNumUses, NewOps);

  // If the User has extra values (phi basic blocks, switch case values), then
  // we need to copy these, too.
  if (WithExtraValues) {
    auto *OldPtr = reinterpret_cast<char *>(OldOps + OldNumUses);
    auto *NewPtr = reinterpret_cast<char *>(NewOps + NewNumUses);
    std::copy(OldPtr, OldPtr + (OldNumUses * sizeof(Value *)), NewPtr);
  }
  Use::zap(OldOps, OldOps + OldNumUses, true);
}

// This is a private struct used by `User` to track the co-allocated descriptor
// section.
struct DescriptorInfo {
  intptr_t SizeInBytes;
};

ArrayRef<const uint8_t> User::getDescriptor() const {
  auto MutableARef = const_cast<User *>(this)->getDescriptor();
  return {MutableARef.begin(), MutableARef.end()};
}

MutableArrayRef<uint8_t> User::getDescriptor() {
  assert(HasDescriptor && "Don't call otherwise!");
  assert(!HasHungOffUses && "Invariant!");

  auto *DI = reinterpret_cast<DescriptorInfo *>(getIntrusiveOperands()) - 1;
  assert(DI->SizeInBytes != 0 && "Should not have had a descriptor otherwise!");

  return MutableArrayRef<uint8_t>(
      reinterpret_cast<uint8_t *>(DI) - DI->SizeInBytes, DI->SizeInBytes);
}

bool User::isDroppable() const {
  if (auto *II = dyn_cast<IntrinsicInst>(this)) {
    switch (II->getIntrinsicID()) {
    default:
      return false;
    case Intrinsic::assume:
    case Intrinsic::pseudoprobe:
    case Intrinsic::experimental_noalias_scope_decl:
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
//                         User operator new Implementations
//===----------------------------------------------------------------------===//

void *User::allocateFixedOperandUser(size_t Size, unsigned Us,
                                     unsigned DescBytes) {
  assert(Us < (1u << NumUserOperandsBits) && "Too many operands");

  static_assert(sizeof(DescriptorInfo) % sizeof(void *) == 0, "Required below");

  unsigned DescBytesToAllocate =
      DescBytes == 0 ? 0 : (DescBytes + sizeof(DescriptorInfo));
  assert(DescBytesToAllocate % sizeof(void *) == 0 &&
         "We need this to satisfy alignment constraints for Uses");

  size_t LeadingSize = DescBytesToAllocate + sizeof(Use) * Us;

  // Ensure we allocate at least one pointer's worth of space before the main
  // user allocation. We use this memory to pass information from the destructor
  // to the deletion operator, so it can recover the true allocation start.
  LeadingSize = std::max(LeadingSize, sizeof(void *));

  uint8_t *Storage = static_cast<uint8_t *>(::operator new(LeadingSize + Size));
  User *Obj = reinterpret_cast<User *>(Storage + LeadingSize);
  Use *Operands = reinterpret_cast<Use *>(Obj) - Us;
  Obj->NumUserOperands = Us;
  Obj->HasHungOffUses = false;
  Obj->HasDescriptor = DescBytes != 0;

  if (DescBytes != 0) {
    auto *DescInfo = reinterpret_cast<DescriptorInfo *>(Operands) - 1;
    DescInfo->SizeInBytes = DescBytes;
  }

  return Obj;
}

void *User::operator new(size_t Size, IntrusiveOperandsAllocMarker allocTrait) {
  return allocateFixedOperandUser(Size, allocTrait.NumOps, 0);
}

void *User::operator new(size_t Size,
                         IntrusiveOperandsAndDescriptorAllocMarker allocTrait) {
  return allocateFixedOperandUser(Size, allocTrait.NumOps,
                                  allocTrait.DescBytes);
}

void *User::operator new(size_t Size, HungOffOperandsAllocMarker) {
  // Allocate space for a single Use*
  void *Storage = ::operator new(Size + sizeof(Use *));
  Use **HungOffOperandList = static_cast<Use **>(Storage);
  User *Obj = reinterpret_cast<User *>(HungOffOperandList + 1);
  Obj->NumUserOperands = 0;
  Obj->HasHungOffUses = true;
  Obj->HasDescriptor = false;
  *HungOffOperandList = nullptr;
  return Obj;
}

//===----------------------------------------------------------------------===//
//                         User operator delete Implementation
//===----------------------------------------------------------------------===//

User::~User() {
  // Hung off uses use a single Use* before the User, while other subclasses
  // use a Use[] allocated prior to the user.
  void *AllocStart = nullptr;
  if (HasHungOffUses) {
    assert(!HasDescriptor && "not supported!");

    Use **HungOffOperandList = reinterpret_cast<Use **>(this) - 1;
    // drop the hung off uses.
    Use::zap(*HungOffOperandList, *HungOffOperandList + NumUserOperands,
             /* Delete */ true);
    AllocStart = HungOffOperandList;
  } else if (HasDescriptor) {
    Use *UseBegin = reinterpret_cast<Use *>(this) - NumUserOperands;
    Use::zap(UseBegin, UseBegin + NumUserOperands, /* Delete */ false);

    auto *DI = reinterpret_cast<DescriptorInfo *>(UseBegin) - 1;
    AllocStart = reinterpret_cast<uint8_t *>(DI) - DI->SizeInBytes;
  } else if (NumUserOperands > 0) {
    Use *Storage = reinterpret_cast<Use *>(this) - NumUserOperands;
    Use::zap(Storage, Storage + NumUserOperands,
             /* Delete */ false);
    AllocStart = Storage;
  } else {
    // Handle the edge case where there are no operands and no descriptor.
    AllocStart = (void **)(this) - 1;
  }

  // Operator delete needs to know where the allocation started. To avoid
  // use-after-destroy, we have to store the allocation start outside the User
  // object memory. The `User` new operator always allocates least one pointer
  // before the User, so we can use that to store the allocation start. As a
  // special case, we avoid this extra prefix allocation for ConstantData
  // instances, since those are extremely common.
  if (!isa<ConstantData>(this))
    ((void **)this)[-1] = AllocStart;
}

void User::operator delete(void *Usr) { ::operator delete(((void **)Usr)[-1]); }

void User::operator delete(void *Usr, HungOffOperandsAllocMarker) {
  Use **HungOffOperandList = static_cast<Use **>(Usr) - 1;
  ::operator delete(HungOffOperandList);
}

void User::operator delete(void *Usr,
                           IntrusiveOperandsAndDescriptorAllocMarker Marker) {
  unsigned NumOps = Marker.NumOps;
  Use *UseBegin = static_cast<Use *>(Usr) - NumOps;
  auto *DI = reinterpret_cast<DescriptorInfo *>(UseBegin) - 1;
  uint8_t *Storage = reinterpret_cast<uint8_t *>(DI) - DI->SizeInBytes;
  ::operator delete(Storage);
}

void User::operator delete(void *Usr, IntrusiveOperandsAllocMarker Marker) {
  unsigned NumOps = Marker.NumOps;
  size_t LeadingSize = sizeof(Use) * NumOps;
  // Handle the edge case where there are no operands and no descriptor.
  LeadingSize = std::max(LeadingSize, sizeof(void *));
  uint8_t *Storage = static_cast<uint8_t *>(Usr) - LeadingSize;
  ::operator delete(Storage);
}
