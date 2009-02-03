//===-- Globals.cpp - Implement the GlobalValue & GlobalVariable class ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the GlobalValue & GlobalVariable classes for the VMCore
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/LeakDetector.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                            GlobalValue Class
//===----------------------------------------------------------------------===//

/// removeDeadUsersOfConstant - If the specified constantexpr is dead, remove
/// it.  This involves recursively eliminating any dead users of the
/// constantexpr.
static bool removeDeadUsersOfConstant(Constant *C) {
  if (isa<GlobalValue>(C)) return false; // Cannot remove this

  while (!C->use_empty()) {
    Constant *User = dyn_cast<Constant>(C->use_back());
    if (!User) return false; // Non-constant usage;
    if (!removeDeadUsersOfConstant(User))
      return false; // Constant wasn't dead
  }

  C->destroyConstant();
  return true;
}

/// removeDeadConstantUsers - If there are any dead constant users dangling
/// off of this global value, remove them.  This method is useful for clients
/// that want to check to see if a global is unused, but don't want to deal
/// with potentially dead constants hanging off of the globals.
void GlobalValue::removeDeadConstantUsers() {
  Value::use_iterator I = use_begin(), E = use_end();
  Value::use_iterator LastNonDeadUser = E;
  while (I != E) {
    if (Constant *User = dyn_cast<Constant>(*I)) {
      if (!removeDeadUsersOfConstant(User)) {
        // If the constant wasn't dead, remember that this was the last live use
        // and move on to the next constant.
        LastNonDeadUser = I;
        ++I;
      } else {
        // If the constant was dead, then the iterator is invalidated.
        if (LastNonDeadUser == E) {
          I = use_begin();
          if (I == E) break;
        } else {
          I = LastNonDeadUser;
          ++I;
        }
      }
    } else {
      LastNonDeadUser = I;
      ++I;
    }
  }
}

/// Override destroyConstant to make sure it doesn't get called on
/// GlobalValue's because they shouldn't be treated like other constants.
void GlobalValue::destroyConstant() {
  assert(0 && "You can't GV->destroyConstant()!");
  abort();
}

/// copyAttributesFrom - copy all additional attributes (those not needed to
/// create a GlobalValue) from the GlobalValue Src to this one.
void GlobalValue::copyAttributesFrom(const GlobalValue *Src) {
  setAlignment(Src->getAlignment());
  setSection(Src->getSection());
  setVisibility(Src->getVisibility());
}


//===----------------------------------------------------------------------===//
// GlobalVariable Implementation
//===----------------------------------------------------------------------===//

GlobalVariable::GlobalVariable(const Type *Ty, bool constant, LinkageTypes Link,
                               Constant *InitVal, const std::string &Name,
                               Module *ParentModule, bool ThreadLocal, 
                               unsigned AddressSpace)
  : GlobalValue(PointerType::get(Ty, AddressSpace), Value::GlobalVariableVal,
                OperandTraits<GlobalVariable>::op_begin(this),
                InitVal != 0, Link, Name),
    isConstantGlobal(constant), isThreadLocalSymbol(ThreadLocal) {
  if (InitVal) {
    assert(InitVal->getType() == Ty &&
           "Initializer should be the same type as the GlobalVariable!");
    Op<0>() = InitVal;
  }

  LeakDetector::addGarbageObject(this);

  if (ParentModule)
    ParentModule->getGlobalList().push_back(this);
}

GlobalVariable::GlobalVariable(const Type *Ty, bool constant, LinkageTypes Link,
                               Constant *InitVal, const std::string &Name,
                               GlobalVariable *Before, bool ThreadLocal,
                               unsigned AddressSpace)
  : GlobalValue(PointerType::get(Ty, AddressSpace), Value::GlobalVariableVal,
                OperandTraits<GlobalVariable>::op_begin(this),
                InitVal != 0, Link, Name),
    isConstantGlobal(constant), isThreadLocalSymbol(ThreadLocal) {
  if (InitVal) {
    assert(InitVal->getType() == Ty &&
           "Initializer should be the same type as the GlobalVariable!");
    Op<0>() = InitVal;
  }
  
  LeakDetector::addGarbageObject(this);
  
  if (Before)
    Before->getParent()->getGlobalList().insert(Before, this);
}

void GlobalVariable::setParent(Module *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

void GlobalVariable::removeFromParent() {
  getParent()->getGlobalList().remove(this);
}

void GlobalVariable::eraseFromParent() {
  getParent()->getGlobalList().erase(this);
}

void GlobalVariable::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                 Use *U) {
  // If you call this, then you better know this GVar has a constant
  // initializer worth replacing. Enforce that here.
  assert(getNumOperands() == 1 &&
         "Attempt to replace uses of Constants on a GVar with no initializer");

  // And, since you know it has an initializer, the From value better be
  // the initializer :)
  assert(getOperand(0) == From &&
         "Attempt to replace wrong constant initializer in GVar");

  // And, you better have a constant for the replacement value
  assert(isa<Constant>(To) &&
         "Attempt to replace GVar initializer with non-constant");

  // Okay, preconditions out of the way, replace the constant initializer.
  this->setOperand(0, cast<Constant>(To));
}

/// copyAttributesFrom - copy all additional attributes (those not needed to
/// create a GlobalVariable) from the GlobalVariable Src to this one.
void GlobalVariable::copyAttributesFrom(const GlobalValue *Src) {
  assert(isa<GlobalVariable>(Src) && "Expected a GlobalVariable!");
  GlobalValue::copyAttributesFrom(Src);
  const GlobalVariable *SrcVar = cast<GlobalVariable>(Src);
  setThreadLocal(SrcVar->isThreadLocal());
}


//===----------------------------------------------------------------------===//
// GlobalAlias Implementation
//===----------------------------------------------------------------------===//

GlobalAlias::GlobalAlias(const Type *Ty, LinkageTypes Link,
                         const std::string &Name, Constant* aliasee,
                         Module *ParentModule)
  : GlobalValue(Ty, Value::GlobalAliasVal, &Op<0>(), 1, Link, Name) {
  LeakDetector::addGarbageObject(this);

  if (aliasee)
    assert(aliasee->getType() == Ty && "Alias and aliasee types should match!");
  Op<0>() = aliasee;

  if (ParentModule)
    ParentModule->getAliasList().push_back(this);
}

void GlobalAlias::setParent(Module *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

void GlobalAlias::removeFromParent() {
  getParent()->getAliasList().remove(this);
}

void GlobalAlias::eraseFromParent() {
  getParent()->getAliasList().erase(this);
}

bool GlobalAlias::isDeclaration() const {
  const GlobalValue* AV = getAliasedGlobal();
  if (AV)
    return AV->isDeclaration();
  else
    return false;
}

void GlobalAlias::setAliasee(Constant *Aliasee) 
{
  if (Aliasee)
    assert(Aliasee->getType() == getType() &&
           "Alias and aliasee types should match!");
  
  setOperand(0, Aliasee);
}

const GlobalValue *GlobalAlias::getAliasedGlobal() const {
  const Constant *C = getAliasee();
  if (C) {
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C))
      return GV;
    else {
      const ConstantExpr *CE = 0;
      if ((CE = dyn_cast<ConstantExpr>(C)) &&
          (CE->getOpcode() == Instruction::BitCast || 
           CE->getOpcode() == Instruction::GetElementPtr))
        return dyn_cast<GlobalValue>(CE->getOperand(0));
      else
        assert(0 && "Unsupported aliasee");
    }
  }
  return 0;
}

const GlobalValue *GlobalAlias::resolveAliasedGlobal(bool stopOnWeak) const {
  SmallPtrSet<const GlobalValue*, 3> Visited;

  // Check if we need to stop early.
  if (stopOnWeak && mayBeOverridden())
    return this;

  const GlobalValue *GV = getAliasedGlobal();
  Visited.insert(GV);

  // Iterate over aliasing chain, stopping on weak alias if necessary.
  while (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV)) {
    if (stopOnWeak && GA->mayBeOverridden())
      break;

    GV = GA->getAliasedGlobal();

    if (!Visited.insert(GV))
      return NULL;
  }

  return GV;
}
