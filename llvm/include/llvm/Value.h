//===-- llvm/Value.h - Definition of the Value class ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Value class. 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_VALUE_H
#define LLVM_VALUE_H

#include "llvm/AbstractTypeUser.h"
#include "llvm/Use.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include <iosfwd>
#include <string>

namespace llvm {

class Constant;
class Argument;
class Instruction;
class BasicBlock;
class GlobalValue;
class Function;
class GlobalVariable;
class GlobalAlias;
class InlineAsm;
class ValueSymbolTable;
class TypeSymbolTable;
template<typename ValueTy> class StringMapEntry;
template <typename ValueTy = Value>
class AssertingVH;
typedef StringMapEntry<Value*> ValueName;
class raw_ostream;
class AssemblyAnnotationWriter;
class ValueHandleBase;
class LLVMContext;

//===----------------------------------------------------------------------===//
//                                 Value Class
//===----------------------------------------------------------------------===//

/// This is a very important LLVM class. It is the base class of all values 
/// computed by a program that may be used as operands to other values. Value is
/// the super class of other important classes such as Instruction and Function.
/// All Values have a Type. Type is not a subclass of Value. All types can have
/// a name and they should belong to some Module. Setting the name on the Value
/// automatically updates the module's symbol table.
///
/// Every value has a "use list" that keeps track of which other Values are
/// using this Value.  A Value can also have an arbitrary number of ValueHandle
/// objects that watch it and listen to RAUW and Destroy events see
/// llvm/Support/ValueHandle.h for details.
///
/// @brief LLVM Value Representation
class Value {
  const unsigned char SubclassID;   // Subclass identifier (for isa/dyn_cast)
  unsigned char HasValueHandle : 1; // Has a ValueHandle pointing to this?
protected:
  /// SubclassOptionalData - This member is similar to SubclassData, however it
  /// is for holding information which may be used to aid optimization, but
  /// which may be cleared to zero without affecting conservative
  /// interpretation.
  unsigned char SubclassOptionalData : 7;

  /// SubclassData - This member is defined by this class, but is not used for
  /// anything.  Subclasses can use it to hold whatever state they find useful.
  /// This field is initialized to zero by the ctor.
  unsigned short SubclassData;
private:
  PATypeHolder VTy;
  Use *UseList;

  friend class ValueSymbolTable; // Allow ValueSymbolTable to directly mod Name.
  friend class SymbolTable;      // Allow SymbolTable to directly poke Name.
  friend class ValueHandleBase;
  ValueName *Name;

  void operator=(const Value &);     // Do not implement
  Value(const Value &);              // Do not implement

public:
  Value(const Type *Ty, unsigned scid);
  virtual ~Value();

  /// dump - Support for debugging, callable in GDB: V->dump()
  //
  virtual void dump() const;

  /// print - Implement operator<< on Value.
  ///
  void print(std::ostream &O, AssemblyAnnotationWriter *AAW = 0) const;
  void print(raw_ostream &O, AssemblyAnnotationWriter *AAW = 0) const;

  /// All values are typed, get the type of this value.
  ///
  inline const Type *getType() const { return VTy; }

  /// All values hold a context through their type.
  LLVMContext &getContext() const;

  // All values can potentially be named...
  inline bool hasName() const { return Name != 0; }
  ValueName *getValueName() const { return Name; }
  
  /// getName() - Return a constant reference to the value's name. This is cheap
  /// and guaranteed to return the same reference as long as the value is not
  /// modified.
  ///
  /// This is currently guaranteed to return a StringRef for which data() points
  /// to a valid null terminated string. The use of StringRef.data() is 
  /// deprecated here, however, and clients should not rely on it. If such 
  /// behavior is needed, clients should use expensive getNameStr(), or switch 
  /// to an interface that does not depend on null termination.
  StringRef getName() const;

  /// getNameStr() - Return the name of the specified value, *constructing a
  /// string* to hold it.  This is guaranteed to construct a string and is very
  /// expensive, clients should use getName() unless necessary.
  std::string getNameStr() const;

  /// setName() - Change the name of the value, choosing a new unique name if
  /// the provided name is taken.
  ///
  /// \arg Name - The new name; or "" if the value's name should be removed.
  void setName(const Twine &Name);

  
  /// takeName - transfer the name from V to this value, setting V's name to
  /// empty.  It is an error to call V->takeName(V). 
  void takeName(Value *V);

  /// replaceAllUsesWith - Go through the uses list for this definition and make
  /// each use point to "V" instead of "this".  After this completes, 'this's
  /// use list is guaranteed to be empty.
  ///
  void replaceAllUsesWith(Value *V);

  // uncheckedReplaceAllUsesWith - Just like replaceAllUsesWith but dangerous.
  // Only use when in type resolution situations!
  void uncheckedReplaceAllUsesWith(Value *V);

  /// clearOptionalData - Clear any optional optimization data from this Value.
  /// Transformation passes must call this method whenever changing the IR
  /// in a way that would affect the values produced by this Value, unless
  /// it takes special care to ensure correctness in some other way.
  void clearOptionalData() { SubclassOptionalData = 0; }

  //----------------------------------------------------------------------
  // Methods for handling the chain of uses of this Value.
  //
  typedef value_use_iterator<User>       use_iterator;
  typedef value_use_iterator<const User> use_const_iterator;

  bool               use_empty() const { return UseList == 0; }
  use_iterator       use_begin()       { return use_iterator(UseList); }
  use_const_iterator use_begin() const { return use_const_iterator(UseList); }
  use_iterator       use_end()         { return use_iterator(0);   }
  use_const_iterator use_end()   const { return use_const_iterator(0);   }
  User              *use_back()        { return *use_begin(); }
  const User        *use_back()  const { return *use_begin(); }

  /// hasOneUse - Return true if there is exactly one user of this value.  This
  /// is specialized because it is a common request and does not require
  /// traversing the whole use list.
  ///
  bool hasOneUse() const {
    use_const_iterator I = use_begin(), E = use_end();
    if (I == E) return false;
    return ++I == E;
  }

  /// hasNUses - Return true if this Value has exactly N users.
  ///
  bool hasNUses(unsigned N) const;

  /// hasNUsesOrMore - Return true if this value has N users or more.  This is
  /// logically equivalent to getNumUses() >= N.
  ///
  bool hasNUsesOrMore(unsigned N) const;

  bool isUsedInBasicBlock(const BasicBlock *BB) const;

  /// getNumUses - This method computes the number of uses of this Value.  This
  /// is a linear time operation.  Use hasOneUse, hasNUses, or hasMoreThanNUses
  /// to check for specific values.
  unsigned getNumUses() const;

  /// addUse - This method should only be used by the Use class.
  ///
  void addUse(Use &U) { U.addToList(&UseList); }

  /// An enumeration for keeping track of the concrete subclass of Value that
  /// is actually instantiated. Values of this enumeration are kept in the 
  /// Value classes SubclassID field. They are used for concrete type
  /// identification.
  enum ValueTy {
    ArgumentVal,              // This is an instance of Argument
    BasicBlockVal,            // This is an instance of BasicBlock
    FunctionVal,              // This is an instance of Function
    GlobalAliasVal,           // This is an instance of GlobalAlias
    GlobalVariableVal,        // This is an instance of GlobalVariable
    UndefValueVal,            // This is an instance of UndefValue
    ConstantExprVal,          // This is an instance of ConstantExpr
    ConstantAggregateZeroVal, // This is an instance of ConstantAggregateNull
    ConstantIntVal,           // This is an instance of ConstantInt
    ConstantFPVal,            // This is an instance of ConstantFP
    ConstantArrayVal,         // This is an instance of ConstantArray
    ConstantStructVal,        // This is an instance of ConstantStruct
    ConstantVectorVal,        // This is an instance of ConstantVector
    ConstantPointerNullVal,   // This is an instance of ConstantPointerNull
    MDNodeVal,                // This is an instance of MDNode
    MDStringVal,              // This is an instance of MDString
    NamedMDNodeVal,           // This is an instance of NamedMDNode
    InlineAsmVal,             // This is an instance of InlineAsm
    PseudoSourceValueVal,     // This is an instance of PseudoSourceValue
    InstructionVal,           // This is an instance of Instruction
    
    // Markers:
    ConstantFirstVal = FunctionVal,
    ConstantLastVal  = ConstantPointerNullVal
  };

  /// getValueID - Return an ID for the concrete type of this object.  This is
  /// used to implement the classof checks.  This should not be used for any
  /// other purpose, as the values may change as LLVM evolves.  Also, note that
  /// for instructions, the Instruction's opcode is added to InstructionVal. So
  /// this means three things:
  /// # there is no value with code InstructionVal (no opcode==0).
  /// # there are more possible values for the value type than in ValueTy enum.
  /// # the InstructionVal enumerator must be the highest valued enumerator in
  ///   the ValueTy enum.
  unsigned getValueID() const {
    return SubclassID;
  }

  /// hasSameSubclassOptionalData - Test whether the optional flags contained
  /// in this value are equal to the optional flags in the given value.
  bool hasSameSubclassOptionalData(const Value *V) const {
    return SubclassOptionalData == V->SubclassOptionalData;
  }

  /// intersectOptionalDataWith - Clear any optional flags in this value
  /// that are not also set in the given value.
  void intersectOptionalDataWith(const Value *V) {
    SubclassOptionalData &= V->SubclassOptionalData;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *) {
    return true; // Values are always values.
  }

  /// getRawType - This should only be used to implement the vmcore library.
  ///
  const Type *getRawType() const { return VTy.getRawType(); }

  /// stripPointerCasts - This method strips off any unneeded pointer
  /// casts from the specified value, returning the original uncasted value.
  /// Note that the returned value has pointer type if the specified value does.
  Value *stripPointerCasts();
  const Value *stripPointerCasts() const {
    return const_cast<Value*>(this)->stripPointerCasts();
  }

  /// getUnderlyingObject - This method strips off any GEP address adjustments
  /// and pointer casts from the specified value, returning the original object
  /// being addressed.  Note that the returned value has pointer type if the
  /// specified value does.
  Value *getUnderlyingObject();
  const Value *getUnderlyingObject() const {
    return const_cast<Value*>(this)->getUnderlyingObject();
  }
  
  /// DoPHITranslation - If this value is a PHI node with CurBB as its parent,
  /// return the value in the PHI node corresponding to PredBB.  If not, return
  /// ourself.  This is useful if you want to know the value something has in a
  /// predecessor block.
  Value *DoPHITranslation(const BasicBlock *CurBB, const BasicBlock *PredBB);

  const Value *DoPHITranslation(const BasicBlock *CurBB,
                                const BasicBlock *PredBB) const{
    return const_cast<Value*>(this)->DoPHITranslation(CurBB, PredBB);
  }
};

inline std::ostream &operator<<(std::ostream &OS, const Value &V) {
  V.print(OS);
  return OS;
}
inline raw_ostream &operator<<(raw_ostream &OS, const Value &V) {
  V.print(OS);
  return OS;
}
  
void Use::set(Value *V) {
  if (Val) removeFromList();
  Val = V;
  if (V) V->addUse(*this);
}


// isa - Provide some specializations of isa so that we don't have to include
// the subtype header files to test to see if the value is a subclass...
//
template <> inline bool isa_impl<Constant, Value>(const Value &Val) {
  return Val.getValueID() >= Value::ConstantFirstVal &&
         Val.getValueID() <= Value::ConstantLastVal;
}
template <> inline bool isa_impl<Argument, Value>(const Value &Val) {
  return Val.getValueID() == Value::ArgumentVal;
}
template <> inline bool isa_impl<InlineAsm, Value>(const Value &Val) {
  return Val.getValueID() == Value::InlineAsmVal;
}
template <> inline bool isa_impl<Instruction, Value>(const Value &Val) {
  return Val.getValueID() >= Value::InstructionVal;
}
template <> inline bool isa_impl<BasicBlock, Value>(const Value &Val) {
  return Val.getValueID() == Value::BasicBlockVal;
}
template <> inline bool isa_impl<Function, Value>(const Value &Val) {
  return Val.getValueID() == Value::FunctionVal;
}
template <> inline bool isa_impl<GlobalVariable, Value>(const Value &Val) {
  return Val.getValueID() == Value::GlobalVariableVal;
}
template <> inline bool isa_impl<GlobalAlias, Value>(const Value &Val) {
  return Val.getValueID() == Value::GlobalAliasVal;
}
template <> inline bool isa_impl<GlobalValue, Value>(const Value &Val) {
  return isa<GlobalVariable>(Val) || isa<Function>(Val) ||
         isa<GlobalAlias>(Val);
}
  
  
// Value* is only 4-byte aligned.
template<>
class PointerLikeTypeTraits<Value*> {
  typedef Value* PT;
public:
  static inline void *getAsVoidPointer(PT P) { return P; }
  static inline PT getFromVoidPointer(void *P) {
    return static_cast<PT>(P);
  }
  enum { NumLowBitsAvailable = 2 };
};

} // End llvm namespace

#endif
