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
typedef StringMapEntry<Value*> ValueName;
class raw_ostream;
class AssemblyAnnotationWriter;

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
/// using this Value.
/// @brief LLVM Value Representation
class Value {
  const unsigned short SubclassID;   // Subclass identifier (for isa/dyn_cast)
protected:
  /// SubclassData - This member is defined by this class, but is not used for
  /// anything.  Subclasses can use it to hold whatever state they find useful.
  /// This field is initialized to zero by the ctor.
  unsigned short SubclassData;
private:
  PATypeHolder VTy;
  Use *UseList;

  friend class ValueSymbolTable; // Allow ValueSymbolTable to directly mod Name.
  friend class SymbolTable;      // Allow SymbolTable to directly poke Name.
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

  // All values can potentially be named...
  inline bool hasName() const { return Name != 0; }
  ValueName *getValueName() const { return Name; }

  /// getNameStart - Return a pointer to a null terminated string for this name.
  /// Note that names can have null characters within the string as well as at
  /// their end.  This always returns a non-null pointer.
  const char *getNameStart() const;
  /// getNameEnd - Return a pointer to the end of the name.
  const char *getNameEnd() const { return getNameStart() + getNameLen(); }
  
  /// isName - Return true if this value has the name specified by the provided
  /// nul terminated string.
  bool isName(const char *N) const;
  
  /// getNameLen - Return the length of the string, correctly handling nul
  /// characters embedded into them.
  unsigned getNameLen() const;

  /// getName()/getNameStr() - Return the name of the specified value, 
  /// *constructing a string* to hold it.  Because these are guaranteed to
  /// construct a string, they are very expensive and should be avoided.
  std::string getName() const { return getNameStr(); }
  std::string getNameStr() const;


  void setName(const std::string &name);
  void setName(const char *Name, unsigned NameLen);
  void setName(const char *Name);  // Takes a null-terminated string.

  
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

} // End llvm namespace

#endif
