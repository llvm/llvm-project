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

#include "llvm/Use.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"

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
template<typename ValueTy> class StringMapEntry;
typedef StringMapEntry<Value*> ValueName;
class raw_ostream;
class AssemblyAnnotationWriter;
class ValueHandleBase;
class LLVMContext;
class Twine;
class MDNode;
class Type;
class StringRef;

//===----------------------------------------------------------------------===//
//                                 Value Class
//===----------------------------------------------------------------------===//

/// This is a very important LLVM class. It is the base class of all values 
/// computed by a program that may be used as operands to other values. Value is
/// the super class of other important classes such as Instruction and Function.
/// All Values have a Type. Type is not a subclass of Value. Some values can
/// have a name and they belong to some Module.  Setting the name on the Value
/// automatically updates the module's symbol table.
///
/// Every value has a "use list" that keeps track of which other Values are
/// using this Value.  A Value can also have an arbitrary number of ValueHandle
/// objects that watch it and listen to RAUW and Destroy events.  See
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

private:
  /// SubclassData - This member is defined by this class, but is not used for
  /// anything.  Subclasses can use it to hold whatever state they find useful.
  /// This field is initialized to zero by the ctor.
  unsigned short SubclassData;

  Type *VTy;
  Use *UseList;

  friend class ValueSymbolTable; // Allow ValueSymbolTable to directly mod Name.
  friend class ValueHandleBase;
  ValueName *Name;

  void operator=(const Value &) LLVM_DELETED_FUNCTION;
  Value(const Value &) LLVM_DELETED_FUNCTION;

protected:
  /// printCustom - Value subclasses can override this to implement custom
  /// printing behavior.
  virtual void printCustom(raw_ostream &O) const;

  Value(Type *Ty, unsigned scid);
public:
  virtual ~Value();

  /// dump - Support for debugging, callable in GDB: V->dump()
  //
  void dump() const;

  /// print - Implement operator<< on Value.
  ///
  void print(raw_ostream &O, AssemblyAnnotationWriter *AAW = 0) const;

  /// All values are typed, get the type of this value.
  ///
  Type *getType() const { return VTy; }

  /// All values hold a context through their type.
  LLVMContext &getContext() const;

  // All values can potentially be named.
  bool hasName() const { return Name != 0 && SubclassID != MDStringVal; }
  ValueName *getValueName() const { return Name; }
  void setValueName(ValueName *VN) { Name = VN; }
  
  /// getName() - Return a constant reference to the value's name. This is cheap
  /// and guaranteed to return the same reference as long as the value is not
  /// modified.
  StringRef getName() const;

  /// setName() - Change the name of the value, choosing a new unique name if
  /// the provided name is taken.
  ///
  /// \param Name The new name; or "" if the value's name should be removed.
  void setName(const Twine &Name);

  
  /// takeName - transfer the name from V to this value, setting V's name to
  /// empty.  It is an error to call V->takeName(V). 
  void takeName(Value *V);

  /// replaceAllUsesWith - Go through the uses list for this definition and make
  /// each use point to "V" instead of "this".  After this completes, 'this's
  /// use list is guaranteed to be empty.
  ///
  void replaceAllUsesWith(Value *V);

  //----------------------------------------------------------------------
  // Methods for handling the chain of uses of this Value.
  //
  typedef value_use_iterator<User>       use_iterator;
  typedef value_use_iterator<const User> const_use_iterator;

  bool               use_empty() const { return UseList == 0; }
  use_iterator       use_begin()       { return use_iterator(UseList); }
  const_use_iterator use_begin() const { return const_use_iterator(UseList); }
  use_iterator       use_end()         { return use_iterator(0);   }
  const_use_iterator use_end()   const { return const_use_iterator(0);   }
  User              *use_back()        { return *use_begin(); }
  const User        *use_back()  const { return *use_begin(); }

  /// hasOneUse - Return true if there is exactly one user of this value.  This
  /// is specialized because it is a common request and does not require
  /// traversing the whole use list.
  ///
  bool hasOneUse() const {
    const_use_iterator I = use_begin(), E = use_end();
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
  /// is a linear time operation.  Use hasOneUse, hasNUses, or hasNUsesOrMore
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
    BlockAddressVal,          // This is an instance of BlockAddress
    ConstantExprVal,          // This is an instance of ConstantExpr
    ConstantAggregateZeroVal, // This is an instance of ConstantAggregateZero
    ConstantDataArrayVal,     // This is an instance of ConstantDataArray
    ConstantDataVectorVal,    // This is an instance of ConstantDataVector
    ConstantIntVal,           // This is an instance of ConstantInt
    ConstantFPVal,            // This is an instance of ConstantFP
    ConstantArrayVal,         // This is an instance of ConstantArray
    ConstantStructVal,        // This is an instance of ConstantStruct
    ConstantVectorVal,        // This is an instance of ConstantVector
    ConstantPointerNullVal,   // This is an instance of ConstantPointerNull
    MDNodeVal,                // This is an instance of MDNode
    MDStringVal,              // This is an instance of MDString
    InlineAsmVal,             // This is an instance of InlineAsm
    PseudoSourceValueVal,     // This is an instance of PseudoSourceValue
    FixedStackPseudoSourceValueVal, // This is an instance of 
                                    // FixedStackPseudoSourceValue
    InstructionVal,           // This is an instance of Instruction
    // Enum values starting at InstructionVal are used for Instructions;
    // don't add new values here!

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

  /// getRawSubclassOptionalData - Return the raw optional flags value
  /// contained in this value. This should only be used when testing two
  /// Values for equivalence.
  unsigned getRawSubclassOptionalData() const {
    return SubclassOptionalData;
  }

  /// clearSubclassOptionalData - Clear the optional flags contained in
  /// this value.
  void clearSubclassOptionalData() {
    SubclassOptionalData = 0;
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

  /// hasValueHandle - Return true if there is a value handle associated with
  /// this value.
  bool hasValueHandle() const { return HasValueHandle; }

  /// stripPointerCasts - This method strips off any unneeded pointer casts and
  /// all-zero GEPs from the specified value, returning the original uncasted
  /// value. If this is called on a non-pointer value, it returns 'this'.
  Value *stripPointerCasts();
  const Value *stripPointerCasts() const {
    return const_cast<Value*>(this)->stripPointerCasts();
  }

  /// stripInBoundsConstantOffsets - This method strips off unneeded pointer casts and
  /// all-constant GEPs from the specified value, returning the original
  /// pointer value. If this is called on a non-pointer value, it returns
  /// 'this'.
  Value *stripInBoundsConstantOffsets();
  const Value *stripInBoundsConstantOffsets() const {
    return const_cast<Value*>(this)->stripInBoundsConstantOffsets();
  }

  /// stripInBoundsOffsets - This method strips off unneeded pointer casts and
  /// any in-bounds Offsets from the specified value, returning the original
  /// pointer value. If this is called on a non-pointer value, it returns
  /// 'this'.
  Value *stripInBoundsOffsets();
  const Value *stripInBoundsOffsets() const {
    return const_cast<Value*>(this)->stripInBoundsOffsets();
  }

  /// isDereferenceablePointer - Test if this value is always a pointer to
  /// allocated and suitably aligned memory for a simple load or store.
  bool isDereferenceablePointer() const;
  
  /// DoPHITranslation - If this value is a PHI node with CurBB as its parent,
  /// return the value in the PHI node corresponding to PredBB.  If not, return
  /// ourself.  This is useful if you want to know the value something has in a
  /// predecessor block.
  Value *DoPHITranslation(const BasicBlock *CurBB, const BasicBlock *PredBB);

  const Value *DoPHITranslation(const BasicBlock *CurBB,
                                const BasicBlock *PredBB) const{
    return const_cast<Value*>(this)->DoPHITranslation(CurBB, PredBB);
  }
  
  /// MaximumAlignment - This is the greatest alignment value supported by
  /// load, store, and alloca instructions, and global values.
  static const unsigned MaximumAlignment = 1u << 29;
  
  /// mutateType - Mutate the type of this Value to be of the specified type.
  /// Note that this is an extremely dangerous operation which can create
  /// completely invalid IR very easily.  It is strongly recommended that you
  /// recreate IR objects with the right types instead of mutating them in
  /// place.
  void mutateType(Type *Ty) {
    VTy = Ty;
  }
  
protected:
  unsigned short getSubclassDataFromValue() const { return SubclassData; }
  void setValueSubclassData(unsigned short D) { SubclassData = D; }
};

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
template <> struct isa_impl<Constant, Value> {
  static inline bool doit(const Value &Val) {
    return Val.getValueID() >= Value::ConstantFirstVal &&
      Val.getValueID() <= Value::ConstantLastVal;
  }
};

template <> struct isa_impl<Argument, Value> {
  static inline bool doit (const Value &Val) {
    return Val.getValueID() == Value::ArgumentVal;
  }
};

template <> struct isa_impl<InlineAsm, Value> { 
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::InlineAsmVal;
  }
};

template <> struct isa_impl<Instruction, Value> { 
  static inline bool doit(const Value &Val) {
    return Val.getValueID() >= Value::InstructionVal;
  }
};

template <> struct isa_impl<BasicBlock, Value> { 
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::BasicBlockVal;
  }
};

template <> struct isa_impl<Function, Value> { 
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::FunctionVal;
  }
};

template <> struct isa_impl<GlobalVariable, Value> { 
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::GlobalVariableVal;
  }
};

template <> struct isa_impl<GlobalAlias, Value> { 
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::GlobalAliasVal;
  }
};

template <> struct isa_impl<GlobalValue, Value> { 
  static inline bool doit(const Value &Val) {
    return isa<GlobalVariable>(Val) || isa<Function>(Val) ||
      isa<GlobalAlias>(Val);
  }
};

template <> struct isa_impl<MDNode, Value> { 
  static inline bool doit(const Value &Val) {
    return Val.getValueID() == Value::MDNodeVal;
  }
};
  
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
