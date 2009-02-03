//===-- Type.cpp - Implement the Type class -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Type class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdarg>
using namespace llvm;

// DEBUG_MERGE_TYPES - Enable this #define to see how and when derived types are
// created and later destroyed, all in an effort to make sure that there is only
// a single canonical version of a type.
//
// #define DEBUG_MERGE_TYPES 1

AbstractTypeUser::~AbstractTypeUser() {}


//===----------------------------------------------------------------------===//
//                         Type Class Implementation
//===----------------------------------------------------------------------===//

// Concrete/Abstract TypeDescriptions - We lazily calculate type descriptions
// for types as they are needed.  Because resolution of types must invalidate
// all of the abstract type descriptions, we keep them in a seperate map to make
// this easy.
static ManagedStatic<std::map<const Type*, 
                              std::string> > ConcreteTypeDescriptions;
static ManagedStatic<std::map<const Type*,
                              std::string> > AbstractTypeDescriptions;

/// Because of the way Type subclasses are allocated, this function is necessary
/// to use the correct kind of "delete" operator to deallocate the Type object.
/// Some type objects (FunctionTy, StructTy) allocate additional space after 
/// the space for their derived type to hold the contained types array of
/// PATypeHandles. Using this allocation scheme means all the PATypeHandles are
/// allocated with the type object, decreasing allocations and eliminating the
/// need for a std::vector to be used in the Type class itself. 
/// @brief Type destruction function
void Type::destroy() const {

  // Structures and Functions allocate their contained types past the end of
  // the type object itself. These need to be destroyed differently than the
  // other types.
  if (isa<FunctionType>(this) || isa<StructType>(this)) {
    // First, make sure we destruct any PATypeHandles allocated by these
    // subclasses.  They must be manually destructed. 
    for (unsigned i = 0; i < NumContainedTys; ++i)
      ContainedTys[i].PATypeHandle::~PATypeHandle();

    // Now call the destructor for the subclass directly because we're going
    // to delete this as an array of char.
    if (isa<FunctionType>(this))
      static_cast<const FunctionType*>(this)->FunctionType::~FunctionType();
    else
      static_cast<const StructType*>(this)->StructType::~StructType();

    // Finally, remove the memory as an array deallocation of the chars it was
    // constructed from.
    operator delete(const_cast<Type *>(this));

    return;
  }

  // For all the other type subclasses, there is either no contained types or 
  // just one (all Sequentials). For Sequentials, the PATypeHandle is not
  // allocated past the type object, its included directly in the SequentialType
  // class. This means we can safely just do "normal" delete of this object and
  // all the destructors that need to run will be run.
  delete this; 
}

const Type *Type::getPrimitiveType(TypeID IDNumber) {
  switch (IDNumber) {
  case VoidTyID      : return VoidTy;
  case FloatTyID     : return FloatTy;
  case DoubleTyID    : return DoubleTy;
  case X86_FP80TyID  : return X86_FP80Ty;
  case FP128TyID     : return FP128Ty;
  case PPC_FP128TyID : return PPC_FP128Ty;
  case LabelTyID     : return LabelTy;
  default:
    return 0;
  }
}

const Type *Type::getVAArgsPromotedType() const {
  if (ID == IntegerTyID && getSubclassData() < 32)
    return Type::Int32Ty;
  else if (ID == FloatTyID)
    return Type::DoubleTy;
  else
    return this;
}

/// isIntOrIntVector - Return true if this is an integer type or a vector of
/// integer types.
///
bool Type::isIntOrIntVector() const {
  if (isInteger())
    return true;
  if (ID != Type::VectorTyID) return false;
  
  return cast<VectorType>(this)->getElementType()->isInteger();
}

/// isFPOrFPVector - Return true if this is a FP type or a vector of FP types.
///
bool Type::isFPOrFPVector() const {
  if (ID == Type::FloatTyID || ID == Type::DoubleTyID || 
      ID == Type::FP128TyID || ID == Type::X86_FP80TyID || 
      ID == Type::PPC_FP128TyID)
    return true;
  if (ID != Type::VectorTyID) return false;
  
  return cast<VectorType>(this)->getElementType()->isFloatingPoint();
}

// canLosslesllyBitCastTo - Return true if this type can be converted to
// 'Ty' without any reinterpretation of bits.  For example, uint to int.
//
bool Type::canLosslesslyBitCastTo(const Type *Ty) const {
  // Identity cast means no change so return true
  if (this == Ty) 
    return true;
  
  // They are not convertible unless they are at least first class types
  if (!this->isFirstClassType() || !Ty->isFirstClassType())
    return false;

  // Vector -> Vector conversions are always lossless if the two vector types
  // have the same size, otherwise not.
  if (const VectorType *thisPTy = dyn_cast<VectorType>(this))
    if (const VectorType *thatPTy = dyn_cast<VectorType>(Ty))
      return thisPTy->getBitWidth() == thatPTy->getBitWidth();

  // At this point we have only various mismatches of the first class types
  // remaining and ptr->ptr. Just select the lossless conversions. Everything
  // else is not lossless.
  if (isa<PointerType>(this))
    return isa<PointerType>(Ty);
  return false;  // Other types have no identity values
}

unsigned Type::getPrimitiveSizeInBits() const {
  switch (getTypeID()) {
  case Type::FloatTyID: return 32;
  case Type::DoubleTyID: return 64;
  case Type::X86_FP80TyID: return 80;
  case Type::FP128TyID: return 128;
  case Type::PPC_FP128TyID: return 128;
  case Type::IntegerTyID: return cast<IntegerType>(this)->getBitWidth();
  case Type::VectorTyID:  return cast<VectorType>(this)->getBitWidth();
  default: return 0;
  }
}

/// isSizedDerivedType - Derived types like structures and arrays are sized
/// iff all of the members of the type are sized as well.  Since asking for
/// their size is relatively uncommon, move this operation out of line.
bool Type::isSizedDerivedType() const {
  if (isa<IntegerType>(this))
    return true;

  if (const ArrayType *ATy = dyn_cast<ArrayType>(this))
    return ATy->getElementType()->isSized();

  if (const VectorType *PTy = dyn_cast<VectorType>(this))
    return PTy->getElementType()->isSized();

  if (!isa<StructType>(this)) 
    return false;

  // Okay, our struct is sized if all of the elements are...
  for (subtype_iterator I = subtype_begin(), E = subtype_end(); I != E; ++I)
    if (!(*I)->isSized()) 
      return false;

  return true;
}

/// getForwardedTypeInternal - This method is used to implement the union-find
/// algorithm for when a type is being forwarded to another type.
const Type *Type::getForwardedTypeInternal() const {
  assert(ForwardType && "This type is not being forwarded to another type!");

  // Check to see if the forwarded type has been forwarded on.  If so, collapse
  // the forwarding links.
  const Type *RealForwardedType = ForwardType->getForwardedType();
  if (!RealForwardedType)
    return ForwardType;  // No it's not forwarded again

  // Yes, it is forwarded again.  First thing, add the reference to the new
  // forward type.
  if (RealForwardedType->isAbstract())
    cast<DerivedType>(RealForwardedType)->addRef();

  // Now drop the old reference.  This could cause ForwardType to get deleted.
  cast<DerivedType>(ForwardType)->dropRef();

  // Return the updated type.
  ForwardType = RealForwardedType;
  return ForwardType;
}

void Type::refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
  abort();
}
void Type::typeBecameConcrete(const DerivedType *AbsTy) {
  abort();
}


// getTypeDescription - This is a recursive function that walks a type hierarchy
// calculating the description for a type.
//
static std::string getTypeDescription(const Type *Ty,
                                      std::vector<const Type *> &TypeStack) {
  if (isa<OpaqueType>(Ty)) {                     // Base case for the recursion
    std::map<const Type*, std::string>::iterator I =
      AbstractTypeDescriptions->find(Ty);
    if (I != AbstractTypeDescriptions->end())
      return I->second;
    std::string Desc = "opaque";
    AbstractTypeDescriptions->insert(std::make_pair(Ty, Desc));
    return Desc;
  }

  if (!Ty->isAbstract()) {                       // Base case for the recursion
    std::map<const Type*, std::string>::iterator I =
      ConcreteTypeDescriptions->find(Ty);
    if (I != ConcreteTypeDescriptions->end()) 
      return I->second;
    
    if (Ty->isPrimitiveType()) {
      switch (Ty->getTypeID()) {
      default: assert(0 && "Unknown prim type!");
      case Type::VoidTyID:   return (*ConcreteTypeDescriptions)[Ty] = "void";
      case Type::FloatTyID:  return (*ConcreteTypeDescriptions)[Ty] = "float";
      case Type::DoubleTyID: return (*ConcreteTypeDescriptions)[Ty] = "double";
      case Type::X86_FP80TyID: 
            return (*ConcreteTypeDescriptions)[Ty] = "x86_fp80";
      case Type::FP128TyID: return (*ConcreteTypeDescriptions)[Ty] = "fp128";
      case Type::PPC_FP128TyID: 
          return (*ConcreteTypeDescriptions)[Ty] = "ppc_fp128";
      case Type::LabelTyID:  return (*ConcreteTypeDescriptions)[Ty] = "label";
      }
    }
  }

  // Check to see if the Type is already on the stack...
  unsigned Slot = 0, CurSize = TypeStack.size();
  while (Slot < CurSize && TypeStack[Slot] != Ty) ++Slot; // Scan for type

  // This is another base case for the recursion.  In this case, we know
  // that we have looped back to a type that we have previously visited.
  // Generate the appropriate upreference to handle this.
  //
  if (Slot < CurSize)
    return "\\" + utostr(CurSize-Slot);         // Here's the upreference

  // Recursive case: derived types...
  std::string Result;
  TypeStack.push_back(Ty);    // Add us to the stack..

  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    const IntegerType *ITy = cast<IntegerType>(Ty);
    Result = "i" + utostr(ITy->getBitWidth());
    break;
  }
  case Type::FunctionTyID: {
    const FunctionType *FTy = cast<FunctionType>(Ty);
    if (!Result.empty())
      Result += " ";
    Result += getTypeDescription(FTy->getReturnType(), TypeStack) + " (";
    for (FunctionType::param_iterator I = FTy->param_begin(),
         E = FTy->param_end(); I != E; ++I) {
      if (I != FTy->param_begin())
        Result += ", ";
      Result += getTypeDescription(*I, TypeStack);
    }
    if (FTy->isVarArg()) {
      if (FTy->getNumParams()) Result += ", ";
      Result += "...";
    }
    Result += ")";
    break;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<StructType>(Ty);
    if (STy->isPacked())
      Result = "<{ ";
    else
      Result = "{ ";
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      if (I != STy->element_begin())
        Result += ", ";
      Result += getTypeDescription(*I, TypeStack);
    }
    Result += " }";
    if (STy->isPacked())
      Result += ">";
    break;
  }
  case Type::PointerTyID: {
    const PointerType *PTy = cast<PointerType>(Ty);
    Result = getTypeDescription(PTy->getElementType(), TypeStack);
    if (unsigned AddressSpace = PTy->getAddressSpace())
      Result += " addrspace(" + utostr(AddressSpace) + ")";
    Result += " *";
    break;
  }
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    unsigned NumElements = ATy->getNumElements();
    Result = "[";
    Result += utostr(NumElements) + " x ";
    Result += getTypeDescription(ATy->getElementType(), TypeStack) + "]";
    break;
  }
  case Type::VectorTyID: {
    const VectorType *PTy = cast<VectorType>(Ty);
    unsigned NumElements = PTy->getNumElements();
    Result = "<";
    Result += utostr(NumElements) + " x ";
    Result += getTypeDescription(PTy->getElementType(), TypeStack) + ">";
    break;
  }
  default:
    Result = "<error>";
    assert(0 && "Unhandled type in getTypeDescription!");
  }

  TypeStack.pop_back();       // Remove self from stack...

  return Result;
}



static const std::string &getOrCreateDesc(std::map<const Type*,std::string>&Map,
                                          const Type *Ty) {
  std::map<const Type*, std::string>::iterator I = Map.find(Ty);
  if (I != Map.end()) return I->second;

  std::vector<const Type *> TypeStack;
  std::string Result = getTypeDescription(Ty, TypeStack);
  return Map[Ty] = Result;
}


const std::string &Type::getDescription() const {
  if (isAbstract())
    return getOrCreateDesc(*AbstractTypeDescriptions, this);
  else
    return getOrCreateDesc(*ConcreteTypeDescriptions, this);
}


bool StructType::indexValid(const Value *V) const {
  // Structure indexes require 32-bit integer constants.
  if (V->getType() == Type::Int32Ty)
    if (const ConstantInt *CU = dyn_cast<ConstantInt>(V))
      return indexValid(CU->getZExtValue());
  return false;
}

bool StructType::indexValid(unsigned V) const {
  return V < NumContainedTys;
}

// getTypeAtIndex - Given an index value into the type, return the type of the
// element.  For a structure type, this must be a constant value...
//
const Type *StructType::getTypeAtIndex(const Value *V) const {
  unsigned Idx = (unsigned)cast<ConstantInt>(V)->getZExtValue();
  return getTypeAtIndex(Idx);
}

const Type *StructType::getTypeAtIndex(unsigned Idx) const {
  assert(indexValid(Idx) && "Invalid structure index!");
  return ContainedTys[Idx];
}

//===----------------------------------------------------------------------===//
//                          Primitive 'Type' data
//===----------------------------------------------------------------------===//

const Type *Type::VoidTy       = new Type(Type::VoidTyID);
const Type *Type::FloatTy      = new Type(Type::FloatTyID);
const Type *Type::DoubleTy     = new Type(Type::DoubleTyID);
const Type *Type::X86_FP80Ty   = new Type(Type::X86_FP80TyID);
const Type *Type::FP128Ty      = new Type(Type::FP128TyID);
const Type *Type::PPC_FP128Ty  = new Type(Type::PPC_FP128TyID);
const Type *Type::LabelTy      = new Type(Type::LabelTyID);

namespace {
  struct BuiltinIntegerType : public IntegerType {
    explicit BuiltinIntegerType(unsigned W) : IntegerType(W) {}
  };
}
const IntegerType *Type::Int1Ty  = new BuiltinIntegerType(1);
const IntegerType *Type::Int8Ty  = new BuiltinIntegerType(8);
const IntegerType *Type::Int16Ty = new BuiltinIntegerType(16);
const IntegerType *Type::Int32Ty = new BuiltinIntegerType(32);
const IntegerType *Type::Int64Ty = new BuiltinIntegerType(64);


//===----------------------------------------------------------------------===//
//                          Derived Type Constructors
//===----------------------------------------------------------------------===//

/// isValidReturnType - Return true if the specified type is valid as a return
/// type.
bool FunctionType::isValidReturnType(const Type *RetTy) {
  if (RetTy->isFirstClassType())
    return true;
  if (RetTy == Type::VoidTy || isa<OpaqueType>(RetTy))
    return true;
  
  // If this is a multiple return case, verify that each return is a first class
  // value and that there is at least one value.
  const StructType *SRetTy = dyn_cast<StructType>(RetTy);
  if (SRetTy == 0 || SRetTy->getNumElements() == 0)
    return false;
  
  for (unsigned i = 0, e = SRetTy->getNumElements(); i != e; ++i)
    if (!SRetTy->getElementType(i)->isFirstClassType())
      return false;
  return true;
}

FunctionType::FunctionType(const Type *Result,
                           const std::vector<const Type*> &Params,
                           bool IsVarArgs)
  : DerivedType(FunctionTyID), isVarArgs(IsVarArgs) {
  ContainedTys = reinterpret_cast<PATypeHandle*>(this+1);
  NumContainedTys = Params.size() + 1; // + 1 for result type
  assert(isValidReturnType(Result) && "invalid return type for function");
    
    
  bool isAbstract = Result->isAbstract();
  new (&ContainedTys[0]) PATypeHandle(Result, this);

  for (unsigned i = 0; i != Params.size(); ++i) {
    assert((Params[i]->isFirstClassType() || isa<OpaqueType>(Params[i])) &&
           "Function arguments must be value types!");
    new (&ContainedTys[i+1]) PATypeHandle(Params[i],this);
    isAbstract |= Params[i]->isAbstract();
  }

  // Calculate whether or not this type is abstract
  setAbstract(isAbstract);
}

StructType::StructType(const std::vector<const Type*> &Types, bool isPacked)
  : CompositeType(StructTyID) {
  ContainedTys = reinterpret_cast<PATypeHandle*>(this + 1);
  NumContainedTys = Types.size();
  setSubclassData(isPacked);
  bool isAbstract = false;
  for (unsigned i = 0; i < Types.size(); ++i) {
    assert(Types[i] != Type::VoidTy && "Void type for structure field!!");
     new (&ContainedTys[i]) PATypeHandle(Types[i], this);
    isAbstract |= Types[i]->isAbstract();
  }

  // Calculate whether or not this type is abstract
  setAbstract(isAbstract);
}

ArrayType::ArrayType(const Type *ElType, uint64_t NumEl)
  : SequentialType(ArrayTyID, ElType) {
  NumElements = NumEl;

  // Calculate whether or not this type is abstract
  setAbstract(ElType->isAbstract());
}

VectorType::VectorType(const Type *ElType, unsigned NumEl)
  : SequentialType(VectorTyID, ElType) {
  NumElements = NumEl;
  setAbstract(ElType->isAbstract());
  assert(NumEl > 0 && "NumEl of a VectorType must be greater than 0");
  assert((ElType->isInteger() || ElType->isFloatingPoint() || 
          isa<OpaqueType>(ElType)) && 
         "Elements of a VectorType must be a primitive type");

}


PointerType::PointerType(const Type *E, unsigned AddrSpace)
  : SequentialType(PointerTyID, E) {
  AddressSpace = AddrSpace;
  // Calculate whether or not this type is abstract
  setAbstract(E->isAbstract());
}

OpaqueType::OpaqueType() : DerivedType(OpaqueTyID) {
  setAbstract(true);
#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *this << "\n";
#endif
}

// dropAllTypeUses - When this (abstract) type is resolved to be equal to
// another (more concrete) type, we must eliminate all references to other
// types, to avoid some circular reference problems.
void DerivedType::dropAllTypeUses() {
  if (NumContainedTys != 0) {
    // The type must stay abstract.  To do this, we insert a pointer to a type
    // that will never get resolved, thus will always be abstract.
    static Type *AlwaysOpaqueTy = OpaqueType::get();
    static PATypeHolder Holder(AlwaysOpaqueTy);
    ContainedTys[0] = AlwaysOpaqueTy;

    // Change the rest of the types to be Int32Ty's.  It doesn't matter what we
    // pick so long as it doesn't point back to this type.  We choose something
    // concrete to avoid overhead for adding to AbstracTypeUser lists and stuff.
    for (unsigned i = 1, e = NumContainedTys; i != e; ++i)
      ContainedTys[i] = Type::Int32Ty;
  }
}


namespace {

/// TypePromotionGraph and graph traits - this is designed to allow us to do
/// efficient SCC processing of type graphs.  This is the exact same as
/// GraphTraits<Type*>, except that we pretend that concrete types have no
/// children to avoid processing them.
struct TypePromotionGraph {
  Type *Ty;
  TypePromotionGraph(Type *T) : Ty(T) {}
};

}

namespace llvm {
  template <> struct GraphTraits<TypePromotionGraph> {
    typedef Type NodeType;
    typedef Type::subtype_iterator ChildIteratorType;

    static inline NodeType *getEntryNode(TypePromotionGraph G) { return G.Ty; }
    static inline ChildIteratorType child_begin(NodeType *N) {
      if (N->isAbstract())
        return N->subtype_begin();
      else           // No need to process children of concrete types.
        return N->subtype_end();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return N->subtype_end();
    }
  };
}


// PromoteAbstractToConcrete - This is a recursive function that walks a type
// graph calculating whether or not a type is abstract.
//
void Type::PromoteAbstractToConcrete() {
  if (!isAbstract()) return;

  scc_iterator<TypePromotionGraph> SI = scc_begin(TypePromotionGraph(this));
  scc_iterator<TypePromotionGraph> SE = scc_end  (TypePromotionGraph(this));

  for (; SI != SE; ++SI) {
    std::vector<Type*> &SCC = *SI;

    // Concrete types are leaves in the tree.  Since an SCC will either be all
    // abstract or all concrete, we only need to check one type.
    if (SCC[0]->isAbstract()) {
      if (isa<OpaqueType>(SCC[0]))
        return;     // Not going to be concrete, sorry.

      // If all of the children of all of the types in this SCC are concrete,
      // then this SCC is now concrete as well.  If not, neither this SCC, nor
      // any parent SCCs will be concrete, so we might as well just exit.
      for (unsigned i = 0, e = SCC.size(); i != e; ++i)
        for (Type::subtype_iterator CI = SCC[i]->subtype_begin(),
               E = SCC[i]->subtype_end(); CI != E; ++CI)
          if ((*CI)->isAbstract())
            // If the child type is in our SCC, it doesn't make the entire SCC
            // abstract unless there is a non-SCC abstract type.
            if (std::find(SCC.begin(), SCC.end(), *CI) == SCC.end())
              return;               // Not going to be concrete, sorry.

      // Okay, we just discovered this whole SCC is now concrete, mark it as
      // such!
      for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
        assert(SCC[i]->isAbstract() && "Why are we processing concrete types?");

        SCC[i]->setAbstract(false);
      }

      for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
        assert(!SCC[i]->isAbstract() && "Concrete type became abstract?");
        // The type just became concrete, notify all users!
        cast<DerivedType>(SCC[i])->notifyUsesThatTypeBecameConcrete();
      }
    }
  }
}


//===----------------------------------------------------------------------===//
//                      Type Structural Equality Testing
//===----------------------------------------------------------------------===//

// TypesEqual - Two types are considered structurally equal if they have the
// same "shape": Every level and element of the types have identical primitive
// ID's, and the graphs have the same edges/nodes in them.  Nodes do not have to
// be pointer equals to be equivalent though.  This uses an optimistic algorithm
// that assumes that two graphs are the same until proven otherwise.
//
static bool TypesEqual(const Type *Ty, const Type *Ty2,
                       std::map<const Type *, const Type *> &EqTypes) {
  if (Ty == Ty2) return true;
  if (Ty->getTypeID() != Ty2->getTypeID()) return false;
  if (isa<OpaqueType>(Ty))
    return false;  // Two unequal opaque types are never equal

  std::map<const Type*, const Type*>::iterator It = EqTypes.find(Ty);
  if (It != EqTypes.end())
    return It->second == Ty2;    // Looping back on a type, check for equality

  // Otherwise, add the mapping to the table to make sure we don't get
  // recursion on the types...
  EqTypes.insert(It, std::make_pair(Ty, Ty2));

  // Two really annoying special cases that breaks an otherwise nice simple
  // algorithm is the fact that arraytypes have sizes that differentiates types,
  // and that function types can be varargs or not.  Consider this now.
  //
  if (const IntegerType *ITy = dyn_cast<IntegerType>(Ty)) {
    const IntegerType *ITy2 = cast<IntegerType>(Ty2);
    return ITy->getBitWidth() == ITy2->getBitWidth();
  } else if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
    const PointerType *PTy2 = cast<PointerType>(Ty2);
    return PTy->getAddressSpace() == PTy2->getAddressSpace() &&
           TypesEqual(PTy->getElementType(), PTy2->getElementType(), EqTypes);
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    const StructType *STy2 = cast<StructType>(Ty2);
    if (STy->getNumElements() != STy2->getNumElements()) return false;
    if (STy->isPacked() != STy2->isPacked()) return false;
    for (unsigned i = 0, e = STy2->getNumElements(); i != e; ++i)
      if (!TypesEqual(STy->getElementType(i), STy2->getElementType(i), EqTypes))
        return false;
    return true;
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    const ArrayType *ATy2 = cast<ArrayType>(Ty2);
    return ATy->getNumElements() == ATy2->getNumElements() &&
           TypesEqual(ATy->getElementType(), ATy2->getElementType(), EqTypes);
  } else if (const VectorType *PTy = dyn_cast<VectorType>(Ty)) {
    const VectorType *PTy2 = cast<VectorType>(Ty2);
    return PTy->getNumElements() == PTy2->getNumElements() &&
           TypesEqual(PTy->getElementType(), PTy2->getElementType(), EqTypes);
  } else if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
    const FunctionType *FTy2 = cast<FunctionType>(Ty2);
    if (FTy->isVarArg() != FTy2->isVarArg() ||
        FTy->getNumParams() != FTy2->getNumParams() ||
        !TypesEqual(FTy->getReturnType(), FTy2->getReturnType(), EqTypes))
      return false;
    for (unsigned i = 0, e = FTy2->getNumParams(); i != e; ++i) {
      if (!TypesEqual(FTy->getParamType(i), FTy2->getParamType(i), EqTypes))
        return false;
    }
    return true;
  } else {
    assert(0 && "Unknown derived type!");
    return false;
  }
}

static bool TypesEqual(const Type *Ty, const Type *Ty2) {
  std::map<const Type *, const Type *> EqTypes;
  return TypesEqual(Ty, Ty2, EqTypes);
}

// AbstractTypeHasCycleThrough - Return true there is a path from CurTy to
// TargetTy in the type graph.  We know that Ty is an abstract type, so if we
// ever reach a non-abstract type, we know that we don't need to search the
// subgraph.
static bool AbstractTypeHasCycleThrough(const Type *TargetTy, const Type *CurTy,
                                SmallPtrSet<const Type*, 128> &VisitedTypes) {
  if (TargetTy == CurTy) return true;
  if (!CurTy->isAbstract()) return false;

  if (!VisitedTypes.insert(CurTy))
    return false;  // Already been here.

  for (Type::subtype_iterator I = CurTy->subtype_begin(),
       E = CurTy->subtype_end(); I != E; ++I)
    if (AbstractTypeHasCycleThrough(TargetTy, *I, VisitedTypes))
      return true;
  return false;
}

static bool ConcreteTypeHasCycleThrough(const Type *TargetTy, const Type *CurTy,
                                SmallPtrSet<const Type*, 128> &VisitedTypes) {
  if (TargetTy == CurTy) return true;

  if (!VisitedTypes.insert(CurTy))
    return false;  // Already been here.

  for (Type::subtype_iterator I = CurTy->subtype_begin(),
       E = CurTy->subtype_end(); I != E; ++I)
    if (ConcreteTypeHasCycleThrough(TargetTy, *I, VisitedTypes))
      return true;
  return false;
}

/// TypeHasCycleThroughItself - Return true if the specified type has a cycle
/// back to itself.
static bool TypeHasCycleThroughItself(const Type *Ty) {
  SmallPtrSet<const Type*, 128> VisitedTypes;

  if (Ty->isAbstract()) {  // Optimized case for abstract types.
    for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
         I != E; ++I)
      if (AbstractTypeHasCycleThrough(Ty, *I, VisitedTypes))
        return true;
  } else {
    for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
         I != E; ++I)
      if (ConcreteTypeHasCycleThrough(Ty, *I, VisitedTypes))
        return true;
  }
  return false;
}

/// getSubElementHash - Generate a hash value for all of the SubType's of this
/// type.  The hash value is guaranteed to be zero if any of the subtypes are 
/// an opaque type.  Otherwise we try to mix them in as well as possible, but do
/// not look at the subtype's subtype's.
static unsigned getSubElementHash(const Type *Ty) {
  unsigned HashVal = 0;
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I) {
    HashVal *= 32;
    const Type *SubTy = I->get();
    HashVal += SubTy->getTypeID();
    switch (SubTy->getTypeID()) {
    default: break;
    case Type::OpaqueTyID: return 0;    // Opaque -> hash = 0 no matter what.
    case Type::IntegerTyID:
      HashVal ^= (cast<IntegerType>(SubTy)->getBitWidth() << 3);
      break;
    case Type::FunctionTyID:
      HashVal ^= cast<FunctionType>(SubTy)->getNumParams()*2 + 
                 cast<FunctionType>(SubTy)->isVarArg();
      break;
    case Type::ArrayTyID:
      HashVal ^= cast<ArrayType>(SubTy)->getNumElements();
      break;
    case Type::VectorTyID:
      HashVal ^= cast<VectorType>(SubTy)->getNumElements();
      break;
    case Type::StructTyID:
      HashVal ^= cast<StructType>(SubTy)->getNumElements();
      break;
    case Type::PointerTyID:
      HashVal ^= cast<PointerType>(SubTy)->getAddressSpace();
      break;
    }
  }
  return HashVal ? HashVal : 1;  // Do not return zero unless opaque subty.
}

//===----------------------------------------------------------------------===//
//                       Derived Type Factory Functions
//===----------------------------------------------------------------------===//

namespace llvm {
class TypeMapBase {
protected:
  /// TypesByHash - Keep track of types by their structure hash value.  Note
  /// that we only keep track of types that have cycles through themselves in
  /// this map.
  ///
  std::multimap<unsigned, PATypeHolder> TypesByHash;

public:
  void RemoveFromTypesByHash(unsigned Hash, const Type *Ty) {
    std::multimap<unsigned, PATypeHolder>::iterator I =
      TypesByHash.lower_bound(Hash);
    for (; I != TypesByHash.end() && I->first == Hash; ++I) {
      if (I->second == Ty) {
        TypesByHash.erase(I);
        return;
      }
    }
    
    // This must be do to an opaque type that was resolved.  Switch down to hash
    // code of zero.
    assert(Hash && "Didn't find type entry!");
    RemoveFromTypesByHash(0, Ty);
  }
  
  /// TypeBecameConcrete - When Ty gets a notification that TheType just became
  /// concrete, drop uses and make Ty non-abstract if we should.
  void TypeBecameConcrete(DerivedType *Ty, const DerivedType *TheType) {
    // If the element just became concrete, remove 'ty' from the abstract
    // type user list for the type.  Do this for as many times as Ty uses
    // OldType.
    for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
         I != E; ++I)
      if (I->get() == TheType)
        TheType->removeAbstractTypeUser(Ty);
    
    // If the type is currently thought to be abstract, rescan all of our
    // subtypes to see if the type has just become concrete!  Note that this
    // may send out notifications to AbstractTypeUsers that types become
    // concrete.
    if (Ty->isAbstract())
      Ty->PromoteAbstractToConcrete();
  }
};
}


// TypeMap - Make sure that only one instance of a particular type may be
// created on any given run of the compiler... note that this involves updating
// our map if an abstract type gets refined somehow.
//
namespace llvm {
template<class ValType, class TypeClass>
class TypeMap : public TypeMapBase {
  std::map<ValType, PATypeHolder> Map;
public:
  typedef typename std::map<ValType, PATypeHolder>::iterator iterator;
  ~TypeMap() { print("ON EXIT"); }

  inline TypeClass *get(const ValType &V) {
    iterator I = Map.find(V);
    return I != Map.end() ? cast<TypeClass>((Type*)I->second.get()) : 0;
  }

  inline void add(const ValType &V, TypeClass *Ty) {
    Map.insert(std::make_pair(V, Ty));

    // If this type has a cycle, remember it.
    TypesByHash.insert(std::make_pair(ValType::hashTypeStructure(Ty), Ty));
    print("add");
  }
  
  /// RefineAbstractType - This method is called after we have merged a type
  /// with another one.  We must now either merge the type away with
  /// some other type or reinstall it in the map with it's new configuration.
  void RefineAbstractType(TypeClass *Ty, const DerivedType *OldType,
                        const Type *NewType) {
#ifdef DEBUG_MERGE_TYPES
    DOUT << "RefineAbstractType(" << (void*)OldType << "[" << *OldType
         << "], " << (void*)NewType << " [" << *NewType << "])\n";
#endif
    
    // Otherwise, we are changing one subelement type into another.  Clearly the
    // OldType must have been abstract, making us abstract.
    assert(Ty->isAbstract() && "Refining a non-abstract type!");
    assert(OldType != NewType);

    // Make a temporary type holder for the type so that it doesn't disappear on
    // us when we erase the entry from the map.
    PATypeHolder TyHolder = Ty;

    // The old record is now out-of-date, because one of the children has been
    // updated.  Remove the obsolete entry from the map.
    unsigned NumErased = Map.erase(ValType::get(Ty));
    assert(NumErased && "Element not found!"); NumErased = NumErased;

    // Remember the structural hash for the type before we start hacking on it,
    // in case we need it later.
    unsigned OldTypeHash = ValType::hashTypeStructure(Ty);

    // Find the type element we are refining... and change it now!
    for (unsigned i = 0, e = Ty->getNumContainedTypes(); i != e; ++i)
      if (Ty->ContainedTys[i] == OldType)
        Ty->ContainedTys[i] = NewType;
    unsigned NewTypeHash = ValType::hashTypeStructure(Ty);
    
    // If there are no cycles going through this node, we can do a simple,
    // efficient lookup in the map, instead of an inefficient nasty linear
    // lookup.
    if (!TypeHasCycleThroughItself(Ty)) {
      typename std::map<ValType, PATypeHolder>::iterator I;
      bool Inserted;

      tie(I, Inserted) = Map.insert(std::make_pair(ValType::get(Ty), Ty));
      if (!Inserted) {
        // Refined to a different type altogether?
        RemoveFromTypesByHash(OldTypeHash, Ty);

        // We already have this type in the table.  Get rid of the newly refined
        // type.
        TypeClass *NewTy = cast<TypeClass>((Type*)I->second.get());
        Ty->refineAbstractTypeTo(NewTy);
        return;
      }
    } else {
      // Now we check to see if there is an existing entry in the table which is
      // structurally identical to the newly refined type.  If so, this type
      // gets refined to the pre-existing type.
      //
      std::multimap<unsigned, PATypeHolder>::iterator I, E, Entry;
      tie(I, E) = TypesByHash.equal_range(NewTypeHash);
      Entry = E;
      for (; I != E; ++I) {
        if (I->second == Ty) {
          // Remember the position of the old type if we see it in our scan.
          Entry = I;
        } else {
          if (TypesEqual(Ty, I->second)) {
            TypeClass *NewTy = cast<TypeClass>((Type*)I->second.get());

            // Remove the old entry form TypesByHash.  If the hash values differ
            // now, remove it from the old place.  Otherwise, continue scanning
            // withing this hashcode to reduce work.
            if (NewTypeHash != OldTypeHash) {
              RemoveFromTypesByHash(OldTypeHash, Ty);
            } else {
              if (Entry == E) {
                // Find the location of Ty in the TypesByHash structure if we
                // haven't seen it already.
                while (I->second != Ty) {
                  ++I;
                  assert(I != E && "Structure doesn't contain type??");
                }
                Entry = I;
              }
              TypesByHash.erase(Entry);
            }
            Ty->refineAbstractTypeTo(NewTy);
            return;
          }
        }
      }

      // If there is no existing type of the same structure, we reinsert an
      // updated record into the map.
      Map.insert(std::make_pair(ValType::get(Ty), Ty));
    }

    // If the hash codes differ, update TypesByHash
    if (NewTypeHash != OldTypeHash) {
      RemoveFromTypesByHash(OldTypeHash, Ty);
      TypesByHash.insert(std::make_pair(NewTypeHash, Ty));
    }
    
    // If the type is currently thought to be abstract, rescan all of our
    // subtypes to see if the type has just become concrete!  Note that this
    // may send out notifications to AbstractTypeUsers that types become
    // concrete.
    if (Ty->isAbstract())
      Ty->PromoteAbstractToConcrete();
  }

  void print(const char *Arg) const {
#ifdef DEBUG_MERGE_TYPES
    DOUT << "TypeMap<>::" << Arg << " table contents:\n";
    unsigned i = 0;
    for (typename std::map<ValType, PATypeHolder>::const_iterator I
           = Map.begin(), E = Map.end(); I != E; ++I)
      DOUT << " " << (++i) << ". " << (void*)I->second.get() << " "
           << *I->second.get() << "\n";
#endif
  }

  void dump() const { print("dump output"); }
};
}


//===----------------------------------------------------------------------===//
// Function Type Factory and Value Class...
//

//===----------------------------------------------------------------------===//
// Integer Type Factory...
//
namespace llvm {
class IntegerValType {
  uint32_t bits;
public:
  IntegerValType(uint16_t numbits) : bits(numbits) {}

  static IntegerValType get(const IntegerType *Ty) {
    return IntegerValType(Ty->getBitWidth());
  }

  static unsigned hashTypeStructure(const IntegerType *Ty) {
    return (unsigned)Ty->getBitWidth();
  }

  inline bool operator<(const IntegerValType &IVT) const {
    return bits < IVT.bits;
  }
};
}

static ManagedStatic<TypeMap<IntegerValType, IntegerType> > IntegerTypes;

const IntegerType *IntegerType::get(unsigned NumBits) {
  assert(NumBits >= MIN_INT_BITS && "bitwidth too small");
  assert(NumBits <= MAX_INT_BITS && "bitwidth too large");

  // Check for the built-in integer types
  switch (NumBits) {
    case  1: return cast<IntegerType>(Type::Int1Ty);
    case  8: return cast<IntegerType>(Type::Int8Ty);
    case 16: return cast<IntegerType>(Type::Int16Ty);
    case 32: return cast<IntegerType>(Type::Int32Ty);
    case 64: return cast<IntegerType>(Type::Int64Ty);
    default: 
      break;
  }

  IntegerValType IVT(NumBits);
  IntegerType *ITy = IntegerTypes->get(IVT);
  if (ITy) return ITy;           // Found a match, return it!

  // Value not found.  Derive a new type!
  ITy = new IntegerType(NumBits);
  IntegerTypes->add(IVT, ITy);

#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *ITy << "\n";
#endif
  return ITy;
}

bool IntegerType::isPowerOf2ByteWidth() const {
  unsigned BitWidth = getBitWidth();
  return (BitWidth > 7) && isPowerOf2_32(BitWidth);
}

APInt IntegerType::getMask() const {
  return APInt::getAllOnesValue(getBitWidth());
}

// FunctionValType - Define a class to hold the key that goes into the TypeMap
//
namespace llvm {
class FunctionValType {
  const Type *RetTy;
  std::vector<const Type*> ArgTypes;
  bool isVarArg;
public:
  FunctionValType(const Type *ret, const std::vector<const Type*> &args,
                  bool isVA) : RetTy(ret), ArgTypes(args), isVarArg(isVA) {}

  static FunctionValType get(const FunctionType *FT);

  static unsigned hashTypeStructure(const FunctionType *FT) {
    unsigned Result = FT->getNumParams()*2 + FT->isVarArg();
    return Result;
  }

  inline bool operator<(const FunctionValType &MTV) const {
    if (RetTy < MTV.RetTy) return true;
    if (RetTy > MTV.RetTy) return false;
    if (isVarArg < MTV.isVarArg) return true;
    if (isVarArg > MTV.isVarArg) return false;
    if (ArgTypes < MTV.ArgTypes) return true;
    if (ArgTypes > MTV.ArgTypes) return false;
    return false;
  }
};
}

// Define the actual map itself now...
static ManagedStatic<TypeMap<FunctionValType, FunctionType> > FunctionTypes;

FunctionValType FunctionValType::get(const FunctionType *FT) {
  // Build up a FunctionValType
  std::vector<const Type *> ParamTypes;
  ParamTypes.reserve(FT->getNumParams());
  for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i)
    ParamTypes.push_back(FT->getParamType(i));
  return FunctionValType(FT->getReturnType(), ParamTypes, FT->isVarArg());
}


// FunctionType::get - The factory function for the FunctionType class...
FunctionType *FunctionType::get(const Type *ReturnType,
                                const std::vector<const Type*> &Params,
                                bool isVarArg) {
  FunctionValType VT(ReturnType, Params, isVarArg);
  FunctionType *FT = FunctionTypes->get(VT);
  if (FT)
    return FT;

  FT = (FunctionType*) operator new(sizeof(FunctionType) +
                                    sizeof(PATypeHandle)*(Params.size()+1));
  new (FT) FunctionType(ReturnType, Params, isVarArg);
  FunctionTypes->add(VT, FT);

#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << FT << "\n";
#endif
  return FT;
}

//===----------------------------------------------------------------------===//
// Array Type Factory...
//
namespace llvm {
class ArrayValType {
  const Type *ValTy;
  uint64_t Size;
public:
  ArrayValType(const Type *val, uint64_t sz) : ValTy(val), Size(sz) {}

  static ArrayValType get(const ArrayType *AT) {
    return ArrayValType(AT->getElementType(), AT->getNumElements());
  }

  static unsigned hashTypeStructure(const ArrayType *AT) {
    return (unsigned)AT->getNumElements();
  }

  inline bool operator<(const ArrayValType &MTV) const {
    if (Size < MTV.Size) return true;
    return Size == MTV.Size && ValTy < MTV.ValTy;
  }
};
}
static ManagedStatic<TypeMap<ArrayValType, ArrayType> > ArrayTypes;


ArrayType *ArrayType::get(const Type *ElementType, uint64_t NumElements) {
  assert(ElementType && "Can't get array of null types!");

  ArrayValType AVT(ElementType, NumElements);
  ArrayType *AT = ArrayTypes->get(AVT);
  if (AT) return AT;           // Found a match, return it!

  // Value not found.  Derive a new type!
  ArrayTypes->add(AVT, AT = new ArrayType(ElementType, NumElements));

#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *AT << "\n";
#endif
  return AT;
}


//===----------------------------------------------------------------------===//
// Vector Type Factory...
//
namespace llvm {
class VectorValType {
  const Type *ValTy;
  unsigned Size;
public:
  VectorValType(const Type *val, int sz) : ValTy(val), Size(sz) {}

  static VectorValType get(const VectorType *PT) {
    return VectorValType(PT->getElementType(), PT->getNumElements());
  }

  static unsigned hashTypeStructure(const VectorType *PT) {
    return PT->getNumElements();
  }

  inline bool operator<(const VectorValType &MTV) const {
    if (Size < MTV.Size) return true;
    return Size == MTV.Size && ValTy < MTV.ValTy;
  }
};
}
static ManagedStatic<TypeMap<VectorValType, VectorType> > VectorTypes;


VectorType *VectorType::get(const Type *ElementType, unsigned NumElements) {
  assert(ElementType && "Can't get vector of null types!");

  VectorValType PVT(ElementType, NumElements);
  VectorType *PT = VectorTypes->get(PVT);
  if (PT) return PT;           // Found a match, return it!

  // Value not found.  Derive a new type!
  VectorTypes->add(PVT, PT = new VectorType(ElementType, NumElements));

#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *PT << "\n";
#endif
  return PT;
}

//===----------------------------------------------------------------------===//
// Struct Type Factory...
//

namespace llvm {
// StructValType - Define a class to hold the key that goes into the TypeMap
//
class StructValType {
  std::vector<const Type*> ElTypes;
  bool packed;
public:
  StructValType(const std::vector<const Type*> &args, bool isPacked)
    : ElTypes(args), packed(isPacked) {}

  static StructValType get(const StructType *ST) {
    std::vector<const Type *> ElTypes;
    ElTypes.reserve(ST->getNumElements());
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i)
      ElTypes.push_back(ST->getElementType(i));

    return StructValType(ElTypes, ST->isPacked());
  }

  static unsigned hashTypeStructure(const StructType *ST) {
    return ST->getNumElements();
  }

  inline bool operator<(const StructValType &STV) const {
    if (ElTypes < STV.ElTypes) return true;
    else if (ElTypes > STV.ElTypes) return false;
    else return (int)packed < (int)STV.packed;
  }
};
}

static ManagedStatic<TypeMap<StructValType, StructType> > StructTypes;

StructType *StructType::get(const std::vector<const Type*> &ETypes, 
                            bool isPacked) {
  StructValType STV(ETypes, isPacked);
  StructType *ST = StructTypes->get(STV);
  if (ST) return ST;

  // Value not found.  Derive a new type!
  ST = (StructType*) operator new(sizeof(StructType) +
                                  sizeof(PATypeHandle) * ETypes.size());
  new (ST) StructType(ETypes, isPacked);
  StructTypes->add(STV, ST);

#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *ST << "\n";
#endif
  return ST;
}

StructType *StructType::get(const Type *type, ...) {
  va_list ap;
  std::vector<const llvm::Type*> StructFields;
  va_start(ap, type);
  while (type) {
    StructFields.push_back(type);
    type = va_arg(ap, llvm::Type*);
  }
  return llvm::StructType::get(StructFields);
}



//===----------------------------------------------------------------------===//
// Pointer Type Factory...
//

// PointerValType - Define a class to hold the key that goes into the TypeMap
//
namespace llvm {
class PointerValType {
  const Type *ValTy;
  unsigned AddressSpace;
public:
  PointerValType(const Type *val, unsigned as) : ValTy(val), AddressSpace(as) {}

  static PointerValType get(const PointerType *PT) {
    return PointerValType(PT->getElementType(), PT->getAddressSpace());
  }

  static unsigned hashTypeStructure(const PointerType *PT) {
    return getSubElementHash(PT);
  }

  bool operator<(const PointerValType &MTV) const {
    if (AddressSpace < MTV.AddressSpace) return true;
    return AddressSpace == MTV.AddressSpace && ValTy < MTV.ValTy;
  }
};
}

static ManagedStatic<TypeMap<PointerValType, PointerType> > PointerTypes;

PointerType *PointerType::get(const Type *ValueType, unsigned AddressSpace) {
  assert(ValueType && "Can't get a pointer to <null> type!");
  assert(ValueType != Type::VoidTy &&
         "Pointer to void is not valid, use sbyte* instead!");
  assert(ValueType != Type::LabelTy && "Pointer to label is not valid!");
  PointerValType PVT(ValueType, AddressSpace);

  PointerType *PT = PointerTypes->get(PVT);
  if (PT) return PT;

  // Value not found.  Derive a new type!
  PointerTypes->add(PVT, PT = new PointerType(ValueType, AddressSpace));

#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *PT << "\n";
#endif
  return PT;
}

//===----------------------------------------------------------------------===//
//                     Derived Type Refinement Functions
//===----------------------------------------------------------------------===//

// removeAbstractTypeUser - Notify an abstract type that a user of the class
// no longer has a handle to the type.  This function is called primarily by
// the PATypeHandle class.  When there are no users of the abstract type, it
// is annihilated, because there is no way to get a reference to it ever again.
//
void Type::removeAbstractTypeUser(AbstractTypeUser *U) const {
  // Search from back to front because we will notify users from back to
  // front.  Also, it is likely that there will be a stack like behavior to
  // users that register and unregister users.
  //
  unsigned i;
  for (i = AbstractTypeUsers.size(); AbstractTypeUsers[i-1] != U; --i)
    assert(i != 0 && "AbstractTypeUser not in user list!");

  --i;  // Convert to be in range 0 <= i < size()
  assert(i < AbstractTypeUsers.size() && "Index out of range!");  // Wraparound?

  AbstractTypeUsers.erase(AbstractTypeUsers.begin()+i);

#ifdef DEBUG_MERGE_TYPES
  DOUT << "  remAbstractTypeUser[" << (void*)this << ", "
       << *this << "][" << i << "] User = " << U << "\n";
#endif

  if (AbstractTypeUsers.empty() && getRefCount() == 0 && isAbstract()) {
#ifdef DEBUG_MERGE_TYPES
    DOUT << "DELETEing unused abstract type: <" << *this
         << ">[" << (void*)this << "]" << "\n";
#endif
    this->destroy();
  }
}

// refineAbstractTypeTo - This function is used when it is discovered that
// the 'this' abstract type is actually equivalent to the NewType specified.
// This causes all users of 'this' to switch to reference the more concrete type
// NewType and for 'this' to be deleted.
//
void DerivedType::refineAbstractTypeTo(const Type *NewType) {
  assert(isAbstract() && "refineAbstractTypeTo: Current type is not abstract!");
  assert(this != NewType && "Can't refine to myself!");
  assert(ForwardType == 0 && "This type has already been refined!");

  // The descriptions may be out of date.  Conservatively clear them all!
  AbstractTypeDescriptions->clear();

#ifdef DEBUG_MERGE_TYPES
  DOUT << "REFINING abstract type [" << (void*)this << " "
       << *this << "] to [" << (void*)NewType << " "
       << *NewType << "]!\n";
#endif

  // Make sure to put the type to be refined to into a holder so that if IT gets
  // refined, that we will not continue using a dead reference...
  //
  PATypeHolder NewTy(NewType);

  // Any PATypeHolders referring to this type will now automatically forward to
  // the type we are resolved to.
  ForwardType = NewType;
  if (NewType->isAbstract())
    cast<DerivedType>(NewType)->addRef();

  // Add a self use of the current type so that we don't delete ourself until
  // after the function exits.
  //
  PATypeHolder CurrentTy(this);

  // To make the situation simpler, we ask the subclass to remove this type from
  // the type map, and to replace any type uses with uses of non-abstract types.
  // This dramatically limits the amount of recursive type trouble we can find
  // ourselves in.
  dropAllTypeUses();

  // Iterate over all of the uses of this type, invoking callback.  Each user
  // should remove itself from our use list automatically.  We have to check to
  // make sure that NewTy doesn't _become_ 'this'.  If it does, resolving types
  // will not cause users to drop off of the use list.  If we resolve to ourself
  // we succeed!
  //
  while (!AbstractTypeUsers.empty() && NewTy != this) {
    AbstractTypeUser *User = AbstractTypeUsers.back();

    unsigned OldSize = AbstractTypeUsers.size(); OldSize=OldSize;
#ifdef DEBUG_MERGE_TYPES
    DOUT << " REFINING user " << OldSize-1 << "[" << (void*)User
         << "] of abstract type [" << (void*)this << " "
         << *this << "] to [" << (void*)NewTy.get() << " "
         << *NewTy << "]!\n";
#endif
    User->refineAbstractType(this, NewTy);

    assert(AbstractTypeUsers.size() != OldSize &&
           "AbsTyUser did not remove self from user list!");
  }

  // If we were successful removing all users from the type, 'this' will be
  // deleted when the last PATypeHolder is destroyed or updated from this type.
  // This may occur on exit of this function, as the CurrentTy object is
  // destroyed.
}

// notifyUsesThatTypeBecameConcrete - Notify AbstractTypeUsers of this type that
// the current type has transitioned from being abstract to being concrete.
//
void DerivedType::notifyUsesThatTypeBecameConcrete() {
#ifdef DEBUG_MERGE_TYPES
  DOUT << "typeIsREFINED type: " << (void*)this << " " << *this << "\n";
#endif

  unsigned OldSize = AbstractTypeUsers.size(); OldSize=OldSize;
  while (!AbstractTypeUsers.empty()) {
    AbstractTypeUser *ATU = AbstractTypeUsers.back();
    ATU->typeBecameConcrete(this);

    assert(AbstractTypeUsers.size() < OldSize-- &&
           "AbstractTypeUser did not remove itself from the use list!");
  }
}

// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void FunctionType::refineAbstractType(const DerivedType *OldType,
                                      const Type *NewType) {
  FunctionTypes->RefineAbstractType(this, OldType, NewType);
}

void FunctionType::typeBecameConcrete(const DerivedType *AbsTy) {
  FunctionTypes->TypeBecameConcrete(this, AbsTy);
}


// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void ArrayType::refineAbstractType(const DerivedType *OldType,
                                   const Type *NewType) {
  ArrayTypes->RefineAbstractType(this, OldType, NewType);
}

void ArrayType::typeBecameConcrete(const DerivedType *AbsTy) {
  ArrayTypes->TypeBecameConcrete(this, AbsTy);
}

// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void VectorType::refineAbstractType(const DerivedType *OldType,
                                   const Type *NewType) {
  VectorTypes->RefineAbstractType(this, OldType, NewType);
}

void VectorType::typeBecameConcrete(const DerivedType *AbsTy) {
  VectorTypes->TypeBecameConcrete(this, AbsTy);
}

// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void StructType::refineAbstractType(const DerivedType *OldType,
                                    const Type *NewType) {
  StructTypes->RefineAbstractType(this, OldType, NewType);
}

void StructType::typeBecameConcrete(const DerivedType *AbsTy) {
  StructTypes->TypeBecameConcrete(this, AbsTy);
}

// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void PointerType::refineAbstractType(const DerivedType *OldType,
                                     const Type *NewType) {
  PointerTypes->RefineAbstractType(this, OldType, NewType);
}

void PointerType::typeBecameConcrete(const DerivedType *AbsTy) {
  PointerTypes->TypeBecameConcrete(this, AbsTy);
}

bool SequentialType::indexValid(const Value *V) const {
  if (const IntegerType *IT = dyn_cast<IntegerType>(V->getType())) 
    return IT->getBitWidth() == 32 || IT->getBitWidth() == 64;
  return false;
}

namespace llvm {
std::ostream &operator<<(std::ostream &OS, const Type *T) {
  if (T == 0)
    OS << "<null> value!\n";
  else
    T->print(OS);
  return OS;
}

std::ostream &operator<<(std::ostream &OS, const Type &T) {
  T.print(OS);
  return OS;
}
}
