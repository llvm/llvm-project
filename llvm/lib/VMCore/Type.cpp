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

#include "LLVMContextImpl.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Mutex.h"
#include "llvm/System/RWMutex.h"
#include "llvm/System/Threading.h"
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

const Type *Type::getPrimitiveType(LLVMContext &C, TypeID IDNumber) {
  switch (IDNumber) {
  case VoidTyID      : return getVoidTy(C);
  case FloatTyID     : return getFloatTy(C);
  case DoubleTyID    : return getDoubleTy(C);
  case X86_FP80TyID  : return getX86_FP80Ty(C);
  case FP128TyID     : return getFP128Ty(C);
  case PPC_FP128TyID : return getPPC_FP128Ty(C);
  case LabelTyID     : return getLabelTy(C);
  case MetadataTyID  : return getMetadataTy(C);
  default:
    return 0;
  }
}

const Type *Type::getVAArgsPromotedType(LLVMContext &C) const {
  if (ID == IntegerTyID && getSubclassData() < 32)
    return Type::getInt32Ty(C);
  else if (ID == FloatTyID)
    return Type::getDoubleTy(C);
  else
    return this;
}

/// getScalarType - If this is a vector type, return the element type,
/// otherwise return this.
const Type *Type::getScalarType() const {
  if (const VectorType *VTy = dyn_cast<VectorType>(this))
    return VTy->getElementType();
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

// canLosslesslyBitCastTo - Return true if this type can be converted to
// 'Ty' without any reinterpretation of bits.  For example, i8* to i32*.
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

/// getScalarSizeInBits - If this is a vector type, return the
/// getPrimitiveSizeInBits value for the element type. Otherwise return the
/// getPrimitiveSizeInBits value for this type.
unsigned Type::getScalarSizeInBits() const {
  return getScalarType()->getPrimitiveSizeInBits();
}

/// getFPMantissaWidth - Return the width of the mantissa of this type.  This
/// is only valid on floating point types.  If the FP type does not
/// have a stable mantissa (e.g. ppc long double), this method returns -1.
int Type::getFPMantissaWidth() const {
  if (const VectorType *VTy = dyn_cast<VectorType>(this))
    return VTy->getElementType()->getFPMantissaWidth();
  assert(isFloatingPoint() && "Not a floating point type!");
  if (ID == FloatTyID) return 24;
  if (ID == DoubleTyID) return 53;
  if (ID == X86_FP80TyID) return 64;
  if (ID == FP128TyID) return 113;
  assert(ID == PPC_FP128TyID && "unknown fp type");
  return -1;
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
  llvm_unreachable("Attempting to refine a derived type!");
}
void Type::typeBecameConcrete(const DerivedType *AbsTy) {
  llvm_unreachable("DerivedType is already a concrete type!");
}


std::string Type::getDescription() const {
  LLVMContextImpl *pImpl = getContext().pImpl;
  TypePrinting &Map =
    isAbstract() ?
      pImpl->AbstractTypeDescriptions :
      pImpl->ConcreteTypeDescriptions;
  
  std::string DescStr;
  raw_string_ostream DescOS(DescStr);
  Map.print(this, DescOS);
  return DescOS.str();
}


bool StructType::indexValid(const Value *V) const {
  // Structure indexes require 32-bit integer constants.
  if (V->getType() == Type::getInt32Ty(V->getContext()))
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

const Type *Type::getVoidTy(LLVMContext &C) {
  return C.pImpl->VoidTy;
}

const Type *Type::getLabelTy(LLVMContext &C) {
  return C.pImpl->LabelTy;
}

const Type *Type::getFloatTy(LLVMContext &C) {
  return C.pImpl->FloatTy;
}

const Type *Type::getDoubleTy(LLVMContext &C) {
  return C.pImpl->DoubleTy;
}

const Type *Type::getMetadataTy(LLVMContext &C) {
  return C.pImpl->MetadataTy;
}

const Type *Type::getX86_FP80Ty(LLVMContext &C) {
  return C.pImpl->X86_FP80Ty;
}

const Type *Type::getFP128Ty(LLVMContext &C) {
  return C.pImpl->FP128Ty;
}

const Type *Type::getPPC_FP128Ty(LLVMContext &C) {
  return C.pImpl->PPC_FP128Ty;
}

const IntegerType *Type::getInt1Ty(LLVMContext &C) {
  return C.pImpl->Int1Ty;
}

const IntegerType *Type::getInt8Ty(LLVMContext &C) {
  return C.pImpl->Int8Ty;
}

const IntegerType *Type::getInt16Ty(LLVMContext &C) {
  return C.pImpl->Int16Ty;
}

const IntegerType *Type::getInt32Ty(LLVMContext &C) {
  return C.pImpl->Int32Ty;
}

const IntegerType *Type::getInt64Ty(LLVMContext &C) {
  return C.pImpl->Int64Ty;
}

//===----------------------------------------------------------------------===//
//                          Derived Type Constructors
//===----------------------------------------------------------------------===//

/// isValidReturnType - Return true if the specified type is valid as a return
/// type.
bool FunctionType::isValidReturnType(const Type *RetTy) {
  if (RetTy->isFirstClassType()) {
    if (const PointerType *PTy = dyn_cast<PointerType>(RetTy))
      return PTy->getElementType() != Type::getMetadataTy(RetTy->getContext());
    return true;
  }
  if (RetTy == Type::getVoidTy(RetTy->getContext()) ||
      RetTy == Type::getMetadataTy(RetTy->getContext()) ||
      isa<OpaqueType>(RetTy))
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

/// isValidArgumentType - Return true if the specified type is valid as an
/// argument type.
bool FunctionType::isValidArgumentType(const Type *ArgTy) {
  if ((!ArgTy->isFirstClassType() && !isa<OpaqueType>(ArgTy)) ||
      (isa<PointerType>(ArgTy) &&
       cast<PointerType>(ArgTy)->getElementType() == 
            Type::getMetadataTy(ArgTy->getContext())))
    return false;

  return true;
}

FunctionType::FunctionType(const Type *Result,
                           const std::vector<const Type*> &Params,
                           bool IsVarArgs)
  : DerivedType(Result->getContext(), FunctionTyID), isVarArgs(IsVarArgs) {
  ContainedTys = reinterpret_cast<PATypeHandle*>(this+1);
  NumContainedTys = Params.size() + 1; // + 1 for result type
  assert(isValidReturnType(Result) && "invalid return type for function");


  bool isAbstract = Result->isAbstract();
  new (&ContainedTys[0]) PATypeHandle(Result, this);

  for (unsigned i = 0; i != Params.size(); ++i) {
    assert(isValidArgumentType(Params[i]) &&
           "Not a valid type for function argument!");
    new (&ContainedTys[i+1]) PATypeHandle(Params[i], this);
    isAbstract |= Params[i]->isAbstract();
  }

  // Calculate whether or not this type is abstract
  setAbstract(isAbstract);
}

StructType::StructType(LLVMContext &C, 
                       const std::vector<const Type*> &Types, bool isPacked)
  : CompositeType(C, StructTyID) {
  ContainedTys = reinterpret_cast<PATypeHandle*>(this + 1);
  NumContainedTys = Types.size();
  setSubclassData(isPacked);
  bool isAbstract = false;
  for (unsigned i = 0; i < Types.size(); ++i) {
    assert(Types[i] && "<null> type for structure field!");
    assert(isValidElementType(Types[i]) &&
           "Invalid type for structure element!");
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
  assert(isValidElementType(ElType) &&
         "Elements of a VectorType must be a primitive type");

}


PointerType::PointerType(const Type *E, unsigned AddrSpace)
  : SequentialType(PointerTyID, E) {
  AddressSpace = AddrSpace;
  // Calculate whether or not this type is abstract
  setAbstract(E->isAbstract());
}

OpaqueType::OpaqueType(LLVMContext &C) : DerivedType(C, OpaqueTyID) {
  setAbstract(true);
#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *this << "\n";
#endif
}

void PATypeHolder::destroy() {
  Ty = 0;
}

// dropAllTypeUses - When this (abstract) type is resolved to be equal to
// another (more concrete) type, we must eliminate all references to other
// types, to avoid some circular reference problems.
void DerivedType::dropAllTypeUses() {
  if (NumContainedTys != 0) {
    // The type must stay abstract.  To do this, we insert a pointer to a type
    // that will never get resolved, thus will always be abstract.
    static Type *AlwaysOpaqueTy = 0;
    static PATypeHolder* Holder = 0;
    Type *tmp = AlwaysOpaqueTy;
    if (llvm_is_multithreaded()) {
      sys::MemoryFence();
      if (!tmp) {
        llvm_acquire_global_lock();
        tmp = AlwaysOpaqueTy;
        if (!tmp) {
          tmp = OpaqueType::get(getContext());
          PATypeHolder* tmp2 = new PATypeHolder(tmp);
          sys::MemoryFence();
          AlwaysOpaqueTy = tmp;
          Holder = tmp2;
        }
      
        llvm_release_global_lock();
      }
    } else {
      AlwaysOpaqueTy = OpaqueType::get(getContext());
      Holder = new PATypeHolder(AlwaysOpaqueTy);
    } 
        
    ContainedTys[0] = AlwaysOpaqueTy;

    // Change the rest of the types to be Int32Ty's.  It doesn't matter what we
    // pick so long as it doesn't point back to this type.  We choose something
    // concrete to avoid overhead for adding to AbstracTypeUser lists and stuff.
    for (unsigned i = 1, e = NumContainedTys; i != e; ++i)
      ContainedTys[i] = Type::getInt32Ty(getContext());
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
    llvm_unreachable("Unknown derived type!");
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

//===----------------------------------------------------------------------===//
// Function Type Factory and Value Class...
//
const IntegerType *IntegerType::get(LLVMContext &C, unsigned NumBits) {
  assert(NumBits >= MIN_INT_BITS && "bitwidth too small");
  assert(NumBits <= MAX_INT_BITS && "bitwidth too large");

  // Check for the built-in integer types
  switch (NumBits) {
    case  1: return cast<IntegerType>(Type::getInt1Ty(C));
    case  8: return cast<IntegerType>(Type::getInt8Ty(C));
    case 16: return cast<IntegerType>(Type::getInt16Ty(C));
    case 32: return cast<IntegerType>(Type::getInt32Ty(C));
    case 64: return cast<IntegerType>(Type::getInt64Ty(C));
    default: 
      break;
  }

  LLVMContextImpl *pImpl = C.pImpl;
  
  IntegerValType IVT(NumBits);
  IntegerType *ITy = 0;
  
  // First, see if the type is already in the table, for which
  // a reader lock suffices.
  sys::SmartScopedLock<true> L(pImpl->TypeMapLock);
  ITy = pImpl->IntegerTypes.get(IVT);
    
  if (!ITy) {
    // Value not found.  Derive a new type!
    ITy = new IntegerType(C, NumBits);
    pImpl->IntegerTypes.add(IVT, ITy);
  }
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
  FunctionType *FT = 0;
  
  LLVMContextImpl *pImpl = ReturnType->getContext().pImpl;
  
  sys::SmartScopedLock<true> L(pImpl->TypeMapLock);
  FT = pImpl->FunctionTypes.get(VT);
  
  if (!FT) {
    FT = (FunctionType*) operator new(sizeof(FunctionType) +
                                    sizeof(PATypeHandle)*(Params.size()+1));
    new (FT) FunctionType(ReturnType, Params, isVarArg);
    pImpl->FunctionTypes.add(VT, FT);
  }

#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << FT << "\n";
#endif
  return FT;
}

ArrayType *ArrayType::get(const Type *ElementType, uint64_t NumElements) {
  assert(ElementType && "Can't get array of <null> types!");
  assert(isValidElementType(ElementType) && "Invalid type for array element!");

  ArrayValType AVT(ElementType, NumElements);
  ArrayType *AT = 0;

  LLVMContextImpl *pImpl = ElementType->getContext().pImpl;
  
  sys::SmartScopedLock<true> L(pImpl->TypeMapLock);
  AT = pImpl->ArrayTypes.get(AVT);
      
  if (!AT) {
    // Value not found.  Derive a new type!
    pImpl->ArrayTypes.add(AVT, AT = new ArrayType(ElementType, NumElements));
  }
#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *AT << "\n";
#endif
  return AT;
}

bool ArrayType::isValidElementType(const Type *ElemTy) {
  if (ElemTy == Type::getVoidTy(ElemTy->getContext()) ||
      ElemTy == Type::getLabelTy(ElemTy->getContext()) ||
      ElemTy == Type::getMetadataTy(ElemTy->getContext()))
    return false;

  if (const PointerType *PTy = dyn_cast<PointerType>(ElemTy))
    if (PTy->getElementType() == Type::getMetadataTy(ElemTy->getContext()))
      return false;

  return true;
}

VectorType *VectorType::get(const Type *ElementType, unsigned NumElements) {
  assert(ElementType && "Can't get vector of <null> types!");

  VectorValType PVT(ElementType, NumElements);
  VectorType *PT = 0;
  
  LLVMContextImpl *pImpl = ElementType->getContext().pImpl;
  
  sys::SmartScopedLock<true> L(pImpl->TypeMapLock);
  PT = pImpl->VectorTypes.get(PVT);
    
  if (!PT) {
    pImpl->VectorTypes.add(PVT, PT = new VectorType(ElementType, NumElements));
  }
#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *PT << "\n";
#endif
  return PT;
}

bool VectorType::isValidElementType(const Type *ElemTy) {
  if (ElemTy->isInteger() || ElemTy->isFloatingPoint() ||
      isa<OpaqueType>(ElemTy))
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// Struct Type Factory...
//

StructType *StructType::get(LLVMContext &Context,
                            const std::vector<const Type*> &ETypes, 
                            bool isPacked) {
  StructValType STV(ETypes, isPacked);
  StructType *ST = 0;
  
  LLVMContextImpl *pImpl = Context.pImpl;
  
  sys::SmartScopedLock<true> L(pImpl->TypeMapLock);
  ST = pImpl->StructTypes.get(STV);
    
  if (!ST) {
    // Value not found.  Derive a new type!
    ST = (StructType*) operator new(sizeof(StructType) +
                                    sizeof(PATypeHandle) * ETypes.size());
    new (ST) StructType(Context, ETypes, isPacked);
    pImpl->StructTypes.add(STV, ST);
  }
#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *ST << "\n";
#endif
  return ST;
}

StructType *StructType::get(LLVMContext &Context, const Type *type, ...) {
  va_list ap;
  std::vector<const llvm::Type*> StructFields;
  va_start(ap, type);
  while (type) {
    StructFields.push_back(type);
    type = va_arg(ap, llvm::Type*);
  }
  return llvm::StructType::get(Context, StructFields);
}

bool StructType::isValidElementType(const Type *ElemTy) {
  if (ElemTy == Type::getVoidTy(ElemTy->getContext()) ||
      ElemTy == Type::getLabelTy(ElemTy->getContext()) ||
      ElemTy == Type::getMetadataTy(ElemTy->getContext()))
    return false;

  if (const PointerType *PTy = dyn_cast<PointerType>(ElemTy))
    if (PTy->getElementType() == Type::getMetadataTy(ElemTy->getContext()))
      return false;

  return true;
}


//===----------------------------------------------------------------------===//
// Pointer Type Factory...
//

PointerType *PointerType::get(const Type *ValueType, unsigned AddressSpace) {
  assert(ValueType && "Can't get a pointer to <null> type!");
  assert(ValueType != Type::getVoidTy(ValueType->getContext()) &&
         "Pointer to void is not valid, use i8* instead!");
  assert(isValidElementType(ValueType) && "Invalid type for pointer element!");
  PointerValType PVT(ValueType, AddressSpace);

  PointerType *PT = 0;
  
  LLVMContextImpl *pImpl = ValueType->getContext().pImpl;
  
  sys::SmartScopedLock<true> L(pImpl->TypeMapLock);
  PT = pImpl->PointerTypes.get(PVT);
  
  if (!PT) {
    // Value not found.  Derive a new type!
    pImpl->PointerTypes.add(PVT, PT = new PointerType(ValueType, AddressSpace));
  }
#ifdef DEBUG_MERGE_TYPES
  DOUT << "Derived new type: " << *PT << "\n";
#endif
  return PT;
}

PointerType *Type::getPointerTo(unsigned addrs) const {
  return PointerType::get(this, addrs);
}

bool PointerType::isValidElementType(const Type *ElemTy) {
  if (ElemTy == Type::getVoidTy(ElemTy->getContext()) ||
      ElemTy == Type::getLabelTy(ElemTy->getContext()))
    return false;

  if (const PointerType *PTy = dyn_cast<PointerType>(ElemTy))
    if (PTy->getElementType() == Type::getMetadataTy(ElemTy->getContext()))
      return false;

  return true;
}


//===----------------------------------------------------------------------===//
//                     Derived Type Refinement Functions
//===----------------------------------------------------------------------===//

// addAbstractTypeUser - Notify an abstract type that there is a new user of
// it.  This function is called primarily by the PATypeHandle class.
void Type::addAbstractTypeUser(AbstractTypeUser *U) const {
  assert(isAbstract() && "addAbstractTypeUser: Current type not abstract!");
  LLVMContextImpl *pImpl = getContext().pImpl;
  pImpl->AbstractTypeUsersLock.acquire();
  AbstractTypeUsers.push_back(U);
  pImpl->AbstractTypeUsersLock.release();
}


// removeAbstractTypeUser - Notify an abstract type that a user of the class
// no longer has a handle to the type.  This function is called primarily by
// the PATypeHandle class.  When there are no users of the abstract type, it
// is annihilated, because there is no way to get a reference to it ever again.
//
void Type::removeAbstractTypeUser(AbstractTypeUser *U) const {
  LLVMContextImpl *pImpl = getContext().pImpl;
  pImpl->AbstractTypeUsersLock.acquire();
  
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
  
  pImpl->AbstractTypeUsersLock.release();
}

// unlockedRefineAbstractTypeTo - This function is used when it is discovered
// that the 'this' abstract type is actually equivalent to the NewType
// specified. This causes all users of 'this' to switch to reference the more 
// concrete type NewType and for 'this' to be deleted.  Only used for internal
// callers.
//
void DerivedType::unlockedRefineAbstractTypeTo(const Type *NewType) {
  assert(isAbstract() && "refineAbstractTypeTo: Current type is not abstract!");
  assert(this != NewType && "Can't refine to myself!");
  assert(ForwardType == 0 && "This type has already been refined!");

  LLVMContextImpl *pImpl = getContext().pImpl;

  // The descriptions may be out of date.  Conservatively clear them all!
  pImpl->AbstractTypeDescriptions.clear();

#ifdef DEBUG_MERGE_TYPES
  DOUT << "REFINING abstract type [" << (void*)this << " "
       << *this << "] to [" << (void*)NewType << " "
       << *NewType << "]!\n";
#endif

  // Make sure to put the type to be refined to into a holder so that if IT gets
  // refined, that we will not continue using a dead reference...
  //
  PATypeHolder NewTy(NewType);
  // Any PATypeHolders referring to this type will now automatically forward o
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
  pImpl->AbstractTypeUsersLock.acquire();
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
  pImpl->AbstractTypeUsersLock.release();

  // If we were successful removing all users from the type, 'this' will be
  // deleted when the last PATypeHolder is destroyed or updated from this type.
  // This may occur on exit of this function, as the CurrentTy object is
  // destroyed.
}

// refineAbstractTypeTo - This function is used by external callers to notify
// us that this abstract type is equivalent to another type.
//
void DerivedType::refineAbstractTypeTo(const Type *NewType) {
  // All recursive calls will go through unlockedRefineAbstractTypeTo,
  // to avoid deadlock problems.
  sys::SmartScopedLock<true> L(NewType->getContext().pImpl->TypeMapLock);
  unlockedRefineAbstractTypeTo(NewType);
}

// notifyUsesThatTypeBecameConcrete - Notify AbstractTypeUsers of this type that
// the current type has transitioned from being abstract to being concrete.
//
void DerivedType::notifyUsesThatTypeBecameConcrete() {
#ifdef DEBUG_MERGE_TYPES
  DOUT << "typeIsREFINED type: " << (void*)this << " " << *this << "\n";
#endif

  LLVMContextImpl *pImpl = getContext().pImpl;

  pImpl->AbstractTypeUsersLock.acquire();
  unsigned OldSize = AbstractTypeUsers.size(); OldSize=OldSize;
  while (!AbstractTypeUsers.empty()) {
    AbstractTypeUser *ATU = AbstractTypeUsers.back();
    ATU->typeBecameConcrete(this);

    assert(AbstractTypeUsers.size() < OldSize-- &&
           "AbstractTypeUser did not remove itself from the use list!");
  }
  pImpl->AbstractTypeUsersLock.release();
}

// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void FunctionType::refineAbstractType(const DerivedType *OldType,
                                      const Type *NewType) {
  LLVMContextImpl *pImpl = OldType->getContext().pImpl;
  pImpl->FunctionTypes.RefineAbstractType(this, OldType, NewType);
}

void FunctionType::typeBecameConcrete(const DerivedType *AbsTy) {
  LLVMContextImpl *pImpl = AbsTy->getContext().pImpl;
  pImpl->FunctionTypes.TypeBecameConcrete(this, AbsTy);
}


// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void ArrayType::refineAbstractType(const DerivedType *OldType,
                                   const Type *NewType) {
  LLVMContextImpl *pImpl = OldType->getContext().pImpl;
  pImpl->ArrayTypes.RefineAbstractType(this, OldType, NewType);
}

void ArrayType::typeBecameConcrete(const DerivedType *AbsTy) {
  LLVMContextImpl *pImpl = AbsTy->getContext().pImpl;
  pImpl->ArrayTypes.TypeBecameConcrete(this, AbsTy);
}

// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void VectorType::refineAbstractType(const DerivedType *OldType,
                                   const Type *NewType) {
  LLVMContextImpl *pImpl = OldType->getContext().pImpl;
  pImpl->VectorTypes.RefineAbstractType(this, OldType, NewType);
}

void VectorType::typeBecameConcrete(const DerivedType *AbsTy) {
  LLVMContextImpl *pImpl = AbsTy->getContext().pImpl;
  pImpl->VectorTypes.TypeBecameConcrete(this, AbsTy);
}

// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void StructType::refineAbstractType(const DerivedType *OldType,
                                    const Type *NewType) {
  LLVMContextImpl *pImpl = OldType->getContext().pImpl;
  pImpl->StructTypes.RefineAbstractType(this, OldType, NewType);
}

void StructType::typeBecameConcrete(const DerivedType *AbsTy) {
  LLVMContextImpl *pImpl = AbsTy->getContext().pImpl;
  pImpl->StructTypes.TypeBecameConcrete(this, AbsTy);
}

// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void PointerType::refineAbstractType(const DerivedType *OldType,
                                     const Type *NewType) {
  LLVMContextImpl *pImpl = OldType->getContext().pImpl;
  pImpl->PointerTypes.RefineAbstractType(this, OldType, NewType);
}

void PointerType::typeBecameConcrete(const DerivedType *AbsTy) {
  LLVMContextImpl *pImpl = AbsTy->getContext().pImpl;
  pImpl->PointerTypes.TypeBecameConcrete(this, AbsTy);
}

bool SequentialType::indexValid(const Value *V) const {
  if (isa<IntegerType>(V->getType())) 
    return true;
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

raw_ostream &operator<<(raw_ostream &OS, const Type &T) {
  T.print(OS);
  return OS;
}
}
