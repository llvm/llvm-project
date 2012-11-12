//===- lib/Linker/LinkModules.cpp - Module Linker Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM module linker.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/TypeFinder.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm-c/Linker.h"
#include <cctype>
using namespace llvm;

//===----------------------------------------------------------------------===//
// TypeMap implementation.
//===----------------------------------------------------------------------===//

namespace {
class TypeMapTy : public ValueMapTypeRemapper {
  /// MappedTypes - This is a mapping from a source type to a destination type
  /// to use.
  DenseMap<Type*, Type*> MappedTypes;

  /// SpeculativeTypes - When checking to see if two subgraphs are isomorphic,
  /// we speculatively add types to MappedTypes, but keep track of them here in
  /// case we need to roll back.
  SmallVector<Type*, 16> SpeculativeTypes;
  
  /// SrcDefinitionsToResolve - This is a list of non-opaque structs in the
  /// source module that are mapped to an opaque struct in the destination
  /// module.
  SmallVector<StructType*, 16> SrcDefinitionsToResolve;
  
  /// DstResolvedOpaqueTypes - This is the set of opaque types in the
  /// destination modules who are getting a body from the source module.
  SmallPtrSet<StructType*, 16> DstResolvedOpaqueTypes;

public:
  /// addTypeMapping - Indicate that the specified type in the destination
  /// module is conceptually equivalent to the specified type in the source
  /// module.
  void addTypeMapping(Type *DstTy, Type *SrcTy);

  /// linkDefinedTypeBodies - Produce a body for an opaque type in the dest
  /// module from a type definition in the source module.
  void linkDefinedTypeBodies();
  
  /// get - Return the mapped type to use for the specified input type from the
  /// source module.
  Type *get(Type *SrcTy);

  FunctionType *get(FunctionType *T) {return cast<FunctionType>(get((Type*)T));}

  /// dump - Dump out the type map for debugging purposes.
  void dump() const {
    for (DenseMap<Type*, Type*>::const_iterator
           I = MappedTypes.begin(), E = MappedTypes.end(); I != E; ++I) {
      dbgs() << "TypeMap: ";
      I->first->dump();
      dbgs() << " => ";
      I->second->dump();
      dbgs() << '\n';
    }
  }

private:
  Type *getImpl(Type *T);
  /// remapType - Implement the ValueMapTypeRemapper interface.
  Type *remapType(Type *SrcTy) {
    return get(SrcTy);
  }
  
  bool areTypesIsomorphic(Type *DstTy, Type *SrcTy);
};
}

void TypeMapTy::addTypeMapping(Type *DstTy, Type *SrcTy) {
  Type *&Entry = MappedTypes[SrcTy];
  if (Entry) return;
  
  if (DstTy == SrcTy) {
    Entry = DstTy;
    return;
  }
  
  // Check to see if these types are recursively isomorphic and establish a
  // mapping between them if so.
  if (!areTypesIsomorphic(DstTy, SrcTy)) {
    // Oops, they aren't isomorphic.  Just discard this request by rolling out
    // any speculative mappings we've established.
    for (unsigned i = 0, e = SpeculativeTypes.size(); i != e; ++i)
      MappedTypes.erase(SpeculativeTypes[i]);
  }
  SpeculativeTypes.clear();
}

/// areTypesIsomorphic - Recursively walk this pair of types, returning true
/// if they are isomorphic, false if they are not.
bool TypeMapTy::areTypesIsomorphic(Type *DstTy, Type *SrcTy) {
  // Two types with differing kinds are clearly not isomorphic.
  if (DstTy->getTypeID() != SrcTy->getTypeID()) return false;

  // If we have an entry in the MappedTypes table, then we have our answer.
  Type *&Entry = MappedTypes[SrcTy];
  if (Entry)
    return Entry == DstTy;

  // Two identical types are clearly isomorphic.  Remember this
  // non-speculatively.
  if (DstTy == SrcTy) {
    Entry = DstTy;
    return true;
  }
  
  // Okay, we have two types with identical kinds that we haven't seen before.

  // If this is an opaque struct type, special case it.
  if (StructType *SSTy = dyn_cast<StructType>(SrcTy)) {
    // Mapping an opaque type to any struct, just keep the dest struct.
    if (SSTy->isOpaque()) {
      Entry = DstTy;
      SpeculativeTypes.push_back(SrcTy);
      return true;
    }

    // Mapping a non-opaque source type to an opaque dest.  If this is the first
    // type that we're mapping onto this destination type then we succeed.  Keep
    // the dest, but fill it in later.  This doesn't need to be speculative.  If
    // this is the second (different) type that we're trying to map onto the
    // same opaque type then we fail.
    if (cast<StructType>(DstTy)->isOpaque()) {
      // We can only map one source type onto the opaque destination type.
      if (!DstResolvedOpaqueTypes.insert(cast<StructType>(DstTy)))
        return false;
      SrcDefinitionsToResolve.push_back(SSTy);
      Entry = DstTy;
      return true;
    }
  }
  
  // If the number of subtypes disagree between the two types, then we fail.
  if (SrcTy->getNumContainedTypes() != DstTy->getNumContainedTypes())
    return false;
  
  // Fail if any of the extra properties (e.g. array size) of the type disagree.
  if (isa<IntegerType>(DstTy))
    return false;  // bitwidth disagrees.
  if (PointerType *PT = dyn_cast<PointerType>(DstTy)) {
    if (PT->getAddressSpace() != cast<PointerType>(SrcTy)->getAddressSpace())
      return false;
    
  } else if (FunctionType *FT = dyn_cast<FunctionType>(DstTy)) {
    if (FT->isVarArg() != cast<FunctionType>(SrcTy)->isVarArg())
      return false;
  } else if (StructType *DSTy = dyn_cast<StructType>(DstTy)) {
    StructType *SSTy = cast<StructType>(SrcTy);
    if (DSTy->isLiteral() != SSTy->isLiteral() ||
        DSTy->isPacked() != SSTy->isPacked())
      return false;
  } else if (ArrayType *DATy = dyn_cast<ArrayType>(DstTy)) {
    if (DATy->getNumElements() != cast<ArrayType>(SrcTy)->getNumElements())
      return false;
  } else if (VectorType *DVTy = dyn_cast<VectorType>(DstTy)) {
    if (DVTy->getNumElements() != cast<ArrayType>(SrcTy)->getNumElements())
      return false;
  }

  // Otherwise, we speculate that these two types will line up and recursively
  // check the subelements.
  Entry = DstTy;
  SpeculativeTypes.push_back(SrcTy);

  for (unsigned i = 0, e = SrcTy->getNumContainedTypes(); i != e; ++i)
    if (!areTypesIsomorphic(DstTy->getContainedType(i),
                            SrcTy->getContainedType(i)))
      return false;
  
  // If everything seems to have lined up, then everything is great.
  return true;
}

/// linkDefinedTypeBodies - Produce a body for an opaque type in the dest
/// module from a type definition in the source module.
void TypeMapTy::linkDefinedTypeBodies() {
  SmallVector<Type*, 16> Elements;
  SmallString<16> TmpName;
  
  // Note that processing entries in this loop (calling 'get') can add new
  // entries to the SrcDefinitionsToResolve vector.
  while (!SrcDefinitionsToResolve.empty()) {
    StructType *SrcSTy = SrcDefinitionsToResolve.pop_back_val();
    StructType *DstSTy = cast<StructType>(MappedTypes[SrcSTy]);
    
    // TypeMap is a many-to-one mapping, if there were multiple types that
    // provide a body for DstSTy then previous iterations of this loop may have
    // already handled it.  Just ignore this case.
    if (!DstSTy->isOpaque()) continue;
    assert(!SrcSTy->isOpaque() && "Not resolving a definition?");
    
    // Map the body of the source type over to a new body for the dest type.
    Elements.resize(SrcSTy->getNumElements());
    for (unsigned i = 0, e = Elements.size(); i != e; ++i)
      Elements[i] = getImpl(SrcSTy->getElementType(i));
    
    DstSTy->setBody(Elements, SrcSTy->isPacked());
    
    // If DstSTy has no name or has a longer name than STy, then viciously steal
    // STy's name.
    if (!SrcSTy->hasName()) continue;
    StringRef SrcName = SrcSTy->getName();
    
    if (!DstSTy->hasName() || DstSTy->getName().size() > SrcName.size()) {
      TmpName.insert(TmpName.end(), SrcName.begin(), SrcName.end());
      SrcSTy->setName("");
      DstSTy->setName(TmpName.str());
      TmpName.clear();
    }
  }
  
  DstResolvedOpaqueTypes.clear();
}

/// get - Return the mapped type to use for the specified input type from the
/// source module.
Type *TypeMapTy::get(Type *Ty) {
  Type *Result = getImpl(Ty);
  
  // If this caused a reference to any struct type, resolve it before returning.
  if (!SrcDefinitionsToResolve.empty())
    linkDefinedTypeBodies();
  return Result;
}

/// getImpl - This is the recursive version of get().
Type *TypeMapTy::getImpl(Type *Ty) {
  // If we already have an entry for this type, return it.
  Type **Entry = &MappedTypes[Ty];
  if (*Entry) return *Entry;
  
  // If this is not a named struct type, then just map all of the elements and
  // then rebuild the type from inside out.
  if (!isa<StructType>(Ty) || cast<StructType>(Ty)->isLiteral()) {
    // If there are no element types to map, then the type is itself.  This is
    // true for the anonymous {} struct, things like 'float', integers, etc.
    if (Ty->getNumContainedTypes() == 0)
      return *Entry = Ty;
    
    // Remap all of the elements, keeping track of whether any of them change.
    bool AnyChange = false;
    SmallVector<Type*, 4> ElementTypes;
    ElementTypes.resize(Ty->getNumContainedTypes());
    for (unsigned i = 0, e = Ty->getNumContainedTypes(); i != e; ++i) {
      ElementTypes[i] = getImpl(Ty->getContainedType(i));
      AnyChange |= ElementTypes[i] != Ty->getContainedType(i);
    }
    
    // If we found our type while recursively processing stuff, just use it.
    Entry = &MappedTypes[Ty];
    if (*Entry) return *Entry;
    
    // If all of the element types mapped directly over, then the type is usable
    // as-is.
    if (!AnyChange)
      return *Entry = Ty;
    
    // Otherwise, rebuild a modified type.
    switch (Ty->getTypeID()) {
    default: llvm_unreachable("unknown derived type to remap");
    case Type::ArrayTyID:
      return *Entry = ArrayType::get(ElementTypes[0],
                                     cast<ArrayType>(Ty)->getNumElements());
    case Type::VectorTyID: 
      return *Entry = VectorType::get(ElementTypes[0],
                                      cast<VectorType>(Ty)->getNumElements());
    case Type::PointerTyID:
      return *Entry = PointerType::get(ElementTypes[0],
                                      cast<PointerType>(Ty)->getAddressSpace());
    case Type::FunctionTyID:
      return *Entry = FunctionType::get(ElementTypes[0],
                                        makeArrayRef(ElementTypes).slice(1),
                                        cast<FunctionType>(Ty)->isVarArg());
    case Type::StructTyID:
      // Note that this is only reached for anonymous structs.
      return *Entry = StructType::get(Ty->getContext(), ElementTypes,
                                      cast<StructType>(Ty)->isPacked());
    }
  }

  // Otherwise, this is an unmapped named struct.  If the struct can be directly
  // mapped over, just use it as-is.  This happens in a case when the linked-in
  // module has something like:
  //   %T = type {%T*, i32}
  //   @GV = global %T* null
  // where T does not exist at all in the destination module.
  //
  // The other case we watch for is when the type is not in the destination
  // module, but that it has to be rebuilt because it refers to something that
  // is already mapped.  For example, if the destination module has:
  //  %A = type { i32 }
  // and the source module has something like
  //  %A' = type { i32 }
  //  %B = type { %A'* }
  //  @GV = global %B* null
  // then we want to create a new type: "%B = type { %A*}" and have it take the
  // pristine "%B" name from the source module.
  //
  // To determine which case this is, we have to recursively walk the type graph
  // speculating that we'll be able to reuse it unmodified.  Only if this is
  // safe would we map the entire thing over.  Because this is an optimization,
  // and is not required for the prettiness of the linked module, we just skip
  // it and always rebuild a type here.
  StructType *STy = cast<StructType>(Ty);
  
  // If the type is opaque, we can just use it directly.
  if (STy->isOpaque())
    return *Entry = STy;
  
  // Otherwise we create a new type and resolve its body later.  This will be
  // resolved by the top level of get().
  SrcDefinitionsToResolve.push_back(STy);
  StructType *DTy = StructType::create(STy->getContext());
  DstResolvedOpaqueTypes.insert(DTy);
  return *Entry = DTy;
}

//===----------------------------------------------------------------------===//
// ModuleLinker implementation.
//===----------------------------------------------------------------------===//

namespace {
  /// ModuleLinker - This is an implementation class for the LinkModules
  /// function, which is the entrypoint for this file.
  class ModuleLinker {
    Module *DstM, *SrcM;
    
    TypeMapTy TypeMap; 

    /// ValueMap - Mapping of values from what they used to be in Src, to what
    /// they are now in DstM.  ValueToValueMapTy is a ValueMap, which involves
    /// some overhead due to the use of Value handles which the Linker doesn't
    /// actually need, but this allows us to reuse the ValueMapper code.
    ValueToValueMapTy ValueMap;
    
    struct AppendingVarInfo {
      GlobalVariable *NewGV;  // New aggregate global in dest module.
      Constant *DstInit;      // Old initializer from dest module.
      Constant *SrcInit;      // Old initializer from src module.
    };
    
    std::vector<AppendingVarInfo> AppendingVars;
    
    unsigned Mode; // Mode to treat source module.
    
    // Set of items not to link in from source.
    SmallPtrSet<const Value*, 16> DoNotLinkFromSource;
    
    // Vector of functions to lazily link in.
    std::vector<Function*> LazilyLinkFunctions;
    
  public:
    std::string ErrorMsg;
    
    ModuleLinker(Module *dstM, Module *srcM, unsigned mode)
      : DstM(dstM), SrcM(srcM), Mode(mode) { }
    
    bool run();
    
  private:
    /// emitError - Helper method for setting a message and returning an error
    /// code.
    bool emitError(const Twine &Message) {
      ErrorMsg = Message.str();
      return true;
    }
    
    /// getLinkageResult - This analyzes the two global values and determines
    /// what the result will look like in the destination module.
    bool getLinkageResult(GlobalValue *Dest, const GlobalValue *Src,
                          GlobalValue::LinkageTypes &LT,
                          GlobalValue::VisibilityTypes &Vis,
                          bool &LinkFromSrc);

    /// getLinkedToGlobal - Given a global in the source module, return the
    /// global in the destination module that is being linked to, if any.
    GlobalValue *getLinkedToGlobal(GlobalValue *SrcGV) {
      // If the source has no name it can't link.  If it has local linkage,
      // there is no name match-up going on.
      if (!SrcGV->hasName() || SrcGV->hasLocalLinkage())
        return 0;
      
      // Otherwise see if we have a match in the destination module's symtab.
      GlobalValue *DGV = DstM->getNamedValue(SrcGV->getName());
      if (DGV == 0) return 0;
        
      // If we found a global with the same name in the dest module, but it has
      // internal linkage, we are really not doing any linkage here.
      if (DGV->hasLocalLinkage())
        return 0;

      // Otherwise, we do in fact link to the destination global.
      return DGV;
    }
    
    void computeTypeMapping();
    bool categorizeModuleFlagNodes(const NamedMDNode *ModFlags,
                                   DenseMap<MDString*, MDNode*> &ErrorNode,
                                   DenseMap<MDString*, MDNode*> &WarningNode,
                                   DenseMap<MDString*, MDNode*> &OverrideNode,
                                   DenseMap<MDString*,
                                   SmallSetVector<MDNode*, 8> > &RequireNodes,
                                   SmallSetVector<MDString*, 16> &SeenIDs);
    
    bool linkAppendingVarProto(GlobalVariable *DstGV, GlobalVariable *SrcGV);
    bool linkGlobalProto(GlobalVariable *SrcGV);
    bool linkFunctionProto(Function *SrcF);
    bool linkAliasProto(GlobalAlias *SrcA);
    bool linkModuleFlagsMetadata();
    
    void linkAppendingVarInit(const AppendingVarInfo &AVI);
    void linkGlobalInits();
    void linkFunctionBody(Function *Dst, Function *Src);
    void linkAliasBodies();
    void linkNamedMDNodes();
  };
}

/// forceRenaming - The LLVM SymbolTable class autorenames globals that conflict
/// in the symbol table.  This is good for all clients except for us.  Go
/// through the trouble to force this back.
static void forceRenaming(GlobalValue *GV, StringRef Name) {
  // If the global doesn't force its name or if it already has the right name,
  // there is nothing for us to do.
  if (GV->hasLocalLinkage() || GV->getName() == Name)
    return;

  Module *M = GV->getParent();

  // If there is a conflict, rename the conflict.
  if (GlobalValue *ConflictGV = M->getNamedValue(Name)) {
    GV->takeName(ConflictGV);
    ConflictGV->setName(Name);    // This will cause ConflictGV to get renamed
    assert(ConflictGV->getName() != Name && "forceRenaming didn't work");
  } else {
    GV->setName(Name);              // Force the name back
  }
}

/// copyGVAttributes - copy additional attributes (those not needed to construct
/// a GlobalValue) from the SrcGV to the DestGV.
static void copyGVAttributes(GlobalValue *DestGV, const GlobalValue *SrcGV) {
  // Use the maximum alignment, rather than just copying the alignment of SrcGV.
  unsigned Alignment = std::max(DestGV->getAlignment(), SrcGV->getAlignment());
  DestGV->copyAttributesFrom(SrcGV);
  DestGV->setAlignment(Alignment);
  
  forceRenaming(DestGV, SrcGV->getName());
}

static bool isLessConstraining(GlobalValue::VisibilityTypes a,
                               GlobalValue::VisibilityTypes b) {
  if (a == GlobalValue::HiddenVisibility)
    return false;
  if (b == GlobalValue::HiddenVisibility)
    return true;
  if (a == GlobalValue::ProtectedVisibility)
    return false;
  if (b == GlobalValue::ProtectedVisibility)
    return true;
  return false;
}

/// getLinkageResult - This analyzes the two global values and determines what
/// the result will look like in the destination module.  In particular, it
/// computes the resultant linkage type and visibility, computes whether the
/// global in the source should be copied over to the destination (replacing
/// the existing one), and computes whether this linkage is an error or not.
bool ModuleLinker::getLinkageResult(GlobalValue *Dest, const GlobalValue *Src,
                                    GlobalValue::LinkageTypes &LT,
                                    GlobalValue::VisibilityTypes &Vis,
                                    bool &LinkFromSrc) {
  assert(Dest && "Must have two globals being queried");
  assert(!Src->hasLocalLinkage() &&
         "If Src has internal linkage, Dest shouldn't be set!");
  
  bool SrcIsDeclaration = Src->isDeclaration() && !Src->isMaterializable();
  bool DestIsDeclaration = Dest->isDeclaration();
  
  if (SrcIsDeclaration) {
    // If Src is external or if both Src & Dest are external..  Just link the
    // external globals, we aren't adding anything.
    if (Src->hasDLLImportLinkage()) {
      // If one of GVs has DLLImport linkage, result should be dllimport'ed.
      if (DestIsDeclaration) {
        LinkFromSrc = true;
        LT = Src->getLinkage();
      }
    } else if (Dest->hasExternalWeakLinkage()) {
      // If the Dest is weak, use the source linkage.
      LinkFromSrc = true;
      LT = Src->getLinkage();
    } else {
      LinkFromSrc = false;
      LT = Dest->getLinkage();
    }
  } else if (DestIsDeclaration && !Dest->hasDLLImportLinkage()) {
    // If Dest is external but Src is not:
    LinkFromSrc = true;
    LT = Src->getLinkage();
  } else if (Src->isWeakForLinker()) {
    // At this point we know that Dest has LinkOnce, External*, Weak, Common,
    // or DLL* linkage.
    if (Dest->hasExternalWeakLinkage() ||
        Dest->hasAvailableExternallyLinkage() ||
        (Dest->hasLinkOnceLinkage() &&
         (Src->hasWeakLinkage() || Src->hasCommonLinkage()))) {
      LinkFromSrc = true;
      LT = Src->getLinkage();
    } else {
      LinkFromSrc = false;
      LT = Dest->getLinkage();
    }
  } else if (Dest->isWeakForLinker()) {
    // At this point we know that Src has External* or DLL* linkage.
    if (Src->hasExternalWeakLinkage()) {
      LinkFromSrc = false;
      LT = Dest->getLinkage();
    } else {
      LinkFromSrc = true;
      LT = GlobalValue::ExternalLinkage;
    }
  } else {
    assert((Dest->hasExternalLinkage()  || Dest->hasDLLImportLinkage() ||
            Dest->hasDLLExportLinkage() || Dest->hasExternalWeakLinkage()) &&
           (Src->hasExternalLinkage()   || Src->hasDLLImportLinkage() ||
            Src->hasDLLExportLinkage()  || Src->hasExternalWeakLinkage()) &&
           "Unexpected linkage type!");
    return emitError("Linking globals named '" + Src->getName() +
                 "': symbol multiply defined!");
  }

  // Compute the visibility. We follow the rules in the System V Application
  // Binary Interface.
  Vis = isLessConstraining(Src->getVisibility(), Dest->getVisibility()) ?
    Dest->getVisibility() : Src->getVisibility();
  return false;
}

/// computeTypeMapping - Loop over all of the linked values to compute type
/// mappings.  For example, if we link "extern Foo *x" and "Foo *x = NULL", then
/// we have two struct types 'Foo' but one got renamed when the module was
/// loaded into the same LLVMContext.
void ModuleLinker::computeTypeMapping() {
  // Incorporate globals.
  for (Module::global_iterator I = SrcM->global_begin(),
       E = SrcM->global_end(); I != E; ++I) {
    GlobalValue *DGV = getLinkedToGlobal(I);
    if (DGV == 0) continue;
    
    if (!DGV->hasAppendingLinkage() || !I->hasAppendingLinkage()) {
      TypeMap.addTypeMapping(DGV->getType(), I->getType());
      continue;      
    }
    
    // Unify the element type of appending arrays.
    ArrayType *DAT = cast<ArrayType>(DGV->getType()->getElementType());
    ArrayType *SAT = cast<ArrayType>(I->getType()->getElementType());
    TypeMap.addTypeMapping(DAT->getElementType(), SAT->getElementType());
  }
  
  // Incorporate functions.
  for (Module::iterator I = SrcM->begin(), E = SrcM->end(); I != E; ++I) {
    if (GlobalValue *DGV = getLinkedToGlobal(I))
      TypeMap.addTypeMapping(DGV->getType(), I->getType());
  }

  // Incorporate types by name, scanning all the types in the source module.
  // At this point, the destination module may have a type "%foo = { i32 }" for
  // example.  When the source module got loaded into the same LLVMContext, if
  // it had the same type, it would have been renamed to "%foo.42 = { i32 }".
  TypeFinder SrcStructTypes;
  SrcStructTypes.run(*SrcM, true);
  SmallPtrSet<StructType*, 32> SrcStructTypesSet(SrcStructTypes.begin(),
                                                 SrcStructTypes.end());

  TypeFinder DstStructTypes;
  DstStructTypes.run(*DstM, true);
  SmallPtrSet<StructType*, 32> DstStructTypesSet(DstStructTypes.begin(),
                                                 DstStructTypes.end());

  for (unsigned i = 0, e = SrcStructTypes.size(); i != e; ++i) {
    StructType *ST = SrcStructTypes[i];
    if (!ST->hasName()) continue;
    
    // Check to see if there is a dot in the name followed by a digit.
    size_t DotPos = ST->getName().rfind('.');
    if (DotPos == 0 || DotPos == StringRef::npos ||
        ST->getName().back() == '.' || !isdigit(ST->getName()[DotPos+1]))
      continue;
    
    // Check to see if the destination module has a struct with the prefix name.
    if (StructType *DST = DstM->getTypeByName(ST->getName().substr(0, DotPos)))
      // Don't use it if this actually came from the source module. They're in
      // the same LLVMContext after all. Also don't use it unless the type is
      // actually used in the destination module. This can happen in situations
      // like this:
      //
      //      Module A                         Module B
      //      --------                         --------
      //   %Z = type { %A }                %B = type { %C.1 }
      //   %A = type { %B.1, [7 x i8] }    %C.1 = type { i8* }
      //   %B.1 = type { %C }              %A.2 = type { %B.3, [5 x i8] }
      //   %C = type { i8* }               %B.3 = type { %C.1 }
      //
      // When we link Module B with Module A, the '%B' in Module B is
      // used. However, that would then use '%C.1'. But when we process '%C.1',
      // we prefer to take the '%C' version. So we are then left with both
      // '%C.1' and '%C' being used for the same types. This leads to some
      // variables using one type and some using the other.
      if (!SrcStructTypesSet.count(DST) && DstStructTypesSet.count(DST))
        TypeMap.addTypeMapping(DST, ST);
  }

  // Don't bother incorporating aliases, they aren't generally typed well.
  
  // Now that we have discovered all of the type equivalences, get a body for
  // any 'opaque' types in the dest module that are now resolved. 
  TypeMap.linkDefinedTypeBodies();
}

/// linkAppendingVarProto - If there were any appending global variables, link
/// them together now.  Return true on error.
bool ModuleLinker::linkAppendingVarProto(GlobalVariable *DstGV,
                                         GlobalVariable *SrcGV) {
 
  if (!SrcGV->hasAppendingLinkage() || !DstGV->hasAppendingLinkage())
    return emitError("Linking globals named '" + SrcGV->getName() +
           "': can only link appending global with another appending global!");
  
  ArrayType *DstTy = cast<ArrayType>(DstGV->getType()->getElementType());
  ArrayType *SrcTy =
    cast<ArrayType>(TypeMap.get(SrcGV->getType()->getElementType()));
  Type *EltTy = DstTy->getElementType();
  
  // Check to see that they two arrays agree on type.
  if (EltTy != SrcTy->getElementType())
    return emitError("Appending variables with different element types!");
  if (DstGV->isConstant() != SrcGV->isConstant())
    return emitError("Appending variables linked with different const'ness!");
  
  if (DstGV->getAlignment() != SrcGV->getAlignment())
    return emitError(
             "Appending variables with different alignment need to be linked!");
  
  if (DstGV->getVisibility() != SrcGV->getVisibility())
    return emitError(
            "Appending variables with different visibility need to be linked!");
  
  if (DstGV->getSection() != SrcGV->getSection())
    return emitError(
          "Appending variables with different section name need to be linked!");
  
  uint64_t NewSize = DstTy->getNumElements() + SrcTy->getNumElements();
  ArrayType *NewType = ArrayType::get(EltTy, NewSize);
  
  // Create the new global variable.
  GlobalVariable *NG =
    new GlobalVariable(*DstGV->getParent(), NewType, SrcGV->isConstant(),
                       DstGV->getLinkage(), /*init*/0, /*name*/"", DstGV,
                       DstGV->getThreadLocalMode(),
                       DstGV->getType()->getAddressSpace());
  
  // Propagate alignment, visibility and section info.
  copyGVAttributes(NG, DstGV);
  
  AppendingVarInfo AVI;
  AVI.NewGV = NG;
  AVI.DstInit = DstGV->getInitializer();
  AVI.SrcInit = SrcGV->getInitializer();
  AppendingVars.push_back(AVI);

  // Replace any uses of the two global variables with uses of the new
  // global.
  ValueMap[SrcGV] = ConstantExpr::getBitCast(NG, TypeMap.get(SrcGV->getType()));

  DstGV->replaceAllUsesWith(ConstantExpr::getBitCast(NG, DstGV->getType()));
  DstGV->eraseFromParent();
  
  // Track the source variable so we don't try to link it.
  DoNotLinkFromSource.insert(SrcGV);
  
  return false;
}

/// linkGlobalProto - Loop through the global variables in the src module and
/// merge them into the dest module.
bool ModuleLinker::linkGlobalProto(GlobalVariable *SGV) {
  GlobalValue *DGV = getLinkedToGlobal(SGV);
  llvm::Optional<GlobalValue::VisibilityTypes> NewVisibility;

  if (DGV) {
    // Concatenation of appending linkage variables is magic and handled later.
    if (DGV->hasAppendingLinkage() || SGV->hasAppendingLinkage())
      return linkAppendingVarProto(cast<GlobalVariable>(DGV), SGV);
    
    // Determine whether linkage of these two globals follows the source
    // module's definition or the destination module's definition.
    GlobalValue::LinkageTypes NewLinkage = GlobalValue::InternalLinkage;
    GlobalValue::VisibilityTypes NV;
    bool LinkFromSrc = false;
    if (getLinkageResult(DGV, SGV, NewLinkage, NV, LinkFromSrc))
      return true;
    NewVisibility = NV;

    // If we're not linking from the source, then keep the definition that we
    // have.
    if (!LinkFromSrc) {
      // Special case for const propagation.
      if (GlobalVariable *DGVar = dyn_cast<GlobalVariable>(DGV))
        if (DGVar->isDeclaration() && SGV->isConstant() && !DGVar->isConstant())
          DGVar->setConstant(true);
      
      // Set calculated linkage and visibility.
      DGV->setLinkage(NewLinkage);
      DGV->setVisibility(*NewVisibility);

      // Make sure to remember this mapping.
      ValueMap[SGV] = ConstantExpr::getBitCast(DGV,TypeMap.get(SGV->getType()));
      
      // Track the source global so that we don't attempt to copy it over when 
      // processing global initializers.
      DoNotLinkFromSource.insert(SGV);
      
      return false;
    }
  }
  
  // No linking to be performed or linking from the source: simply create an
  // identical version of the symbol over in the dest module... the
  // initializer will be filled in later by LinkGlobalInits.
  GlobalVariable *NewDGV =
    new GlobalVariable(*DstM, TypeMap.get(SGV->getType()->getElementType()),
                       SGV->isConstant(), SGV->getLinkage(), /*init*/0,
                       SGV->getName(), /*insertbefore*/0,
                       SGV->getThreadLocalMode(),
                       SGV->getType()->getAddressSpace());
  // Propagate alignment, visibility and section info.
  copyGVAttributes(NewDGV, SGV);
  if (NewVisibility)
    NewDGV->setVisibility(*NewVisibility);

  if (DGV) {
    DGV->replaceAllUsesWith(ConstantExpr::getBitCast(NewDGV, DGV->getType()));
    DGV->eraseFromParent();
  }
  
  // Make sure to remember this mapping.
  ValueMap[SGV] = NewDGV;
  return false;
}

/// linkFunctionProto - Link the function in the source module into the
/// destination module if needed, setting up mapping information.
bool ModuleLinker::linkFunctionProto(Function *SF) {
  GlobalValue *DGV = getLinkedToGlobal(SF);
  llvm::Optional<GlobalValue::VisibilityTypes> NewVisibility;

  if (DGV) {
    GlobalValue::LinkageTypes NewLinkage = GlobalValue::InternalLinkage;
    bool LinkFromSrc = false;
    GlobalValue::VisibilityTypes NV;
    if (getLinkageResult(DGV, SF, NewLinkage, NV, LinkFromSrc))
      return true;
    NewVisibility = NV;

    if (!LinkFromSrc) {
      // Set calculated linkage
      DGV->setLinkage(NewLinkage);
      DGV->setVisibility(*NewVisibility);

      // Make sure to remember this mapping.
      ValueMap[SF] = ConstantExpr::getBitCast(DGV, TypeMap.get(SF->getType()));
      
      // Track the function from the source module so we don't attempt to remap 
      // it.
      DoNotLinkFromSource.insert(SF);
      
      return false;
    }
  }
  
  // If there is no linkage to be performed or we are linking from the source,
  // bring SF over.
  Function *NewDF = Function::Create(TypeMap.get(SF->getFunctionType()),
                                     SF->getLinkage(), SF->getName(), DstM);
  copyGVAttributes(NewDF, SF);
  if (NewVisibility)
    NewDF->setVisibility(*NewVisibility);

  if (DGV) {
    // Any uses of DF need to change to NewDF, with cast.
    DGV->replaceAllUsesWith(ConstantExpr::getBitCast(NewDF, DGV->getType()));
    DGV->eraseFromParent();
  } else {
    // Internal, LO_ODR, or LO linkage - stick in set to ignore and lazily link.
    if (SF->hasLocalLinkage() || SF->hasLinkOnceLinkage() ||
        SF->hasAvailableExternallyLinkage()) {
      DoNotLinkFromSource.insert(SF);
      LazilyLinkFunctions.push_back(SF);
    }
  }
  
  ValueMap[SF] = NewDF;
  return false;
}

/// LinkAliasProto - Set up prototypes for any aliases that come over from the
/// source module.
bool ModuleLinker::linkAliasProto(GlobalAlias *SGA) {
  GlobalValue *DGV = getLinkedToGlobal(SGA);
  llvm::Optional<GlobalValue::VisibilityTypes> NewVisibility;

  if (DGV) {
    GlobalValue::LinkageTypes NewLinkage = GlobalValue::InternalLinkage;
    GlobalValue::VisibilityTypes NV;
    bool LinkFromSrc = false;
    if (getLinkageResult(DGV, SGA, NewLinkage, NV, LinkFromSrc))
      return true;
    NewVisibility = NV;

    if (!LinkFromSrc) {
      // Set calculated linkage.
      DGV->setLinkage(NewLinkage);
      DGV->setVisibility(*NewVisibility);

      // Make sure to remember this mapping.
      ValueMap[SGA] = ConstantExpr::getBitCast(DGV,TypeMap.get(SGA->getType()));
      
      // Track the alias from the source module so we don't attempt to remap it.
      DoNotLinkFromSource.insert(SGA);
      
      return false;
    }
  }
  
  // If there is no linkage to be performed or we're linking from the source,
  // bring over SGA.
  GlobalAlias *NewDA = new GlobalAlias(TypeMap.get(SGA->getType()),
                                       SGA->getLinkage(), SGA->getName(),
                                       /*aliasee*/0, DstM);
  copyGVAttributes(NewDA, SGA);
  if (NewVisibility)
    NewDA->setVisibility(*NewVisibility);

  if (DGV) {
    // Any uses of DGV need to change to NewDA, with cast.
    DGV->replaceAllUsesWith(ConstantExpr::getBitCast(NewDA, DGV->getType()));
    DGV->eraseFromParent();
  }
  
  ValueMap[SGA] = NewDA;
  return false;
}

static void getArrayElements(Constant *C, SmallVectorImpl<Constant*> &Dest) {
  unsigned NumElements = cast<ArrayType>(C->getType())->getNumElements();

  for (unsigned i = 0; i != NumElements; ++i)
    Dest.push_back(C->getAggregateElement(i));
}
                             
void ModuleLinker::linkAppendingVarInit(const AppendingVarInfo &AVI) {
  // Merge the initializer.
  SmallVector<Constant*, 16> Elements;
  getArrayElements(AVI.DstInit, Elements);
  
  Constant *SrcInit = MapValue(AVI.SrcInit, ValueMap, RF_None, &TypeMap);
  getArrayElements(SrcInit, Elements);
  
  ArrayType *NewType = cast<ArrayType>(AVI.NewGV->getType()->getElementType());
  AVI.NewGV->setInitializer(ConstantArray::get(NewType, Elements));
}

/// linkGlobalInits - Update the initializers in the Dest module now that all
/// globals that may be referenced are in Dest.
void ModuleLinker::linkGlobalInits() {
  // Loop over all of the globals in the src module, mapping them over as we go
  for (Module::const_global_iterator I = SrcM->global_begin(),
       E = SrcM->global_end(); I != E; ++I) {
    
    // Only process initialized GV's or ones not already in dest.
    if (!I->hasInitializer() || DoNotLinkFromSource.count(I)) continue;          
    
    // Grab destination global variable.
    GlobalVariable *DGV = cast<GlobalVariable>(ValueMap[I]);
    // Figure out what the initializer looks like in the dest module.
    DGV->setInitializer(MapValue(I->getInitializer(), ValueMap,
                                 RF_None, &TypeMap));
  }
}

/// linkFunctionBody - Copy the source function over into the dest function and
/// fix up references to values.  At this point we know that Dest is an external
/// function, and that Src is not.
void ModuleLinker::linkFunctionBody(Function *Dst, Function *Src) {
  assert(Src && Dst && Dst->isDeclaration() && !Src->isDeclaration());

  // Go through and convert function arguments over, remembering the mapping.
  Function::arg_iterator DI = Dst->arg_begin();
  for (Function::arg_iterator I = Src->arg_begin(), E = Src->arg_end();
       I != E; ++I, ++DI) {
    DI->setName(I->getName());  // Copy the name over.

    // Add a mapping to our mapping.
    ValueMap[I] = DI;
  }

  if (Mode == Linker::DestroySource) {
    // Splice the body of the source function into the dest function.
    Dst->getBasicBlockList().splice(Dst->end(), Src->getBasicBlockList());
    
    // At this point, all of the instructions and values of the function are now
    // copied over.  The only problem is that they are still referencing values in
    // the Source function as operands.  Loop through all of the operands of the
    // functions and patch them up to point to the local versions.
    for (Function::iterator BB = Dst->begin(), BE = Dst->end(); BB != BE; ++BB)
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
        RemapInstruction(I, ValueMap, RF_IgnoreMissingEntries, &TypeMap);
    
  } else {
    // Clone the body of the function into the dest function.
    SmallVector<ReturnInst*, 8> Returns; // Ignore returns.
    CloneFunctionInto(Dst, Src, ValueMap, false, Returns, "", NULL, &TypeMap);
  }
  
  // There is no need to map the arguments anymore.
  for (Function::arg_iterator I = Src->arg_begin(), E = Src->arg_end();
       I != E; ++I)
    ValueMap.erase(I);
  
}

/// linkAliasBodies - Insert all of the aliases in Src into the Dest module.
void ModuleLinker::linkAliasBodies() {
  for (Module::alias_iterator I = SrcM->alias_begin(), E = SrcM->alias_end();
       I != E; ++I) {
    if (DoNotLinkFromSource.count(I))
      continue;
    if (Constant *Aliasee = I->getAliasee()) {
      GlobalAlias *DA = cast<GlobalAlias>(ValueMap[I]);
      DA->setAliasee(MapValue(Aliasee, ValueMap, RF_None, &TypeMap));
    }
  }
}

/// linkNamedMDNodes - Insert all of the named MDNodes in Src into the Dest
/// module.
void ModuleLinker::linkNamedMDNodes() {
  const NamedMDNode *SrcModFlags = SrcM->getModuleFlagsMetadata();
  for (Module::const_named_metadata_iterator I = SrcM->named_metadata_begin(),
       E = SrcM->named_metadata_end(); I != E; ++I) {
    // Don't link module flags here. Do them separately.
    if (&*I == SrcModFlags) continue;
    NamedMDNode *DestNMD = DstM->getOrInsertNamedMetadata(I->getName());
    // Add Src elements into Dest node.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
      DestNMD->addOperand(MapValue(I->getOperand(i), ValueMap,
                                   RF_None, &TypeMap));
  }
}

/// categorizeModuleFlagNodes - Categorize the module flags according to their
/// type: Error, Warning, Override, and Require.
bool ModuleLinker::
categorizeModuleFlagNodes(const NamedMDNode *ModFlags,
                          DenseMap<MDString*, MDNode*> &ErrorNode,
                          DenseMap<MDString*, MDNode*> &WarningNode,
                          DenseMap<MDString*, MDNode*> &OverrideNode,
                          DenseMap<MDString*,
                            SmallSetVector<MDNode*, 8> > &RequireNodes,
                          SmallSetVector<MDString*, 16> &SeenIDs) {
  bool HasErr = false;

  for (unsigned I = 0, E = ModFlags->getNumOperands(); I != E; ++I) {
    MDNode *Op = ModFlags->getOperand(I);
    assert(Op->getNumOperands() == 3 && "Invalid module flag metadata!");
    assert(isa<ConstantInt>(Op->getOperand(0)) &&
           "Module flag's first operand must be an integer!");
    assert(isa<MDString>(Op->getOperand(1)) &&
           "Module flag's second operand must be an MDString!");

    ConstantInt *Behavior = cast<ConstantInt>(Op->getOperand(0));
    MDString *ID = cast<MDString>(Op->getOperand(1));
    Value *Val = Op->getOperand(2);
    switch (Behavior->getZExtValue()) {
    default:
      assert(false && "Invalid behavior in module flag metadata!");
      break;
    case Module::Error: {
      MDNode *&ErrNode = ErrorNode[ID];
      if (!ErrNode) ErrNode = Op;
      if (ErrNode->getOperand(2) != Val)
        HasErr = emitError("linking module flags '" + ID->getString() +
                           "': IDs have conflicting values");
      break;
    }
    case Module::Warning: {
      MDNode *&WarnNode = WarningNode[ID];
      if (!WarnNode) WarnNode = Op;
      if (WarnNode->getOperand(2) != Val)
        errs() << "WARNING: linking module flags '" << ID->getString()
               << "': IDs have conflicting values";
      break;
    }
    case Module::Require:  RequireNodes[ID].insert(Op);     break;
    case Module::Override: {
      MDNode *&OvrNode = OverrideNode[ID];
      if (!OvrNode) OvrNode = Op;
      if (OvrNode->getOperand(2) != Val)
        HasErr = emitError("linking module flags '" + ID->getString() +
                           "': IDs have conflicting override values");
      break;
    }
    }

    SeenIDs.insert(ID);
  }

  return HasErr;
}

/// linkModuleFlagsMetadata - Merge the linker flags in Src into the Dest
/// module.
bool ModuleLinker::linkModuleFlagsMetadata() {
  const NamedMDNode *SrcModFlags = SrcM->getModuleFlagsMetadata();
  if (!SrcModFlags) return false;

  NamedMDNode *DstModFlags = DstM->getOrInsertModuleFlagsMetadata();

  // If the destination module doesn't have module flags yet, then just copy
  // over the source module's flags.
  if (DstModFlags->getNumOperands() == 0) {
    for (unsigned I = 0, E = SrcModFlags->getNumOperands(); I != E; ++I)
      DstModFlags->addOperand(SrcModFlags->getOperand(I));

    return false;
  }

  bool HasErr = false;

  // Otherwise, we have to merge them based on their behaviors. First,
  // categorize all of the nodes in the modules' module flags. If an error or
  // warning occurs, then emit the appropriate message(s).
  DenseMap<MDString*, MDNode*> ErrorNode;
  DenseMap<MDString*, MDNode*> WarningNode;
  DenseMap<MDString*, MDNode*> OverrideNode;
  DenseMap<MDString*, SmallSetVector<MDNode*, 8> > RequireNodes;
  SmallSetVector<MDString*, 16> SeenIDs;

  HasErr |= categorizeModuleFlagNodes(SrcModFlags, ErrorNode, WarningNode,
                                      OverrideNode, RequireNodes, SeenIDs);
  HasErr |= categorizeModuleFlagNodes(DstModFlags, ErrorNode, WarningNode,
                                      OverrideNode, RequireNodes, SeenIDs);

  // Check that there isn't both an error and warning node for a flag.
  for (SmallSetVector<MDString*, 16>::iterator
         I = SeenIDs.begin(), E = SeenIDs.end(); I != E; ++I) {
    MDString *ID = *I;
    if (ErrorNode[ID] && WarningNode[ID])
      HasErr = emitError("linking module flags '" + ID->getString() +
                         "': IDs have conflicting behaviors");
  }

  // Early exit if we had an error.
  if (HasErr) return true;

  // Get the destination's module flags ready for new operands.
  DstModFlags->dropAllReferences();

  // Add all of the module flags to the destination module.
  DenseMap<MDString*, SmallVector<MDNode*, 4> > AddedNodes;
  for (SmallSetVector<MDString*, 16>::iterator
         I = SeenIDs.begin(), E = SeenIDs.end(); I != E; ++I) {
    MDString *ID = *I;
    if (OverrideNode[ID]) {
      DstModFlags->addOperand(OverrideNode[ID]);
      AddedNodes[ID].push_back(OverrideNode[ID]);
    } else if (ErrorNode[ID]) {
      DstModFlags->addOperand(ErrorNode[ID]);
      AddedNodes[ID].push_back(ErrorNode[ID]);
    } else if (WarningNode[ID]) {
      DstModFlags->addOperand(WarningNode[ID]);
      AddedNodes[ID].push_back(WarningNode[ID]);
    }

    for (SmallSetVector<MDNode*, 8>::iterator
           II = RequireNodes[ID].begin(), IE = RequireNodes[ID].end();
         II != IE; ++II)
      DstModFlags->addOperand(*II);
  }

  // Now check that all of the requirements have been satisfied.
  for (SmallSetVector<MDString*, 16>::iterator
         I = SeenIDs.begin(), E = SeenIDs.end(); I != E; ++I) {
    MDString *ID = *I;
    SmallSetVector<MDNode*, 8> &Set = RequireNodes[ID];

    for (SmallSetVector<MDNode*, 8>::iterator
           II = Set.begin(), IE = Set.end(); II != IE; ++II) {
      MDNode *Node = *II;
      assert(isa<MDNode>(Node->getOperand(2)) &&
             "Module flag's third operand must be an MDNode!");
      MDNode *Val = cast<MDNode>(Node->getOperand(2));

      MDString *ReqID = cast<MDString>(Val->getOperand(0));
      Value *ReqVal = Val->getOperand(1);

      bool HasValue = false;
      for (SmallVectorImpl<MDNode*>::iterator
             RI = AddedNodes[ReqID].begin(), RE = AddedNodes[ReqID].end();
           RI != RE; ++RI) {
        MDNode *ReqNode = *RI;
        if (ReqNode->getOperand(2) == ReqVal) {
          HasValue = true;
          break;
        }
      }

      if (!HasValue)
        HasErr = emitError("linking module flags '" + ReqID->getString() +
                           "': does not have the required value");
    }
  }

  return HasErr;
}
  
bool ModuleLinker::run() {
  assert(DstM && "Null destination module");
  assert(SrcM && "Null source module");

  // Inherit the target data from the source module if the destination module
  // doesn't have one already.
  if (DstM->getDataLayout().empty() && !SrcM->getDataLayout().empty())
    DstM->setDataLayout(SrcM->getDataLayout());

  // Copy the target triple from the source to dest if the dest's is empty.
  if (DstM->getTargetTriple().empty() && !SrcM->getTargetTriple().empty())
    DstM->setTargetTriple(SrcM->getTargetTriple());

  if (!SrcM->getDataLayout().empty() && !DstM->getDataLayout().empty() &&
      SrcM->getDataLayout() != DstM->getDataLayout())
    errs() << "WARNING: Linking two modules of different data layouts!\n";
  if (!SrcM->getTargetTriple().empty() &&
      DstM->getTargetTriple() != SrcM->getTargetTriple()) {
    errs() << "WARNING: Linking two modules of different target triples: ";
    if (!SrcM->getModuleIdentifier().empty())
      errs() << SrcM->getModuleIdentifier() << ": ";
    errs() << "'" << SrcM->getTargetTriple() << "' and '" 
           << DstM->getTargetTriple() << "'\n";
  }

  // Append the module inline asm string.
  if (!SrcM->getModuleInlineAsm().empty()) {
    if (DstM->getModuleInlineAsm().empty())
      DstM->setModuleInlineAsm(SrcM->getModuleInlineAsm());
    else
      DstM->setModuleInlineAsm(DstM->getModuleInlineAsm()+"\n"+
                               SrcM->getModuleInlineAsm());
  }

  // Update the destination module's dependent libraries list with the libraries
  // from the source module. There's no opportunity for duplicates here as the
  // Module ensures that duplicate insertions are discarded.
  for (Module::lib_iterator SI = SrcM->lib_begin(), SE = SrcM->lib_end();
       SI != SE; ++SI)
    DstM->addLibrary(*SI);
  
  // If the source library's module id is in the dependent library list of the
  // destination library, remove it since that module is now linked in.
  StringRef ModuleId = SrcM->getModuleIdentifier();
  if (!ModuleId.empty())
    DstM->removeLibrary(sys::path::stem(ModuleId));
  
  // Loop over all of the linked values to compute type mappings.
  computeTypeMapping();

  // Insert all of the globals in src into the DstM module... without linking
  // initializers (which could refer to functions not yet mapped over).
  for (Module::global_iterator I = SrcM->global_begin(),
       E = SrcM->global_end(); I != E; ++I)
    if (linkGlobalProto(I))
      return true;

  // Link the functions together between the two modules, without doing function
  // bodies... this just adds external function prototypes to the DstM
  // function...  We do this so that when we begin processing function bodies,
  // all of the global values that may be referenced are available in our
  // ValueMap.
  for (Module::iterator I = SrcM->begin(), E = SrcM->end(); I != E; ++I)
    if (linkFunctionProto(I))
      return true;

  // If there were any aliases, link them now.
  for (Module::alias_iterator I = SrcM->alias_begin(),
       E = SrcM->alias_end(); I != E; ++I)
    if (linkAliasProto(I))
      return true;

  for (unsigned i = 0, e = AppendingVars.size(); i != e; ++i)
    linkAppendingVarInit(AppendingVars[i]);
  
  // Update the initializers in the DstM module now that all globals that may
  // be referenced are in DstM.
  linkGlobalInits();

  // Link in the function bodies that are defined in the source module into
  // DstM.
  for (Module::iterator SF = SrcM->begin(), E = SrcM->end(); SF != E; ++SF) {
    // Skip if not linking from source.
    if (DoNotLinkFromSource.count(SF)) continue;
    
    // Skip if no body (function is external) or materialize.
    if (SF->isDeclaration()) {
      if (!SF->isMaterializable())
        continue;
      if (SF->Materialize(&ErrorMsg))
        return true;
    }
    
    linkFunctionBody(cast<Function>(ValueMap[SF]), SF);
    SF->Dematerialize();
  }

  // Resolve all uses of aliases with aliasees.
  linkAliasBodies();

  // Remap all of the named MDNodes in Src into the DstM module. We do this
  // after linking GlobalValues so that MDNodes that reference GlobalValues
  // are properly remapped.
  linkNamedMDNodes();

  // Merge the module flags into the DstM module.
  if (linkModuleFlagsMetadata())
    return true;

  // Process vector of lazily linked in functions.
  bool LinkedInAnyFunctions;
  do {
    LinkedInAnyFunctions = false;
    
    for(std::vector<Function*>::iterator I = LazilyLinkFunctions.begin(),
        E = LazilyLinkFunctions.end(); I != E; ++I) {
      if (!*I)
        continue;
      
      Function *SF = *I;
      Function *DF = cast<Function>(ValueMap[SF]);
      
      if (!DF->use_empty()) {
        
        // Materialize if necessary.
        if (SF->isDeclaration()) {
          if (!SF->isMaterializable())
            continue;
          if (SF->Materialize(&ErrorMsg))
            return true;
        }
        
        // Link in function body.
        linkFunctionBody(DF, SF);
        SF->Dematerialize();

        // "Remove" from vector by setting the element to 0.
        *I = 0;
        
        // Set flag to indicate we may have more functions to lazily link in
        // since we linked in a function.
        LinkedInAnyFunctions = true;
      }
    }
  } while (LinkedInAnyFunctions);
  
  // Remove any prototypes of functions that were not actually linked in.
  for(std::vector<Function*>::iterator I = LazilyLinkFunctions.begin(),
      E = LazilyLinkFunctions.end(); I != E; ++I) {
    if (!*I)
      continue;
    
    Function *SF = *I;
    Function *DF = cast<Function>(ValueMap[SF]);
    if (DF->use_empty())
      DF->eraseFromParent();
  }
  
  // Now that all of the types from the source are used, resolve any structs
  // copied over to the dest that didn't exist there.
  TypeMap.linkDefinedTypeBodies();
  
  return false;
}

//===----------------------------------------------------------------------===//
// LinkModules entrypoint.
//===----------------------------------------------------------------------===//

/// LinkModules - This function links two modules together, with the resulting
/// left module modified to be the composite of the two input modules.  If an
/// error occurs, true is returned and ErrorMsg (if not null) is set to indicate
/// the problem.  Upon failure, the Dest module could be in a modified state,
/// and shouldn't be relied on to be consistent.
bool Linker::LinkModules(Module *Dest, Module *Src, unsigned Mode, 
                         std::string *ErrorMsg) {
  ModuleLinker TheLinker(Dest, Src, Mode);
  if (TheLinker.run()) {
    if (ErrorMsg) *ErrorMsg = TheLinker.ErrorMsg;
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// C API.
//===----------------------------------------------------------------------===//

LLVMBool LLVMLinkModules(LLVMModuleRef Dest, LLVMModuleRef Src,
                         LLVMLinkerMode Mode, char **OutMessages) {
  std::string Messages;
  LLVMBool Result = Linker::LinkModules(unwrap(Dest), unwrap(Src),
                                        Mode, OutMessages? &Messages : 0);
  if (OutMessages)
    *OutMessages = strdup(Messages.c_str());
  return Result;
}
