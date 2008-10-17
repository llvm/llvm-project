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
// Specifically, this:
//  * Merges global variables between the two modules
//    * Uninit + Uninit = Init, Init + Uninit = Init, Init + Init = Error if !=
//  * Merges functions between two modules
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/Instructions.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Path.h"
#include "llvm/ADT/DenseMap.h"
#include <sstream>
using namespace llvm;

// Error - Simple wrapper function to conditionally assign to E and return true.
// This just makes error return conditions a little bit simpler...
static inline bool Error(std::string *E, const std::string &Message) {
  if (E) *E = Message;
  return true;
}

// Function: ResolveTypes()
//
// Description:
//  Attempt to link the two specified types together.
//
// Inputs:
//  DestTy - The type to which we wish to resolve.
//  SrcTy  - The original type which we want to resolve.
//
// Outputs:
//  DestST - The symbol table in which the new type should be placed.
//
// Return value:
//  true  - There is an error and the types cannot yet be linked.
//  false - No errors.
//
static bool ResolveTypes(const Type *DestTy, const Type *SrcTy) {
  if (DestTy == SrcTy) return false;       // If already equal, noop
  assert(DestTy && SrcTy && "Can't handle null types");

  if (const OpaqueType *OT = dyn_cast<OpaqueType>(DestTy)) {
    // Type _is_ in module, just opaque...
    const_cast<OpaqueType*>(OT)->refineAbstractTypeTo(SrcTy);
  } else if (const OpaqueType *OT = dyn_cast<OpaqueType>(SrcTy)) {
    const_cast<OpaqueType*>(OT)->refineAbstractTypeTo(DestTy);
  } else {
    return true;  // Cannot link types... not-equal and neither is opaque.
  }
  return false;
}

/// LinkerTypeMap - This implements a map of types that is stable
/// even if types are resolved/refined to other types.  This is not a general
/// purpose map, it is specific to the linker's use.
namespace {
class LinkerTypeMap : public AbstractTypeUser {
  typedef DenseMap<const Type*, PATypeHolder> TheMapTy;
  TheMapTy TheMap;
  
  LinkerTypeMap(const LinkerTypeMap&); // DO NOT IMPLEMENT
  void operator=(const LinkerTypeMap&); // DO NOT IMPLEMENT
public:
  LinkerTypeMap() {}
  ~LinkerTypeMap() {
    for (DenseMap<const Type*, PATypeHolder>::iterator I = TheMap.begin(),
         E = TheMap.end(); I != E; ++I)
      I->first->removeAbstractTypeUser(this);
  }
  
  /// lookup - Return the value for the specified type or null if it doesn't
  /// exist.
  const Type *lookup(const Type *Ty) const {
    TheMapTy::const_iterator I = TheMap.find(Ty);
    if (I != TheMap.end()) return I->second;
    return 0;
  }
  
  /// erase - Remove the specified type, returning true if it was in the set.
  bool erase(const Type *Ty) {
    if (!TheMap.erase(Ty))
      return false;
    if (Ty->isAbstract())
      Ty->removeAbstractTypeUser(this);
    return true;
  }
  
  /// insert - This returns true if the pointer was new to the set, false if it
  /// was already in the set.
  bool insert(const Type *Src, const Type *Dst) {
    if (!TheMap.insert(std::make_pair(Src, PATypeHolder(Dst))).second)
      return false;  // Already in map.
    if (Src->isAbstract())
      Src->addAbstractTypeUser(this);
    return true;
  }
  
protected:
  /// refineAbstractType - The callback method invoked when an abstract type is
  /// resolved to another type.  An object must override this method to update
  /// its internal state to reference NewType instead of OldType.
  ///
  virtual void refineAbstractType(const DerivedType *OldTy,
                                  const Type *NewTy) {
    TheMapTy::iterator I = TheMap.find(OldTy);
    const Type *DstTy = I->second;
    
    TheMap.erase(I);
    if (OldTy->isAbstract())
      OldTy->removeAbstractTypeUser(this);

    // Don't reinsert into the map if the key is concrete now.
    if (NewTy->isAbstract())
      insert(NewTy, DstTy);
  }
  
  /// The other case which AbstractTypeUsers must be aware of is when a type
  /// makes the transition from being abstract (where it has clients on it's
  /// AbstractTypeUsers list) to concrete (where it does not).  This method
  /// notifies ATU's when this occurs for a type.
  virtual void typeBecameConcrete(const DerivedType *AbsTy) {
    TheMap.erase(AbsTy);
    AbsTy->removeAbstractTypeUser(this);
  }
  
  // for debugging...
  virtual void dump() const {
    cerr << "AbstractTypeSet!\n";
  }
};
}


// RecursiveResolveTypes - This is just like ResolveTypes, except that it
// recurses down into derived types, merging the used types if the parent types
// are compatible.
static bool RecursiveResolveTypesI(const Type *DstTy, const Type *SrcTy,
                                   LinkerTypeMap &Pointers) {
  if (DstTy == SrcTy) return false;       // If already equal, noop

  // If we found our opaque type, resolve it now!
  if (isa<OpaqueType>(DstTy) || isa<OpaqueType>(SrcTy))
    return ResolveTypes(DstTy, SrcTy);

  // Two types cannot be resolved together if they are of different primitive
  // type.  For example, we cannot resolve an int to a float.
  if (DstTy->getTypeID() != SrcTy->getTypeID()) return true;

  // If neither type is abstract, then they really are just different types.
  if (!DstTy->isAbstract() && !SrcTy->isAbstract())
    return true;
  
  // Otherwise, resolve the used type used by this derived type...
  switch (DstTy->getTypeID()) {
  default:
    return true;
  case Type::FunctionTyID: {
    const FunctionType *DstFT = cast<FunctionType>(DstTy);
    const FunctionType *SrcFT = cast<FunctionType>(SrcTy);
    if (DstFT->isVarArg() != SrcFT->isVarArg() ||
        DstFT->getNumContainedTypes() != SrcFT->getNumContainedTypes())
      return true;
    
    // Use TypeHolder's so recursive resolution won't break us.
    PATypeHolder ST(SrcFT), DT(DstFT);
    for (unsigned i = 0, e = DstFT->getNumContainedTypes(); i != e; ++i) {
      const Type *SE = ST->getContainedType(i), *DE = DT->getContainedType(i);
      if (SE != DE && RecursiveResolveTypesI(DE, SE, Pointers))
        return true;
    }
    return false;
  }
  case Type::StructTyID: {
    const StructType *DstST = cast<StructType>(DstTy);
    const StructType *SrcST = cast<StructType>(SrcTy);
    if (DstST->getNumContainedTypes() != SrcST->getNumContainedTypes())
      return true;
    
    PATypeHolder ST(SrcST), DT(DstST);
    for (unsigned i = 0, e = DstST->getNumContainedTypes(); i != e; ++i) {
      const Type *SE = ST->getContainedType(i), *DE = DT->getContainedType(i);
      if (SE != DE && RecursiveResolveTypesI(DE, SE, Pointers))
        return true;
    }
    return false;
  }
  case Type::ArrayTyID: {
    const ArrayType *DAT = cast<ArrayType>(DstTy);
    const ArrayType *SAT = cast<ArrayType>(SrcTy);
    if (DAT->getNumElements() != SAT->getNumElements()) return true;
    return RecursiveResolveTypesI(DAT->getElementType(), SAT->getElementType(),
                                  Pointers);
  }
  case Type::VectorTyID: {
    const VectorType *DVT = cast<VectorType>(DstTy);
    const VectorType *SVT = cast<VectorType>(SrcTy);
    if (DVT->getNumElements() != SVT->getNumElements()) return true;
    return RecursiveResolveTypesI(DVT->getElementType(), SVT->getElementType(),
                                  Pointers);
  }
  case Type::PointerTyID: {
    const PointerType *DstPT = cast<PointerType>(DstTy);
    const PointerType *SrcPT = cast<PointerType>(SrcTy);
    
    if (DstPT->getAddressSpace() != SrcPT->getAddressSpace())
      return true;
    
    // If this is a pointer type, check to see if we have already seen it.  If
    // so, we are in a recursive branch.  Cut off the search now.  We cannot use
    // an associative container for this search, because the type pointers (keys
    // in the container) change whenever types get resolved.
    if (SrcPT->isAbstract())
      if (const Type *ExistingDestTy = Pointers.lookup(SrcPT))
        return ExistingDestTy != DstPT;
    
    if (DstPT->isAbstract())
      if (const Type *ExistingSrcTy = Pointers.lookup(DstPT))
        return ExistingSrcTy != SrcPT;
    // Otherwise, add the current pointers to the vector to stop recursion on
    // this pair.
    if (DstPT->isAbstract())
      Pointers.insert(DstPT, SrcPT);
    if (SrcPT->isAbstract())
      Pointers.insert(SrcPT, DstPT);
    
    return RecursiveResolveTypesI(DstPT->getElementType(),
                                  SrcPT->getElementType(), Pointers);
  }
  }
}

static bool RecursiveResolveTypes(const Type *DestTy, const Type *SrcTy) {
  LinkerTypeMap PointerTypes;
  return RecursiveResolveTypesI(DestTy, SrcTy, PointerTypes);
}


// LinkTypes - Go through the symbol table of the Src module and see if any
// types are named in the src module that are not named in the Dst module.
// Make sure there are no type name conflicts.
static bool LinkTypes(Module *Dest, const Module *Src, std::string *Err) {
        TypeSymbolTable *DestST = &Dest->getTypeSymbolTable();
  const TypeSymbolTable *SrcST  = &Src->getTypeSymbolTable();

  // Look for a type plane for Type's...
  TypeSymbolTable::const_iterator TI = SrcST->begin();
  TypeSymbolTable::const_iterator TE = SrcST->end();
  if (TI == TE) return false;  // No named types, do nothing.

  // Some types cannot be resolved immediately because they depend on other
  // types being resolved to each other first.  This contains a list of types we
  // are waiting to recheck.
  std::vector<std::string> DelayedTypesToResolve;

  for ( ; TI != TE; ++TI ) {
    const std::string &Name = TI->first;
    const Type *RHS = TI->second;

    // Check to see if this type name is already in the dest module.
    Type *Entry = DestST->lookup(Name);

    // If the name is just in the source module, bring it over to the dest.
    if (Entry == 0) {
      if (!Name.empty())
        DestST->insert(Name, const_cast<Type*>(RHS));
    } else if (ResolveTypes(Entry, RHS)) {
      // They look different, save the types 'till later to resolve.
      DelayedTypesToResolve.push_back(Name);
    }
  }

  // Iteratively resolve types while we can...
  while (!DelayedTypesToResolve.empty()) {
    // Loop over all of the types, attempting to resolve them if possible...
    unsigned OldSize = DelayedTypesToResolve.size();

    // Try direct resolution by name...
    for (unsigned i = 0; i != DelayedTypesToResolve.size(); ++i) {
      const std::string &Name = DelayedTypesToResolve[i];
      Type *T1 = SrcST->lookup(Name);
      Type *T2 = DestST->lookup(Name);
      if (!ResolveTypes(T2, T1)) {
        // We are making progress!
        DelayedTypesToResolve.erase(DelayedTypesToResolve.begin()+i);
        --i;
      }
    }

    // Did we not eliminate any types?
    if (DelayedTypesToResolve.size() == OldSize) {
      // Attempt to resolve subelements of types.  This allows us to merge these
      // two types: { int* } and { opaque* }
      for (unsigned i = 0, e = DelayedTypesToResolve.size(); i != e; ++i) {
        const std::string &Name = DelayedTypesToResolve[i];
        if (!RecursiveResolveTypes(SrcST->lookup(Name), DestST->lookup(Name))) {
          // We are making progress!
          DelayedTypesToResolve.erase(DelayedTypesToResolve.begin()+i);

          // Go back to the main loop, perhaps we can resolve directly by name
          // now...
          break;
        }
      }

      // If we STILL cannot resolve the types, then there is something wrong.
      if (DelayedTypesToResolve.size() == OldSize) {
        // Remove the symbol name from the destination.
        DelayedTypesToResolve.pop_back();
      }
    }
  }


  return false;
}

#ifndef NDEBUG
static void PrintMap(const std::map<const Value*, Value*> &M) {
  for (std::map<const Value*, Value*>::const_iterator I = M.begin(), E =M.end();
       I != E; ++I) {
    cerr << " Fr: " << (void*)I->first << " ";
    I->first->dump();
    cerr << " To: " << (void*)I->second << " ";
    I->second->dump();
    cerr << "\n";
  }
}
#endif


// RemapOperand - Use ValueMap to convert constants from one module to another.
static Value *RemapOperand(const Value *In,
                           std::map<const Value*, Value*> &ValueMap) {
  std::map<const Value*,Value*>::const_iterator I = ValueMap.find(In);
  if (I != ValueMap.end()) 
    return I->second;

  // Check to see if it's a constant that we are interested in transforming.
  Value *Result = 0;
  if (const Constant *CPV = dyn_cast<Constant>(In)) {
    if ((!isa<DerivedType>(CPV->getType()) && !isa<ConstantExpr>(CPV)) ||
        isa<ConstantInt>(CPV) || isa<ConstantAggregateZero>(CPV))
      return const_cast<Constant*>(CPV);   // Simple constants stay identical.

    if (const ConstantArray *CPA = dyn_cast<ConstantArray>(CPV)) {
      std::vector<Constant*> Operands(CPA->getNumOperands());
      for (unsigned i = 0, e = CPA->getNumOperands(); i != e; ++i)
        Operands[i] =cast<Constant>(RemapOperand(CPA->getOperand(i), ValueMap));
      Result = ConstantArray::get(cast<ArrayType>(CPA->getType()), Operands);
    } else if (const ConstantStruct *CPS = dyn_cast<ConstantStruct>(CPV)) {
      std::vector<Constant*> Operands(CPS->getNumOperands());
      for (unsigned i = 0, e = CPS->getNumOperands(); i != e; ++i)
        Operands[i] =cast<Constant>(RemapOperand(CPS->getOperand(i), ValueMap));
      Result = ConstantStruct::get(cast<StructType>(CPS->getType()), Operands);
    } else if (isa<ConstantPointerNull>(CPV) || isa<UndefValue>(CPV)) {
      Result = const_cast<Constant*>(CPV);
    } else if (const ConstantVector *CP = dyn_cast<ConstantVector>(CPV)) {
      std::vector<Constant*> Operands(CP->getNumOperands());
      for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
        Operands[i] = cast<Constant>(RemapOperand(CP->getOperand(i), ValueMap));
      Result = ConstantVector::get(Operands);
    } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CPV)) {
      std::vector<Constant*> Ops;
      for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i)
        Ops.push_back(cast<Constant>(RemapOperand(CE->getOperand(i),ValueMap)));
      Result = CE->getWithOperands(Ops);
    } else {
      assert(!isa<GlobalValue>(CPV) && "Unmapped global?");
      assert(0 && "Unknown type of derived type constant value!");
    }
  } else if (isa<InlineAsm>(In)) {
    Result = const_cast<Value*>(In);
  }
  
  // Cache the mapping in our local map structure
  if (Result) {
    ValueMap[In] = Result;
    return Result;
  }
  
#ifndef NDEBUG
  cerr << "LinkModules ValueMap: \n";
  PrintMap(ValueMap);

  cerr << "Couldn't remap value: " << (void*)In << " " << *In << "\n";
  assert(0 && "Couldn't remap value!");
#endif
  return 0;
}

/// ForceRenaming - The LLVM SymbolTable class autorenames globals that conflict
/// in the symbol table.  This is good for all clients except for us.  Go
/// through the trouble to force this back.
static void ForceRenaming(GlobalValue *GV, const std::string &Name) {
  assert(GV->getName() != Name && "Can't force rename to self");
  ValueSymbolTable &ST = GV->getParent()->getValueSymbolTable();

  // If there is a conflict, rename the conflict.
  if (GlobalValue *ConflictGV = cast_or_null<GlobalValue>(ST.lookup(Name))) {
    assert(ConflictGV->hasInternalLinkage() &&
           "Not conflicting with a static global, should link instead!");
    GV->takeName(ConflictGV);
    ConflictGV->setName(Name);    // This will cause ConflictGV to get renamed
    assert(ConflictGV->getName() != Name && "ForceRenaming didn't work");
  } else {
    GV->setName(Name);              // Force the name back
  }
}

/// CopyGVAttributes - copy additional attributes (those not needed to construct
/// a GlobalValue) from the SrcGV to the DestGV. 
static void CopyGVAttributes(GlobalValue *DestGV, const GlobalValue *SrcGV) {
  // Use the maximum alignment, rather than just copying the alignment of SrcGV.
  unsigned Alignment = std::max(DestGV->getAlignment(), SrcGV->getAlignment());
  DestGV->copyAttributesFrom(SrcGV);
  DestGV->setAlignment(Alignment);
}

/// GetLinkageResult - This analyzes the two global values and determines what
/// the result will look like in the destination module.  In particular, it
/// computes the resultant linkage type, computes whether the global in the
/// source should be copied over to the destination (replacing the existing
/// one), and computes whether this linkage is an error or not. It also performs
/// visibility checks: we cannot link together two symbols with different
/// visibilities.
static bool GetLinkageResult(GlobalValue *Dest, const GlobalValue *Src,
                             GlobalValue::LinkageTypes &LT, bool &LinkFromSrc,
                             std::string *Err) {
  assert((!Dest || !Src->hasInternalLinkage()) &&
         "If Src has internal linkage, Dest shouldn't be set!");
  if (!Dest) {
    // Linking something to nothing.
    LinkFromSrc = true;
    LT = Src->getLinkage();
  } else if (Src->isDeclaration()) {
    // If Src is external or if both Src & Dest are external..  Just link the
    // external globals, we aren't adding anything.
    if (Src->hasDLLImportLinkage()) {
      // If one of GVs has DLLImport linkage, result should be dllimport'ed.
      if (Dest->isDeclaration()) {
        LinkFromSrc = true;
        LT = Src->getLinkage();
      }      
    } else if (Dest->hasExternalWeakLinkage()) {
      //If the Dest is weak, use the source linkage
      LinkFromSrc = true;
      LT = Src->getLinkage();
    } else {
      LinkFromSrc = false;
      LT = Dest->getLinkage();
    }
  } else if (Dest->isDeclaration() && !Dest->hasDLLImportLinkage()) {
    // If Dest is external but Src is not:
    LinkFromSrc = true;
    LT = Src->getLinkage();
  } else if (Src->hasAppendingLinkage() || Dest->hasAppendingLinkage()) {
    if (Src->getLinkage() != Dest->getLinkage())
      return Error(Err, "Linking globals named '" + Src->getName() +
            "': can only link appending global with another appending global!");
    LinkFromSrc = true; // Special cased.
    LT = Src->getLinkage();
  } else if (Src->mayBeOverridden()) {
    // At this point we know that Dest has LinkOnce, External*, Weak, Common,
    // or DLL* linkage.
    if ((Dest->hasLinkOnceLinkage() &&
          (Src->hasWeakLinkage() || Src->hasCommonLinkage())) ||
        Dest->hasExternalWeakLinkage()) {
      LinkFromSrc = true;
      LT = Src->getLinkage();
    } else {
      LinkFromSrc = false;
      LT = Dest->getLinkage();
    }
  } else if (Dest->mayBeOverridden()) {
    // At this point we know that Src has External* or DLL* linkage.
    if (Src->hasExternalWeakLinkage()) {
      LinkFromSrc = false;
      LT = Dest->getLinkage();
    } else {
      LinkFromSrc = true;
      LT = GlobalValue::ExternalLinkage;
    }
  } else {
    assert((Dest->hasExternalLinkage() ||
            Dest->hasDLLImportLinkage() ||
            Dest->hasDLLExportLinkage() ||
            Dest->hasExternalWeakLinkage()) &&
           (Src->hasExternalLinkage() ||
            Src->hasDLLImportLinkage() ||
            Src->hasDLLExportLinkage() ||
            Src->hasExternalWeakLinkage()) &&
           "Unexpected linkage type!");
    return Error(Err, "Linking globals named '" + Src->getName() +
                 "': symbol multiply defined!");
  }

  // Check visibility
  if (Dest && Src->getVisibility() != Dest->getVisibility())
    if (!Src->isDeclaration() && !Dest->isDeclaration())
      return Error(Err, "Linking globals named '" + Src->getName() +
                   "': symbols have different visibilities!");
  return false;
}

// LinkGlobals - Loop through the global variables in the src module and merge
// them into the dest module.
static bool LinkGlobals(Module *Dest, const Module *Src,
                        std::map<const Value*, Value*> &ValueMap,
                    std::multimap<std::string, GlobalVariable *> &AppendingVars,
                        std::string *Err) {
  ValueSymbolTable &DestSymTab = Dest->getValueSymbolTable();
  
  // Loop over all of the globals in the src module, mapping them over as we go
  for (Module::const_global_iterator I = Src->global_begin(),
       E = Src->global_end(); I != E; ++I) {
    const GlobalVariable *SGV = I;
    GlobalValue *DGV = 0;

    // Check to see if may have to link the global with the global, alias or
    // function.
    if (SGV->hasName() && !SGV->hasInternalLinkage())
      DGV = cast_or_null<GlobalValue>(DestSymTab.lookup(SGV->getNameStart(),
                                                        SGV->getNameEnd()));
    
    // If we found a global with the same name in the dest module, but it has
    // internal linkage, we are really not doing any linkage here.
    if (DGV && DGV->hasInternalLinkage())
      DGV = 0;
    
    // If types don't agree due to opaque types, try to resolve them.
    if (DGV && DGV->getType() != SGV->getType())
      RecursiveResolveTypes(SGV->getType(), DGV->getType());

    assert((SGV->hasInitializer() || SGV->hasExternalWeakLinkage() ||
            SGV->hasExternalLinkage() || SGV->hasDLLImportLinkage()) &&
           "Global must either be external or have an initializer!");

    GlobalValue::LinkageTypes NewLinkage = GlobalValue::InternalLinkage;
    bool LinkFromSrc = false;
    if (GetLinkageResult(DGV, SGV, NewLinkage, LinkFromSrc, Err))
      return true;

    if (DGV == 0) {
      // No linking to be performed, simply create an identical version of the
      // symbol over in the dest module... the initializer will be filled in
      // later by LinkGlobalInits.
      GlobalVariable *NewDGV =
        new GlobalVariable(SGV->getType()->getElementType(),
                           SGV->isConstant(), SGV->getLinkage(), /*init*/0,
                           SGV->getName(), Dest, false,
                           SGV->getType()->getAddressSpace());
      // Propagate alignment, visibility and section info.
      CopyGVAttributes(NewDGV, SGV);

      // If the LLVM runtime renamed the global, but it is an externally visible
      // symbol, DGV must be an existing global with internal linkage.  Rename
      // it.
      if (!NewDGV->hasInternalLinkage() && NewDGV->getName() != SGV->getName())
        ForceRenaming(NewDGV, SGV->getName());

      // Make sure to remember this mapping.
      ValueMap[SGV] = NewDGV;

      // Keep track that this is an appending variable.
      if (SGV->hasAppendingLinkage())
        AppendingVars.insert(std::make_pair(SGV->getName(), NewDGV));
      continue;
    }
    
    // If the visibilities of the symbols disagree and the destination is a
    // prototype, take the visibility of its input.
    if (DGV->isDeclaration())
      DGV->setVisibility(SGV->getVisibility());
    
    if (DGV->hasAppendingLinkage()) {
      // No linking is performed yet.  Just insert a new copy of the global, and
      // keep track of the fact that it is an appending variable in the
      // AppendingVars map.  The name is cleared out so that no linkage is
      // performed.
      GlobalVariable *NewDGV =
        new GlobalVariable(SGV->getType()->getElementType(),
                           SGV->isConstant(), SGV->getLinkage(), /*init*/0,
                           "", Dest, false,
                           SGV->getType()->getAddressSpace());

      // Set alignment allowing CopyGVAttributes merge it with alignment of SGV.
      NewDGV->setAlignment(DGV->getAlignment());
      // Propagate alignment, section and visibility info.
      CopyGVAttributes(NewDGV, SGV);

      // Make sure to remember this mapping...
      ValueMap[SGV] = NewDGV;

      // Keep track that this is an appending variable...
      AppendingVars.insert(std::make_pair(SGV->getName(), NewDGV));
      continue;
    }
    
    if (LinkFromSrc) {
      if (isa<GlobalAlias>(DGV))
        return Error(Err, "Global-Alias Collision on '" + SGV->getName() +
                     "': symbol multiple defined");
      
      // If the types don't match, and if we are to link from the source, nuke
      // DGV and create a new one of the appropriate type.  Note that the thing
      // we are replacing may be a function (if a prototype, weak, etc) or a
      // global variable.
      GlobalVariable *NewDGV =
        new GlobalVariable(SGV->getType()->getElementType(), SGV->isConstant(),
                           NewLinkage, /*init*/0, DGV->getName(), Dest, false,
                           SGV->getType()->getAddressSpace());
      
      // Propagate alignment, section, and visibility info.
      CopyGVAttributes(NewDGV, SGV);
      DGV->replaceAllUsesWith(ConstantExpr::getBitCast(NewDGV, DGV->getType()));
      
      // DGV will conflict with NewDGV because they both had the same
      // name. We must erase this now so ForceRenaming doesn't assert
      // because DGV might not have internal linkage.
      if (GlobalVariable *Var = dyn_cast<GlobalVariable>(DGV))
        Var->eraseFromParent();
      else
        cast<Function>(DGV)->eraseFromParent();
      DGV = NewDGV;

      // If the symbol table renamed the global, but it is an externally visible
      // symbol, DGV must be an existing global with internal linkage.  Rename.
      if (NewDGV->getName() != SGV->getName() && !NewDGV->hasInternalLinkage())
        ForceRenaming(NewDGV, SGV->getName());
      
      // Inherit const as appropriate.
      NewDGV->setConstant(SGV->isConstant());
      
      // Make sure to remember this mapping.
      ValueMap[SGV] = NewDGV;
      continue;
    }
    
    // Not "link from source", keep the one in the DestModule and remap the
    // input onto it.
    
    // Special case for const propagation.
    if (GlobalVariable *DGVar = dyn_cast<GlobalVariable>(DGV))
      if (DGVar->isDeclaration() && SGV->isConstant() && !DGVar->isConstant())
        DGVar->setConstant(true);

    // SGV is global, but DGV is alias. The only valid mapping is when SGV is
    // external declaration, which is effectively a no-op. Also make sure
    // linkage calculation was correct.
    if (isa<GlobalAlias>(DGV) && !SGV->isDeclaration())
      return Error(Err, "Global-Alias Collision on '" + SGV->getName() +
                   "': symbol multiple defined");
    
    // Set calculated linkage
    DGV->setLinkage(NewLinkage);
    
    // Make sure to remember this mapping...
    ValueMap[SGV] = ConstantExpr::getBitCast(DGV, SGV->getType());
  }
  return false;
}

static GlobalValue::LinkageTypes
CalculateAliasLinkage(const GlobalValue *SGV, const GlobalValue *DGV) {
  if (SGV->hasExternalLinkage() || DGV->hasExternalLinkage())
    return GlobalValue::ExternalLinkage;
  else if (SGV->hasWeakLinkage() || DGV->hasWeakLinkage())
    return GlobalValue::WeakLinkage;
  else {
    assert(SGV->hasInternalLinkage() && DGV->hasInternalLinkage() &&
           "Unexpected linkage type");
    return GlobalValue::InternalLinkage;
  }
}

// LinkAlias - Loop through the alias in the src module and link them into the
// dest module. We're assuming, that all functions/global variables were already
// linked in.
static bool LinkAlias(Module *Dest, const Module *Src,
                      std::map<const Value*, Value*> &ValueMap,
                      std::string *Err) {
  // Loop over all alias in the src module
  for (Module::const_alias_iterator I = Src->alias_begin(),
         E = Src->alias_end(); I != E; ++I) {
    const GlobalAlias *SGA = I;
    const GlobalValue *SAliasee = SGA->getAliasedGlobal();
    GlobalAlias *NewGA = NULL;

    // Globals were already linked, thus we can just query ValueMap for variant
    // of SAliasee in Dest.
    std::map<const Value*,Value*>::const_iterator VMI = ValueMap.find(SAliasee);
    assert(VMI != ValueMap.end() && "Aliasee not linked");
    GlobalValue* DAliasee = cast<GlobalValue>(VMI->second);
    GlobalValue* DGV = NULL;

    // Try to find something 'similar' to SGA in destination module.
    if (!DGV && !SGA->hasInternalLinkage()) {
      DGV = Dest->getNamedAlias(SGA->getName());

      // If types don't agree due to opaque types, try to resolve them.
      if (DGV && DGV->getType() != SGA->getType())
        RecursiveResolveTypes(SGA->getType(), DGV->getType());
    }

    if (!DGV && !SGA->hasInternalLinkage()) {
      DGV = Dest->getGlobalVariable(SGA->getName());

      // If types don't agree due to opaque types, try to resolve them.
      if (DGV && DGV->getType() != SGA->getType())
        RecursiveResolveTypes(SGA->getType(), DGV->getType());
    }

    if (!DGV && !SGA->hasInternalLinkage()) {
      DGV = Dest->getFunction(SGA->getName());

      // If types don't agree due to opaque types, try to resolve them.
      if (DGV && DGV->getType() != SGA->getType())
        RecursiveResolveTypes(SGA->getType(), DGV->getType());
    }

    // No linking to be performed on internal stuff.
    if (DGV && DGV->hasInternalLinkage())
      DGV = NULL;

    if (GlobalAlias *DGA = dyn_cast_or_null<GlobalAlias>(DGV)) {
      // Types are known to be the same, check whether aliasees equal. As
      // globals are already linked we just need query ValueMap to find the
      // mapping.
      if (DAliasee == DGA->getAliasedGlobal()) {
        // This is just two copies of the same alias. Propagate linkage, if
        // necessary.
        DGA->setLinkage(CalculateAliasLinkage(SGA, DGA));

        NewGA = DGA;
        // Proceed to 'common' steps
      } else
        return Error(Err, "Alias Collision on '"  + SGA->getName()+
                     "': aliases have different aliasees");
    } else if (GlobalVariable *DGVar = dyn_cast_or_null<GlobalVariable>(DGV)) {
      // The only allowed way is to link alias with external declaration or weak
      // symbol..
      if (DGVar->isDeclaration() || DGVar->mayBeOverridden()) {
        // But only if aliasee is global too...
        if (!isa<GlobalVariable>(DAliasee))
          return Error(Err, "Global-Alias Collision on '" + SGA->getName() +
                       "': aliasee is not global variable");

        NewGA = new GlobalAlias(SGA->getType(), SGA->getLinkage(),
                                SGA->getName(), DAliasee, Dest);
        CopyGVAttributes(NewGA, SGA);

        // Any uses of DGV need to change to NewGA, with cast, if needed.
        if (SGA->getType() != DGVar->getType())
          DGVar->replaceAllUsesWith(ConstantExpr::getBitCast(NewGA,
                                                             DGVar->getType()));
        else
          DGVar->replaceAllUsesWith(NewGA);

        // DGVar will conflict with NewGA because they both had the same
        // name. We must erase this now so ForceRenaming doesn't assert
        // because DGV might not have internal linkage.
        DGVar->eraseFromParent();

        // Proceed to 'common' steps
      } else
        return Error(Err, "Global-Alias Collision on '" + SGA->getName() +
                     "': symbol multiple defined");
    } else if (Function *DF = dyn_cast_or_null<Function>(DGV)) {
      // The only allowed way is to link alias with external declaration or weak
      // symbol...
      if (DF->isDeclaration() || DF->mayBeOverridden()) {
        // But only if aliasee is function too...
        if (!isa<Function>(DAliasee))
          return Error(Err, "Function-Alias Collision on '" + SGA->getName() +
                       "': aliasee is not function");

        NewGA = new GlobalAlias(SGA->getType(), SGA->getLinkage(),
                                SGA->getName(), DAliasee, Dest);
        CopyGVAttributes(NewGA, SGA);

        // Any uses of DF need to change to NewGA, with cast, if needed.
        if (SGA->getType() != DF->getType())
          DF->replaceAllUsesWith(ConstantExpr::getBitCast(NewGA,
                                                          DF->getType()));
        else
          DF->replaceAllUsesWith(NewGA);

        // DF will conflict with NewGA because they both had the same
        // name. We must erase this now so ForceRenaming doesn't assert
        // because DF might not have internal linkage.
        DF->eraseFromParent();

        // Proceed to 'common' steps
      } else
        return Error(Err, "Function-Alias Collision on '" + SGA->getName() +
                     "': symbol multiple defined");
    } else {
      // No linking to be performed, simply create an identical version of the
      // alias over in the dest module...

      NewGA = new GlobalAlias(SGA->getType(), SGA->getLinkage(),
                              SGA->getName(), DAliasee, Dest);
      CopyGVAttributes(NewGA, SGA);

      // Proceed to 'common' steps
    }

    assert(NewGA && "No alias was created in destination module!");

    // If the symbol table renamed the alias, but it is an externally visible
    // symbol, DGA must be an global value with internal linkage. Rename it.
    if (NewGA->getName() != SGA->getName() &&
        !NewGA->hasInternalLinkage())
      ForceRenaming(NewGA, SGA->getName());

    // Remember this mapping so uses in the source module get remapped
    // later by RemapOperand.
    ValueMap[SGA] = NewGA;
  }

  return false;
}


// LinkGlobalInits - Update the initializers in the Dest module now that all
// globals that may be referenced are in Dest.
static bool LinkGlobalInits(Module *Dest, const Module *Src,
                            std::map<const Value*, Value*> &ValueMap,
                            std::string *Err) {
  // Loop over all of the globals in the src module, mapping them over as we go
  for (Module::const_global_iterator I = Src->global_begin(),
       E = Src->global_end(); I != E; ++I) {
    const GlobalVariable *SGV = I;

    if (SGV->hasInitializer()) {      // Only process initialized GV's
      // Figure out what the initializer looks like in the dest module...
      Constant *SInit =
        cast<Constant>(RemapOperand(SGV->getInitializer(), ValueMap));

      GlobalVariable *DGV =
        cast<GlobalVariable>(ValueMap[SGV]->stripPointerCasts());
      if (DGV->hasInitializer()) {
        if (SGV->hasExternalLinkage()) {
          if (DGV->getInitializer() != SInit)
            return Error(Err, "Global Variable Collision on '" + SGV->getName() +
                         "': global variables have different initializers");
        } else if (DGV->mayBeOverridden()) {
          // Nothing is required, mapped values will take the new global
          // automatically.
        } else if (SGV->mayBeOverridden()) {
          // Nothing is required, mapped values will take the new global
          // automatically.
        } else if (DGV->hasAppendingLinkage()) {
          assert(0 && "Appending linkage unimplemented!");
        } else {
          assert(0 && "Unknown linkage!");
        }
      } else {
        // Copy the initializer over now...
        DGV->setInitializer(SInit);
      }
    }
  }
  return false;
}

// LinkFunctionProtos - Link the functions together between the two modules,
// without doing function bodies... this just adds external function prototypes
// to the Dest function...
//
static bool LinkFunctionProtos(Module *Dest, const Module *Src,
                               std::map<const Value*, Value*> &ValueMap,
                               std::string *Err) {
  ValueSymbolTable &DestSymTab = Dest->getValueSymbolTable();
  
  // Loop over all of the functions in the src module, mapping them over
  for (Module::const_iterator I = Src->begin(), E = Src->end(); I != E; ++I) {
    const Function *SF = I;   // SrcFunction
    GlobalValue *DGV = 0;
    
    // Check to see if may have to link the function with the global, alias or
    // function.
    if (SF->hasName() && !SF->hasInternalLinkage())
      DGV = cast_or_null<GlobalValue>(DestSymTab.lookup(SF->getNameStart(),
                                                        SF->getNameEnd()));
    
    // If we found a global with the same name in the dest module, but it has
    // internal linkage, we are really not doing any linkage here.
    if (DGV && DGV->hasInternalLinkage())
      DGV = 0;

    // If types don't agree due to opaque types, try to resolve them.
    if (DGV && DGV->getType() != SF->getType())
      RecursiveResolveTypes(SF->getType(), DGV->getType());

    GlobalValue::LinkageTypes NewLinkage = GlobalValue::InternalLinkage;
    bool LinkFromSrc = false;
    if (GetLinkageResult(DGV, SF, NewLinkage, LinkFromSrc, Err))
      return true;
    
    // If there is no linkage to be performed, just bring over SF without
    // modifying it.
    if (DGV == 0) {
      // Function does not already exist, simply insert an function signature
      // identical to SF into the dest module.
      Function *NewDF = Function::Create(SF->getFunctionType(),
                                         SF->getLinkage(),
                                         SF->getName(), Dest);
      CopyGVAttributes(NewDF, SF);
      
      // If the LLVM runtime renamed the function, but it is an externally
      // visible symbol, DF must be an existing function with internal linkage.
      // Rename it.
      if (!NewDF->hasInternalLinkage() && NewDF->getName() != SF->getName())
        ForceRenaming(NewDF, SF->getName());
      
      // ... and remember this mapping...
      ValueMap[SF] = NewDF;
      continue;
    }
    
    // If the visibilities of the symbols disagree and the destination is a
    // prototype, take the visibility of its input.
    if (DGV->isDeclaration())
      DGV->setVisibility(SF->getVisibility());
    
    if (LinkFromSrc) {
      if (isa<GlobalAlias>(DGV))
        return Error(Err, "Function-Alias Collision on '" + SF->getName() +
                     "': symbol multiple defined");
      
      // We have a definition of the same name but different type in the
      // source module. Copy the prototype to the destination and replace
      // uses of the destination's prototype with the new prototype.
      Function *NewDF = Function::Create(SF->getFunctionType(), NewLinkage,
                                         SF->getName(), Dest);
      CopyGVAttributes(NewDF, SF);
      
      // Any uses of DF need to change to NewDF, with cast
      DGV->replaceAllUsesWith(ConstantExpr::getBitCast(NewDF, DGV->getType()));
      
      // DF will conflict with NewDF because they both had the same. We must
      // erase this now so ForceRenaming doesn't assert because DF might
      // not have internal linkage. 
      if (GlobalVariable *Var = dyn_cast<GlobalVariable>(DGV))
        Var->eraseFromParent();
      else
        cast<Function>(DGV)->eraseFromParent();
      
      // If the symbol table renamed the function, but it is an externally
      // visible symbol, DF must be an existing function with internal 
      // linkage.  Rename it.
      if (NewDF->getName() != SF->getName() && !NewDF->hasInternalLinkage())
        ForceRenaming(NewDF, SF->getName());
      
      // Remember this mapping so uses in the source module get remapped
      // later by RemapOperand.
      ValueMap[SF] = NewDF;
      continue;
    }
    
    // Not "link from source", keep the one in the DestModule and remap the
    // input onto it.
    
    if (isa<GlobalAlias>(DGV)) {
      // The only valid mappings are:
      // - SF is external declaration, which is effectively a no-op.
      // - SF is weak, when we just need to throw SF out.
      if (!SF->isDeclaration() && !SF->mayBeOverridden())
        return Error(Err, "Function-Alias Collision on '" + SF->getName() +
                     "': symbol multiple defined");
    }

    // Set calculated linkage
    DGV->setLinkage(NewLinkage);

    // Make sure to remember this mapping.
    ValueMap[SF] = ConstantExpr::getBitCast(DGV, SF->getType());
  }
  return false;
}

// LinkFunctionBody - Copy the source function over into the dest function and
// fix up references to values.  At this point we know that Dest is an external
// function, and that Src is not.
static bool LinkFunctionBody(Function *Dest, Function *Src,
                             std::map<const Value*, Value*> &ValueMap,
                             std::string *Err) {
  assert(Src && Dest && Dest->isDeclaration() && !Src->isDeclaration());

  // Go through and convert function arguments over, remembering the mapping.
  Function::arg_iterator DI = Dest->arg_begin();
  for (Function::arg_iterator I = Src->arg_begin(), E = Src->arg_end();
       I != E; ++I, ++DI) {
    DI->setName(I->getName());  // Copy the name information over...

    // Add a mapping to our local map
    ValueMap[I] = DI;
  }

  // Splice the body of the source function into the dest function.
  Dest->getBasicBlockList().splice(Dest->end(), Src->getBasicBlockList());

  // At this point, all of the instructions and values of the function are now
  // copied over.  The only problem is that they are still referencing values in
  // the Source function as operands.  Loop through all of the operands of the
  // functions and patch them up to point to the local versions...
  //
  for (Function::iterator BB = Dest->begin(), BE = Dest->end(); BB != BE; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      for (Instruction::op_iterator OI = I->op_begin(), OE = I->op_end();
           OI != OE; ++OI)
        if (!isa<Instruction>(*OI) && !isa<BasicBlock>(*OI))
          *OI = RemapOperand(*OI, ValueMap);

  // There is no need to map the arguments anymore.
  for (Function::arg_iterator I = Src->arg_begin(), E = Src->arg_end();
       I != E; ++I)
    ValueMap.erase(I);

  return false;
}


// LinkFunctionBodies - Link in the function bodies that are defined in the
// source module into the DestModule.  This consists basically of copying the
// function over and fixing up references to values.
static bool LinkFunctionBodies(Module *Dest, Module *Src,
                               std::map<const Value*, Value*> &ValueMap,
                               std::string *Err) {

  // Loop over all of the functions in the src module, mapping them over as we
  // go
  for (Module::iterator SF = Src->begin(), E = Src->end(); SF != E; ++SF) {
    if (!SF->isDeclaration()) {               // No body if function is external
      Function *DF = dyn_cast<Function>(ValueMap[SF]); // Destination function

      // DF not external SF external?
      if (DF && DF->isDeclaration())
        // Only provide the function body if there isn't one already.
        if (LinkFunctionBody(DF, SF, ValueMap, Err))
          return true;
    }
  }
  return false;
}

// LinkAppendingVars - If there were any appending global variables, link them
// together now.  Return true on error.
static bool LinkAppendingVars(Module *M,
                  std::multimap<std::string, GlobalVariable *> &AppendingVars,
                              std::string *ErrorMsg) {
  if (AppendingVars.empty()) return false; // Nothing to do.

  // Loop over the multimap of appending vars, processing any variables with the
  // same name, forming a new appending global variable with both of the
  // initializers merged together, then rewrite references to the old variables
  // and delete them.
  std::vector<Constant*> Inits;
  while (AppendingVars.size() > 1) {
    // Get the first two elements in the map...
    std::multimap<std::string,
      GlobalVariable*>::iterator Second = AppendingVars.begin(), First=Second++;

    // If the first two elements are for different names, there is no pair...
    // Otherwise there is a pair, so link them together...
    if (First->first == Second->first) {
      GlobalVariable *G1 = First->second, *G2 = Second->second;
      const ArrayType *T1 = cast<ArrayType>(G1->getType()->getElementType());
      const ArrayType *T2 = cast<ArrayType>(G2->getType()->getElementType());

      // Check to see that they two arrays agree on type...
      if (T1->getElementType() != T2->getElementType())
        return Error(ErrorMsg,
         "Appending variables with different element types need to be linked!");
      if (G1->isConstant() != G2->isConstant())
        return Error(ErrorMsg,
                     "Appending variables linked with different const'ness!");

      if (G1->getAlignment() != G2->getAlignment())
        return Error(ErrorMsg,
         "Appending variables with different alignment need to be linked!");

      if (G1->getVisibility() != G2->getVisibility())
        return Error(ErrorMsg,
         "Appending variables with different visibility need to be linked!");

      if (G1->getSection() != G2->getSection())
        return Error(ErrorMsg,
         "Appending variables with different section name need to be linked!");
      
      unsigned NewSize = T1->getNumElements() + T2->getNumElements();
      ArrayType *NewType = ArrayType::get(T1->getElementType(), NewSize);

      G1->setName("");   // Clear G1's name in case of a conflict!
      
      // Create the new global variable...
      GlobalVariable *NG =
        new GlobalVariable(NewType, G1->isConstant(), G1->getLinkage(),
                           /*init*/0, First->first, M, G1->isThreadLocal(),
                           G1->getType()->getAddressSpace());

      // Propagate alignment, visibility and section info.
      CopyGVAttributes(NG, G1);

      // Merge the initializer...
      Inits.reserve(NewSize);
      if (ConstantArray *I = dyn_cast<ConstantArray>(G1->getInitializer())) {
        for (unsigned i = 0, e = T1->getNumElements(); i != e; ++i)
          Inits.push_back(I->getOperand(i));
      } else {
        assert(isa<ConstantAggregateZero>(G1->getInitializer()));
        Constant *CV = Constant::getNullValue(T1->getElementType());
        for (unsigned i = 0, e = T1->getNumElements(); i != e; ++i)
          Inits.push_back(CV);
      }
      if (ConstantArray *I = dyn_cast<ConstantArray>(G2->getInitializer())) {
        for (unsigned i = 0, e = T2->getNumElements(); i != e; ++i)
          Inits.push_back(I->getOperand(i));
      } else {
        assert(isa<ConstantAggregateZero>(G2->getInitializer()));
        Constant *CV = Constant::getNullValue(T2->getElementType());
        for (unsigned i = 0, e = T2->getNumElements(); i != e; ++i)
          Inits.push_back(CV);
      }
      NG->setInitializer(ConstantArray::get(NewType, Inits));
      Inits.clear();

      // Replace any uses of the two global variables with uses of the new
      // global...

      // FIXME: This should rewrite simple/straight-forward uses such as
      // getelementptr instructions to not use the Cast!
      G1->replaceAllUsesWith(ConstantExpr::getBitCast(NG, G1->getType()));
      G2->replaceAllUsesWith(ConstantExpr::getBitCast(NG, G2->getType()));

      // Remove the two globals from the module now...
      M->getGlobalList().erase(G1);
      M->getGlobalList().erase(G2);

      // Put the new global into the AppendingVars map so that we can handle
      // linking of more than two vars...
      Second->second = NG;
    }
    AppendingVars.erase(First);
  }

  return false;
}

static bool ResolveAliases(Module *Dest) {
  for (Module::alias_iterator I = Dest->alias_begin(), E = Dest->alias_end();
       I != E; ++I)
    if (const GlobalValue *GV = I->resolveAliasedGlobal())
      if (GV != I && !GV->isDeclaration())
        I->replaceAllUsesWith(const_cast<GlobalValue*>(GV));

  return false;
}

// LinkModules - This function links two modules together, with the resulting
// left module modified to be the composite of the two input modules.  If an
// error occurs, true is returned and ErrorMsg (if not null) is set to indicate
// the problem.  Upon failure, the Dest module could be in a modified state, and
// shouldn't be relied on to be consistent.
bool
Linker::LinkModules(Module *Dest, Module *Src, std::string *ErrorMsg) {
  assert(Dest != 0 && "Invalid Destination module");
  assert(Src  != 0 && "Invalid Source Module");

  if (Dest->getDataLayout().empty()) {
    if (!Src->getDataLayout().empty()) {
      Dest->setDataLayout(Src->getDataLayout());
    } else {
      std::string DataLayout;

      if (Dest->getEndianness() == Module::AnyEndianness) {
        if (Src->getEndianness() == Module::BigEndian)
          DataLayout.append("E");
        else if (Src->getEndianness() == Module::LittleEndian)
          DataLayout.append("e");
      }

      if (Dest->getPointerSize() == Module::AnyPointerSize) {
        if (Src->getPointerSize() == Module::Pointer64)
          DataLayout.append(DataLayout.length() == 0 ? "p:64:64" : "-p:64:64");
        else if (Src->getPointerSize() == Module::Pointer32)
          DataLayout.append(DataLayout.length() == 0 ? "p:32:32" : "-p:32:32");
      }
      Dest->setDataLayout(DataLayout);
    }
  }

  // Copy the target triple from the source to dest if the dest's is empty.
  if (Dest->getTargetTriple().empty() && !Src->getTargetTriple().empty())
    Dest->setTargetTriple(Src->getTargetTriple());
      
  if (!Src->getDataLayout().empty() && !Dest->getDataLayout().empty() &&
      Src->getDataLayout() != Dest->getDataLayout())
    cerr << "WARNING: Linking two modules of different data layouts!\n";
  if (!Src->getTargetTriple().empty() &&
      Dest->getTargetTriple() != Src->getTargetTriple())
    cerr << "WARNING: Linking two modules of different target triples!\n";

  // Append the module inline asm string.
  if (!Src->getModuleInlineAsm().empty()) {
    if (Dest->getModuleInlineAsm().empty())
      Dest->setModuleInlineAsm(Src->getModuleInlineAsm());
    else
      Dest->setModuleInlineAsm(Dest->getModuleInlineAsm()+"\n"+
                               Src->getModuleInlineAsm());
  }
  
  // Update the destination module's dependent libraries list with the libraries
  // from the source module. There's no opportunity for duplicates here as the
  // Module ensures that duplicate insertions are discarded.
  for (Module::lib_iterator SI = Src->lib_begin(), SE = Src->lib_end();
       SI != SE; ++SI) 
    Dest->addLibrary(*SI);

  // LinkTypes - Go through the symbol table of the Src module and see if any
  // types are named in the src module that are not named in the Dst module.
  // Make sure there are no type name conflicts.
  if (LinkTypes(Dest, Src, ErrorMsg)) 
    return true;

  // ValueMap - Mapping of values from what they used to be in Src, to what they
  // are now in Dest.
  std::map<const Value*, Value*> ValueMap;

  // AppendingVars - Keep track of global variables in the destination module
  // with appending linkage.  After the module is linked together, they are
  // appended and the module is rewritten.
  std::multimap<std::string, GlobalVariable *> AppendingVars;
  for (Module::global_iterator I = Dest->global_begin(), E = Dest->global_end();
       I != E; ++I) {
    // Add all of the appending globals already in the Dest module to
    // AppendingVars.
    if (I->hasAppendingLinkage())
      AppendingVars.insert(std::make_pair(I->getName(), I));
  }

  // Insert all of the globals in src into the Dest module... without linking
  // initializers (which could refer to functions not yet mapped over).
  if (LinkGlobals(Dest, Src, ValueMap, AppendingVars, ErrorMsg))
    return true;

  // Link the functions together between the two modules, without doing function
  // bodies... this just adds external function prototypes to the Dest
  // function...  We do this so that when we begin processing function bodies,
  // all of the global values that may be referenced are available in our
  // ValueMap.
  if (LinkFunctionProtos(Dest, Src, ValueMap, ErrorMsg))
    return true;

  // If there were any alias, link them now. We really need to do this now,
  // because all of the aliases that may be referenced need to be available in
  // ValueMap
  if (LinkAlias(Dest, Src, ValueMap, ErrorMsg)) return true;

  // Update the initializers in the Dest module now that all globals that may
  // be referenced are in Dest.
  if (LinkGlobalInits(Dest, Src, ValueMap, ErrorMsg)) return true;

  // Link in the function bodies that are defined in the source module into the
  // DestModule.  This consists basically of copying the function over and
  // fixing up references to values.
  if (LinkFunctionBodies(Dest, Src, ValueMap, ErrorMsg)) return true;

  // If there were any appending global variables, link them together now.
  if (LinkAppendingVars(Dest, AppendingVars, ErrorMsg)) return true;

  // Resolve all uses of aliases with aliasees
  if (ResolveAliases(Dest)) return true;

  // If the source library's module id is in the dependent library list of the
  // destination library, remove it since that module is now linked in.
  sys::Path modId;
  modId.set(Src->getModuleIdentifier());
  if (!modId.isEmpty())
    Dest->removeLibrary(modId.getBasename());

  return false;
}

// vim: sw=2
