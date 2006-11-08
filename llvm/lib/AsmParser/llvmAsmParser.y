//===-- llvmAsmParser.y - Parser for llvm assembly files --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the bison parser for LLVM assembly languages files.
//
//===----------------------------------------------------------------------===//

%{
#include "ParserInternals.h"
#include "llvm/CallingConv.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Assembly/AutoUpgrade.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <iostream>
#include <list>
#include <utility>

// The following is a gross hack. In order to rid the libAsmParser library of
// exceptions, we have to have a way of getting the yyparse function to go into
// an error situation. So, whenever we want an error to occur, the GenerateError
// function (see bottom of file) sets TriggerError. Then, at the end of each 
// production in the grammer we use CHECK_FOR_ERROR which will invoke YYERROR 
// (a goto) to put YACC in error state. Furthermore, several calls to 
// GenerateError are made from inside productions and they must simulate the
// previous exception behavior by exiting the production immediately. We have
// replaced these with the GEN_ERROR macro which calls GeneratError and then
// immediately invokes YYERROR. This would be so much cleaner if it was a 
// recursive descent parser.
static bool TriggerError = false;
#define CHECK_FOR_ERROR { if (TriggerError) { TriggerError = false; YYABORT; } }
#define GEN_ERROR(msg) { GenerateError(msg); YYERROR; }

int yyerror(const char *ErrorMsg); // Forward declarations to prevent "implicit
int yylex();                       // declaration" of xxx warnings.
int yyparse();

namespace llvm {
  std::string CurFilename;
}
using namespace llvm;

static Module *ParserResult;

// DEBUG_UPREFS - Define this symbol if you want to enable debugging output
// relating to upreferences in the input stream.
//
//#define DEBUG_UPREFS 1
#ifdef DEBUG_UPREFS
#define UR_OUT(X) std::cerr << X
#else
#define UR_OUT(X)
#endif

#define YYERROR_VERBOSE 1

static bool ObsoleteVarArgs;
static bool NewVarArgs;
static BasicBlock *CurBB;
static GlobalVariable *CurGV;


// This contains info used when building the body of a function.  It is
// destroyed when the function is completed.
//
typedef std::vector<Value *> ValueList;           // Numbered defs
static void 
ResolveDefinitions(std::map<const Type *,ValueList> &LateResolvers,
                   std::map<const Type *,ValueList> *FutureLateResolvers = 0);

static struct PerModuleInfo {
  Module *CurrentModule;
  std::map<const Type *, ValueList> Values; // Module level numbered definitions
  std::map<const Type *,ValueList> LateResolveValues;
  std::vector<PATypeHolder>    Types;
  std::map<ValID, PATypeHolder> LateResolveTypes;

  /// PlaceHolderInfo - When temporary placeholder objects are created, remember
  /// how they were referenced and on which line of the input they came from so
  /// that we can resolve them later and print error messages as appropriate.
  std::map<Value*, std::pair<ValID, int> > PlaceHolderInfo;

  // GlobalRefs - This maintains a mapping between <Type, ValID>'s and forward
  // references to global values.  Global values may be referenced before they
  // are defined, and if so, the temporary object that they represent is held
  // here.  This is used for forward references of GlobalValues.
  //
  typedef std::map<std::pair<const PointerType *,
                             ValID>, GlobalValue*> GlobalRefsType;
  GlobalRefsType GlobalRefs;

  void ModuleDone() {
    // If we could not resolve some functions at function compilation time
    // (calls to functions before they are defined), resolve them now...  Types
    // are resolved when the constant pool has been completely parsed.
    //
    ResolveDefinitions(LateResolveValues);
    if (TriggerError)
      return;

    // Check to make sure that all global value forward references have been
    // resolved!
    //
    if (!GlobalRefs.empty()) {
      std::string UndefinedReferences = "Unresolved global references exist:\n";

      for (GlobalRefsType::iterator I = GlobalRefs.begin(), E =GlobalRefs.end();
           I != E; ++I) {
        UndefinedReferences += "  " + I->first.first->getDescription() + " " +
                               I->first.second.getName() + "\n";
      }
      GenerateError(UndefinedReferences);
      return;
    }

    // Look for intrinsic functions and CallInst that need to be upgraded
    for (Module::iterator FI = CurrentModule->begin(),
         FE = CurrentModule->end(); FI != FE; )
      UpgradeCallsToIntrinsic(FI++);

    Values.clear();         // Clear out function local definitions
    Types.clear();
    CurrentModule = 0;
  }

  // GetForwardRefForGlobal - Check to see if there is a forward reference
  // for this global.  If so, remove it from the GlobalRefs map and return it.
  // If not, just return null.
  GlobalValue *GetForwardRefForGlobal(const PointerType *PTy, ValID ID) {
    // Check to see if there is a forward reference to this global variable...
    // if there is, eliminate it and patch the reference to use the new def'n.
    GlobalRefsType::iterator I = GlobalRefs.find(std::make_pair(PTy, ID));
    GlobalValue *Ret = 0;
    if (I != GlobalRefs.end()) {
      Ret = I->second;
      GlobalRefs.erase(I);
    }
    return Ret;
  }
} CurModule;

static struct PerFunctionInfo {
  Function *CurrentFunction;     // Pointer to current function being created

  std::map<const Type*, ValueList> Values; // Keep track of #'d definitions
  std::map<const Type*, ValueList> LateResolveValues;
  bool isDeclare;                    // Is this function a forward declararation?
  GlobalValue::LinkageTypes Linkage; // Linkage for forward declaration.

  /// BBForwardRefs - When we see forward references to basic blocks, keep
  /// track of them here.
  std::map<BasicBlock*, std::pair<ValID, int> > BBForwardRefs;
  std::vector<BasicBlock*> NumberedBlocks;
  unsigned NextBBNum;

  inline PerFunctionInfo() {
    CurrentFunction = 0;
    isDeclare = false;
    Linkage = GlobalValue::ExternalLinkage;    
  }

  inline void FunctionStart(Function *M) {
    CurrentFunction = M;
    NextBBNum = 0;
  }

  void FunctionDone() {
    NumberedBlocks.clear();

    // Any forward referenced blocks left?
    if (!BBForwardRefs.empty()) {
      GenerateError("Undefined reference to label " +
                     BBForwardRefs.begin()->first->getName());
      return;
    }

    // Resolve all forward references now.
    ResolveDefinitions(LateResolveValues, &CurModule.LateResolveValues);

    Values.clear();         // Clear out function local definitions
    CurrentFunction = 0;
    isDeclare = false;
    Linkage = GlobalValue::ExternalLinkage;
  }
} CurFun;  // Info for the current function...

static bool inFunctionScope() { return CurFun.CurrentFunction != 0; }


//===----------------------------------------------------------------------===//
//               Code to handle definitions of all the types
//===----------------------------------------------------------------------===//

static int InsertValue(Value *V,
                  std::map<const Type*,ValueList> &ValueTab = CurFun.Values) {
  if (V->hasName()) return -1;           // Is this a numbered definition?

  // Yes, insert the value into the value table...
  ValueList &List = ValueTab[V->getType()];
  List.push_back(V);
  return List.size()-1;
}

static const Type *getTypeVal(const ValID &D, bool DoNotImprovise = false) {
  switch (D.Type) {
  case ValID::NumberVal:               // Is it a numbered definition?
    // Module constants occupy the lowest numbered slots...
    if ((unsigned)D.Num < CurModule.Types.size())
      return CurModule.Types[(unsigned)D.Num];
    break;
  case ValID::NameVal:                 // Is it a named definition?
    if (const Type *N = CurModule.CurrentModule->getTypeByName(D.Name)) {
      D.destroy();  // Free old strdup'd memory...
      return N;
    }
    break;
  default:
    GenerateError("Internal parser error: Invalid symbol type reference!");
    return 0;
  }

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  if (DoNotImprovise) return 0;  // Do we just want a null to be returned?


  if (inFunctionScope()) {
    if (D.Type == ValID::NameVal) {
      GenerateError("Reference to an undefined type: '" + D.getName() + "'");
      return 0;
    } else {
      GenerateError("Reference to an undefined type: #" + itostr(D.Num));
      return 0;
    }
  }

  std::map<ValID, PATypeHolder>::iterator I =CurModule.LateResolveTypes.find(D);
  if (I != CurModule.LateResolveTypes.end())
    return I->second;

  Type *Typ = OpaqueType::get();
  CurModule.LateResolveTypes.insert(std::make_pair(D, Typ));
  return Typ;
 }

static Value *lookupInSymbolTable(const Type *Ty, const std::string &Name) {
  SymbolTable &SymTab =
    inFunctionScope() ? CurFun.CurrentFunction->getSymbolTable() :
                        CurModule.CurrentModule->getSymbolTable();
  return SymTab.lookup(Ty, Name);
}

// getValNonImprovising - Look up the value specified by the provided type and
// the provided ValID.  If the value exists and has already been defined, return
// it.  Otherwise return null.
//
static Value *getValNonImprovising(const Type *Ty, const ValID &D) {
  if (isa<FunctionType>(Ty)) {
    GenerateError("Functions are not values and "
                   "must be referenced as pointers");
    return 0;
  }

  switch (D.Type) {
  case ValID::NumberVal: {                 // Is it a numbered definition?
    unsigned Num = (unsigned)D.Num;

    // Module constants occupy the lowest numbered slots...
    std::map<const Type*,ValueList>::iterator VI = CurModule.Values.find(Ty);
    if (VI != CurModule.Values.end()) {
      if (Num < VI->second.size())
        return VI->second[Num];
      Num -= VI->second.size();
    }

    // Make sure that our type is within bounds
    VI = CurFun.Values.find(Ty);
    if (VI == CurFun.Values.end()) return 0;

    // Check that the number is within bounds...
    if (VI->second.size() <= Num) return 0;

    return VI->second[Num];
  }

  case ValID::NameVal: {                // Is it a named definition?
    Value *N = lookupInSymbolTable(Ty, std::string(D.Name));
    if (N == 0) return 0;

    D.destroy();  // Free old strdup'd memory...
    return N;
  }

  // Check to make sure that "Ty" is an integral type, and that our
  // value will fit into the specified type...
  case ValID::ConstSIntVal:    // Is it a constant pool reference??
    if (!ConstantInt::isValueValidForType(Ty, D.ConstPool64)) {
      GenerateError("Signed integral constant '" +
                     itostr(D.ConstPool64) + "' is invalid for type '" +
                     Ty->getDescription() + "'!");
      return 0;
    }
    return ConstantInt::get(Ty, D.ConstPool64);

  case ValID::ConstUIntVal:     // Is it an unsigned const pool reference?
    if (!ConstantInt::isValueValidForType(Ty, D.UConstPool64)) {
      if (!ConstantInt::isValueValidForType(Ty, D.ConstPool64)) {
        GenerateError("Integral constant '" + utostr(D.UConstPool64) +
                       "' is invalid or out of range!");
        return 0;
      } else {     // This is really a signed reference.  Transmogrify.
        return ConstantInt::get(Ty, D.ConstPool64);
      }
    } else {
      return ConstantInt::get(Ty, D.UConstPool64);
    }

  case ValID::ConstFPVal:        // Is it a floating point const pool reference?
    if (!ConstantFP::isValueValidForType(Ty, D.ConstPoolFP)) {
      GenerateError("FP constant invalid for type!!");
      return 0;
    }
    return ConstantFP::get(Ty, D.ConstPoolFP);

  case ValID::ConstNullVal:      // Is it a null value?
    if (!isa<PointerType>(Ty)) {
      GenerateError("Cannot create a a non pointer null!");
      return 0;
    }
    return ConstantPointerNull::get(cast<PointerType>(Ty));

  case ValID::ConstUndefVal:      // Is it an undef value?
    return UndefValue::get(Ty);

  case ValID::ConstZeroVal:      // Is it a zero value?
    return Constant::getNullValue(Ty);
    
  case ValID::ConstantVal:       // Fully resolved constant?
    if (D.ConstantValue->getType() != Ty) {
      GenerateError("Constant expression type different from required type!");
      return 0;
    }
    return D.ConstantValue;

  case ValID::InlineAsmVal: {    // Inline asm expression
    const PointerType *PTy = dyn_cast<PointerType>(Ty);
    const FunctionType *FTy =
      PTy ? dyn_cast<FunctionType>(PTy->getElementType()) : 0;
    if (!FTy || !InlineAsm::Verify(FTy, D.IAD->Constraints)) {
      GenerateError("Invalid type for asm constraint string!");
      return 0;
    }
    InlineAsm *IA = InlineAsm::get(FTy, D.IAD->AsmString, D.IAD->Constraints,
                                   D.IAD->HasSideEffects);
    D.destroy();   // Free InlineAsmDescriptor.
    return IA;
  }
  default:
    assert(0 && "Unhandled case!");
    return 0;
  }   // End of switch

  assert(0 && "Unhandled case!");
  return 0;
}

// getVal - This function is identical to getValNonImprovising, except that if a
// value is not already defined, it "improvises" by creating a placeholder var
// that looks and acts just like the requested variable.  When the value is
// defined later, all uses of the placeholder variable are replaced with the
// real thing.
//
static Value *getVal(const Type *Ty, const ValID &ID) {
  if (Ty == Type::LabelTy) {
    GenerateError("Cannot use a basic block here");
    return 0;
  }

  // See if the value has already been defined.
  Value *V = getValNonImprovising(Ty, ID);
  if (V) return V;
  if (TriggerError) return 0;

  if (!Ty->isFirstClassType() && !isa<OpaqueType>(Ty)) {
    GenerateError("Invalid use of a composite type!");
    return 0;
  }

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  V = new Argument(Ty);

  // Remember where this forward reference came from.  FIXME, shouldn't we try
  // to recycle these things??
  CurModule.PlaceHolderInfo.insert(std::make_pair(V, std::make_pair(ID,
                                                               llvmAsmlineno)));

  if (inFunctionScope())
    InsertValue(V, CurFun.LateResolveValues);
  else
    InsertValue(V, CurModule.LateResolveValues);
  return V;
}

/// getBBVal - This is used for two purposes:
///  * If isDefinition is true, a new basic block with the specified ID is being
///    defined.
///  * If isDefinition is true, this is a reference to a basic block, which may
///    or may not be a forward reference.
///
static BasicBlock *getBBVal(const ValID &ID, bool isDefinition = false) {
  assert(inFunctionScope() && "Can't get basic block at global scope!");

  std::string Name;
  BasicBlock *BB = 0;
  switch (ID.Type) {
  default: 
    GenerateError("Illegal label reference " + ID.getName());
    return 0;
  case ValID::NumberVal:                // Is it a numbered definition?
    if (unsigned(ID.Num) >= CurFun.NumberedBlocks.size())
      CurFun.NumberedBlocks.resize(ID.Num+1);
    BB = CurFun.NumberedBlocks[ID.Num];
    break;
  case ValID::NameVal:                  // Is it a named definition?
    Name = ID.Name;
    if (Value *N = CurFun.CurrentFunction->
                   getSymbolTable().lookup(Type::LabelTy, Name))
      BB = cast<BasicBlock>(N);
    break;
  }

  // See if the block has already been defined.
  if (BB) {
    // If this is the definition of the block, make sure the existing value was
    // just a forward reference.  If it was a forward reference, there will be
    // an entry for it in the PlaceHolderInfo map.
    if (isDefinition && !CurFun.BBForwardRefs.erase(BB)) {
      // The existing value was a definition, not a forward reference.
      GenerateError("Redefinition of label " + ID.getName());
      return 0;
    }

    ID.destroy();                       // Free strdup'd memory.
    return BB;
  }

  // Otherwise this block has not been seen before.
  BB = new BasicBlock("", CurFun.CurrentFunction);
  if (ID.Type == ValID::NameVal) {
    BB->setName(ID.Name);
  } else {
    CurFun.NumberedBlocks[ID.Num] = BB;
  }

  // If this is not a definition, keep track of it so we can use it as a forward
  // reference.
  if (!isDefinition) {
    // Remember where this forward reference came from.
    CurFun.BBForwardRefs[BB] = std::make_pair(ID, llvmAsmlineno);
  } else {
    // The forward declaration could have been inserted anywhere in the
    // function: insert it into the correct place now.
    CurFun.CurrentFunction->getBasicBlockList().remove(BB);
    CurFun.CurrentFunction->getBasicBlockList().push_back(BB);
  }
  ID.destroy();
  return BB;
}


//===----------------------------------------------------------------------===//
//              Code to handle forward references in instructions
//===----------------------------------------------------------------------===//
//
// This code handles the late binding needed with statements that reference
// values not defined yet... for example, a forward branch, or the PHI node for
// a loop body.
//
// This keeps a table (CurFun.LateResolveValues) of all such forward references
// and back patchs after we are done.
//

// ResolveDefinitions - If we could not resolve some defs at parsing
// time (forward branches, phi functions for loops, etc...) resolve the
// defs now...
//
static void 
ResolveDefinitions(std::map<const Type*,ValueList> &LateResolvers,
                   std::map<const Type*,ValueList> *FutureLateResolvers) {
  // Loop over LateResolveDefs fixing up stuff that couldn't be resolved
  for (std::map<const Type*,ValueList>::iterator LRI = LateResolvers.begin(),
         E = LateResolvers.end(); LRI != E; ++LRI) {
    ValueList &List = LRI->second;
    while (!List.empty()) {
      Value *V = List.back();
      List.pop_back();

      std::map<Value*, std::pair<ValID, int> >::iterator PHI =
        CurModule.PlaceHolderInfo.find(V);
      assert(PHI != CurModule.PlaceHolderInfo.end() && "Placeholder error!");

      ValID &DID = PHI->second.first;

      Value *TheRealValue = getValNonImprovising(LRI->first, DID);
      if (TriggerError)
        return;
      if (TheRealValue) {
        V->replaceAllUsesWith(TheRealValue);
        delete V;
        CurModule.PlaceHolderInfo.erase(PHI);
      } else if (FutureLateResolvers) {
        // Functions have their unresolved items forwarded to the module late
        // resolver table
        InsertValue(V, *FutureLateResolvers);
      } else {
        if (DID.Type == ValID::NameVal) {
          GenerateError("Reference to an invalid definition: '" +DID.getName()+
                         "' of type '" + V->getType()->getDescription() + "'",
                         PHI->second.second);
          return;
        } else {
          GenerateError("Reference to an invalid definition: #" +
                         itostr(DID.Num) + " of type '" +
                         V->getType()->getDescription() + "'",
                         PHI->second.second);
          return;
        }
      }
    }
  }

  LateResolvers.clear();
}

// ResolveTypeTo - A brand new type was just declared.  This means that (if
// name is not null) things referencing Name can be resolved.  Otherwise, things
// refering to the number can be resolved.  Do this now.
//
static void ResolveTypeTo(char *Name, const Type *ToTy) {
  ValID D;
  if (Name) D = ValID::create(Name);
  else      D = ValID::create((int)CurModule.Types.size());

  std::map<ValID, PATypeHolder>::iterator I =
    CurModule.LateResolveTypes.find(D);
  if (I != CurModule.LateResolveTypes.end()) {
    ((DerivedType*)I->second.get())->refineAbstractTypeTo(ToTy);
    CurModule.LateResolveTypes.erase(I);
  }
}

// setValueName - Set the specified value to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is free'd by this function.
//
static void setValueName(Value *V, char *NameStr) {
  if (NameStr) {
    std::string Name(NameStr);      // Copy string
    free(NameStr);                  // Free old string

    if (V->getType() == Type::VoidTy) {
      GenerateError("Can't assign name '" + Name+"' to value with void type!");
      return;
    }

    assert(inFunctionScope() && "Must be in function scope!");
    SymbolTable &ST = CurFun.CurrentFunction->getSymbolTable();
    if (ST.lookup(V->getType(), Name)) {
      GenerateError("Redefinition of value named '" + Name + "' in the '" +
                     V->getType()->getDescription() + "' type plane!");
      return;
    }

    // Set the name.
    V->setName(Name);
  }
}

/// ParseGlobalVariable - Handle parsing of a global.  If Initializer is null,
/// this is a declaration, otherwise it is a definition.
static GlobalVariable *
ParseGlobalVariable(char *NameStr,GlobalValue::LinkageTypes Linkage,
                    bool isConstantGlobal, const Type *Ty,
                    Constant *Initializer) {
  if (isa<FunctionType>(Ty)) {
    GenerateError("Cannot declare global vars of function type!");
    return 0;
  }

  const PointerType *PTy = PointerType::get(Ty);

  std::string Name;
  if (NameStr) {
    Name = NameStr;      // Copy string
    free(NameStr);       // Free old string
  }

  // See if this global value was forward referenced.  If so, recycle the
  // object.
  ValID ID;
  if (!Name.empty()) {
    ID = ValID::create((char*)Name.c_str());
  } else {
    ID = ValID::create((int)CurModule.Values[PTy].size());
  }

  if (GlobalValue *FWGV = CurModule.GetForwardRefForGlobal(PTy, ID)) {
    // Move the global to the end of the list, from whereever it was
    // previously inserted.
    GlobalVariable *GV = cast<GlobalVariable>(FWGV);
    CurModule.CurrentModule->getGlobalList().remove(GV);
    CurModule.CurrentModule->getGlobalList().push_back(GV);
    GV->setInitializer(Initializer);
    GV->setLinkage(Linkage);
    GV->setConstant(isConstantGlobal);
    InsertValue(GV, CurModule.Values);
    return GV;
  }

  // If this global has a name, check to see if there is already a definition
  // of this global in the module.  If so, merge as appropriate.  Note that
  // this is really just a hack around problems in the CFE.  :(
  if (!Name.empty()) {
    // We are a simple redefinition of a value, check to see if it is defined
    // the same as the old one.
    if (GlobalVariable *EGV =
                CurModule.CurrentModule->getGlobalVariable(Name, Ty)) {
      // We are allowed to redefine a global variable in two circumstances:
      // 1. If at least one of the globals is uninitialized or
      // 2. If both initializers have the same value.
      //
      if (!EGV->hasInitializer() || !Initializer ||
          EGV->getInitializer() == Initializer) {

        // Make sure the existing global version gets the initializer!  Make
        // sure that it also gets marked const if the new version is.
        if (Initializer && !EGV->hasInitializer())
          EGV->setInitializer(Initializer);
        if (isConstantGlobal)
          EGV->setConstant(true);
        EGV->setLinkage(Linkage);
        return EGV;
      }

      GenerateError("Redefinition of global variable named '" + Name +
                     "' in the '" + Ty->getDescription() + "' type plane!");
      return 0;
    }
  }

  // Otherwise there is no existing GV to use, create one now.
  GlobalVariable *GV =
    new GlobalVariable(Ty, isConstantGlobal, Linkage, Initializer, Name,
                       CurModule.CurrentModule);
  InsertValue(GV, CurModule.Values);
  return GV;
}

// setTypeName - Set the specified type to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is freed by this function.
//
// This function returns true if the type has already been defined, but is
// allowed to be redefined in the specified context.  If the name is a new name
// for the type plane, it is inserted and false is returned.
static bool setTypeName(const Type *T, char *NameStr) {
  assert(!inFunctionScope() && "Can't give types function-local names!");
  if (NameStr == 0) return false;
 
  std::string Name(NameStr);      // Copy string
  free(NameStr);                  // Free old string

  // We don't allow assigning names to void type
  if (T == Type::VoidTy) {
    GenerateError("Can't assign name '" + Name + "' to the void type!");
    return false;
  }

  // Set the type name, checking for conflicts as we do so.
  bool AlreadyExists = CurModule.CurrentModule->addTypeName(Name, T);

  if (AlreadyExists) {   // Inserting a name that is already defined???
    const Type *Existing = CurModule.CurrentModule->getTypeByName(Name);
    assert(Existing && "Conflict but no matching type?");

    // There is only one case where this is allowed: when we are refining an
    // opaque type.  In this case, Existing will be an opaque type.
    if (const OpaqueType *OpTy = dyn_cast<OpaqueType>(Existing)) {
      // We ARE replacing an opaque type!
      const_cast<OpaqueType*>(OpTy)->refineAbstractTypeTo(T);
      return true;
    }

    // Otherwise, this is an attempt to redefine a type. That's okay if
    // the redefinition is identical to the original. This will be so if
    // Existing and T point to the same Type object. In this one case we
    // allow the equivalent redefinition.
    if (Existing == T) return true;  // Yes, it's equal.

    // Any other kind of (non-equivalent) redefinition is an error.
    GenerateError("Redefinition of type named '" + Name + "' in the '" +
                   T->getDescription() + "' type plane!");
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Code for handling upreferences in type names...
//

// TypeContains - Returns true if Ty directly contains E in it.
//
static bool TypeContains(const Type *Ty, const Type *E) {
  return std::find(Ty->subtype_begin(), Ty->subtype_end(),
                   E) != Ty->subtype_end();
}

namespace {
  struct UpRefRecord {
    // NestingLevel - The number of nesting levels that need to be popped before
    // this type is resolved.
    unsigned NestingLevel;

    // LastContainedTy - This is the type at the current binding level for the
    // type.  Every time we reduce the nesting level, this gets updated.
    const Type *LastContainedTy;

    // UpRefTy - This is the actual opaque type that the upreference is
    // represented with.
    OpaqueType *UpRefTy;

    UpRefRecord(unsigned NL, OpaqueType *URTy)
      : NestingLevel(NL), LastContainedTy(URTy), UpRefTy(URTy) {}
  };
}

// UpRefs - A list of the outstanding upreferences that need to be resolved.
static std::vector<UpRefRecord> UpRefs;

/// HandleUpRefs - Every time we finish a new layer of types, this function is
/// called.  It loops through the UpRefs vector, which is a list of the
/// currently active types.  For each type, if the up reference is contained in
/// the newly completed type, we decrement the level count.  When the level
/// count reaches zero, the upreferenced type is the type that is passed in:
/// thus we can complete the cycle.
///
static PATypeHolder HandleUpRefs(const Type *ty) {
  // If Ty isn't abstract, or if there are no up-references in it, then there is
  // nothing to resolve here.
  if (!ty->isAbstract() || UpRefs.empty()) return ty;
  
  PATypeHolder Ty(ty);
  UR_OUT("Type '" << Ty->getDescription() <<
         "' newly formed.  Resolving upreferences.\n" <<
         UpRefs.size() << " upreferences active!\n");

  // If we find any resolvable upreferences (i.e., those whose NestingLevel goes
  // to zero), we resolve them all together before we resolve them to Ty.  At
  // the end of the loop, if there is anything to resolve to Ty, it will be in
  // this variable.
  OpaqueType *TypeToResolve = 0;

  for (unsigned i = 0; i != UpRefs.size(); ++i) {
    UR_OUT("  UR#" << i << " - TypeContains(" << Ty->getDescription() << ", "
           << UpRefs[i].second->getDescription() << ") = "
           << (TypeContains(Ty, UpRefs[i].second) ? "true" : "false") << "\n");
    if (TypeContains(Ty, UpRefs[i].LastContainedTy)) {
      // Decrement level of upreference
      unsigned Level = --UpRefs[i].NestingLevel;
      UpRefs[i].LastContainedTy = Ty;
      UR_OUT("  Uplevel Ref Level = " << Level << "\n");
      if (Level == 0) {                     // Upreference should be resolved!
        if (!TypeToResolve) {
          TypeToResolve = UpRefs[i].UpRefTy;
        } else {
          UR_OUT("  * Resolving upreference for "
                 << UpRefs[i].second->getDescription() << "\n";
                 std::string OldName = UpRefs[i].UpRefTy->getDescription());
          UpRefs[i].UpRefTy->refineAbstractTypeTo(TypeToResolve);
          UR_OUT("  * Type '" << OldName << "' refined upreference to: "
                 << (const void*)Ty << ", " << Ty->getDescription() << "\n");
        }
        UpRefs.erase(UpRefs.begin()+i);     // Remove from upreference list...
        --i;                                // Do not skip the next element...
      }
    }
  }

  if (TypeToResolve) {
    UR_OUT("  * Resolving upreference for "
           << UpRefs[i].second->getDescription() << "\n";
           std::string OldName = TypeToResolve->getDescription());
    TypeToResolve->refineAbstractTypeTo(Ty);
  }

  return Ty;
}

/// This function is used to obtain the correct opcode for an instruction when 
/// an obsolete opcode is encountered. The OI parameter (OpcodeInfo) has both 
/// an opcode and an "obsolete" flag. These are generated by the lexer and 
/// the "obsolete" member will be true when the lexer encounters the token for
/// an obsolete opcode. For example, "div" was replaced by [usf]div but we need
/// to maintain backwards compatibility for asm files that still have the "div"
/// instruction. This function handles converting div -> [usf]div appropriately.
/// @brief Convert obsolete opcodes to new values
static void 
sanitizeOpCode(OpcodeInfo<Instruction::BinaryOps> &OI, const PATypeHolder& PATy)
{
  // If its not obsolete, don't do anything
  if (!OI.obsolete) 
    return;

  // If its a packed type we want to use the element type
  const Type* Ty = PATy;
  if (const PackedType* PTy = dyn_cast<PackedType>(Ty))
    Ty = PTy->getElementType();

  // Depending on the opcode ..
  switch (OI.opcode) {
    default:
      GenerateError("Invalid obsolete opCode (check Lexer.l)");
      break;
    case Instruction::UDiv:
      // Handle cases where the opcode needs to change
      if (Ty->isFloatingPoint()) 
        OI.opcode = Instruction::FDiv;
      else if (Ty->isSigned())
        OI.opcode = Instruction::SDiv;
      break;
    case Instruction::URem:
      if (Ty->isFloatingPoint()) 
        OI.opcode = Instruction::FRem;
      else if (Ty->isSigned())
        OI.opcode = Instruction::SRem;
      break;
  }
  // Its not obsolete any more, we fixed it.
  OI.obsolete = false;
}
  
// common code from the two 'RunVMAsmParser' functions
static Module* RunParser(Module * M) {

  llvmAsmlineno = 1;      // Reset the current line number...
  ObsoleteVarArgs = false;
  NewVarArgs = false;
  CurModule.CurrentModule = M;

  // Check to make sure the parser succeeded
  if (yyparse()) {
    if (ParserResult)
      delete ParserResult;
    return 0;
  }

  // Check to make sure that parsing produced a result
  if (!ParserResult)
    return 0;

  // Reset ParserResult variable while saving its value for the result.
  Module *Result = ParserResult;
  ParserResult = 0;

  //Not all functions use vaarg, so make a second check for ObsoleteVarArgs
  {
    Function* F;
    if ((F = Result->getNamedFunction("llvm.va_start"))
        && F->getFunctionType()->getNumParams() == 0)
      ObsoleteVarArgs = true;
    if((F = Result->getNamedFunction("llvm.va_copy"))
       && F->getFunctionType()->getNumParams() == 1)
      ObsoleteVarArgs = true;
  }

  if (ObsoleteVarArgs && NewVarArgs) {
    GenerateError(
      "This file is corrupt: it uses both new and old style varargs");
    return 0;
  }

  if(ObsoleteVarArgs) {
    if(Function* F = Result->getNamedFunction("llvm.va_start")) {
      if (F->arg_size() != 0) {
        GenerateError("Obsolete va_start takes 0 argument!");
        return 0;
      }
      
      //foo = va_start()
      // ->
      //bar = alloca typeof(foo)
      //va_start(bar)
      //foo = load bar

      const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
      const Type* ArgTy = F->getFunctionType()->getReturnType();
      const Type* ArgTyPtr = PointerType::get(ArgTy);
      Function* NF = Result->getOrInsertFunction("llvm.va_start", 
                                                 RetTy, ArgTyPtr, (Type *)0);

      while (!F->use_empty()) {
        CallInst* CI = cast<CallInst>(F->use_back());
        AllocaInst* bar = new AllocaInst(ArgTy, 0, "vastart.fix.1", CI);
        new CallInst(NF, bar, "", CI);
        Value* foo = new LoadInst(bar, "vastart.fix.2", CI);
        CI->replaceAllUsesWith(foo);
        CI->getParent()->getInstList().erase(CI);
      }
      Result->getFunctionList().erase(F);
    }
    
    if(Function* F = Result->getNamedFunction("llvm.va_end")) {
      if(F->arg_size() != 1) {
        GenerateError("Obsolete va_end takes 1 argument!");
        return 0;
      }

      //vaend foo
      // ->
      //bar = alloca 1 of typeof(foo)
      //vaend bar
      const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
      const Type* ArgTy = F->getFunctionType()->getParamType(0);
      const Type* ArgTyPtr = PointerType::get(ArgTy);
      Function* NF = Result->getOrInsertFunction("llvm.va_end", 
                                                 RetTy, ArgTyPtr, (Type *)0);

      while (!F->use_empty()) {
        CallInst* CI = cast<CallInst>(F->use_back());
        AllocaInst* bar = new AllocaInst(ArgTy, 0, "vaend.fix.1", CI);
        new StoreInst(CI->getOperand(1), bar, CI);
        new CallInst(NF, bar, "", CI);
        CI->getParent()->getInstList().erase(CI);
      }
      Result->getFunctionList().erase(F);
    }

    if(Function* F = Result->getNamedFunction("llvm.va_copy")) {
      if(F->arg_size() != 1) {
        GenerateError("Obsolete va_copy takes 1 argument!");
        return 0;
      }
      //foo = vacopy(bar)
      // ->
      //a = alloca 1 of typeof(foo)
      //b = alloca 1 of typeof(foo)
      //store bar -> b
      //vacopy(a, b)
      //foo = load a
      
      const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
      const Type* ArgTy = F->getFunctionType()->getReturnType();
      const Type* ArgTyPtr = PointerType::get(ArgTy);
      Function* NF = Result->getOrInsertFunction("llvm.va_copy", 
                                                 RetTy, ArgTyPtr, ArgTyPtr,
                                                 (Type *)0);

      while (!F->use_empty()) {
        CallInst* CI = cast<CallInst>(F->use_back());
        AllocaInst* a = new AllocaInst(ArgTy, 0, "vacopy.fix.1", CI);
        AllocaInst* b = new AllocaInst(ArgTy, 0, "vacopy.fix.2", CI);
        new StoreInst(CI->getOperand(1), b, CI);
        new CallInst(NF, a, b, "", CI);
        Value* foo = new LoadInst(a, "vacopy.fix.3", CI);
        CI->replaceAllUsesWith(foo);
        CI->getParent()->getInstList().erase(CI);
      }
      Result->getFunctionList().erase(F);
    }
  }

  return Result;
}

//===----------------------------------------------------------------------===//
//            RunVMAsmParser - Define an interface to this parser
//===----------------------------------------------------------------------===//
//
Module *llvm::RunVMAsmParser(const std::string &Filename, FILE *F) {
  set_scan_file(F);

  CurFilename = Filename;
  return RunParser(new Module(CurFilename));
}

Module *llvm::RunVMAsmParser(const char * AsmString, Module * M) {
  set_scan_string(AsmString);

  CurFilename = "from_memory";
  if (M == NULL) {
    return RunParser(new Module (CurFilename));
  } else {
    return RunParser(M);
  }
}

%}

%union {
  llvm::Module                           *ModuleVal;
  llvm::Function                         *FunctionVal;
  std::pair<llvm::PATypeHolder*, char*>  *ArgVal;
  llvm::BasicBlock                       *BasicBlockVal;
  llvm::TerminatorInst                   *TermInstVal;
  llvm::Instruction                      *InstVal;
  llvm::Constant                         *ConstVal;

  const llvm::Type                       *PrimType;
  llvm::PATypeHolder                     *TypeVal;
  llvm::Value                            *ValueVal;

  std::vector<std::pair<llvm::PATypeHolder*,char*> > *ArgList;
  std::vector<llvm::Value*>              *ValueList;
  std::list<llvm::PATypeHolder>          *TypeList;
  // Represent the RHS of PHI node
  std::list<std::pair<llvm::Value*,
                      llvm::BasicBlock*> > *PHIList;
  std::vector<std::pair<llvm::Constant*, llvm::BasicBlock*> > *JumpTable;
  std::vector<llvm::Constant*>           *ConstVector;

  llvm::GlobalValue::LinkageTypes         Linkage;
  int64_t                           SInt64Val;
  uint64_t                          UInt64Val;
  int                               SIntVal;
  unsigned                          UIntVal;
  double                            FPVal;
  bool                              BoolVal;

  char                             *StrVal;   // This memory is strdup'd!
  llvm::ValID                       ValIDVal; // strdup'd memory maybe!

  BinaryOpInfo                      BinaryOpVal;
  TermOpInfo                        TermOpVal;
  MemOpInfo                         MemOpVal;
  OtherOpInfo                       OtherOpVal;
  llvm::Module::Endianness          Endianness;
}

%type <ModuleVal>     Module FunctionList
%type <FunctionVal>   Function FunctionProto FunctionHeader BasicBlockList
%type <BasicBlockVal> BasicBlock InstructionList
%type <TermInstVal>   BBTerminatorInst
%type <InstVal>       Inst InstVal MemoryInst
%type <ConstVal>      ConstVal ConstExpr
%type <ConstVector>   ConstVector
%type <ArgList>       ArgList ArgListH
%type <ArgVal>        ArgVal
%type <PHIList>       PHIList
%type <ValueList>     ValueRefList ValueRefListE  // For call param lists
%type <ValueList>     IndexList                   // For GEP derived indices
%type <TypeList>      TypeListI ArgTypeListI
%type <JumpTable>     JumpTable
%type <BoolVal>       GlobalType                  // GLOBAL or CONSTANT?
%type <BoolVal>       OptVolatile                 // 'volatile' or not
%type <BoolVal>       OptTailCall                 // TAIL CALL or plain CALL.
%type <BoolVal>       OptSideEffect               // 'sideeffect' or not.
%type <Linkage>       OptLinkage
%type <Endianness>    BigOrLittle

// ValueRef - Unresolved reference to a definition or BB
%type <ValIDVal>      ValueRef ConstValueRef SymbolicValueRef
%type <ValueVal>      ResolvedVal            // <type> <valref> pair
// Tokens and types for handling constant integer values
//
// ESINT64VAL - A negative number within long long range
%token <SInt64Val> ESINT64VAL

// EUINT64VAL - A positive number within uns. long long range
%token <UInt64Val> EUINT64VAL
%type  <SInt64Val> EINT64VAL

%token  <SIntVal>   SINTVAL   // Signed 32 bit ints...
%token  <UIntVal>   UINTVAL   // Unsigned 32 bit ints...
%type   <SIntVal>   INTVAL
%token  <FPVal>     FPVAL     // Float or Double constant

// Built in types...
%type  <TypeVal> Types TypesV UpRTypes UpRTypesV
%type  <PrimType> SIntType UIntType IntType FPType PrimType   // Classifications
%token <PrimType> VOID BOOL SBYTE UBYTE SHORT USHORT INT UINT LONG ULONG
%token <PrimType> FLOAT DOUBLE TYPE LABEL

%token <StrVal> VAR_ID LABELSTR STRINGCONSTANT
%type  <StrVal> Name OptName OptAssign
%type  <UIntVal> OptAlign OptCAlign
%type <StrVal> OptSection SectionString

%token IMPLEMENTATION ZEROINITIALIZER TRUETOK FALSETOK BEGINTOK ENDTOK
%token DECLARE GLOBAL CONSTANT SECTION VOLATILE
%token TO DOTDOTDOT NULL_TOK UNDEF CONST INTERNAL LINKONCE WEAK APPENDING
%token DLLIMPORT DLLEXPORT EXTERN_WEAK
%token OPAQUE NOT EXTERNAL TARGET TRIPLE ENDIAN POINTERSIZE LITTLE BIG ALIGN
%token DEPLIBS CALL TAIL ASM_TOK MODULE SIDEEFFECT
%token CC_TOK CCC_TOK CSRETCC_TOK FASTCC_TOK COLDCC_TOK
%token X86_STDCALLCC_TOK X86_FASTCALLCC_TOK
%token DATALAYOUT
%type <UIntVal> OptCallingConv

// Basic Block Terminating Operators
%token <TermOpVal> RET BR SWITCH INVOKE UNWIND UNREACHABLE

// Binary Operators
%type  <BinaryOpVal> ArithmeticOps LogicalOps SetCondOps // Binops Subcatagories
%token <BinaryOpVal> ADD SUB MUL UDIV SDIV FDIV UREM SREM FREM AND OR XOR
%token <BinaryOpVal> SETLE SETGE SETLT SETGT SETEQ SETNE  // Binary Comparators

// Memory Instructions
%token <MemOpVal> MALLOC ALLOCA FREE LOAD STORE GETELEMENTPTR

// Other Operators
%type  <OtherOpVal> ShiftOps
%token <OtherOpVal> PHI_TOK CAST SELECT SHL SHR VAARG
%token <OtherOpVal> EXTRACTELEMENT INSERTELEMENT SHUFFLEVECTOR
%token VAARG_old VANEXT_old //OBSOLETE


%start Module
%%

// Handle constant integer size restriction and conversion...
//
INTVAL : SINTVAL;
INTVAL : UINTVAL {
  if ($1 > (uint32_t)INT32_MAX)     // Outside of my range!
    GEN_ERROR("Value too large for type!");
  $$ = (int32_t)$1;
  CHECK_FOR_ERROR
};


EINT64VAL : ESINT64VAL;      // These have same type and can't cause problems...
EINT64VAL : EUINT64VAL {
  if ($1 > (uint64_t)INT64_MAX)     // Outside of my range!
    GEN_ERROR("Value too large for type!");
  $$ = (int64_t)$1;
  CHECK_FOR_ERROR
};

// Operations that are notably excluded from this list include:
// RET, BR, & SWITCH because they end basic blocks and are treated specially.
//
ArithmeticOps: ADD | SUB | MUL | UDIV | SDIV | FDIV | UREM | SREM | FREM;
LogicalOps   : AND | OR | XOR;
SetCondOps   : SETLE | SETGE | SETLT | SETGT | SETEQ | SETNE;

ShiftOps  : SHL | SHR;

// These are some types that allow classification if we only want a particular 
// thing... for example, only a signed, unsigned, or integral type.
SIntType :  LONG |  INT |  SHORT | SBYTE;
UIntType : ULONG | UINT | USHORT | UBYTE;
IntType  : SIntType | UIntType;
FPType   : FLOAT | DOUBLE;

// OptAssign - Value producing statements have an optional assignment component
OptAssign : Name '=' {
    $$ = $1;
    CHECK_FOR_ERROR
  }
  | /*empty*/ {
    $$ = 0;
    CHECK_FOR_ERROR
  };

OptLinkage : INTERNAL    { $$ = GlobalValue::InternalLinkage; } |
             LINKONCE    { $$ = GlobalValue::LinkOnceLinkage; } |
             WEAK        { $$ = GlobalValue::WeakLinkage; } |
             APPENDING   { $$ = GlobalValue::AppendingLinkage; } |
             DLLIMPORT   { $$ = GlobalValue::DLLImportLinkage; } |
             DLLEXPORT   { $$ = GlobalValue::DLLExportLinkage; } |
             EXTERN_WEAK { $$ = GlobalValue::ExternalWeakLinkage; } |
             /*empty*/   { $$ = GlobalValue::ExternalLinkage; };

OptCallingConv : /*empty*/          { $$ = CallingConv::C; } |
                 CCC_TOK            { $$ = CallingConv::C; } |
                 CSRETCC_TOK        { $$ = CallingConv::CSRet; } |
                 FASTCC_TOK         { $$ = CallingConv::Fast; } |
                 COLDCC_TOK         { $$ = CallingConv::Cold; } |
                 X86_STDCALLCC_TOK  { $$ = CallingConv::X86_StdCall; } |
                 X86_FASTCALLCC_TOK { $$ = CallingConv::X86_FastCall; } |
                 CC_TOK EUINT64VAL  {
                   if ((unsigned)$2 != $2)
                     GEN_ERROR("Calling conv too large!");
                   $$ = $2;
                  CHECK_FOR_ERROR
                 };

// OptAlign/OptCAlign - An optional alignment, and an optional alignment with
// a comma before it.
OptAlign : /*empty*/        { $$ = 0; } |
           ALIGN EUINT64VAL {
  $$ = $2;
  if ($$ != 0 && !isPowerOf2_32($$))
    GEN_ERROR("Alignment must be a power of two!");
  CHECK_FOR_ERROR
};
OptCAlign : /*empty*/            { $$ = 0; } |
            ',' ALIGN EUINT64VAL {
  $$ = $3;
  if ($$ != 0 && !isPowerOf2_32($$))
    GEN_ERROR("Alignment must be a power of two!");
  CHECK_FOR_ERROR
};


SectionString : SECTION STRINGCONSTANT {
  for (unsigned i = 0, e = strlen($2); i != e; ++i)
    if ($2[i] == '"' || $2[i] == '\\')
      GEN_ERROR("Invalid character in section name!");
  $$ = $2;
  CHECK_FOR_ERROR
};

OptSection : /*empty*/ { $$ = 0; } |
             SectionString { $$ = $1; };

// GlobalVarAttributes - Used to pass the attributes string on a global.  CurGV
// is set to be the global we are processing.
//
GlobalVarAttributes : /* empty */ {} |
                     ',' GlobalVarAttribute GlobalVarAttributes {};
GlobalVarAttribute : SectionString {
    CurGV->setSection($1);
    free($1);
    CHECK_FOR_ERROR
  } 
  | ALIGN EUINT64VAL {
    if ($2 != 0 && !isPowerOf2_32($2))
      GEN_ERROR("Alignment must be a power of two!");
    CurGV->setAlignment($2);
    CHECK_FOR_ERROR
  };

//===----------------------------------------------------------------------===//
// Types includes all predefined types... except void, because it can only be
// used in specific contexts (function returning void for example).  To have
// access to it, a user must explicitly use TypesV.
//

// TypesV includes all of 'Types', but it also includes the void type.
TypesV    : Types    | VOID { $$ = new PATypeHolder($1); };
UpRTypesV : UpRTypes | VOID { $$ = new PATypeHolder($1); };

Types     : UpRTypes {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    $$ = $1;
    CHECK_FOR_ERROR
  };


// Derived types are added later...
//
PrimType : BOOL | SBYTE | UBYTE | SHORT  | USHORT | INT   | UINT ;
PrimType : LONG | ULONG | FLOAT | DOUBLE | TYPE   | LABEL;
UpRTypes : OPAQUE {
    $$ = new PATypeHolder(OpaqueType::get());
    CHECK_FOR_ERROR
  }
  | PrimType {
    $$ = new PATypeHolder($1);
    CHECK_FOR_ERROR
  };
UpRTypes : SymbolicValueRef {            // Named types are also simple types...
  const Type* tmp = getTypeVal($1);
  CHECK_FOR_ERROR
  $$ = new PATypeHolder(tmp);
};

// Include derived types in the Types production.
//
UpRTypes : '\\' EUINT64VAL {                   // Type UpReference
    if ($2 > (uint64_t)~0U) GEN_ERROR("Value out of range!");
    OpaqueType *OT = OpaqueType::get();        // Use temporary placeholder
    UpRefs.push_back(UpRefRecord((unsigned)$2, OT));  // Add to vector...
    $$ = new PATypeHolder(OT);
    UR_OUT("New Upreference!\n");
    CHECK_FOR_ERROR
  }
  | UpRTypesV '(' ArgTypeListI ')' {           // Function derived type?
    std::vector<const Type*> Params;
    for (std::list<llvm::PATypeHolder>::iterator I = $3->begin(),
           E = $3->end(); I != E; ++I)
      Params.push_back(*I);
    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    $$ = new PATypeHolder(HandleUpRefs(FunctionType::get(*$1,Params,isVarArg)));
    delete $3;      // Delete the argument list
    delete $1;      // Delete the return type handle
    CHECK_FOR_ERROR
  }
  | '[' EUINT64VAL 'x' UpRTypes ']' {          // Sized array type?
    $$ = new PATypeHolder(HandleUpRefs(ArrayType::get(*$4, (unsigned)$2)));
    delete $4;
    CHECK_FOR_ERROR
  }
  | '<' EUINT64VAL 'x' UpRTypes '>' {          // Packed array type?
     const llvm::Type* ElemTy = $4->get();
     if ((unsigned)$2 != $2)
        GEN_ERROR("Unsigned result not equal to signed result");
     if (!ElemTy->isPrimitiveType())
        GEN_ERROR("Elemental type of a PackedType must be primitive");
     if (!isPowerOf2_32($2))
       GEN_ERROR("Vector length should be a power of 2!");
     $$ = new PATypeHolder(HandleUpRefs(PackedType::get(*$4, (unsigned)$2)));
     delete $4;
     CHECK_FOR_ERROR
  }
  | '{' TypeListI '}' {                        // Structure type?
    std::vector<const Type*> Elements;
    for (std::list<llvm::PATypeHolder>::iterator I = $2->begin(),
           E = $2->end(); I != E; ++I)
      Elements.push_back(*I);

    $$ = new PATypeHolder(HandleUpRefs(StructType::get(Elements)));
    delete $2;
    CHECK_FOR_ERROR
  }
  | '{' '}' {                                  // Empty structure type?
    $$ = new PATypeHolder(StructType::get(std::vector<const Type*>()));
    CHECK_FOR_ERROR
  }
  | UpRTypes '*' {                             // Pointer type?
    if (*$1 == Type::LabelTy)
      GEN_ERROR("Cannot form a pointer to a basic block");
    $$ = new PATypeHolder(HandleUpRefs(PointerType::get(*$1)));
    delete $1;
    CHECK_FOR_ERROR
  };

// TypeList - Used for struct declarations and as a basis for function type 
// declaration type lists
//
TypeListI : UpRTypes {
    $$ = new std::list<PATypeHolder>();
    $$->push_back(*$1); delete $1;
    CHECK_FOR_ERROR
  }
  | TypeListI ',' UpRTypes {
    ($$=$1)->push_back(*$3); delete $3;
    CHECK_FOR_ERROR
  };

// ArgTypeList - List of types for a function type declaration...
ArgTypeListI : TypeListI
  | TypeListI ',' DOTDOTDOT {
    ($$=$1)->push_back(Type::VoidTy);
    CHECK_FOR_ERROR
  }
  | DOTDOTDOT {
    ($$ = new std::list<PATypeHolder>())->push_back(Type::VoidTy);
    CHECK_FOR_ERROR
  }
  | /*empty*/ {
    $$ = new std::list<PATypeHolder>();
    CHECK_FOR_ERROR
  };

// ConstVal - The various declarations that go into the constant pool.  This
// production is used ONLY to represent constants that show up AFTER a 'const',
// 'constant' or 'global' token at global scope.  Constants that can be inlined
// into other expressions (such as integers and constexprs) are handled by the
// ResolvedVal, ValueRef and ConstValueRef productions.
//
ConstVal: Types '[' ConstVector ']' { // Nonempty unsized arr
    const ArrayType *ATy = dyn_cast<ArrayType>($1->get());
    if (ATy == 0)
      GEN_ERROR("Cannot make array constant with type: '" + 
                     (*$1)->getDescription() + "'!");
    const Type *ETy = ATy->getElementType();
    int NumElements = ATy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != -1 && NumElements != (int)$3->size())
      GEN_ERROR("Type mismatch: constant sized array initialized with " +
                     utostr($3->size()) +  " arguments, but has size of " + 
                     itostr(NumElements) + "!");

    // Verify all elements are correct type!
    for (unsigned i = 0; i < $3->size(); i++) {
      if (ETy != (*$3)[i]->getType())
        GEN_ERROR("Element #" + utostr(i) + " is not of type '" + 
                       ETy->getDescription() +"' as required!\nIt is of type '"+
                       (*$3)[i]->getType()->getDescription() + "'.");
    }

    $$ = ConstantArray::get(ATy, *$3);
    delete $1; delete $3;
    CHECK_FOR_ERROR
  }
  | Types '[' ']' {
    const ArrayType *ATy = dyn_cast<ArrayType>($1->get());
    if (ATy == 0)
      GEN_ERROR("Cannot make array constant with type: '" + 
                     (*$1)->getDescription() + "'!");

    int NumElements = ATy->getNumElements();
    if (NumElements != -1 && NumElements != 0) 
      GEN_ERROR("Type mismatch: constant sized array initialized with 0"
                     " arguments, but has size of " + itostr(NumElements) +"!");
    $$ = ConstantArray::get(ATy, std::vector<Constant*>());
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types 'c' STRINGCONSTANT {
    const ArrayType *ATy = dyn_cast<ArrayType>($1->get());
    if (ATy == 0)
      GEN_ERROR("Cannot make array constant with type: '" + 
                     (*$1)->getDescription() + "'!");

    int NumElements = ATy->getNumElements();
    const Type *ETy = ATy->getElementType();
    char *EndStr = UnEscapeLexed($3, true);
    if (NumElements != -1 && NumElements != (EndStr-$3))
      GEN_ERROR("Can't build string constant of size " + 
                     itostr((int)(EndStr-$3)) +
                     " when array has size " + itostr(NumElements) + "!");
    std::vector<Constant*> Vals;
    if (ETy == Type::SByteTy) {
      for (signed char *C = (signed char *)$3; C != (signed char *)EndStr; ++C)
        Vals.push_back(ConstantInt::get(ETy, *C));
    } else if (ETy == Type::UByteTy) {
      for (unsigned char *C = (unsigned char *)$3; 
           C != (unsigned char*)EndStr; ++C)
        Vals.push_back(ConstantInt::get(ETy, *C));
    } else {
      free($3);
      GEN_ERROR("Cannot build string arrays of non byte sized elements!");
    }
    free($3);
    $$ = ConstantArray::get(ATy, Vals);
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types '<' ConstVector '>' { // Nonempty unsized arr
    const PackedType *PTy = dyn_cast<PackedType>($1->get());
    if (PTy == 0)
      GEN_ERROR("Cannot make packed constant with type: '" + 
                     (*$1)->getDescription() + "'!");
    const Type *ETy = PTy->getElementType();
    int NumElements = PTy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != -1 && NumElements != (int)$3->size())
      GEN_ERROR("Type mismatch: constant sized packed initialized with " +
                     utostr($3->size()) +  " arguments, but has size of " + 
                     itostr(NumElements) + "!");

    // Verify all elements are correct type!
    for (unsigned i = 0; i < $3->size(); i++) {
      if (ETy != (*$3)[i]->getType())
        GEN_ERROR("Element #" + utostr(i) + " is not of type '" + 
           ETy->getDescription() +"' as required!\nIt is of type '"+
           (*$3)[i]->getType()->getDescription() + "'.");
    }

    $$ = ConstantPacked::get(PTy, *$3);
    delete $1; delete $3;
    CHECK_FOR_ERROR
  }
  | Types '{' ConstVector '}' {
    const StructType *STy = dyn_cast<StructType>($1->get());
    if (STy == 0)
      GEN_ERROR("Cannot make struct constant with type: '" + 
                     (*$1)->getDescription() + "'!");

    if ($3->size() != STy->getNumContainedTypes())
      GEN_ERROR("Illegal number of initializers for structure type!");

    // Check to ensure that constants are compatible with the type initializer!
    for (unsigned i = 0, e = $3->size(); i != e; ++i)
      if ((*$3)[i]->getType() != STy->getElementType(i))
        GEN_ERROR("Expected type '" +
                       STy->getElementType(i)->getDescription() +
                       "' for element #" + utostr(i) +
                       " of structure initializer!");

    $$ = ConstantStruct::get(STy, *$3);
    delete $1; delete $3;
    CHECK_FOR_ERROR
  }
  | Types '{' '}' {
    const StructType *STy = dyn_cast<StructType>($1->get());
    if (STy == 0)
      GEN_ERROR("Cannot make struct constant with type: '" + 
                     (*$1)->getDescription() + "'!");

    if (STy->getNumContainedTypes() != 0)
      GEN_ERROR("Illegal number of initializers for structure type!");

    $$ = ConstantStruct::get(STy, std::vector<Constant*>());
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types NULL_TOK {
    const PointerType *PTy = dyn_cast<PointerType>($1->get());
    if (PTy == 0)
      GEN_ERROR("Cannot make null pointer constant with type: '" + 
                     (*$1)->getDescription() + "'!");

    $$ = ConstantPointerNull::get(PTy);
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types UNDEF {
    $$ = UndefValue::get($1->get());
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types SymbolicValueRef {
    const PointerType *Ty = dyn_cast<PointerType>($1->get());
    if (Ty == 0)
      GEN_ERROR("Global const reference must be a pointer type!");

    // ConstExprs can exist in the body of a function, thus creating
    // GlobalValues whenever they refer to a variable.  Because we are in
    // the context of a function, getValNonImprovising will search the functions
    // symbol table instead of the module symbol table for the global symbol,
    // which throws things all off.  To get around this, we just tell
    // getValNonImprovising that we are at global scope here.
    //
    Function *SavedCurFn = CurFun.CurrentFunction;
    CurFun.CurrentFunction = 0;

    Value *V = getValNonImprovising(Ty, $2);
    CHECK_FOR_ERROR

    CurFun.CurrentFunction = SavedCurFn;

    // If this is an initializer for a constant pointer, which is referencing a
    // (currently) undefined variable, create a stub now that shall be replaced
    // in the future with the right type of variable.
    //
    if (V == 0) {
      assert(isa<PointerType>(Ty) && "Globals may only be used as pointers!");
      const PointerType *PT = cast<PointerType>(Ty);

      // First check to see if the forward references value is already created!
      PerModuleInfo::GlobalRefsType::iterator I =
        CurModule.GlobalRefs.find(std::make_pair(PT, $2));
    
      if (I != CurModule.GlobalRefs.end()) {
        V = I->second;             // Placeholder already exists, use it...
        $2.destroy();
      } else {
        std::string Name;
        if ($2.Type == ValID::NameVal) Name = $2.Name;

        // Create the forward referenced global.
        GlobalValue *GV;
        if (const FunctionType *FTy = 
                 dyn_cast<FunctionType>(PT->getElementType())) {
          GV = new Function(FTy, GlobalValue::ExternalLinkage, Name,
                            CurModule.CurrentModule);
        } else {
          GV = new GlobalVariable(PT->getElementType(), false,
                                  GlobalValue::ExternalLinkage, 0,
                                  Name, CurModule.CurrentModule);
        }

        // Keep track of the fact that we have a forward ref to recycle it
        CurModule.GlobalRefs.insert(std::make_pair(std::make_pair(PT, $2), GV));
        V = GV;
      }
    }

    $$ = cast<GlobalValue>(V);
    delete $1;            // Free the type handle
    CHECK_FOR_ERROR
  }
  | Types ConstExpr {
    if ($1->get() != $2->getType())
      GEN_ERROR("Mismatched types for constant expression!");
    $$ = $2;
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types ZEROINITIALIZER {
    const Type *Ty = $1->get();
    if (isa<FunctionType>(Ty) || Ty == Type::LabelTy || isa<OpaqueType>(Ty))
      GEN_ERROR("Cannot create a null initialized value of this type!");
    $$ = Constant::getNullValue(Ty);
    delete $1;
    CHECK_FOR_ERROR
  };

ConstVal : SIntType EINT64VAL {      // integral constants
    if (!ConstantInt::isValueValidForType($1, $2))
      GEN_ERROR("Constant value doesn't fit in type!");
    $$ = ConstantInt::get($1, $2);
    CHECK_FOR_ERROR
  }
  | UIntType EUINT64VAL {            // integral constants
    if (!ConstantInt::isValueValidForType($1, $2))
      GEN_ERROR("Constant value doesn't fit in type!");
    $$ = ConstantInt::get($1, $2);
    CHECK_FOR_ERROR
  }
  | BOOL TRUETOK {                      // Boolean constants
    $$ = ConstantBool::getTrue();
    CHECK_FOR_ERROR
  }
  | BOOL FALSETOK {                     // Boolean constants
    $$ = ConstantBool::getFalse();
    CHECK_FOR_ERROR
  }
  | FPType FPVAL {                   // Float & Double constants
    if (!ConstantFP::isValueValidForType($1, $2))
      GEN_ERROR("Floating point constant invalid for type!!");
    $$ = ConstantFP::get($1, $2);
    CHECK_FOR_ERROR
  };


ConstExpr: CAST '(' ConstVal TO Types ')' {
    if (!$3->getType()->isFirstClassType())
      GEN_ERROR("cast constant expression from a non-primitive type: '" +
                     $3->getType()->getDescription() + "'!");
    if (!$5->get()->isFirstClassType())
      GEN_ERROR("cast constant expression to a non-primitive type: '" +
                     $5->get()->getDescription() + "'!");
    $$ = ConstantExpr::getCast($3, $5->get());
    delete $5;
    CHECK_FOR_ERROR
  }
  | GETELEMENTPTR '(' ConstVal IndexList ')' {
    if (!isa<PointerType>($3->getType()))
      GEN_ERROR("GetElementPtr requires a pointer operand!");

    // LLVM 1.2 and earlier used ubyte struct indices.  Convert any ubyte struct
    // indices to uint struct indices for compatibility.
    generic_gep_type_iterator<std::vector<Value*>::iterator>
      GTI = gep_type_begin($3->getType(), $4->begin(), $4->end()),
      GTE = gep_type_end($3->getType(), $4->begin(), $4->end());
    for (unsigned i = 0, e = $4->size(); i != e && GTI != GTE; ++i, ++GTI)
      if (isa<StructType>(*GTI))        // Only change struct indices
        if (ConstantInt *CUI = dyn_cast<ConstantInt>((*$4)[i]))
          if (CUI->getType() == Type::UByteTy)
            (*$4)[i] = ConstantExpr::getCast(CUI, Type::UIntTy);

    const Type *IdxTy =
      GetElementPtrInst::getIndexedType($3->getType(), *$4, true);
    if (!IdxTy)
      GEN_ERROR("Index list invalid for constant getelementptr!");

    std::vector<Constant*> IdxVec;
    for (unsigned i = 0, e = $4->size(); i != e; ++i)
      if (Constant *C = dyn_cast<Constant>((*$4)[i]))
        IdxVec.push_back(C);
      else
        GEN_ERROR("Indices to constant getelementptr must be constants!");

    delete $4;

    $$ = ConstantExpr::getGetElementPtr($3, IdxVec);
    CHECK_FOR_ERROR
  }
  | SELECT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    if ($3->getType() != Type::BoolTy)
      GEN_ERROR("Select condition must be of boolean type!");
    if ($5->getType() != $7->getType())
      GEN_ERROR("Select operand types must match!");
    $$ = ConstantExpr::getSelect($3, $5, $7);
    CHECK_FOR_ERROR
  }
  | ArithmeticOps '(' ConstVal ',' ConstVal ')' {
    if ($3->getType() != $5->getType())
      GEN_ERROR("Binary operator types must match!");
    // First, make sure we're dealing with the right opcode by upgrading from
    // obsolete versions.
    sanitizeOpCode($1,$3->getType());
    CHECK_FOR_ERROR;

    // HACK: llvm 1.3 and earlier used to emit invalid pointer constant exprs.
    // To retain backward compatibility with these early compilers, we emit a
    // cast to the appropriate integer type automatically if we are in the
    // broken case.  See PR424 for more information.
    if (!isa<PointerType>($3->getType())) {
      $$ = ConstantExpr::get($1.opcode, $3, $5);
    } else {
      const Type *IntPtrTy = 0;
      switch (CurModule.CurrentModule->getPointerSize()) {
      case Module::Pointer32: IntPtrTy = Type::IntTy; break;
      case Module::Pointer64: IntPtrTy = Type::LongTy; break;
      default: GEN_ERROR("invalid pointer binary constant expr!");
      }
      $$ = ConstantExpr::get($1.opcode, ConstantExpr::getCast($3, IntPtrTy),
                             ConstantExpr::getCast($5, IntPtrTy));
      $$ = ConstantExpr::getCast($$, $3->getType());
    }
    CHECK_FOR_ERROR
  }
  | LogicalOps '(' ConstVal ',' ConstVal ')' {
    if ($3->getType() != $5->getType())
      GEN_ERROR("Logical operator types must match!");
    if (!$3->getType()->isIntegral()) {
      if (!isa<PackedType>($3->getType()) || 
          !cast<PackedType>($3->getType())->getElementType()->isIntegral())
        GEN_ERROR("Logical operator requires integral operands!");
    }
    $$ = ConstantExpr::get($1.opcode, $3, $5);
    CHECK_FOR_ERROR
  }
  | SetCondOps '(' ConstVal ',' ConstVal ')' {
    if ($3->getType() != $5->getType())
      GEN_ERROR("setcc operand types must match!");
    $$ = ConstantExpr::get($1.opcode, $3, $5);
    CHECK_FOR_ERROR
  }
  | ShiftOps '(' ConstVal ',' ConstVal ')' {
    if ($5->getType() != Type::UByteTy)
      GEN_ERROR("Shift count for shift constant must be unsigned byte!");
    if (!$3->getType()->isInteger())
      GEN_ERROR("Shift constant expression requires integer operand!");
    $$ = ConstantExpr::get($1.opcode, $3, $5);
    CHECK_FOR_ERROR
  }
  | EXTRACTELEMENT '(' ConstVal ',' ConstVal ')' {
    if (!ExtractElementInst::isValidOperands($3, $5))
      GEN_ERROR("Invalid extractelement operands!");
    $$ = ConstantExpr::getExtractElement($3, $5);
    CHECK_FOR_ERROR
  }
  | INSERTELEMENT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    if (!InsertElementInst::isValidOperands($3, $5, $7))
      GEN_ERROR("Invalid insertelement operands!");
    $$ = ConstantExpr::getInsertElement($3, $5, $7);
    CHECK_FOR_ERROR
  }
  | SHUFFLEVECTOR '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    if (!ShuffleVectorInst::isValidOperands($3, $5, $7))
      GEN_ERROR("Invalid shufflevector operands!");
    $$ = ConstantExpr::getShuffleVector($3, $5, $7);
    CHECK_FOR_ERROR
  };


// ConstVector - A list of comma separated constants.
ConstVector : ConstVector ',' ConstVal {
    ($$ = $1)->push_back($3);
    CHECK_FOR_ERROR
  }
  | ConstVal {
    $$ = new std::vector<Constant*>();
    $$->push_back($1);
    CHECK_FOR_ERROR
  };


// GlobalType - Match either GLOBAL or CONSTANT for global declarations...
GlobalType : GLOBAL { $$ = false; } | CONSTANT { $$ = true; };


//===----------------------------------------------------------------------===//
//                             Rules to match Modules
//===----------------------------------------------------------------------===//

// Module rule: Capture the result of parsing the whole file into a result
// variable...
//
Module : FunctionList {
  $$ = ParserResult = $1;
  CurModule.ModuleDone();
  CHECK_FOR_ERROR;
};

// FunctionList - A list of functions, preceeded by a constant pool.
//
FunctionList : FunctionList Function {
    $$ = $1;
    CurFun.FunctionDone();
    CHECK_FOR_ERROR
  } 
  | FunctionList FunctionProto {
    $$ = $1;
    CHECK_FOR_ERROR
  }
  | FunctionList MODULE ASM_TOK AsmBlock {
    $$ = $1;
    CHECK_FOR_ERROR
  }  
  | FunctionList IMPLEMENTATION {
    $$ = $1;
    CHECK_FOR_ERROR
  }
  | ConstPool {
    $$ = CurModule.CurrentModule;
    // Emit an error if there are any unresolved types left.
    if (!CurModule.LateResolveTypes.empty()) {
      const ValID &DID = CurModule.LateResolveTypes.begin()->first;
      if (DID.Type == ValID::NameVal) {
        GEN_ERROR("Reference to an undefined type: '"+DID.getName() + "'");
      } else {
        GEN_ERROR("Reference to an undefined type: #" + itostr(DID.Num));
      }
    }
    CHECK_FOR_ERROR
  };

// ConstPool - Constants with optional names assigned to them.
ConstPool : ConstPool OptAssign TYPE TypesV {
    // Eagerly resolve types.  This is not an optimization, this is a
    // requirement that is due to the fact that we could have this:
    //
    // %list = type { %list * }
    // %list = type { %list * }    ; repeated type decl
    //
    // If types are not resolved eagerly, then the two types will not be
    // determined to be the same type!
    //
    ResolveTypeTo($2, *$4);

    if (!setTypeName(*$4, $2) && !$2) {
      CHECK_FOR_ERROR
      // If this is a named type that is not a redefinition, add it to the slot
      // table.
      CurModule.Types.push_back(*$4);
    }

    delete $4;
    CHECK_FOR_ERROR
  }
  | ConstPool FunctionProto {       // Function prototypes can be in const pool
    CHECK_FOR_ERROR
  }
  | ConstPool MODULE ASM_TOK AsmBlock {  // Asm blocks can be in the const pool
    CHECK_FOR_ERROR
  }
  | ConstPool OptAssign OptLinkage GlobalType ConstVal {
    if ($5 == 0) 
      GEN_ERROR("Global value initializer is not a constant!");
    CurGV = ParseGlobalVariable($2, $3, $4, $5->getType(), $5);
    CHECK_FOR_ERROR
  } GlobalVarAttributes {
    CurGV = 0;
  }
  | ConstPool OptAssign EXTERNAL GlobalType Types {
    CurGV = ParseGlobalVariable($2, GlobalValue::ExternalLinkage, $4, *$5, 0);
    CHECK_FOR_ERROR
    delete $5;
  } GlobalVarAttributes {
    CurGV = 0;
    CHECK_FOR_ERROR
  }
  | ConstPool OptAssign DLLIMPORT GlobalType Types {
    CurGV = ParseGlobalVariable($2, GlobalValue::DLLImportLinkage, $4, *$5, 0);
    CHECK_FOR_ERROR
    delete $5;
  } GlobalVarAttributes {
    CurGV = 0;
    CHECK_FOR_ERROR
  }
  | ConstPool OptAssign EXTERN_WEAK GlobalType Types {
    CurGV = 
      ParseGlobalVariable($2, GlobalValue::ExternalWeakLinkage, $4, *$5, 0);
    CHECK_FOR_ERROR
    delete $5;
  } GlobalVarAttributes {
    CurGV = 0;
    CHECK_FOR_ERROR
  }
  | ConstPool TARGET TargetDefinition { 
    CHECK_FOR_ERROR
  }
  | ConstPool DEPLIBS '=' LibrariesDefinition {
    CHECK_FOR_ERROR
  }
  | /* empty: end of list */ { 
  };


AsmBlock : STRINGCONSTANT {
  const std::string &AsmSoFar = CurModule.CurrentModule->getModuleInlineAsm();
  char *EndStr = UnEscapeLexed($1, true);
  std::string NewAsm($1, EndStr);
  free($1);

  if (AsmSoFar.empty())
    CurModule.CurrentModule->setModuleInlineAsm(NewAsm);
  else
    CurModule.CurrentModule->setModuleInlineAsm(AsmSoFar+"\n"+NewAsm);
  CHECK_FOR_ERROR
};

BigOrLittle : BIG    { $$ = Module::BigEndian; };
BigOrLittle : LITTLE { $$ = Module::LittleEndian; };

TargetDefinition : ENDIAN '=' BigOrLittle {
    CurModule.CurrentModule->setEndianness($3);
    CHECK_FOR_ERROR
  }
  | POINTERSIZE '=' EUINT64VAL {
    if ($3 == 32)
      CurModule.CurrentModule->setPointerSize(Module::Pointer32);
    else if ($3 == 64)
      CurModule.CurrentModule->setPointerSize(Module::Pointer64);
    else
      GEN_ERROR("Invalid pointer size: '" + utostr($3) + "'!");
    CHECK_FOR_ERROR
  }
  | TRIPLE '=' STRINGCONSTANT {
    CurModule.CurrentModule->setTargetTriple($3);
    free($3);
  }
  | DATALAYOUT '=' STRINGCONSTANT {
    CurModule.CurrentModule->setDataLayout($3);
    free($3);
  };

LibrariesDefinition : '[' LibList ']';

LibList : LibList ',' STRINGCONSTANT {
          CurModule.CurrentModule->addLibrary($3);
          free($3);
          CHECK_FOR_ERROR
        }
        | STRINGCONSTANT {
          CurModule.CurrentModule->addLibrary($1);
          free($1);
          CHECK_FOR_ERROR
        }
        | /* empty: end of list */ {
          CHECK_FOR_ERROR
        }
        ;

//===----------------------------------------------------------------------===//
//                       Rules to match Function Headers
//===----------------------------------------------------------------------===//

Name : VAR_ID | STRINGCONSTANT;
OptName : Name | /*empty*/ { $$ = 0; };

ArgVal : Types OptName {
  if (*$1 == Type::VoidTy)
    GEN_ERROR("void typed arguments are invalid!");
  $$ = new std::pair<PATypeHolder*, char*>($1, $2);
  CHECK_FOR_ERROR
};

ArgListH : ArgListH ',' ArgVal {
    $$ = $1;
    $1->push_back(*$3);
    delete $3;
    CHECK_FOR_ERROR
  }
  | ArgVal {
    $$ = new std::vector<std::pair<PATypeHolder*,char*> >();
    $$->push_back(*$1);
    delete $1;
    CHECK_FOR_ERROR
  };

ArgList : ArgListH {
    $$ = $1;
    CHECK_FOR_ERROR
  }
  | ArgListH ',' DOTDOTDOT {
    $$ = $1;
    $$->push_back(std::pair<PATypeHolder*,
                            char*>(new PATypeHolder(Type::VoidTy), 0));
    CHECK_FOR_ERROR
  }
  | DOTDOTDOT {
    $$ = new std::vector<std::pair<PATypeHolder*,char*> >();
    $$->push_back(std::make_pair(new PATypeHolder(Type::VoidTy), (char*)0));
    CHECK_FOR_ERROR
  }
  | /* empty */ {
    $$ = 0;
    CHECK_FOR_ERROR
  };

FunctionHeaderH : OptCallingConv TypesV Name '(' ArgList ')' 
                  OptSection OptAlign {
  UnEscapeLexed($3);
  std::string FunctionName($3);
  free($3);  // Free strdup'd memory!
  
  if (!(*$2)->isFirstClassType() && *$2 != Type::VoidTy)
    GEN_ERROR("LLVM functions cannot return aggregate types!");

  std::vector<const Type*> ParamTypeList;
  if ($5) {   // If there are arguments...
    for (std::vector<std::pair<PATypeHolder*,char*> >::iterator I = $5->begin();
         I != $5->end(); ++I)
      ParamTypeList.push_back(I->first->get());
  }

  bool isVarArg = ParamTypeList.size() && ParamTypeList.back() == Type::VoidTy;
  if (isVarArg) ParamTypeList.pop_back();

  const FunctionType *FT = FunctionType::get(*$2, ParamTypeList, isVarArg);
  const PointerType *PFT = PointerType::get(FT);
  delete $2;

  ValID ID;
  if (!FunctionName.empty()) {
    ID = ValID::create((char*)FunctionName.c_str());
  } else {
    ID = ValID::create((int)CurModule.Values[PFT].size());
  }

  Function *Fn = 0;
  // See if this function was forward referenced.  If so, recycle the object.
  if (GlobalValue *FWRef = CurModule.GetForwardRefForGlobal(PFT, ID)) {
    // Move the function to the end of the list, from whereever it was 
    // previously inserted.
    Fn = cast<Function>(FWRef);
    CurModule.CurrentModule->getFunctionList().remove(Fn);
    CurModule.CurrentModule->getFunctionList().push_back(Fn);
  } else if (!FunctionName.empty() &&     // Merge with an earlier prototype?
             (Fn = CurModule.CurrentModule->getFunction(FunctionName, FT))) {
    // If this is the case, either we need to be a forward decl, or it needs 
    // to be.
    if (!CurFun.isDeclare && !Fn->isExternal())
      GEN_ERROR("Redefinition of function '" + FunctionName + "'!");
    
    // Make sure to strip off any argument names so we can't get conflicts.
    if (Fn->isExternal())
      for (Function::arg_iterator AI = Fn->arg_begin(), AE = Fn->arg_end();
           AI != AE; ++AI)
        AI->setName("");
  } else  {  // Not already defined?
    Fn = new Function(FT, GlobalValue::ExternalLinkage, FunctionName,
                      CurModule.CurrentModule);

    InsertValue(Fn, CurModule.Values);
  }

  CurFun.FunctionStart(Fn);

  if (CurFun.isDeclare) {
    // If we have declaration, always overwrite linkage.  This will allow us to
    // correctly handle cases, when pointer to function is passed as argument to
    // another function.
    Fn->setLinkage(CurFun.Linkage);
  }
  Fn->setCallingConv($1);
  Fn->setAlignment($8);
  if ($7) {
    Fn->setSection($7);
    free($7);
  }

  // Add all of the arguments we parsed to the function...
  if ($5) {                     // Is null if empty...
    if (isVarArg) {  // Nuke the last entry
      assert($5->back().first->get() == Type::VoidTy && $5->back().second == 0&&
             "Not a varargs marker!");
      delete $5->back().first;
      $5->pop_back();  // Delete the last entry
    }
    Function::arg_iterator ArgIt = Fn->arg_begin();
    for (std::vector<std::pair<PATypeHolder*,char*> >::iterator I = $5->begin();
         I != $5->end(); ++I, ++ArgIt) {
      delete I->first;                          // Delete the typeholder...

      setValueName(ArgIt, I->second);           // Insert arg into symtab...
      CHECK_FOR_ERROR
      InsertValue(ArgIt);
    }

    delete $5;                     // We're now done with the argument list
  }
  CHECK_FOR_ERROR
};

BEGIN : BEGINTOK | '{';                // Allow BEGIN or '{' to start a function

FunctionHeader : OptLinkage FunctionHeaderH BEGIN {
  $$ = CurFun.CurrentFunction;

  // Make sure that we keep track of the linkage type even if there was a
  // previous "declare".
  $$->setLinkage($1);
};

END : ENDTOK | '}';                    // Allow end of '}' to end a function

Function : BasicBlockList END {
  $$ = $1;
  CHECK_FOR_ERROR
};

FnDeclareLinkage: /*default*/ |
                  DLLIMPORT   { CurFun.Linkage = GlobalValue::DLLImportLinkage; } |
                  EXTERN_WEAK { CurFun.Linkage = GlobalValue::DLLImportLinkage; };
  
FunctionProto : DECLARE { CurFun.isDeclare = true; } FnDeclareLinkage FunctionHeaderH {
    $$ = CurFun.CurrentFunction;
    CurFun.FunctionDone();
    CHECK_FOR_ERROR
  };

//===----------------------------------------------------------------------===//
//                        Rules to match Basic Blocks
//===----------------------------------------------------------------------===//

OptSideEffect : /* empty */ {
    $$ = false;
    CHECK_FOR_ERROR
  }
  | SIDEEFFECT {
    $$ = true;
    CHECK_FOR_ERROR
  };

ConstValueRef : ESINT64VAL {    // A reference to a direct constant
    $$ = ValID::create($1);
    CHECK_FOR_ERROR
  }
  | EUINT64VAL {
    $$ = ValID::create($1);
    CHECK_FOR_ERROR
  }
  | FPVAL {                     // Perhaps it's an FP constant?
    $$ = ValID::create($1);
    CHECK_FOR_ERROR
  }
  | TRUETOK {
    $$ = ValID::create(ConstantBool::getTrue());
    CHECK_FOR_ERROR
  } 
  | FALSETOK {
    $$ = ValID::create(ConstantBool::getFalse());
    CHECK_FOR_ERROR
  }
  | NULL_TOK {
    $$ = ValID::createNull();
    CHECK_FOR_ERROR
  }
  | UNDEF {
    $$ = ValID::createUndef();
    CHECK_FOR_ERROR
  }
  | ZEROINITIALIZER {     // A vector zero constant.
    $$ = ValID::createZeroInit();
    CHECK_FOR_ERROR
  }
  | '<' ConstVector '>' { // Nonempty unsized packed vector
    const Type *ETy = (*$2)[0]->getType();
    int NumElements = $2->size(); 
    
    PackedType* pt = PackedType::get(ETy, NumElements);
    PATypeHolder* PTy = new PATypeHolder(
                                         HandleUpRefs(
                                            PackedType::get(
                                                ETy, 
                                                NumElements)
                                            )
                                         );
    
    // Verify all elements are correct type!
    for (unsigned i = 0; i < $2->size(); i++) {
      if (ETy != (*$2)[i]->getType())
        GEN_ERROR("Element #" + utostr(i) + " is not of type '" + 
                     ETy->getDescription() +"' as required!\nIt is of type '" +
                     (*$2)[i]->getType()->getDescription() + "'.");
    }

    $$ = ValID::create(ConstantPacked::get(pt, *$2));
    delete PTy; delete $2;
    CHECK_FOR_ERROR
  }
  | ConstExpr {
    $$ = ValID::create($1);
    CHECK_FOR_ERROR
  }
  | ASM_TOK OptSideEffect STRINGCONSTANT ',' STRINGCONSTANT {
    char *End = UnEscapeLexed($3, true);
    std::string AsmStr = std::string($3, End);
    End = UnEscapeLexed($5, true);
    std::string Constraints = std::string($5, End);
    $$ = ValID::createInlineAsm(AsmStr, Constraints, $2);
    free($3);
    free($5);
    CHECK_FOR_ERROR
  };

// SymbolicValueRef - Reference to one of two ways of symbolically refering to
// another value.
//
SymbolicValueRef : INTVAL {  // Is it an integer reference...?
    $$ = ValID::create($1);
    CHECK_FOR_ERROR
  }
  | Name {                   // Is it a named reference...?
    $$ = ValID::create($1);
    CHECK_FOR_ERROR
  };

// ValueRef - A reference to a definition... either constant or symbolic
ValueRef : SymbolicValueRef | ConstValueRef;


// ResolvedVal - a <type> <value> pair.  This is used only in cases where the
// type immediately preceeds the value reference, and allows complex constant
// pool references (for things like: 'ret [2 x int] [ int 12, int 42]')
ResolvedVal : Types ValueRef {
    $$ = getVal(*$1, $2); delete $1;
    CHECK_FOR_ERROR
  };

BasicBlockList : BasicBlockList BasicBlock {
    $$ = $1;
    CHECK_FOR_ERROR
  }
  | FunctionHeader BasicBlock { // Do not allow functions with 0 basic blocks   
    $$ = $1;
    CHECK_FOR_ERROR
  };


// Basic blocks are terminated by branching instructions: 
// br, br/cc, switch, ret
//
BasicBlock : InstructionList OptAssign BBTerminatorInst  {
    setValueName($3, $2);
    CHECK_FOR_ERROR
    InsertValue($3);

    $1->getInstList().push_back($3);
    InsertValue($1);
    $$ = $1;
    CHECK_FOR_ERROR
  };

InstructionList : InstructionList Inst {
    $1->getInstList().push_back($2);
    $$ = $1;
    CHECK_FOR_ERROR
  }
  | /* empty */ {
    $$ = CurBB = getBBVal(ValID::create((int)CurFun.NextBBNum++), true);
    CHECK_FOR_ERROR

    // Make sure to move the basic block to the correct location in the
    // function, instead of leaving it inserted wherever it was first
    // referenced.
    Function::BasicBlockListType &BBL = 
      CurFun.CurrentFunction->getBasicBlockList();
    BBL.splice(BBL.end(), BBL, $$);
    CHECK_FOR_ERROR
  }
  | LABELSTR {
    $$ = CurBB = getBBVal(ValID::create($1), true);
    CHECK_FOR_ERROR

    // Make sure to move the basic block to the correct location in the
    // function, instead of leaving it inserted wherever it was first
    // referenced.
    Function::BasicBlockListType &BBL = 
      CurFun.CurrentFunction->getBasicBlockList();
    BBL.splice(BBL.end(), BBL, $$);
    CHECK_FOR_ERROR
  };

BBTerminatorInst : RET ResolvedVal {              // Return with a result...
    $$ = new ReturnInst($2);
    CHECK_FOR_ERROR
  }
  | RET VOID {                                       // Return with no result...
    $$ = new ReturnInst();
    CHECK_FOR_ERROR
  }
  | BR LABEL ValueRef {                         // Unconditional Branch...
    BasicBlock* tmpBB = getBBVal($3);
    CHECK_FOR_ERROR
    $$ = new BranchInst(tmpBB);
  }                                                  // Conditional Branch...
  | BR BOOL ValueRef ',' LABEL ValueRef ',' LABEL ValueRef {  
    BasicBlock* tmpBBA = getBBVal($6);
    CHECK_FOR_ERROR
    BasicBlock* tmpBBB = getBBVal($9);
    CHECK_FOR_ERROR
    Value* tmpVal = getVal(Type::BoolTy, $3);
    CHECK_FOR_ERROR
    $$ = new BranchInst(tmpBBA, tmpBBB, tmpVal);
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' JumpTable ']' {
    Value* tmpVal = getVal($2, $3);
    CHECK_FOR_ERROR
    BasicBlock* tmpBB = getBBVal($6);
    CHECK_FOR_ERROR
    SwitchInst *S = new SwitchInst(tmpVal, tmpBB, $8->size());
    $$ = S;

    std::vector<std::pair<Constant*,BasicBlock*> >::iterator I = $8->begin(),
      E = $8->end();
    for (; I != E; ++I) {
      if (ConstantInt *CI = dyn_cast<ConstantInt>(I->first))
          S->addCase(CI, I->second);
      else
        GEN_ERROR("Switch case is constant, but not a simple integer!");
    }
    delete $8;
    CHECK_FOR_ERROR
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' ']' {
    Value* tmpVal = getVal($2, $3);
    CHECK_FOR_ERROR
    BasicBlock* tmpBB = getBBVal($6);
    CHECK_FOR_ERROR
    SwitchInst *S = new SwitchInst(tmpVal, tmpBB, 0);
    $$ = S;
    CHECK_FOR_ERROR
  }
  | INVOKE OptCallingConv TypesV ValueRef '(' ValueRefListE ')'
    TO LABEL ValueRef UNWIND LABEL ValueRef {
    const PointerType *PFTy;
    const FunctionType *Ty;

    if (!(PFTy = dyn_cast<PointerType>($3->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      if ($6) {
        for (std::vector<Value*>::iterator I = $6->begin(), E = $6->end();
             I != E; ++I)
          ParamTypes.push_back((*I)->getType());
      }

      bool isVarArg = ParamTypes.size() && ParamTypes.back() == Type::VoidTy;
      if (isVarArg) ParamTypes.pop_back();

      Ty = FunctionType::get($3->get(), ParamTypes, isVarArg);
      PFTy = PointerType::get(Ty);
    }

    Value *V = getVal(PFTy, $4);   // Get the function we're calling...
    CHECK_FOR_ERROR
    BasicBlock *Normal = getBBVal($10);
    CHECK_FOR_ERROR
    BasicBlock *Except = getBBVal($13);
    CHECK_FOR_ERROR

    // Create the call node...
    if (!$6) {                                   // Has no arguments?
      $$ = new InvokeInst(V, Normal, Except, std::vector<Value*>());
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly!
      //
      FunctionType::param_iterator I = Ty->param_begin();
      FunctionType::param_iterator E = Ty->param_end();
      std::vector<Value*>::iterator ArgI = $6->begin(), ArgE = $6->end();

      for (; ArgI != ArgE && I != E; ++ArgI, ++I)
        if ((*ArgI)->getType() != *I)
          GEN_ERROR("Parameter " +(*ArgI)->getName()+ " is not of type '" +
                         (*I)->getDescription() + "'!");

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
        GEN_ERROR("Invalid number of parameters detected!");

      $$ = new InvokeInst(V, Normal, Except, *$6);
    }
    cast<InvokeInst>($$)->setCallingConv($2);
  
    delete $3;
    delete $6;
    CHECK_FOR_ERROR
  }
  | UNWIND {
    $$ = new UnwindInst();
    CHECK_FOR_ERROR
  }
  | UNREACHABLE {
    $$ = new UnreachableInst();
    CHECK_FOR_ERROR
  };



JumpTable : JumpTable IntType ConstValueRef ',' LABEL ValueRef {
    $$ = $1;
    Constant *V = cast<Constant>(getValNonImprovising($2, $3));
    CHECK_FOR_ERROR
    if (V == 0)
      GEN_ERROR("May only switch on a constant pool value!");

    BasicBlock* tmpBB = getBBVal($6);
    CHECK_FOR_ERROR
    $$->push_back(std::make_pair(V, tmpBB));
  }
  | IntType ConstValueRef ',' LABEL ValueRef {
    $$ = new std::vector<std::pair<Constant*, BasicBlock*> >();
    Constant *V = cast<Constant>(getValNonImprovising($1, $2));
    CHECK_FOR_ERROR

    if (V == 0)
      GEN_ERROR("May only switch on a constant pool value!");

    BasicBlock* tmpBB = getBBVal($5);
    CHECK_FOR_ERROR
    $$->push_back(std::make_pair(V, tmpBB)); 
  };

Inst : OptAssign InstVal {
  // Is this definition named?? if so, assign the name...
  setValueName($2, $1);
  CHECK_FOR_ERROR
  InsertValue($2);
  $$ = $2;
  CHECK_FOR_ERROR
};

PHIList : Types '[' ValueRef ',' ValueRef ']' {    // Used for PHI nodes
    $$ = new std::list<std::pair<Value*, BasicBlock*> >();
    Value* tmpVal = getVal(*$1, $3);
    CHECK_FOR_ERROR
    BasicBlock* tmpBB = getBBVal($5);
    CHECK_FOR_ERROR
    $$->push_back(std::make_pair(tmpVal, tmpBB));
    delete $1;
  }
  | PHIList ',' '[' ValueRef ',' ValueRef ']' {
    $$ = $1;
    Value* tmpVal = getVal($1->front().first->getType(), $4);
    CHECK_FOR_ERROR
    BasicBlock* tmpBB = getBBVal($6);
    CHECK_FOR_ERROR
    $1->push_back(std::make_pair(tmpVal, tmpBB));
  };


ValueRefList : ResolvedVal {    // Used for call statements, and memory insts...
    $$ = new std::vector<Value*>();
    $$->push_back($1);
  }
  | ValueRefList ',' ResolvedVal {
    $$ = $1;
    $1->push_back($3);
    CHECK_FOR_ERROR
  };

// ValueRefListE - Just like ValueRefList, except that it may also be empty!
ValueRefListE : ValueRefList | /*empty*/ { $$ = 0; };

OptTailCall : TAIL CALL {
    $$ = true;
    CHECK_FOR_ERROR
  }
  | CALL {
    $$ = false;
    CHECK_FOR_ERROR
  };

InstVal : ArithmeticOps Types ValueRef ',' ValueRef {
    if (!(*$2)->isInteger() && !(*$2)->isFloatingPoint() && 
        !isa<PackedType>((*$2).get()))
      GEN_ERROR(
        "Arithmetic operator requires integer, FP, or packed operands!");
    if (isa<PackedType>((*$2).get()) && 
        ($1.opcode == Instruction::URem || 
         $1.opcode == Instruction::SRem ||
         $1.opcode == Instruction::FRem))
      GEN_ERROR("U/S/FRem not supported on packed types!");
    // Upgrade the opcode from obsolete versions before we do anything with it.
    sanitizeOpCode($1,*$2);
    CHECK_FOR_ERROR;
    Value* val1 = getVal(*$2, $3); 
    CHECK_FOR_ERROR
    Value* val2 = getVal(*$2, $5);
    CHECK_FOR_ERROR
    $$ = BinaryOperator::create($1.opcode, val1, val2);
    if ($$ == 0)
      GEN_ERROR("binary operator returned null!");
    delete $2;
  }
  | LogicalOps Types ValueRef ',' ValueRef {
    if (!(*$2)->isIntegral()) {
      if (!isa<PackedType>($2->get()) ||
          !cast<PackedType>($2->get())->getElementType()->isIntegral())
        GEN_ERROR("Logical operator requires integral operands!");
    }
    Value* tmpVal1 = getVal(*$2, $3);
    CHECK_FOR_ERROR
    Value* tmpVal2 = getVal(*$2, $5);
    CHECK_FOR_ERROR
    $$ = BinaryOperator::create($1.opcode, tmpVal1, tmpVal2);
    if ($$ == 0)
      GEN_ERROR("binary operator returned null!");
    delete $2;
  }
  | SetCondOps Types ValueRef ',' ValueRef {
    if(isa<PackedType>((*$2).get())) {
      GEN_ERROR(
        "PackedTypes currently not supported in setcc instructions!");
    }
    Value* tmpVal1 = getVal(*$2, $3);
    CHECK_FOR_ERROR
    Value* tmpVal2 = getVal(*$2, $5);
    CHECK_FOR_ERROR
    $$ = new SetCondInst($1.opcode, tmpVal1, tmpVal2);
    if ($$ == 0)
      GEN_ERROR("binary operator returned null!");
    delete $2;
  }
  | NOT ResolvedVal {
    std::cerr << "WARNING: Use of eliminated 'not' instruction:"
              << " Replacing with 'xor'.\n";

    Value *Ones = ConstantIntegral::getAllOnesValue($2->getType());
    if (Ones == 0)
      GEN_ERROR("Expected integral type for not instruction!");

    $$ = BinaryOperator::create(Instruction::Xor, $2, Ones);
    if ($$ == 0)
      GEN_ERROR("Could not create a xor instruction!");
    CHECK_FOR_ERROR
  }
  | ShiftOps ResolvedVal ',' ResolvedVal {
    if ($4->getType() != Type::UByteTy)
      GEN_ERROR("Shift amount must be ubyte!");
    if (!$2->getType()->isInteger())
      GEN_ERROR("Shift constant expression requires integer operand!");
    $$ = new ShiftInst($1.opcode, $2, $4);
    CHECK_FOR_ERROR
  }
  | CAST ResolvedVal TO Types {
    if (!$4->get()->isFirstClassType())
      GEN_ERROR("cast instruction to a non-primitive type: '" +
                     $4->get()->getDescription() + "'!");
    $$ = new CastInst($2, *$4);
    delete $4;
    CHECK_FOR_ERROR
  }
  | SELECT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    if ($2->getType() != Type::BoolTy)
      GEN_ERROR("select condition must be boolean!");
    if ($4->getType() != $6->getType())
      GEN_ERROR("select value types should match!");
    $$ = new SelectInst($2, $4, $6);
    CHECK_FOR_ERROR
  }
  | VAARG ResolvedVal ',' Types {
    NewVarArgs = true;
    $$ = new VAArgInst($2, *$4);
    delete $4;
    CHECK_FOR_ERROR
  }
  | VAARG_old ResolvedVal ',' Types {
    ObsoleteVarArgs = true;
    const Type* ArgTy = $2->getType();
    Function* NF = CurModule.CurrentModule->
      getOrInsertFunction("llvm.va_copy", ArgTy, ArgTy, (Type *)0);

    //b = vaarg a, t -> 
    //foo = alloca 1 of t
    //bar = vacopy a 
    //store bar -> foo
    //b = vaarg foo, t
    AllocaInst* foo = new AllocaInst(ArgTy, 0, "vaarg.fix");
    CurBB->getInstList().push_back(foo);
    CallInst* bar = new CallInst(NF, $2);
    CurBB->getInstList().push_back(bar);
    CurBB->getInstList().push_back(new StoreInst(bar, foo));
    $$ = new VAArgInst(foo, *$4);
    delete $4;
    CHECK_FOR_ERROR
  }
  | VANEXT_old ResolvedVal ',' Types {
    ObsoleteVarArgs = true;
    const Type* ArgTy = $2->getType();
    Function* NF = CurModule.CurrentModule->
      getOrInsertFunction("llvm.va_copy", ArgTy, ArgTy, (Type *)0);

    //b = vanext a, t ->
    //foo = alloca 1 of t
    //bar = vacopy a
    //store bar -> foo
    //tmp = vaarg foo, t
    //b = load foo
    AllocaInst* foo = new AllocaInst(ArgTy, 0, "vanext.fix");
    CurBB->getInstList().push_back(foo);
    CallInst* bar = new CallInst(NF, $2);
    CurBB->getInstList().push_back(bar);
    CurBB->getInstList().push_back(new StoreInst(bar, foo));
    Instruction* tmp = new VAArgInst(foo, *$4);
    CurBB->getInstList().push_back(tmp);
    $$ = new LoadInst(foo);
    delete $4;
    CHECK_FOR_ERROR
  }
  | EXTRACTELEMENT ResolvedVal ',' ResolvedVal {
    if (!ExtractElementInst::isValidOperands($2, $4))
      GEN_ERROR("Invalid extractelement operands!");
    $$ = new ExtractElementInst($2, $4);
    CHECK_FOR_ERROR
  }
  | INSERTELEMENT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    if (!InsertElementInst::isValidOperands($2, $4, $6))
      GEN_ERROR("Invalid insertelement operands!");
    $$ = new InsertElementInst($2, $4, $6);
    CHECK_FOR_ERROR
  }
  | SHUFFLEVECTOR ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    if (!ShuffleVectorInst::isValidOperands($2, $4, $6))
      GEN_ERROR("Invalid shufflevector operands!");
    $$ = new ShuffleVectorInst($2, $4, $6);
    CHECK_FOR_ERROR
  }
  | PHI_TOK PHIList {
    const Type *Ty = $2->front().first->getType();
    if (!Ty->isFirstClassType())
      GEN_ERROR("PHI node operands must be of first class type!");
    $$ = new PHINode(Ty);
    ((PHINode*)$$)->reserveOperandSpace($2->size());
    while ($2->begin() != $2->end()) {
      if ($2->front().first->getType() != Ty) 
        GEN_ERROR("All elements of a PHI node must be of the same type!");
      cast<PHINode>($$)->addIncoming($2->front().first, $2->front().second);
      $2->pop_front();
    }
    delete $2;  // Free the list...
    CHECK_FOR_ERROR
  }
  | OptTailCall OptCallingConv TypesV ValueRef '(' ValueRefListE ')'  {
    const PointerType *PFTy;
    const FunctionType *Ty;

    if (!(PFTy = dyn_cast<PointerType>($3->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      if ($6) {
        for (std::vector<Value*>::iterator I = $6->begin(), E = $6->end();
             I != E; ++I)
          ParamTypes.push_back((*I)->getType());
      }

      bool isVarArg = ParamTypes.size() && ParamTypes.back() == Type::VoidTy;
      if (isVarArg) ParamTypes.pop_back();

      if (!(*$3)->isFirstClassType() && *$3 != Type::VoidTy)
        GEN_ERROR("LLVM functions cannot return aggregate types!");

      Ty = FunctionType::get($3->get(), ParamTypes, isVarArg);
      PFTy = PointerType::get(Ty);
    }

    Value *V = getVal(PFTy, $4);   // Get the function we're calling...
    CHECK_FOR_ERROR

    // Create the call node...
    if (!$6) {                                   // Has no arguments?
      // Make sure no arguments is a good thing!
      if (Ty->getNumParams() != 0)
        GEN_ERROR("No arguments passed to a function that "
                       "expects arguments!");

      $$ = new CallInst(V, std::vector<Value*>());
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly!
      //
      FunctionType::param_iterator I = Ty->param_begin();
      FunctionType::param_iterator E = Ty->param_end();
      std::vector<Value*>::iterator ArgI = $6->begin(), ArgE = $6->end();

      for (; ArgI != ArgE && I != E; ++ArgI, ++I)
        if ((*ArgI)->getType() != *I)
          GEN_ERROR("Parameter " +(*ArgI)->getName()+ " is not of type '" +
                         (*I)->getDescription() + "'!");

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
        GEN_ERROR("Invalid number of parameters detected!");

      $$ = new CallInst(V, *$6);
    }
    cast<CallInst>($$)->setTailCall($1);
    cast<CallInst>($$)->setCallingConv($2);
    delete $3;
    delete $6;
    CHECK_FOR_ERROR
  }
  | MemoryInst {
    $$ = $1;
    CHECK_FOR_ERROR
  };


// IndexList - List of indices for GEP based instructions...
IndexList : ',' ValueRefList { 
    $$ = $2; 
    CHECK_FOR_ERROR
  } | /* empty */ { 
    $$ = new std::vector<Value*>(); 
    CHECK_FOR_ERROR
  };

OptVolatile : VOLATILE {
    $$ = true;
    CHECK_FOR_ERROR
  }
  | /* empty */ {
    $$ = false;
    CHECK_FOR_ERROR
  };



MemoryInst : MALLOC Types OptCAlign {
    $$ = new MallocInst(*$2, 0, $3);
    delete $2;
    CHECK_FOR_ERROR
  }
  | MALLOC Types ',' UINT ValueRef OptCAlign {
    Value* tmpVal = getVal($4, $5);
    CHECK_FOR_ERROR
    $$ = new MallocInst(*$2, tmpVal, $6);
    delete $2;
  }
  | ALLOCA Types OptCAlign {
    $$ = new AllocaInst(*$2, 0, $3);
    delete $2;
    CHECK_FOR_ERROR
  }
  | ALLOCA Types ',' UINT ValueRef OptCAlign {
    Value* tmpVal = getVal($4, $5);
    CHECK_FOR_ERROR
    $$ = new AllocaInst(*$2, tmpVal, $6);
    delete $2;
  }
  | FREE ResolvedVal {
    if (!isa<PointerType>($2->getType()))
      GEN_ERROR("Trying to free nonpointer type " + 
                     $2->getType()->getDescription() + "!");
    $$ = new FreeInst($2);
    CHECK_FOR_ERROR
  }

  | OptVolatile LOAD Types ValueRef {
    if (!isa<PointerType>($3->get()))
      GEN_ERROR("Can't load from nonpointer type: " +
                     (*$3)->getDescription());
    if (!cast<PointerType>($3->get())->getElementType()->isFirstClassType())
      GEN_ERROR("Can't load from pointer of non-first-class type: " +
                     (*$3)->getDescription());
    Value* tmpVal = getVal(*$3, $4);
    CHECK_FOR_ERROR
    $$ = new LoadInst(tmpVal, "", $1);
    delete $3;
  }
  | OptVolatile STORE ResolvedVal ',' Types ValueRef {
    const PointerType *PT = dyn_cast<PointerType>($5->get());
    if (!PT)
      GEN_ERROR("Can't store to a nonpointer type: " +
                     (*$5)->getDescription());
    const Type *ElTy = PT->getElementType();
    if (ElTy != $3->getType())
      GEN_ERROR("Can't store '" + $3->getType()->getDescription() +
                     "' into space of type '" + ElTy->getDescription() + "'!");

    Value* tmpVal = getVal(*$5, $6);
    CHECK_FOR_ERROR
    $$ = new StoreInst($3, tmpVal, $1);
    delete $5;
  }
  | GETELEMENTPTR Types ValueRef IndexList {
    if (!isa<PointerType>($2->get()))
      GEN_ERROR("getelementptr insn requires pointer operand!");

    // LLVM 1.2 and earlier used ubyte struct indices.  Convert any ubyte struct
    // indices to uint struct indices for compatibility.
    generic_gep_type_iterator<std::vector<Value*>::iterator>
      GTI = gep_type_begin($2->get(), $4->begin(), $4->end()),
      GTE = gep_type_end($2->get(), $4->begin(), $4->end());
    for (unsigned i = 0, e = $4->size(); i != e && GTI != GTE; ++i, ++GTI)
      if (isa<StructType>(*GTI))        // Only change struct indices
        if (ConstantInt *CUI = dyn_cast<ConstantInt>((*$4)[i]))
          if (CUI->getType() == Type::UByteTy)
            (*$4)[i] = ConstantExpr::getCast(CUI, Type::UIntTy);

    if (!GetElementPtrInst::getIndexedType(*$2, *$4, true))
      GEN_ERROR("Invalid getelementptr indices for type '" +
                     (*$2)->getDescription()+ "'!");
    Value* tmpVal = getVal(*$2, $3);
    CHECK_FOR_ERROR
    $$ = new GetElementPtrInst(tmpVal, *$4);
    delete $2; 
    delete $4;
  };


%%

void llvm::GenerateError(const std::string &message, int LineNo) {
  if (LineNo == -1) LineNo = llvmAsmlineno;
  // TODO: column number in exception
  if (TheParseError)
    TheParseError->setError(CurFilename, message, LineNo);
  TriggerError = 1;
}

int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + utostr((unsigned) llvmAsmlineno) + ": ";
  std::string errMsg = std::string(ErrorMsg) + "\n" + where + " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(llvmAsmtext, llvmAsmleng) + "'";
  GenerateError(errMsg);
  return 0;
}
