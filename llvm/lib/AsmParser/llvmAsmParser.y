//===-- llvmAsmParser.y - Parser for llvm assembly files --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ValueSymbolTable.h"
#include "llvm/AutoUpgrade.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Streams.h"
#include <algorithm>
#include <list>
#include <map>
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
using namespace llvm;

static Module *ParserResult;

// DEBUG_UPREFS - Define this symbol if you want to enable debugging output
// relating to upreferences in the input stream.
//
//#define DEBUG_UPREFS 1
#ifdef DEBUG_UPREFS
#define UR_OUT(X) cerr << X
#else
#define UR_OUT(X)
#endif

#define YYERROR_VERBOSE 1

static GlobalVariable *CurGV;


// This contains info used when building the body of a function.  It is
// destroyed when the function is completed.
//
typedef std::vector<Value *> ValueList;           // Numbered defs

static void
ResolveDefinitions(ValueList &LateResolvers, ValueList *FutureLateResolvers=0);

static struct PerModuleInfo {
  Module *CurrentModule;
  ValueList Values; // Module level numbered definitions
  ValueList LateResolveValues;
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
      UpgradeCallsToIntrinsic(FI++); // must be post-increment, as we remove

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

  bool TypeIsUnresolved(PATypeHolder* PATy) {
    // If it isn't abstract, its resolved
    const Type* Ty = PATy->get();
    if (!Ty->isAbstract())
      return false;
    // Traverse the type looking for abstract types. If it isn't abstract then
    // we don't need to traverse that leg of the type.
    std::vector<const Type*> WorkList, SeenList;
    WorkList.push_back(Ty);
    while (!WorkList.empty()) {
      const Type* Ty = WorkList.back();
      SeenList.push_back(Ty);
      WorkList.pop_back();
      if (const OpaqueType* OpTy = dyn_cast<OpaqueType>(Ty)) {
        // Check to see if this is an unresolved type
        std::map<ValID, PATypeHolder>::iterator I = LateResolveTypes.begin();
        std::map<ValID, PATypeHolder>::iterator E = LateResolveTypes.end();
        for ( ; I != E; ++I) {
          if (I->second.get() == OpTy)
            return true;
        }
      } else if (const SequentialType* SeqTy = dyn_cast<SequentialType>(Ty)) {
        const Type* TheTy = SeqTy->getElementType();
        if (TheTy->isAbstract() && TheTy != Ty) {
          std::vector<const Type*>::iterator I = SeenList.begin(),
                                             E = SeenList.end();
          for ( ; I != E; ++I)
            if (*I == TheTy)
              break;
          if (I == E)
            WorkList.push_back(TheTy);
        }
      } else if (const StructType* StrTy = dyn_cast<StructType>(Ty)) {
        for (unsigned i = 0; i < StrTy->getNumElements(); ++i) {
          const Type* TheTy = StrTy->getElementType(i);
          if (TheTy->isAbstract() && TheTy != Ty) {
            std::vector<const Type*>::iterator I = SeenList.begin(),
                                               E = SeenList.end();
            for ( ; I != E; ++I)
              if (*I == TheTy)
                break;
            if (I == E)
              WorkList.push_back(TheTy);
          }
        }
      }
    }
    return false;
  }
} CurModule;

static struct PerFunctionInfo {
  Function *CurrentFunction;     // Pointer to current function being created

  ValueList Values; // Keep track of #'d definitions
  unsigned NextValNum;
  ValueList LateResolveValues;
  bool isDeclare;                   // Is this function a forward declararation?
  GlobalValue::LinkageTypes Linkage; // Linkage for forward declaration.
  GlobalValue::VisibilityTypes Visibility;

  /// BBForwardRefs - When we see forward references to basic blocks, keep
  /// track of them here.
  std::map<ValID, BasicBlock*> BBForwardRefs;

  inline PerFunctionInfo() {
    CurrentFunction = 0;
    isDeclare = false;
    Linkage = GlobalValue::ExternalLinkage;
    Visibility = GlobalValue::DefaultVisibility;
  }

  inline void FunctionStart(Function *M) {
    CurrentFunction = M;
    NextValNum = 0;
  }

  void FunctionDone() {
    // Any forward referenced blocks left?
    if (!BBForwardRefs.empty()) {
      GenerateError("Undefined reference to label " +
                     BBForwardRefs.begin()->second->getName());
      return;
    }

    // Resolve all forward references now.
    ResolveDefinitions(LateResolveValues, &CurModule.LateResolveValues);

    Values.clear();         // Clear out function local definitions
    BBForwardRefs.clear();
    CurrentFunction = 0;
    isDeclare = false;
    Linkage = GlobalValue::ExternalLinkage;
    Visibility = GlobalValue::DefaultVisibility;
  }
} CurFun;  // Info for the current function...

static bool inFunctionScope() { return CurFun.CurrentFunction != 0; }


//===----------------------------------------------------------------------===//
//               Code to handle definitions of all the types
//===----------------------------------------------------------------------===//

/// InsertValue - Insert a value into the value table.  If it is named, this
/// returns -1, otherwise it returns the slot number for the value.
static int InsertValue(Value *V, ValueList &ValueTab = CurFun.Values) {
  // Things that have names or are void typed don't get slot numbers
  if (V->hasName() || (V->getType() == Type::VoidTy))
    return -1;

  // In the case of function values, we have to allow for the forward reference
  // of basic blocks, which are included in the numbering. Consequently, we keep
  // track of the next insertion location with NextValNum. When a BB gets
  // inserted, it could change the size of the CurFun.Values vector.
  if (&ValueTab == &CurFun.Values) {
    if (ValueTab.size() <= CurFun.NextValNum)
      ValueTab.resize(CurFun.NextValNum+1);
    ValueTab[CurFun.NextValNum++] = V;
    return CurFun.NextValNum-1;
  }
  // For all other lists, its okay to just tack it on the back of the vector.
  ValueTab.push_back(V);
  return ValueTab.size()-1;
}

static const Type *getTypeVal(const ValID &D, bool DoNotImprovise = false) {
  switch (D.Type) {
  case ValID::LocalID:               // Is it a numbered definition?
    // Module constants occupy the lowest numbered slots...
    if (D.Num < CurModule.Types.size())
      return CurModule.Types[D.Num];
    break;
  case ValID::LocalName:                 // Is it a named definition?
    if (const Type *N = CurModule.CurrentModule->getTypeByName(D.getName())) {
      D.destroy();  // Free old strdup'd memory...
      return N;
    }
    break;
  default:
    GenerateError("Internal parser error: Invalid symbol type reference");
    return 0;
  }

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  if (DoNotImprovise) return 0;  // Do we just want a null to be returned?


  if (inFunctionScope()) {
    if (D.Type == ValID::LocalName) {
      GenerateError("Reference to an undefined type: '" + D.getName() + "'");
      return 0;
    } else {
      GenerateError("Reference to an undefined type: #" + utostr(D.Num));
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

// getExistingVal - Look up the value specified by the provided type and
// the provided ValID.  If the value exists and has already been defined, return
// it.  Otherwise return null.
//
static Value *getExistingVal(const Type *Ty, const ValID &D) {
  if (isa<FunctionType>(Ty)) {
    GenerateError("Functions are not values and "
                   "must be referenced as pointers");
    return 0;
  }

  switch (D.Type) {
  case ValID::LocalID: {                 // Is it a numbered definition?
    // Check that the number is within bounds.
    if (D.Num >= CurFun.Values.size())
      return 0;
    Value *Result = CurFun.Values[D.Num];
    if (Ty != Result->getType()) {
      GenerateError("Numbered value (%" + utostr(D.Num) + ") of type '" +
                    Result->getType()->getDescription() + "' does not match "
                    "expected type, '" + Ty->getDescription() + "'");
      return 0;
    }
    return Result;
  }
  case ValID::GlobalID: {                 // Is it a numbered definition?
    if (D.Num >= CurModule.Values.size())
      return 0;
    Value *Result = CurModule.Values[D.Num];
    if (Ty != Result->getType()) {
      GenerateError("Numbered value (@" + utostr(D.Num) + ") of type '" +
                    Result->getType()->getDescription() + "' does not match "
                    "expected type, '" + Ty->getDescription() + "'");
      return 0;
    }
    return Result;
  }

  case ValID::LocalName: {                // Is it a named definition?
    if (!inFunctionScope())
      return 0;
    ValueSymbolTable &SymTab = CurFun.CurrentFunction->getValueSymbolTable();
    Value *N = SymTab.lookup(D.getName());
    if (N == 0)
      return 0;
    if (N->getType() != Ty)
      return 0;

    D.destroy();  // Free old strdup'd memory...
    return N;
  }
  case ValID::GlobalName: {                // Is it a named definition?
    ValueSymbolTable &SymTab = CurModule.CurrentModule->getValueSymbolTable();
    Value *N = SymTab.lookup(D.getName());
    if (N == 0)
      return 0;
    if (N->getType() != Ty)
      return 0;

    D.destroy();  // Free old strdup'd memory...
    return N;
  }

  // Check to make sure that "Ty" is an integral type, and that our
  // value will fit into the specified type...
  case ValID::ConstSIntVal:    // Is it a constant pool reference??
    if (!isa<IntegerType>(Ty) ||
        !ConstantInt::isValueValidForType(Ty, D.ConstPool64)) {
      GenerateError("Signed integral constant '" +
                     itostr(D.ConstPool64) + "' is invalid for type '" +
                     Ty->getDescription() + "'");
      return 0;
    }
    return ConstantInt::get(Ty, D.ConstPool64, true);

  case ValID::ConstUIntVal:     // Is it an unsigned const pool reference?
    if (isa<IntegerType>(Ty) &&
        ConstantInt::isValueValidForType(Ty, D.UConstPool64))
      return ConstantInt::get(Ty, D.UConstPool64);

    if (!isa<IntegerType>(Ty) ||
        !ConstantInt::isValueValidForType(Ty, D.ConstPool64)) {
      GenerateError("Integral constant '" + utostr(D.UConstPool64) +
                    "' is invalid or out of range for type '" +
                    Ty->getDescription() + "'");
      return 0;
    }
    // This is really a signed reference.  Transmogrify.
    return ConstantInt::get(Ty, D.ConstPool64, true);

  case ValID::ConstAPInt:     // Is it an unsigned const pool reference?
    if (!isa<IntegerType>(Ty)) {
      GenerateError("Integral constant '" + D.getName() +
                    "' is invalid or out of range for type '" +
                    Ty->getDescription() + "'");
      return 0;
    }

    {
      APSInt Tmp = *D.ConstPoolInt;
      Tmp.extOrTrunc(Ty->getPrimitiveSizeInBits());
      return ConstantInt::get(Tmp);
    }

  case ValID::ConstFPVal:        // Is it a floating point const pool reference?
    if (!Ty->isFloatingPoint() ||
        !ConstantFP::isValueValidForType(Ty, *D.ConstPoolFP)) {
      GenerateError("FP constant invalid for type");
      return 0;
    }
    // Lexer has no type info, so builds all float and double FP constants
    // as double.  Fix this here.  Long double does not need this.
    if (&D.ConstPoolFP->getSemantics() == &APFloat::IEEEdouble &&
        Ty==Type::FloatTy)
      D.ConstPoolFP->convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven);
    return ConstantFP::get(*D.ConstPoolFP);

  case ValID::ConstNullVal:      // Is it a null value?
    if (!isa<PointerType>(Ty)) {
      GenerateError("Cannot create a a non pointer null");
      return 0;
    }
    return ConstantPointerNull::get(cast<PointerType>(Ty));

  case ValID::ConstUndefVal:      // Is it an undef value?
    return UndefValue::get(Ty);

  case ValID::ConstZeroVal:      // Is it a zero value?
    return Constant::getNullValue(Ty);

  case ValID::ConstantVal:       // Fully resolved constant?
    if (D.ConstantValue->getType() != Ty) {
      GenerateError("Constant expression type different from required type");
      return 0;
    }
    return D.ConstantValue;

  case ValID::InlineAsmVal: {    // Inline asm expression
    const PointerType *PTy = dyn_cast<PointerType>(Ty);
    const FunctionType *FTy =
      PTy ? dyn_cast<FunctionType>(PTy->getElementType()) : 0;
    if (!FTy || !InlineAsm::Verify(FTy, D.IAD->Constraints)) {
      GenerateError("Invalid type for asm constraint string");
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

// getVal - This function is identical to getExistingVal, except that if a
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
  Value *V = getExistingVal(Ty, ID);
  if (V) return V;
  if (TriggerError) return 0;

  if (!Ty->isFirstClassType() && !isa<OpaqueType>(Ty)) {
    GenerateError("Invalid use of a non-first-class type");
    return 0;
  }

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  switch (ID.Type) {
  case ValID::GlobalName:
  case ValID::GlobalID: {
   const PointerType *PTy = dyn_cast<PointerType>(Ty);
   if (!PTy) {
     GenerateError("Invalid type for reference to global" );
     return 0;
   }
   const Type* ElTy = PTy->getElementType();
   if (const FunctionType *FTy = dyn_cast<FunctionType>(ElTy))
     V = Function::Create(FTy, GlobalValue::ExternalLinkage);
   else
     V = new GlobalVariable(ElTy, false, GlobalValue::ExternalLinkage, 0, "",
                            (Module*)0, false, PTy->getAddressSpace());
   break;
  }
  default:
   V = new Argument(Ty);
  }

  // Remember where this forward reference came from.  FIXME, shouldn't we try
  // to recycle these things??
  CurModule.PlaceHolderInfo.insert(std::make_pair(V, std::make_pair(ID,
                                                              LLLgetLineNo())));

  if (inFunctionScope())
    InsertValue(V, CurFun.LateResolveValues);
  else
    InsertValue(V, CurModule.LateResolveValues);
  return V;
}

/// defineBBVal - This is a definition of a new basic block with the specified
/// identifier which must be the same as CurFun.NextValNum, if its numeric.
static BasicBlock *defineBBVal(const ValID &ID) {
  assert(inFunctionScope() && "Can't get basic block at global scope!");

  BasicBlock *BB = 0;

  // First, see if this was forward referenced

  std::map<ValID, BasicBlock*>::iterator BBI = CurFun.BBForwardRefs.find(ID);
  if (BBI != CurFun.BBForwardRefs.end()) {
    BB = BBI->second;
    // The forward declaration could have been inserted anywhere in the
    // function: insert it into the correct place now.
    CurFun.CurrentFunction->getBasicBlockList().remove(BB);
    CurFun.CurrentFunction->getBasicBlockList().push_back(BB);

    // We're about to erase the entry, save the key so we can clean it up.
    ValID Tmp = BBI->first;

    // Erase the forward ref from the map as its no longer "forward"
    CurFun.BBForwardRefs.erase(ID);

    // The key has been removed from the map but so we don't want to leave
    // strdup'd memory around so destroy it too.
    Tmp.destroy();

    // If its a numbered definition, bump the number and set the BB value.
    if (ID.Type == ValID::LocalID) {
      assert(ID.Num == CurFun.NextValNum && "Invalid new block number");
      InsertValue(BB);
    }
  } else {
    // We haven't seen this BB before and its first mention is a definition.
    // Just create it and return it.
    std::string Name (ID.Type == ValID::LocalName ? ID.getName() : "");
    BB = BasicBlock::Create(Name, CurFun.CurrentFunction);
    if (ID.Type == ValID::LocalID) {
      assert(ID.Num == CurFun.NextValNum && "Invalid new block number");
      InsertValue(BB);
    }
  }

  ID.destroy();
  return BB;
}

/// getBBVal - get an existing BB value or create a forward reference for it.
///
static BasicBlock *getBBVal(const ValID &ID) {
  assert(inFunctionScope() && "Can't get basic block at global scope!");

  BasicBlock *BB =  0;

  std::map<ValID, BasicBlock*>::iterator BBI = CurFun.BBForwardRefs.find(ID);
  if (BBI != CurFun.BBForwardRefs.end()) {
    BB = BBI->second;
  } if (ID.Type == ValID::LocalName) {
    std::string Name = ID.getName();
    Value *N = CurFun.CurrentFunction->getValueSymbolTable().lookup(Name);
    if (N) {
      if (N->getType()->getTypeID() == Type::LabelTyID)
        BB = cast<BasicBlock>(N);
      else
        GenerateError("Reference to label '" + Name + "' is actually of type '"+
          N->getType()->getDescription() + "'");
    }
  } else if (ID.Type == ValID::LocalID) {
    if (ID.Num < CurFun.NextValNum && ID.Num < CurFun.Values.size()) {
      if (CurFun.Values[ID.Num]->getType()->getTypeID() == Type::LabelTyID)
        BB = cast<BasicBlock>(CurFun.Values[ID.Num]);
      else
        GenerateError("Reference to label '%" + utostr(ID.Num) +
          "' is actually of type '"+
          CurFun.Values[ID.Num]->getType()->getDescription() + "'");
    }
  } else {
    GenerateError("Illegal label reference " + ID.getName());
    return 0;
  }

  // If its already been defined, return it now.
  if (BB) {
    ID.destroy(); // Free strdup'd memory.
    return BB;
  }

  // Otherwise, this block has not been seen before, create it.
  std::string Name;
  if (ID.Type == ValID::LocalName)
    Name = ID.getName();
  BB = BasicBlock::Create(Name, CurFun.CurrentFunction);

  // Insert it in the forward refs map.
  CurFun.BBForwardRefs[ID] = BB;

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
ResolveDefinitions(ValueList &LateResolvers, ValueList *FutureLateResolvers) {
  // Loop over LateResolveDefs fixing up stuff that couldn't be resolved
  while (!LateResolvers.empty()) {
    Value *V = LateResolvers.back();
    LateResolvers.pop_back();

    std::map<Value*, std::pair<ValID, int> >::iterator PHI =
      CurModule.PlaceHolderInfo.find(V);
    assert(PHI != CurModule.PlaceHolderInfo.end() && "Placeholder error!");

    ValID &DID = PHI->second.first;

    Value *TheRealValue = getExistingVal(V->getType(), DID);
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
      if (DID.Type == ValID::LocalName || DID.Type == ValID::GlobalName) {
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
  LateResolvers.clear();
}

// ResolveTypeTo - A brand new type was just declared.  This means that (if
// name is not null) things referencing Name can be resolved.  Otherwise, things
// refering to the number can be resolved.  Do this now.
//
static void ResolveTypeTo(std::string *Name, const Type *ToTy) {
  ValID D;
  if (Name)
    D = ValID::createLocalName(*Name);
  else
    D = ValID::createLocalID(CurModule.Types.size());

  std::map<ValID, PATypeHolder>::iterator I =
    CurModule.LateResolveTypes.find(D);
  if (I != CurModule.LateResolveTypes.end()) {
    ((DerivedType*)I->second.get())->refineAbstractTypeTo(ToTy);
    CurModule.LateResolveTypes.erase(I);
  }
  D.destroy();
}

// setValueName - Set the specified value to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is free'd by this function.
//
static void setValueName(Value *V, std::string *NameStr) {
  if (!NameStr) return;
  std::string Name(*NameStr);      // Copy string
  delete NameStr;                  // Free old string

  if (V->getType() == Type::VoidTy) {
    GenerateError("Can't assign name '" + Name+"' to value with void type");
    return;
  }

  assert(inFunctionScope() && "Must be in function scope!");
  ValueSymbolTable &ST = CurFun.CurrentFunction->getValueSymbolTable();
  if (ST.lookup(Name)) {
    GenerateError("Redefinition of value '" + Name + "' of type '" +
                   V->getType()->getDescription() + "'");
    return;
  }

  // Set the name.
  V->setName(Name);
}

/// ParseGlobalVariable - Handle parsing of a global.  If Initializer is null,
/// this is a declaration, otherwise it is a definition.
static GlobalVariable *
ParseGlobalVariable(std::string *NameStr,
                    GlobalValue::LinkageTypes Linkage,
                    GlobalValue::VisibilityTypes Visibility,
                    bool isConstantGlobal, const Type *Ty,
                    Constant *Initializer, bool IsThreadLocal,
                    unsigned AddressSpace = 0) {
  if (isa<FunctionType>(Ty)) {
    GenerateError("Cannot declare global vars of function type");
    return 0;
  }
  if (Ty == Type::LabelTy) {
    GenerateError("Cannot declare global vars of label type");
    return 0;
  }

  const PointerType *PTy = PointerType::get(Ty, AddressSpace);

  std::string Name;
  if (NameStr) {
    Name = *NameStr;      // Copy string
    delete NameStr;       // Free old string
  }

  // See if this global value was forward referenced.  If so, recycle the
  // object.
  ValID ID;
  if (!Name.empty()) {
    ID = ValID::createGlobalName(Name);
  } else {
    ID = ValID::createGlobalID(CurModule.Values.size());
  }

  if (GlobalValue *FWGV = CurModule.GetForwardRefForGlobal(PTy, ID)) {
    // Move the global to the end of the list, from whereever it was
    // previously inserted.
    GlobalVariable *GV = cast<GlobalVariable>(FWGV);
    CurModule.CurrentModule->getGlobalList().remove(GV);
    CurModule.CurrentModule->getGlobalList().push_back(GV);
    GV->setInitializer(Initializer);
    GV->setLinkage(Linkage);
    GV->setVisibility(Visibility);
    GV->setConstant(isConstantGlobal);
    GV->setThreadLocal(IsThreadLocal);
    InsertValue(GV, CurModule.Values);
    ID.destroy();
    return GV;
  }

  ID.destroy();

  // If this global has a name
  if (!Name.empty()) {
    // if the global we're parsing has an initializer (is a definition) and
    // has external linkage.
    if (Initializer && Linkage != GlobalValue::InternalLinkage)
      // If there is already a global with external linkage with this name
      if (CurModule.CurrentModule->getGlobalVariable(Name, false)) {
        // If we allow this GVar to get created, it will be renamed in the
        // symbol table because it conflicts with an existing GVar. We can't
        // allow redefinition of GVars whose linking indicates that their name
        // must stay the same. Issue the error.
        GenerateError("Redefinition of global variable named '" + Name +
                       "' of type '" + Ty->getDescription() + "'");
        return 0;
      }
  }

  // Otherwise there is no existing GV to use, create one now.
  GlobalVariable *GV =
    new GlobalVariable(Ty, isConstantGlobal, Linkage, Initializer, Name,
                       CurModule.CurrentModule, IsThreadLocal, AddressSpace);
  GV->setVisibility(Visibility);
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
static bool setTypeName(const Type *T, std::string *NameStr) {
  assert(!inFunctionScope() && "Can't give types function-local names!");
  if (NameStr == 0) return false;

  std::string Name(*NameStr);      // Copy string
  delete NameStr;                  // Free old string

  // We don't allow assigning names to void type
  if (T == Type::VoidTy) {
    GenerateError("Can't assign name '" + Name + "' to the void type");
    return false;
  }

  // Set the type name, checking for conflicts as we do so.
  bool AlreadyExists = CurModule.CurrentModule->addTypeName(Name, T);

  if (AlreadyExists) {   // Inserting a name that is already defined???
    const Type *Existing = CurModule.CurrentModule->getTypeByName(Name);
    assert(Existing && "Conflict but no matching type?!");

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
    GenerateError("Redefinition of type named '" + Name + "' of type '" +
                   T->getDescription() + "'");
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

//===----------------------------------------------------------------------===//
//            RunVMAsmParser - Define an interface to this parser
//===----------------------------------------------------------------------===//
//
static Module* RunParser(Module * M);

Module *llvm::RunVMAsmParser(llvm::MemoryBuffer *MB) {
  InitLLLexer(MB);
  Module *M = RunParser(new Module(LLLgetFilename()));
  FreeLexer();
  return M;
}

%}

%union {
  llvm::Module                           *ModuleVal;
  llvm::Function                         *FunctionVal;
  llvm::BasicBlock                       *BasicBlockVal;
  llvm::TerminatorInst                   *TermInstVal;
  llvm::Instruction                      *InstVal;
  llvm::Constant                         *ConstVal;

  const llvm::Type                       *PrimType;
  std::list<llvm::PATypeHolder>          *TypeList;
  llvm::PATypeHolder                     *TypeVal;
  llvm::Value                            *ValueVal;
  std::vector<llvm::Value*>              *ValueList;
  std::vector<unsigned>                  *ConstantList;
  llvm::ArgListType                      *ArgList;
  llvm::TypeWithAttrs                     TypeWithAttrs;
  llvm::TypeWithAttrsList                *TypeWithAttrsList;
  llvm::ParamList                        *ParamList;

  // Represent the RHS of PHI node
  std::list<std::pair<llvm::Value*,
                      llvm::BasicBlock*> > *PHIList;
  std::vector<std::pair<llvm::Constant*, llvm::BasicBlock*> > *JumpTable;
  std::vector<llvm::Constant*>           *ConstVector;

  llvm::GlobalValue::LinkageTypes         Linkage;
  llvm::GlobalValue::VisibilityTypes      Visibility;
  llvm::Attributes                  Attributes;
  llvm::APInt                       *APIntVal;
  int64_t                           SInt64Val;
  uint64_t                          UInt64Val;
  int                               SIntVal;
  unsigned                          UIntVal;
  llvm::APFloat                    *FPVal;
  bool                              BoolVal;

  std::string                      *StrVal;   // This memory must be deleted
  llvm::ValID                       ValIDVal;

  llvm::Instruction::BinaryOps      BinaryOpVal;
  llvm::Instruction::TermOps        TermOpVal;
  llvm::Instruction::MemoryOps      MemOpVal;
  llvm::Instruction::CastOps        CastOpVal;
  llvm::Instruction::OtherOps       OtherOpVal;
  llvm::ICmpInst::Predicate         IPredicate;
  llvm::FCmpInst::Predicate         FPredicate;
}

%type <ModuleVal>     Module
%type <FunctionVal>   Function FunctionProto FunctionHeader BasicBlockList
%type <BasicBlockVal> BasicBlock InstructionList
%type <TermInstVal>   BBTerminatorInst
%type <InstVal>       Inst InstVal MemoryInst
%type <ConstVal>      ConstVal ConstExpr AliaseeRef
%type <ConstVector>   ConstVector
%type <ArgList>       ArgList ArgListH
%type <PHIList>       PHIList
%type <ParamList>     ParamList      // For call param lists & GEP indices
%type <ValueList>     IndexList         // For GEP indices
%type <ConstantList>  ConstantIndexList // For insertvalue/extractvalue indices
%type <TypeList>      TypeListI
%type <TypeWithAttrsList> ArgTypeList ArgTypeListI
%type <TypeWithAttrs> ArgType
%type <JumpTable>     JumpTable
%type <BoolVal>       GlobalType                  // GLOBAL or CONSTANT?
%type <BoolVal>       ThreadLocal                 // 'thread_local' or not
%type <BoolVal>       OptVolatile                 // 'volatile' or not
%type <BoolVal>       OptTailCall                 // TAIL CALL or plain CALL.
%type <BoolVal>       OptSideEffect               // 'sideeffect' or not.
%type <Linkage>       GVInternalLinkage GVExternalLinkage
%type <Linkage>       FunctionDefineLinkage FunctionDeclareLinkage
%type <Linkage>       AliasLinkage
%type <Visibility>    GVVisibilityStyle

// ValueRef - Unresolved reference to a definition or BB
%type <ValIDVal>      ValueRef ConstValueRef SymbolicValueRef
%type <ValueVal>      ResolvedVal            // <type> <valref> pair
%type <ValueList>     ReturnedVal
// Tokens and types for handling constant integer values
//
// ESINT64VAL - A negative number within long long range
%token <SInt64Val> ESINT64VAL

// EUINT64VAL - A positive number within uns. long long range
%token <UInt64Val> EUINT64VAL

// ESAPINTVAL - A negative number with arbitrary precision
%token <APIntVal>  ESAPINTVAL

// EUAPINTVAL - A positive number with arbitrary precision
%token <APIntVal>  EUAPINTVAL

%token  <UIntVal>   LOCALVAL_ID GLOBALVAL_ID  // %123 @123
%token  <FPVal>     FPVAL     // Float or Double constant

// Built in types...
%type  <TypeVal> Types ResultTypes
%type  <PrimType> IntType FPType PrimType           // Classifications
%token <PrimType> VOID INTTYPE
%token <PrimType> FLOAT DOUBLE X86_FP80 FP128 PPC_FP128 LABEL
%token TYPE


%token<StrVal> LOCALVAR GLOBALVAR LABELSTR
%token<StrVal> STRINGCONSTANT ATSTRINGCONSTANT PCTSTRINGCONSTANT
%type <StrVal> LocalName OptLocalName OptLocalAssign
%type <StrVal> GlobalName OptGlobalAssign GlobalAssign
%type <StrVal> OptSection SectionString OptGC

%type <UIntVal> OptAlign OptCAlign OptAddrSpace

%token ZEROINITIALIZER TRUETOK FALSETOK BEGINTOK ENDTOK
%token DECLARE DEFINE GLOBAL CONSTANT SECTION ALIAS VOLATILE THREAD_LOCAL
%token TO DOTDOTDOT NULL_TOK UNDEF INTERNAL LINKONCE WEAK APPENDING
%token DLLIMPORT DLLEXPORT EXTERN_WEAK COMMON
%token OPAQUE EXTERNAL TARGET TRIPLE ALIGN ADDRSPACE
%token DEPLIBS CALL TAIL ASM_TOK MODULE SIDEEFFECT
%token CC_TOK CCC_TOK FASTCC_TOK COLDCC_TOK X86_STDCALLCC_TOK X86_FASTCALLCC_TOK
%token DATALAYOUT
%type <UIntVal> OptCallingConv LocalNumber
%type <Attributes> OptAttributes Attribute
%type <Attributes> OptFuncAttrs  FuncAttr
%type <Attributes> OptRetAttrs  RetAttr

// Basic Block Terminating Operators
%token <TermOpVal> RET BR SWITCH INVOKE UNWIND UNREACHABLE

// Binary Operators
%type  <BinaryOpVal> ArithmeticOps LogicalOps // Binops Subcatagories
%token <BinaryOpVal> ADD SUB MUL UDIV SDIV FDIV UREM SREM FREM AND OR XOR
%token <BinaryOpVal> SHL LSHR ASHR

%token <OtherOpVal> ICMP FCMP VICMP VFCMP
%type  <IPredicate> IPredicates
%type  <FPredicate> FPredicates
%token  EQ NE SLT SGT SLE SGE ULT UGT ULE UGE
%token  OEQ ONE OLT OGT OLE OGE ORD UNO UEQ UNE

// Memory Instructions
%token <MemOpVal> MALLOC ALLOCA FREE LOAD STORE GETELEMENTPTR

// Cast Operators
%type <CastOpVal> CastOps
%token <CastOpVal> TRUNC ZEXT SEXT FPTRUNC FPEXT BITCAST
%token <CastOpVal> UITOFP SITOFP FPTOUI FPTOSI INTTOPTR PTRTOINT

// Other Operators
%token <OtherOpVal> PHI_TOK SELECT VAARG
%token <OtherOpVal> EXTRACTELEMENT INSERTELEMENT SHUFFLEVECTOR
%token <OtherOpVal> GETRESULT
%token <OtherOpVal> EXTRACTVALUE INSERTVALUE

// Function Attributes
%token SIGNEXT ZEROEXT NORETURN INREG SRET NOUNWIND NOALIAS BYVAL NEST
%token READNONE READONLY GC OPTSIZE NOINLINE ALWAYSINLINE

// Visibility Styles
%token DEFAULT HIDDEN PROTECTED

%start Module
%%


// Operations that are notably excluded from this list include:
// RET, BR, & SWITCH because they end basic blocks and are treated specially.
//
ArithmeticOps: ADD | SUB | MUL | UDIV | SDIV | FDIV | UREM | SREM | FREM;
LogicalOps   : SHL | LSHR | ASHR | AND | OR | XOR;
CastOps      : TRUNC | ZEXT | SEXT | FPTRUNC | FPEXT | BITCAST |
               UITOFP | SITOFP | FPTOUI | FPTOSI | INTTOPTR | PTRTOINT;

IPredicates
  : EQ   { $$ = ICmpInst::ICMP_EQ; }  | NE   { $$ = ICmpInst::ICMP_NE; }
  | SLT  { $$ = ICmpInst::ICMP_SLT; } | SGT  { $$ = ICmpInst::ICMP_SGT; }
  | SLE  { $$ = ICmpInst::ICMP_SLE; } | SGE  { $$ = ICmpInst::ICMP_SGE; }
  | ULT  { $$ = ICmpInst::ICMP_ULT; } | UGT  { $$ = ICmpInst::ICMP_UGT; }
  | ULE  { $$ = ICmpInst::ICMP_ULE; } | UGE  { $$ = ICmpInst::ICMP_UGE; }
  ;

FPredicates
  : OEQ  { $$ = FCmpInst::FCMP_OEQ; } | ONE  { $$ = FCmpInst::FCMP_ONE; }
  | OLT  { $$ = FCmpInst::FCMP_OLT; } | OGT  { $$ = FCmpInst::FCMP_OGT; }
  | OLE  { $$ = FCmpInst::FCMP_OLE; } | OGE  { $$ = FCmpInst::FCMP_OGE; }
  | ORD  { $$ = FCmpInst::FCMP_ORD; } | UNO  { $$ = FCmpInst::FCMP_UNO; }
  | UEQ  { $$ = FCmpInst::FCMP_UEQ; } | UNE  { $$ = FCmpInst::FCMP_UNE; }
  | ULT  { $$ = FCmpInst::FCMP_ULT; } | UGT  { $$ = FCmpInst::FCMP_UGT; }
  | ULE  { $$ = FCmpInst::FCMP_ULE; } | UGE  { $$ = FCmpInst::FCMP_UGE; }
  | TRUETOK { $$ = FCmpInst::FCMP_TRUE; }
  | FALSETOK { $$ = FCmpInst::FCMP_FALSE; }
  ;

// These are some types that allow classification if we only want a particular
// thing... for example, only a signed, unsigned, or integral type.
IntType :  INTTYPE;
FPType   : FLOAT | DOUBLE | PPC_FP128 | FP128 | X86_FP80;

LocalName : LOCALVAR | STRINGCONSTANT | PCTSTRINGCONSTANT ;
OptLocalName : LocalName | /*empty*/ { $$ = 0; };

OptAddrSpace : ADDRSPACE '(' EUINT64VAL ')' { $$=$3; }
             | /*empty*/                    { $$=0; };

/// OptLocalAssign - Value producing statements have an optional assignment
/// component.
OptLocalAssign : LocalName '=' {
    $$ = $1;
    CHECK_FOR_ERROR
  }
  | /*empty*/ {
    $$ = 0;
    CHECK_FOR_ERROR
  };

LocalNumber : LOCALVAL_ID '=' {
  $$ = $1;
  CHECK_FOR_ERROR
};


GlobalName : GLOBALVAR | ATSTRINGCONSTANT ;

OptGlobalAssign : GlobalAssign
  | /*empty*/ {
    $$ = 0;
    CHECK_FOR_ERROR
  };

GlobalAssign : GlobalName '=' {
    $$ = $1;
    CHECK_FOR_ERROR
  };

GVInternalLinkage
  : INTERNAL    { $$ = GlobalValue::InternalLinkage; }
  | WEAK        { $$ = GlobalValue::WeakLinkage; }
  | LINKONCE    { $$ = GlobalValue::LinkOnceLinkage; }
  | APPENDING   { $$ = GlobalValue::AppendingLinkage; }
  | DLLEXPORT   { $$ = GlobalValue::DLLExportLinkage; }
  | COMMON      { $$ = GlobalValue::CommonLinkage; }
  ;

GVExternalLinkage
  : DLLIMPORT   { $$ = GlobalValue::DLLImportLinkage; }
  | EXTERN_WEAK { $$ = GlobalValue::ExternalWeakLinkage; }
  | EXTERNAL    { $$ = GlobalValue::ExternalLinkage; }
  ;

GVVisibilityStyle
  : /*empty*/ { $$ = GlobalValue::DefaultVisibility;   }
  | DEFAULT   { $$ = GlobalValue::DefaultVisibility;   }
  | HIDDEN    { $$ = GlobalValue::HiddenVisibility;    }
  | PROTECTED { $$ = GlobalValue::ProtectedVisibility; }
  ;

FunctionDeclareLinkage
  : /*empty*/   { $$ = GlobalValue::ExternalLinkage; }
  | DLLIMPORT   { $$ = GlobalValue::DLLImportLinkage; }
  | EXTERN_WEAK { $$ = GlobalValue::ExternalWeakLinkage; }
  ;

FunctionDefineLinkage
  : /*empty*/   { $$ = GlobalValue::ExternalLinkage; }
  | INTERNAL    { $$ = GlobalValue::InternalLinkage; }
  | LINKONCE    { $$ = GlobalValue::LinkOnceLinkage; }
  | WEAK        { $$ = GlobalValue::WeakLinkage; }
  | DLLEXPORT   { $$ = GlobalValue::DLLExportLinkage; }
  ;

AliasLinkage
  : /*empty*/   { $$ = GlobalValue::ExternalLinkage; }
  | WEAK        { $$ = GlobalValue::WeakLinkage; }
  | INTERNAL    { $$ = GlobalValue::InternalLinkage; }
  ;

OptCallingConv : /*empty*/          { $$ = CallingConv::C; } |
                 CCC_TOK            { $$ = CallingConv::C; } |
                 FASTCC_TOK         { $$ = CallingConv::Fast; } |
                 COLDCC_TOK         { $$ = CallingConv::Cold; } |
                 X86_STDCALLCC_TOK  { $$ = CallingConv::X86_StdCall; } |
                 X86_FASTCALLCC_TOK { $$ = CallingConv::X86_FastCall; } |
                 CC_TOK EUINT64VAL  {
                   if ((unsigned)$2 != $2)
                     GEN_ERROR("Calling conv too large");
                   $$ = $2;
                  CHECK_FOR_ERROR
                 };

Attribute     : ZEROEXT { $$ = Attribute::ZExt;      }
              | ZEXT    { $$ = Attribute::ZExt;      }
              | SIGNEXT { $$ = Attribute::SExt;      }
              | SEXT    { $$ = Attribute::SExt;      }
              | INREG   { $$ = Attribute::InReg;     }
              | SRET    { $$ = Attribute::StructRet; }
              | NOALIAS { $$ = Attribute::NoAlias;   }
              | BYVAL   { $$ = Attribute::ByVal;     }
              | NEST    { $$ = Attribute::Nest;      }
              | ALIGN EUINT64VAL { $$ =
                          Attribute::constructAlignmentFromInt($2);    }
              ;

OptAttributes : /* empty */  { $$ = Attribute::None; }
              | OptAttributes Attribute {
                $$ = $1 | $2;
              }
              ;

RetAttr       : INREG    { $$ = Attribute::InReg;     }
              | ZEROEXT  { $$ = Attribute::ZExt;     }
              | SIGNEXT  { $$ = Attribute::SExt;     }
              ;

OptRetAttrs  : /* empty */ { $$ = Attribute::None; }
             | OptRetAttrs RetAttr {
               $$ = $1 | $2;
             }
             ;


FuncAttr      : NORETURN { $$ = Attribute::NoReturn; }
              | NOUNWIND { $$ = Attribute::NoUnwind; }
              | INREG    { $$ = Attribute::InReg;     }
              | ZEROEXT  { $$ = Attribute::ZExt;     }
              | SIGNEXT  { $$ = Attribute::SExt;     }
              | READNONE { $$ = Attribute::ReadNone; }
              | READONLY { $$ = Attribute::ReadOnly; }
              | NOINLINE { $$ = Attribute::NoInline; }
              | ALWAYSINLINE { $$ = Attribute::AlwaysInline; }
              | OPTSIZE { $$ = Attribute::OptimizeForSize; }
              ;

OptFuncAttrs  : /* empty */ { $$ = Attribute::None; }
              | OptFuncAttrs FuncAttr {
                $$ = $1 | $2;
              }
              ;


OptGC         : /* empty */ { $$ = 0; }
              | GC STRINGCONSTANT {
                $$ = $2;
              }
              ;

// OptAlign/OptCAlign - An optional alignment, and an optional alignment with
// a comma before it.
OptAlign : /*empty*/        { $$ = 0; } |
           ALIGN EUINT64VAL {
  $$ = $2;
  if ($$ != 0 && !isPowerOf2_32($$))
    GEN_ERROR("Alignment must be a power of two");
  CHECK_FOR_ERROR
};
OptCAlign : /*empty*/            { $$ = 0; } |
            ',' ALIGN EUINT64VAL {
  $$ = $3;
  if ($$ != 0 && !isPowerOf2_32($$))
    GEN_ERROR("Alignment must be a power of two");
  CHECK_FOR_ERROR
};



SectionString : SECTION STRINGCONSTANT {
  for (unsigned i = 0, e = $2->length(); i != e; ++i)
    if ((*$2)[i] == '"' || (*$2)[i] == '\\')
      GEN_ERROR("Invalid character in section name");
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
    CurGV->setSection(*$1);
    delete $1;
    CHECK_FOR_ERROR
  }
  | ALIGN EUINT64VAL {
    if ($2 != 0 && !isPowerOf2_32($2))
      GEN_ERROR("Alignment must be a power of two");
    CurGV->setAlignment($2);
    CHECK_FOR_ERROR
  };

//===----------------------------------------------------------------------===//
// Types includes all predefined types... except void, because it can only be
// used in specific contexts (function returning void for example).

// Derived types are added later...
//
PrimType : INTTYPE | FLOAT | DOUBLE | PPC_FP128 | FP128 | X86_FP80 | LABEL ;

Types
  : OPAQUE {
    $$ = new PATypeHolder(OpaqueType::get());
    CHECK_FOR_ERROR
  }
  | PrimType {
    $$ = new PATypeHolder($1);
    CHECK_FOR_ERROR
  }
  | Types OptAddrSpace '*' {                             // Pointer type?
    if (*$1 == Type::LabelTy)
      GEN_ERROR("Cannot form a pointer to a basic block");
    $$ = new PATypeHolder(HandleUpRefs(PointerType::get(*$1, $2)));
    delete $1;
    CHECK_FOR_ERROR
  }
  | SymbolicValueRef {            // Named types are also simple types...
    const Type* tmp = getTypeVal($1);
    CHECK_FOR_ERROR
    $$ = new PATypeHolder(tmp);
  }
  | '\\' EUINT64VAL {                   // Type UpReference
    if ($2 > (uint64_t)~0U) GEN_ERROR("Value out of range");
    OpaqueType *OT = OpaqueType::get();        // Use temporary placeholder
    UpRefs.push_back(UpRefRecord((unsigned)$2, OT));  // Add to vector...
    $$ = new PATypeHolder(OT);
    UR_OUT("New Upreference!\n");
    CHECK_FOR_ERROR
  }
  | Types '(' ArgTypeListI ')' OptFuncAttrs {
    // Allow but ignore attributes on function types; this permits auto-upgrade.
    // FIXME: remove in LLVM 3.0.
    const Type *RetTy = *$1;
    if (!FunctionType::isValidReturnType(RetTy))
      GEN_ERROR("Invalid result type for LLVM function");

    std::vector<const Type*> Params;
    TypeWithAttrsList::iterator I = $3->begin(), E = $3->end();
    for (; I != E; ++I ) {
      const Type *Ty = I->Ty->get();
      Params.push_back(Ty);
    }

    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    for (unsigned i = 0; i != Params.size(); ++i)
      if (!(Params[i]->isFirstClassType() || isa<OpaqueType>(Params[i])))
        GEN_ERROR("Function arguments must be value types!");

    CHECK_FOR_ERROR

    FunctionType *FT = FunctionType::get(RetTy, Params, isVarArg);
    delete $1;   // Delete the return type handle
    $$ = new PATypeHolder(HandleUpRefs(FT));

    // Delete the argument list
    for (I = $3->begin() ; I != E; ++I ) {
      delete I->Ty;
    }
    delete $3;

    CHECK_FOR_ERROR
  }
  | VOID '(' ArgTypeListI ')' OptFuncAttrs {
    // Allow but ignore attributes on function types; this permits auto-upgrade.
    // FIXME: remove in LLVM 3.0.
    std::vector<const Type*> Params;
    TypeWithAttrsList::iterator I = $3->begin(), E = $3->end();
    for ( ; I != E; ++I ) {
      const Type* Ty = I->Ty->get();
      Params.push_back(Ty);
    }

    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    for (unsigned i = 0; i != Params.size(); ++i)
      if (!(Params[i]->isFirstClassType() || isa<OpaqueType>(Params[i])))
        GEN_ERROR("Function arguments must be value types!");

    CHECK_FOR_ERROR

    FunctionType *FT = FunctionType::get($1, Params, isVarArg);
    $$ = new PATypeHolder(HandleUpRefs(FT));

    // Delete the argument list
    for (I = $3->begin() ; I != E; ++I ) {
      delete I->Ty;
    }
    delete $3;

    CHECK_FOR_ERROR
  }

  | '[' EUINT64VAL 'x' Types ']' {          // Sized array type?
    $$ = new PATypeHolder(HandleUpRefs(ArrayType::get(*$4, $2)));
    delete $4;
    CHECK_FOR_ERROR
  }
  | '<' EUINT64VAL 'x' Types '>' {          // Vector type?
     const llvm::Type* ElemTy = $4->get();
     if ((unsigned)$2 != $2)
        GEN_ERROR("Unsigned result not equal to signed result");
     if (!ElemTy->isFloatingPoint() && !ElemTy->isInteger())
        GEN_ERROR("Element type of a VectorType must be primitive");
     $$ = new PATypeHolder(HandleUpRefs(VectorType::get(*$4, (unsigned)$2)));
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
  | '<' '{' TypeListI '}' '>' {
    std::vector<const Type*> Elements;
    for (std::list<llvm::PATypeHolder>::iterator I = $3->begin(),
           E = $3->end(); I != E; ++I)
      Elements.push_back(*I);

    $$ = new PATypeHolder(HandleUpRefs(StructType::get(Elements, true)));
    delete $3;
    CHECK_FOR_ERROR
  }
  | '<' '{' '}' '>' {                         // Empty structure type?
    $$ = new PATypeHolder(StructType::get(std::vector<const Type*>(), true));
    CHECK_FOR_ERROR
  }
  ;

ArgType
  : Types OptAttributes {
    // Allow but ignore attributes on function types; this permits auto-upgrade.
    // FIXME: remove in LLVM 3.0.
    $$.Ty = $1;
    $$.Attrs = Attribute::None;
  }
  ;

ResultTypes
  : Types {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    if (!(*$1)->isFirstClassType() && !isa<StructType>($1->get()))
      GEN_ERROR("LLVM functions cannot return aggregate types");
    $$ = $1;
  }
  | VOID {
    $$ = new PATypeHolder(Type::VoidTy);
  }
  ;

ArgTypeList : ArgType {
    $$ = new TypeWithAttrsList();
    $$->push_back($1);
    CHECK_FOR_ERROR
  }
  | ArgTypeList ',' ArgType {
    ($$=$1)->push_back($3);
    CHECK_FOR_ERROR
  }
  ;

ArgTypeListI
  : ArgTypeList
  | ArgTypeList ',' DOTDOTDOT {
    $$=$1;
    TypeWithAttrs TWA; TWA.Attrs = Attribute::None;
    TWA.Ty = new PATypeHolder(Type::VoidTy);
    $$->push_back(TWA);
    CHECK_FOR_ERROR
  }
  | DOTDOTDOT {
    $$ = new TypeWithAttrsList;
    TypeWithAttrs TWA; TWA.Attrs = Attribute::None;
    TWA.Ty = new PATypeHolder(Type::VoidTy);
    $$->push_back(TWA);
    CHECK_FOR_ERROR
  }
  | /*empty*/ {
    $$ = new TypeWithAttrsList();
    CHECK_FOR_ERROR
  };

// TypeList - Used for struct declarations and as a basis for function type
// declaration type lists
//
TypeListI : Types {
    $$ = new std::list<PATypeHolder>();
    $$->push_back(*$1);
    delete $1;
    CHECK_FOR_ERROR
  }
  | TypeListI ',' Types {
    ($$=$1)->push_back(*$3);
    delete $3;
    CHECK_FOR_ERROR
  };

// ConstVal - The various declarations that go into the constant pool.  This
// production is used ONLY to represent constants that show up AFTER a 'const',
// 'constant' or 'global' token at global scope.  Constants that can be inlined
// into other expressions (such as integers and constexprs) are handled by the
// ResolvedVal, ValueRef and ConstValueRef productions.
//
ConstVal: Types '[' ConstVector ']' { // Nonempty unsized arr
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    const ArrayType *ATy = dyn_cast<ArrayType>($1->get());
    if (ATy == 0)
      GEN_ERROR("Cannot make array constant with type: '" +
                     (*$1)->getDescription() + "'");
    const Type *ETy = ATy->getElementType();
    uint64_t NumElements = ATy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != uint64_t(-1) && NumElements != $3->size())
      GEN_ERROR("Type mismatch: constant sized array initialized with " +
                     utostr($3->size()) +  " arguments, but has size of " +
                     utostr(NumElements) + "");

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
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    const ArrayType *ATy = dyn_cast<ArrayType>($1->get());
    if (ATy == 0)
      GEN_ERROR("Cannot make array constant with type: '" +
                     (*$1)->getDescription() + "'");

    uint64_t NumElements = ATy->getNumElements();
    if (NumElements != uint64_t(-1) && NumElements != 0)
      GEN_ERROR("Type mismatch: constant sized array initialized with 0"
                     " arguments, but has size of " + utostr(NumElements) +"");
    $$ = ConstantArray::get(ATy, std::vector<Constant*>());
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types 'c' STRINGCONSTANT {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    const ArrayType *ATy = dyn_cast<ArrayType>($1->get());
    if (ATy == 0)
      GEN_ERROR("Cannot make array constant with type: '" +
                     (*$1)->getDescription() + "'");

    uint64_t NumElements = ATy->getNumElements();
    const Type *ETy = ATy->getElementType();
    if (NumElements != uint64_t(-1) && NumElements != $3->length())
      GEN_ERROR("Can't build string constant of size " +
                     utostr($3->length()) +
                     " when array has size " + utostr(NumElements) + "");
    std::vector<Constant*> Vals;
    if (ETy == Type::Int8Ty) {
      for (uint64_t i = 0; i < $3->length(); ++i)
        Vals.push_back(ConstantInt::get(ETy, (*$3)[i]));
    } else {
      delete $3;
      GEN_ERROR("Cannot build string arrays of non byte sized elements");
    }
    delete $3;
    $$ = ConstantArray::get(ATy, Vals);
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types '<' ConstVector '>' { // Nonempty unsized arr
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    const VectorType *PTy = dyn_cast<VectorType>($1->get());
    if (PTy == 0)
      GEN_ERROR("Cannot make packed constant with type: '" +
                     (*$1)->getDescription() + "'");
    const Type *ETy = PTy->getElementType();
    unsigned NumElements = PTy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != unsigned(-1) && NumElements != (unsigned)$3->size())
      GEN_ERROR("Type mismatch: constant sized packed initialized with " +
                     utostr($3->size()) +  " arguments, but has size of " +
                     utostr(NumElements) + "");

    // Verify all elements are correct type!
    for (unsigned i = 0; i < $3->size(); i++) {
      if (ETy != (*$3)[i]->getType())
        GEN_ERROR("Element #" + utostr(i) + " is not of type '" +
           ETy->getDescription() +"' as required!\nIt is of type '"+
           (*$3)[i]->getType()->getDescription() + "'.");
    }

    $$ = ConstantVector::get(PTy, *$3);
    delete $1; delete $3;
    CHECK_FOR_ERROR
  }
  | Types '{' ConstVector '}' {
    const StructType *STy = dyn_cast<StructType>($1->get());
    if (STy == 0)
      GEN_ERROR("Cannot make struct constant with type: '" +
                     (*$1)->getDescription() + "'");

    if ($3->size() != STy->getNumContainedTypes())
      GEN_ERROR("Illegal number of initializers for structure type");

    // Check to ensure that constants are compatible with the type initializer!
    for (unsigned i = 0, e = $3->size(); i != e; ++i)
      if ((*$3)[i]->getType() != STy->getElementType(i))
        GEN_ERROR("Expected type '" +
                       STy->getElementType(i)->getDescription() +
                       "' for element #" + utostr(i) +
                       " of structure initializer");

    // Check to ensure that Type is not packed
    if (STy->isPacked())
      GEN_ERROR("Unpacked Initializer to vector type '" +
                STy->getDescription() + "'");

    $$ = ConstantStruct::get(STy, *$3);
    delete $1; delete $3;
    CHECK_FOR_ERROR
  }
  | Types '{' '}' {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    const StructType *STy = dyn_cast<StructType>($1->get());
    if (STy == 0)
      GEN_ERROR("Cannot make struct constant with type: '" +
                     (*$1)->getDescription() + "'");

    if (STy->getNumContainedTypes() != 0)
      GEN_ERROR("Illegal number of initializers for structure type");

    // Check to ensure that Type is not packed
    if (STy->isPacked())
      GEN_ERROR("Unpacked Initializer to vector type '" +
                STy->getDescription() + "'");

    $$ = ConstantStruct::get(STy, std::vector<Constant*>());
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types '<' '{' ConstVector '}' '>' {
    const StructType *STy = dyn_cast<StructType>($1->get());
    if (STy == 0)
      GEN_ERROR("Cannot make struct constant with type: '" +
                     (*$1)->getDescription() + "'");

    if ($4->size() != STy->getNumContainedTypes())
      GEN_ERROR("Illegal number of initializers for structure type");

    // Check to ensure that constants are compatible with the type initializer!
    for (unsigned i = 0, e = $4->size(); i != e; ++i)
      if ((*$4)[i]->getType() != STy->getElementType(i))
        GEN_ERROR("Expected type '" +
                       STy->getElementType(i)->getDescription() +
                       "' for element #" + utostr(i) +
                       " of structure initializer");

    // Check to ensure that Type is packed
    if (!STy->isPacked())
      GEN_ERROR("Vector initializer to non-vector type '" +
                STy->getDescription() + "'");

    $$ = ConstantStruct::get(STy, *$4);
    delete $1; delete $4;
    CHECK_FOR_ERROR
  }
  | Types '<' '{' '}' '>' {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    const StructType *STy = dyn_cast<StructType>($1->get());
    if (STy == 0)
      GEN_ERROR("Cannot make struct constant with type: '" +
                     (*$1)->getDescription() + "'");

    if (STy->getNumContainedTypes() != 0)
      GEN_ERROR("Illegal number of initializers for structure type");

    // Check to ensure that Type is packed
    if (!STy->isPacked())
      GEN_ERROR("Vector initializer to non-vector type '" +
                STy->getDescription() + "'");

    $$ = ConstantStruct::get(STy, std::vector<Constant*>());
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types NULL_TOK {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    const PointerType *PTy = dyn_cast<PointerType>($1->get());
    if (PTy == 0)
      GEN_ERROR("Cannot make null pointer constant with type: '" +
                     (*$1)->getDescription() + "'");

    $$ = ConstantPointerNull::get(PTy);
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types UNDEF {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    $$ = UndefValue::get($1->get());
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types SymbolicValueRef {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    const PointerType *Ty = dyn_cast<PointerType>($1->get());
    if (Ty == 0)
      GEN_ERROR("Global const reference must be a pointer type " + (*$1)->getDescription());

    // ConstExprs can exist in the body of a function, thus creating
    // GlobalValues whenever they refer to a variable.  Because we are in
    // the context of a function, getExistingVal will search the functions
    // symbol table instead of the module symbol table for the global symbol,
    // which throws things all off.  To get around this, we just tell
    // getExistingVal that we are at global scope here.
    //
    Function *SavedCurFn = CurFun.CurrentFunction;
    CurFun.CurrentFunction = 0;

    Value *V = getExistingVal(Ty, $2);
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
        if ($2.Type == ValID::GlobalName)
          Name = $2.getName();
        else if ($2.Type != ValID::GlobalID)
          GEN_ERROR("Invalid reference to global");

        // Create the forward referenced global.
        GlobalValue *GV;
        if (const FunctionType *FTy =
                 dyn_cast<FunctionType>(PT->getElementType())) {
          GV = Function::Create(FTy, GlobalValue::ExternalWeakLinkage, Name,
                                CurModule.CurrentModule);
        } else {
          GV = new GlobalVariable(PT->getElementType(), false,
                                  GlobalValue::ExternalWeakLinkage, 0,
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
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    if ($1->get() != $2->getType())
      GEN_ERROR("Mismatched types for constant expression: " +
        (*$1)->getDescription() + " and " + $2->getType()->getDescription());
    $$ = $2;
    delete $1;
    CHECK_FOR_ERROR
  }
  | Types ZEROINITIALIZER {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    const Type *Ty = $1->get();
    if (isa<FunctionType>(Ty) || Ty == Type::LabelTy || isa<OpaqueType>(Ty))
      GEN_ERROR("Cannot create a null initialized value of this type");
    $$ = Constant::getNullValue(Ty);
    delete $1;
    CHECK_FOR_ERROR
  }
  | IntType ESINT64VAL {      // integral constants
    if (!ConstantInt::isValueValidForType($1, $2))
      GEN_ERROR("Constant value doesn't fit in type");
    $$ = ConstantInt::get($1, $2, true);
    CHECK_FOR_ERROR
  }
  | IntType ESAPINTVAL {      // arbitrary precision integer constants
    uint32_t BitWidth = cast<IntegerType>($1)->getBitWidth();
    if ($2->getBitWidth() > BitWidth) {
      GEN_ERROR("Constant value does not fit in type");
    }
    $2->sextOrTrunc(BitWidth);
    $$ = ConstantInt::get(*$2);
    delete $2;
    CHECK_FOR_ERROR
  }
  | IntType EUINT64VAL {      // integral constants
    if (!ConstantInt::isValueValidForType($1, $2))
      GEN_ERROR("Constant value doesn't fit in type");
    $$ = ConstantInt::get($1, $2, false);
    CHECK_FOR_ERROR
  }
  | IntType EUAPINTVAL {      // arbitrary precision integer constants
    uint32_t BitWidth = cast<IntegerType>($1)->getBitWidth();
    if ($2->getBitWidth() > BitWidth) {
      GEN_ERROR("Constant value does not fit in type");
    }
    $2->zextOrTrunc(BitWidth);
    $$ = ConstantInt::get(*$2);
    delete $2;
    CHECK_FOR_ERROR
  }
  | INTTYPE TRUETOK {                      // Boolean constants
    if (cast<IntegerType>($1)->getBitWidth() != 1)
      GEN_ERROR("Constant true must have type i1");
    $$ = ConstantInt::getTrue();
    CHECK_FOR_ERROR
  }
  | INTTYPE FALSETOK {                     // Boolean constants
    if (cast<IntegerType>($1)->getBitWidth() != 1)
      GEN_ERROR("Constant false must have type i1");
    $$ = ConstantInt::getFalse();
    CHECK_FOR_ERROR
  }
  | FPType FPVAL {                   // Floating point constants
    if (!ConstantFP::isValueValidForType($1, *$2))
      GEN_ERROR("Floating point constant invalid for type");
    // Lexer has no type info, so builds all float and double FP constants
    // as double.  Fix this here.  Long double is done right.
    if (&$2->getSemantics()==&APFloat::IEEEdouble && $1==Type::FloatTy)
      $2->convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven);
    $$ = ConstantFP::get(*$2);
    delete $2;
    CHECK_FOR_ERROR
  };


ConstExpr: CastOps '(' ConstVal TO Types ')' {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$5)->getDescription());
    Constant *Val = $3;
    const Type *DestTy = $5->get();
    if (!CastInst::castIsValid($1, $3, DestTy))
      GEN_ERROR("invalid cast opcode for cast from '" +
                Val->getType()->getDescription() + "' to '" +
                DestTy->getDescription() + "'");
    $$ = ConstantExpr::getCast($1, $3, DestTy);
    delete $5;
  }
  | GETELEMENTPTR '(' ConstVal IndexList ')' {
    if (!isa<PointerType>($3->getType()))
      GEN_ERROR("GetElementPtr requires a pointer operand");

    const Type *IdxTy =
      GetElementPtrInst::getIndexedType($3->getType(), $4->begin(), $4->end());
    if (!IdxTy)
      GEN_ERROR("Index list invalid for constant getelementptr");

    SmallVector<Constant*, 8> IdxVec;
    for (unsigned i = 0, e = $4->size(); i != e; ++i)
      if (Constant *C = dyn_cast<Constant>((*$4)[i]))
        IdxVec.push_back(C);
      else
        GEN_ERROR("Indices to constant getelementptr must be constants");

    delete $4;

    $$ = ConstantExpr::getGetElementPtr($3, &IdxVec[0], IdxVec.size());
    CHECK_FOR_ERROR
  }
  | SELECT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    if ($3->getType() != Type::Int1Ty)
      GEN_ERROR("Select condition must be of boolean type");
    if ($5->getType() != $7->getType())
      GEN_ERROR("Select operand types must match");
    $$ = ConstantExpr::getSelect($3, $5, $7);
    CHECK_FOR_ERROR
  }
  | ArithmeticOps '(' ConstVal ',' ConstVal ')' {
    if ($3->getType() != $5->getType())
      GEN_ERROR("Binary operator types must match");
    CHECK_FOR_ERROR;
    $$ = ConstantExpr::get($1, $3, $5);
  }
  | LogicalOps '(' ConstVal ',' ConstVal ')' {
    if ($3->getType() != $5->getType())
      GEN_ERROR("Logical operator types must match");
    if (!$3->getType()->isInteger()) {
      if (!isa<VectorType>($3->getType()) ||
          !cast<VectorType>($3->getType())->getElementType()->isInteger())
        GEN_ERROR("Logical operator requires integral operands");
    }
    $$ = ConstantExpr::get($1, $3, $5);
    CHECK_FOR_ERROR
  }
  | ICMP IPredicates '(' ConstVal ',' ConstVal ')' {
    if ($4->getType() != $6->getType())
      GEN_ERROR("icmp operand types must match");
    $$ = ConstantExpr::getICmp($2, $4, $6);
  }
  | FCMP FPredicates '(' ConstVal ',' ConstVal ')' {
    if ($4->getType() != $6->getType())
      GEN_ERROR("fcmp operand types must match");
    $$ = ConstantExpr::getFCmp($2, $4, $6);
  }
  | VICMP IPredicates '(' ConstVal ',' ConstVal ')' {
    if ($4->getType() != $6->getType())
      GEN_ERROR("vicmp operand types must match");
    $$ = ConstantExpr::getVICmp($2, $4, $6);
  }
  | VFCMP FPredicates '(' ConstVal ',' ConstVal ')' {
    if ($4->getType() != $6->getType())
      GEN_ERROR("vfcmp operand types must match");
    $$ = ConstantExpr::getVFCmp($2, $4, $6);
  }
  | EXTRACTELEMENT '(' ConstVal ',' ConstVal ')' {
    if (!ExtractElementInst::isValidOperands($3, $5))
      GEN_ERROR("Invalid extractelement operands");
    $$ = ConstantExpr::getExtractElement($3, $5);
    CHECK_FOR_ERROR
  }
  | INSERTELEMENT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    if (!InsertElementInst::isValidOperands($3, $5, $7))
      GEN_ERROR("Invalid insertelement operands");
    $$ = ConstantExpr::getInsertElement($3, $5, $7);
    CHECK_FOR_ERROR
  }
  | SHUFFLEVECTOR '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    if (!ShuffleVectorInst::isValidOperands($3, $5, $7))
      GEN_ERROR("Invalid shufflevector operands");
    $$ = ConstantExpr::getShuffleVector($3, $5, $7);
    CHECK_FOR_ERROR
  }
  | EXTRACTVALUE '(' ConstVal ConstantIndexList ')' {
    if (!isa<StructType>($3->getType()) && !isa<ArrayType>($3->getType()))
      GEN_ERROR("ExtractValue requires an aggregate operand");

    $$ = ConstantExpr::getExtractValue($3, &(*$4)[0], $4->size());
    delete $4;
    CHECK_FOR_ERROR
  }
  | INSERTVALUE '(' ConstVal ',' ConstVal ConstantIndexList ')' {
    if (!isa<StructType>($3->getType()) && !isa<ArrayType>($3->getType()))
      GEN_ERROR("InsertValue requires an aggregate operand");

    $$ = ConstantExpr::getInsertValue($3, $5, &(*$6)[0], $6->size());
    delete $6;
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

// ThreadLocal
ThreadLocal : THREAD_LOCAL { $$ = true; } | { $$ = false; };

// AliaseeRef - Match either GlobalValue or bitcast to GlobalValue.
AliaseeRef : ResultTypes SymbolicValueRef {
    const Type* VTy = $1->get();
    Value *V = getVal(VTy, $2);
    CHECK_FOR_ERROR
    GlobalValue* Aliasee = dyn_cast<GlobalValue>(V);
    if (!Aliasee)
      GEN_ERROR("Aliases can be created only to global values");

    $$ = Aliasee;
    CHECK_FOR_ERROR
    delete $1;
   }
   | BITCAST '(' AliaseeRef TO Types ')' {
    Constant *Val = $3;
    const Type *DestTy = $5->get();
    if (!CastInst::castIsValid($1, $3, DestTy))
      GEN_ERROR("invalid cast opcode for cast from '" +
                Val->getType()->getDescription() + "' to '" +
                DestTy->getDescription() + "'");

    $$ = ConstantExpr::getCast($1, $3, DestTy);
    CHECK_FOR_ERROR
    delete $5;
   };

//===----------------------------------------------------------------------===//
//                             Rules to match Modules
//===----------------------------------------------------------------------===//

// Module rule: Capture the result of parsing the whole file into a result
// variable...
//
Module
  : DefinitionList {
    $$ = ParserResult = CurModule.CurrentModule;
    CurModule.ModuleDone();
    CHECK_FOR_ERROR;
  }
  | /*empty*/ {
    $$ = ParserResult = CurModule.CurrentModule;
    CurModule.ModuleDone();
    CHECK_FOR_ERROR;
  }
  ;

DefinitionList
  : Definition
  | DefinitionList Definition
  ;

Definition
  : DEFINE { CurFun.isDeclare = false; } Function {
    CurFun.FunctionDone();
    CHECK_FOR_ERROR
  }
  | DECLARE { CurFun.isDeclare = true; } FunctionProto {
    CHECK_FOR_ERROR
  }
  | MODULE ASM_TOK AsmBlock {
    CHECK_FOR_ERROR
  }
  | OptLocalAssign TYPE Types {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$3)->getDescription());
    // Eagerly resolve types.  This is not an optimization, this is a
    // requirement that is due to the fact that we could have this:
    //
    // %list = type { %list * }
    // %list = type { %list * }    ; repeated type decl
    //
    // If types are not resolved eagerly, then the two types will not be
    // determined to be the same type!
    //
    ResolveTypeTo($1, *$3);

    if (!setTypeName(*$3, $1) && !$1) {
      CHECK_FOR_ERROR
      // If this is a named type that is not a redefinition, add it to the slot
      // table.
      CurModule.Types.push_back(*$3);
    }

    delete $3;
    CHECK_FOR_ERROR
  }
  | OptLocalAssign TYPE VOID {
    ResolveTypeTo($1, $3);

    if (!setTypeName($3, $1) && !$1) {
      CHECK_FOR_ERROR
      // If this is a named type that is not a redefinition, add it to the slot
      // table.
      CurModule.Types.push_back($3);
    }
    CHECK_FOR_ERROR
  }
  | OptGlobalAssign GVVisibilityStyle ThreadLocal GlobalType ConstVal
    OptAddrSpace {
    /* "Externally Visible" Linkage */
    if ($5 == 0)
      GEN_ERROR("Global value initializer is not a constant");
    CurGV = ParseGlobalVariable($1, GlobalValue::ExternalLinkage,
                                $2, $4, $5->getType(), $5, $3, $6);
    CHECK_FOR_ERROR
  } GlobalVarAttributes {
    CurGV = 0;
  }
  | OptGlobalAssign GVInternalLinkage GVVisibilityStyle ThreadLocal GlobalType
    ConstVal OptAddrSpace {
    if ($6 == 0)
      GEN_ERROR("Global value initializer is not a constant");
    CurGV = ParseGlobalVariable($1, $2, $3, $5, $6->getType(), $6, $4, $7);
    CHECK_FOR_ERROR
  } GlobalVarAttributes {
    CurGV = 0;
  }
  | OptGlobalAssign GVExternalLinkage GVVisibilityStyle ThreadLocal GlobalType
    Types OptAddrSpace {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$6)->getDescription());
    CurGV = ParseGlobalVariable($1, $2, $3, $5, *$6, 0, $4, $7);
    CHECK_FOR_ERROR
    delete $6;
  } GlobalVarAttributes {
    CurGV = 0;
    CHECK_FOR_ERROR
  }
  | OptGlobalAssign GVVisibilityStyle ALIAS AliasLinkage AliaseeRef {
    std::string Name;
    if ($1) {
      Name = *$1;
      delete $1;
    }
    if (Name.empty())
      GEN_ERROR("Alias name cannot be empty");

    Constant* Aliasee = $5;
    if (Aliasee == 0)
      GEN_ERROR(std::string("Invalid aliasee for alias: ") + Name);

    GlobalAlias* GA = new GlobalAlias(Aliasee->getType(), $4, Name, Aliasee,
                                      CurModule.CurrentModule);
    GA->setVisibility($2);
    InsertValue(GA, CurModule.Values);


    // If there was a forward reference of this alias, resolve it now.

    ValID ID;
    if (!Name.empty())
      ID = ValID::createGlobalName(Name);
    else
      ID = ValID::createGlobalID(CurModule.Values.size()-1);

    if (GlobalValue *FWGV =
          CurModule.GetForwardRefForGlobal(GA->getType(), ID)) {
      // Replace uses of the fwdref with the actual alias.
      FWGV->replaceAllUsesWith(GA);
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(FWGV))
        GV->eraseFromParent();
      else
        cast<Function>(FWGV)->eraseFromParent();
    }
    ID.destroy();

    CHECK_FOR_ERROR
  }
  | TARGET TargetDefinition {
    CHECK_FOR_ERROR
  }
  | DEPLIBS '=' LibrariesDefinition {
    CHECK_FOR_ERROR
  }
  ;


AsmBlock : STRINGCONSTANT {
  const std::string &AsmSoFar = CurModule.CurrentModule->getModuleInlineAsm();
  if (AsmSoFar.empty())
    CurModule.CurrentModule->setModuleInlineAsm(*$1);
  else
    CurModule.CurrentModule->setModuleInlineAsm(AsmSoFar+"\n"+*$1);
  delete $1;
  CHECK_FOR_ERROR
};

TargetDefinition : TRIPLE '=' STRINGCONSTANT {
    CurModule.CurrentModule->setTargetTriple(*$3);
    delete $3;
  }
  | DATALAYOUT '=' STRINGCONSTANT {
    CurModule.CurrentModule->setDataLayout(*$3);
    delete $3;
  };

LibrariesDefinition : '[' LibList ']';

LibList : LibList ',' STRINGCONSTANT {
          CurModule.CurrentModule->addLibrary(*$3);
          delete $3;
          CHECK_FOR_ERROR
        }
        | STRINGCONSTANT {
          CurModule.CurrentModule->addLibrary(*$1);
          delete $1;
          CHECK_FOR_ERROR
        }
        | /* empty: end of list */ {
          CHECK_FOR_ERROR
        }
        ;

//===----------------------------------------------------------------------===//
//                       Rules to match Function Headers
//===----------------------------------------------------------------------===//

ArgListH : ArgListH ',' Types OptAttributes OptLocalName {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$3)->getDescription());
    if (!(*$3)->isFirstClassType())
      GEN_ERROR("Argument types must be first-class");
    ArgListEntry E; E.Attrs = $4; E.Ty = $3; E.Name = $5;
    $$ = $1;
    $1->push_back(E);
    CHECK_FOR_ERROR
  }
  | Types OptAttributes OptLocalName {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    if (!(*$1)->isFirstClassType())
      GEN_ERROR("Argument types must be first-class");
    ArgListEntry E; E.Attrs = $2; E.Ty = $1; E.Name = $3;
    $$ = new ArgListType;
    $$->push_back(E);
    CHECK_FOR_ERROR
  };

ArgList : ArgListH {
    $$ = $1;
    CHECK_FOR_ERROR
  }
  | ArgListH ',' DOTDOTDOT {
    $$ = $1;
    struct ArgListEntry E;
    E.Ty = new PATypeHolder(Type::VoidTy);
    E.Name = 0;
    E.Attrs = Attribute::None;
    $$->push_back(E);
    CHECK_FOR_ERROR
  }
  | DOTDOTDOT {
    $$ = new ArgListType;
    struct ArgListEntry E;
    E.Ty = new PATypeHolder(Type::VoidTy);
    E.Name = 0;
    E.Attrs = Attribute::None;
    $$->push_back(E);
    CHECK_FOR_ERROR
  }
  | /* empty */ {
    $$ = 0;
    CHECK_FOR_ERROR
  };

FunctionHeaderH : OptCallingConv OptRetAttrs ResultTypes GlobalName '(' ArgList ')'
                  OptFuncAttrs OptSection OptAlign OptGC {
  std::string FunctionName(*$4);
  delete $4;  // Free strdup'd memory!

  // Check the function result for abstractness if this is a define. We should
  // have no abstract types at this point
  if (!CurFun.isDeclare && CurModule.TypeIsUnresolved($3))
    GEN_ERROR("Reference to abstract result: "+ $3->get()->getDescription());

  if (!FunctionType::isValidReturnType(*$3))
    GEN_ERROR("Invalid result type for LLVM function");

  std::vector<const Type*> ParamTypeList;
  SmallVector<AttributeWithIndex, 8> Attrs;
  //FIXME : In 3.0, stop accepting zext, sext and inreg as optional function 
  //attributes.
  Attributes RetAttrs = $2;
  if ($8 != Attribute::None) {
    if ($8 & Attribute::ZExt) {
      RetAttrs = RetAttrs | Attribute::ZExt;
      $8 = $8 ^ Attribute::ZExt;
    }
    if ($8 & Attribute::SExt) {
      RetAttrs = RetAttrs | Attribute::SExt;
      $8 = $8 ^ Attribute::SExt;
    }
    if ($8 & Attribute::InReg) {
      RetAttrs = RetAttrs | Attribute::InReg;
      $8 = $8 ^ Attribute::InReg;
    }
  }
  if (RetAttrs != Attribute::None)
    Attrs.push_back(AttributeWithIndex::get(0, RetAttrs));
  if ($6) {   // If there are arguments...
    unsigned index = 1;
    for (ArgListType::iterator I = $6->begin(); I != $6->end(); ++I, ++index) {
      const Type* Ty = I->Ty->get();
      if (!CurFun.isDeclare && CurModule.TypeIsUnresolved(I->Ty))
        GEN_ERROR("Reference to abstract argument: " + Ty->getDescription());
      ParamTypeList.push_back(Ty);
      if (Ty != Type::VoidTy && I->Attrs != Attribute::None)
        Attrs.push_back(AttributeWithIndex::get(index, I->Attrs));
    }
  }
  if ($8 != Attribute::None)
    Attrs.push_back(AttributeWithIndex::get(~0, $8));

  bool isVarArg = ParamTypeList.size() && ParamTypeList.back() == Type::VoidTy;
  if (isVarArg) ParamTypeList.pop_back();

  AttrListPtr PAL;
  if (!Attrs.empty())
    PAL = AttrListPtr::get(Attrs.begin(), Attrs.end());

  FunctionType *FT = FunctionType::get(*$3, ParamTypeList, isVarArg);
  const PointerType *PFT = PointerType::getUnqual(FT);
  delete $3;

  ValID ID;
  if (!FunctionName.empty()) {
    ID = ValID::createGlobalName((char*)FunctionName.c_str());
  } else {
    ID = ValID::createGlobalID(CurModule.Values.size());
  }

  Function *Fn = 0;
  // See if this function was forward referenced.  If so, recycle the object.
  if (GlobalValue *FWRef = CurModule.GetForwardRefForGlobal(PFT, ID)) {
    // Move the function to the end of the list, from whereever it was
    // previously inserted.
    Fn = cast<Function>(FWRef);
    assert(Fn->getAttributes().isEmpty() &&
           "Forward reference has parameter attributes!");
    CurModule.CurrentModule->getFunctionList().remove(Fn);
    CurModule.CurrentModule->getFunctionList().push_back(Fn);
  } else if (!FunctionName.empty() &&     // Merge with an earlier prototype?
             (Fn = CurModule.CurrentModule->getFunction(FunctionName))) {
    if (Fn->getFunctionType() != FT ) {
      // The existing function doesn't have the same type. This is an overload
      // error.
      GEN_ERROR("Overload of function '" + FunctionName + "' not permitted.");
    } else if (Fn->getAttributes() != PAL) {
      // The existing function doesn't have the same parameter attributes.
      // This is an overload error.
      GEN_ERROR("Overload of function '" + FunctionName + "' not permitted.");
    } else if (!CurFun.isDeclare && !Fn->isDeclaration()) {
      // Neither the existing or the current function is a declaration and they
      // have the same name and same type. Clearly this is a redefinition.
      GEN_ERROR("Redefinition of function '" + FunctionName + "'");
    } else if (Fn->isDeclaration()) {
      // Make sure to strip off any argument names so we can't get conflicts.
      for (Function::arg_iterator AI = Fn->arg_begin(), AE = Fn->arg_end();
           AI != AE; ++AI)
        AI->setName("");
    }
  } else  {  // Not already defined?
    Fn = Function::Create(FT, GlobalValue::ExternalWeakLinkage, FunctionName,
                          CurModule.CurrentModule);
    InsertValue(Fn, CurModule.Values);
  }

  ID.destroy();
  CurFun.FunctionStart(Fn);

  if (CurFun.isDeclare) {
    // If we have declaration, always overwrite linkage.  This will allow us to
    // correctly handle cases, when pointer to function is passed as argument to
    // another function.
    Fn->setLinkage(CurFun.Linkage);
    Fn->setVisibility(CurFun.Visibility);
  }
  Fn->setCallingConv($1);
  Fn->setAttributes(PAL);
  Fn->setAlignment($10);
  if ($9) {
    Fn->setSection(*$9);
    delete $9;
  }
  if ($11) {
    Fn->setGC($11->c_str());
    delete $11;
  }

  // Add all of the arguments we parsed to the function...
  if ($6) {                     // Is null if empty...
    if (isVarArg) {  // Nuke the last entry
      assert($6->back().Ty->get() == Type::VoidTy && $6->back().Name == 0 &&
             "Not a varargs marker!");
      delete $6->back().Ty;
      $6->pop_back();  // Delete the last entry
    }
    Function::arg_iterator ArgIt = Fn->arg_begin();
    Function::arg_iterator ArgEnd = Fn->arg_end();
    unsigned Idx = 1;
    for (ArgListType::iterator I = $6->begin();
         I != $6->end() && ArgIt != ArgEnd; ++I, ++ArgIt) {
      delete I->Ty;                          // Delete the typeholder...
      setValueName(ArgIt, I->Name);       // Insert arg into symtab...
      CHECK_FOR_ERROR
      InsertValue(ArgIt);
      Idx++;
    }

    delete $6;                     // We're now done with the argument list
  }
  CHECK_FOR_ERROR
};

BEGIN : BEGINTOK | '{';                // Allow BEGIN or '{' to start a function

FunctionHeader : FunctionDefineLinkage GVVisibilityStyle FunctionHeaderH BEGIN {
  $$ = CurFun.CurrentFunction;

  // Make sure that we keep track of the linkage type even if there was a
  // previous "declare".
  $$->setLinkage($1);
  $$->setVisibility($2);
};

END : ENDTOK | '}';                    // Allow end of '}' to end a function

Function : BasicBlockList END {
  $$ = $1;
  CHECK_FOR_ERROR
};

FunctionProto : FunctionDeclareLinkage GVVisibilityStyle FunctionHeaderH {
    CurFun.CurrentFunction->setLinkage($1);
    CurFun.CurrentFunction->setVisibility($2);
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
  | ESAPINTVAL {      // arbitrary precision integer constants
    $$ = ValID::create(*$1, true);
    delete $1;
    CHECK_FOR_ERROR
  }
  | EUAPINTVAL {      // arbitrary precision integer constants
    $$ = ValID::create(*$1, false);
    delete $1;
    CHECK_FOR_ERROR
  }
  | FPVAL {                     // Perhaps it's an FP constant?
    $$ = ValID::create($1);
    CHECK_FOR_ERROR
  }
  | TRUETOK {
    $$ = ValID::create(ConstantInt::getTrue());
    CHECK_FOR_ERROR
  }
  | FALSETOK {
    $$ = ValID::create(ConstantInt::getFalse());
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
    unsigned NumElements = $2->size();

    if (!ETy->isInteger() && !ETy->isFloatingPoint())
      GEN_ERROR("Invalid vector element type: " + ETy->getDescription());

    VectorType* pt = VectorType::get(ETy, NumElements);
    PATypeHolder* PTy = new PATypeHolder(HandleUpRefs(pt));

    // Verify all elements are correct type!
    for (unsigned i = 0; i < $2->size(); i++) {
      if (ETy != (*$2)[i]->getType())
        GEN_ERROR("Element #" + utostr(i) + " is not of type '" +
                     ETy->getDescription() +"' as required!\nIt is of type '" +
                     (*$2)[i]->getType()->getDescription() + "'.");
    }

    $$ = ValID::create(ConstantVector::get(pt, *$2));
    delete PTy; delete $2;
    CHECK_FOR_ERROR
  }
  | '[' ConstVector ']' { // Nonempty unsized arr
    const Type *ETy = (*$2)[0]->getType();
    uint64_t NumElements = $2->size();

    if (!ETy->isFirstClassType())
      GEN_ERROR("Invalid array element type: " + ETy->getDescription());

    ArrayType *ATy = ArrayType::get(ETy, NumElements);
    PATypeHolder* PTy = new PATypeHolder(HandleUpRefs(ATy));

    // Verify all elements are correct type!
    for (unsigned i = 0; i < $2->size(); i++) {
      if (ETy != (*$2)[i]->getType())
        GEN_ERROR("Element #" + utostr(i) + " is not of type '" +
                       ETy->getDescription() +"' as required!\nIt is of type '"+
                       (*$2)[i]->getType()->getDescription() + "'.");
    }

    $$ = ValID::create(ConstantArray::get(ATy, *$2));
    delete PTy; delete $2;
    CHECK_FOR_ERROR
  }
  | '[' ']' {
    // Use undef instead of an array because it's inconvenient to determine
    // the element type at this point, there being no elements to examine.
    $$ = ValID::createUndef();
    CHECK_FOR_ERROR
  }
  | 'c' STRINGCONSTANT {
    uint64_t NumElements = $2->length();
    const Type *ETy = Type::Int8Ty;

    ArrayType *ATy = ArrayType::get(ETy, NumElements);

    std::vector<Constant*> Vals;
    for (unsigned i = 0; i < $2->length(); ++i)
      Vals.push_back(ConstantInt::get(ETy, (*$2)[i]));
    delete $2;
    $$ = ValID::create(ConstantArray::get(ATy, Vals));
    CHECK_FOR_ERROR
  }
  | '{' ConstVector '}' {
    std::vector<const Type*> Elements($2->size());
    for (unsigned i = 0, e = $2->size(); i != e; ++i)
      Elements[i] = (*$2)[i]->getType();

    const StructType *STy = StructType::get(Elements);
    PATypeHolder* PTy = new PATypeHolder(HandleUpRefs(STy));

    $$ = ValID::create(ConstantStruct::get(STy, *$2));
    delete PTy; delete $2;
    CHECK_FOR_ERROR
  }
  | '{' '}' {
    const StructType *STy = StructType::get(std::vector<const Type*>());
    $$ = ValID::create(ConstantStruct::get(STy, std::vector<Constant*>()));
    CHECK_FOR_ERROR
  }
  | '<' '{' ConstVector '}' '>' {
    std::vector<const Type*> Elements($3->size());
    for (unsigned i = 0, e = $3->size(); i != e; ++i)
      Elements[i] = (*$3)[i]->getType();

    const StructType *STy = StructType::get(Elements, /*isPacked=*/true);
    PATypeHolder* PTy = new PATypeHolder(HandleUpRefs(STy));

    $$ = ValID::create(ConstantStruct::get(STy, *$3));
    delete PTy; delete $3;
    CHECK_FOR_ERROR
  }
  | '<' '{' '}' '>' {
    const StructType *STy = StructType::get(std::vector<const Type*>(),
                                            /*isPacked=*/true);
    $$ = ValID::create(ConstantStruct::get(STy, std::vector<Constant*>()));
    CHECK_FOR_ERROR
  }
  | ConstExpr {
    $$ = ValID::create($1);
    CHECK_FOR_ERROR
  }
  | ASM_TOK OptSideEffect STRINGCONSTANT ',' STRINGCONSTANT {
    $$ = ValID::createInlineAsm(*$3, *$5, $2);
    delete $3;
    delete $5;
    CHECK_FOR_ERROR
  };

// SymbolicValueRef - Reference to one of two ways of symbolically refering to
// another value.
//
SymbolicValueRef : LOCALVAL_ID {  // Is it an integer reference...?
    $$ = ValID::createLocalID($1);
    CHECK_FOR_ERROR
  }
  | GLOBALVAL_ID {
    $$ = ValID::createGlobalID($1);
    CHECK_FOR_ERROR
  }
  | LocalName {                   // Is it a named reference...?
    $$ = ValID::createLocalName(*$1);
    delete $1;
    CHECK_FOR_ERROR
  }
  | GlobalName {                   // Is it a named reference...?
    $$ = ValID::createGlobalName(*$1);
    delete $1;
    CHECK_FOR_ERROR
  };

// ValueRef - A reference to a definition... either constant or symbolic
ValueRef : SymbolicValueRef | ConstValueRef;


// ResolvedVal - a <type> <value> pair.  This is used only in cases where the
// type immediately preceeds the value reference, and allows complex constant
// pool references (for things like: 'ret [2 x int] [ int 12, int 42]')
ResolvedVal : Types ValueRef {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    $$ = getVal(*$1, $2);
    delete $1;
    CHECK_FOR_ERROR
  }
  ;

ReturnedVal : ResolvedVal {
    $$ = new std::vector<Value *>();
    $$->push_back($1);
    CHECK_FOR_ERROR
  }
  | ReturnedVal ',' ResolvedVal {
    ($$=$1)->push_back($3);
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
BasicBlock : InstructionList OptLocalAssign BBTerminatorInst {
    setValueName($3, $2);
    CHECK_FOR_ERROR
    InsertValue($3);
    $1->getInstList().push_back($3);
    $$ = $1;
    CHECK_FOR_ERROR
  };

BasicBlock : InstructionList LocalNumber BBTerminatorInst {
  CHECK_FOR_ERROR
  int ValNum = InsertValue($3);
  if (ValNum != (int)$2)
    GEN_ERROR("Result value number %" + utostr($2) +
              " is incorrect, expected %" + utostr((unsigned)ValNum));

  $1->getInstList().push_back($3);
  $$ = $1;
  CHECK_FOR_ERROR
};


InstructionList : InstructionList Inst {
    if (CastInst *CI1 = dyn_cast<CastInst>($2))
      if (CastInst *CI2 = dyn_cast<CastInst>(CI1->getOperand(0)))
        if (CI2->getParent() == 0)
          $1->getInstList().push_back(CI2);
    $1->getInstList().push_back($2);
    $$ = $1;
    CHECK_FOR_ERROR
  }
  | /* empty */ {          // Empty space between instruction lists
    $$ = defineBBVal(ValID::createLocalID(CurFun.NextValNum));
    CHECK_FOR_ERROR
  }
  | LABELSTR {             // Labelled (named) basic block
    $$ = defineBBVal(ValID::createLocalName(*$1));
    delete $1;
    CHECK_FOR_ERROR

  };

BBTerminatorInst :
  RET ReturnedVal  { // Return with a result...
    ValueList &VL = *$2;
    assert(!VL.empty() && "Invalid ret operands!");
    const Type *ReturnType = CurFun.CurrentFunction->getReturnType();
    if (VL.size() > 1 ||
        (isa<StructType>(ReturnType) &&
         (VL.empty() || VL[0]->getType() != ReturnType))) {
      Value *RV = UndefValue::get(ReturnType);
      for (unsigned i = 0, e = VL.size(); i != e; ++i) {
        Instruction *I = InsertValueInst::Create(RV, VL[i], i, "mrv");
        ($<BasicBlockVal>-1)->getInstList().push_back(I);
        RV = I;
      }
      $$ = ReturnInst::Create(RV);
    } else {
      $$ = ReturnInst::Create(VL[0]);
    }
    delete $2;
    CHECK_FOR_ERROR
  }
  | RET VOID {                                    // Return with no result...
    $$ = ReturnInst::Create();
    CHECK_FOR_ERROR
  }
  | BR LABEL ValueRef {                           // Unconditional Branch...
    BasicBlock* tmpBB = getBBVal($3);
    CHECK_FOR_ERROR
    $$ = BranchInst::Create(tmpBB);
  }                                               // Conditional Branch...
  | BR INTTYPE ValueRef ',' LABEL ValueRef ',' LABEL ValueRef {
    if (cast<IntegerType>($2)->getBitWidth() != 1)
      GEN_ERROR("Branch condition must have type i1");
    BasicBlock* tmpBBA = getBBVal($6);
    CHECK_FOR_ERROR
    BasicBlock* tmpBBB = getBBVal($9);
    CHECK_FOR_ERROR
    Value* tmpVal = getVal(Type::Int1Ty, $3);
    CHECK_FOR_ERROR
    $$ = BranchInst::Create(tmpBBA, tmpBBB, tmpVal);
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' JumpTable ']' {
    Value* tmpVal = getVal($2, $3);
    CHECK_FOR_ERROR
    BasicBlock* tmpBB = getBBVal($6);
    CHECK_FOR_ERROR
    SwitchInst *S = SwitchInst::Create(tmpVal, tmpBB, $8->size());
    $$ = S;

    std::vector<std::pair<Constant*,BasicBlock*> >::iterator I = $8->begin(),
      E = $8->end();
    for (; I != E; ++I) {
      if (ConstantInt *CI = dyn_cast<ConstantInt>(I->first))
          S->addCase(CI, I->second);
      else
        GEN_ERROR("Switch case is constant, but not a simple integer");
    }
    delete $8;
    CHECK_FOR_ERROR
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' ']' {
    Value* tmpVal = getVal($2, $3);
    CHECK_FOR_ERROR
    BasicBlock* tmpBB = getBBVal($6);
    CHECK_FOR_ERROR
    SwitchInst *S = SwitchInst::Create(tmpVal, tmpBB, 0);
    $$ = S;
    CHECK_FOR_ERROR
  }
  | INVOKE OptCallingConv OptRetAttrs ResultTypes ValueRef '(' ParamList ')' 
    OptFuncAttrs TO LABEL ValueRef UNWIND LABEL ValueRef {

    // Handle the short syntax
    const PointerType *PFTy = 0;
    const FunctionType *Ty = 0;
    if (!(PFTy = dyn_cast<PointerType>($4->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      ParamList::iterator I = $7->begin(), E = $7->end();
      for (; I != E; ++I) {
        const Type *Ty = I->Val->getType();
        if (Ty == Type::VoidTy)
          GEN_ERROR("Short call syntax cannot be used with varargs");
        ParamTypes.push_back(Ty);
      }

      if (!FunctionType::isValidReturnType(*$4))
        GEN_ERROR("Invalid result type for LLVM function");

      Ty = FunctionType::get($4->get(), ParamTypes, false);
      PFTy = PointerType::getUnqual(Ty);
    }

    delete $4;

    Value *V = getVal(PFTy, $5);   // Get the function we're calling...
    CHECK_FOR_ERROR
    BasicBlock *Normal = getBBVal($12);
    CHECK_FOR_ERROR
    BasicBlock *Except = getBBVal($15);
    CHECK_FOR_ERROR

    SmallVector<AttributeWithIndex, 8> Attrs;
    //FIXME : In 3.0, stop accepting zext, sext and inreg as optional function 
    //attributes.
    Attributes RetAttrs = $3;
    if ($9 != Attribute::None) {
      if ($9 & Attribute::ZExt) {
        RetAttrs = RetAttrs | Attribute::ZExt;
        $9 = $9 ^ Attribute::ZExt;
      }
      if ($9 & Attribute::SExt) {
        RetAttrs = RetAttrs | Attribute::SExt;
        $9 = $9 ^ Attribute::SExt;
      }
      if ($9 & Attribute::InReg) {
        RetAttrs = RetAttrs | Attribute::InReg;
        $9 = $9 ^ Attribute::InReg;
      }
    }
    if (RetAttrs != Attribute::None)
      Attrs.push_back(AttributeWithIndex::get(0, RetAttrs));
    
    // Check the arguments
    ValueList Args;
    if ($7->empty()) {                                   // Has no arguments?
      // Make sure no arguments is a good thing!
      if (Ty->getNumParams() != 0)
        GEN_ERROR("No arguments passed to a function that "
                       "expects arguments");
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly!
      FunctionType::param_iterator I = Ty->param_begin();
      FunctionType::param_iterator E = Ty->param_end();
      ParamList::iterator ArgI = $7->begin(), ArgE = $7->end();
      unsigned index = 1;

      for (; ArgI != ArgE && I != E; ++ArgI, ++I, ++index) {
        if (ArgI->Val->getType() != *I)
          GEN_ERROR("Parameter " + ArgI->Val->getName()+ " is not of type '" +
                         (*I)->getDescription() + "'");
        Args.push_back(ArgI->Val);
        if (ArgI->Attrs != Attribute::None)
          Attrs.push_back(AttributeWithIndex::get(index, ArgI->Attrs));
      }

      if (Ty->isVarArg()) {
        if (I == E)
          for (; ArgI != ArgE; ++ArgI, ++index) {
            Args.push_back(ArgI->Val); // push the remaining varargs
            if (ArgI->Attrs != Attribute::None)
              Attrs.push_back(AttributeWithIndex::get(index, ArgI->Attrs));
          }
      } else if (I != E || ArgI != ArgE)
        GEN_ERROR("Invalid number of parameters detected");
    }
    if ($9 != Attribute::None)
      Attrs.push_back(AttributeWithIndex::get(~0, $9));
    AttrListPtr PAL;
    if (!Attrs.empty())
      PAL = AttrListPtr::get(Attrs.begin(), Attrs.end());

    // Create the InvokeInst
    InvokeInst *II = InvokeInst::Create(V, Normal, Except,
                                        Args.begin(), Args.end());
    II->setCallingConv($2);
    II->setAttributes(PAL);
    $$ = II;
    delete $7;
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
    Constant *V = cast<Constant>(getExistingVal($2, $3));
    CHECK_FOR_ERROR
    if (V == 0)
      GEN_ERROR("May only switch on a constant pool value");

    BasicBlock* tmpBB = getBBVal($6);
    CHECK_FOR_ERROR
    $$->push_back(std::make_pair(V, tmpBB));
  }
  | IntType ConstValueRef ',' LABEL ValueRef {
    $$ = new std::vector<std::pair<Constant*, BasicBlock*> >();
    Constant *V = cast<Constant>(getExistingVal($1, $2));
    CHECK_FOR_ERROR

    if (V == 0)
      GEN_ERROR("May only switch on a constant pool value");

    BasicBlock* tmpBB = getBBVal($5);
    CHECK_FOR_ERROR
    $$->push_back(std::make_pair(V, tmpBB));
  };

Inst : OptLocalAssign InstVal {
    // Is this definition named?? if so, assign the name...
    setValueName($2, $1);
    CHECK_FOR_ERROR
    InsertValue($2);
    $$ = $2;
    CHECK_FOR_ERROR
  };

Inst : LocalNumber InstVal {
    CHECK_FOR_ERROR
    int ValNum = InsertValue($2);

    if (ValNum != (int)$1)
      GEN_ERROR("Result value number %" + utostr($1) +
                " is incorrect, expected %" + utostr((unsigned)ValNum));

    $$ = $2;
    CHECK_FOR_ERROR
  };


PHIList : Types '[' ValueRef ',' ValueRef ']' {    // Used for PHI nodes
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
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


ParamList : Types OptAttributes ValueRef OptAttributes {
    // FIXME: Remove trailing OptAttributes in LLVM 3.0, it was a mistake in 2.0
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$1)->getDescription());
    // Used for call and invoke instructions
    $$ = new ParamList();
    ParamListEntry E; E.Attrs = $2 | $4; E.Val = getVal($1->get(), $3);
    $$->push_back(E);
    delete $1;
    CHECK_FOR_ERROR
  }
  | LABEL OptAttributes ValueRef OptAttributes {
    // FIXME: Remove trailing OptAttributes in LLVM 3.0, it was a mistake in 2.0
    // Labels are only valid in ASMs
    $$ = new ParamList();
    ParamListEntry E; E.Attrs = $2 | $4; E.Val = getBBVal($3);
    $$->push_back(E);
    CHECK_FOR_ERROR
  }
  | ParamList ',' Types OptAttributes ValueRef OptAttributes {
    // FIXME: Remove trailing OptAttributes in LLVM 3.0, it was a mistake in 2.0
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$3)->getDescription());
    $$ = $1;
    ParamListEntry E; E.Attrs = $4 | $6; E.Val = getVal($3->get(), $5);
    $$->push_back(E);
    delete $3;
    CHECK_FOR_ERROR
  }
  | ParamList ',' LABEL OptAttributes ValueRef OptAttributes {
    // FIXME: Remove trailing OptAttributes in LLVM 3.0, it was a mistake in 2.0
    $$ = $1;
    ParamListEntry E; E.Attrs = $4 | $6; E.Val = getBBVal($5);
    $$->push_back(E);
    CHECK_FOR_ERROR
  }
  | /*empty*/ { $$ = new ParamList(); };

IndexList       // Used for gep instructions and constant expressions
  : /*empty*/ { $$ = new std::vector<Value*>(); }
  | IndexList ',' ResolvedVal {
    $$ = $1;
    $$->push_back($3);
    CHECK_FOR_ERROR
  }
  ;

ConstantIndexList       // Used for insertvalue and extractvalue instructions
  : ',' EUINT64VAL {
    $$ = new std::vector<unsigned>();
    if ((unsigned)$2 != $2)
      GEN_ERROR("Index " + utostr($2) + " is not valid for insertvalue or extractvalue.");
    $$->push_back($2);
  }
  | ConstantIndexList ',' EUINT64VAL {
    $$ = $1;
    if ((unsigned)$3 != $3)
      GEN_ERROR("Index " + utostr($3) + " is not valid for insertvalue or extractvalue.");
    $$->push_back($3);
    CHECK_FOR_ERROR
  }
  ;

OptTailCall : TAIL CALL {
    $$ = true;
    CHECK_FOR_ERROR
  }
  | CALL {
    $$ = false;
    CHECK_FOR_ERROR
  };

InstVal : ArithmeticOps Types ValueRef ',' ValueRef {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$2)->getDescription());
    if (!(*$2)->isInteger() && !(*$2)->isFloatingPoint() &&
        !isa<VectorType>((*$2).get()))
      GEN_ERROR(
        "Arithmetic operator requires integer, FP, or packed operands");
    Value* val1 = getVal(*$2, $3);
    CHECK_FOR_ERROR
    Value* val2 = getVal(*$2, $5);
    CHECK_FOR_ERROR
    $$ = BinaryOperator::Create($1, val1, val2);
    if ($$ == 0)
      GEN_ERROR("binary operator returned null");
    delete $2;
  }
  | LogicalOps Types ValueRef ',' ValueRef {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$2)->getDescription());
    if (!(*$2)->isInteger()) {
      if (!isa<VectorType>($2->get()) ||
          !cast<VectorType>($2->get())->getElementType()->isInteger())
        GEN_ERROR("Logical operator requires integral operands");
    }
    Value* tmpVal1 = getVal(*$2, $3);
    CHECK_FOR_ERROR
    Value* tmpVal2 = getVal(*$2, $5);
    CHECK_FOR_ERROR
    $$ = BinaryOperator::Create($1, tmpVal1, tmpVal2);
    if ($$ == 0)
      GEN_ERROR("binary operator returned null");
    delete $2;
  }
  | ICMP IPredicates Types ValueRef ',' ValueRef  {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$3)->getDescription());
    Value* tmpVal1 = getVal(*$3, $4);
    CHECK_FOR_ERROR
    Value* tmpVal2 = getVal(*$3, $6);
    CHECK_FOR_ERROR
    $$ = CmpInst::Create($1, $2, tmpVal1, tmpVal2);
    if ($$ == 0)
      GEN_ERROR("icmp operator returned null");
    delete $3;
  }
  | FCMP FPredicates Types ValueRef ',' ValueRef  {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$3)->getDescription());
    Value* tmpVal1 = getVal(*$3, $4);
    CHECK_FOR_ERROR
    Value* tmpVal2 = getVal(*$3, $6);
    CHECK_FOR_ERROR
    $$ = CmpInst::Create($1, $2, tmpVal1, tmpVal2);
    if ($$ == 0)
      GEN_ERROR("fcmp operator returned null");
    delete $3;
  }
  | VICMP IPredicates Types ValueRef ',' ValueRef  {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$3)->getDescription());
    if (!isa<VectorType>((*$3).get()))
      GEN_ERROR("Scalar types not supported by vicmp instruction");
    Value* tmpVal1 = getVal(*$3, $4);
    CHECK_FOR_ERROR
    Value* tmpVal2 = getVal(*$3, $6);
    CHECK_FOR_ERROR
    $$ = CmpInst::Create($1, $2, tmpVal1, tmpVal2);
    if ($$ == 0)
      GEN_ERROR("vicmp operator returned null");
    delete $3;
  }
  | VFCMP FPredicates Types ValueRef ',' ValueRef  {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$3)->getDescription());
    if (!isa<VectorType>((*$3).get()))
      GEN_ERROR("Scalar types not supported by vfcmp instruction");
    Value* tmpVal1 = getVal(*$3, $4);
    CHECK_FOR_ERROR
    Value* tmpVal2 = getVal(*$3, $6);
    CHECK_FOR_ERROR
    $$ = CmpInst::Create($1, $2, tmpVal1, tmpVal2);
    if ($$ == 0)
      GEN_ERROR("vfcmp operator returned null");
    delete $3;
  }
  | CastOps ResolvedVal TO Types {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$4)->getDescription());
    Value* Val = $2;
    const Type* DestTy = $4->get();
    if (!CastInst::castIsValid($1, Val, DestTy))
      GEN_ERROR("invalid cast opcode for cast from '" +
                Val->getType()->getDescription() + "' to '" +
                DestTy->getDescription() + "'");
    $$ = CastInst::Create($1, Val, DestTy);
    delete $4;
  }
  | SELECT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    if (isa<VectorType>($2->getType())) {
      // vector select
      if (!isa<VectorType>($4->getType())
      || !isa<VectorType>($6->getType()) )
        GEN_ERROR("vector select value types must be vector types");
      const VectorType* cond_type = cast<VectorType>($2->getType());
      const VectorType* select_type = cast<VectorType>($4->getType());
      if (cond_type->getElementType() != Type::Int1Ty)
        GEN_ERROR("vector select condition element type must be boolean");
      if (cond_type->getNumElements() != select_type->getNumElements())
        GEN_ERROR("vector select number of elements must be the same");
    } else {
      if ($2->getType() != Type::Int1Ty)
        GEN_ERROR("select condition must be boolean");
    }
    if ($4->getType() != $6->getType())
      GEN_ERROR("select value types must match");
    $$ = SelectInst::Create($2, $4, $6);
    CHECK_FOR_ERROR
  }
  | VAARG ResolvedVal ',' Types {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$4)->getDescription());
    $$ = new VAArgInst($2, *$4);
    delete $4;
    CHECK_FOR_ERROR
  }
  | EXTRACTELEMENT ResolvedVal ',' ResolvedVal {
    if (!ExtractElementInst::isValidOperands($2, $4))
      GEN_ERROR("Invalid extractelement operands");
    $$ = new ExtractElementInst($2, $4);
    CHECK_FOR_ERROR
  }
  | INSERTELEMENT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    if (!InsertElementInst::isValidOperands($2, $4, $6))
      GEN_ERROR("Invalid insertelement operands");
    $$ = InsertElementInst::Create($2, $4, $6);
    CHECK_FOR_ERROR
  }
  | SHUFFLEVECTOR ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    if (!ShuffleVectorInst::isValidOperands($2, $4, $6))
      GEN_ERROR("Invalid shufflevector operands");
    $$ = new ShuffleVectorInst($2, $4, $6);
    CHECK_FOR_ERROR
  }
  | PHI_TOK PHIList {
    const Type *Ty = $2->front().first->getType();
    if (!Ty->isFirstClassType())
      GEN_ERROR("PHI node operands must be of first class type");
    $$ = PHINode::Create(Ty);
    ((PHINode*)$$)->reserveOperandSpace($2->size());
    while ($2->begin() != $2->end()) {
      if ($2->front().first->getType() != Ty)
        GEN_ERROR("All elements of a PHI node must be of the same type");
      cast<PHINode>($$)->addIncoming($2->front().first, $2->front().second);
      $2->pop_front();
    }
    delete $2;  // Free the list...
    CHECK_FOR_ERROR
  }
  | OptTailCall OptCallingConv OptRetAttrs ResultTypes ValueRef '(' ParamList ')'
    OptFuncAttrs {

    // Handle the short syntax
    const PointerType *PFTy = 0;
    const FunctionType *Ty = 0;
    if (!(PFTy = dyn_cast<PointerType>($4->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      ParamList::iterator I = $7->begin(), E = $7->end();
      for (; I != E; ++I) {
        const Type *Ty = I->Val->getType();
        if (Ty == Type::VoidTy)
          GEN_ERROR("Short call syntax cannot be used with varargs");
        ParamTypes.push_back(Ty);
      }

      if (!FunctionType::isValidReturnType(*$4))
        GEN_ERROR("Invalid result type for LLVM function");

      Ty = FunctionType::get($4->get(), ParamTypes, false);
      PFTy = PointerType::getUnqual(Ty);
    }

    Value *V = getVal(PFTy, $5);   // Get the function we're calling...
    CHECK_FOR_ERROR

    // Check for call to invalid intrinsic to avoid crashing later.
    if (Function *theF = dyn_cast<Function>(V)) {
      if (theF->hasName() && (theF->getValueName()->getKeyLength() >= 5) &&
          (0 == strncmp(theF->getValueName()->getKeyData(), "llvm.", 5)) &&
          !theF->getIntrinsicID(true))
        GEN_ERROR("Call to invalid LLVM intrinsic function '" +
                  theF->getName() + "'");
    }

    // Set up the Attributes for the function
    SmallVector<AttributeWithIndex, 8> Attrs;
    //FIXME : In 3.0, stop accepting zext, sext and inreg as optional function 
    //attributes.
    Attributes RetAttrs = $3;
    if ($9 != Attribute::None) {
      if ($9 & Attribute::ZExt) {
        RetAttrs = RetAttrs | Attribute::ZExt;
        $9 = $9 ^ Attribute::ZExt;
      }
      if ($9 & Attribute::SExt) {
        RetAttrs = RetAttrs | Attribute::SExt;
        $9 = $9 ^ Attribute::SExt;
      }
      if ($9 & Attribute::InReg) {
        RetAttrs = RetAttrs | Attribute::InReg;
        $9 = $9 ^ Attribute::InReg;
      }
    }
    if (RetAttrs != Attribute::None)
      Attrs.push_back(AttributeWithIndex::get(0, RetAttrs));
    
    // Check the arguments
    ValueList Args;
    if ($7->empty()) {                                   // Has no arguments?
      // Make sure no arguments is a good thing!
      if (Ty->getNumParams() != 0)
        GEN_ERROR("No arguments passed to a function that "
                       "expects arguments");
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly.  Also, gather any parameter attributes.
      FunctionType::param_iterator I = Ty->param_begin();
      FunctionType::param_iterator E = Ty->param_end();
      ParamList::iterator ArgI = $7->begin(), ArgE = $7->end();
      unsigned index = 1;

      for (; ArgI != ArgE && I != E; ++ArgI, ++I, ++index) {
        if (ArgI->Val->getType() != *I)
          GEN_ERROR("Parameter " + ArgI->Val->getName()+ " is not of type '" +
                         (*I)->getDescription() + "'");
        Args.push_back(ArgI->Val);
        if (ArgI->Attrs != Attribute::None)
          Attrs.push_back(AttributeWithIndex::get(index, ArgI->Attrs));
      }
      if (Ty->isVarArg()) {
        if (I == E)
          for (; ArgI != ArgE; ++ArgI, ++index) {
            Args.push_back(ArgI->Val); // push the remaining varargs
            if (ArgI->Attrs != Attribute::None)
              Attrs.push_back(AttributeWithIndex::get(index, ArgI->Attrs));
          }
      } else if (I != E || ArgI != ArgE)
        GEN_ERROR("Invalid number of parameters detected");
    }
    if ($9 != Attribute::None)
      Attrs.push_back(AttributeWithIndex::get(~0, $9));

    // Finish off the Attributes and check them
    AttrListPtr PAL;
    if (!Attrs.empty())
      PAL = AttrListPtr::get(Attrs.begin(), Attrs.end());

    // Create the call node
    CallInst *CI = CallInst::Create(V, Args.begin(), Args.end());
    CI->setTailCall($1);
    CI->setCallingConv($2);
    CI->setAttributes(PAL);
    $$ = CI;
    delete $7;
    delete $4;
    CHECK_FOR_ERROR
  }
  | MemoryInst {
    $$ = $1;
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
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$2)->getDescription());
    $$ = new MallocInst(*$2, 0, $3);
    delete $2;
    CHECK_FOR_ERROR
  }
  | MALLOC Types ',' INTTYPE ValueRef OptCAlign {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$2)->getDescription());
    if ($4 != Type::Int32Ty)
      GEN_ERROR("Malloc array size is not a 32-bit integer!");
    Value* tmpVal = getVal($4, $5);
    CHECK_FOR_ERROR
    $$ = new MallocInst(*$2, tmpVal, $6);
    delete $2;
  }
  | ALLOCA Types OptCAlign {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$2)->getDescription());
    $$ = new AllocaInst(*$2, 0, $3);
    delete $2;
    CHECK_FOR_ERROR
  }
  | ALLOCA Types ',' INTTYPE ValueRef OptCAlign {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$2)->getDescription());
    if ($4 != Type::Int32Ty)
      GEN_ERROR("Alloca array size is not a 32-bit integer!");
    Value* tmpVal = getVal($4, $5);
    CHECK_FOR_ERROR
    $$ = new AllocaInst(*$2, tmpVal, $6);
    delete $2;
  }
  | FREE ResolvedVal {
    if (!isa<PointerType>($2->getType()))
      GEN_ERROR("Trying to free nonpointer type " +
                     $2->getType()->getDescription() + "");
    $$ = new FreeInst($2);
    CHECK_FOR_ERROR
  }

  | OptVolatile LOAD Types ValueRef OptCAlign {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$3)->getDescription());
    if (!isa<PointerType>($3->get()))
      GEN_ERROR("Can't load from nonpointer type: " +
                     (*$3)->getDescription());
    if (!cast<PointerType>($3->get())->getElementType()->isFirstClassType())
      GEN_ERROR("Can't load from pointer of non-first-class type: " +
                     (*$3)->getDescription());
    Value* tmpVal = getVal(*$3, $4);
    CHECK_FOR_ERROR
    $$ = new LoadInst(tmpVal, "", $1, $5);
    delete $3;
  }
  | OptVolatile STORE ResolvedVal ',' Types ValueRef OptCAlign {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$5)->getDescription());
    const PointerType *PT = dyn_cast<PointerType>($5->get());
    if (!PT)
      GEN_ERROR("Can't store to a nonpointer type: " +
                     (*$5)->getDescription());
    const Type *ElTy = PT->getElementType();
    if (ElTy != $3->getType())
      GEN_ERROR("Can't store '" + $3->getType()->getDescription() +
                     "' into space of type '" + ElTy->getDescription() + "'");

    Value* tmpVal = getVal(*$5, $6);
    CHECK_FOR_ERROR
    $$ = new StoreInst($3, tmpVal, $1, $7);
    delete $5;
  }
  | GETRESULT Types ValueRef ',' EUINT64VAL  {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$2)->getDescription());
    if (!isa<StructType>($2->get()) && !isa<ArrayType>($2->get()))
      GEN_ERROR("getresult insn requires an aggregate operand");
    if (!ExtractValueInst::getIndexedType(*$2, $5))
      GEN_ERROR("Invalid getresult index for type '" +
                     (*$2)->getDescription()+ "'");

    Value *tmpVal = getVal(*$2, $3);
    CHECK_FOR_ERROR
    $$ = ExtractValueInst::Create(tmpVal, $5);
    delete $2;
  }
  | GETELEMENTPTR Types ValueRef IndexList {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$2)->getDescription());
    if (!isa<PointerType>($2->get()))
      GEN_ERROR("getelementptr insn requires pointer operand");

    if (!GetElementPtrInst::getIndexedType(*$2, $4->begin(), $4->end()))
      GEN_ERROR("Invalid getelementptr indices for type '" +
                     (*$2)->getDescription()+ "'");
    Value* tmpVal = getVal(*$2, $3);
    CHECK_FOR_ERROR
    $$ = GetElementPtrInst::Create(tmpVal, $4->begin(), $4->end());
    delete $2;
    delete $4;
  }
  | EXTRACTVALUE Types ValueRef ConstantIndexList {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$2)->getDescription());
    if (!isa<StructType>($2->get()) && !isa<ArrayType>($2->get()))
      GEN_ERROR("extractvalue insn requires an aggregate operand");

    if (!ExtractValueInst::getIndexedType(*$2, $4->begin(), $4->end()))
      GEN_ERROR("Invalid extractvalue indices for type '" +
                     (*$2)->getDescription()+ "'");
    Value* tmpVal = getVal(*$2, $3);
    CHECK_FOR_ERROR
    $$ = ExtractValueInst::Create(tmpVal, $4->begin(), $4->end());
    delete $2;
    delete $4;
  }
  | INSERTVALUE Types ValueRef ',' Types ValueRef ConstantIndexList {
    if (!UpRefs.empty())
      GEN_ERROR("Invalid upreference in type: " + (*$2)->getDescription());
    if (!isa<StructType>($2->get()) && !isa<ArrayType>($2->get()))
      GEN_ERROR("extractvalue insn requires an aggregate operand");

    if (ExtractValueInst::getIndexedType(*$2, $7->begin(), $7->end()) != $5->get())
      GEN_ERROR("Invalid insertvalue indices for type '" +
                     (*$2)->getDescription()+ "'");
    Value* aggVal = getVal(*$2, $3);
    Value* tmpVal = getVal(*$5, $6);
    CHECK_FOR_ERROR
    $$ = InsertValueInst::Create(aggVal, tmpVal, $7->begin(), $7->end());
    delete $2;
    delete $5;
    delete $7;
  };


%%

// common code from the two 'RunVMAsmParser' functions
static Module* RunParser(Module * M) {
  CurModule.CurrentModule = M;
  // Check to make sure the parser succeeded
  if (yyparse()) {
    if (ParserResult)
      delete ParserResult;
    return 0;
  }

  // Emit an error if there are any unresolved types left.
  if (!CurModule.LateResolveTypes.empty()) {
    const ValID &DID = CurModule.LateResolveTypes.begin()->first;
    if (DID.Type == ValID::LocalName) {
      GenerateError("Undefined type remains at eof: '"+DID.getName() + "'");
    } else {
      GenerateError("Undefined type remains at eof: #" + itostr(DID.Num));
    }
    if (ParserResult)
      delete ParserResult;
    return 0;
  }

  // Emit an error if there are any unresolved values left.
  if (!CurModule.LateResolveValues.empty()) {
    Value *V = CurModule.LateResolveValues.back();
    std::map<Value*, std::pair<ValID, int> >::iterator I =
      CurModule.PlaceHolderInfo.find(V);

    if (I != CurModule.PlaceHolderInfo.end()) {
      ValID &DID = I->second.first;
      if (DID.Type == ValID::LocalName) {
        GenerateError("Undefined value remains at eof: "+DID.getName() + "'");
      } else {
        GenerateError("Undefined value remains at eof: #" + itostr(DID.Num));
      }
      if (ParserResult)
        delete ParserResult;
      return 0;
    }
  }

  // Check to make sure that parsing produced a result
  if (!ParserResult)
    return 0;

  // Reset ParserResult variable while saving its value for the result.
  Module *Result = ParserResult;
  ParserResult = 0;

  return Result;
}

void llvm::GenerateError(const std::string &message, int LineNo) {
  if (LineNo == -1) LineNo = LLLgetLineNo();
  // TODO: column number in exception
  if (TheParseError)
    TheParseError->setError(LLLgetFilename(), message, LineNo);
  TriggerError = 1;
}

int yyerror(const char *ErrorMsg) {
  std::string where = LLLgetFilename() + ":" + utostr(LLLgetLineNo()) + ": ";
  std::string errMsg = where + "error: " + std::string(ErrorMsg);
  if (yychar != YYEMPTY && yychar != 0) {
    errMsg += " while reading token: '";
    errMsg += std::string(LLLgetTokenStart(),
                          LLLgetTokenStart()+LLLgetTokenLength()) + "'";
  }
  GenerateError(errMsg);
  return 0;
}
