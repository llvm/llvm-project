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
#include "UpgradeInternals.h"
#include "llvm/CallingConv.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <list>
#include <utility>

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
#define YYINCLUDED_STDLIB_H
#define YYDEBUG 1

int yylex();
int yyparse();

int yyerror(const char*);
static void warning(const std::string& WarningMsg);

namespace llvm {

std::istream* LexInput;
static std::string CurFilename;

// This bool controls whether attributes are ever added to function declarations
// definitions and calls.
static bool AddAttributes = false;

static Module *ParserResult;
static bool ObsoleteVarArgs;
static bool NewVarArgs;
static BasicBlock *CurBB;
static GlobalVariable *CurGV;
static unsigned lastCallingConv;

// This contains info used when building the body of a function.  It is
// destroyed when the function is completed.
//
typedef std::vector<Value *> ValueList;           // Numbered defs

typedef std::pair<std::string,TypeInfo> RenameMapKey;
typedef std::map<RenameMapKey,std::string> RenameMapType;

static void 
ResolveDefinitions(std::map<const Type *,ValueList> &LateResolvers,
                   std::map<const Type *,ValueList> *FutureLateResolvers = 0);

static struct PerModuleInfo {
  Module *CurrentModule;
  std::map<const Type *, ValueList> Values; // Module level numbered definitions
  std::map<const Type *,ValueList> LateResolveValues;
  std::vector<PATypeHolder> Types;
  std::vector<Signedness> TypeSigns;
  std::map<std::string,Signedness> NamedTypeSigns;
  std::map<std::string,Signedness> NamedValueSigns;
  std::map<ValID, PATypeHolder> LateResolveTypes;
  static Module::Endianness Endian;
  static Module::PointerSize PointerSize;
  RenameMapType RenameMap;

  /// PlaceHolderInfo - When temporary placeholder objects are created, remember
  /// how they were referenced and on which line of the input they came from so
  /// that we can resolve them later and print error messages as appropriate.
  std::map<Value*, std::pair<ValID, int> > PlaceHolderInfo;

  // GlobalRefs - This maintains a mapping between <Type, ValID>'s and forward
  // references to global values.  Global values may be referenced before they
  // are defined, and if so, the temporary object that they represent is held
  // here.  This is used for forward references of GlobalValues.
  //
  typedef std::map<std::pair<const PointerType *, ValID>, GlobalValue*> 
    GlobalRefsType;
  GlobalRefsType GlobalRefs;

  void ModuleDone() {
    // If we could not resolve some functions at function compilation time
    // (calls to functions before they are defined), resolve them now...  Types
    // are resolved when the constant pool has been completely parsed.
    //
    ResolveDefinitions(LateResolveValues);

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
      error(UndefinedReferences);
      return;
    }

    if (CurrentModule->getDataLayout().empty()) {
      std::string dataLayout;
      if (Endian != Module::AnyEndianness)
        dataLayout.append(Endian == Module::BigEndian ? "E" : "e");
      if (PointerSize != Module::AnyPointerSize) {
        if (!dataLayout.empty())
          dataLayout += "-";
        dataLayout.append(PointerSize == Module::Pointer64 ? 
                          "p:64:64" : "p:32:32");
      }
      CurrentModule->setDataLayout(dataLayout);
    }

    Values.clear();         // Clear out function local definitions
    Types.clear();
    TypeSigns.clear();
    NamedTypeSigns.clear();
    NamedValueSigns.clear();
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
  void setEndianness(Module::Endianness E) { Endian = E; }
  void setPointerSize(Module::PointerSize sz) { PointerSize = sz; }
} CurModule;

Module::Endianness  PerModuleInfo::Endian = Module::AnyEndianness;
Module::PointerSize PerModuleInfo::PointerSize = Module::AnyPointerSize;

static struct PerFunctionInfo {
  Function *CurrentFunction;     // Pointer to current function being created

  std::map<const Type*, ValueList> Values; // Keep track of #'d definitions
  std::map<const Type*, ValueList> LateResolveValues;
  bool isDeclare;                   // Is this function a forward declararation?
  GlobalValue::LinkageTypes Linkage;// Linkage for forward declaration.

  /// BBForwardRefs - When we see forward references to basic blocks, keep
  /// track of them here.
  std::map<BasicBlock*, std::pair<ValID, int> > BBForwardRefs;
  std::vector<BasicBlock*> NumberedBlocks;
  RenameMapType RenameMap;
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
      error("Undefined reference to label " + 
            BBForwardRefs.begin()->first->getName());
      return;
    }

    // Resolve all forward references now.
    ResolveDefinitions(LateResolveValues, &CurModule.LateResolveValues);

    Values.clear();         // Clear out function local definitions
    RenameMap.clear();
    CurrentFunction = 0;
    isDeclare = false;
    Linkage = GlobalValue::ExternalLinkage;
  }
} CurFun;  // Info for the current function...

static bool inFunctionScope() { return CurFun.CurrentFunction != 0; }

/// This function is just a utility to make a Key value for the rename map.
/// The Key is a combination of the name, type, Signedness of the original 
/// value (global/function). This just constructs the key and ensures that
/// named Signedness values are resolved to the actual Signedness.
/// @brief Make a key for the RenameMaps
static RenameMapKey makeRenameMapKey(const std::string &Name, const Type* Ty, 
                                     const Signedness &Sign) {
  TypeInfo TI; 
  TI.T = Ty; 
  if (Sign.isNamed())
    // Don't allow Named Signedness nodes because they won't match. The actual
    // Signedness must be looked up in the NamedTypeSigns map.
    TI.S.copy(CurModule.NamedTypeSigns[Sign.getName()]);
  else
    TI.S.copy(Sign);
  return std::make_pair(Name, TI);
}


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

static const Type *getType(const ValID &D, bool DoNotImprovise = false) {
  switch (D.Type) {
  case ValID::NumberVal:               // Is it a numbered definition?
    // Module constants occupy the lowest numbered slots...
    if ((unsigned)D.Num < CurModule.Types.size()) {
      return CurModule.Types[(unsigned)D.Num];
    }
    break;
  case ValID::NameVal:                 // Is it a named definition?
    if (const Type *N = CurModule.CurrentModule->getTypeByName(D.Name)) {
      return N;
    }
    break;
  default:
    error("Internal parser error: Invalid symbol type reference");
    return 0;
  }

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  if (DoNotImprovise) return 0;  // Do we just want a null to be returned?

  if (inFunctionScope()) {
    if (D.Type == ValID::NameVal) {
      error("Reference to an undefined type: '" + D.getName() + "'");
      return 0;
    } else {
      error("Reference to an undefined type: #" + itostr(D.Num));
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

/// This is like the getType method except that instead of looking up the type
/// for a given ID, it looks up that type's sign.
/// @brief Get the signedness of a referenced type
static Signedness getTypeSign(const ValID &D) {
  switch (D.Type) {
  case ValID::NumberVal:               // Is it a numbered definition?
    // Module constants occupy the lowest numbered slots...
    if ((unsigned)D.Num < CurModule.TypeSigns.size()) {
      return CurModule.TypeSigns[(unsigned)D.Num];
    }
    break;
  case ValID::NameVal: {               // Is it a named definition?
    std::map<std::string,Signedness>::const_iterator I = 
      CurModule.NamedTypeSigns.find(D.Name);
    if (I != CurModule.NamedTypeSigns.end())
      return I->second;
    // Perhaps its a named forward .. just cache the name
    Signedness S;
    S.makeNamed(D.Name);
    return S;
  }
  default: 
    break;
  }
  // If we don't find it, its signless
  Signedness S;
  S.makeSignless();
  return S;
}

/// This function is analagous to getElementType in LLVM. It provides the same
/// function except that it looks up the Signedness instead of the type. This is
/// used when processing GEP instructions that need to extract the type of an
/// indexed struct/array/ptr member. 
/// @brief Look up an element's sign.
static Signedness getElementSign(const ValueInfo& VI, 
                                 const std::vector<Value*> &Indices) {
  const Type *Ptr = VI.V->getType();
  assert(isa<PointerType>(Ptr) && "Need pointer type");

  unsigned CurIdx = 0;
  Signedness S(VI.S);
  while (const CompositeType *CT = dyn_cast<CompositeType>(Ptr)) {
    if (CurIdx == Indices.size())
      break;

    Value *Index = Indices[CurIdx++];
    assert(!isa<PointerType>(CT) || CurIdx == 1 && "Invalid type");
    Ptr = CT->getTypeAtIndex(Index);
    if (const Type* Ty = Ptr->getForwardedType())
      Ptr = Ty;
    assert(S.isComposite() && "Bad Signedness type");
    if (isa<StructType>(CT)) {
      S = S.get(cast<ConstantInt>(Index)->getZExtValue());
    } else {
      S = S.get(0UL);
    }
    if (S.isNamed())
      S = CurModule.NamedTypeSigns[S.getName()];
  }
  Signedness Result;
  Result.makeComposite(S);
  return Result;
}

/// This function just translates a ConstantInfo into a ValueInfo and calls
/// getElementSign(ValueInfo,...). Its just a convenience.
/// @brief ConstantInfo version of getElementSign.
static Signedness getElementSign(const ConstInfo& CI, 
                                 const std::vector<Constant*> &Indices) {
  ValueInfo VI;
  VI.V = CI.C;
  VI.S.copy(CI.S);
  std::vector<Value*> Idx;
  for (unsigned i = 0; i < Indices.size(); ++i)
    Idx.push_back(Indices[i]);
  Signedness result = getElementSign(VI, Idx);
  VI.destroy();
  return result;
}

/// This function determines if two function types differ only in their use of
/// the sret parameter attribute in the first argument. If they are identical 
/// in all other respects, it returns true. Otherwise, it returns false.
static bool FuncTysDifferOnlyBySRet(const FunctionType *F1, 
                                    const FunctionType *F2) {
  if (F1->getReturnType() != F2->getReturnType() ||
      F1->getNumParams() != F2->getNumParams())
    return false;
  const ParamAttrsList *PAL1 = F1->getParamAttrs();
  const ParamAttrsList *PAL2 = F2->getParamAttrs();
  if (PAL1 && !PAL2 || PAL2 && !PAL1)
    return false;
  if (PAL1 && PAL2 && ((PAL1->size() != PAL2->size()) ||
      (PAL1->getParamAttrs(0) != PAL2->getParamAttrs(0)))) 
    return false;
  unsigned SRetMask = ~unsigned(ParamAttr::StructRet);
  for (unsigned i = 0; i < F1->getNumParams(); ++i) {
    if (F1->getParamType(i) != F2->getParamType(i) || (PAL1 && PAL2 &&
        (unsigned(PAL1->getParamAttrs(i+1)) & SRetMask !=
         unsigned(PAL2->getParamAttrs(i+1)) & SRetMask)))
      return false;
  }
  return true;
}

/// This function determines if the type of V and Ty differ only by the SRet
/// parameter attribute. This is a more generalized case of
/// FuncTysDIfferOnlyBySRet since it doesn't require FunctionType arguments.
static bool TypesDifferOnlyBySRet(Value *V, const Type* Ty) {
  if (V->getType() == Ty)
    return true;
  const PointerType *PF1 = dyn_cast<PointerType>(Ty);
  const PointerType *PF2 = dyn_cast<PointerType>(V->getType());
  if (PF1 && PF2) {
    const FunctionType* FT1 = dyn_cast<FunctionType>(PF1->getElementType());
    const FunctionType* FT2 = dyn_cast<FunctionType>(PF2->getElementType());
    if (FT1 && FT2)
      return FuncTysDifferOnlyBySRet(FT1, FT2);
  }
  return false;
}

// The upgrade of csretcc to sret param attribute may have caused a function 
// to not be found because the param attribute changed the type of the called 
// function. This helper function, used in getExistingValue, detects that
// situation and bitcasts the function to the correct type.
static Value* handleSRetFuncTypeMerge(Value *V, const Type* Ty) {
  // Handle degenerate cases
  if (!V)
    return 0;
  if (V->getType() == Ty)
    return V;

  const PointerType *PF1 = dyn_cast<PointerType>(Ty);
  const PointerType *PF2 = dyn_cast<PointerType>(V->getType());
  if (PF1 && PF2) {
    const FunctionType *FT1 = dyn_cast<FunctionType>(PF1->getElementType());
    const FunctionType *FT2 = dyn_cast<FunctionType>(PF2->getElementType());
    if (FT1 && FT2 && FuncTysDifferOnlyBySRet(FT1, FT2)) {
      const ParamAttrsList *PAL2 = FT2->getParamAttrs();
      if (PAL2 && PAL2->paramHasAttr(1, ParamAttr::StructRet))
        return V;
      else if (Constant *C = dyn_cast<Constant>(V))
        return ConstantExpr::getBitCast(C, PF1);
      else
        return new BitCastInst(V, PF1, "upgrd.cast", CurBB);
    }
      
  }
  return 0;
}

// getExistingValue - Look up the value specified by the provided type and
// the provided ValID.  If the value exists and has already been defined, return
// it.  Otherwise return null.
//
static Value *getExistingValue(const Type *Ty, const ValID &D) {
  if (isa<FunctionType>(Ty)) {
    error("Functions are not values and must be referenced as pointers");
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
    // Get the name out of the ID
    RenameMapKey Key = makeRenameMapKey(D.Name, Ty, D.S);
    Value *V = 0;
    if (inFunctionScope()) {
      // See if the name was renamed
      RenameMapType::const_iterator I = CurFun.RenameMap.find(Key);
      std::string LookupName;
      if (I != CurFun.RenameMap.end())
        LookupName = I->second;
      else
        LookupName = D.Name;
      ValueSymbolTable &SymTab = CurFun.CurrentFunction->getValueSymbolTable();
      V = SymTab.lookup(LookupName);
      if (V && V->getType() != Ty)
        V = handleSRetFuncTypeMerge(V, Ty);
      assert((!V || TypesDifferOnlyBySRet(V, Ty)) && "Found wrong type");
    }
    if (!V) {
      RenameMapType::const_iterator I = CurModule.RenameMap.find(Key);
      std::string LookupName;
      if (I != CurModule.RenameMap.end())
        LookupName = I->second;
      else
        LookupName = D.Name;
      V = CurModule.CurrentModule->getValueSymbolTable().lookup(LookupName);
      if (V && V->getType() != Ty)
        V = handleSRetFuncTypeMerge(V, Ty);
      assert((!V || TypesDifferOnlyBySRet(V, Ty)) && "Found wrong type");
    }
    if (!V) 
      return 0;

    D.destroy();  // Free old strdup'd memory...
    return V;
  }

  // Check to make sure that "Ty" is an integral type, and that our
  // value will fit into the specified type...
  case ValID::ConstSIntVal:    // Is it a constant pool reference??
    if (!ConstantInt::isValueValidForType(Ty, D.ConstPool64)) {
      error("Signed integral constant '" + itostr(D.ConstPool64) + 
            "' is invalid for type '" + Ty->getDescription() + "'");
    }
    return ConstantInt::get(Ty, D.ConstPool64);

  case ValID::ConstUIntVal:     // Is it an unsigned const pool reference?
    if (!ConstantInt::isValueValidForType(Ty, D.UConstPool64)) {
      if (!ConstantInt::isValueValidForType(Ty, D.ConstPool64))
        error("Integral constant '" + utostr(D.UConstPool64) + 
              "' is invalid or out of range");
      else     // This is really a signed reference.  Transmogrify.
        return ConstantInt::get(Ty, D.ConstPool64);
    } else
      return ConstantInt::get(Ty, D.UConstPool64);

  case ValID::ConstFPVal:        // Is it a floating point const pool reference?
    if (!ConstantFP::isValueValidForType(Ty, D.ConstPoolFP))
      error("FP constant invalid for type");
    return ConstantFP::get(Ty, D.ConstPoolFP);

  case ValID::ConstNullVal:      // Is it a null value?
    if (!isa<PointerType>(Ty))
      error("Cannot create a a non pointer null");
    return ConstantPointerNull::get(cast<PointerType>(Ty));

  case ValID::ConstUndefVal:      // Is it an undef value?
    return UndefValue::get(Ty);

  case ValID::ConstZeroVal:      // Is it a zero value?
    return Constant::getNullValue(Ty);
    
  case ValID::ConstantVal:       // Fully resolved constant?
    if (D.ConstantValue->getType() != Ty) 
      error("Constant expression type different from required type");
    return D.ConstantValue;

  case ValID::InlineAsmVal: {    // Inline asm expression
    const PointerType *PTy = dyn_cast<PointerType>(Ty);
    const FunctionType *FTy =
      PTy ? dyn_cast<FunctionType>(PTy->getElementType()) : 0;
    if (!FTy || !InlineAsm::Verify(FTy, D.IAD->Constraints))
      error("Invalid type for asm constraint string");
    InlineAsm *IA = InlineAsm::get(FTy, D.IAD->AsmString, D.IAD->Constraints,
                                   D.IAD->HasSideEffects);
    D.destroy();   // Free InlineAsmDescriptor.
    return IA;
  }
  default:
    assert(0 && "Unhandled case");
    return 0;
  }   // End of switch

  assert(0 && "Unhandled case");
  return 0;
}

// getVal - This function is identical to getExistingValue, except that if a
// value is not already defined, it "improvises" by creating a placeholder var
// that looks and acts just like the requested variable.  When the value is
// defined later, all uses of the placeholder variable are replaced with the
// real thing.
//
static Value *getVal(const Type *Ty, const ValID &ID) {
  if (Ty == Type::LabelTy)
    error("Cannot use a basic block here");

  // See if the value has already been defined.
  Value *V = getExistingValue(Ty, ID);
  if (V) return V;

  if (!Ty->isFirstClassType() && !isa<OpaqueType>(Ty))
    error("Invalid use of a composite type");

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  V = new Argument(Ty);

  // Remember where this forward reference came from.  FIXME, shouldn't we try
  // to recycle these things??
  CurModule.PlaceHolderInfo.insert(
    std::make_pair(V, std::make_pair(ID, Upgradelineno)));

  if (inFunctionScope())
    InsertValue(V, CurFun.LateResolveValues);
  else
    InsertValue(V, CurModule.LateResolveValues);
  return V;
}

/// @brief This just makes any name given to it unique, up to MAX_UINT times.
static std::string makeNameUnique(const std::string& Name) {
  static unsigned UniqueNameCounter = 1;
  std::string Result(Name);
  Result += ".upgrd." + llvm::utostr(UniqueNameCounter++);
  return Result;
}

/// getBBVal - This is used for two purposes:
///  * If isDefinition is true, a new basic block with the specified ID is being
///    defined.
///  * If isDefinition is true, this is a reference to a basic block, which may
///    or may not be a forward reference.
///
static BasicBlock *getBBVal(const ValID &ID, bool isDefinition = false) {
  assert(inFunctionScope() && "Can't get basic block at global scope");

  std::string Name;
  BasicBlock *BB = 0;
  switch (ID.Type) {
  default: 
    error("Illegal label reference " + ID.getName());
    break;
  case ValID::NumberVal:                // Is it a numbered definition?
    if (unsigned(ID.Num) >= CurFun.NumberedBlocks.size())
      CurFun.NumberedBlocks.resize(ID.Num+1);
    BB = CurFun.NumberedBlocks[ID.Num];
    break;
  case ValID::NameVal:                  // Is it a named definition?
    Name = ID.Name;
    if (Value *N = CurFun.CurrentFunction->getValueSymbolTable().lookup(Name)) {
      if (N->getType() != Type::LabelTy) {
        // Register names didn't use to conflict with basic block names
        // because of type planes. Now they all have to be unique. So, we just
        // rename the register and treat this name as if no basic block
        // had been found.
        RenameMapKey Key = makeRenameMapKey(ID.Name, N->getType(), ID.S);
        N->setName(makeNameUnique(N->getName()));
        CurModule.RenameMap[Key] = N->getName();
        BB = 0;
      } else {
        BB = cast<BasicBlock>(N);
      }
    }
    break;
  }

  // See if the block has already been defined.
  if (BB) {
    // If this is the definition of the block, make sure the existing value was
    // just a forward reference.  If it was a forward reference, there will be
    // an entry for it in the PlaceHolderInfo map.
    if (isDefinition && !CurFun.BBForwardRefs.erase(BB))
      // The existing value was a definition, not a forward reference.
      error("Redefinition of label " + ID.getName());

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
    CurFun.BBForwardRefs[BB] = std::make_pair(ID, Upgradelineno);
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
    const Type* Ty = LRI->first;
    ValueList &List = LRI->second;
    while (!List.empty()) {
      Value *V = List.back();
      List.pop_back();

      std::map<Value*, std::pair<ValID, int> >::iterator PHI =
        CurModule.PlaceHolderInfo.find(V);
      assert(PHI != CurModule.PlaceHolderInfo.end() && "Placeholder error");

      ValID &DID = PHI->second.first;

      Value *TheRealValue = getExistingValue(Ty, DID);
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
          error("Reference to an invalid definition: '" + DID.getName() +
                "' of type '" + V->getType()->getDescription() + "'",
                PHI->second.second);
            return;
        } else {
          error("Reference to an invalid definition: #" +
                itostr(DID.Num) + " of type '" + 
                V->getType()->getDescription() + "'", PHI->second.second);
          return;
        }
      }
    }
  }

  LateResolvers.clear();
}

/// This function is used for type resolution and upref handling. When a type
/// becomes concrete, this function is called to adjust the signedness for the
/// concrete type.
static void ResolveTypeSign(const Type* oldTy, const Signedness &Sign) {
  std::string TyName = CurModule.CurrentModule->getTypeName(oldTy);
  if (!TyName.empty())
    CurModule.NamedTypeSigns[TyName] = Sign;
}

/// ResolveTypeTo - A brand new type was just declared.  This means that (if
/// name is not null) things referencing Name can be resolved.  Otherwise, 
/// things refering to the number can be resolved.  Do this now.
static void ResolveTypeTo(char *Name, const Type *ToTy, const Signedness& Sign){
  ValID D;
  if (Name)
    D = ValID::create(Name);
  else      
    D = ValID::create((int)CurModule.Types.size());
  D.S.copy(Sign);

  if (Name)
    CurModule.NamedTypeSigns[Name] = Sign;

  std::map<ValID, PATypeHolder>::iterator I =
    CurModule.LateResolveTypes.find(D);
  if (I != CurModule.LateResolveTypes.end()) {
    const Type *OldTy = I->second.get();
    ((DerivedType*)OldTy)->refineAbstractTypeTo(ToTy);
    CurModule.LateResolveTypes.erase(I);
  }
}

/// This is the implementation portion of TypeHasInteger. It traverses the
/// type given, avoiding recursive types, and returns true as soon as it finds
/// an integer type. If no integer type is found, it returns false.
static bool TypeHasIntegerI(const Type *Ty, std::vector<const Type*> Stack) {
  // Handle some easy cases
  if (Ty->isPrimitiveType() || (Ty->getTypeID() == Type::OpaqueTyID))
    return false;
  if (Ty->isInteger())
    return true;
  if (const SequentialType *STy = dyn_cast<SequentialType>(Ty))
    return STy->getElementType()->isInteger();

  // Avoid type structure recursion
  for (std::vector<const Type*>::iterator I = Stack.begin(), E = Stack.end();
       I != E; ++I)
    if (Ty == *I)
      return false;

  // Push us on the type stack
  Stack.push_back(Ty);

  if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
    if (TypeHasIntegerI(FTy->getReturnType(), Stack)) 
      return true;
    FunctionType::param_iterator I = FTy->param_begin();
    FunctionType::param_iterator E = FTy->param_end();
    for (; I != E; ++I)
      if (TypeHasIntegerI(*I, Stack))
        return true;
    return false;
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    StructType::element_iterator I = STy->element_begin();
    StructType::element_iterator E = STy->element_end();
    for (; I != E; ++I) {
      if (TypeHasIntegerI(*I, Stack))
        return true;
    }
    return false;
  }
  // There shouldn't be anything else, but its definitely not integer
  assert(0 && "What type is this?");
  return false;
}

/// This is the interface to TypeHasIntegerI. It just provides the type stack,
/// to avoid recursion, and then calls TypeHasIntegerI.
static inline bool TypeHasInteger(const Type *Ty) {
  std::vector<const Type*> TyStack;
  return TypeHasIntegerI(Ty, TyStack);
}

// setValueName - Set the specified value to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is free'd by this function.
//
static void setValueName(const ValueInfo &V, char *NameStr) {
  if (NameStr) {
    std::string Name(NameStr);      // Copy string
    free(NameStr);                  // Free old string

    if (V.V->getType() == Type::VoidTy) {
      error("Can't assign name '" + Name + "' to value with void type");
      return;
    }

    assert(inFunctionScope() && "Must be in function scope");

    // Search the function's symbol table for an existing value of this name
    ValueSymbolTable &ST = CurFun.CurrentFunction->getValueSymbolTable();
    Value* Existing = ST.lookup(Name);
    if (Existing) {
      // An existing value of the same name was found. This might have happened
      // because of the integer type planes collapsing in LLVM 2.0. 
      if (Existing->getType() == V.V->getType() &&
          !TypeHasInteger(Existing->getType())) {
        // If the type does not contain any integers in them then this can't be
        // a type plane collapsing issue. It truly is a redefinition and we 
        // should error out as the assembly is invalid.
        error("Redefinition of value named '" + Name + "' of type '" +
              V.V->getType()->getDescription() + "'");
        return;
      } 
      // In LLVM 2.0 we don't allow names to be re-used for any values in a 
      // function, regardless of Type. Previously re-use of names was okay as 
      // long as they were distinct types. With type planes collapsing because
      // of the signedness change and because of PR411, this can no longer be
      // supported. We must search the entire symbol table for a conflicting
      // name and make the name unique. No warning is needed as this can't 
      // cause a problem.
      std::string NewName = makeNameUnique(Name);
      // We're changing the name but it will probably be used by other 
      // instructions as operands later on. Consequently we have to retain
      // a mapping of the renaming that we're doing.
      RenameMapKey Key = makeRenameMapKey(Name, V.V->getType(), V.S);
      CurFun.RenameMap[Key] = NewName;
      Name = NewName;
    }

    // Set the name.
    V.V->setName(Name);
  }
}

/// ParseGlobalVariable - Handle parsing of a global.  If Initializer is null,
/// this is a declaration, otherwise it is a definition.
static GlobalVariable *
ParseGlobalVariable(char *NameStr,GlobalValue::LinkageTypes Linkage,
                    bool isConstantGlobal, const Type *Ty,
                    Constant *Initializer,
                    const Signedness &Sign) {
  if (isa<FunctionType>(Ty))
    error("Cannot declare global vars of function type");

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
  ID.S.makeComposite(Sign);

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
  // of this global in the module and emit warnings if there are conflicts.
  if (!Name.empty()) {
    // The global has a name. See if there's an existing one of the same name.
    if (CurModule.CurrentModule->getNamedGlobal(Name) ||
        CurModule.CurrentModule->getFunction(Name)) {
      // We found an existing global of the same name. This isn't allowed 
      // in LLVM 2.0. Consequently, we must alter the name of the global so it
      // can at least compile. This can happen because of type planes 
      // There is alread a global of the same name which means there is a
      // conflict. Let's see what we can do about it.
      std::string NewName(makeNameUnique(Name));
      if (Linkage != GlobalValue::InternalLinkage) {
        // The linkage of this gval is external so we can't reliably rename 
        // it because it could potentially create a linking problem.  
        // However, we can't leave the name conflict in the output either or 
        // it won't assemble with LLVM 2.0.  So, all we can do is rename 
        // this one to something unique and emit a warning about the problem.
        warning("Renaming global variable '" + Name + "' to '" + NewName + 
                  "' may cause linkage errors");
      }

      // Put the renaming in the global rename map
      RenameMapKey Key = makeRenameMapKey(Name, PointerType::get(Ty), ID.S);
      CurModule.RenameMap[Key] = NewName;

      // Rename it
      Name = NewName;
    }
  }

  // Otherwise there is no existing GV to use, create one now.
  GlobalVariable *GV =
    new GlobalVariable(Ty, isConstantGlobal, Linkage, Initializer, Name,
                       CurModule.CurrentModule);
  InsertValue(GV, CurModule.Values);
  // Remember the sign of this global.
  CurModule.NamedValueSigns[Name] = ID.S;
  return GV;
}

// setTypeName - Set the specified type to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is freed by this function.
//
// This function returns true if the type has already been defined, but is
// allowed to be redefined in the specified context.  If the name is a new name
// for the type plane, it is inserted and false is returned.
static bool setTypeName(const PATypeInfo& TI, char *NameStr) {
  assert(!inFunctionScope() && "Can't give types function-local names");
  if (NameStr == 0) return false;
 
  std::string Name(NameStr);      // Copy string
  free(NameStr);                  // Free old string

  const Type* Ty = TI.PAT->get();

  // We don't allow assigning names to void type
  if (Ty == Type::VoidTy) {
    error("Can't assign name '" + Name + "' to the void type");
    return false;
  }

  // Set the type name, checking for conflicts as we do so.
  bool AlreadyExists = CurModule.CurrentModule->addTypeName(Name, Ty);

  // Save the sign information for later use 
  CurModule.NamedTypeSigns[Name] = TI.S;

  if (AlreadyExists) {   // Inserting a name that is already defined???
    const Type *Existing = CurModule.CurrentModule->getTypeByName(Name);
    assert(Existing && "Conflict but no matching type?");

    // There is only one case where this is allowed: when we are refining an
    // opaque type.  In this case, Existing will be an opaque type.
    if (const OpaqueType *OpTy = dyn_cast<OpaqueType>(Existing)) {
      // We ARE replacing an opaque type!
      const_cast<OpaqueType*>(OpTy)->refineAbstractTypeTo(Ty);
      return true;
    }

    // Otherwise, this is an attempt to redefine a type. That's okay if
    // the redefinition is identical to the original. This will be so if
    // Existing and T point to the same Type object. In this one case we
    // allow the equivalent redefinition.
    if (Existing == Ty) return true;  // Yes, it's equal.

    // Any other kind of (non-equivalent) redefinition is an error.
    error("Redefinition of type named '" + Name + "' in the '" +
          Ty->getDescription() + "' type plane");
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
      : NestingLevel(NL), LastContainedTy(URTy), UpRefTy(URTy) { }
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
static PATypeHolder HandleUpRefs(const Type *ty, const Signedness& Sign) {
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

  unsigned i = 0;
  for (; i != UpRefs.size(); ++i) {
    UR_OUT("  UR#" << i << " - TypeContains(" << Ty->getDescription() << ", "
           << UpRefs[i].UpRefTy->getDescription() << ") = "
           << (TypeContains(Ty, UpRefs[i].UpRefTy) ? "true" : "false") << "\n");
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
                 << UpRefs[i].UpRefTy->getDescription() << "\n";
          std::string OldName = UpRefs[i].UpRefTy->getDescription());
          ResolveTypeSign(UpRefs[i].UpRefTy, Sign);
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
           << UpRefs[i].UpRefTy->getDescription() << "\n";
           std::string OldName = TypeToResolve->getDescription());
    ResolveTypeSign(TypeToResolve, Sign);
    TypeToResolve->refineAbstractTypeTo(Ty);
  }

  return Ty;
}

bool Signedness::operator<(const Signedness &that) const {
  if (isNamed()) {
    if (that.isNamed()) 
      return *(this->name) < *(that.name);
    else
      return CurModule.NamedTypeSigns[*name] < that;
  } else if (that.isNamed()) {
    return *this < CurModule.NamedTypeSigns[*that.name];
  }

  if (isComposite() && that.isComposite()) {
    if (sv->size() == that.sv->size()) {
      SignVector::const_iterator thisI = sv->begin(), thisE = sv->end();
      SignVector::const_iterator thatI = that.sv->begin(), 
                                 thatE = that.sv->end();
      for (; thisI != thisE; ++thisI, ++thatI) {
        if (*thisI < *thatI)
          return true;
        else if (!(*thisI == *thatI))
          return false;
      }
      return false;
    }
    return sv->size() < that.sv->size();
  }  
  return kind < that.kind;
}

bool Signedness::operator==(const Signedness &that) const {
  if (isNamed())
    if (that.isNamed())
      return *(this->name) == *(that.name);
    else 
      return CurModule.NamedTypeSigns[*(this->name)] == that;
  else if (that.isNamed())
    return *this == CurModule.NamedTypeSigns[*(that.name)];
  if (isComposite() && that.isComposite()) {
    if (sv->size() == that.sv->size()) {
      SignVector::const_iterator thisI = sv->begin(), thisE = sv->end();
      SignVector::const_iterator thatI = that.sv->begin(), 
                                 thatE = that.sv->end();
      for (; thisI != thisE; ++thisI, ++thatI) {
        if (!(*thisI == *thatI))
          return false;
      }
      return true;
    }
    return false;
  }
  return kind == that.kind;
}

void Signedness::copy(const Signedness &that) {
  if (that.isNamed()) {
    kind = Named;
    name = new std::string(*that.name);
  } else if (that.isComposite()) {
    kind = Composite;
    sv = new SignVector();
    *sv = *that.sv;
  } else {
    kind = that.kind;
    sv = 0;
  }
}

void Signedness::destroy() {
  if (isNamed()) {
    delete name;
  } else if (isComposite()) {
    delete sv;
  } 
}

#ifndef NDEBUG
void Signedness::dump() const {
  if (isComposite()) {
    if (sv->size() == 1) {
      (*sv)[0].dump();
      std::cerr << "*";
    } else {
      std::cerr << "{ " ;
      for (unsigned i = 0; i < sv->size(); ++i) {
        if (i != 0)
          std::cerr << ", ";
        (*sv)[i].dump();
      }
      std::cerr << "} " ;
    }
  } else if (isNamed()) {
    std::cerr << *name;
  } else if (isSigned()) {
    std::cerr << "S";
  } else if (isUnsigned()) {
    std::cerr << "U";
  } else
    std::cerr << ".";
}
#endif

static inline Instruction::TermOps 
getTermOp(TermOps op) {
  switch (op) {
    default           : assert(0 && "Invalid OldTermOp");
    case RetOp        : return Instruction::Ret;
    case BrOp         : return Instruction::Br;
    case SwitchOp     : return Instruction::Switch;
    case InvokeOp     : return Instruction::Invoke;
    case UnwindOp     : return Instruction::Unwind;
    case UnreachableOp: return Instruction::Unreachable;
  }
}

static inline Instruction::BinaryOps 
getBinaryOp(BinaryOps op, const Type *Ty, const Signedness& Sign) {
  switch (op) {
    default     : assert(0 && "Invalid OldBinaryOps");
    case SetEQ  : 
    case SetNE  : 
    case SetLE  :
    case SetGE  :
    case SetLT  :
    case SetGT  : assert(0 && "Should use getCompareOp");
    case AddOp  : return Instruction::Add;
    case SubOp  : return Instruction::Sub;
    case MulOp  : return Instruction::Mul;
    case DivOp  : {
      // This is an obsolete instruction so we must upgrade it based on the
      // types of its operands.
      bool isFP = Ty->isFloatingPoint();
      if (const VectorType* PTy = dyn_cast<VectorType>(Ty))
        // If its a vector type we want to use the element type
        isFP = PTy->getElementType()->isFloatingPoint();
      if (isFP)
        return Instruction::FDiv;
      else if (Sign.isSigned())
        return Instruction::SDiv;
      return Instruction::UDiv;
    }
    case UDivOp : return Instruction::UDiv;
    case SDivOp : return Instruction::SDiv;
    case FDivOp : return Instruction::FDiv;
    case RemOp  : {
      // This is an obsolete instruction so we must upgrade it based on the
      // types of its operands.
      bool isFP = Ty->isFloatingPoint();
      if (const VectorType* PTy = dyn_cast<VectorType>(Ty))
        // If its a vector type we want to use the element type
        isFP = PTy->getElementType()->isFloatingPoint();
      // Select correct opcode
      if (isFP)
        return Instruction::FRem;
      else if (Sign.isSigned())
        return Instruction::SRem;
      return Instruction::URem;
    }
    case URemOp : return Instruction::URem;
    case SRemOp : return Instruction::SRem;
    case FRemOp : return Instruction::FRem;
    case LShrOp : return Instruction::LShr;
    case AShrOp : return Instruction::AShr;
    case ShlOp  : return Instruction::Shl;
    case ShrOp  : 
      if (Sign.isSigned())
        return Instruction::AShr;
      return Instruction::LShr;
    case AndOp  : return Instruction::And;
    case OrOp   : return Instruction::Or;
    case XorOp  : return Instruction::Xor;
  }
}

static inline Instruction::OtherOps 
getCompareOp(BinaryOps op, unsigned short &predicate, const Type* &Ty,
             const Signedness &Sign) {
  bool isSigned = Sign.isSigned();
  bool isFP = Ty->isFloatingPoint();
  switch (op) {
    default     : assert(0 && "Invalid OldSetCC");
    case SetEQ  : 
      if (isFP) {
        predicate = FCmpInst::FCMP_OEQ;
        return Instruction::FCmp;
      } else {
        predicate = ICmpInst::ICMP_EQ;
        return Instruction::ICmp;
      }
    case SetNE  : 
      if (isFP) {
        predicate = FCmpInst::FCMP_UNE;
        return Instruction::FCmp;
      } else {
        predicate = ICmpInst::ICMP_NE;
        return Instruction::ICmp;
      }
    case SetLE  : 
      if (isFP) {
        predicate = FCmpInst::FCMP_OLE;
        return Instruction::FCmp;
      } else {
        if (isSigned)
          predicate = ICmpInst::ICMP_SLE;
        else
          predicate = ICmpInst::ICMP_ULE;
        return Instruction::ICmp;
      }
    case SetGE  : 
      if (isFP) {
        predicate = FCmpInst::FCMP_OGE;
        return Instruction::FCmp;
      } else {
        if (isSigned)
          predicate = ICmpInst::ICMP_SGE;
        else
          predicate = ICmpInst::ICMP_UGE;
        return Instruction::ICmp;
      }
    case SetLT  : 
      if (isFP) {
        predicate = FCmpInst::FCMP_OLT;
        return Instruction::FCmp;
      } else {
        if (isSigned)
          predicate = ICmpInst::ICMP_SLT;
        else
          predicate = ICmpInst::ICMP_ULT;
        return Instruction::ICmp;
      }
    case SetGT  : 
      if (isFP) {
        predicate = FCmpInst::FCMP_OGT;
        return Instruction::FCmp;
      } else {
        if (isSigned)
          predicate = ICmpInst::ICMP_SGT;
        else
          predicate = ICmpInst::ICMP_UGT;
        return Instruction::ICmp;
      }
  }
}

static inline Instruction::MemoryOps getMemoryOp(MemoryOps op) {
  switch (op) {
    default              : assert(0 && "Invalid OldMemoryOps");
    case MallocOp        : return Instruction::Malloc;
    case FreeOp          : return Instruction::Free;
    case AllocaOp        : return Instruction::Alloca;
    case LoadOp          : return Instruction::Load;
    case StoreOp         : return Instruction::Store;
    case GetElementPtrOp : return Instruction::GetElementPtr;
  }
}

static inline Instruction::OtherOps 
getOtherOp(OtherOps op, const Signedness &Sign) {
  switch (op) {
    default               : assert(0 && "Invalid OldOtherOps");
    case PHIOp            : return Instruction::PHI;
    case CallOp           : return Instruction::Call;
    case SelectOp         : return Instruction::Select;
    case UserOp1          : return Instruction::UserOp1;
    case UserOp2          : return Instruction::UserOp2;
    case VAArg            : return Instruction::VAArg;
    case ExtractElementOp : return Instruction::ExtractElement;
    case InsertElementOp  : return Instruction::InsertElement;
    case ShuffleVectorOp  : return Instruction::ShuffleVector;
    case ICmpOp           : return Instruction::ICmp;
    case FCmpOp           : return Instruction::FCmp;
  };
}

static inline Value*
getCast(CastOps op, Value *Src, const Signedness &SrcSign, const Type *DstTy, 
        const Signedness &DstSign, bool ForceInstruction = false) {
  Instruction::CastOps Opcode;
  const Type* SrcTy = Src->getType();
  if (op == CastOp) {
    if (SrcTy->isFloatingPoint() && isa<PointerType>(DstTy)) {
      // fp -> ptr cast is no longer supported but we must upgrade this
      // by doing a double cast: fp -> int -> ptr
      SrcTy = Type::Int64Ty;
      Opcode = Instruction::IntToPtr;
      if (isa<Constant>(Src)) {
        Src = ConstantExpr::getCast(Instruction::FPToUI, 
                                     cast<Constant>(Src), SrcTy);
      } else {
        std::string NewName(makeNameUnique(Src->getName()));
        Src = new FPToUIInst(Src, SrcTy, NewName, CurBB);
      }
    } else if (isa<IntegerType>(DstTy) &&
               cast<IntegerType>(DstTy)->getBitWidth() == 1) {
      // cast type %x to bool was previously defined as setne type %x, null
      // The cast semantic is now to truncate, not compare so we must retain
      // the original intent by replacing the cast with a setne
      Constant* Null = Constant::getNullValue(SrcTy);
      Instruction::OtherOps Opcode = Instruction::ICmp;
      unsigned short predicate = ICmpInst::ICMP_NE;
      if (SrcTy->isFloatingPoint()) {
        Opcode = Instruction::FCmp;
        predicate = FCmpInst::FCMP_ONE;
      } else if (!SrcTy->isInteger() && !isa<PointerType>(SrcTy)) {
        error("Invalid cast to bool");
      }
      if (isa<Constant>(Src) && !ForceInstruction)
        return ConstantExpr::getCompare(predicate, cast<Constant>(Src), Null);
      else
        return CmpInst::create(Opcode, predicate, Src, Null);
    }
    // Determine the opcode to use by calling CastInst::getCastOpcode
    Opcode = 
      CastInst::getCastOpcode(Src, SrcSign.isSigned(), DstTy, 
                              DstSign.isSigned());

  } else switch (op) {
    default: assert(0 && "Invalid cast token");
    case TruncOp:    Opcode = Instruction::Trunc; break;
    case ZExtOp:     Opcode = Instruction::ZExt; break;
    case SExtOp:     Opcode = Instruction::SExt; break;
    case FPTruncOp:  Opcode = Instruction::FPTrunc; break;
    case FPExtOp:    Opcode = Instruction::FPExt; break;
    case FPToUIOp:   Opcode = Instruction::FPToUI; break;
    case FPToSIOp:   Opcode = Instruction::FPToSI; break;
    case UIToFPOp:   Opcode = Instruction::UIToFP; break;
    case SIToFPOp:   Opcode = Instruction::SIToFP; break;
    case PtrToIntOp: Opcode = Instruction::PtrToInt; break;
    case IntToPtrOp: Opcode = Instruction::IntToPtr; break;
    case BitCastOp:  Opcode = Instruction::BitCast; break;
  }

  if (isa<Constant>(Src) && !ForceInstruction)
    return ConstantExpr::getCast(Opcode, cast<Constant>(Src), DstTy);
  return CastInst::create(Opcode, Src, DstTy);
}

static Instruction *
upgradeIntrinsicCall(const Type* RetTy, const ValID &ID, 
                     std::vector<Value*>& Args) {

  std::string Name = ID.Type == ValID::NameVal ? ID.Name : "";
  if (Name.length() <= 5 || Name[0] != 'l' || Name[1] != 'l' || 
      Name[2] != 'v' || Name[3] != 'm' || Name[4] != '.')
    return 0;

  switch (Name[5]) {
    case 'i':
      if (Name == "llvm.isunordered.f32" || Name == "llvm.isunordered.f64") {
        if (Args.size() != 2)
          error("Invalid prototype for " + Name);
        return new FCmpInst(FCmpInst::FCMP_UNO, Args[0], Args[1]);
      }
      break;
    case 'b':
      if (Name.length() == 14 && !memcmp(&Name[5], "bswap.i", 7)) {
        const Type* ArgTy = Args[0]->getType();
        Name += ".i" + utostr(cast<IntegerType>(ArgTy)->getBitWidth());
        Function *F = cast<Function>(
          CurModule.CurrentModule->getOrInsertFunction(Name, RetTy, ArgTy, 
                                                       (void*)0));
        return new CallInst(F, Args[0]);
      }
      break;
    case 'c':
      if ((Name.length() <= 14 && !memcmp(&Name[5], "ctpop.i", 7)) ||
          (Name.length() <= 13 && !memcmp(&Name[5], "ctlz.i", 6)) ||
          (Name.length() <= 13 && !memcmp(&Name[5], "cttz.i", 6))) {
        // These intrinsics changed their result type.
        const Type* ArgTy = Args[0]->getType();
        Function *OldF = CurModule.CurrentModule->getFunction(Name);
        if (OldF)
          OldF->setName("upgrd.rm." + Name);

        Function *NewF = cast<Function>(
          CurModule.CurrentModule->getOrInsertFunction(Name, Type::Int32Ty, 
                                                       ArgTy, (void*)0));

        Instruction *Call = new CallInst(NewF, Args[0], "", CurBB);
        return CastInst::createIntegerCast(Call, RetTy, false);
      }
      break;

    case 'v' : {
      const Type* PtrTy = PointerType::get(Type::Int8Ty);
      std::vector<const Type*> Params;
      if (Name == "llvm.va_start" || Name == "llvm.va_end") {
        if (Args.size() != 1)
          error("Invalid prototype for " + Name + " prototype");
        Params.push_back(PtrTy);
        const FunctionType *FTy = 
          FunctionType::get(Type::VoidTy, Params, false);
        const PointerType *PFTy = PointerType::get(FTy);
        Value* Func = getVal(PFTy, ID);
        Args[0] = new BitCastInst(Args[0], PtrTy, makeNameUnique("va"), CurBB);
        return new CallInst(Func, &Args[0], Args.size());
      } else if (Name == "llvm.va_copy") {
        if (Args.size() != 2)
          error("Invalid prototype for " + Name + " prototype");
        Params.push_back(PtrTy);
        Params.push_back(PtrTy);
        const FunctionType *FTy = 
          FunctionType::get(Type::VoidTy, Params, false);
        const PointerType *PFTy = PointerType::get(FTy);
        Value* Func = getVal(PFTy, ID);
        std::string InstName0(makeNameUnique("va0"));
        std::string InstName1(makeNameUnique("va1"));
        Args[0] = new BitCastInst(Args[0], PtrTy, InstName0, CurBB);
        Args[1] = new BitCastInst(Args[1], PtrTy, InstName1, CurBB);
        return new CallInst(Func, &Args[0], Args.size());
      }
    }
  }
  return 0;
}

const Type* upgradeGEPCEIndices(const Type* PTy, 
                                std::vector<ValueInfo> *Indices, 
                                std::vector<Constant*> &Result) {
  const Type *Ty = PTy;
  Result.clear();
  for (unsigned i = 0, e = Indices->size(); i != e ; ++i) {
    Constant *Index = cast<Constant>((*Indices)[i].V);

    if (ConstantInt *CI = dyn_cast<ConstantInt>(Index)) {
      // LLVM 1.2 and earlier used ubyte struct indices.  Convert any ubyte 
      // struct indices to i32 struct indices with ZExt for compatibility.
      if (CI->getBitWidth() < 32)
        Index = ConstantExpr::getCast(Instruction::ZExt, CI, Type::Int32Ty);
    }
    
    if (isa<SequentialType>(Ty)) {
      // Make sure that unsigned SequentialType indices are zext'd to 
      // 64-bits if they were smaller than that because LLVM 2.0 will sext 
      // all indices for SequentialType elements. We must retain the same 
      // semantic (zext) for unsigned types.
      if (const IntegerType *Ity = dyn_cast<IntegerType>(Index->getType())) {
        if (Ity->getBitWidth() < 64 && (*Indices)[i].S.isUnsigned()) {
          Index = ConstantExpr::getCast(Instruction::ZExt, Index,Type::Int64Ty);
        }
      }
    }
    Result.push_back(Index);
    Ty = GetElementPtrInst::getIndexedType(PTy, (Value**)&Result[0], 
                                           Result.size(),true);
    if (!Ty)
      error("Index list invalid for constant getelementptr");
  }
  return Ty;
}

const Type* upgradeGEPInstIndices(const Type* PTy, 
                                  std::vector<ValueInfo> *Indices, 
                                  std::vector<Value*>    &Result) {
  const Type *Ty = PTy;
  Result.clear();
  for (unsigned i = 0, e = Indices->size(); i != e ; ++i) {
    Value *Index = (*Indices)[i].V;

    if (ConstantInt *CI = dyn_cast<ConstantInt>(Index)) {
      // LLVM 1.2 and earlier used ubyte struct indices.  Convert any ubyte 
      // struct indices to i32 struct indices with ZExt for compatibility.
      if (CI->getBitWidth() < 32)
        Index = ConstantExpr::getCast(Instruction::ZExt, CI, Type::Int32Ty);
    }
    

    if (isa<StructType>(Ty)) {        // Only change struct indices
      if (!isa<Constant>(Index)) {
        error("Invalid non-constant structure index");
        return 0;
      }
    } else {
      // Make sure that unsigned SequentialType indices are zext'd to 
      // 64-bits if they were smaller than that because LLVM 2.0 will sext 
      // all indices for SequentialType elements. We must retain the same 
      // semantic (zext) for unsigned types.
      if (const IntegerType *Ity = dyn_cast<IntegerType>(Index->getType())) {
        if (Ity->getBitWidth() < 64 && (*Indices)[i].S.isUnsigned()) {
          if (isa<Constant>(Index))
            Index = ConstantExpr::getCast(Instruction::ZExt, 
              cast<Constant>(Index), Type::Int64Ty);
          else
            Index = CastInst::create(Instruction::ZExt, Index, Type::Int64Ty,
              makeNameUnique("gep"), CurBB);
        }
      }
    }
    Result.push_back(Index);
    Ty = GetElementPtrInst::getIndexedType(PTy, &Result[0], Result.size(),true);
    if (!Ty)
      error("Index list invalid for constant getelementptr");
  }
  return Ty;
}

unsigned upgradeCallingConv(unsigned CC) {
  switch (CC) {
    case OldCallingConv::C           : return CallingConv::C;
    case OldCallingConv::CSRet       : return CallingConv::C;
    case OldCallingConv::Fast        : return CallingConv::Fast;
    case OldCallingConv::Cold        : return CallingConv::Cold;
    case OldCallingConv::X86_StdCall : return CallingConv::X86_StdCall;
    case OldCallingConv::X86_FastCall: return CallingConv::X86_FastCall;
    default:
      return CC;
  }
}

Module* UpgradeAssembly(const std::string &infile, std::istream& in, 
                              bool debug, bool addAttrs)
{
  Upgradelineno = 1; 
  CurFilename = infile;
  LexInput = &in;
  yydebug = debug;
  AddAttributes = addAttrs;
  ObsoleteVarArgs = false;
  NewVarArgs = false;

  CurModule.CurrentModule = new Module(CurFilename);

  // Check to make sure the parser succeeded
  if (yyparse()) {
    if (ParserResult)
      delete ParserResult;
    std::cerr << "llvm-upgrade: parse failed.\n";
    return 0;
  }

  // Check to make sure that parsing produced a result
  if (!ParserResult) {
    std::cerr << "llvm-upgrade: no parse result.\n";
    return 0;
  }

  // Reset ParserResult variable while saving its value for the result.
  Module *Result = ParserResult;
  ParserResult = 0;

  //Not all functions use vaarg, so make a second check for ObsoleteVarArgs
  {
    Function* F;
    if ((F = Result->getFunction("llvm.va_start"))
        && F->getFunctionType()->getNumParams() == 0)
      ObsoleteVarArgs = true;
    if((F = Result->getFunction("llvm.va_copy"))
       && F->getFunctionType()->getNumParams() == 1)
      ObsoleteVarArgs = true;
  }

  if (ObsoleteVarArgs && NewVarArgs) {
    error("This file is corrupt: it uses both new and old style varargs");
    return 0;
  }

  if(ObsoleteVarArgs) {
    if(Function* F = Result->getFunction("llvm.va_start")) {
      if (F->arg_size() != 0) {
        error("Obsolete va_start takes 0 argument");
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
      Function* NF = cast<Function>(Result->getOrInsertFunction(
        "llvm.va_start", RetTy, ArgTyPtr, (Type *)0));

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
    
    if(Function* F = Result->getFunction("llvm.va_end")) {
      if(F->arg_size() != 1) {
        error("Obsolete va_end takes 1 argument");
        return 0;
      }

      //vaend foo
      // ->
      //bar = alloca 1 of typeof(foo)
      //vaend bar
      const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
      const Type* ArgTy = F->getFunctionType()->getParamType(0);
      const Type* ArgTyPtr = PointerType::get(ArgTy);
      Function* NF = cast<Function>(Result->getOrInsertFunction(
        "llvm.va_end", RetTy, ArgTyPtr, (Type *)0));

      while (!F->use_empty()) {
        CallInst* CI = cast<CallInst>(F->use_back());
        AllocaInst* bar = new AllocaInst(ArgTy, 0, "vaend.fix.1", CI);
        new StoreInst(CI->getOperand(1), bar, CI);
        new CallInst(NF, bar, "", CI);
        CI->getParent()->getInstList().erase(CI);
      }
      Result->getFunctionList().erase(F);
    }

    if(Function* F = Result->getFunction("llvm.va_copy")) {
      if(F->arg_size() != 1) {
        error("Obsolete va_copy takes 1 argument");
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
      Function* NF = cast<Function>(Result->getOrInsertFunction(
        "llvm.va_copy", RetTy, ArgTyPtr, ArgTyPtr, (Type *)0));

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

} // end llvm namespace

using namespace llvm;

%}

%union {
  llvm::Module                           *ModuleVal;
  llvm::Function                         *FunctionVal;
  std::pair<llvm::PATypeInfo, char*>     *ArgVal;
  llvm::BasicBlock                       *BasicBlockVal;
  llvm::TermInstInfo                     TermInstVal;
  llvm::InstrInfo                        InstVal;
  llvm::ConstInfo                        ConstVal;
  llvm::ValueInfo                        ValueVal;
  llvm::PATypeInfo                       TypeVal;
  llvm::TypeInfo                         PrimType;
  llvm::PHIListInfo                      PHIList;
  std::list<llvm::PATypeInfo>            *TypeList;
  std::vector<llvm::ValueInfo>           *ValueList;
  std::vector<llvm::ConstInfo>           *ConstVector;


  std::vector<std::pair<llvm::PATypeInfo,char*> > *ArgList;
  // Represent the RHS of PHI node
  std::vector<std::pair<llvm::Constant*, llvm::BasicBlock*> > *JumpTable;

  llvm::GlobalValue::LinkageTypes         Linkage;
  int64_t                           SInt64Val;
  uint64_t                          UInt64Val;
  int                               SIntVal;
  unsigned                          UIntVal;
  double                            FPVal;
  bool                              BoolVal;

  char                             *StrVal;   // This memory is strdup'd!
  llvm::ValID                       ValIDVal; // strdup'd memory maybe!

  llvm::BinaryOps                   BinaryOpVal;
  llvm::TermOps                     TermOpVal;
  llvm::MemoryOps                   MemOpVal;
  llvm::OtherOps                    OtherOpVal;
  llvm::CastOps                     CastOpVal;
  llvm::ICmpInst::Predicate         IPred;
  llvm::FCmpInst::Predicate         FPred;
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
%type <Linkage>       OptLinkage FnDeclareLinkage
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
%type  <PrimType> SIntType UIntType IntType FPType PrimType // Classifications
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
%token <TermOpVal> RET BR SWITCH INVOKE UNREACHABLE
%token UNWIND EXCEPT

// Binary Operators
%type  <BinaryOpVal> ArithmeticOps LogicalOps SetCondOps // Binops Subcatagories
%type  <BinaryOpVal> ShiftOps
%token <BinaryOpVal> ADD SUB MUL DIV UDIV SDIV FDIV REM UREM SREM FREM 
%token <BinaryOpVal> AND OR XOR SHL SHR ASHR LSHR 
%token <BinaryOpVal> SETLE SETGE SETLT SETGT SETEQ SETNE  // Binary Comparators
%token <OtherOpVal> ICMP FCMP

// Memory Instructions
%token <MemOpVal> MALLOC ALLOCA FREE LOAD STORE GETELEMENTPTR

// Other Operators
%token <OtherOpVal> PHI_TOK SELECT VAARG
%token <OtherOpVal> EXTRACTELEMENT INSERTELEMENT SHUFFLEVECTOR
%token VAARG_old VANEXT_old //OBSOLETE

// Support for ICmp/FCmp Predicates, which is 1.9++ but not 2.0
%type  <IPred> IPredicates
%type  <FPred> FPredicates
%token  EQ NE SLT SGT SLE SGE ULT UGT ULE UGE 
%token  OEQ ONE OLT OGT OLE OGE ORD UNO UEQ UNE

%token <CastOpVal> CAST TRUNC ZEXT SEXT FPTRUNC FPEXT FPTOUI FPTOSI 
%token <CastOpVal> UITOFP SITOFP PTRTOINT INTTOPTR BITCAST 
%type  <CastOpVal> CastOps

%start Module

%%

// Handle constant integer size restriction and conversion...
//
INTVAL 
  : SINTVAL
  | UINTVAL {
    if ($1 > (uint32_t)INT32_MAX)     // Outside of my range!
      error("Value too large for type");
    $$ = (int32_t)$1;
  }
  ;

EINT64VAL 
  : ESINT64VAL       // These have same type and can't cause problems...
  | EUINT64VAL {
    if ($1 > (uint64_t)INT64_MAX)     // Outside of my range!
      error("Value too large for type");
    $$ = (int64_t)$1;
  };

// Operations that are notably excluded from this list include:
// RET, BR, & SWITCH because they end basic blocks and are treated specially.
//
ArithmeticOps
  : ADD | SUB | MUL | DIV | UDIV | SDIV | FDIV | REM | UREM | SREM | FREM
  ;

LogicalOps   
  : AND | OR | XOR
  ;

SetCondOps   
  : SETLE | SETGE | SETLT | SETGT | SETEQ | SETNE
  ;

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
ShiftOps  
  : SHL | SHR | ASHR | LSHR
  ;

CastOps      
  : TRUNC | ZEXT | SEXT | FPTRUNC | FPEXT | FPTOUI | FPTOSI 
  | UITOFP | SITOFP | PTRTOINT | INTTOPTR | BITCAST | CAST
  ;

// These are some types that allow classification if we only want a particular 
// thing... for example, only a signed, unsigned, or integral type.
SIntType 
  :  LONG |  INT |  SHORT | SBYTE
  ;

UIntType 
  : ULONG | UINT | USHORT | UBYTE
  ;

IntType  
  : SIntType | UIntType
  ;

FPType   
  : FLOAT | DOUBLE
  ;

// OptAssign - Value producing statements have an optional assignment component
OptAssign 
  : Name '=' {
    $$ = $1;
  }
  | /*empty*/ {
    $$ = 0;
  };

OptLinkage 
  : INTERNAL    { $$ = GlobalValue::InternalLinkage; }
  | LINKONCE    { $$ = GlobalValue::LinkOnceLinkage; } 
  | WEAK        { $$ = GlobalValue::WeakLinkage; } 
  | APPENDING   { $$ = GlobalValue::AppendingLinkage; } 
  | DLLIMPORT   { $$ = GlobalValue::DLLImportLinkage; } 
  | DLLEXPORT   { $$ = GlobalValue::DLLExportLinkage; } 
  | EXTERN_WEAK { $$ = GlobalValue::ExternalWeakLinkage; }
  | /*empty*/   { $$ = GlobalValue::ExternalLinkage; }
  ;

OptCallingConv 
  : /*empty*/          { $$ = lastCallingConv = OldCallingConv::C; } 
  | CCC_TOK            { $$ = lastCallingConv = OldCallingConv::C; } 
  | CSRETCC_TOK        { $$ = lastCallingConv = OldCallingConv::CSRet; } 
  | FASTCC_TOK         { $$ = lastCallingConv = OldCallingConv::Fast; } 
  | COLDCC_TOK         { $$ = lastCallingConv = OldCallingConv::Cold; } 
  | X86_STDCALLCC_TOK  { $$ = lastCallingConv = OldCallingConv::X86_StdCall; } 
  | X86_FASTCALLCC_TOK { $$ = lastCallingConv = OldCallingConv::X86_FastCall; } 
  | CC_TOK EUINT64VAL  {
    if ((unsigned)$2 != $2)
      error("Calling conv too large");
    $$ = lastCallingConv = $2;
  }
  ;

// OptAlign/OptCAlign - An optional alignment, and an optional alignment with
// a comma before it.
OptAlign 
  : /*empty*/        { $$ = 0; } 
  | ALIGN EUINT64VAL {
    $$ = $2;
    if ($$ != 0 && !isPowerOf2_32($$))
      error("Alignment must be a power of two");
  }
  ;

OptCAlign 
  : /*empty*/ { $$ = 0; } 
  | ',' ALIGN EUINT64VAL {
    $$ = $3;
    if ($$ != 0 && !isPowerOf2_32($$))
      error("Alignment must be a power of two");
  }
  ;

SectionString 
  : SECTION STRINGCONSTANT {
    for (unsigned i = 0, e = strlen($2); i != e; ++i)
      if ($2[i] == '"' || $2[i] == '\\')
        error("Invalid character in section name");
    $$ = $2;
  }
  ;

OptSection 
  : /*empty*/ { $$ = 0; } 
  | SectionString { $$ = $1; }
  ;

// GlobalVarAttributes - Used to pass the attributes string on a global.  CurGV
// is set to be the global we are processing.
//
GlobalVarAttributes 
  : /* empty */ {} 
  | ',' GlobalVarAttribute GlobalVarAttributes {}
  ;

GlobalVarAttribute
  : SectionString {
    CurGV->setSection($1);
    free($1);
  } 
  | ALIGN EUINT64VAL {
    if ($2 != 0 && !isPowerOf2_32($2))
      error("Alignment must be a power of two");
    CurGV->setAlignment($2);
    
  }
  ;

//===----------------------------------------------------------------------===//
// Types includes all predefined types... except void, because it can only be
// used in specific contexts (function returning void for example).  To have
// access to it, a user must explicitly use TypesV.
//

// TypesV includes all of 'Types', but it also includes the void type.
TypesV    
  : Types
  | VOID { 
    $$.PAT = new PATypeHolder($1.T); 
    $$.S.makeSignless();
  }
  ;

UpRTypesV 
  : UpRTypes 
  | VOID { 
    $$.PAT = new PATypeHolder($1.T); 
    $$.S.makeSignless();
  }
  ;

Types
  : UpRTypes {
    if (!UpRefs.empty())
      error("Invalid upreference in type: " + (*$1.PAT)->getDescription());
    $$ = $1;
  }
  ;

PrimType
  : BOOL | SBYTE | UBYTE | SHORT  | USHORT | INT   | UINT 
  | LONG | ULONG | FLOAT | DOUBLE | LABEL
  ;

// Derived types are added later...
UpRTypes 
  : PrimType { 
    $$.PAT = new PATypeHolder($1.T);
    $$.S.copy($1.S);
  }
  | OPAQUE {
    $$.PAT = new PATypeHolder(OpaqueType::get());
    $$.S.makeSignless();
  }
  | SymbolicValueRef {            // Named types are also simple types...
    $$.S.copy(getTypeSign($1));
    const Type* tmp = getType($1);
    $$.PAT = new PATypeHolder(tmp);
  }
  | '\\' EUINT64VAL {                   // Type UpReference
    if ($2 > (uint64_t)~0U) 
      error("Value out of range");
    OpaqueType *OT = OpaqueType::get();        // Use temporary placeholder
    UpRefs.push_back(UpRefRecord((unsigned)$2, OT));  // Add to vector...
    $$.PAT = new PATypeHolder(OT);
    $$.S.makeSignless();
    UR_OUT("New Upreference!\n");
  }
  | UpRTypesV '(' ArgTypeListI ')' {           // Function derived type?
    $$.S.makeComposite($1.S);
    std::vector<const Type*> Params;
    for (std::list<llvm::PATypeInfo>::iterator I = $3->begin(),
           E = $3->end(); I != E; ++I) {
      Params.push_back(I->PAT->get());
      $$.S.add(I->S);
    }
    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    ParamAttrsList *PAL = 0;
    if (lastCallingConv == OldCallingConv::CSRet) {
      ParamAttrsVector Attrs;
      ParamAttrsWithIndex PAWI;
      PAWI.index = 1;  PAWI.attrs = ParamAttr::StructRet; // first arg
      Attrs.push_back(PAWI);
      PAL = ParamAttrsList::get(Attrs);
    }

    const FunctionType *FTy =
      FunctionType::get($1.PAT->get(), Params, isVarArg, PAL);

    $$.PAT = new PATypeHolder( HandleUpRefs(FTy, $$.S) );
    delete $1.PAT;  // Delete the return type handle
    delete $3;      // Delete the argument list
  }
  | '[' EUINT64VAL 'x' UpRTypes ']' {          // Sized array type?
    $$.S.makeComposite($4.S);
    $$.PAT = new PATypeHolder(HandleUpRefs(ArrayType::get($4.PAT->get(), 
                                           (unsigned)$2), $$.S));
    delete $4.PAT;
  }
  | '<' EUINT64VAL 'x' UpRTypes '>' {          // Vector type?
    const llvm::Type* ElemTy = $4.PAT->get();
    if ((unsigned)$2 != $2)
       error("Unsigned result not equal to signed result");
    if (!(ElemTy->isInteger() || ElemTy->isFloatingPoint()))
       error("Elements of a VectorType must be integer or floating point");
    if (!isPowerOf2_32($2))
      error("VectorType length should be a power of 2");
    $$.S.makeComposite($4.S);
    $$.PAT = new PATypeHolder(HandleUpRefs(VectorType::get(ElemTy, 
                                         (unsigned)$2), $$.S));
    delete $4.PAT;
  }
  | '{' TypeListI '}' {                        // Structure type?
    std::vector<const Type*> Elements;
    $$.S.makeComposite();
    for (std::list<llvm::PATypeInfo>::iterator I = $2->begin(),
           E = $2->end(); I != E; ++I) {
      Elements.push_back(I->PAT->get());
      $$.S.add(I->S);
    }
    $$.PAT = new PATypeHolder(HandleUpRefs(StructType::get(Elements), $$.S));
    delete $2;
  }
  | '{' '}' {                                  // Empty structure type?
    $$.PAT = new PATypeHolder(StructType::get(std::vector<const Type*>()));
    $$.S.makeComposite();
  }
  | '<' '{' TypeListI '}' '>' {                // Packed Structure type?
    $$.S.makeComposite();
    std::vector<const Type*> Elements;
    for (std::list<llvm::PATypeInfo>::iterator I = $3->begin(),
           E = $3->end(); I != E; ++I) {
      Elements.push_back(I->PAT->get());
      $$.S.add(I->S);
      delete I->PAT;
    }
    $$.PAT = new PATypeHolder(HandleUpRefs(StructType::get(Elements, true), 
                                           $$.S));
    delete $3;
  }
  | '<' '{' '}' '>' {                          // Empty packed structure type?
    $$.PAT = new PATypeHolder(StructType::get(std::vector<const Type*>(),true));
    $$.S.makeComposite();
  }
  | UpRTypes '*' {                             // Pointer type?
    if ($1.PAT->get() == Type::LabelTy)
      error("Cannot form a pointer to a basic block");
    $$.S.makeComposite($1.S);
    $$.PAT = new PATypeHolder(HandleUpRefs(PointerType::get($1.PAT->get()),
                                           $$.S));
    delete $1.PAT;
  }
  ;

// TypeList - Used for struct declarations and as a basis for function type 
// declaration type lists
//
TypeListI 
  : UpRTypes {
    $$ = new std::list<PATypeInfo>();
    $$->push_back($1); 
  }
  | TypeListI ',' UpRTypes {
    ($$=$1)->push_back($3);
  }
  ;

// ArgTypeList - List of types for a function type declaration...
ArgTypeListI 
  : TypeListI
  | TypeListI ',' DOTDOTDOT {
    PATypeInfo VoidTI;
    VoidTI.PAT = new PATypeHolder(Type::VoidTy);
    VoidTI.S.makeSignless();
    ($$=$1)->push_back(VoidTI);
  }
  | DOTDOTDOT {
    $$ = new std::list<PATypeInfo>();
    PATypeInfo VoidTI;
    VoidTI.PAT = new PATypeHolder(Type::VoidTy);
    VoidTI.S.makeSignless();
    $$->push_back(VoidTI);
  }
  | /*empty*/ {
    $$ = new std::list<PATypeInfo>();
  }
  ;

// ConstVal - The various declarations that go into the constant pool.  This
// production is used ONLY to represent constants that show up AFTER a 'const',
// 'constant' or 'global' token at global scope.  Constants that can be inlined
// into other expressions (such as integers and constexprs) are handled by the
// ResolvedVal, ValueRef and ConstValueRef productions.
//
ConstVal
  : Types '[' ConstVector ']' { // Nonempty unsized arr
    const ArrayType *ATy = dyn_cast<ArrayType>($1.PAT->get());
    if (ATy == 0)
      error("Cannot make array constant with type: '" + 
            $1.PAT->get()->getDescription() + "'");
    const Type *ETy = ATy->getElementType();
    int NumElements = ATy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != -1 && NumElements != (int)$3->size())
      error("Type mismatch: constant sized array initialized with " +
            utostr($3->size()) +  " arguments, but has size of " + 
            itostr(NumElements) + "");

    // Verify all elements are correct type!
    std::vector<Constant*> Elems;
    for (unsigned i = 0; i < $3->size(); i++) {
      Constant *C = (*$3)[i].C;
      const Type* ValTy = C->getType();
      if (ETy != ValTy)
        error("Element #" + utostr(i) + " is not of type '" + 
              ETy->getDescription() +"' as required!\nIt is of type '"+
              ValTy->getDescription() + "'");
      Elems.push_back(C);
    }
    $$.C = ConstantArray::get(ATy, Elems);
    $$.S.copy($1.S);
    delete $1.PAT; 
    delete $3;
  }
  | Types '[' ']' {
    const ArrayType *ATy = dyn_cast<ArrayType>($1.PAT->get());
    if (ATy == 0)
      error("Cannot make array constant with type: '" + 
            $1.PAT->get()->getDescription() + "'");
    int NumElements = ATy->getNumElements();
    if (NumElements != -1 && NumElements != 0) 
      error("Type mismatch: constant sized array initialized with 0"
            " arguments, but has size of " + itostr(NumElements) +"");
    $$.C = ConstantArray::get(ATy, std::vector<Constant*>());
    $$.S.copy($1.S);
    delete $1.PAT;
  }
  | Types 'c' STRINGCONSTANT {
    const ArrayType *ATy = dyn_cast<ArrayType>($1.PAT->get());
    if (ATy == 0)
      error("Cannot make array constant with type: '" + 
            $1.PAT->get()->getDescription() + "'");
    int NumElements = ATy->getNumElements();
    const Type *ETy = dyn_cast<IntegerType>(ATy->getElementType());
    if (!ETy || cast<IntegerType>(ETy)->getBitWidth() != 8)
      error("String arrays require type i8, not '" + ETy->getDescription() + 
            "'");
    char *EndStr = UnEscapeLexed($3, true);
    if (NumElements != -1 && NumElements != (EndStr-$3))
      error("Can't build string constant of size " + 
            itostr((int)(EndStr-$3)) + " when array has size " + 
            itostr(NumElements) + "");
    std::vector<Constant*> Vals;
    for (char *C = (char *)$3; C != (char *)EndStr; ++C)
      Vals.push_back(ConstantInt::get(ETy, *C));
    free($3);
    $$.C = ConstantArray::get(ATy, Vals);
    $$.S.copy($1.S);
    delete $1.PAT;
  }
  | Types '<' ConstVector '>' { // Nonempty unsized arr
    const VectorType *PTy = dyn_cast<VectorType>($1.PAT->get());
    if (PTy == 0)
      error("Cannot make packed constant with type: '" + 
            $1.PAT->get()->getDescription() + "'");
    const Type *ETy = PTy->getElementType();
    int NumElements = PTy->getNumElements();
    // Verify that we have the correct size...
    if (NumElements != -1 && NumElements != (int)$3->size())
      error("Type mismatch: constant sized packed initialized with " +
            utostr($3->size()) +  " arguments, but has size of " + 
            itostr(NumElements) + "");
    // Verify all elements are correct type!
    std::vector<Constant*> Elems;
    for (unsigned i = 0; i < $3->size(); i++) {
      Constant *C = (*$3)[i].C;
      const Type* ValTy = C->getType();
      if (ETy != ValTy)
        error("Element #" + utostr(i) + " is not of type '" + 
              ETy->getDescription() +"' as required!\nIt is of type '"+
              ValTy->getDescription() + "'");
      Elems.push_back(C);
    }
    $$.C = ConstantVector::get(PTy, Elems);
    $$.S.copy($1.S);
    delete $1.PAT;
    delete $3;
  }
  | Types '{' ConstVector '}' {
    const StructType *STy = dyn_cast<StructType>($1.PAT->get());
    if (STy == 0)
      error("Cannot make struct constant with type: '" + 
            $1.PAT->get()->getDescription() + "'");
    if ($3->size() != STy->getNumContainedTypes())
      error("Illegal number of initializers for structure type");

    // Check to ensure that constants are compatible with the type initializer!
    std::vector<Constant*> Fields;
    for (unsigned i = 0, e = $3->size(); i != e; ++i) {
      Constant *C = (*$3)[i].C;
      if (C->getType() != STy->getElementType(i))
        error("Expected type '" + STy->getElementType(i)->getDescription() +
              "' for element #" + utostr(i) + " of structure initializer");
      Fields.push_back(C);
    }
    $$.C = ConstantStruct::get(STy, Fields);
    $$.S.copy($1.S);
    delete $1.PAT;
    delete $3;
  }
  | Types '{' '}' {
    const StructType *STy = dyn_cast<StructType>($1.PAT->get());
    if (STy == 0)
      error("Cannot make struct constant with type: '" + 
              $1.PAT->get()->getDescription() + "'");
    if (STy->getNumContainedTypes() != 0)
      error("Illegal number of initializers for structure type");
    $$.C = ConstantStruct::get(STy, std::vector<Constant*>());
    $$.S.copy($1.S);
    delete $1.PAT;
  }
  | Types '<' '{' ConstVector '}' '>' {
    const StructType *STy = dyn_cast<StructType>($1.PAT->get());
    if (STy == 0)
      error("Cannot make packed struct constant with type: '" + 
            $1.PAT->get()->getDescription() + "'");
    if ($4->size() != STy->getNumContainedTypes())
      error("Illegal number of initializers for packed structure type");

    // Check to ensure that constants are compatible with the type initializer!
    std::vector<Constant*> Fields;
    for (unsigned i = 0, e = $4->size(); i != e; ++i) {
      Constant *C = (*$4)[i].C;
      if (C->getType() != STy->getElementType(i))
        error("Expected type '" + STy->getElementType(i)->getDescription() +
              "' for element #" + utostr(i) + " of packed struct initializer");
      Fields.push_back(C);
    }
    $$.C = ConstantStruct::get(STy, Fields);
    $$.S.copy($1.S);
    delete $1.PAT; 
    delete $4;
  }
  | Types '<' '{' '}' '>' {
    const StructType *STy = dyn_cast<StructType>($1.PAT->get());
    if (STy == 0)
      error("Cannot make packed struct constant with type: '" + 
              $1.PAT->get()->getDescription() + "'");
    if (STy->getNumContainedTypes() != 0)
      error("Illegal number of initializers for packed structure type");
    $$.C = ConstantStruct::get(STy, std::vector<Constant*>());
    $$.S.copy($1.S);
    delete $1.PAT;
  }
  | Types NULL_TOK {
    const PointerType *PTy = dyn_cast<PointerType>($1.PAT->get());
    if (PTy == 0)
      error("Cannot make null pointer constant with type: '" + 
            $1.PAT->get()->getDescription() + "'");
    $$.C = ConstantPointerNull::get(PTy);
    $$.S.copy($1.S);
    delete $1.PAT;
  }
  | Types UNDEF {
    $$.C = UndefValue::get($1.PAT->get());
    $$.S.copy($1.S);
    delete $1.PAT;
  }
  | Types SymbolicValueRef {
    const PointerType *Ty = dyn_cast<PointerType>($1.PAT->get());
    if (Ty == 0)
      error("Global const reference must be a pointer type, not" +
            $1.PAT->get()->getDescription());

    // ConstExprs can exist in the body of a function, thus creating
    // GlobalValues whenever they refer to a variable.  Because we are in
    // the context of a function, getExistingValue will search the functions
    // symbol table instead of the module symbol table for the global symbol,
    // which throws things all off.  To get around this, we just tell
    // getExistingValue that we are at global scope here.
    //
    Function *SavedCurFn = CurFun.CurrentFunction;
    CurFun.CurrentFunction = 0;
    $2.S.copy($1.S);
    Value *V = getExistingValue(Ty, $2);
    CurFun.CurrentFunction = SavedCurFn;

    // If this is an initializer for a constant pointer, which is referencing a
    // (currently) undefined variable, create a stub now that shall be replaced
    // in the future with the right type of variable.
    //
    if (V == 0) {
      assert(isa<PointerType>(Ty) && "Globals may only be used as pointers");
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
    $$.C = cast<GlobalValue>(V);
    $$.S.copy($1.S);
    delete $1.PAT;            // Free the type handle
  }
  | Types ConstExpr {
    if ($1.PAT->get() != $2.C->getType())
      error("Mismatched types for constant expression");
    $$ = $2;
    $$.S.copy($1.S);
    delete $1.PAT;
  }
  | Types ZEROINITIALIZER {
    const Type *Ty = $1.PAT->get();
    if (isa<FunctionType>(Ty) || Ty == Type::LabelTy || isa<OpaqueType>(Ty))
      error("Cannot create a null initialized value of this type");
    $$.C = Constant::getNullValue(Ty);
    $$.S.copy($1.S);
    delete $1.PAT;
  }
  | SIntType EINT64VAL {      // integral constants
    const Type *Ty = $1.T;
    if (!ConstantInt::isValueValidForType(Ty, $2))
      error("Constant value doesn't fit in type");
    $$.C = ConstantInt::get(Ty, $2);
    $$.S.makeSigned();
  }
  | UIntType EUINT64VAL {            // integral constants
    const Type *Ty = $1.T;
    if (!ConstantInt::isValueValidForType(Ty, $2))
      error("Constant value doesn't fit in type");
    $$.C = ConstantInt::get(Ty, $2);
    $$.S.makeUnsigned();
  }
  | BOOL TRUETOK {                      // Boolean constants
    $$.C = ConstantInt::get(Type::Int1Ty, true);
    $$.S.makeUnsigned();
  }
  | BOOL FALSETOK {                     // Boolean constants
    $$.C = ConstantInt::get(Type::Int1Ty, false);
    $$.S.makeUnsigned();
  }
  | FPType FPVAL {                   // Float & Double constants
    if (!ConstantFP::isValueValidForType($1.T, $2))
      error("Floating point constant invalid for type");
    $$.C = ConstantFP::get($1.T, $2);
    $$.S.makeSignless();
  }
  ;

ConstExpr
  : CastOps '(' ConstVal TO Types ')' {
    const Type* SrcTy = $3.C->getType();
    const Type* DstTy = $5.PAT->get();
    Signedness SrcSign($3.S);
    Signedness DstSign($5.S);
    if (!SrcTy->isFirstClassType())
      error("cast constant expression from a non-primitive type: '" +
            SrcTy->getDescription() + "'");
    if (!DstTy->isFirstClassType())
      error("cast constant expression to a non-primitive type: '" +
            DstTy->getDescription() + "'");
    $$.C = cast<Constant>(getCast($1, $3.C, SrcSign, DstTy, DstSign));
    $$.S.copy(DstSign);
    delete $5.PAT;
  }
  | GETELEMENTPTR '(' ConstVal IndexList ')' {
    const Type *Ty = $3.C->getType();
    if (!isa<PointerType>(Ty))
      error("GetElementPtr requires a pointer operand");

    std::vector<Constant*> CIndices;
    upgradeGEPCEIndices($3.C->getType(), $4, CIndices);

    delete $4;
    $$.C = ConstantExpr::getGetElementPtr($3.C, &CIndices[0], CIndices.size());
    $$.S.copy(getElementSign($3, CIndices));
  }
  | SELECT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    if (!$3.C->getType()->isInteger() ||
        cast<IntegerType>($3.C->getType())->getBitWidth() != 1)
      error("Select condition must be bool type");
    if ($5.C->getType() != $7.C->getType())
      error("Select operand types must match");
    $$.C = ConstantExpr::getSelect($3.C, $5.C, $7.C);
    $$.S.copy($5.S);
  }
  | ArithmeticOps '(' ConstVal ',' ConstVal ')' {
    const Type *Ty = $3.C->getType();
    if (Ty != $5.C->getType())
      error("Binary operator types must match");
    // First, make sure we're dealing with the right opcode by upgrading from
    // obsolete versions.
    Instruction::BinaryOps Opcode = getBinaryOp($1, Ty, $3.S);

    // HACK: llvm 1.3 and earlier used to emit invalid pointer constant exprs.
    // To retain backward compatibility with these early compilers, we emit a
    // cast to the appropriate integer type automatically if we are in the
    // broken case.  See PR424 for more information.
    if (!isa<PointerType>(Ty)) {
      $$.C = ConstantExpr::get(Opcode, $3.C, $5.C);
    } else {
      const Type *IntPtrTy = 0;
      switch (CurModule.CurrentModule->getPointerSize()) {
      case Module::Pointer32: IntPtrTy = Type::Int32Ty; break;
      case Module::Pointer64: IntPtrTy = Type::Int64Ty; break;
      default: error("invalid pointer binary constant expr");
      }
      $$.C = ConstantExpr::get(Opcode, 
             ConstantExpr::getCast(Instruction::PtrToInt, $3.C, IntPtrTy),
             ConstantExpr::getCast(Instruction::PtrToInt, $5.C, IntPtrTy));
      $$.C = ConstantExpr::getCast(Instruction::IntToPtr, $$.C, Ty);
    }
    $$.S.copy($3.S); 
  }
  | LogicalOps '(' ConstVal ',' ConstVal ')' {
    const Type* Ty = $3.C->getType();
    if (Ty != $5.C->getType())
      error("Logical operator types must match");
    if (!Ty->isInteger()) {
      if (!isa<VectorType>(Ty) || 
          !cast<VectorType>(Ty)->getElementType()->isInteger())
        error("Logical operator requires integer operands");
    }
    Instruction::BinaryOps Opcode = getBinaryOp($1, Ty, $3.S);
    $$.C = ConstantExpr::get(Opcode, $3.C, $5.C);
    $$.S.copy($3.S);
  }
  | SetCondOps '(' ConstVal ',' ConstVal ')' {
    const Type* Ty = $3.C->getType();
    if (Ty != $5.C->getType())
      error("setcc operand types must match");
    unsigned short pred;
    Instruction::OtherOps Opcode = getCompareOp($1, pred, Ty, $3.S);
    $$.C = ConstantExpr::getCompare(Opcode, $3.C, $5.C);
    $$.S.makeUnsigned();
  }
  | ICMP IPredicates '(' ConstVal ',' ConstVal ')' {
    if ($4.C->getType() != $6.C->getType()) 
      error("icmp operand types must match");
    $$.C = ConstantExpr::getCompare($2, $4.C, $6.C);
    $$.S.makeUnsigned();
  }
  | FCMP FPredicates '(' ConstVal ',' ConstVal ')' {
    if ($4.C->getType() != $6.C->getType()) 
      error("fcmp operand types must match");
    $$.C = ConstantExpr::getCompare($2, $4.C, $6.C);
    $$.S.makeUnsigned();
  }
  | ShiftOps '(' ConstVal ',' ConstVal ')' {
    if (!$5.C->getType()->isInteger() ||
        cast<IntegerType>($5.C->getType())->getBitWidth() != 8)
      error("Shift count for shift constant must be unsigned byte");
    const Type* Ty = $3.C->getType();
    if (!$3.C->getType()->isInteger())
      error("Shift constant expression requires integer operand");
    Constant *ShiftAmt = ConstantExpr::getZExt($5.C, Ty);
    $$.C = ConstantExpr::get(getBinaryOp($1, Ty, $3.S), $3.C, ShiftAmt);
    $$.S.copy($3.S);
  }
  | EXTRACTELEMENT '(' ConstVal ',' ConstVal ')' {
    if (!ExtractElementInst::isValidOperands($3.C, $5.C))
      error("Invalid extractelement operands");
    $$.C = ConstantExpr::getExtractElement($3.C, $5.C);
    $$.S.copy($3.S.get(0));
  }
  | INSERTELEMENT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    if (!InsertElementInst::isValidOperands($3.C, $5.C, $7.C))
      error("Invalid insertelement operands");
    $$.C = ConstantExpr::getInsertElement($3.C, $5.C, $7.C);
    $$.S.copy($3.S);
  }
  | SHUFFLEVECTOR '(' ConstVal ',' ConstVal ',' ConstVal ')' {
    if (!ShuffleVectorInst::isValidOperands($3.C, $5.C, $7.C))
      error("Invalid shufflevector operands");
    $$.C = ConstantExpr::getShuffleVector($3.C, $5.C, $7.C);
    $$.S.copy($3.S);
  }
  ;


// ConstVector - A list of comma separated constants.
ConstVector 
  : ConstVector ',' ConstVal { ($$ = $1)->push_back($3); }
  | ConstVal {
    $$ = new std::vector<ConstInfo>();
    $$->push_back($1);
  }
  ;


// GlobalType - Match either GLOBAL or CONSTANT for global declarations...
GlobalType 
  : GLOBAL { $$ = false; } 
  | CONSTANT { $$ = true; }
  ;


//===----------------------------------------------------------------------===//
//                             Rules to match Modules
//===----------------------------------------------------------------------===//

// Module rule: Capture the result of parsing the whole file into a result
// variable...
//
Module 
  : FunctionList {
    $$ = ParserResult = $1;
    CurModule.ModuleDone();
  }
  ;

// FunctionList - A list of functions, preceeded by a constant pool.
//
FunctionList 
  : FunctionList Function { $$ = $1; CurFun.FunctionDone(); } 
  | FunctionList FunctionProto { $$ = $1; }
  | FunctionList MODULE ASM_TOK AsmBlock { $$ = $1; }  
  | FunctionList IMPLEMENTATION { $$ = $1; }
  | ConstPool {
    $$ = CurModule.CurrentModule;
    // Emit an error if there are any unresolved types left.
    if (!CurModule.LateResolveTypes.empty()) {
      const ValID &DID = CurModule.LateResolveTypes.begin()->first;
      if (DID.Type == ValID::NameVal) {
        error("Reference to an undefined type: '"+DID.getName() + "'");
      } else {
        error("Reference to an undefined type: #" + itostr(DID.Num));
      }
    }
  }
  ;

// ConstPool - Constants with optional names assigned to them.
ConstPool 
  : ConstPool OptAssign TYPE TypesV {
    // Eagerly resolve types.  This is not an optimization, this is a
    // requirement that is due to the fact that we could have this:
    //
    // %list = type { %list * }
    // %list = type { %list * }    ; repeated type decl
    //
    // If types are not resolved eagerly, then the two types will not be
    // determined to be the same type!
    //
    ResolveTypeTo($2, $4.PAT->get(), $4.S);

    if (!setTypeName($4, $2) && !$2) {
      // If this is a numbered type that is not a redefinition, add it to the 
      // slot table.
      CurModule.Types.push_back($4.PAT->get());
      CurModule.TypeSigns.push_back($4.S);
    }
    delete $4.PAT;
  }
  | ConstPool FunctionProto {       // Function prototypes can be in const pool
  }
  | ConstPool MODULE ASM_TOK AsmBlock {  // Asm blocks can be in the const pool
  }
  | ConstPool OptAssign OptLinkage GlobalType ConstVal {
    if ($5.C == 0) 
      error("Global value initializer is not a constant");
    CurGV = ParseGlobalVariable($2, $3, $4, $5.C->getType(), $5.C, $5.S);
  } GlobalVarAttributes {
    CurGV = 0;
  }
  | ConstPool OptAssign EXTERNAL GlobalType Types {
    const Type *Ty = $5.PAT->get();
    CurGV = ParseGlobalVariable($2, GlobalValue::ExternalLinkage, $4, Ty, 0,
                                $5.S);
    delete $5.PAT;
  } GlobalVarAttributes {
    CurGV = 0;
  }
  | ConstPool OptAssign DLLIMPORT GlobalType Types {
    const Type *Ty = $5.PAT->get();
    CurGV = ParseGlobalVariable($2, GlobalValue::DLLImportLinkage, $4, Ty, 0,
                                $5.S);
    delete $5.PAT;
  } GlobalVarAttributes {
    CurGV = 0;
  }
  | ConstPool OptAssign EXTERN_WEAK GlobalType Types {
    const Type *Ty = $5.PAT->get();
    CurGV = 
      ParseGlobalVariable($2, GlobalValue::ExternalWeakLinkage, $4, Ty, 0, 
                          $5.S);
    delete $5.PAT;
  } GlobalVarAttributes {
    CurGV = 0;
  }
  | ConstPool TARGET TargetDefinition { 
  }
  | ConstPool DEPLIBS '=' LibrariesDefinition {
  }
  | /* empty: end of list */ { 
  }
  ;

AsmBlock 
  : STRINGCONSTANT {
    const std::string &AsmSoFar = CurModule.CurrentModule->getModuleInlineAsm();
    char *EndStr = UnEscapeLexed($1, true);
    std::string NewAsm($1, EndStr);
    free($1);

    if (AsmSoFar.empty())
      CurModule.CurrentModule->setModuleInlineAsm(NewAsm);
    else
      CurModule.CurrentModule->setModuleInlineAsm(AsmSoFar+"\n"+NewAsm);
  }
  ;

BigOrLittle 
  : BIG    { $$ = Module::BigEndian; }
  | LITTLE { $$ = Module::LittleEndian; }
  ;

TargetDefinition 
  : ENDIAN '=' BigOrLittle {
    CurModule.setEndianness($3);
  }
  | POINTERSIZE '=' EUINT64VAL {
    if ($3 == 32)
      CurModule.setPointerSize(Module::Pointer32);
    else if ($3 == 64)
      CurModule.setPointerSize(Module::Pointer64);
    else
      error("Invalid pointer size: '" + utostr($3) + "'");
  }
  | TRIPLE '=' STRINGCONSTANT {
    CurModule.CurrentModule->setTargetTriple($3);
    free($3);
  }
  | DATALAYOUT '=' STRINGCONSTANT {
    CurModule.CurrentModule->setDataLayout($3);
    free($3);
  }
  ;

LibrariesDefinition 
  : '[' LibList ']'
  ;

LibList 
  : LibList ',' STRINGCONSTANT {
      CurModule.CurrentModule->addLibrary($3);
      free($3);
  }
  | STRINGCONSTANT {
    CurModule.CurrentModule->addLibrary($1);
    free($1);
  }
  | /* empty: end of list */ { }
  ;

//===----------------------------------------------------------------------===//
//                       Rules to match Function Headers
//===----------------------------------------------------------------------===//

Name 
  : VAR_ID | STRINGCONSTANT
  ;

OptName 
  : Name 
  | /*empty*/ { $$ = 0; }
  ;

ArgVal 
  : Types OptName {
    if ($1.PAT->get() == Type::VoidTy)
      error("void typed arguments are invalid");
    $$ = new std::pair<PATypeInfo, char*>($1, $2);
  }
  ;

ArgListH 
  : ArgListH ',' ArgVal {
    $$ = $1;
    $$->push_back(*$3);
    delete $3;
  }
  | ArgVal {
    $$ = new std::vector<std::pair<PATypeInfo,char*> >();
    $$->push_back(*$1);
    delete $1;
  }
  ;

ArgList 
  : ArgListH { $$ = $1; }
  | ArgListH ',' DOTDOTDOT {
    $$ = $1;
    PATypeInfo VoidTI;
    VoidTI.PAT = new PATypeHolder(Type::VoidTy);
    VoidTI.S.makeSignless();
    $$->push_back(std::pair<PATypeInfo, char*>(VoidTI, 0));
  }
  | DOTDOTDOT {
    $$ = new std::vector<std::pair<PATypeInfo,char*> >();
    PATypeInfo VoidTI;
    VoidTI.PAT = new PATypeHolder(Type::VoidTy);
    VoidTI.S.makeSignless();
    $$->push_back(std::pair<PATypeInfo, char*>(VoidTI, 0));
  }
  | /* empty */ { $$ = 0; }
  ;

FunctionHeaderH 
  : OptCallingConv TypesV Name '(' ArgList ')' OptSection OptAlign {
    UnEscapeLexed($3);
    std::string FunctionName($3);
    free($3);  // Free strdup'd memory!

    const Type* RetTy = $2.PAT->get();
    
    if (!RetTy->isFirstClassType() && RetTy != Type::VoidTy)
      error("LLVM functions cannot return aggregate types");

    Signedness FTySign;
    FTySign.makeComposite($2.S);
    std::vector<const Type*> ParamTyList;

    // In LLVM 2.0 the signatures of three varargs intrinsics changed to take
    // i8*. We check here for those names and override the parameter list
    // types to ensure the prototype is correct.
    if (FunctionName == "llvm.va_start" || FunctionName == "llvm.va_end") {
      ParamTyList.push_back(PointerType::get(Type::Int8Ty));
    } else if (FunctionName == "llvm.va_copy") {
      ParamTyList.push_back(PointerType::get(Type::Int8Ty));
      ParamTyList.push_back(PointerType::get(Type::Int8Ty));
    } else if ($5) {   // If there are arguments...
      for (std::vector<std::pair<PATypeInfo,char*> >::iterator 
           I = $5->begin(), E = $5->end(); I != E; ++I) {
        const Type *Ty = I->first.PAT->get();
        ParamTyList.push_back(Ty);
        FTySign.add(I->first.S);
      }
    }

    bool isVarArg = ParamTyList.size() && ParamTyList.back() == Type::VoidTy;
    if (isVarArg) 
      ParamTyList.pop_back();

    // Convert the CSRet calling convention into the corresponding parameter
    // attribute.
    ParamAttrsList *PAL = 0;
    if ($1 == OldCallingConv::CSRet) {
      ParamAttrsVector Attrs;
      ParamAttrsWithIndex PAWI;
      PAWI.index = 1;  PAWI.attrs = ParamAttr::StructRet; // first arg
      Attrs.push_back(PAWI);
      PAL = ParamAttrsList::get(Attrs);
    }

    const FunctionType *FT = 
      FunctionType::get(RetTy, ParamTyList, isVarArg, PAL);
    const PointerType *PFT = PointerType::get(FT);
    delete $2.PAT;

    ValID ID;
    if (!FunctionName.empty()) {
      ID = ValID::create((char*)FunctionName.c_str());
    } else {
      ID = ValID::create((int)CurModule.Values[PFT].size());
    }
    ID.S.makeComposite(FTySign);

    Function *Fn = 0;
    Module* M = CurModule.CurrentModule;

    // See if this function was forward referenced.  If so, recycle the object.
    if (GlobalValue *FWRef = CurModule.GetForwardRefForGlobal(PFT, ID)) {
      // Move the function to the end of the list, from whereever it was 
      // previously inserted.
      Fn = cast<Function>(FWRef);
      M->getFunctionList().remove(Fn);
      M->getFunctionList().push_back(Fn);
    } else if (!FunctionName.empty()) {
      GlobalValue *Conflict = M->getFunction(FunctionName);
      if (!Conflict)
        Conflict = M->getNamedGlobal(FunctionName);
      if (Conflict && PFT == Conflict->getType()) {
        if (!CurFun.isDeclare && !Conflict->isDeclaration()) {
          // We have two function definitions that conflict, same type, same
          // name. We should really check to make sure that this is the result
          // of integer type planes collapsing and generate an error if it is
          // not, but we'll just rename on the assumption that it is. However,
          // let's do it intelligently and rename the internal linkage one
          // if there is one.
          std::string NewName(makeNameUnique(FunctionName));
          if (Conflict->hasInternalLinkage()) {
            Conflict->setName(NewName);
            RenameMapKey Key = 
              makeRenameMapKey(FunctionName, Conflict->getType(), ID.S);
            CurModule.RenameMap[Key] = NewName;
            Fn = new Function(FT, CurFun.Linkage, FunctionName, M);
            InsertValue(Fn, CurModule.Values);
          } else {
            Fn = new Function(FT, CurFun.Linkage, NewName, M);
            InsertValue(Fn, CurModule.Values);
            RenameMapKey Key = 
              makeRenameMapKey(FunctionName, PFT, ID.S);
            CurModule.RenameMap[Key] = NewName;
          }
        } else {
          // If they are not both definitions, then just use the function we
          // found since the types are the same.
          Fn = cast<Function>(Conflict);

          // Make sure to strip off any argument names so we can't get 
          // conflicts.
          if (Fn->isDeclaration())
            for (Function::arg_iterator AI = Fn->arg_begin(), 
                 AE = Fn->arg_end(); AI != AE; ++AI)
              AI->setName("");
        }
      } else if (Conflict) {
        // We have two globals with the same name and different types. 
        // Previously, this was permitted because the symbol table had 
        // "type planes" and names only needed to be distinct within a 
        // type plane. After PR411 was fixed, this is no loner the case. 
        // To resolve this we must rename one of the two. 
        if (Conflict->hasInternalLinkage()) {
          // We can safely rename the Conflict.
          RenameMapKey Key = 
            makeRenameMapKey(Conflict->getName(), Conflict->getType(), 
              CurModule.NamedValueSigns[Conflict->getName()]);
          Conflict->setName(makeNameUnique(Conflict->getName()));
          CurModule.RenameMap[Key] = Conflict->getName();
          Fn = new Function(FT, CurFun.Linkage, FunctionName, M);
          InsertValue(Fn, CurModule.Values);
        } else { 
          // We can't quietly rename either of these things, but we must
          // rename one of them. Only if the function's linkage is internal can
          // we forgo a warning message about the renamed function. 
          std::string NewName = makeNameUnique(FunctionName);
          if (CurFun.Linkage != GlobalValue::InternalLinkage) {
            warning("Renaming function '" + FunctionName + "' as '" + NewName +
                    "' may cause linkage errors");
          }
          // Elect to rename the thing we're now defining.
          Fn = new Function(FT, CurFun.Linkage, NewName, M);
          InsertValue(Fn, CurModule.Values);
          RenameMapKey Key = makeRenameMapKey(FunctionName, PFT, ID.S);
          CurModule.RenameMap[Key] = NewName;
        } 
      } else {
        // There's no conflict, just define the function
        Fn = new Function(FT, CurFun.Linkage, FunctionName, M);
        InsertValue(Fn, CurModule.Values);
      }
    } else {
      // There's no conflict, just define the function
      Fn = new Function(FT, CurFun.Linkage, FunctionName, M);
      InsertValue(Fn, CurModule.Values);
    }


    CurFun.FunctionStart(Fn);

    if (CurFun.isDeclare) {
      // If we have declaration, always overwrite linkage.  This will allow us 
      // to correctly handle cases, when pointer to function is passed as 
      // argument to another function.
      Fn->setLinkage(CurFun.Linkage);
    }
    Fn->setCallingConv(upgradeCallingConv($1));
    Fn->setAlignment($8);
    if ($7) {
      Fn->setSection($7);
      free($7);
    }

    // Add all of the arguments we parsed to the function...
    if ($5) {                     // Is null if empty...
      if (isVarArg) {  // Nuke the last entry
        assert($5->back().first.PAT->get() == Type::VoidTy && 
               $5->back().second == 0 && "Not a varargs marker");
        delete $5->back().first.PAT;
        $5->pop_back();  // Delete the last entry
      }
      Function::arg_iterator ArgIt = Fn->arg_begin();
      Function::arg_iterator ArgEnd = Fn->arg_end();
      std::vector<std::pair<PATypeInfo,char*> >::iterator I = $5->begin();
      std::vector<std::pair<PATypeInfo,char*> >::iterator E = $5->end();
      for ( ; I != E && ArgIt != ArgEnd; ++I, ++ArgIt) {
        delete I->first.PAT;                      // Delete the typeholder...
        ValueInfo VI; VI.V = ArgIt; VI.S.copy(I->first.S); 
        setValueName(VI, I->second);           // Insert arg into symtab...
        InsertValue(ArgIt);
      }
      delete $5;                     // We're now done with the argument list
    }
    lastCallingConv = OldCallingConv::C;
  }
  ;

BEGIN 
  : BEGINTOK | '{'                // Allow BEGIN or '{' to start a function
  ;

FunctionHeader 
  : OptLinkage { CurFun.Linkage = $1; } FunctionHeaderH BEGIN {
    $$ = CurFun.CurrentFunction;

    // Make sure that we keep track of the linkage type even if there was a
    // previous "declare".
    $$->setLinkage($1);
  }
  ;

END 
  : ENDTOK | '}'                    // Allow end of '}' to end a function
  ;

Function 
  : BasicBlockList END {
    $$ = $1;
  };

FnDeclareLinkage
  : /*default*/ { $$ = GlobalValue::ExternalLinkage; }
  | DLLIMPORT   { $$ = GlobalValue::DLLImportLinkage; } 
  | EXTERN_WEAK { $$ = GlobalValue::ExternalWeakLinkage; }
  ;
  
FunctionProto 
  : DECLARE { CurFun.isDeclare = true; } 
     FnDeclareLinkage { CurFun.Linkage = $3; } FunctionHeaderH {
    $$ = CurFun.CurrentFunction;
    CurFun.FunctionDone();
    
  }
  ;

//===----------------------------------------------------------------------===//
//                        Rules to match Basic Blocks
//===----------------------------------------------------------------------===//

OptSideEffect 
  : /* empty */ { $$ = false; }
  | SIDEEFFECT { $$ = true; }
  ;

ConstValueRef 
    // A reference to a direct constant
  : ESINT64VAL { $$ = ValID::create($1); }
  | EUINT64VAL { $$ = ValID::create($1); }
  | FPVAL { $$ = ValID::create($1); } 
  | TRUETOK { 
    $$ = ValID::create(ConstantInt::get(Type::Int1Ty, true));
    $$.S.makeUnsigned();
  }
  | FALSETOK { 
    $$ = ValID::create(ConstantInt::get(Type::Int1Ty, false)); 
    $$.S.makeUnsigned();
  }
  | NULL_TOK { $$ = ValID::createNull(); }
  | UNDEF { $$ = ValID::createUndef(); }
  | ZEROINITIALIZER { $$ = ValID::createZeroInit(); }
  | '<' ConstVector '>' { // Nonempty unsized packed vector
    const Type *ETy = (*$2)[0].C->getType();
    int NumElements = $2->size(); 
    VectorType* pt = VectorType::get(ETy, NumElements);
    $$.S.makeComposite((*$2)[0].S);
    PATypeHolder* PTy = new PATypeHolder(HandleUpRefs(pt, $$.S));
    
    // Verify all elements are correct type!
    std::vector<Constant*> Elems;
    for (unsigned i = 0; i < $2->size(); i++) {
      Constant *C = (*$2)[i].C;
      const Type *CTy = C->getType();
      if (ETy != CTy)
        error("Element #" + utostr(i) + " is not of type '" + 
              ETy->getDescription() +"' as required!\nIt is of type '" +
              CTy->getDescription() + "'");
      Elems.push_back(C);
    }
    $$ = ValID::create(ConstantVector::get(pt, Elems));
    delete PTy; delete $2;
  }
  | ConstExpr {
    $$ = ValID::create($1.C);
    $$.S.copy($1.S);
  }
  | ASM_TOK OptSideEffect STRINGCONSTANT ',' STRINGCONSTANT {
    char *End = UnEscapeLexed($3, true);
    std::string AsmStr = std::string($3, End);
    End = UnEscapeLexed($5, true);
    std::string Constraints = std::string($5, End);
    $$ = ValID::createInlineAsm(AsmStr, Constraints, $2);
    free($3);
    free($5);
  }
  ;

// SymbolicValueRef - Reference to one of two ways of symbolically refering to // another value.
//
SymbolicValueRef 
  : INTVAL {  $$ = ValID::create($1); $$.S.makeSignless(); }
  | Name   {  $$ = ValID::create($1); $$.S.makeSignless(); }
  ;

// ValueRef - A reference to a definition... either constant or symbolic
ValueRef 
  : SymbolicValueRef | ConstValueRef
  ;


// ResolvedVal - a <type> <value> pair.  This is used only in cases where the
// type immediately preceeds the value reference, and allows complex constant
// pool references (for things like: 'ret [2 x int] [ int 12, int 42]')
ResolvedVal 
  : Types ValueRef { 
    const Type *Ty = $1.PAT->get();
    $2.S.copy($1.S);
    $$.V = getVal(Ty, $2); 
    $$.S.copy($1.S);
    delete $1.PAT;
  }
  ;

BasicBlockList 
  : BasicBlockList BasicBlock {
    $$ = $1;
  }
  | FunctionHeader BasicBlock { // Do not allow functions with 0 basic blocks   
    $$ = $1;
  };


// Basic blocks are terminated by branching instructions: 
// br, br/cc, switch, ret
//
BasicBlock 
  : InstructionList OptAssign BBTerminatorInst  {
    ValueInfo VI; VI.V = $3.TI; VI.S.copy($3.S);
    setValueName(VI, $2);
    InsertValue($3.TI);
    $1->getInstList().push_back($3.TI);
    InsertValue($1);
    $$ = $1;
  }
  ;

InstructionList
  : InstructionList Inst {
    if ($2.I)
      $1->getInstList().push_back($2.I);
    $$ = $1;
  }
  | /* empty */ {
    $$ = CurBB = getBBVal(ValID::create((int)CurFun.NextBBNum++),true);
    // Make sure to move the basic block to the correct location in the
    // function, instead of leaving it inserted wherever it was first
    // referenced.
    Function::BasicBlockListType &BBL = 
      CurFun.CurrentFunction->getBasicBlockList();
    BBL.splice(BBL.end(), BBL, $$);
  }
  | LABELSTR {
    $$ = CurBB = getBBVal(ValID::create($1), true);
    // Make sure to move the basic block to the correct location in the
    // function, instead of leaving it inserted wherever it was first
    // referenced.
    Function::BasicBlockListType &BBL = 
      CurFun.CurrentFunction->getBasicBlockList();
    BBL.splice(BBL.end(), BBL, $$);
  }
  ;

Unwind : UNWIND | EXCEPT;

BBTerminatorInst 
  : RET ResolvedVal {              // Return with a result...
    $$.TI = new ReturnInst($2.V);
    $$.S.makeSignless();
  }
  | RET VOID {                                       // Return with no result...
    $$.TI = new ReturnInst();
    $$.S.makeSignless();
  }
  | BR LABEL ValueRef {                         // Unconditional Branch...
    BasicBlock* tmpBB = getBBVal($3);
    $$.TI = new BranchInst(tmpBB);
    $$.S.makeSignless();
  }                                                  // Conditional Branch...
  | BR BOOL ValueRef ',' LABEL ValueRef ',' LABEL ValueRef {  
    $6.S.makeSignless();
    $9.S.makeSignless();
    BasicBlock* tmpBBA = getBBVal($6);
    BasicBlock* tmpBBB = getBBVal($9);
    $3.S.makeUnsigned();
    Value* tmpVal = getVal(Type::Int1Ty, $3);
    $$.TI = new BranchInst(tmpBBA, tmpBBB, tmpVal);
    $$.S.makeSignless();
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' JumpTable ']' {
    $3.S.copy($2.S);
    Value* tmpVal = getVal($2.T, $3);
    $6.S.makeSignless();
    BasicBlock* tmpBB = getBBVal($6);
    SwitchInst *S = new SwitchInst(tmpVal, tmpBB, $8->size());
    $$.TI = S;
    $$.S.makeSignless();
    std::vector<std::pair<Constant*,BasicBlock*> >::iterator I = $8->begin(),
      E = $8->end();
    for (; I != E; ++I) {
      if (ConstantInt *CI = dyn_cast<ConstantInt>(I->first))
          S->addCase(CI, I->second);
      else
        error("Switch case is constant, but not a simple integer");
    }
    delete $8;
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' ']' {
    $3.S.copy($2.S);
    Value* tmpVal = getVal($2.T, $3);
    $6.S.makeSignless();
    BasicBlock* tmpBB = getBBVal($6);
    SwitchInst *S = new SwitchInst(tmpVal, tmpBB, 0);
    $$.TI = S;
    $$.S.makeSignless();
  }
  | INVOKE OptCallingConv TypesV ValueRef '(' ValueRefListE ')'
    TO LABEL ValueRef Unwind LABEL ValueRef {
    const PointerType *PFTy;
    const FunctionType *Ty;
    Signedness FTySign;

    if (!(PFTy = dyn_cast<PointerType>($3.PAT->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      FTySign.makeComposite($3.S);
      if ($6) {
        for (std::vector<ValueInfo>::iterator I = $6->begin(), E = $6->end();
             I != E; ++I) {
          ParamTypes.push_back((*I).V->getType());
          FTySign.add(I->S);
        }
      }
      ParamAttrsList *PAL = 0;
      if ($2 == OldCallingConv::CSRet) {
        ParamAttrsVector Attrs;
        ParamAttrsWithIndex PAWI;
        PAWI.index = 1;  PAWI.attrs = ParamAttr::StructRet; // first arg
        Attrs.push_back(PAWI);
        PAL = ParamAttrsList::get(Attrs);
      }
      bool isVarArg = ParamTypes.size() && ParamTypes.back() == Type::VoidTy;
      if (isVarArg) ParamTypes.pop_back();
      Ty = FunctionType::get($3.PAT->get(), ParamTypes, isVarArg, PAL);
      PFTy = PointerType::get(Ty);
      $$.S.copy($3.S);
    } else {
      FTySign = $3.S;
      // Get the signedness of the result type. $3 is the pointer to the
      // function type so we get the 0th element to extract the function type,
      // and then the 0th element again to get the result type.
      $$.S.copy($3.S.get(0).get(0)); 
    }

    $4.S.makeComposite(FTySign);
    Value *V = getVal(PFTy, $4);   // Get the function we're calling...
    BasicBlock *Normal = getBBVal($10);
    BasicBlock *Except = getBBVal($13);

    // Create the call node...
    if (!$6) {                                   // Has no arguments?
      $$.TI = new InvokeInst(V, Normal, Except, 0, 0);
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly!
      //
      FunctionType::param_iterator I = Ty->param_begin();
      FunctionType::param_iterator E = Ty->param_end();
      std::vector<ValueInfo>::iterator ArgI = $6->begin(), ArgE = $6->end();

      std::vector<Value*> Args;
      for (; ArgI != ArgE && I != E; ++ArgI, ++I) {
        if ((*ArgI).V->getType() != *I)
          error("Parameter " +(*ArgI).V->getName()+ " is not of type '" +
                (*I)->getDescription() + "'");
        Args.push_back((*ArgI).V);
      }

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
        error("Invalid number of parameters detected");

      $$.TI = new InvokeInst(V, Normal, Except, &Args[0], Args.size());
    }
    cast<InvokeInst>($$.TI)->setCallingConv(upgradeCallingConv($2));
    delete $3.PAT;
    delete $6;
    lastCallingConv = OldCallingConv::C;
  }
  | Unwind {
    $$.TI = new UnwindInst();
    $$.S.makeSignless();
  }
  | UNREACHABLE {
    $$.TI = new UnreachableInst();
    $$.S.makeSignless();
  }
  ;

JumpTable 
  : JumpTable IntType ConstValueRef ',' LABEL ValueRef {
    $$ = $1;
    $3.S.copy($2.S);
    Constant *V = cast<Constant>(getExistingValue($2.T, $3));
    
    if (V == 0)
      error("May only switch on a constant pool value");

    $6.S.makeSignless();
    BasicBlock* tmpBB = getBBVal($6);
    $$->push_back(std::make_pair(V, tmpBB));
  }
  | IntType ConstValueRef ',' LABEL ValueRef {
    $$ = new std::vector<std::pair<Constant*, BasicBlock*> >();
    $2.S.copy($1.S);
    Constant *V = cast<Constant>(getExistingValue($1.T, $2));

    if (V == 0)
      error("May only switch on a constant pool value");

    $5.S.makeSignless();
    BasicBlock* tmpBB = getBBVal($5);
    $$->push_back(std::make_pair(V, tmpBB)); 
  }
  ;

Inst 
  : OptAssign InstVal {
    bool omit = false;
    if ($1)
      if (BitCastInst *BCI = dyn_cast<BitCastInst>($2.I))
        if (BCI->getSrcTy() == BCI->getDestTy() && 
            BCI->getOperand(0)->getName() == $1)
          // This is a useless bit cast causing a name redefinition. It is
          // a bit cast from a type to the same type of an operand with the
          // same name as the name we would give this instruction. Since this
          // instruction results in no code generation, it is safe to omit
          // the instruction. This situation can occur because of collapsed
          // type planes. For example:
          //   %X = add int %Y, %Z
          //   %X = cast int %Y to uint
          // After upgrade, this looks like:
          //   %X = add i32 %Y, %Z
          //   %X = bitcast i32 to i32
          // The bitcast is clearly useless so we omit it.
          omit = true;
    if (omit) {
      $$.I = 0;
      $$.S.makeSignless();
    } else {
      ValueInfo VI; VI.V = $2.I; VI.S.copy($2.S);
      setValueName(VI, $1);
      InsertValue($2.I);
      $$ = $2;
    }
  };

PHIList : Types '[' ValueRef ',' ValueRef ']' {    // Used for PHI nodes
    $$.P = new std::list<std::pair<Value*, BasicBlock*> >();
    $$.S.copy($1.S);
    $3.S.copy($1.S);
    Value* tmpVal = getVal($1.PAT->get(), $3);
    $5.S.makeSignless();
    BasicBlock* tmpBB = getBBVal($5);
    $$.P->push_back(std::make_pair(tmpVal, tmpBB));
    delete $1.PAT;
  }
  | PHIList ',' '[' ValueRef ',' ValueRef ']' {
    $$ = $1;
    $4.S.copy($1.S);
    Value* tmpVal = getVal($1.P->front().first->getType(), $4);
    $6.S.makeSignless();
    BasicBlock* tmpBB = getBBVal($6);
    $1.P->push_back(std::make_pair(tmpVal, tmpBB));
  }
  ;

ValueRefList : ResolvedVal {    // Used for call statements, and memory insts...
    $$ = new std::vector<ValueInfo>();
    $$->push_back($1);
  }
  | ValueRefList ',' ResolvedVal {
    $$ = $1;
    $1->push_back($3);
  };

// ValueRefListE - Just like ValueRefList, except that it may also be empty!
ValueRefListE 
  : ValueRefList 
  | /*empty*/ { $$ = 0; }
  ;

OptTailCall 
  : TAIL CALL {
    $$ = true;
  }
  | CALL {
    $$ = false;
  }
  ;

InstVal 
  : ArithmeticOps Types ValueRef ',' ValueRef {
    $3.S.copy($2.S);
    $5.S.copy($2.S);
    const Type* Ty = $2.PAT->get();
    if (!Ty->isInteger() && !Ty->isFloatingPoint() && !isa<VectorType>(Ty))
      error("Arithmetic operator requires integer, FP, or packed operands");
    if (isa<VectorType>(Ty) && 
        ($1 == URemOp || $1 == SRemOp || $1 == FRemOp || $1 == RemOp))
      error("Remainder not supported on vector types");
    // Upgrade the opcode from obsolete versions before we do anything with it.
    Instruction::BinaryOps Opcode = getBinaryOp($1, Ty, $2.S);
    Value* val1 = getVal(Ty, $3); 
    Value* val2 = getVal(Ty, $5);
    $$.I = BinaryOperator::create(Opcode, val1, val2);
    if ($$.I == 0)
      error("binary operator returned null");
    $$.S.copy($2.S);
    delete $2.PAT;
  }
  | LogicalOps Types ValueRef ',' ValueRef {
    $3.S.copy($2.S);
    $5.S.copy($2.S);
    const Type *Ty = $2.PAT->get();
    if (!Ty->isInteger()) {
      if (!isa<VectorType>(Ty) ||
          !cast<VectorType>(Ty)->getElementType()->isInteger())
        error("Logical operator requires integral operands");
    }
    Instruction::BinaryOps Opcode = getBinaryOp($1, Ty, $2.S);
    Value* tmpVal1 = getVal(Ty, $3);
    Value* tmpVal2 = getVal(Ty, $5);
    $$.I = BinaryOperator::create(Opcode, tmpVal1, tmpVal2);
    if ($$.I == 0)
      error("binary operator returned null");
    $$.S.copy($2.S);
    delete $2.PAT;
  }
  | SetCondOps Types ValueRef ',' ValueRef {
    $3.S.copy($2.S);
    $5.S.copy($2.S);
    const Type* Ty = $2.PAT->get();
    if(isa<VectorType>(Ty))
      error("VectorTypes currently not supported in setcc instructions");
    unsigned short pred;
    Instruction::OtherOps Opcode = getCompareOp($1, pred, Ty, $2.S);
    Value* tmpVal1 = getVal(Ty, $3);
    Value* tmpVal2 = getVal(Ty, $5);
    $$.I = CmpInst::create(Opcode, pred, tmpVal1, tmpVal2);
    if ($$.I == 0)
      error("binary operator returned null");
    $$.S.makeUnsigned();
    delete $2.PAT;
  }
  | ICMP IPredicates Types ValueRef ',' ValueRef {
    $4.S.copy($3.S);
    $6.S.copy($3.S);
    const Type *Ty = $3.PAT->get();
    if (isa<VectorType>(Ty)) 
      error("VectorTypes currently not supported in icmp instructions");
    else if (!Ty->isInteger() && !isa<PointerType>(Ty))
      error("icmp requires integer or pointer typed operands");
    Value* tmpVal1 = getVal(Ty, $4);
    Value* tmpVal2 = getVal(Ty, $6);
    $$.I = new ICmpInst($2, tmpVal1, tmpVal2);
    $$.S.makeUnsigned();
    delete $3.PAT;
  }
  | FCMP FPredicates Types ValueRef ',' ValueRef {
    $4.S.copy($3.S);
    $6.S.copy($3.S);
    const Type *Ty = $3.PAT->get();
    if (isa<VectorType>(Ty))
      error("VectorTypes currently not supported in fcmp instructions");
    else if (!Ty->isFloatingPoint())
      error("fcmp instruction requires floating point operands");
    Value* tmpVal1 = getVal(Ty, $4);
    Value* tmpVal2 = getVal(Ty, $6);
    $$.I = new FCmpInst($2, tmpVal1, tmpVal2);
    $$.S.makeUnsigned();
    delete $3.PAT;
  }
  | NOT ResolvedVal {
    warning("Use of obsolete 'not' instruction: Replacing with 'xor");
    const Type *Ty = $2.V->getType();
    Value *Ones = ConstantInt::getAllOnesValue(Ty);
    if (Ones == 0)
      error("Expected integral type for not instruction");
    $$.I = BinaryOperator::create(Instruction::Xor, $2.V, Ones);
    if ($$.I == 0)
      error("Could not create a xor instruction");
    $$.S.copy($2.S);
  }
  | ShiftOps ResolvedVal ',' ResolvedVal {
    if (!$4.V->getType()->isInteger() ||
        cast<IntegerType>($4.V->getType())->getBitWidth() != 8)
      error("Shift amount must be int8");
    const Type* Ty = $2.V->getType();
    if (!Ty->isInteger())
      error("Shift constant expression requires integer operand");
    Value* ShiftAmt = 0;
    if (cast<IntegerType>(Ty)->getBitWidth() > Type::Int8Ty->getBitWidth())
      if (Constant *C = dyn_cast<Constant>($4.V))
        ShiftAmt = ConstantExpr::getZExt(C, Ty);
      else
        ShiftAmt = new ZExtInst($4.V, Ty, makeNameUnique("shift"), CurBB);
    else
      ShiftAmt = $4.V;
    $$.I = BinaryOperator::create(getBinaryOp($1, Ty, $2.S), $2.V, ShiftAmt);
    $$.S.copy($2.S);
  }
  | CastOps ResolvedVal TO Types {
    const Type *DstTy = $4.PAT->get();
    if (!DstTy->isFirstClassType())
      error("cast instruction to a non-primitive type: '" +
            DstTy->getDescription() + "'");
    $$.I = cast<Instruction>(getCast($1, $2.V, $2.S, DstTy, $4.S, true));
    $$.S.copy($4.S);
    delete $4.PAT;
  }
  | SELECT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    if (!$2.V->getType()->isInteger() ||
        cast<IntegerType>($2.V->getType())->getBitWidth() != 1)
      error("select condition must be bool");
    if ($4.V->getType() != $6.V->getType())
      error("select value types should match");
    $$.I = new SelectInst($2.V, $4.V, $6.V);
    $$.S.copy($4.S);
  }
  | VAARG ResolvedVal ',' Types {
    const Type *Ty = $4.PAT->get();
    NewVarArgs = true;
    $$.I = new VAArgInst($2.V, Ty);
    $$.S.copy($4.S);
    delete $4.PAT;
  }
  | VAARG_old ResolvedVal ',' Types {
    const Type* ArgTy = $2.V->getType();
    const Type* DstTy = $4.PAT->get();
    ObsoleteVarArgs = true;
    Function* NF = cast<Function>(CurModule.CurrentModule->
      getOrInsertFunction("llvm.va_copy", ArgTy, ArgTy, (Type *)0));

    //b = vaarg a, t -> 
    //foo = alloca 1 of t
    //bar = vacopy a 
    //store bar -> foo
    //b = vaarg foo, t
    AllocaInst* foo = new AllocaInst(ArgTy, 0, "vaarg.fix");
    CurBB->getInstList().push_back(foo);
    CallInst* bar = new CallInst(NF, $2.V);
    CurBB->getInstList().push_back(bar);
    CurBB->getInstList().push_back(new StoreInst(bar, foo));
    $$.I = new VAArgInst(foo, DstTy);
    $$.S.copy($4.S);
    delete $4.PAT;
  }
  | VANEXT_old ResolvedVal ',' Types {
    const Type* ArgTy = $2.V->getType();
    const Type* DstTy = $4.PAT->get();
    ObsoleteVarArgs = true;
    Function* NF = cast<Function>(CurModule.CurrentModule->
      getOrInsertFunction("llvm.va_copy", ArgTy, ArgTy, (Type *)0));

    //b = vanext a, t ->
    //foo = alloca 1 of t
    //bar = vacopy a
    //store bar -> foo
    //tmp = vaarg foo, t
    //b = load foo
    AllocaInst* foo = new AllocaInst(ArgTy, 0, "vanext.fix");
    CurBB->getInstList().push_back(foo);
    CallInst* bar = new CallInst(NF, $2.V);
    CurBB->getInstList().push_back(bar);
    CurBB->getInstList().push_back(new StoreInst(bar, foo));
    Instruction* tmp = new VAArgInst(foo, DstTy);
    CurBB->getInstList().push_back(tmp);
    $$.I = new LoadInst(foo);
    $$.S.copy($4.S);
    delete $4.PAT;
  }
  | EXTRACTELEMENT ResolvedVal ',' ResolvedVal {
    if (!ExtractElementInst::isValidOperands($2.V, $4.V))
      error("Invalid extractelement operands");
    $$.I = new ExtractElementInst($2.V, $4.V);
    $$.S.copy($2.S.get(0));
  }
  | INSERTELEMENT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    if (!InsertElementInst::isValidOperands($2.V, $4.V, $6.V))
      error("Invalid insertelement operands");
    $$.I = new InsertElementInst($2.V, $4.V, $6.V);
    $$.S.copy($2.S);
  }
  | SHUFFLEVECTOR ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    if (!ShuffleVectorInst::isValidOperands($2.V, $4.V, $6.V))
      error("Invalid shufflevector operands");
    $$.I = new ShuffleVectorInst($2.V, $4.V, $6.V);
    $$.S.copy($2.S);
  }
  | PHI_TOK PHIList {
    const Type *Ty = $2.P->front().first->getType();
    if (!Ty->isFirstClassType())
      error("PHI node operands must be of first class type");
    PHINode *PHI = new PHINode(Ty);
    PHI->reserveOperandSpace($2.P->size());
    while ($2.P->begin() != $2.P->end()) {
      if ($2.P->front().first->getType() != Ty) 
        error("All elements of a PHI node must be of the same type");
      PHI->addIncoming($2.P->front().first, $2.P->front().second);
      $2.P->pop_front();
    }
    $$.I = PHI;
    $$.S.copy($2.S);
    delete $2.P;  // Free the list...
  }
  | OptTailCall OptCallingConv TypesV ValueRef '(' ValueRefListE ')' {
    // Handle the short call syntax
    const PointerType *PFTy;
    const FunctionType *FTy;
    Signedness FTySign;
    if (!(PFTy = dyn_cast<PointerType>($3.PAT->get())) ||
        !(FTy = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      FTySign.makeComposite($3.S);
      if ($6) {
        for (std::vector<ValueInfo>::iterator I = $6->begin(), E = $6->end();
             I != E; ++I) {
          ParamTypes.push_back((*I).V->getType());
          FTySign.add(I->S);
        }
      }

      bool isVarArg = ParamTypes.size() && ParamTypes.back() == Type::VoidTy;
      if (isVarArg) ParamTypes.pop_back();

      const Type *RetTy = $3.PAT->get();
      if (!RetTy->isFirstClassType() && RetTy != Type::VoidTy)
        error("Functions cannot return aggregate types");

      // Deal with CSRetCC
      ParamAttrsList *PAL = 0;
      if ($2 == OldCallingConv::CSRet) {
        ParamAttrsVector Attrs;
        ParamAttrsWithIndex PAWI;
        PAWI.index = 1;  PAWI.attrs = ParamAttr::StructRet; // first arg
        Attrs.push_back(PAWI);
        PAL = ParamAttrsList::get(Attrs);
      }

      FTy = FunctionType::get(RetTy, ParamTypes, isVarArg, PAL);
      PFTy = PointerType::get(FTy);
      $$.S.copy($3.S);
    } else {
      FTySign = $3.S;
      // Get the signedness of the result type. $3 is the pointer to the
      // function type so we get the 0th element to extract the function type,
      // and then the 0th element again to get the result type.
      $$.S.copy($3.S.get(0).get(0)); 
    }
    $4.S.makeComposite(FTySign);

    // First upgrade any intrinsic calls.
    std::vector<Value*> Args;
    if ($6)
      for (unsigned i = 0, e = $6->size(); i < e; ++i) 
        Args.push_back((*$6)[i].V);
    Instruction *Inst = upgradeIntrinsicCall(FTy->getReturnType(), $4, Args);

    // If we got an upgraded intrinsic
    if (Inst) {
      $$.I = Inst;
    } else {
      // Get the function we're calling
      Value *V = getVal(PFTy, $4);

      // Check the argument values match
      if (!$6) {                                   // Has no arguments?
        // Make sure no arguments is a good thing!
        if (FTy->getNumParams() != 0)
          error("No arguments passed to a function that expects arguments");
      } else {                                     // Has arguments?
        // Loop through FunctionType's arguments and ensure they are specified
        // correctly!
        //
        FunctionType::param_iterator I = FTy->param_begin();
        FunctionType::param_iterator E = FTy->param_end();
        std::vector<ValueInfo>::iterator ArgI = $6->begin(), ArgE = $6->end();

        for (; ArgI != ArgE && I != E; ++ArgI, ++I)
          if ((*ArgI).V->getType() != *I)
            error("Parameter " +(*ArgI).V->getName()+ " is not of type '" +
                  (*I)->getDescription() + "'");

        if (I != E || (ArgI != ArgE && !FTy->isVarArg()))
          error("Invalid number of parameters detected");
      }

      // Create the call instruction
      CallInst *CI = new CallInst(V, &Args[0], Args.size());
      CI->setTailCall($1);
      CI->setCallingConv(upgradeCallingConv($2));
      $$.I = CI;
    }
    delete $3.PAT;
    delete $6;
    lastCallingConv = OldCallingConv::C;
  }
  | MemoryInst {
    $$ = $1;
  }
  ;


// IndexList - List of indices for GEP based instructions...
IndexList 
  : ',' ValueRefList { $$ = $2; } 
  | /* empty */ { $$ = new std::vector<ValueInfo>(); }
  ;

OptVolatile 
  : VOLATILE { $$ = true; }
  | /* empty */ { $$ = false; }
  ;

MemoryInst 
  : MALLOC Types OptCAlign {
    const Type *Ty = $2.PAT->get();
    $$.S.makeComposite($2.S);
    $$.I = new MallocInst(Ty, 0, $3);
    delete $2.PAT;
  }
  | MALLOC Types ',' UINT ValueRef OptCAlign {
    const Type *Ty = $2.PAT->get();
    $5.S.makeUnsigned();
    $$.S.makeComposite($2.S);
    $$.I = new MallocInst(Ty, getVal($4.T, $5), $6);
    delete $2.PAT;
  }
  | ALLOCA Types OptCAlign {
    const Type *Ty = $2.PAT->get();
    $$.S.makeComposite($2.S);
    $$.I = new AllocaInst(Ty, 0, $3);
    delete $2.PAT;
  }
  | ALLOCA Types ',' UINT ValueRef OptCAlign {
    const Type *Ty = $2.PAT->get();
    $5.S.makeUnsigned();
    $$.S.makeComposite($4.S);
    $$.I = new AllocaInst(Ty, getVal($4.T, $5), $6);
    delete $2.PAT;
  }
  | FREE ResolvedVal {
    const Type *PTy = $2.V->getType();
    if (!isa<PointerType>(PTy))
      error("Trying to free nonpointer type '" + PTy->getDescription() + "'");
    $$.I = new FreeInst($2.V);
    $$.S.makeSignless();
  }
  | OptVolatile LOAD Types ValueRef {
    const Type* Ty = $3.PAT->get();
    $4.S.copy($3.S);
    if (!isa<PointerType>(Ty))
      error("Can't load from nonpointer type: " + Ty->getDescription());
    if (!cast<PointerType>(Ty)->getElementType()->isFirstClassType())
      error("Can't load from pointer of non-first-class type: " +
                     Ty->getDescription());
    Value* tmpVal = getVal(Ty, $4);
    $$.I = new LoadInst(tmpVal, "", $1);
    $$.S.copy($3.S.get(0));
    delete $3.PAT;
  }
  | OptVolatile STORE ResolvedVal ',' Types ValueRef {
    $6.S.copy($5.S);
    const PointerType *PTy = dyn_cast<PointerType>($5.PAT->get());
    if (!PTy)
      error("Can't store to a nonpointer type: " + 
             $5.PAT->get()->getDescription());
    const Type *ElTy = PTy->getElementType();
    Value *StoreVal = $3.V;
    Value* tmpVal = getVal(PTy, $6);
    if (ElTy != $3.V->getType()) {
      StoreVal = handleSRetFuncTypeMerge($3.V, ElTy);
      if (!StoreVal)
        error("Can't store '" + $3.V->getType()->getDescription() +
              "' into space of type '" + ElTy->getDescription() + "'");
      else {
        PTy = PointerType::get(StoreVal->getType());
        if (Constant *C = dyn_cast<Constant>(tmpVal))
          tmpVal = ConstantExpr::getBitCast(C, PTy);
        else
          tmpVal = new BitCastInst(tmpVal, PTy, "upgrd.cast", CurBB);
      }
    }
    $$.I = new StoreInst(StoreVal, tmpVal, $1);
    $$.S.makeSignless();
    delete $5.PAT;
  }
  | GETELEMENTPTR Types ValueRef IndexList {
    $3.S.copy($2.S);
    const Type* Ty = $2.PAT->get();
    if (!isa<PointerType>(Ty))
      error("getelementptr insn requires pointer operand");

    std::vector<Value*> VIndices;
    upgradeGEPInstIndices(Ty, $4, VIndices);

    Value* tmpVal = getVal(Ty, $3);
    $$.I = new GetElementPtrInst(tmpVal, &VIndices[0], VIndices.size());
    ValueInfo VI; VI.V = tmpVal; VI.S.copy($2.S);
    $$.S.copy(getElementSign(VI, VIndices));
    delete $2.PAT;
    delete $4;
  };


%%

int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + llvm::utostr((unsigned) Upgradelineno) + ": ";
  std::string errMsg = where + "error: " + std::string(ErrorMsg);
  if (yychar != YYEMPTY && yychar != 0)
    errMsg += " while reading token '" + std::string(Upgradetext, Upgradeleng) +
              "'.";
  std::cerr << "llvm-upgrade: " << errMsg << '\n';
  std::cout << "llvm-upgrade: parse failed.\n";
  exit(1);
}

void warning(const std::string& ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + llvm::utostr((unsigned) Upgradelineno) + ": ";
  std::string errMsg = where + "warning: " + std::string(ErrorMsg);
  if (yychar != YYEMPTY && yychar != 0)
    errMsg += " while reading token '" + std::string(Upgradetext, Upgradeleng) +
              "'.";
  std::cerr << "llvm-upgrade: " << errMsg << '\n';
}

void error(const std::string& ErrorMsg, int LineNo) {
  if (LineNo == -1) LineNo = Upgradelineno;
  Upgradelineno = LineNo;
  yyerror(ErrorMsg.c_str());
}

