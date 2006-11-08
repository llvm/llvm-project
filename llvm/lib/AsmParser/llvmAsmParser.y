//===-- llvmAsmParser.y - Parser for llvm assembly files ---------*- C++ -*--=//
//
//  This file implements the bison parser for LLVM assembly languages files.
//
//===------------------------------------------------------------------------=//

//
// TODO: Parse comments and add them to an internal node... so that they may
// be saved in the bytecode format as well as everything else.  Very important
// for a general IR format.
//

%{
#include "ParserInternals.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/SymbolTable.h"
#include "llvm/Module.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/Support/STLExtras.h"
#include "llvm/Support/DepthFirstIterator.h"
#include <list>
#include <utility>            // Get definition of pair class
#include <algorithm>
#include <stdio.h>            // This embarasment is due to our flex lexer...

int yyerror(const char *ErrorMsg); // Forward declarations to prevent "implicit 
int yylex();                       // declaration" of xxx warnings.
int yyparse();

static Module *ParserResult;
string CurFilename;

// DEBUG_UPREFS - Define this symbol if you want to enable debugging output
// relating to upreferences in the input stream.
//
//#define DEBUG_UPREFS 1
#ifdef DEBUG_UPREFS
#define UR_OUT(X) cerr << X
#else
#define UR_OUT(X)
#endif

// This contains info used when building the body of a method.  It is destroyed
// when the method is completed.
//
typedef vector<Value *> ValueList;           // Numbered defs
static void ResolveDefinitions(vector<ValueList> &LateResolvers);
static void ResolveTypes      (vector<PATypeHolder<Type> > &LateResolveTypes);

static struct PerModuleInfo {
  Module *CurrentModule;
  vector<ValueList>    Values;     // Module level numbered definitions
  vector<ValueList>    LateResolveValues;
  vector<PATypeHolder<Type> > Types, LateResolveTypes;

  void ModuleDone() {
    // If we could not resolve some methods at method compilation time (calls to
    // methods before they are defined), resolve them now...  Types are resolved
    // when the constant pool has been completely parsed.
    //
    ResolveDefinitions(LateResolveValues);

    Values.clear();         // Clear out method local definitions
    Types.clear();
    CurrentModule = 0;
  }
} CurModule;

static struct PerMethodInfo {
  Method *CurrentMethod;         // Pointer to current method being created

  vector<ValueList> Values;      // Keep track of numbered definitions
  vector<ValueList> LateResolveValues;
  vector<PATypeHolder<Type> > Types, LateResolveTypes;
  bool isDeclare;                // Is this method a forward declararation?

  inline PerMethodInfo() {
    CurrentMethod = 0;
    isDeclare = false;
  }

  inline ~PerMethodInfo() {}

  inline void MethodStart(Method *M) {
    CurrentMethod = M;
  }

  void MethodDone() {
    // If we could not resolve some blocks at parsing time (forward branches)
    // resolve the branches now...
    ResolveDefinitions(LateResolveValues);

    Values.clear();         // Clear out method local definitions
    Types.clear();
    CurrentMethod = 0;
    isDeclare = false;
  }
} CurMeth;  // Info for the current method...


//===----------------------------------------------------------------------===//
//               Code to handle definitions of all the types
//===----------------------------------------------------------------------===//

static void InsertValue(Value *D, vector<ValueList> &ValueTab = CurMeth.Values){
  if (!D->hasName()) {             // Is this a numbered definition?
    unsigned type = D->getType()->getUniqueID();
    if (ValueTab.size() <= type)
      ValueTab.resize(type+1, ValueList());
    //printf("Values[%d][%d] = %d\n", type, ValueTab[type].size(), D);
    ValueTab[type].push_back(D);
  }
}

// TODO: FIXME when Type are not const
static void InsertType(const Type *Ty, vector<PATypeHolder<Type> > &Types) {
  Types.push_back(Ty);
}

static const Type *getTypeVal(const ValID &D, bool DoNotImprovise = false) {
  switch (D.Type) {
  case 0: {                 // Is it a numbered definition?
    unsigned Num = (unsigned)D.Num;

    // Module constants occupy the lowest numbered slots...
    if (Num < CurModule.Types.size()) 
      return CurModule.Types[Num];

    Num -= CurModule.Types.size();

    // Check that the number is within bounds...
    if (Num <= CurMeth.Types.size())
      return CurMeth.Types[Num];
  }
  case 1: {                // Is it a named definition?
    string Name(D.Name);
    SymbolTable *SymTab = 0;
    if (CurMeth.CurrentMethod) 
      SymTab = CurMeth.CurrentMethod->getSymbolTable();
    Value *N = SymTab ? SymTab->lookup(Type::TypeTy, Name) : 0;

    if (N == 0) {
      // Symbol table doesn't automatically chain yet... because the method
      // hasn't been added to the module...
      //
      SymTab = CurModule.CurrentModule->getSymbolTable();
      if (SymTab)
        N = SymTab->lookup(Type::TypeTy, Name);
      if (N == 0) break;
    }

    D.destroy();  // Free old strdup'd memory...
    return N->castTypeAsserting();
  }
  default:
    ThrowException("Invalid symbol type reference!");
  }

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  if (DoNotImprovise) return 0;  // Do we just want a null to be returned?

  vector<PATypeHolder<Type> > *LateResolver = CurMeth.CurrentMethod ? 
    &CurMeth.LateResolveTypes : &CurModule.LateResolveTypes;

  Type *Typ = new TypePlaceHolder(Type::TypeTy, D);
  InsertType(Typ, *LateResolver);
  return Typ;
}

static Value *getVal(const Type *Ty, const ValID &D, 
                     bool DoNotImprovise = false) {
  assert(Ty != Type::TypeTy && "Should use getTypeVal for types!");

  switch (D.Type) {
  case 0: {                 // Is it a numbered definition?
    unsigned type = Ty->getUniqueID();
    unsigned Num = (unsigned)D.Num;

    // Module constants occupy the lowest numbered slots...
    if (type < CurModule.Values.size()) {
      if (Num < CurModule.Values[type].size()) 
        return CurModule.Values[type][Num];

      Num -= CurModule.Values[type].size();
    }

    // Make sure that our type is within bounds
    if (CurMeth.Values.size() <= type)
      break;

    // Check that the number is within bounds...
    if (CurMeth.Values[type].size() <= Num)
      break;
  
    return CurMeth.Values[type][Num];
  }
  case 1: {                // Is it a named definition?
    string Name(D.Name);
    SymbolTable *SymTab = 0;
    if (CurMeth.CurrentMethod) 
      SymTab = CurMeth.CurrentMethod->getSymbolTable();
    Value *N = SymTab ? SymTab->lookup(Ty, Name) : 0;

    if (N == 0) {
      // Symbol table doesn't automatically chain yet... because the method
      // hasn't been added to the module...
      //
      SymTab = CurModule.CurrentModule->getSymbolTable();
      if (SymTab)
        N = SymTab->lookup(Ty, Name);
      if (N == 0) break;
    }

    D.destroy();  // Free old strdup'd memory...
    return N;
  }

  case 2:                 // Is it a constant pool reference??
  case 3:                 // Is it an unsigned const pool reference?
  case 4:                 // Is it a string const pool reference?
  case 5:{                // Is it a floating point const pool reference?
    ConstPoolVal *CPV = 0;

    // Check to make sure that "Ty" is an integral type, and that our 
    // value will fit into the specified type...
    switch (D.Type) {
    case 2:
      if (Ty == Type::BoolTy) {  // Special handling for boolean data
        CPV = ConstPoolBool::get(D.ConstPool64 != 0);
      } else {
        if (!ConstPoolSInt::isValueValidForType(Ty, D.ConstPool64))
          ThrowException("Symbolic constant pool value '" +
			 itostr(D.ConstPool64) + "' is invalid for type '" + 
			 Ty->getName() + "'!");
        CPV = ConstPoolSInt::get(Ty, D.ConstPool64);
      }
      break;
    case 3:
      if (!ConstPoolUInt::isValueValidForType(Ty, D.UConstPool64)) {
        if (!ConstPoolSInt::isValueValidForType(Ty, D.ConstPool64)) {
          ThrowException("Integral constant pool reference is invalid!");
        } else {     // This is really a signed reference.  Transmogrify.
          CPV = ConstPoolSInt::get(Ty, D.ConstPool64);
        }
      } else {
        CPV = ConstPoolUInt::get(Ty, D.UConstPool64);
      }
      break;
    case 4:
      cerr << "FIXME: TODO: String constants [sbyte] not implemented yet!\n";
      abort();
      break;
    case 5:
      if (!ConstPoolFP::isValueValidForType(Ty, D.ConstPoolFP))
	ThrowException("FP constant invalid for type!!");
      else
	CPV = ConstPoolFP::get(Ty, D.ConstPoolFP);
      break;
    }
    assert(CPV && "How did we escape creating a constant??");
    return CPV;
  }   // End of case 2,3,4
  default:
    assert(0 && "Unhandled case!");
  }   // End of switch


  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  if (DoNotImprovise) return 0;  // Do we just want a null to be returned?

  Value *d = 0;
  vector<ValueList> *LateResolver =  (CurMeth.CurrentMethod) ? 
    &CurMeth.LateResolveValues : &CurModule.LateResolveValues;

  switch (Ty->getPrimitiveID()) {
  case Type::LabelTyID:  d = new   BBPlaceHolder(Ty, D); break;
  case Type::MethodTyID: d = new MethPlaceHolder(Ty, D); 
                         LateResolver = &CurModule.LateResolveValues; break;
  default:               d = new ValuePlaceHolder(Ty, D); break;
  }

  assert(d != 0 && "How did we not make something?");
  InsertValue(d, *LateResolver);
  return d;
}


//===----------------------------------------------------------------------===//
//              Code to handle forward references in instructions
//===----------------------------------------------------------------------===//
//
// This code handles the late binding needed with statements that reference
// values not defined yet... for example, a forward branch, or the PHI node for
// a loop body.
//
// This keeps a table (CurMeth.LateResolveValues) of all such forward references
// and back patchs after we are done.
//

// ResolveDefinitions - If we could not resolve some defs at parsing 
// time (forward branches, phi functions for loops, etc...) resolve the 
// defs now...
//
static void ResolveDefinitions(vector<ValueList> &LateResolvers) {
  // Loop over LateResolveDefs fixing up stuff that couldn't be resolved
  for (unsigned ty = 0; ty < LateResolvers.size(); ty++) {
    while (!LateResolvers[ty].empty()) {
      Value *V = LateResolvers[ty].back();
      LateResolvers[ty].pop_back();
      ValID &DID = getValIDFromPlaceHolder(V);

      Value *TheRealValue = getVal(Type::getUniqueIDType(ty), DID, true);

      if (TheRealValue == 0) {
	if (DID.Type == 1)
	  ThrowException("Reference to an invalid definition: '" +DID.getName()+
			 "' of type '" + V->getType()->getDescription() + "'",
			 getLineNumFromPlaceHolder(V));
	else
	  ThrowException("Reference to an invalid definition: #" +
			 itostr(DID.Num) + " of type '" + 
			 V->getType()->getDescription() + "'",
			 getLineNumFromPlaceHolder(V));
      }

      assert(!V->isType() && "Types should be in LateResolveTypes!");

      V->replaceAllUsesWith(TheRealValue);
      delete V;
    }
  }

  LateResolvers.clear();
}


// ResolveTypes - This goes through the forward referenced type table and makes
// sure that all type references are complete.  This code is executed after the
// constant pool of a method or module is completely parsed.
//
static void ResolveTypes(vector<PATypeHolder<Type> > &LateResolveTypes) {
  while (!LateResolveTypes.empty()) {
    const Type *Ty = LateResolveTypes.back();
    ValID &DID = getValIDFromPlaceHolder(Ty);

    const Type *TheRealType = getTypeVal(DID, true);
    if (TheRealType == 0) {
      if (DID.Type == 1)
	ThrowException("Reference to an invalid type: '" +DID.getName(),
		       getLineNumFromPlaceHolder(Ty));
      else
	ThrowException("Reference to an invalid type: #" + itostr(DID.Num),
		       getLineNumFromPlaceHolder(Ty));
    }

    // FIXME: When types are not const
    DerivedType *DTy = const_cast<DerivedType*>(Ty->castDerivedTypeAsserting());
    
    // Refine the opaque type we had to the new type we are getting.
    DTy->refineAbstractTypeTo(TheRealType);

    // No need to delete type, refine does that for us.
    LateResolveTypes.pop_back();
  }
}

// setValueName - Set the specified value to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is freed by this function.
//
static void setValueName(Value *V, char *NameStr) {
  if (NameStr == 0) return;
  string Name(NameStr);           // Copy string
  free(NameStr);                  // Free old string

  SymbolTable *ST = CurMeth.CurrentMethod ? 
    CurMeth.CurrentMethod->getSymbolTableSure() : 
    CurModule.CurrentModule->getSymbolTableSure();

  Value *Existing = ST->lookup(V->getType(), Name);
  if (Existing) {    // Inserting a name that is already defined???
    // There is only one case where this is allowed: when we are refining an
    // opaque type.  In this case, Existing will be an opaque type.
    if (const Type *Ty = Existing->castType())
      if (Ty->isOpaqueType()) {
	// We ARE replacing an opaque type!

	// TODO: FIXME when types are not const!
	const_cast<DerivedType*>(Ty->castDerivedTypeAsserting())->refineAbstractTypeTo(V->castTypeAsserting());
	return;
      }

    // Otherwise, we are a simple redefinition of a value, baaad
    ThrowException("Redefinition of value name '" + Name + "' in the '" +
		   V->getType()->getDescription() + "' type plane!");
  }

  V->setName(Name, ST);
}


//===----------------------------------------------------------------------===//
// Code for handling upreferences in type names...
//

// TypeContains - Returns true if Ty contains E in it.
//
static bool TypeContains(const Type *Ty, const Type *E) {
  return find(df_begin(Ty), df_end(Ty), E) != df_end(Ty);
}


static vector<pair<unsigned, OpaqueType *> > UpRefs;

static PATypeHolder<Type> HandleUpRefs(const Type *ty) {
  PATypeHolder<Type> Ty(ty);
  UR_OUT(UpRefs.size() << " upreferences active!\n");
  for (unsigned i = 0; i < UpRefs.size(); ) {
    UR_OUT("TypeContains(" << Ty->getDescription() << ", " 
	   << UpRefs[i].second->getDescription() << ") = " 
	   << TypeContains(Ty, UpRefs[i].second) << endl);
    if (TypeContains(Ty, UpRefs[i].second)) {
      unsigned Level = --UpRefs[i].first;   // Decrement level of upreference
      UR_OUT("Uplevel Ref Level = " << Level << endl);
      if (Level == 0) {                     // Upreference should be resolved! 
	UR_OUT("About to resolve upreference!\n";
	       string OldName = UpRefs[i].second->getDescription());
	UpRefs[i].second->refineAbstractTypeTo(Ty);
	UpRefs.erase(UpRefs.begin()+i);     // Remove from upreference list...
	UR_OUT("Type '" << OldName << "' refined upreference to: "
	       << (const void*)Ty << ", " << Ty->getDescription() << endl);
	continue;
      }
    }

    ++i;                                  // Otherwise, no resolve, move on...
  }
  // FIXME: TODO: this should return the updated type
  return Ty;
}

template <class TypeTy>
inline static void TypeDone(PATypeHolder<TypeTy> *Ty) {
  if (UpRefs.size())
    ThrowException("Invalid upreference in type: " + (*Ty)->getDescription());
}

// newTH - Allocate a new type holder for the specified type
template <class TypeTy>
inline static PATypeHolder<TypeTy> *newTH(const TypeTy *Ty) {
  return new PATypeHolder<TypeTy>(Ty);
}
template <class TypeTy>
inline static PATypeHolder<TypeTy> *newTH(const PATypeHolder<TypeTy> &TH) {
  return new PATypeHolder<TypeTy>(TH);
}


// newTHC - Allocate a new type holder for the specified type that can be
// casted to a new Type type.
template <class TypeTy, class OldTy>
inline static PATypeHolder<TypeTy> *newTHC(const PATypeHolder<OldTy> &Old) {
  return new PATypeHolder<TypeTy>((const TypeTy*)Old.get());
}


//===----------------------------------------------------------------------===//
//            RunVMAsmParser - Define an interface to this parser
//===----------------------------------------------------------------------===//
//
Module *RunVMAsmParser(const string &Filename, FILE *F) {
  llvmAsmin = F;
  CurFilename = Filename;
  llvmAsmlineno = 1;      // Reset the current line number...

  CurModule.CurrentModule = new Module();  // Allocate a new module to read
  yyparse();       // Parse the file.
  Module *Result = ParserResult;
  llvmAsmin = stdin;    // F is about to go away, don't use it anymore...
  ParserResult = 0;

  return Result;
}

%}

%union {
  Module                           *ModuleVal;
  Method                           *MethodVal;
  MethodArgument                   *MethArgVal;
  BasicBlock                       *BasicBlockVal;
  TerminatorInst                   *TermInstVal;
  Instruction                      *InstVal;
  ConstPoolVal                     *ConstVal;

  const Type                       *PrimType;
  PATypeHolder<Type>               *TypeVal;
  PATypeHolder<ArrayType>          *ArrayTypeTy;
  PATypeHolder<StructType>         *StructTypeTy;
  Value                            *ValueVal;

  list<MethodArgument*>            *MethodArgList;
  list<Value*>                     *ValueList;
  list<PATypeHolder<Type> >        *TypeList;
  list<pair<Value*, BasicBlock*> > *PHIList;   // Represent the RHS of PHI node
  list<pair<ConstPoolVal*, BasicBlock*> > *JumpTable;
  vector<ConstPoolVal*>            *ConstVector;

  int64_t                           SInt64Val;
  uint64_t                          UInt64Val;
  int                               SIntVal;
  unsigned                          UIntVal;
  double                            FPVal;
  bool                              BoolVal;

  char                             *StrVal;   // This memory is strdup'd!
  ValID                             ValIDVal; // strdup'd memory maybe!

  Instruction::UnaryOps             UnaryOpVal;
  Instruction::BinaryOps            BinaryOpVal;
  Instruction::TermOps              TermOpVal;
  Instruction::MemoryOps            MemOpVal;
  Instruction::OtherOps             OtherOpVal;
}

%type <ModuleVal>     Module MethodList
%type <MethodVal>     Method MethodProto MethodHeader BasicBlockList
%type <BasicBlockVal> BasicBlock InstructionList
%type <TermInstVal>   BBTerminatorInst
%type <InstVal>       Inst InstVal MemoryInst
%type <ConstVal>      ConstVal ExtendedConstVal
%type <ConstVector>   ConstVector UByteList
%type <MethodArgList> ArgList ArgListH
%type <MethArgVal>    ArgVal
%type <PHIList>       PHIList
%type <ValueList>     ValueRefList ValueRefListE  // For call param lists
%type <TypeList>      TypeListI ArgTypeListI
%type <JumpTable>     JumpTable
%type <BoolVal>       GlobalType                  // GLOBAL or CONSTANT?

%type <ValIDVal>      ValueRef ConstValueRef // Reference to a definition or BB
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
%token <TypeVal>  OPAQUE
%token <PrimType> VOID BOOL SBYTE UBYTE SHORT USHORT INT UINT LONG ULONG
%token <PrimType> FLOAT DOUBLE TYPE LABEL
%type  <ArrayTypeTy> ArrayType ArrayTypeI
%type  <StructTypeTy> StructType StructTypeI

%token <StrVal>     VAR_ID LABELSTR STRINGCONSTANT
%type  <StrVal>  OptVAR_ID OptAssign


%token IMPLEMENTATION TRUE FALSE BEGINTOK END DECLARE GLOBAL CONSTANT UNINIT
%token TO DOTDOTDOT STRING

// Basic Block Terminating Operators 
%token <TermOpVal> RET BR SWITCH

// Unary Operators 
%type  <UnaryOpVal> UnaryOps  // all the unary operators
%token <UnaryOpVal> NOT

// Binary Operators 
%type  <BinaryOpVal> BinaryOps  // all the binary operators
%token <BinaryOpVal> ADD SUB MUL DIV REM
%token <BinaryOpVal> SETLE SETGE SETLT SETGT SETEQ SETNE  // Binary Comarators

// Memory Instructions
%token <MemoryOpVal> MALLOC ALLOCA FREE LOAD STORE GETELEMENTPTR

// Other Operators
%type  <OtherOpVal> ShiftOps
%token <OtherOpVal> PHI CALL CAST SHL SHR

%start Module
%%

// Handle constant integer size restriction and conversion...
//

INTVAL : SINTVAL
INTVAL : UINTVAL {
  if ($1 > (uint32_t)INT32_MAX)     // Outside of my range!
    ThrowException("Value too large for type!");
  $$ = (int32_t)$1;
}


EINT64VAL : ESINT64VAL       // These have same type and can't cause problems...
EINT64VAL : EUINT64VAL {
  if ($1 > (uint64_t)INT64_MAX)     // Outside of my range!
    ThrowException("Value too large for type!");
  $$ = (int64_t)$1;
}

// Operations that are notably excluded from this list include: 
// RET, BR, & SWITCH because they end basic blocks and are treated specially.
//
UnaryOps  : NOT
BinaryOps : ADD | SUB | MUL | DIV | REM
BinaryOps : SETLE | SETGE | SETLT | SETGT | SETEQ | SETNE
ShiftOps  : SHL | SHR

// These are some types that allow classification if we only want a particular 
// thing... for example, only a signed, unsigned, or integral type.
SIntType :  LONG |  INT |  SHORT | SBYTE
UIntType : ULONG | UINT | USHORT | UBYTE
IntType  : SIntType | UIntType
FPType   : FLOAT | DOUBLE

// OptAssign - Value producing statements have an optional assignment component
OptAssign : VAR_ID '=' {
    $$ = $1;
  }
  | /*empty*/ { 
    $$ = 0; 
  }


//===----------------------------------------------------------------------===//
// Types includes all predefined types... except void, because it can only be
// used in specific contexts (method returning void for example).  To have
// access to it, a user must explicitly use TypesV.
//

// TypesV includes all of 'Types', but it also includes the void type.
TypesV    : Types    | VOID { $$ = newTH($1); }
UpRTypesV : UpRTypes | VOID { $$ = newTH($1); }

Types     : UpRTypes {
    TypeDone($$ = $1);
  }


// Derived types are added later...
//
PrimType : BOOL | SBYTE | UBYTE | SHORT  | USHORT | INT   | UINT 
PrimType : LONG | ULONG | FLOAT | DOUBLE | TYPE   | LABEL
UpRTypes : OPAQUE | PrimType { $$ = newTH($1); }
UpRTypes : ValueRef {                    // Named types are also simple types...
  $$ = newTH(getTypeVal($1));
}

// ArrayTypeI - Internal version of ArrayType that can have incomplete uprefs
//
ArrayTypeI : '[' UpRTypesV ']' {               // Unsized array type?
    $$ = newTHC<ArrayType>(HandleUpRefs(ArrayType::get(*$2)));
    delete $2;
  }
  | '[' EUINT64VAL 'x' UpRTypes ']' {          // Sized array type?
    $$ = newTHC<ArrayType>(HandleUpRefs(ArrayType::get(*$4, (int)$2)));
    delete $4;
  }

StructTypeI : '{' TypeListI '}' {              // Structure type?
    vector<const Type*> Elements;
    mapto($2->begin(), $2->end(), back_inserter(Elements), 
	mem_fun_ref(&PATypeHandle<Type>::get));

    $$ = newTHC<StructType>(HandleUpRefs(StructType::get(Elements)));
    delete $2;
  }
  | '{' '}' {                                  // Empty structure type?
    $$ = newTH(StructType::get(vector<const Type*>()));
  }


// Include derived types in the Types production.
//
UpRTypes : '\\' EUINT64VAL {                   // Type UpReference
    if ($2 > (uint64_t)INT64_MAX) ThrowException("Value out of range!");
    OpaqueType *OT = OpaqueType::get();        // Use temporary placeholder
    UpRefs.push_back(make_pair((unsigned)$2, OT));  // Add to vector...
    $$ = newTH<Type>(OT);
    UR_OUT("New Upreference!\n");
  }
  | UpRTypesV '(' ArgTypeListI ')' {           // Method derived type?
    vector<const Type*> Params;
    mapto($3->begin(), $3->end(), back_inserter(Params), 
	  mem_fun_ref(&PATypeHandle<Type>::get));
    $$ = newTH(HandleUpRefs(MethodType::get(*$1, Params)));
    delete $3;      // Delete the argument list
    delete $1;      // Delete the old type handle
  }
  | ArrayTypeI {                               // [Un]sized array type?
    $$ = newTHC<Type>(*$1); delete $1;
  }
  | StructTypeI {                              // Structure type?
    $$ = newTHC<Type>(*$1); delete $1;
  }
  | UpRTypes '*' {                             // Pointer type?
    $$ = newTH(HandleUpRefs(PointerType::get(*$1)));
    delete $1;  // Delete the type handle
  }

// Define some helpful top level types that do not allow UpReferences to escape
//
ArrayType  : ArrayTypeI  { TypeDone($$ = $1); }
StructType : StructTypeI { TypeDone($$ = $1); }



// TypeList - Used for struct declarations and as a basis for method type 
// declaration type lists
//
TypeListI : UpRTypes {
    $$ = new list<PATypeHolder<Type> >();
    $$->push_back(*$1); delete $1;
  }
  | TypeListI ',' UpRTypes {
    ($$=$1)->push_back(*$3); delete $3;
  }

// ArgTypeList - List of types for a method type declaration...
ArgTypeListI : TypeListI
  | TypeListI ',' DOTDOTDOT {
    ($$=$1)->push_back(Type::VoidTy);
  }
  | DOTDOTDOT {
    ($$ = new list<PATypeHolder<Type> >())->push_back(Type::VoidTy);
  }
  | /*empty*/ {
    $$ = new list<PATypeHolder<Type> >();
  }


// ConstVal - The various declarations that go into the constant pool.  This
// includes all forward declarations of types, constants, and functions.
//
// This is broken into two sections: ExtendedConstVal and ConstVal
//
ExtendedConstVal: ArrayType '[' ConstVector ']' { // Nonempty unsized arr
    const ArrayType *ATy = *$1;
    const Type *ETy = ATy->getElementType();
    int NumElements = ATy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != -1 && NumElements != (int)$3->size())
      ThrowException("Type mismatch: constant sized array initialized with " +
		     utostr($3->size()) +  " arguments, but has size of " + 
		     itostr(NumElements) + "!");

    // Verify all elements are correct type!
    for (unsigned i = 0; i < $3->size(); i++) {
      if (ETy != (*$3)[i]->getType())
	ThrowException("Element #" + utostr(i) + " is not of type '" + 
		       ETy->getName() + "' as required!\nIt is of type '" +
		       (*$3)[i]->getType()->getName() + "'.");
    }

    $$ = ConstPoolArray::get(ATy, *$3);
    delete $1; delete $3;
  }
  | ArrayType '[' ']' {
    int NumElements = (*$1)->getNumElements();
    if (NumElements != -1 && NumElements != 0) 
      ThrowException("Type mismatch: constant sized array initialized with 0"
		     " arguments, but has size of " + itostr(NumElements) +"!");
    $$ = ConstPoolArray::get((*$1), vector<ConstPoolVal*>());
    delete $1;
  }
  | ArrayType 'c' STRINGCONSTANT {
    const ArrayType *ATy = *$1;
    int NumElements = ATy->getNumElements();
    const Type *ETy = ATy->getElementType();
    char *EndStr = UnEscapeLexed($3, true);
    if (NumElements != -1 && NumElements != (EndStr-$3))
      ThrowException("Can't build string constant of size " + 
		     itostr((int)(EndStr-$3)) +
		     " when array has size " + itostr(NumElements) + "!");
    vector<ConstPoolVal*> Vals;
    if (ETy == Type::SByteTy) {
      for (char *C = $3; C != EndStr; ++C)
	Vals.push_back(ConstPoolSInt::get(ETy, *C));
    } else if (ETy == Type::UByteTy) {
      for (char *C = $3; C != EndStr; ++C)
	Vals.push_back(ConstPoolUInt::get(ETy, *C));
    } else {
      free($3);
      ThrowException("Cannot build string arrays of non byte sized elements!");
    }
    free($3);
    $$ = ConstPoolArray::get(ATy, Vals);
    delete $1;
  }
  | StructType '{' ConstVector '}' {
    // FIXME: TODO: Check to see that the constants are compatible with the type
    // initializer!
    $$ = ConstPoolStruct::get(*$1, *$3);
    delete $1; delete $3;
  }
/*
  | Types '*' ConstVal {
    assert(0);
    $$ = 0;
  }
*/

ConstVal : ExtendedConstVal {
    $$ = $1;
  }
  | SIntType EINT64VAL {     // integral constants
    if (!ConstPoolSInt::isValueValidForType($1, $2))
      ThrowException("Constant value doesn't fit in type!");
    $$ = ConstPoolSInt::get($1, $2);
  } 
  | UIntType EUINT64VAL {           // integral constants
    if (!ConstPoolUInt::isValueValidForType($1, $2))
      ThrowException("Constant value doesn't fit in type!");
    $$ = ConstPoolUInt::get($1, $2);
  } 
  | BOOL TRUE {                     // Boolean constants
    $$ = ConstPoolBool::True;
  }
  | BOOL FALSE {                    // Boolean constants
    $$ = ConstPoolBool::False;
  }
  | FPType FPVAL {                   // Float & Double constants
    $$ = ConstPoolFP::get($1, $2);
  }

// ConstVector - A list of comma seperated constants.
ConstVector : ConstVector ',' ConstVal {
    ($$ = $1)->push_back($3);
  }
  | ConstVal {
    $$ = new vector<ConstPoolVal*>();
    $$->push_back($1);
  }


// GlobalType - Match either GLOBAL or CONSTANT for global declarations...
GlobalType : GLOBAL { $$ = false; } | CONSTANT { $$ = true; }


// ConstPool - Constants with optional names assigned to them.
ConstPool : ConstPool OptAssign ConstVal { 
    setValueName($3, $2);
    InsertValue($3);
  }
  | ConstPool OptAssign TYPE TypesV {  // Types can be defined in the const pool
    // TODO: FIXME when Type are not const
    setValueName(const_cast<Type*>($4->get()), $2);

    if (!$2) {
      InsertType($4->get(),
		 CurMeth.CurrentMethod ? CurMeth.Types : CurModule.Types);
    }
    delete $4;
  }
  | ConstPool MethodProto {            // Method prototypes can be in const pool
  }
  | ConstPool OptAssign GlobalType ResolvedVal {
    const Type *Ty = $4->getType();
    // Global declarations appear in Constant Pool
    ConstPoolVal *Initializer = $4->castConstant();
    if (Initializer == 0)
      ThrowException("Global value initializer is not a constant!");
	 
    GlobalVariable *GV = new GlobalVariable(PointerType::get(Ty), $3,
					    Initializer);
    setValueName(GV, $2);

    CurModule.CurrentModule->getGlobalList().push_back(GV);
    InsertValue(GV, CurModule.Values);
  }
  | ConstPool OptAssign UNINIT GlobalType Types {
    const Type *Ty = *$5;
    // Global declarations appear in Constant Pool
    if (Ty->isArrayType() && Ty->castArrayType()->isUnsized()) {
      ThrowException("Type '" + Ty->getDescription() +
		     "' is not a sized type!");
    }

    GlobalVariable *GV = new GlobalVariable(PointerType::get(Ty), $4);
    setValueName(GV, $2);

    CurModule.CurrentModule->getGlobalList().push_back(GV);
    InsertValue(GV, CurModule.Values);
  }
  | /* empty: end of list */ { 
  }


//===----------------------------------------------------------------------===//
//                             Rules to match Modules
//===----------------------------------------------------------------------===//

// Module rule: Capture the result of parsing the whole file into a result
// variable...
//
Module : MethodList {
  $$ = ParserResult = $1;
  CurModule.ModuleDone();
}

// MethodList - A list of methods, preceeded by a constant pool.
//
MethodList : MethodList Method {
    $$ = $1;
    if (!$2->getParent())
      $1->getMethodList().push_back($2);
    CurMeth.MethodDone();
  } 
  | MethodList MethodProto {
    $$ = $1;
  }
  | ConstPool IMPLEMENTATION {
    $$ = CurModule.CurrentModule;
    // Resolve circular types before we parse the body of the module
    ResolveTypes(CurModule.LateResolveTypes);
  }


//===----------------------------------------------------------------------===//
//                       Rules to match Method Headers
//===----------------------------------------------------------------------===//

OptVAR_ID : VAR_ID | /*empty*/ { $$ = 0; }

ArgVal : Types OptVAR_ID {
  $$ = new MethodArgument(*$1); delete $1;
  setValueName($$, $2);
}

ArgListH : ArgVal ',' ArgListH {
    $$ = $3;
    $3->push_front($1);
  }
  | ArgVal {
    $$ = new list<MethodArgument*>();
    $$->push_front($1);
  }
  | DOTDOTDOT {
    $$ = new list<MethodArgument*>();
    $$->push_back(new MethodArgument(Type::VoidTy));
  }

ArgList : ArgListH {
    $$ = $1;
  }
  | /* empty */ {
    $$ = 0;
  }

MethodHeaderH : TypesV STRINGCONSTANT '(' ArgList ')' {
  UnEscapeLexed($2);
  vector<const Type*> ParamTypeList;
  if ($4)
    for (list<MethodArgument*>::iterator I = $4->begin(); I != $4->end(); ++I)
      ParamTypeList.push_back((*I)->getType());

  const MethodType *MT = MethodType::get(*$1, ParamTypeList);
  delete $1;

  Method *M = 0;
  if (SymbolTable *ST = CurModule.CurrentModule->getSymbolTable()) {
    if (Value *V = ST->lookup(MT, $2)) {  // Method already in symtab?
      M = V->castMethodAsserting();

      // Yes it is.  If this is the case, either we need to be a forward decl,
      // or it needs to be.
      if (!CurMeth.isDeclare && !M->isExternal())
	ThrowException("Redefinition of method '" + string($2) + "'!");      
    }
  }

  if (M == 0) {  // Not already defined?
    M = new Method(MT, $2);
    InsertValue(M, CurModule.Values);
  }

  free($2);  // Free strdup'd memory!

  CurMeth.MethodStart(M);

  // Add all of the arguments we parsed to the method...
  if ($4 && !CurMeth.isDeclare) {        // Is null if empty...
    Method::ArgumentListType &ArgList = M->getArgumentList();

    for (list<MethodArgument*>::iterator I = $4->begin(); I != $4->end(); ++I) {
      InsertValue(*I);
      ArgList.push_back(*I);
    }
    delete $4;                     // We're now done with the argument list
  }
}

MethodHeader : MethodHeaderH ConstPool BEGINTOK {
  $$ = CurMeth.CurrentMethod;

  // Resolve circular types before we parse the body of the method.
  ResolveTypes(CurMeth.LateResolveTypes);
}

Method : BasicBlockList END {
  $$ = $1;
}

MethodProto : DECLARE { CurMeth.isDeclare = true; } MethodHeaderH {
  $$ = CurMeth.CurrentMethod;
  if (!$$->getParent())
    CurModule.CurrentModule->getMethodList().push_back($$);
  CurMeth.MethodDone();
}

//===----------------------------------------------------------------------===//
//                        Rules to match Basic Blocks
//===----------------------------------------------------------------------===//

ConstValueRef : ESINT64VAL {    // A reference to a direct constant
    $$ = ValID::create($1);
  }
  | EUINT64VAL {
    $$ = ValID::create($1);
  }
  | FPVAL {                     // Perhaps it's an FP constant?
    $$ = ValID::create($1);
  }
  | TRUE {
    $$ = ValID::create((int64_t)1);
  } 
  | FALSE {
    $$ = ValID::create((int64_t)0);
  }
/*
  | STRINGCONSTANT {        // Quoted strings work too... especially for methods
    $$ = ValID::create_conststr($1);
  }
*/

// ValueRef - A reference to a definition... 
ValueRef : INTVAL {           // Is it an integer reference...?
    $$ = ValID::create($1);
  }
  | VAR_ID {                 // Is it a named reference...?
    $$ = ValID::create($1);
  }
  | ConstValueRef {
    $$ = $1;
  }

// ResolvedVal - a <type> <value> pair.  This is used only in cases where the
// type immediately preceeds the value reference, and allows complex constant
// pool references (for things like: 'ret [2 x int] [ int 12, int 42]')
ResolvedVal : ExtendedConstVal {
    $$ = $1;
  }
  | Types ValueRef {
    $$ = getVal(*$1, $2); delete $1;
  }


BasicBlockList : BasicBlockList BasicBlock {
    $1->getBasicBlocks().push_back($2);
    $$ = $1;
  }
  | MethodHeader BasicBlock { // Do not allow methods with 0 basic blocks   
    $$ = $1;                  // in them...
    $1->getBasicBlocks().push_back($2);
  }


// Basic blocks are terminated by branching instructions: 
// br, br/cc, switch, ret
//
BasicBlock : InstructionList BBTerminatorInst  {
    $1->getInstList().push_back($2);
    InsertValue($1);
    $$ = $1;
  }
  | LABELSTR InstructionList BBTerminatorInst  {
    $2->getInstList().push_back($3);
    setValueName($2, $1);

    InsertValue($2);
    $$ = $2;
  }

InstructionList : InstructionList Inst {
    $1->getInstList().push_back($2);
    $$ = $1;
  }
  | /* empty */ {
    $$ = new BasicBlock();
  }

BBTerminatorInst : RET ResolvedVal {              // Return with a result...
    $$ = new ReturnInst($2);
  }
  | RET VOID {                                       // Return with no result...
    $$ = new ReturnInst();
  }
  | BR LABEL ValueRef {                         // Unconditional Branch...
    $$ = new BranchInst(getVal(Type::LabelTy, $3)->castBasicBlockAsserting());
  }                                                  // Conditional Branch...
  | BR BOOL ValueRef ',' LABEL ValueRef ',' LABEL ValueRef {  
    $$ = new BranchInst(getVal(Type::LabelTy, $6)->castBasicBlockAsserting(), 
			getVal(Type::LabelTy, $9)->castBasicBlockAsserting(),
			getVal(Type::BoolTy, $3));
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' JumpTable ']' {
    SwitchInst *S = new SwitchInst(getVal($2, $3), 
                          getVal(Type::LabelTy, $6)->castBasicBlockAsserting());
    $$ = S;

    list<pair<ConstPoolVal*, BasicBlock*> >::iterator I = $8->begin(), 
                                                      end = $8->end();
    for (; I != end; ++I)
      S->dest_push_back(I->first, I->second);
  }

JumpTable : JumpTable IntType ConstValueRef ',' LABEL ValueRef {
    $$ = $1;
    ConstPoolVal *V = getVal($2, $3, true)->castConstantAsserting();
    if (V == 0)
      ThrowException("May only switch on a constant pool value!");

    $$->push_back(make_pair(V, getVal($5, $6)->castBasicBlockAsserting()));
  }
  | IntType ConstValueRef ',' LABEL ValueRef {
    $$ = new list<pair<ConstPoolVal*, BasicBlock*> >();
    ConstPoolVal *V = getVal($1, $2, true)->castConstantAsserting();

    if (V == 0)
      ThrowException("May only switch on a constant pool value!");

    $$->push_back(make_pair(V, getVal($4, $5)->castBasicBlockAsserting()));
  }

Inst : OptAssign InstVal {
  setValueName($2, $1);  // Is this definition named?? if so, assign the name...

  InsertValue($2);
  $$ = $2;
}

PHIList : Types '[' ValueRef ',' ValueRef ']' {    // Used for PHI nodes
    $$ = new list<pair<Value*, BasicBlock*> >();
    $$->push_back(make_pair(getVal(*$1, $3), 
			 getVal(Type::LabelTy, $5)->castBasicBlockAsserting()));
    delete $1;
  }
  | PHIList ',' '[' ValueRef ',' ValueRef ']' {
    $$ = $1;
    $1->push_back(make_pair(getVal($1->front().first->getType(), $4),
			 getVal(Type::LabelTy, $6)->castBasicBlockAsserting()));
  }


ValueRefList : ResolvedVal {    // Used for call statements, and memory insts...
    $$ = new list<Value*>();
    $$->push_back($1);
  }
  | ValueRefList ',' ResolvedVal {
    $$ = $1;
    $1->push_back($3);
  }

// ValueRefListE - Just like ValueRefList, except that it may also be empty!
ValueRefListE : ValueRefList | /*empty*/ { $$ = 0; }

InstVal : BinaryOps Types ValueRef ',' ValueRef {
    $$ = BinaryOperator::create($1, getVal(*$2, $3), getVal(*$2, $5));
    if ($$ == 0)
      ThrowException("binary operator returned null!");
    delete $2;
  }
  | UnaryOps ResolvedVal {
    $$ = UnaryOperator::create($1, $2);
    if ($$ == 0)
      ThrowException("unary operator returned null!");
  }
  | ShiftOps ResolvedVal ',' ResolvedVal {
    if ($4->getType() != Type::UByteTy)
      ThrowException("Shift amount must be ubyte!");
    $$ = new ShiftInst($1, $2, $4);
  }
  | CAST ResolvedVal TO Types {
    $$ = new CastInst($2, *$4);
    delete $4;
  }
  | PHI PHIList {
    const Type *Ty = $2->front().first->getType();
    $$ = new PHINode(Ty);
    while ($2->begin() != $2->end()) {
      if ($2->front().first->getType() != Ty) 
	ThrowException("All elements of a PHI node must be of the same type!");
      ((PHINode*)$$)->addIncoming($2->front().first, $2->front().second);
      $2->pop_front();
    }
    delete $2;  // Free the list...
  } 
  | CALL TypesV ValueRef '(' ValueRefListE ')' {
    const MethodType *Ty;

    if (!(Ty = (*$2)->dyncastMethodType())) {
      // Pull out the types of all of the arguments...
      vector<const Type*> ParamTypes;
      for (list<Value*>::iterator I = $5->begin(), E = $5->end(); I != E; ++I)
	ParamTypes.push_back((*I)->getType());
      Ty = MethodType::get(*$2, ParamTypes);
    }
    delete $2;

    Value *V = getVal(Ty, $3);   // Get the method we're calling...

    // Create the call node...
    if (!$5) {                                   // Has no arguments?
      $$ = new CallInst(V->castMethodAsserting(), vector<Value*>());
    } else {                                     // Has arguments?
      // Loop through MethodType's arguments and ensure they are specified
      // correctly!
      //
      MethodType::ParamTypes::const_iterator I = Ty->getParamTypes().begin();
      MethodType::ParamTypes::const_iterator E = Ty->getParamTypes().end();
      list<Value*>::iterator ArgI = $5->begin(), ArgE = $5->end();

      for (; ArgI != ArgE && I != E; ++ArgI, ++I)
	if ((*ArgI)->getType() != *I)
	  ThrowException("Parameter " +(*ArgI)->getName()+ " is not of type '" +
			 (*I)->getName() + "'!");

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
	ThrowException("Invalid number of parameters detected!");

      $$ = new CallInst(V->castMethodAsserting(),
			vector<Value*>($5->begin(), $5->end()));
    }
    delete $5;
  }
  | MemoryInst {
    $$ = $1;
  }

// UByteList - List of ubyte values for load and store instructions
UByteList : ',' ConstVector { 
  $$ = $2; 
} | /* empty */ { 
  $$ = new vector<ConstPoolVal*>(); 
}

MemoryInst : MALLOC Types {
    $$ = new MallocInst(PointerType::get(*$2));
    delete $2;
  }
  | MALLOC Types ',' UINT ValueRef {
    if (!(*$2)->isArrayType() || ((const ArrayType*)$2->get())->isSized())
      ThrowException("Trying to allocate " + (*$2)->getName() + 
		     " as unsized array!");
    const Type *Ty = PointerType::get(*$2);
    $$ = new MallocInst(Ty, getVal($4, $5));
    delete $2;
  }
  | ALLOCA Types {
    $$ = new AllocaInst(PointerType::get(*$2));
    delete $2;
  }
  | ALLOCA Types ',' UINT ValueRef {
    if (!(*$2)->isArrayType() || ((const ArrayType*)$2->get())->isSized())
      ThrowException("Trying to allocate " + (*$2)->getName() + 
		     " as unsized array!");
    const Type *Ty = PointerType::get(*$2);
    Value *ArrSize = getVal($4, $5);
    $$ = new AllocaInst(Ty, ArrSize);
    delete $2;
  }
  | FREE ResolvedVal {
    if (!$2->getType()->isPointerType())
      ThrowException("Trying to free nonpointer type " + 
                     $2->getType()->getName() + "!");
    $$ = new FreeInst($2);
  }

  | LOAD Types ValueRef UByteList {
    if (!(*$2)->isPointerType())
      ThrowException("Can't load from nonpointer type: " + (*$2)->getName());
    if (LoadInst::getIndexedType(*$2, *$4) == 0)
      ThrowException("Invalid indices for load instruction!");

    $$ = new LoadInst(getVal(*$2, $3), *$4);
    delete $4;   // Free the vector...
    delete $2;
  }
  | STORE ResolvedVal ',' Types ValueRef UByteList {
    if (!(*$4)->isPointerType())
      ThrowException("Can't store to a nonpointer type: " + (*$4)->getName());
    const Type *ElTy = StoreInst::getIndexedType(*$4, *$6);
    if (ElTy == 0)
      ThrowException("Can't store into that field list!");
    if (ElTy != $2->getType())
      ThrowException("Can't store '" + $2->getType()->getName() +
                     "' into space of type '" + ElTy->getName() + "'!");
    $$ = new StoreInst($2, getVal(*$4, $5), *$6);
    delete $4; delete $6;
  }
  | GETELEMENTPTR Types ValueRef UByteList {
    if (!(*$2)->isPointerType())
      ThrowException("getelementptr insn requires pointer operand!");
    if (!GetElementPtrInst::getIndexedType(*$2, *$4, true))
      ThrowException("Can't get element ptr '" + (*$2)->getName() + "'!");
    $$ = new GetElementPtrInst(getVal(*$2, $3), *$4);
    delete $2; delete $4;
  }

%%
int yyerror(const char *ErrorMsg) {
  ThrowException(string("Parse error: ") + ErrorMsg);
  return 0;
}
