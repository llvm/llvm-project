//===-- AsmWriter.cpp - Printing LLVM as an assembly file -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/Assembly/Writer.h
//
// Note that these routines must be extremely tolerant of various errors in the
// LLVM code, because it can be used for debugging transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/Writer.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Assembly/AsmAnnotationWriter.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instruction.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
using namespace llvm;

// Make virtual table appear in this compilation unit.
AssemblyAnnotationWriter::~AssemblyAnnotationWriter() {}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static const Module *getModuleFromVal(const Value *V) {
  if (const Argument *MA = dyn_cast<Argument>(V))
    return MA->getParent() ? MA->getParent()->getParent() : 0;
  
  if (const BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return BB->getParent() ? BB->getParent()->getParent() : 0;
  
  if (const Instruction *I = dyn_cast<Instruction>(V)) {
    const Function *M = I->getParent() ? I->getParent()->getParent() : 0;
    return M ? M->getParent() : 0;
  }
  
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
    return GV->getParent();
  return 0;
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
static void PrintEscapedString(const char *Str, unsigned Length,
                               raw_ostream &Out) {
  for (unsigned i = 0; i != Length; ++i) {
    unsigned char C = Str[i];
    if (isprint(C) && C != '\\' && C != '"' && isprint(C))
      Out << C;
    else
      Out << '\\' << hexdigit(C >> 4) << hexdigit(C & 0x0F);
  }
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
static void PrintEscapedString(const std::string &Str, raw_ostream &Out) {
  PrintEscapedString(Str.c_str(), Str.size(), Out);
}

enum PrefixType {
  GlobalPrefix,
  LabelPrefix,
  LocalPrefix,
  NoPrefix
};

/// PrintLLVMName - Turn the specified name into an 'LLVM name', which is either
/// prefixed with % (if the string only contains simple characters) or is
/// surrounded with ""'s (if it has special chars in it).  Print it out.
static void PrintLLVMName(raw_ostream &OS, const char *NameStr,
                          unsigned NameLen, PrefixType Prefix) {
  assert(NameStr && "Cannot get empty name!");
  switch (Prefix) {
  default: assert(0 && "Bad prefix!");
  case NoPrefix: break;
  case GlobalPrefix: OS << '@'; break;
  case LabelPrefix:  break;
  case LocalPrefix:  OS << '%'; break;
  }      
  
  // Scan the name to see if it needs quotes first.
  bool NeedsQuotes = isdigit(NameStr[0]);
  if (!NeedsQuotes) {
    for (unsigned i = 0; i != NameLen; ++i) {
      char C = NameStr[i];
      if (!isalnum(C) && C != '-' && C != '.' && C != '_') {
        NeedsQuotes = true;
        break;
      }
    }
  }
  
  // If we didn't need any quotes, just write out the name in one blast.
  if (!NeedsQuotes) {
    OS.write(NameStr, NameLen);
    return;
  }
  
  // Okay, we need quotes.  Output the quotes and escape any scary characters as
  // needed.
  OS << '"';
  PrintEscapedString(NameStr, NameLen, OS);
  OS << '"';
}

/// getLLVMName - Turn the specified string into an 'LLVM name', which is
/// surrounded with ""'s and escaped if it has special chars in it.
static std::string getLLVMName(const std::string &Name) {
  assert(!Name.empty() && "Cannot get empty name!");
  std::string result;
  raw_string_ostream OS(result);
  PrintLLVMName(OS, Name.c_str(), Name.length(), NoPrefix);
  return OS.str();
}

/// PrintLLVMName - Turn the specified name into an 'LLVM name', which is either
/// prefixed with % (if the string only contains simple characters) or is
/// surrounded with ""'s (if it has special chars in it).  Print it out.
static void PrintLLVMName(raw_ostream &OS, const Value *V) {
  PrintLLVMName(OS, V->getNameStart(), V->getNameLen(),
                isa<GlobalValue>(V) ? GlobalPrefix : LocalPrefix);
}



//===----------------------------------------------------------------------===//
// SlotTracker Class: Enumerate slot numbers for unnamed values
//===----------------------------------------------------------------------===//

namespace {

/// This class provides computation of slot numbers for LLVM Assembly writing.
///
class SlotTracker {
public:
  /// ValueMap - A mapping of Values to slot numbers
  typedef DenseMap<const Value*, unsigned> ValueMap;
  
private:  
  /// TheModule - The module for which we are holding slot numbers
  const Module* TheModule;
  
  /// TheFunction - The function for which we are holding slot numbers
  const Function* TheFunction;
  bool FunctionProcessed;
  
  /// mMap - The TypePlanes map for the module level data
  ValueMap mMap;
  unsigned mNext;
  
  /// fMap - The TypePlanes map for the function level data
  ValueMap fMap;
  unsigned fNext;
  
public:
  /// Construct from a module
  explicit SlotTracker(const Module *M);
  /// Construct from a function, starting out in incorp state.
  explicit SlotTracker(const Function *F);

  /// Return the slot number of the specified value in it's type
  /// plane.  If something is not in the SlotTracker, return -1.
  int getLocalSlot(const Value *V);
  int getGlobalSlot(const GlobalValue *V);

  /// If you'd like to deal with a function instead of just a module, use
  /// this method to get its data into the SlotTracker.
  void incorporateFunction(const Function *F) {
    TheFunction = F;
    FunctionProcessed = false;
  }

  /// After calling incorporateFunction, use this method to remove the
  /// most recently incorporated function from the SlotTracker. This
  /// will reset the state of the machine back to just the module contents.
  void purgeFunction();

  // Implementation Details
private:
  /// This function does the actual initialization.
  inline void initialize();

  /// CreateModuleSlot - Insert the specified GlobalValue* into the slot table.
  void CreateModuleSlot(const GlobalValue *V);
  
  /// CreateFunctionSlot - Insert the specified Value* into the slot table.
  void CreateFunctionSlot(const Value *V);

  /// Add all of the module level global variables (and their initializers)
  /// and function declarations, but not the contents of those functions.
  void processModule();

  /// Add all of the functions arguments, basic blocks, and instructions
  void processFunction();

  SlotTracker(const SlotTracker &);  // DO NOT IMPLEMENT
  void operator=(const SlotTracker &);  // DO NOT IMPLEMENT
};

}  // end anonymous namespace


static SlotTracker *createSlotTracker(const Value *V) {
  if (const Argument *FA = dyn_cast<Argument>(V))
    return new SlotTracker(FA->getParent());
  
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return new SlotTracker(I->getParent()->getParent());
  
  if (const BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return new SlotTracker(BB->getParent());
  
  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return new SlotTracker(GV->getParent());
  
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
    return new SlotTracker(GA->getParent());    
  
  if (const Function *Func = dyn_cast<Function>(V))
    return new SlotTracker(Func);
  
  return 0;
}

#if 0
#define ST_DEBUG(X) cerr << X
#else
#define ST_DEBUG(X)
#endif

// Module level constructor. Causes the contents of the Module (sans functions)
// to be added to the slot table.
SlotTracker::SlotTracker(const Module *M)
  : TheModule(M), TheFunction(0), FunctionProcessed(false), mNext(0), fNext(0) {
}

// Function level constructor. Causes the contents of the Module and the one
// function provided to be added to the slot table.
SlotTracker::SlotTracker(const Function *F)
  : TheModule(F ? F->getParent() : 0), TheFunction(F), FunctionProcessed(false),
    mNext(0), fNext(0) {
}

inline void SlotTracker::initialize() {
  if (TheModule) {
    processModule();
    TheModule = 0; ///< Prevent re-processing next time we're called.
  }
  
  if (TheFunction && !FunctionProcessed)
    processFunction();
}

// Iterate through all the global variables, functions, and global
// variable initializers and create slots for them.
void SlotTracker::processModule() {
  ST_DEBUG("begin processModule!\n");
  
  // Add all of the unnamed global variables to the value table.
  for (Module::const_global_iterator I = TheModule->global_begin(),
       E = TheModule->global_end(); I != E; ++I)
    if (!I->hasName()) 
      CreateModuleSlot(I);
  
  // Add all the unnamed functions to the table.
  for (Module::const_iterator I = TheModule->begin(), E = TheModule->end();
       I != E; ++I)
    if (!I->hasName())
      CreateModuleSlot(I);
  
  ST_DEBUG("end processModule!\n");
}


// Process the arguments, basic blocks, and instructions  of a function.
void SlotTracker::processFunction() {
  ST_DEBUG("begin processFunction!\n");
  fNext = 0;
  
  // Add all the function arguments with no names.
  for(Function::const_arg_iterator AI = TheFunction->arg_begin(),
      AE = TheFunction->arg_end(); AI != AE; ++AI)
    if (!AI->hasName())
      CreateFunctionSlot(AI);
  
  ST_DEBUG("Inserting Instructions:\n");
  
  // Add all of the basic blocks and instructions with no names.
  for (Function::const_iterator BB = TheFunction->begin(),
       E = TheFunction->end(); BB != E; ++BB) {
    if (!BB->hasName())
      CreateFunctionSlot(BB);
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (I->getType() != Type::VoidTy && !I->hasName())
        CreateFunctionSlot(I);
  }
  
  FunctionProcessed = true;
  
  ST_DEBUG("end processFunction!\n");
}

/// Clean up after incorporating a function. This is the only way to get out of
/// the function incorporation state that affects get*Slot/Create*Slot. Function
/// incorporation state is indicated by TheFunction != 0.
void SlotTracker::purgeFunction() {
  ST_DEBUG("begin purgeFunction!\n");
  fMap.clear(); // Simply discard the function level map
  TheFunction = 0;
  FunctionProcessed = false;
  ST_DEBUG("end purgeFunction!\n");
}

/// getGlobalSlot - Get the slot number of a global value.
int SlotTracker::getGlobalSlot(const GlobalValue *V) {
  // Check for uninitialized state and do lazy initialization.
  initialize();
  
  // Find the type plane in the module map
  ValueMap::iterator MI = mMap.find(V);
  return MI == mMap.end() ? -1 : (int)MI->second;
}


/// getLocalSlot - Get the slot number for a value that is local to a function.
int SlotTracker::getLocalSlot(const Value *V) {
  assert(!isa<Constant>(V) && "Can't get a constant or global slot with this!");
  
  // Check for uninitialized state and do lazy initialization.
  initialize();
  
  ValueMap::iterator FI = fMap.find(V);
  return FI == fMap.end() ? -1 : (int)FI->second;
}


/// CreateModuleSlot - Insert the specified GlobalValue* into the slot table.
void SlotTracker::CreateModuleSlot(const GlobalValue *V) {
  assert(V && "Can't insert a null Value into SlotTracker!");
  assert(V->getType() != Type::VoidTy && "Doesn't need a slot!");
  assert(!V->hasName() && "Doesn't need a slot!");
  
  unsigned DestSlot = mNext++;
  mMap[V] = DestSlot;
  
  ST_DEBUG("  Inserting value [" << V->getType() << "] = " << V << " slot=" <<
           DestSlot << " [");
  // G = Global, F = Function, A = Alias, o = other
  ST_DEBUG((isa<GlobalVariable>(V) ? 'G' :
            (isa<Function>(V) ? 'F' :
             (isa<GlobalAlias>(V) ? 'A' : 'o'))) << "]\n");
}


/// CreateSlot - Create a new slot for the specified value if it has no name.
void SlotTracker::CreateFunctionSlot(const Value *V) {
  assert(V->getType() != Type::VoidTy && !V->hasName() &&
         "Doesn't need a slot!");
  
  unsigned DestSlot = fNext++;
  fMap[V] = DestSlot;
  
  // G = Global, F = Function, o = other
  ST_DEBUG("  Inserting value [" << V->getType() << "] = " << V << " slot=" <<
           DestSlot << " [o]\n");
}  



//===----------------------------------------------------------------------===//
// AsmWriter Implementation
//===----------------------------------------------------------------------===//

static void WriteAsOperandInternal(raw_ostream &Out, const Value *V,
                               std::map<const Type *, std::string> &TypeTable,
                                   SlotTracker *Machine);



/// fillTypeNameTable - If the module has a symbol table, take all global types
/// and stuff their names into the TypeNames map.
///
static void fillTypeNameTable(const Module *M,
                              std::map<const Type *, std::string> &TypeNames) {
  if (!M) return;
  const TypeSymbolTable &ST = M->getTypeSymbolTable();
  TypeSymbolTable::const_iterator TI = ST.begin();
  for (; TI != ST.end(); ++TI) {
    // As a heuristic, don't insert pointer to primitive types, because
    // they are used too often to have a single useful name.
    //
    const Type *Ty = cast<Type>(TI->second);
    if (!isa<PointerType>(Ty) ||
        !cast<PointerType>(Ty)->getElementType()->isPrimitiveType() ||
        !cast<PointerType>(Ty)->getElementType()->isInteger() ||
        isa<OpaqueType>(cast<PointerType>(Ty)->getElementType()))
      TypeNames.insert(std::make_pair(Ty, '%' + getLLVMName(TI->first)));
  }
}



static void calcTypeName(const Type *Ty,
                         std::vector<const Type *> &TypeStack,
                         std::map<const Type *, std::string> &TypeNames,
                         std::string &Result) {
  if (Ty->isInteger() || (Ty->isPrimitiveType() && !isa<OpaqueType>(Ty))) {
    Result += Ty->getDescription();  // Base case
    return;
  }

  // Check to see if the type is named.
  std::map<const Type *, std::string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) {
    Result += I->second;
    return;
  }

  if (isa<OpaqueType>(Ty)) {
    Result += "opaque";
    return;
  }

  // Check to see if the Type is already on the stack...
  unsigned Slot = 0, CurSize = TypeStack.size();
  while (Slot < CurSize && TypeStack[Slot] != Ty) ++Slot; // Scan for type

  // This is another base case for the recursion.  In this case, we know
  // that we have looped back to a type that we have previously visited.
  // Generate the appropriate upreference to handle this.
  if (Slot < CurSize) {
    Result += "\\" + utostr(CurSize-Slot);     // Here's the upreference
    return;
  }

  TypeStack.push_back(Ty);    // Recursive case: Add us to the stack..

  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    unsigned BitWidth = cast<IntegerType>(Ty)->getBitWidth();
    Result += "i" + utostr(BitWidth);
    break;
  }
  case Type::FunctionTyID: {
    const FunctionType *FTy = cast<FunctionType>(Ty);
    calcTypeName(FTy->getReturnType(), TypeStack, TypeNames, Result);
    Result += " (";
    for (FunctionType::param_iterator I = FTy->param_begin(),
         E = FTy->param_end(); I != E; ++I) {
      if (I != FTy->param_begin())
        Result += ", ";
      calcTypeName(*I, TypeStack, TypeNames, Result);
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
      Result += '<';
    Result += "{ ";
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      calcTypeName(*I, TypeStack, TypeNames, Result);
      if (next(I) != STy->element_end())
        Result += ',';
      Result += ' ';
    }
    Result += '}';
    if (STy->isPacked())
      Result += '>';
    break;
  }
  case Type::PointerTyID: {
    const PointerType *PTy = cast<PointerType>(Ty);
    calcTypeName(PTy->getElementType(), TypeStack, TypeNames, Result);
    if (unsigned AddressSpace = PTy->getAddressSpace())
      Result += " addrspace(" + utostr(AddressSpace) + ")";
    Result += "*";
    break;
  }
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    Result += "[" + utostr(ATy->getNumElements()) + " x ";
    calcTypeName(ATy->getElementType(), TypeStack, TypeNames, Result);
    Result += "]";
    break;
  }
  case Type::VectorTyID: {
    const VectorType *PTy = cast<VectorType>(Ty);
    Result += "<" + utostr(PTy->getNumElements()) + " x ";
    calcTypeName(PTy->getElementType(), TypeStack, TypeNames, Result);
    Result += ">";
    break;
  }
  case Type::OpaqueTyID:
    Result += "opaque";
    break;
  default:
    Result += "<unrecognized-type>";
    break;
  }

  TypeStack.pop_back();       // Remove self from stack...
}


/// printTypeInt - The internal guts of printing out a type that has a
/// potentially named portion.
///
static void printTypeInt(raw_ostream &Out, const Type *Ty,
                         std::map<const Type *, std::string> &TypeNames) {
  // Primitive types always print out their description, regardless of whether
  // they have been named or not.
  //
  if (Ty->isInteger() || (Ty->isPrimitiveType() && !isa<OpaqueType>(Ty))) {
    Out << Ty->getDescription();
    return;
  }

  // Check to see if the type is named.
  std::map<const Type *, std::string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) {
    Out << I->second;
    return;
  }

  // Otherwise we have a type that has not been named but is a derived type.
  // Carefully recurse the type hierarchy to print out any contained symbolic
  // names.
  //
  std::vector<const Type *> TypeStack;
  std::string TypeName;
  calcTypeName(Ty, TypeStack, TypeNames, TypeName);
  TypeNames.insert(std::make_pair(Ty, TypeName));//Cache type name for later use
  Out << TypeName;
}


/// WriteTypeSymbolic - This attempts to write the specified type as a symbolic
/// type, iff there is an entry in the modules symbol table for the specified
/// type or one of it's component types. This is slower than a simple x << Type
///
void llvm::WriteTypeSymbolic(std::ostream &Out, const Type *Ty,
                             const Module *M) {
  raw_os_ostream RO(Out);
  WriteTypeSymbolic(RO, Ty, M);
}

void llvm::WriteTypeSymbolic(raw_ostream &Out, const Type *Ty, const Module *M){
  Out << ' ';

  // If they want us to print out a type, but there is no context, we can't
  // print it symbolically.
  if (!M) {
    Out << Ty->getDescription();
  } else {
    std::map<const Type *, std::string> TypeNames;
    fillTypeNameTable(M, TypeNames);
    printTypeInt(Out, Ty, TypeNames);
  }
}

static const char *getPredicateText(unsigned predicate) {
  const char * pred = "unknown";
  switch (predicate) {
    case FCmpInst::FCMP_FALSE: pred = "false"; break;
    case FCmpInst::FCMP_OEQ:   pred = "oeq"; break;
    case FCmpInst::FCMP_OGT:   pred = "ogt"; break;
    case FCmpInst::FCMP_OGE:   pred = "oge"; break;
    case FCmpInst::FCMP_OLT:   pred = "olt"; break;
    case FCmpInst::FCMP_OLE:   pred = "ole"; break;
    case FCmpInst::FCMP_ONE:   pred = "one"; break;
    case FCmpInst::FCMP_ORD:   pred = "ord"; break;
    case FCmpInst::FCMP_UNO:   pred = "uno"; break;
    case FCmpInst::FCMP_UEQ:   pred = "ueq"; break;
    case FCmpInst::FCMP_UGT:   pred = "ugt"; break;
    case FCmpInst::FCMP_UGE:   pred = "uge"; break;
    case FCmpInst::FCMP_ULT:   pred = "ult"; break;
    case FCmpInst::FCMP_ULE:   pred = "ule"; break;
    case FCmpInst::FCMP_UNE:   pred = "une"; break;
    case FCmpInst::FCMP_TRUE:  pred = "true"; break;
    case ICmpInst::ICMP_EQ:    pred = "eq"; break;
    case ICmpInst::ICMP_NE:    pred = "ne"; break;
    case ICmpInst::ICMP_SGT:   pred = "sgt"; break;
    case ICmpInst::ICMP_SGE:   pred = "sge"; break;
    case ICmpInst::ICMP_SLT:   pred = "slt"; break;
    case ICmpInst::ICMP_SLE:   pred = "sle"; break;
    case ICmpInst::ICMP_UGT:   pred = "ugt"; break;
    case ICmpInst::ICMP_UGE:   pred = "uge"; break;
    case ICmpInst::ICMP_ULT:   pred = "ult"; break;
    case ICmpInst::ICMP_ULE:   pred = "ule"; break;
  }
  return pred;
}

static void WriteConstantInt(raw_ostream &Out, const Constant *CV,
                             std::map<const Type *, std::string> &TypeTable,
                             SlotTracker *Machine) {
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    if (CI->getType() == Type::Int1Ty) {
      Out << (CI->getZExtValue() ? "true" : "false");
      return;
    }
    Out << CI->getValue();
    return;
  }
  
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    if (&CFP->getValueAPF().getSemantics() == &APFloat::IEEEdouble ||
        &CFP->getValueAPF().getSemantics() == &APFloat::IEEEsingle) {
      // We would like to output the FP constant value in exponential notation,
      // but we cannot do this if doing so will lose precision.  Check here to
      // make sure that we only output it in exponential format if we can parse
      // the value back and get the same value.
      //
      bool ignored;
      bool isDouble = &CFP->getValueAPF().getSemantics()==&APFloat::IEEEdouble;
      double Val = isDouble ? CFP->getValueAPF().convertToDouble() :
                              CFP->getValueAPF().convertToFloat();
      std::string StrVal = ftostr(CFP->getValueAPF());

      // Check to make sure that the stringized number is not some string like
      // "Inf" or NaN, that atof will accept, but the lexer will not.  Check
      // that the string matches the "[-+]?[0-9]" regex.
      //
      if ((StrVal[0] >= '0' && StrVal[0] <= '9') ||
          ((StrVal[0] == '-' || StrVal[0] == '+') &&
           (StrVal[1] >= '0' && StrVal[1] <= '9'))) {
        // Reparse stringized version!
        if (atof(StrVal.c_str()) == Val) {
          Out << StrVal;
          return;
        }
      }
      // Otherwise we could not reparse it to exactly the same value, so we must
      // output the string in hexadecimal format!  Note that loading and storing
      // floating point types changes the bits of NaNs on some hosts, notably
      // x86, so we must not use these types.
      assert(sizeof(double) == sizeof(uint64_t) &&
             "assuming that double is 64 bits!");
      char Buffer[40];
      APFloat apf = CFP->getValueAPF();
      // Floats are represented in ASCII IR as double, convert.
      if (!isDouble)
        apf.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, 
                          &ignored);
      Out << "0x" << 
              utohex_buffer(uint64_t(apf.bitcastToAPInt().getZExtValue()), 
                            Buffer+40);
      return;
    }
    
    // Some form of long double.  These appear as a magic letter identifying
    // the type, then a fixed number of hex digits.
    Out << "0x";
    if (&CFP->getValueAPF().getSemantics() == &APFloat::x87DoubleExtended)
      Out << 'K';
    else if (&CFP->getValueAPF().getSemantics() == &APFloat::IEEEquad)
      Out << 'L';
    else if (&CFP->getValueAPF().getSemantics() == &APFloat::PPCDoubleDouble)
      Out << 'M';
    else
      assert(0 && "Unsupported floating point type");
    // api needed to prevent premature destruction
    APInt api = CFP->getValueAPF().bitcastToAPInt();
    const uint64_t* p = api.getRawData();
    uint64_t word = *p;
    int shiftcount=60;
    int width = api.getBitWidth();
    for (int j=0; j<width; j+=4, shiftcount-=4) {
      unsigned int nibble = (word>>shiftcount) & 15;
      if (nibble < 10)
        Out << (unsigned char)(nibble + '0');
      else
        Out << (unsigned char)(nibble - 10 + 'A');
      if (shiftcount == 0 && j+4 < width) {
        word = *(++p);
        shiftcount = 64;
        if (width-j-4 < 64)
          shiftcount = width-j-4;
      }
    }
    return;
  }
  
  if (isa<ConstantAggregateZero>(CV)) {
    Out << "zeroinitializer";
    return;
  }
  
  if (const ConstantArray *CA = dyn_cast<ConstantArray>(CV)) {
    // As a special case, print the array as a string if it is an array of
    // i8 with ConstantInt values.
    //
    const Type *ETy = CA->getType()->getElementType();
    if (CA->isString()) {
      Out << "c\"";
      PrintEscapedString(CA->getAsString(), Out);
      Out << '"';
    } else {                // Cannot output in string format...
      Out << '[';
      if (CA->getNumOperands()) {
        Out << ' ';
        printTypeInt(Out, ETy, TypeTable);
        Out << ' ';
        WriteAsOperandInternal(Out, CA->getOperand(0),
                               TypeTable, Machine);
        for (unsigned i = 1, e = CA->getNumOperands(); i != e; ++i) {
          Out << ", ";
          printTypeInt(Out, ETy, TypeTable);
          Out << ' ';
          WriteAsOperandInternal(Out, CA->getOperand(i), TypeTable, Machine);
        }
        Out << ' ';
      }
      Out << ']';
    }
    return;
  }
  
  if (const ConstantStruct *CS = dyn_cast<ConstantStruct>(CV)) {
    if (CS->getType()->isPacked())
      Out << '<';
    Out << '{';
    unsigned N = CS->getNumOperands();
    if (N) {
      Out << ' ';
      printTypeInt(Out, CS->getOperand(0)->getType(), TypeTable);
      Out << ' ';

      WriteAsOperandInternal(Out, CS->getOperand(0), TypeTable, Machine);

      for (unsigned i = 1; i < N; i++) {
        Out << ", ";
        printTypeInt(Out, CS->getOperand(i)->getType(), TypeTable);
        Out << ' ';

        WriteAsOperandInternal(Out, CS->getOperand(i), TypeTable, Machine);
      }
      Out << ' ';
    }
 
    Out << '}';
    if (CS->getType()->isPacked())
      Out << '>';
    return;
  }
  
  if (const ConstantVector *CP = dyn_cast<ConstantVector>(CV)) {
    const Type *ETy = CP->getType()->getElementType();
    assert(CP->getNumOperands() > 0 &&
           "Number of operands for a PackedConst must be > 0");
    Out << "< ";
    printTypeInt(Out, ETy, TypeTable);
    Out << ' ';
    WriteAsOperandInternal(Out, CP->getOperand(0), TypeTable, Machine);
    for (unsigned i = 1, e = CP->getNumOperands(); i != e; ++i) {
      Out << ", ";
      printTypeInt(Out, ETy, TypeTable);
      Out << ' ';
      WriteAsOperandInternal(Out, CP->getOperand(i), TypeTable, Machine);
    }
    Out << " >";
    return;
  }
  
  if (isa<ConstantPointerNull>(CV)) {
    Out << "null";
    return;
  }
  
  if (isa<UndefValue>(CV)) {
    Out << "undef";
    return;
  }

  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    Out << CE->getOpcodeName();
    if (CE->isCompare())
      Out << ' ' << getPredicateText(CE->getPredicate());
    Out << " (";

    for (User::const_op_iterator OI=CE->op_begin(); OI != CE->op_end(); ++OI) {
      printTypeInt(Out, (*OI)->getType(), TypeTable);
      Out << ' ';
      WriteAsOperandInternal(Out, *OI, TypeTable, Machine);
      if (OI+1 != CE->op_end())
        Out << ", ";
    }

    if (CE->hasIndices()) {
      const SmallVector<unsigned, 4> &Indices = CE->getIndices();
      for (unsigned i = 0, e = Indices.size(); i != e; ++i)
        Out << ", " << Indices[i];
    }

    if (CE->isCast()) {
      Out << " to ";
      printTypeInt(Out, CE->getType(), TypeTable);
    }

    Out << ')';
    return;
  }
  
  Out << "<placeholder or erroneous Constant>";
}


/// WriteAsOperand - Write the name of the specified value out to the specified
/// ostream.  This can be useful when you just want to print int %reg126, not
/// the whole instruction that generated it.
///
static void WriteAsOperandInternal(raw_ostream &Out, const Value *V,
                                  std::map<const Type*, std::string> &TypeTable,
                                   SlotTracker *Machine) {
  if (V->hasName()) {
    PrintLLVMName(Out, V);
    return;
  }
  
  const Constant *CV = dyn_cast<Constant>(V);
  if (CV && !isa<GlobalValue>(CV)) {
    WriteConstantInt(Out, CV, TypeTable, Machine);
    return;
  }
  
  if (const InlineAsm *IA = dyn_cast<InlineAsm>(V)) {
    Out << "asm ";
    if (IA->hasSideEffects())
      Out << "sideeffect ";
    Out << '"';
    PrintEscapedString(IA->getAsmString(), Out);
    Out << "\", \"";
    PrintEscapedString(IA->getConstraintString(), Out);
    Out << '"';
    return;
  }
  
  char Prefix = '%';
  int Slot;
  if (Machine) {
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
      Slot = Machine->getGlobalSlot(GV);
      Prefix = '@';
    } else {
      Slot = Machine->getLocalSlot(V);
    }
  } else {
    Machine = createSlotTracker(V);
    if (Machine) {
      if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
        Slot = Machine->getGlobalSlot(GV);
        Prefix = '@';
      } else {
        Slot = Machine->getLocalSlot(V);
      }
    } else {
      Slot = -1;
    }
    delete Machine;
  }
  
  if (Slot != -1)
    Out << Prefix << Slot;
  else
    Out << "<badref>";
}

/// WriteAsOperand - Write the name of the specified value out to the specified
/// ostream.  This can be useful when you just want to print int %reg126, not
/// the whole instruction that generated it.
///
void llvm::WriteAsOperand(std::ostream &Out, const Value *V, bool PrintType,
                          const Module *Context) {
  raw_os_ostream OS(Out);
  WriteAsOperand(OS, V, PrintType, Context);
}

void llvm::WriteAsOperand(raw_ostream &Out, const Value *V, bool PrintType,
                          const Module *Context) {
  std::map<const Type *, std::string> TypeNames;
  if (Context == 0) Context = getModuleFromVal(V);

  if (Context)
    fillTypeNameTable(Context, TypeNames);

  if (PrintType) {
    printTypeInt(Out, V->getType(), TypeNames);
    Out << ' ';
  }

  WriteAsOperandInternal(Out, V, TypeNames, 0);
}


namespace {

class AssemblyWriter {
  raw_ostream &Out;
  SlotTracker &Machine;
  const Module *TheModule;
  std::map<const Type *, std::string> TypeNames;
  AssemblyAnnotationWriter *AnnotationWriter;
public:
  inline AssemblyWriter(raw_ostream &o, SlotTracker &Mac, const Module *M,
                        AssemblyAnnotationWriter *AAW)
    : Out(o), Machine(Mac), TheModule(M), AnnotationWriter(AAW) {

    // If the module has a symbol table, take all global types and stuff their
    // names into the TypeNames map.
    //
    fillTypeNameTable(M, TypeNames);
  }

  void write(const Module *M) { printModule(M);       }
  
  void write(const GlobalValue *G) {
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(G))
      printGlobal(GV);
    else if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(G))
      printAlias(GA);
    else if (const Function *F = dyn_cast<Function>(G))
      printFunction(F);
    else
      assert(0 && "Unknown global");
  }
  
  void write(const BasicBlock *BB)    { printBasicBlock(BB);  }
  void write(const Instruction *I)    { printInstruction(*I); }
  void write(const Type *Ty)          { printType(Ty);        }

  void writeOperand(const Value *Op, bool PrintType);
  void writeParamOperand(const Value *Operand, Attributes Attrs);

  const Module* getModule() { return TheModule; }

private:
  void printModule(const Module *M);
  void printTypeSymbolTable(const TypeSymbolTable &ST);
  void printGlobal(const GlobalVariable *GV);
  void printAlias(const GlobalAlias *GV);
  void printFunction(const Function *F);
  void printArgument(const Argument *FA, Attributes Attrs);
  void printBasicBlock(const BasicBlock *BB);
  void printInstruction(const Instruction &I);

  // printType - Go to extreme measures to attempt to print out a short,
  // symbolic version of a type name.
  //
  void printType(const Type *Ty) {
    printTypeInt(Out, Ty, TypeNames);
  }

  // printTypeAtLeastOneLevel - Print out one level of the possibly complex type
  // without considering any symbolic types that we may have equal to it.
  //
  void printTypeAtLeastOneLevel(const Type *Ty);

  // printInfoComment - Print a little comment after the instruction indicating
  // which slot it occupies.
  void printInfoComment(const Value &V);
};
}  // end of llvm namespace

/// printTypeAtLeastOneLevel - Print out one level of the possibly complex type
/// without considering any symbolic types that we may have equal to it.
///
void AssemblyWriter::printTypeAtLeastOneLevel(const Type *Ty) {
  if (const IntegerType *ITy = dyn_cast<IntegerType>(Ty)) {
    Out << "i" << utostr(ITy->getBitWidth());
    return;
  }
  
  if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
    printType(FTy->getReturnType());
    Out << " (";
    for (FunctionType::param_iterator I = FTy->param_begin(),
           E = FTy->param_end(); I != E; ++I) {
      if (I != FTy->param_begin())
        Out << ", ";
      printType(*I);
    }
    if (FTy->isVarArg()) {
      if (FTy->getNumParams()) Out << ", ";
      Out << "...";
    }
    Out << ')';
    return;
  }
  
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    if (STy->isPacked())
      Out << '<';
    Out << "{ ";
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      if (I != STy->element_begin())
        Out << ", ";
      printType(*I);
    }
    Out << " }";
    if (STy->isPacked())
      Out << '>';
    return;
  }
  
  if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
    printType(PTy->getElementType());
    if (unsigned AddressSpace = PTy->getAddressSpace())
      Out << " addrspace(" << AddressSpace << ")";
    Out << '*';
    return;
  } 
  
  if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    Out << '[' << ATy->getNumElements() << " x ";
    printType(ATy->getElementType());
    Out << ']';
    return;
  }
  
  if (const VectorType *PTy = dyn_cast<VectorType>(Ty)) {
    Out << '<' << PTy->getNumElements() << " x ";
    printType(PTy->getElementType());
    Out << '>';
    return;
  }
  
  if (isa<OpaqueType>(Ty)) {
    Out << "opaque";
    return;
  }
  
  if (!Ty->isPrimitiveType())
    Out << "<unknown derived type>";
  printType(Ty);
}


void AssemblyWriter::writeOperand(const Value *Operand, bool PrintType) {
  if (Operand == 0) {
    Out << "<null operand!>";
  } else {
    if (PrintType) {
      printType(Operand->getType());
      Out << ' ';
    }
    WriteAsOperandInternal(Out, Operand, TypeNames, &Machine);
  }
}

void AssemblyWriter::writeParamOperand(const Value *Operand, 
                                       Attributes Attrs) {
  if (Operand == 0) {
    Out << "<null operand!>";
  } else {
    // Print the type
    printType(Operand->getType());
    // Print parameter attributes list
    if (Attrs != Attribute::None)
      Out << ' ' << Attribute::getAsString(Attrs);
    Out << ' ';
    // Print the operand
    WriteAsOperandInternal(Out, Operand, TypeNames, &Machine);
  }
}

void AssemblyWriter::printModule(const Module *M) {
  if (!M->getModuleIdentifier().empty() &&
      // Don't print the ID if it will start a new line (which would
      // require a comment char before it).
      M->getModuleIdentifier().find('\n') == std::string::npos)
    Out << "; ModuleID = '" << M->getModuleIdentifier() << "'\n";

  if (!M->getDataLayout().empty())
    Out << "target datalayout = \"" << M->getDataLayout() << "\"\n";
  if (!M->getTargetTriple().empty())
    Out << "target triple = \"" << M->getTargetTriple() << "\"\n";

  if (!M->getModuleInlineAsm().empty()) {
    // Split the string into lines, to make it easier to read the .ll file.
    std::string Asm = M->getModuleInlineAsm();
    size_t CurPos = 0;
    size_t NewLine = Asm.find_first_of('\n', CurPos);
    while (NewLine != std::string::npos) {
      // We found a newline, print the portion of the asm string from the
      // last newline up to this newline.
      Out << "module asm \"";
      PrintEscapedString(std::string(Asm.begin()+CurPos, Asm.begin()+NewLine),
                         Out);
      Out << "\"\n";
      CurPos = NewLine+1;
      NewLine = Asm.find_first_of('\n', CurPos);
    }
    Out << "module asm \"";
    PrintEscapedString(std::string(Asm.begin()+CurPos, Asm.end()), Out);
    Out << "\"\n";
  }
  
  // Loop over the dependent libraries and emit them.
  Module::lib_iterator LI = M->lib_begin();
  Module::lib_iterator LE = M->lib_end();
  if (LI != LE) {
    Out << "deplibs = [ ";
    while (LI != LE) {
      Out << '"' << *LI << '"';
      ++LI;
      if (LI != LE)
        Out << ", ";
    }
    Out << " ]\n";
  }

  // Loop over the symbol table, emitting all named constants.
  printTypeSymbolTable(M->getTypeSymbolTable());

  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
    printGlobal(I);
  
  // Output all aliases.
  if (!M->alias_empty()) Out << "\n";
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    printAlias(I);

  // Output all of the functions.
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    printFunction(I);
}

static void PrintLinkage(GlobalValue::LinkageTypes LT, raw_ostream &Out) {
  switch (LT) {
  case GlobalValue::PrivateLinkage:      Out << "private "; break;
  case GlobalValue::InternalLinkage:     Out << "internal "; break;
  case GlobalValue::LinkOnceLinkage:     Out << "linkonce "; break;
  case GlobalValue::WeakLinkage:         Out << "weak "; break;
  case GlobalValue::CommonLinkage:       Out << "common "; break;
  case GlobalValue::AppendingLinkage:    Out << "appending "; break;
  case GlobalValue::DLLImportLinkage:    Out << "dllimport "; break;
  case GlobalValue::DLLExportLinkage:    Out << "dllexport "; break;
  case GlobalValue::ExternalWeakLinkage: Out << "extern_weak "; break;      
  case GlobalValue::ExternalLinkage: break;
  case GlobalValue::GhostLinkage:
    Out << "GhostLinkage not allowed in AsmWriter!\n";
    abort();
  }
}
      

static void PrintVisibility(GlobalValue::VisibilityTypes Vis,
                            raw_ostream &Out) {
  switch (Vis) {
  default: assert(0 && "Invalid visibility style!");
  case GlobalValue::DefaultVisibility: break;
  case GlobalValue::HiddenVisibility:    Out << "hidden "; break;
  case GlobalValue::ProtectedVisibility: Out << "protected "; break;
  }
}

void AssemblyWriter::printGlobal(const GlobalVariable *GV) {
  if (GV->hasName()) {
    PrintLLVMName(Out, GV);
    Out << " = ";
  }

  if (!GV->hasInitializer() && GV->hasExternalLinkage())
    Out << "external ";
  
  PrintLinkage(GV->getLinkage(), Out);
  PrintVisibility(GV->getVisibility(), Out);

  if (GV->isThreadLocal()) Out << "thread_local ";
  if (unsigned AddressSpace = GV->getType()->getAddressSpace())
    Out << "addrspace(" << AddressSpace << ") ";
  Out << (GV->isConstant() ? "constant " : "global ");
  printType(GV->getType()->getElementType());

  if (GV->hasInitializer()) {
    Out << ' ';
    writeOperand(GV->getInitializer(), false);
  }
    
  if (GV->hasSection())
    Out << ", section \"" << GV->getSection() << '"';
  if (GV->getAlignment())
    Out << ", align " << GV->getAlignment();

  printInfoComment(*GV);
  Out << '\n';
}

void AssemblyWriter::printAlias(const GlobalAlias *GA) {
  // Don't crash when dumping partially built GA
  if (!GA->hasName())
    Out << "<<nameless>> = ";
  else {
    PrintLLVMName(Out, GA);
    Out << " = ";
  }
  PrintVisibility(GA->getVisibility(), Out);

  Out << "alias ";

  PrintLinkage(GA->getLinkage(), Out);
  
  const Constant *Aliasee = GA->getAliasee();
    
  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Aliasee)) {
    printType(GV->getType());
    Out << ' ';
    PrintLLVMName(Out, GV);
  } else if (const Function *F = dyn_cast<Function>(Aliasee)) {
    printType(F->getFunctionType());
    Out << "* ";

    if (F->hasName())
      PrintLLVMName(Out, F);
    else
      Out << "@\"\"";
  } else if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(Aliasee)) {
    printType(GA->getType());
    Out << " ";
    PrintLLVMName(Out, GA);
  } else {
    const ConstantExpr *CE = 0;
    if ((CE = dyn_cast<ConstantExpr>(Aliasee)) &&
        (CE->getOpcode() == Instruction::BitCast)) {
      writeOperand(CE, false);    
    } else
      assert(0 && "Unsupported aliasee");
  }
  
  printInfoComment(*GA);
  Out << '\n';
}

void AssemblyWriter::printTypeSymbolTable(const TypeSymbolTable &ST) {
  // Print the types.
  for (TypeSymbolTable::const_iterator TI = ST.begin(), TE = ST.end();
       TI != TE; ++TI) {
    Out << '\t';
    PrintLLVMName(Out, &TI->first[0], TI->first.size(), LocalPrefix);
    Out << " = type ";

    // Make sure we print out at least one level of the type structure, so
    // that we do not get %FILE = type %FILE
    //
    printTypeAtLeastOneLevel(TI->second);
    Out << '\n';
  }
}

/// printFunction - Print all aspects of a function.
///
void AssemblyWriter::printFunction(const Function *F) {
  // Print out the return type and name.
  Out << '\n';

  if (AnnotationWriter) AnnotationWriter->emitFunctionAnnot(F, Out);

  if (F->isDeclaration())
    Out << "declare ";
  else
    Out << "define ";
  
  PrintLinkage(F->getLinkage(), Out);
  PrintVisibility(F->getVisibility(), Out);

  // Print the calling convention.
  switch (F->getCallingConv()) {
  case CallingConv::C: break;   // default
  case CallingConv::Fast:         Out << "fastcc "; break;
  case CallingConv::Cold:         Out << "coldcc "; break;
  case CallingConv::X86_StdCall:  Out << "x86_stdcallcc "; break;
  case CallingConv::X86_FastCall: Out << "x86_fastcallcc "; break; 
  default: Out << "cc" << F->getCallingConv() << " "; break;
  }

  const FunctionType *FT = F->getFunctionType();
  const AttrListPtr &Attrs = F->getAttributes();
  Attributes RetAttrs = Attrs.getRetAttributes();
  if (RetAttrs != Attribute::None)
    Out <<  Attribute::getAsString(Attrs.getRetAttributes()) << ' ';
  printType(F->getReturnType());
  Out << ' ';
  if (F->hasName())
    PrintLLVMName(Out, F);
  else
    Out << "@\"\"";
  Out << '(';
  Machine.incorporateFunction(F);

  // Loop over the arguments, printing them...

  unsigned Idx = 1;
  if (!F->isDeclaration()) {
    // If this isn't a declaration, print the argument names as well.
    for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I) {
      // Insert commas as we go... the first arg doesn't get a comma
      if (I != F->arg_begin()) Out << ", ";
      printArgument(I, Attrs.getParamAttributes(Idx));
      Idx++;
    }
  } else {
    // Otherwise, print the types from the function type.
    for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i) {
      // Insert commas as we go... the first arg doesn't get a comma
      if (i) Out << ", ";
      
      // Output type...
      printType(FT->getParamType(i));
      
      Attributes ArgAttrs = Attrs.getParamAttributes(i+1);
      if (ArgAttrs != Attribute::None)
        Out << ' ' << Attribute::getAsString(ArgAttrs);
    }
  }

  // Finish printing arguments...
  if (FT->isVarArg()) {
    if (FT->getNumParams()) Out << ", ";
    Out << "...";  // Output varargs portion of signature!
  }
  Out << ')';
  Attributes FnAttrs = Attrs.getFnAttributes();
  if (FnAttrs != Attribute::None)
    Out << ' ' << Attribute::getAsString(Attrs.getFnAttributes());
  if (F->hasSection())
    Out << " section \"" << F->getSection() << '"';
  if (F->getAlignment())
    Out << " align " << F->getAlignment();
  if (F->hasGC())
    Out << " gc \"" << F->getGC() << '"';
  if (F->isDeclaration()) {
    Out << "\n";
  } else {
    Out << " {";

    // Output all of its basic blocks... for the function
    for (Function::const_iterator I = F->begin(), E = F->end(); I != E; ++I)
      printBasicBlock(I);

    Out << "}\n";
  }

  Machine.purgeFunction();
}

/// printArgument - This member is called for every argument that is passed into
/// the function.  Simply print it out
///
void AssemblyWriter::printArgument(const Argument *Arg, 
                                   Attributes Attrs) {
  // Output type...
  printType(Arg->getType());

  // Output parameter attributes list
  if (Attrs != Attribute::None)
    Out << ' ' << Attribute::getAsString(Attrs);

  // Output name, if available...
  if (Arg->hasName()) {
    Out << ' ';
    PrintLLVMName(Out, Arg);
  }
}

/// printBasicBlock - This member is called for each basic block in a method.
///
void AssemblyWriter::printBasicBlock(const BasicBlock *BB) {
  if (BB->hasName()) {              // Print out the label if it exists...
    Out << "\n";
    PrintLLVMName(Out, BB->getNameStart(), BB->getNameLen(), LabelPrefix);
    Out << ':';
  } else if (!BB->use_empty()) {      // Don't print block # of no uses...
    Out << "\n; <label>:";
    int Slot = Machine.getLocalSlot(BB);
    if (Slot != -1)
      Out << Slot;
    else
      Out << "<badref>";
  }

  if (BB->getParent() == 0)
    Out << "\t\t; Error: Block without parent!";
  else if (BB != &BB->getParent()->getEntryBlock()) {  // Not the entry block?
    // Output predecessors for the block...
    Out << "\t\t;";
    pred_const_iterator PI = pred_begin(BB), PE = pred_end(BB);
    
    if (PI == PE) {
      Out << " No predecessors!";
    } else {
      Out << " preds = ";
      writeOperand(*PI, false);
      for (++PI; PI != PE; ++PI) {
        Out << ", ";
        writeOperand(*PI, false);
      }
    }
  }

  Out << "\n";

  if (AnnotationWriter) AnnotationWriter->emitBasicBlockStartAnnot(BB, Out);

  // Output all of the instructions in the basic block...
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    printInstruction(*I);

  if (AnnotationWriter) AnnotationWriter->emitBasicBlockEndAnnot(BB, Out);
}


/// printInfoComment - Print a little comment after the instruction indicating
/// which slot it occupies.
///
void AssemblyWriter::printInfoComment(const Value &V) {
  if (V.getType() != Type::VoidTy) {
    Out << "\t\t; <";
    printType(V.getType());
    Out << '>';

    if (!V.hasName() && !isa<Instruction>(V)) {
      int SlotNum;
      if (const GlobalValue *GV = dyn_cast<GlobalValue>(&V))
        SlotNum = Machine.getGlobalSlot(GV);
      else
        SlotNum = Machine.getLocalSlot(&V);
      if (SlotNum == -1)
        Out << ":<badref>";
      else
        Out << ':' << SlotNum; // Print out the def slot taken.
    }
    Out << " [#uses=" << V.getNumUses() << ']';  // Output # uses
  }
}

// This member is called for each Instruction in a function..
void AssemblyWriter::printInstruction(const Instruction &I) {
  if (AnnotationWriter) AnnotationWriter->emitInstructionAnnot(&I, Out);

  Out << '\t';

  // Print out name if it exists...
  if (I.hasName()) {
    PrintLLVMName(Out, &I);
    Out << " = ";
  } else if (I.getType() != Type::VoidTy) {
    // Print out the def slot taken.
    int SlotNum = Machine.getLocalSlot(&I);
    if (SlotNum == -1)
      Out << "<badref> = ";
    else
      Out << '%' << SlotNum << " = ";
  }

  // If this is a volatile load or store, print out the volatile marker.
  if ((isa<LoadInst>(I)  && cast<LoadInst>(I).isVolatile()) ||
      (isa<StoreInst>(I) && cast<StoreInst>(I).isVolatile())) {
      Out << "volatile ";
  } else if (isa<CallInst>(I) && cast<CallInst>(I).isTailCall()) {
    // If this is a call, check if it's a tail call.
    Out << "tail ";
  }

  // Print out the opcode...
  Out << I.getOpcodeName();

  // Print out the compare instruction predicates
  if (const CmpInst *CI = dyn_cast<CmpInst>(&I))
    Out << ' ' << getPredicateText(CI->getPredicate());

  // Print out the type of the operands...
  const Value *Operand = I.getNumOperands() ? I.getOperand(0) : 0;

  // Special case conditional branches to swizzle the condition out to the front
  if (isa<BranchInst>(I) && I.getNumOperands() > 1) {
    Out << ' ';
    writeOperand(I.getOperand(2), true);
    Out << ", ";
    writeOperand(Operand, true);
    Out << ", ";
    writeOperand(I.getOperand(1), true);

  } else if (isa<SwitchInst>(I)) {
    // Special case switch statement to get formatting nice and correct...
    Out << ' ';
    writeOperand(Operand        , true);
    Out << ", ";
    writeOperand(I.getOperand(1), true);
    Out << " [";

    for (unsigned op = 2, Eop = I.getNumOperands(); op < Eop; op += 2) {
      Out << "\n\t\t";
      writeOperand(I.getOperand(op  ), true);
      Out << ", ";
      writeOperand(I.getOperand(op+1), true);
    }
    Out << "\n\t]";
  } else if (isa<PHINode>(I)) {
    Out << ' ';
    printType(I.getType());
    Out << ' ';

    for (unsigned op = 0, Eop = I.getNumOperands(); op < Eop; op += 2) {
      if (op) Out << ", ";
      Out << "[ ";
      writeOperand(I.getOperand(op  ), false); Out << ", ";
      writeOperand(I.getOperand(op+1), false); Out << " ]";
    }
  } else if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(&I)) {
    Out << ' ';
    writeOperand(I.getOperand(0), true);
    for (const unsigned *i = EVI->idx_begin(), *e = EVI->idx_end(); i != e; ++i)
      Out << ", " << *i;
  } else if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(&I)) {
    Out << ' ';
    writeOperand(I.getOperand(0), true); Out << ", ";
    writeOperand(I.getOperand(1), true);
    for (const unsigned *i = IVI->idx_begin(), *e = IVI->idx_end(); i != e; ++i)
      Out << ", " << *i;
  } else if (isa<ReturnInst>(I) && !Operand) {
    Out << " void";
  } else if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
    // Print the calling convention being used.
    switch (CI->getCallingConv()) {
    case CallingConv::C: break;   // default
    case CallingConv::Fast:  Out << " fastcc"; break;
    case CallingConv::Cold:  Out << " coldcc"; break;
    case CallingConv::X86_StdCall:  Out << " x86_stdcallcc"; break;
    case CallingConv::X86_FastCall: Out << " x86_fastcallcc"; break; 
    default: Out << " cc" << CI->getCallingConv(); break;
    }

    const PointerType    *PTy = cast<PointerType>(Operand->getType());
    const FunctionType   *FTy = cast<FunctionType>(PTy->getElementType());
    const Type         *RetTy = FTy->getReturnType();
    const AttrListPtr &PAL = CI->getAttributes();

    if (PAL.getRetAttributes() != Attribute::None)
      Out << ' ' << Attribute::getAsString(PAL.getRetAttributes());

    // If possible, print out the short form of the call instruction.  We can
    // only do this if the first argument is a pointer to a nonvararg function,
    // and if the return type is not a pointer to a function.
    //
    Out << ' ';
    if (!FTy->isVarArg() &&
        (!isa<PointerType>(RetTy) ||
         !isa<FunctionType>(cast<PointerType>(RetTy)->getElementType()))) {
      printType(RetTy);
      Out << ' ';
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }
    Out << '(';
    for (unsigned op = 1, Eop = I.getNumOperands(); op < Eop; ++op) {
      if (op > 1)
        Out << ", ";
      writeParamOperand(I.getOperand(op), PAL.getParamAttributes(op));
    }
    Out << ')';
    if (PAL.getFnAttributes() != Attribute::None)
      Out << ' ' << Attribute::getAsString(PAL.getFnAttributes());
  } else if (const InvokeInst *II = dyn_cast<InvokeInst>(&I)) {
    const PointerType    *PTy = cast<PointerType>(Operand->getType());
    const FunctionType   *FTy = cast<FunctionType>(PTy->getElementType());
    const Type         *RetTy = FTy->getReturnType();
    const AttrListPtr &PAL = II->getAttributes();

    // Print the calling convention being used.
    switch (II->getCallingConv()) {
    case CallingConv::C: break;   // default
    case CallingConv::Fast:  Out << " fastcc"; break;
    case CallingConv::Cold:  Out << " coldcc"; break;
    case CallingConv::X86_StdCall:  Out << " x86_stdcallcc"; break;
    case CallingConv::X86_FastCall: Out << " x86_fastcallcc"; break;
    default: Out << " cc" << II->getCallingConv(); break;
    }

    if (PAL.getRetAttributes() != Attribute::None)
      Out << ' ' << Attribute::getAsString(PAL.getRetAttributes());

    // If possible, print out the short form of the invoke instruction. We can
    // only do this if the first argument is a pointer to a nonvararg function,
    // and if the return type is not a pointer to a function.
    //
    Out << ' ';
    if (!FTy->isVarArg() &&
        (!isa<PointerType>(RetTy) ||
         !isa<FunctionType>(cast<PointerType>(RetTy)->getElementType()))) {
      printType(RetTy);
      Out << ' ';
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }
    Out << '(';
    for (unsigned op = 3, Eop = I.getNumOperands(); op < Eop; ++op) {
      if (op > 3)
        Out << ", ";
      writeParamOperand(I.getOperand(op), PAL.getParamAttributes(op-2));
    }

    Out << ')';
    if (PAL.getFnAttributes() != Attribute::None)
      Out << ' ' << Attribute::getAsString(PAL.getFnAttributes());

    Out << "\n\t\t\tto ";
    writeOperand(II->getNormalDest(), true);
    Out << " unwind ";
    writeOperand(II->getUnwindDest(), true);

  } else if (const AllocationInst *AI = dyn_cast<AllocationInst>(&I)) {
    Out << ' ';
    printType(AI->getType()->getElementType());
    if (AI->isArrayAllocation()) {
      Out << ", ";
      writeOperand(AI->getArraySize(), true);
    }
    if (AI->getAlignment()) {
      Out << ", align " << AI->getAlignment();
    }
  } else if (isa<CastInst>(I)) {
    if (Operand) {
      Out << ' ';
      writeOperand(Operand, true);   // Work with broken code
    }
    Out << " to ";
    printType(I.getType());
  } else if (isa<VAArgInst>(I)) {
    if (Operand) {
      Out << ' ';
      writeOperand(Operand, true);   // Work with broken code
    }
    Out << ", ";
    printType(I.getType());
  } else if (Operand) {   // Print the normal way...

    // PrintAllTypes - Instructions who have operands of all the same type
    // omit the type from all but the first operand.  If the instruction has
    // different type operands (for example br), then they are all printed.
    bool PrintAllTypes = false;
    const Type *TheType = Operand->getType();

    // Select, Store and ShuffleVector always print all types.
    if (isa<SelectInst>(I) || isa<StoreInst>(I) || isa<ShuffleVectorInst>(I)
        || isa<ReturnInst>(I)) {
      PrintAllTypes = true;
    } else {
      for (unsigned i = 1, E = I.getNumOperands(); i != E; ++i) {
        Operand = I.getOperand(i);
        // note that Operand shouldn't be null, but the test helps make dump()
        // more tolerant of malformed IR
        if (Operand && Operand->getType() != TheType) {
          PrintAllTypes = true;    // We have differing types!  Print them all!
          break;
        }
      }
    }

    if (!PrintAllTypes) {
      Out << ' ';
      printType(TheType);
    }

    Out << ' ';
    for (unsigned i = 0, E = I.getNumOperands(); i != E; ++i) {
      if (i) Out << ", ";
      writeOperand(I.getOperand(i), PrintAllTypes);
    }
  }
  
  // Print post operand alignment for load/store
  if (isa<LoadInst>(I) && cast<LoadInst>(I).getAlignment()) {
    Out << ", align " << cast<LoadInst>(I).getAlignment();
  } else if (isa<StoreInst>(I) && cast<StoreInst>(I).getAlignment()) {
    Out << ", align " << cast<StoreInst>(I).getAlignment();
  }

  printInfoComment(I);
  Out << '\n';
}


//===----------------------------------------------------------------------===//
//                       External Interface declarations
//===----------------------------------------------------------------------===//

void Module::print(std::ostream &o, AssemblyAnnotationWriter *AAW) const {
  raw_os_ostream OS(o);
  print(OS, AAW);
}
void Module::print(raw_ostream &OS, AssemblyAnnotationWriter *AAW) const {
  SlotTracker SlotTable(this);
  AssemblyWriter W(OS, SlotTable, this, AAW);
  W.write(this);
}

void Type::print(std::ostream &o) const {
  raw_os_ostream OS(o);
  print(OS);
}

void Type::print(raw_ostream &o) const {
  if (this == 0)
    o << "<null Type>";
  else
    o << getDescription();
}

void Value::print(raw_ostream &OS, AssemblyAnnotationWriter *AAW) const {
  if (this == 0) {
    OS << "printing a <null> value\n";
    return;
  }

  if (const Instruction *I = dyn_cast<Instruction>(this)) {
    const Function *F = I->getParent() ? I->getParent()->getParent() : 0;
    SlotTracker SlotTable(F);
    AssemblyWriter W(OS, SlotTable, F ? F->getParent() : 0, AAW);
    W.write(I);
  } else if (const BasicBlock *BB = dyn_cast<BasicBlock>(this)) {
    SlotTracker SlotTable(BB->getParent());
    AssemblyWriter W(OS, SlotTable,
                     BB->getParent() ? BB->getParent()->getParent() : 0, AAW);
    W.write(BB);
  } else if (const GlobalValue *GV = dyn_cast<GlobalValue>(this)) {
    SlotTracker SlotTable(GV->getParent());
    AssemblyWriter W(OS, SlotTable, GV->getParent(), 0);
    W.write(GV);
  } else if (const Constant *C = dyn_cast<Constant>(this)) {
    OS << C->getType()->getDescription() << ' ';
    std::map<const Type *, std::string> TypeTable;
    WriteConstantInt(OS, C, TypeTable, 0);
  } else if (const Argument *A = dyn_cast<Argument>(this)) {
    WriteAsOperand(OS, this, true,
                   A->getParent() ? A->getParent()->getParent() : 0);
  } else if (isa<InlineAsm>(this)) {
    WriteAsOperand(OS, this, true, 0);
  } else {
    assert(0 && "Unknown value to print out!");
  }
}

void Value::print(std::ostream &O, AssemblyAnnotationWriter *AAW) const {
  raw_os_ostream OS(O);
  print(OS, AAW);
}

// Value::dump - allow easy printing of Values from the debugger.
void Value::dump() const { print(errs()); errs() << '\n'; errs().flush(); }

// Type::dump - allow easy printing of Types from the debugger.
void Type::dump() const { print(errs()); errs() << '\n'; errs().flush(); }

// Type::dump - allow easy printing of Types from the debugger.
// This one uses type names from the given context module
void Type::dump(const Module *Context) const {
  WriteTypeSymbolic(errs(), this, Context);
  errs() << '\n';
  errs().flush();
}

// Module::dump() - Allow printing of Modules from the debugger.
void Module::dump() const { print(errs(), 0); errs().flush(); }


