//===-- Writer.cpp - Library for converting LLVM code to C ----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This library converts LLVM code to C code, compilable by GCC and other C
// compilers.
//
//===----------------------------------------------------------------------===//

#include "CTargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/SymbolTable.h"
#include "llvm/Intrinsics.h"
#include "llvm/Analysis/ConstantsScanner.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/Mangler.h"
#include "Support/StringExtras.h"
#include "Support/MathExtras.h"
#include "Config/config.h"
#include <algorithm>
#include <iostream>
#include <sstream>
using namespace llvm;

namespace {
  // Register the target.
  RegisterTarget<CTargetMachine> X("c", "  C backend");

  /// NameAllUsedStructs - This pass inserts names for any unnamed structure
  /// types that are used by the program.
  ///
  class CBackendNameAllUsedStructs : public Pass {
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<FindUsedTypes>();
    }

    virtual const char *getPassName() const {
      return "C backend type canonicalizer";
    }

    virtual bool run(Module &M);
  };
  
  /// CWriter - This class is the main chunk of code that converts an LLVM
  /// module to a C translation unit.
  class CWriter : public FunctionPass, public InstVisitor<CWriter> {
    std::ostream &Out; 
    IntrinsicLowering &IL;
    Mangler *Mang;
    LoopInfo *LI;
    const Module *TheModule;
    std::map<const Type *, std::string> TypeNames;

    std::map<const ConstantFP *, unsigned> FPConstantMap;
  public:
    CWriter(std::ostream &o, IntrinsicLowering &il) : Out(o), IL(il) {}

    virtual const char *getPassName() const { return "C backend"; }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.setPreservesAll();
    }

    virtual bool doInitialization(Module &M);

    bool runOnFunction(Function &F) {
      LI = &getAnalysis<LoopInfo>();

      // Output all floating point constants that cannot be printed accurately.
      printFloatingPointConstants(F);
  
      lowerIntrinsics(F);
      printFunction(F);
      FPConstantMap.clear();
      return false;
    }

    virtual bool doFinalization(Module &M) {
      // Free memory...
      delete Mang;
      TypeNames.clear();
      return false;
    }

    std::ostream &printType(std::ostream &Out, const Type *Ty,
                            const std::string &VariableName = "",
                            bool IgnoreName = false);

    void writeOperand(Value *Operand);
    void writeOperandInternal(Value *Operand);

  private :
    void lowerIntrinsics(Function &F);

    bool nameAllUsedStructureTypes(Module &M);
    void printModule(Module *M);
    void printModuleTypes(const SymbolTable &ST);
    void printContainedStructs(const Type *Ty, std::set<const StructType *> &);
    void printFloatingPointConstants(Function &F);
    void printFunctionSignature(const Function *F, bool Prototype);

    void printFunction(Function &);
    void printBasicBlock(BasicBlock *BB);
    void printLoop(Loop *L);

    void printConstant(Constant *CPV);
    void printConstantArray(ConstantArray *CPA);

    // isInlinableInst - Attempt to inline instructions into their uses to build
    // trees as much as possible.  To do this, we have to consistently decide
    // what is acceptable to inline, so that variable declarations don't get
    // printed and an extra copy of the expr is not emitted.
    //
    static bool isInlinableInst(const Instruction &I) {
      // Always inline setcc instructions, even if they are shared by multiple
      // expressions.  GCC generates horrible code if we don't.
      if (isa<SetCondInst>(I)) return true;

      // Must be an expression, must be used exactly once.  If it is dead, we
      // emit it inline where it would go.
      if (I.getType() == Type::VoidTy || !I.hasOneUse() ||
          isa<TerminatorInst>(I) || isa<CallInst>(I) || isa<PHINode>(I) || 
          isa<LoadInst>(I) || isa<VAArgInst>(I) || isa<VANextInst>(I))
        // Don't inline a load across a store or other bad things!
        return false;

      // Only inline instruction it it's use is in the same BB as the inst.
      return I.getParent() == cast<Instruction>(I.use_back())->getParent();
    }

    // isDirectAlloca - Define fixed sized allocas in the entry block as direct
    // variables which are accessed with the & operator.  This causes GCC to
    // generate significantly better code than to emit alloca calls directly.
    //
    static const AllocaInst *isDirectAlloca(const Value *V) {
      const AllocaInst *AI = dyn_cast<AllocaInst>(V);
      if (!AI) return false;
      if (AI->isArrayAllocation())
        return 0;   // FIXME: we can also inline fixed size array allocas!
      if (AI->getParent() != &AI->getParent()->getParent()->getEntryBlock())
        return 0;
      return AI;
    }

    // Instruction visitation functions
    friend class InstVisitor<CWriter>;

    void visitReturnInst(ReturnInst &I);
    void visitBranchInst(BranchInst &I);
    void visitSwitchInst(SwitchInst &I);
    void visitInvokeInst(InvokeInst &I) {
      assert(0 && "Lowerinvoke pass didn't work!");
    }

    void visitUnwindInst(UnwindInst &I) {
      assert(0 && "Lowerinvoke pass didn't work!");
    }

    void visitPHINode(PHINode &I);
    void visitBinaryOperator(Instruction &I);

    void visitCastInst (CastInst &I);
    void visitSelectInst(SelectInst &I);
    void visitCallInst (CallInst &I);
    void visitCallSite (CallSite CS);
    void visitShiftInst(ShiftInst &I) { visitBinaryOperator(I); }

    void visitMallocInst(MallocInst &I);
    void visitAllocaInst(AllocaInst &I);
    void visitFreeInst  (FreeInst   &I);
    void visitLoadInst  (LoadInst   &I);
    void visitStoreInst (StoreInst  &I);
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitVANextInst(VANextInst &I);
    void visitVAArgInst (VAArgInst &I);

    void visitInstruction(Instruction &I) {
      std::cerr << "C Writer does not know about " << I;
      abort();
    }

    void outputLValue(Instruction *I) {
      Out << "  " << Mang->getValueName(I) << " = ";
    }

    bool isGotoCodeNecessary(BasicBlock *From, BasicBlock *To);
    void printPHICopiesForSuccessors(BasicBlock *CurBlock, 
                                     unsigned Indent);
    void printBranchToBlock(BasicBlock *CurBlock, BasicBlock *SuccBlock,
                            unsigned Indent);
    void printIndexingExpression(Value *Ptr, gep_type_iterator I,
                                 gep_type_iterator E);
  };
}

/// This method inserts names for any unnamed structure types that are used by
/// the program, and removes names from structure types that are not used by the
/// program.
///
bool CBackendNameAllUsedStructs::run(Module &M) {
  // Get a set of types that are used by the program...
  std::set<const Type *> UT = getAnalysis<FindUsedTypes>().getTypes();
  
  // Loop over the module symbol table, removing types from UT that are
  // already named, and removing names for structure types that are not used.
  //
  SymbolTable &MST = M.getSymbolTable();
  for (SymbolTable::type_iterator TI = MST.type_begin(), TE = MST.type_end();
       TI != TE; ) {
    SymbolTable::type_iterator I = TI++;
    if (const StructType *STy = dyn_cast<StructType>(I->second)) {
      // If this is not used, remove it from the symbol table.
      std::set<const Type *>::iterator UTI = UT.find(STy);
      if (UTI == UT.end())
        MST.remove(I);
      else
        UT.erase(UTI);
    }
  }

  // UT now contains types that are not named.  Loop over it, naming
  // structure types.
  //
  bool Changed = false;
  unsigned RenameCounter = 0;
  for (std::set<const Type *>::const_iterator I = UT.begin(), E = UT.end();
       I != E; ++I)
    if (const StructType *ST = dyn_cast<StructType>(*I)) {
      while (M.addTypeName("unnamed"+utostr(RenameCounter), ST))
        ++RenameCounter;
      Changed = true;
    }
  return Changed;
}


// Pass the Type* and the variable name and this prints out the variable
// declaration.
//
std::ostream &CWriter::printType(std::ostream &Out, const Type *Ty,
                                 const std::string &NameSoFar,
                                 bool IgnoreName) {
  if (Ty->isPrimitiveType())
    switch (Ty->getTypeID()) {
    case Type::VoidTyID:   return Out << "void "               << NameSoFar;
    case Type::BoolTyID:   return Out << "bool "               << NameSoFar;
    case Type::UByteTyID:  return Out << "unsigned char "      << NameSoFar;
    case Type::SByteTyID:  return Out << "signed char "        << NameSoFar;
    case Type::UShortTyID: return Out << "unsigned short "     << NameSoFar;
    case Type::ShortTyID:  return Out << "short "              << NameSoFar;
    case Type::UIntTyID:   return Out << "unsigned "           << NameSoFar;
    case Type::IntTyID:    return Out << "int "                << NameSoFar;
    case Type::ULongTyID:  return Out << "unsigned long long " << NameSoFar;
    case Type::LongTyID:   return Out << "signed long long "   << NameSoFar;
    case Type::FloatTyID:  return Out << "float "              << NameSoFar;
    case Type::DoubleTyID: return Out << "double "             << NameSoFar;
    default :
      std::cerr << "Unknown primitive type: " << *Ty << "\n";
      abort();
    }
  
  // Check to see if the type is named.
  if (!IgnoreName || isa<OpaqueType>(Ty)) {
    std::map<const Type *, std::string>::iterator I = TypeNames.find(Ty);
    if (I != TypeNames.end()) return Out << I->second << " " << NameSoFar;
  }

  switch (Ty->getTypeID()) {
  case Type::FunctionTyID: {
    const FunctionType *MTy = cast<FunctionType>(Ty);
    std::stringstream FunctionInnards; 
    FunctionInnards << " (" << NameSoFar << ") (";
    for (FunctionType::param_iterator I = MTy->param_begin(),
           E = MTy->param_end(); I != E; ++I) {
      if (I != MTy->param_begin())
        FunctionInnards << ", ";
      printType(FunctionInnards, *I, "");
    }
    if (MTy->isVarArg()) {
      if (MTy->getNumParams()) 
        FunctionInnards << ", ...";
    } else if (!MTy->getNumParams()) {
      FunctionInnards << "void";
    }
    FunctionInnards << ")";
    std::string tstr = FunctionInnards.str();
    printType(Out, MTy->getReturnType(), tstr);
    return Out;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<StructType>(Ty);
    Out << NameSoFar + " {\n";
    unsigned Idx = 0;
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      Out << "  ";
      printType(Out, *I, "field" + utostr(Idx++));
      Out << ";\n";
    }
    return Out << "}";
  }  

  case Type::PointerTyID: {
    const PointerType *PTy = cast<PointerType>(Ty);
    std::string ptrName = "*" + NameSoFar;

    if (isa<ArrayType>(PTy->getElementType()))
      ptrName = "(" + ptrName + ")";

    return printType(Out, PTy->getElementType(), ptrName);
  }

  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    unsigned NumElements = ATy->getNumElements();
    return printType(Out, ATy->getElementType(),
                     NameSoFar + "[" + utostr(NumElements) + "]");
  }

  case Type::OpaqueTyID: {
    static int Count = 0;
    std::string TyName = "struct opaque_" + itostr(Count++);
    assert(TypeNames.find(Ty) == TypeNames.end());
    TypeNames[Ty] = TyName;
    return Out << TyName << " " << NameSoFar;
  }
  default:
    assert(0 && "Unhandled case in getTypeProps!");
    abort();
  }

  return Out;
}

void CWriter::printConstantArray(ConstantArray *CPA) {

  // As a special case, print the array as a string if it is an array of
  // ubytes or an array of sbytes with positive values.
  // 
  const Type *ETy = CPA->getType()->getElementType();
  bool isString = (ETy == Type::SByteTy || ETy == Type::UByteTy);

  // Make sure the last character is a null char, as automatically added by C
  if (isString && (CPA->getNumOperands() == 0 ||
                   !cast<Constant>(*(CPA->op_end()-1))->isNullValue()))
    isString = false;
  
  if (isString) {
    Out << "\"";
    // Keep track of whether the last number was a hexadecimal escape
    bool LastWasHex = false;

    // Do not include the last character, which we know is null
    for (unsigned i = 0, e = CPA->getNumOperands()-1; i != e; ++i) {
      unsigned char C = cast<ConstantInt>(CPA->getOperand(i))->getRawValue();
      
      // Print it out literally if it is a printable character.  The only thing
      // to be careful about is when the last letter output was a hex escape
      // code, in which case we have to be careful not to print out hex digits
      // explicitly (the C compiler thinks it is a continuation of the previous
      // character, sheesh...)
      //
      if (isprint(C) && (!LastWasHex || !isxdigit(C))) {
        LastWasHex = false;
        if (C == '"' || C == '\\')
          Out << "\\" << C;
        else
          Out << C;
      } else {
        LastWasHex = false;
        switch (C) {
        case '\n': Out << "\\n"; break;
        case '\t': Out << "\\t"; break;
        case '\r': Out << "\\r"; break;
        case '\v': Out << "\\v"; break;
        case '\a': Out << "\\a"; break;
        case '\"': Out << "\\\""; break;
        case '\'': Out << "\\\'"; break;           
        default:
          Out << "\\x";
          Out << (char)(( C/16  < 10) ? ( C/16 +'0') : ( C/16 -10+'A'));
          Out << (char)(((C&15) < 10) ? ((C&15)+'0') : ((C&15)-10+'A'));
          LastWasHex = true;
          break;
        }
      }
    }
    Out << "\"";
  } else {
    Out << "{";
    if (CPA->getNumOperands()) {
      Out << " ";
      printConstant(cast<Constant>(CPA->getOperand(0)));
      for (unsigned i = 1, e = CPA->getNumOperands(); i != e; ++i) {
        Out << ", ";
        printConstant(cast<Constant>(CPA->getOperand(i)));
      }
    }
    Out << " }";
  }
}

// isFPCSafeToPrint - Returns true if we may assume that CFP may be written out
// textually as a double (rather than as a reference to a stack-allocated
// variable). We decide this by converting CFP to a string and back into a
// double, and then checking whether the conversion results in a bit-equal
// double to the original value of CFP. This depends on us and the target C
// compiler agreeing on the conversion process (which is pretty likely since we
// only deal in IEEE FP).
//
static bool isFPCSafeToPrint(const ConstantFP *CFP) {
#if HAVE_PRINTF_A
  char Buffer[100];
  sprintf(Buffer, "%a", CFP->getValue());

  if (!strncmp(Buffer, "0x", 2) ||
      !strncmp(Buffer, "-0x", 3) ||
      !strncmp(Buffer, "+0x", 3))
    return atof(Buffer) == CFP->getValue();
  return false;
#else
  std::string StrVal = ftostr(CFP->getValue());

  while (StrVal[0] == ' ')
    StrVal.erase(StrVal.begin());

  // Check to make sure that the stringized number is not some string like "Inf"
  // or NaN.  Check that the string matches the "[-+]?[0-9]" regex.
  if ((StrVal[0] >= '0' && StrVal[0] <= '9') ||
      ((StrVal[0] == '-' || StrVal[0] == '+') &&
       (StrVal[1] >= '0' && StrVal[1] <= '9')))
    // Reparse stringized version!
    return atof(StrVal.c_str()) == CFP->getValue();
  return false;
#endif
}

// printConstant - The LLVM Constant to C Constant converter.
void CWriter::printConstant(Constant *CPV) {
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CPV)) {
    switch (CE->getOpcode()) {
    case Instruction::Cast:
      Out << "((";
      printType(Out, CPV->getType());
      Out << ")";
      printConstant(CE->getOperand(0));
      Out << ")";
      return;

    case Instruction::GetElementPtr:
      Out << "(&(";
      printIndexingExpression(CE->getOperand(0), gep_type_begin(CPV),
                              gep_type_end(CPV));
      Out << "))";
      return;
    case Instruction::Select:
      Out << "(";
      printConstant(CE->getOperand(0));
      Out << "?";
      printConstant(CE->getOperand(1));
      Out << ":";
      printConstant(CE->getOperand(2));
      Out << ")";
      return;
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::Div:
    case Instruction::Rem:
    case Instruction::SetEQ:
    case Instruction::SetNE:
    case Instruction::SetLT:
    case Instruction::SetLE:
    case Instruction::SetGT:
    case Instruction::SetGE:
    case Instruction::Shl:
    case Instruction::Shr:
      Out << "(";
      printConstant(CE->getOperand(0));
      switch (CE->getOpcode()) {
      case Instruction::Add: Out << " + "; break;
      case Instruction::Sub: Out << " - "; break;
      case Instruction::Mul: Out << " * "; break;
      case Instruction::Div: Out << " / "; break;
      case Instruction::Rem: Out << " % "; break;
      case Instruction::SetEQ: Out << " == "; break;
      case Instruction::SetNE: Out << " != "; break;
      case Instruction::SetLT: Out << " < "; break;
      case Instruction::SetLE: Out << " <= "; break;
      case Instruction::SetGT: Out << " > "; break;
      case Instruction::SetGE: Out << " >= "; break;
      case Instruction::Shl: Out << " << "; break;
      case Instruction::Shr: Out << " >> "; break;
      default: assert(0 && "Illegal opcode here!");
      }
      printConstant(CE->getOperand(1));
      Out << ")";
      return;

    default:
      std::cerr << "CWriter Error: Unhandled constant expression: "
                << *CE << "\n";
      abort();
    }
  }

  switch (CPV->getType()->getTypeID()) {
  case Type::BoolTyID:
    Out << (CPV == ConstantBool::False ? "0" : "1"); break;
  case Type::SByteTyID:
  case Type::ShortTyID:
    Out << cast<ConstantSInt>(CPV)->getValue(); break;
  case Type::IntTyID:
    if ((int)cast<ConstantSInt>(CPV)->getValue() == (int)0x80000000)
      Out << "((int)0x80000000)";   // Handle MININT specially to avoid warning
    else
      Out << cast<ConstantSInt>(CPV)->getValue();
    break;

  case Type::LongTyID:
    Out << cast<ConstantSInt>(CPV)->getValue() << "ll"; break;

  case Type::UByteTyID:
  case Type::UShortTyID:
    Out << cast<ConstantUInt>(CPV)->getValue(); break;
  case Type::UIntTyID:
    Out << cast<ConstantUInt>(CPV)->getValue() << "u"; break;
  case Type::ULongTyID:
    Out << cast<ConstantUInt>(CPV)->getValue() << "ull"; break;

  case Type::FloatTyID:
  case Type::DoubleTyID: {
    ConstantFP *FPC = cast<ConstantFP>(CPV);
    std::map<const ConstantFP*, unsigned>::iterator I = FPConstantMap.find(FPC);
    if (I != FPConstantMap.end()) {
      // Because of FP precision problems we must load from a stack allocated
      // value that holds the value in hex.
      Out << "(*(" << (FPC->getType() == Type::FloatTy ? "float" : "double")
          << "*)&FPConstant" << I->second << ")";
    } else {
      std::string Num;
#if HAVE_PRINTF_A
      // Print out the constant as a floating point number.
      char Buffer[100];
      sprintf(Buffer, "%a", FPC->getValue());
      Num = Buffer;
#else
      Num = ftostr(FPC->getValue());
#endif

      if (IsNAN(FPC->getValue())) {
        // The value is NaN
        if (FPC->getType() == Type::FloatTy)
          Out << "LLVM_NANF(\"0\") /*nan*/ ";
        else
          Out << "LLVM_NAN(\"0\") /*nan*/ ";
      } else if (IsInf(FPC->getValue())) {
        // The value is Inf
        if (FPC->getValue() < 0) Out << "-";
        if (FPC->getType() == Type::FloatTy)
          Out << "LLVM_INFF /*inf*/ ";
        else
          Out << "LLVM_INF /*inf*/ ";
      } else {
        Out << Num;
      }
    }
    break;
  }

  case Type::ArrayTyID:
    if (isa<ConstantAggregateZero>(CPV)) {
      const ArrayType *AT = cast<ArrayType>(CPV->getType());
      Out << "{";
      if (AT->getNumElements()) {
        Out << " ";
        Constant *CZ = Constant::getNullValue(AT->getElementType());
        printConstant(CZ);
        for (unsigned i = 1, e = AT->getNumElements(); i != e; ++i) {
          Out << ", ";
          printConstant(CZ);
        }
      }
      Out << " }";
    } else {
      printConstantArray(cast<ConstantArray>(CPV));
    }
    break;

  case Type::StructTyID:
    if (isa<ConstantAggregateZero>(CPV)) {
      const StructType *ST = cast<StructType>(CPV->getType());
      Out << "{";
      if (ST->getNumElements()) {
        Out << " ";
        printConstant(Constant::getNullValue(ST->getElementType(0)));
        for (unsigned i = 1, e = ST->getNumElements(); i != e; ++i) {
          Out << ", ";
          printConstant(Constant::getNullValue(ST->getElementType(i)));
        }
      }
      Out << " }";
    } else {
      Out << "{";
      if (CPV->getNumOperands()) {
        Out << " ";
        printConstant(cast<Constant>(CPV->getOperand(0)));
        for (unsigned i = 1, e = CPV->getNumOperands(); i != e; ++i) {
          Out << ", ";
          printConstant(cast<Constant>(CPV->getOperand(i)));
        }
      }
      Out << " }";
    }
    break;

  case Type::PointerTyID:
    if (isa<ConstantPointerNull>(CPV)) {
      Out << "((";
      printType(Out, CPV->getType());
      Out << ")/*NULL*/0)";
      break;
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(CPV)) {
      writeOperand(GV);
      break;
    }
    // FALL THROUGH
  default:
    std::cerr << "Unknown constant type: " << *CPV << "\n";
    abort();
  }
}

void CWriter::writeOperandInternal(Value *Operand) {
  if (Instruction *I = dyn_cast<Instruction>(Operand))
    if (isInlinableInst(*I) && !isDirectAlloca(I)) {
      // Should we inline this instruction to build a tree?
      Out << "(";
      visit(*I);
      Out << ")";    
      return;
    }
  
  Constant* CPV = dyn_cast<Constant>(Operand);
  if (CPV && !isa<GlobalValue>(CPV)) {
    printConstant(CPV); 
  } else {
    Out << Mang->getValueName(Operand);
  }
}

void CWriter::writeOperand(Value *Operand) {
  if (isa<GlobalVariable>(Operand) || isDirectAlloca(Operand))
    Out << "(&";  // Global variables are references as their addresses by llvm

  writeOperandInternal(Operand);

  if (isa<GlobalVariable>(Operand) || isDirectAlloca(Operand))
    Out << ")";
}

// generateCompilerSpecificCode - This is where we add conditional compilation
// directives to cater to specific compilers as need be.
//
static void generateCompilerSpecificCode(std::ostream& Out) {
  // Alloca is hard to get, and we don't want to include stdlib.h here...
  Out << "/* get a declaration for alloca */\n"
      << "#if defined(sun) || defined(__CYGWIN__) || defined(__APPLE__)\n"
      << "extern void *__builtin_alloca(unsigned long);\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#else\n"
      << "#ifndef __FreeBSD__\n"
      << "#include <alloca.h>\n"
      << "#endif\n"
      << "#endif\n\n";

  // We output GCC specific attributes to preserve 'linkonce'ness on globals.
  // If we aren't being compiled with GCC, just drop these attributes.
  Out << "#ifndef __GNUC__  /* Can only support \"linkonce\" vars with GCC */\n"
      << "#define __attribute__(X)\n"
      << "#endif\n\n";

#if 0
  // At some point, we should support "external weak" vs. "weak" linkages.
  // On Mac OS X, "external weak" is spelled "__attribute__((weak_import))".
  Out << "#if defined(__GNUC__) && defined(__APPLE_CC__)\n"
      << "#define __EXTERNAL_WEAK__ __attribute__((weak_import))\n"
      << "#elif defined(__GNUC__)\n"
      << "#define __EXTERNAL_WEAK__ __attribute__((weak))\n"
      << "#else\n"
      << "#define __EXTERNAL_WEAK__\n"
      << "#endif\n\n";
#endif

  // For now, turn off the weak linkage attribute on Mac OS X. (See above.)
  Out << "#if defined(__GNUC__) && defined(__APPLE_CC__)\n"
      << "#define __ATTRIBUTE_WEAK__\n"
      << "#elif defined(__GNUC__)\n"
      << "#define __ATTRIBUTE_WEAK__ __attribute__((weak))\n"
      << "#else\n"
      << "#define __ATTRIBUTE_WEAK__\n"
      << "#endif\n\n";

  // Define NaN and Inf as GCC builtins if using GCC, as 0 otherwise
  // From the GCC documentation:
  // 
  //   double __builtin_nan (const char *str)
  //
  // This is an implementation of the ISO C99 function nan.
  //
  // Since ISO C99 defines this function in terms of strtod, which we do
  // not implement, a description of the parsing is in order. The string is
  // parsed as by strtol; that is, the base is recognized by leading 0 or
  // 0x prefixes. The number parsed is placed in the significand such that
  // the least significant bit of the number is at the least significant
  // bit of the significand. The number is truncated to fit the significand
  // field provided. The significand is forced to be a quiet NaN.
  //
  // This function, if given a string literal, is evaluated early enough
  // that it is considered a compile-time constant.
  //
  //   float __builtin_nanf (const char *str)
  //
  // Similar to __builtin_nan, except the return type is float.
  //
  //   double __builtin_inf (void)
  //
  // Similar to __builtin_huge_val, except a warning is generated if the
  // target floating-point format does not support infinities. This
  // function is suitable for implementing the ISO C99 macro INFINITY.
  //
  //   float __builtin_inff (void)
  //
  // Similar to __builtin_inf, except the return type is float.
  Out << "#ifdef __GNUC__\n"
      << "#define LLVM_NAN(NanStr)  __builtin_nan(NanStr)   /* Double */\n"
      << "#define LLVM_NANF(NanStr) __builtin_nan(NanStr)   /* Float */\n"
      << "#define LLVM_INF          __builtin_inf()         /* Double */\n"
      << "#define LLVM_INFF         __builtin_inff()        /* Float */\n"
      << "#else\n"
      << "#define LLVM_NAN(NanStr)  ((double)0.0)           /* Double */\n"
      << "#define LLVM_NANF(NanStr) 0.0F                    /* Float */\n"
      << "#define LLVM_INF          ((double)0.0)           /* Double */\n"
      << "#define LLVM_INFF         0.0F                    /* Float */\n"
      << "#endif\n";
}

bool CWriter::doInitialization(Module &M) {
  // Initialize
  TheModule = &M;

  IL.AddPrototypes(M);
  
  // Ensure that all structure types have names...
  Mang = new Mangler(M);

  // get declaration for alloca
  Out << "/* Provide Declarations */\n";
  Out << "#include <stdarg.h>\n";      // Varargs support
  Out << "#include <setjmp.h>\n";      // Unwind support
  generateCompilerSpecificCode(Out);

  // Provide a definition for `bool' if not compiling with a C++ compiler.
  Out << "\n"
      << "#ifndef __cplusplus\ntypedef unsigned char bool;\n#endif\n"
    
      << "\n\n/* Support for floating point constants */\n"
      << "typedef unsigned long long ConstantDoubleTy;\n"
      << "typedef unsigned int        ConstantFloatTy;\n"
    
      << "\n\n/* Global Declarations */\n";

  // First output all the declarations for the program, because C requires
  // Functions & globals to be declared before they are used.
  //

  // Loop over the symbol table, emitting all named constants...
  printModuleTypes(M.getSymbolTable());

  // Global variable declarations...
  if (!M.gempty()) {
    Out << "\n/* External Global Variable Declarations */\n";
    for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I) {
      if (I->hasExternalLinkage()) {
        Out << "extern ";
        printType(Out, I->getType()->getElementType(), Mang->getValueName(I));
        Out << ";\n";
      }
    }
  }

  // Function declarations
  if (!M.empty()) {
    Out << "\n/* Function Declarations */\n";
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
      // Don't print declarations for intrinsic functions.
      if (!I->getIntrinsicID() && 
          I->getName() != "setjmp" && I->getName() != "longjmp") {
        printFunctionSignature(I, true);
        if (I->hasWeakLinkage()) Out << " __ATTRIBUTE_WEAK__";
        if (I->hasLinkOnceLinkage()) Out << " __ATTRIBUTE_WEAK__";
        Out << ";\n";
      }
    }
  }

  // Output the global variable declarations
  if (!M.gempty()) {
    Out << "\n\n/* Global Variable Declarations */\n";
    for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
      if (!I->isExternal()) {
        Out << "extern ";
        printType(Out, I->getType()->getElementType(), Mang->getValueName(I));

        if (I->hasLinkOnceLinkage())
          Out << " __attribute__((common))";
        else if (I->hasWeakLinkage())
          Out << " __ATTRIBUTE_WEAK__";
        Out << ";\n";
      }
  }

  // Output the global variable definitions and contents...
  if (!M.gempty()) {
    Out << "\n\n/* Global Variable Definitions and Initialization */\n";
    for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
      if (!I->isExternal()) {
        if (I->hasInternalLinkage())
          Out << "static ";
        printType(Out, I->getType()->getElementType(), Mang->getValueName(I));
        if (I->hasLinkOnceLinkage())
          Out << " __attribute__((common))";
        else if (I->hasWeakLinkage())
          Out << " __ATTRIBUTE_WEAK__";

        // If the initializer is not null, emit the initializer.  If it is null,
        // we try to avoid emitting large amounts of zeros.  The problem with
        // this, however, occurs when the variable has weak linkage.  In this
        // case, the assembler will complain about the variable being both weak
        // and common, so we disable this optimization.
        if (!I->getInitializer()->isNullValue()) {
          Out << " = " ;
          writeOperand(I->getInitializer());
        } else if (I->hasWeakLinkage()) {
          // We have to specify an initializer, but it doesn't have to be
          // complete.  If the value is an aggregate, print out { 0 }, and let
          // the compiler figure out the rest of the zeros.
          Out << " = " ;
          if (isa<StructType>(I->getInitializer()->getType()) ||
              isa<ArrayType>(I->getInitializer()->getType())) {
            Out << "{ 0 }";
          } else {
            // Just print it out normally.
            writeOperand(I->getInitializer());
          }
        }
        Out << ";\n";
      }
  }

  if (!M.empty())
    Out << "\n\n/* Function Bodies */\n";
  return false;
}


/// Output all floating point constants that cannot be printed accurately...
void CWriter::printFloatingPointConstants(Function &F) {
  union {
    double D;
    uint64_t U;
  } DBLUnion;

  union {
    float F;
    unsigned U;
  } FLTUnion;

  // Scan the module for floating point constants.  If any FP constant is used
  // in the function, we want to redirect it here so that we do not depend on
  // the precision of the printed form, unless the printed form preserves
  // precision.
  //
  static unsigned FPCounter = 0;
  for (constant_iterator I = constant_begin(&F), E = constant_end(&F);
       I != E; ++I)
    if (const ConstantFP *FPC = dyn_cast<ConstantFP>(*I))
      if (!isFPCSafeToPrint(FPC) && // Do not put in FPConstantMap if safe.
          !FPConstantMap.count(FPC)) {
        double Val = FPC->getValue();
        
        FPConstantMap[FPC] = FPCounter;  // Number the FP constants
        
        if (FPC->getType() == Type::DoubleTy) {
          DBLUnion.D = Val;
          Out << "static const ConstantDoubleTy FPConstant" << FPCounter++
              << " = 0x" << std::hex << DBLUnion.U << std::dec
              << "ULL;    /* " << Val << " */\n";
        } else if (FPC->getType() == Type::FloatTy) {
          FLTUnion.F = Val;
          Out << "static const ConstantFloatTy FPConstant" << FPCounter++
              << " = 0x" << std::hex << FLTUnion.U << std::dec
              << "U;    /* " << Val << " */\n";
        } else
          assert(0 && "Unknown float type!");
      }
  
  Out << "\n";
}


/// printSymbolTable - Run through symbol table looking for type names.  If a
/// type name is found, emit it's declaration...
///
void CWriter::printModuleTypes(const SymbolTable &ST) {
  // If there are no type names, exit early.
  if ( ! ST.hasTypes() )
    return;

  // We are only interested in the type plane of the symbol table...
  SymbolTable::type_const_iterator I   = ST.type_begin();
  SymbolTable::type_const_iterator End = ST.type_end();
  
  // Print out forward declarations for structure types before anything else!
  Out << "/* Structure forward decls */\n";
  for (; I != End; ++I)
    if (const Type *STy = dyn_cast<StructType>(I->second)) {
      std::string Name = "struct l_" + Mangler::makeNameProper(I->first);
      Out << Name << ";\n";
      TypeNames.insert(std::make_pair(STy, Name));
    }

  Out << "\n";

  // Now we can print out typedefs...
  Out << "/* Typedefs */\n";
  for (I = ST.type_begin(); I != End; ++I) {
    const Type *Ty = cast<Type>(I->second);
    std::string Name = "l_" + Mangler::makeNameProper(I->first);
    Out << "typedef ";
    printType(Out, Ty, Name);
    Out << ";\n";
  }
  
  Out << "\n";

  // Keep track of which structures have been printed so far...
  std::set<const StructType *> StructPrinted;

  // Loop over all structures then push them into the stack so they are
  // printed in the correct order.
  //
  Out << "/* Structure contents */\n";
  for (I = ST.type_begin(); I != End; ++I)
    if (const StructType *STy = dyn_cast<StructType>(I->second))
      // Only print out used types!
      printContainedStructs(STy, StructPrinted);
}

// Push the struct onto the stack and recursively push all structs
// this one depends on.
void CWriter::printContainedStructs(const Type *Ty,
                                    std::set<const StructType*> &StructPrinted){
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    //Check to see if we have already printed this struct
    if (StructPrinted.count(STy) == 0) {
      // Print all contained types first...
      for (StructType::element_iterator I = STy->element_begin(),
             E = STy->element_end(); I != E; ++I) {
        const Type *Ty1 = I->get();
        if (isa<StructType>(Ty1) || isa<ArrayType>(Ty1))
          printContainedStructs(*I, StructPrinted);
      }
      
      //Print structure type out..
      StructPrinted.insert(STy);
      std::string Name = TypeNames[STy];  
      printType(Out, STy, Name, true);
      Out << ";\n\n";
    }

    // If it is an array, check contained types and continue
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)){
    const Type *Ty1 = ATy->getElementType();
    if (isa<StructType>(Ty1) || isa<ArrayType>(Ty1))
      printContainedStructs(Ty1, StructPrinted);
  }
}


void CWriter::printFunctionSignature(const Function *F, bool Prototype) {
  if (F->hasInternalLinkage()) Out << "static ";
  
  // Loop over the arguments, printing them...
  const FunctionType *FT = cast<FunctionType>(F->getFunctionType());
  
  std::stringstream FunctionInnards; 
    
  // Print out the name...
  FunctionInnards << Mang->getValueName(F) << "(";
    
  if (!F->isExternal()) {
    if (!F->aempty()) {
      std::string ArgName;
      if (F->abegin()->hasName() || !Prototype)
        ArgName = Mang->getValueName(F->abegin());
      printType(FunctionInnards, F->afront().getType(), ArgName);
      for (Function::const_aiterator I = ++F->abegin(), E = F->aend();
           I != E; ++I) {
        FunctionInnards << ", ";
        if (I->hasName() || !Prototype)
          ArgName = Mang->getValueName(I);
        else 
          ArgName = "";
        printType(FunctionInnards, I->getType(), ArgName);
      }
    }
  } else {
    // Loop over the arguments, printing them...
    for (FunctionType::param_iterator I = FT->param_begin(),
           E = FT->param_end(); I != E; ++I) {
      if (I != FT->param_begin()) FunctionInnards << ", ";
      printType(FunctionInnards, *I);
    }
  }

  // Finish printing arguments... if this is a vararg function, print the ...,
  // unless there are no known types, in which case, we just emit ().
  //
  if (FT->isVarArg() && FT->getNumParams()) {
    if (FT->getNumParams()) FunctionInnards << ", ";
    FunctionInnards << "...";  // Output varargs portion of signature!
  } else if (!FT->isVarArg() && FT->getNumParams() == 0) {
    FunctionInnards << "void"; // ret() -> ret(void) in C.
  }
  FunctionInnards << ")";
  // Print out the return type and the entire signature for that matter
  printType(Out, F->getReturnType(), FunctionInnards.str());
}

void CWriter::printFunction(Function &F) {
  printFunctionSignature(&F, false);
  Out << " {\n";

  // print local variable information for the function
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I)
    if (const AllocaInst *AI = isDirectAlloca(&*I)) {
      Out << "  ";
      printType(Out, AI->getAllocatedType(), Mang->getValueName(AI));
      Out << ";    /* Address exposed local */\n";
    } else if (I->getType() != Type::VoidTy && !isInlinableInst(*I)) {
      Out << "  ";
      printType(Out, I->getType(), Mang->getValueName(&*I));
      Out << ";\n";
      
      if (isa<PHINode>(*I)) {  // Print out PHI node temporaries as well...
        Out << "  ";
        printType(Out, I->getType(),
                  Mang->getValueName(&*I)+"__PHI_TEMPORARY");
        Out << ";\n";
      }
    }

  Out << "\n";

  // print the basic blocks
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    if (Loop *L = LI->getLoopFor(BB)) {
      if (L->getHeader() == BB && L->getParentLoop() == 0)
        printLoop(L);
    } else {
      printBasicBlock(BB);
    }
  }
  
  Out << "}\n\n";
}

void CWriter::printLoop(Loop *L) {
  Out << "  do {     /* Syntactic loop '" << L->getHeader()->getName()
      << "' to make GCC happy */\n";
  for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i) {
    BasicBlock *BB = L->getBlocks()[i];
    Loop *BBLoop = LI->getLoopFor(BB);
    if (BBLoop == L)
      printBasicBlock(BB);
    else if (BB == BBLoop->getHeader() && BBLoop->getParentLoop() == L)
      printLoop(BBLoop);      
  }
  Out << "  } while (1); /* end of syntactic loop '"
      << L->getHeader()->getName() << "' */\n";
}

void CWriter::printBasicBlock(BasicBlock *BB) {

  // Don't print the label for the basic block if there are no uses, or if
  // the only terminator use is the predecessor basic block's terminator.
  // We have to scan the use list because PHI nodes use basic blocks too but
  // do not require a label to be generated.
  //
  bool NeedsLabel = false;
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
    if (isGotoCodeNecessary(*PI, BB)) {
      NeedsLabel = true;
      break;
    }
      
  if (NeedsLabel) Out << Mang->getValueName(BB) << ":\n";
      
  // Output all of the instructions in the basic block...
  for (BasicBlock::iterator II = BB->begin(), E = --BB->end(); II != E;
       ++II) {
    if (!isInlinableInst(*II) && !isDirectAlloca(II)) {
      if (II->getType() != Type::VoidTy)
        outputLValue(II);
      else
        Out << "  ";
      visit(*II);
      Out << ";\n";
    }
  }
      
  // Don't emit prefix or suffix for the terminator...
  visit(*BB->getTerminator());
}


// Specific Instruction type classes... note that all of the casts are
// necessary because we use the instruction classes as opaque types...
//
void CWriter::visitReturnInst(ReturnInst &I) {
  // Don't output a void return if this is the last basic block in the function
  if (I.getNumOperands() == 0 && 
      &*--I.getParent()->getParent()->end() == I.getParent() &&
      !I.getParent()->size() == 1) {
    return;
  }

  Out << "  return";
  if (I.getNumOperands()) {
    Out << " ";
    writeOperand(I.getOperand(0));
  }
  Out << ";\n";
}

void CWriter::visitSwitchInst(SwitchInst &SI) {
  printPHICopiesForSuccessors(SI.getParent(), 0);

  Out << "  switch (";
  writeOperand(SI.getOperand(0));
  Out << ") {\n  default:\n";
  printBranchToBlock(SI.getParent(), SI.getDefaultDest(), 2);
  Out << ";\n";
  for (unsigned i = 2, e = SI.getNumOperands(); i != e; i += 2) {
    Out << "  case ";
    writeOperand(SI.getOperand(i));
    Out << ":\n";
    BasicBlock *Succ = cast<BasicBlock>(SI.getOperand(i+1));
    printBranchToBlock(SI.getParent(), Succ, 2);
    if (Succ == SI.getParent()->getNext())
      Out << "    break;\n";
  }
  Out << "  }\n";
}

bool CWriter::isGotoCodeNecessary(BasicBlock *From, BasicBlock *To) {
  /// FIXME: This should be reenabled, but loop reordering safe!!
  return true;

  if (From->getNext() != To) // Not the direct successor, we need a goto
    return true; 

  //isa<SwitchInst>(From->getTerminator())


  if (LI->getLoopFor(From) != LI->getLoopFor(To))
    return true;
  return false;
}

void CWriter::printPHICopiesForSuccessors(BasicBlock *CurBlock, 
                                          unsigned Indent) {
  for (succ_iterator SI = succ_begin(CurBlock), E = succ_end(CurBlock);
       SI != E; ++SI)
    for (BasicBlock::iterator I = SI->begin();
         PHINode *PN = dyn_cast<PHINode>(I); ++I) {
      //  now we have to do the printing
      Out << std::string(Indent, ' ');
      Out << "  " << Mang->getValueName(I) << "__PHI_TEMPORARY = ";
      writeOperand(PN->getIncomingValue(PN->getBasicBlockIndex(CurBlock)));
      Out << ";   /* for PHI node */\n";
    }
}


void CWriter::printBranchToBlock(BasicBlock *CurBB, BasicBlock *Succ,
                                 unsigned Indent) {
  if (isGotoCodeNecessary(CurBB, Succ)) {
    Out << std::string(Indent, ' ') << "  goto ";
    writeOperand(Succ);
    Out << ";\n";
  }
}

// Branch instruction printing - Avoid printing out a branch to a basic block
// that immediately succeeds the current one.
//
void CWriter::visitBranchInst(BranchInst &I) {
  printPHICopiesForSuccessors(I.getParent(), 0);

  if (I.isConditional()) {
    if (isGotoCodeNecessary(I.getParent(), I.getSuccessor(0))) {
      Out << "  if (";
      writeOperand(I.getCondition());
      Out << ") {\n";
      
      printBranchToBlock(I.getParent(), I.getSuccessor(0), 2);
      
      if (isGotoCodeNecessary(I.getParent(), I.getSuccessor(1))) {
        Out << "  } else {\n";
        printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
      }
    } else {
      // First goto not necessary, assume second one is...
      Out << "  if (!";
      writeOperand(I.getCondition());
      Out << ") {\n";

      printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
    }

    Out << "  }\n";
  } else {
    printBranchToBlock(I.getParent(), I.getSuccessor(0), 0);
  }
  Out << "\n";
}

// PHI nodes get copied into temporary values at the end of predecessor basic
// blocks.  We now need to copy these temporary values into the REAL value for
// the PHI.
void CWriter::visitPHINode(PHINode &I) {
  writeOperand(&I);
  Out << "__PHI_TEMPORARY";
}


void CWriter::visitBinaryOperator(Instruction &I) {
  // binary instructions, shift instructions, setCond instructions.
  assert(!isa<PointerType>(I.getType()));

  // We must cast the results of binary operations which might be promoted.
  bool needsCast = false;
  if ((I.getType() == Type::UByteTy) || (I.getType() == Type::SByteTy)
      || (I.getType() == Type::UShortTy) || (I.getType() == Type::ShortTy)
      || (I.getType() == Type::FloatTy)) {
    needsCast = true;
    Out << "((";
    printType(Out, I.getType());
    Out << ")(";
  }
      
  writeOperand(I.getOperand(0));

  switch (I.getOpcode()) {
  case Instruction::Add: Out << " + "; break;
  case Instruction::Sub: Out << " - "; break;
  case Instruction::Mul: Out << "*"; break;
  case Instruction::Div: Out << "/"; break;
  case Instruction::Rem: Out << "%"; break;
  case Instruction::And: Out << " & "; break;
  case Instruction::Or: Out << " | "; break;
  case Instruction::Xor: Out << " ^ "; break;
  case Instruction::SetEQ: Out << " == "; break;
  case Instruction::SetNE: Out << " != "; break;
  case Instruction::SetLE: Out << " <= "; break;
  case Instruction::SetGE: Out << " >= "; break;
  case Instruction::SetLT: Out << " < "; break;
  case Instruction::SetGT: Out << " > "; break;
  case Instruction::Shl : Out << " << "; break;
  case Instruction::Shr : Out << " >> "; break;
  default: std::cerr << "Invalid operator type!" << I; abort();
  }

  writeOperand(I.getOperand(1));

  if (needsCast) {
    Out << "))";
  }
}

void CWriter::visitCastInst(CastInst &I) {
  if (I.getType() == Type::BoolTy) {
    Out << "(";
    writeOperand(I.getOperand(0));
    Out << " != 0)";
    return;
  }
  Out << "(";
  printType(Out, I.getType());
  Out << ")";
  if (isa<PointerType>(I.getType())&&I.getOperand(0)->getType()->isIntegral() ||
      isa<PointerType>(I.getOperand(0)->getType())&&I.getType()->isIntegral()) {
    // Avoid "cast to pointer from integer of different size" warnings
    Out << "(long)";  
  }
  
  writeOperand(I.getOperand(0));
}

void CWriter::visitSelectInst(SelectInst &I) {
  Out << "((";
  writeOperand(I.getCondition());
  Out << ") ? (";
  writeOperand(I.getTrueValue());
  Out << ") : (";
  writeOperand(I.getFalseValue());
  Out << "))";    
}


void CWriter::lowerIntrinsics(Function &F) {
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(I++))
        if (Function *F = CI->getCalledFunction())
          switch (F->getIntrinsicID()) {
          case Intrinsic::not_intrinsic:
          case Intrinsic::vastart:
          case Intrinsic::vacopy:
          case Intrinsic::vaend:
          case Intrinsic::returnaddress:
          case Intrinsic::frameaddress:
          case Intrinsic::setjmp:
          case Intrinsic::longjmp:
            // We directly implement these intrinsics
            break;
          default:
            // All other intrinsic calls we must lower.
            Instruction *Before = CI->getPrev();
            IL.LowerIntrinsicCall(CI);
            if (Before) {        // Move iterator to instruction after call
              I = Before; ++I;
            } else {
              I = BB->begin();
            }
          }
}



void CWriter::visitCallInst(CallInst &I) {
  // Handle intrinsic function calls first...
  if (Function *F = I.getCalledFunction())
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID()) {
      switch (ID) {
      default: assert(0 && "Unknown LLVM intrinsic!");
      case Intrinsic::vastart: 
        Out << "0; ";
        
        Out << "va_start(*(va_list*)&" << Mang->getValueName(&I) << ", ";
        // Output the last argument to the enclosing function...
        if (I.getParent()->getParent()->aempty()) {
          std::cerr << "The C backend does not currently support zero "
                    << "argument varargs functions, such as '"
                    << I.getParent()->getParent()->getName() << "'!\n";
          abort();
        }
        writeOperand(&I.getParent()->getParent()->aback());
        Out << ")";
        return;
      case Intrinsic::vaend:
        if (!isa<ConstantPointerNull>(I.getOperand(1))) {
          Out << "va_end(*(va_list*)&";
          writeOperand(I.getOperand(1));
          Out << ")";
        } else {
          Out << "va_end(*(va_list*)0)";
        }
        return;
      case Intrinsic::vacopy:
        Out << "0;";
        Out << "va_copy(*(va_list*)&" << Mang->getValueName(&I) << ", ";
        Out << "*(va_list*)&";
        writeOperand(I.getOperand(1));
        Out << ")";
        return;
      case Intrinsic::returnaddress:
        Out << "__builtin_return_address(";
        writeOperand(I.getOperand(1));
        Out << ")";
        return;
      case Intrinsic::frameaddress:
        Out << "__builtin_frame_address(";
        writeOperand(I.getOperand(1));
        Out << ")";
        return;
      case Intrinsic::setjmp:
        Out << "setjmp(*(jmp_buf*)";
        writeOperand(I.getOperand(1));
        Out << ")";
        return;
      case Intrinsic::longjmp:
        Out << "longjmp(*(jmp_buf*)";
        writeOperand(I.getOperand(1));
        Out << ", ";
        writeOperand(I.getOperand(2));
        Out << ")";
        return;
      }
    }
  visitCallSite(&I);
}

void CWriter::visitCallSite(CallSite CS) {
  const PointerType  *PTy   = cast<PointerType>(CS.getCalledValue()->getType());
  const FunctionType *FTy   = cast<FunctionType>(PTy->getElementType());
  const Type         *RetTy = FTy->getReturnType();
  
  writeOperand(CS.getCalledValue());
  Out << "(";

  if (CS.arg_begin() != CS.arg_end()) {
    CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
    writeOperand(*AI);

    for (++AI; AI != AE; ++AI) {
      Out << ", ";
      writeOperand(*AI);
    }
  }
  Out << ")";
}  

void CWriter::visitMallocInst(MallocInst &I) {
  assert(0 && "lowerallocations pass didn't work!");
}

void CWriter::visitAllocaInst(AllocaInst &I) {
  Out << "(";
  printType(Out, I.getType());
  Out << ") alloca(sizeof(";
  printType(Out, I.getType()->getElementType());
  Out << ")";
  if (I.isArrayAllocation()) {
    Out << " * " ;
    writeOperand(I.getOperand(0));
  }
  Out << ")";
}

void CWriter::visitFreeInst(FreeInst &I) {
  assert(0 && "lowerallocations pass didn't work!");
}

void CWriter::printIndexingExpression(Value *Ptr, gep_type_iterator I,
                                      gep_type_iterator E) {
  bool HasImplicitAddress = false;
  // If accessing a global value with no indexing, avoid *(&GV) syndrome
  if (GlobalValue *V = dyn_cast<GlobalValue>(Ptr)) {
    HasImplicitAddress = true;
  } else if (isDirectAlloca(Ptr)) {
    HasImplicitAddress = true;
  }

  if (I == E) {
    if (!HasImplicitAddress)
      Out << "*";  // Implicit zero first argument: '*x' is equivalent to 'x[0]'

    writeOperandInternal(Ptr);
    return;
  }

  const Constant *CI = dyn_cast<Constant>(I.getOperand());
  if (HasImplicitAddress && (!CI || !CI->isNullValue()))
    Out << "(&";

  writeOperandInternal(Ptr);

  if (HasImplicitAddress && (!CI || !CI->isNullValue())) {
    Out << ")";
    HasImplicitAddress = false;  // HIA is only true if we haven't addressed yet
  }

  assert(!HasImplicitAddress || (CI && CI->isNullValue()) &&
         "Can only have implicit address with direct accessing");

  if (HasImplicitAddress) {
    ++I;
  } else if (CI && CI->isNullValue()) {
    gep_type_iterator TmpI = I; ++TmpI;

    // Print out the -> operator if possible...
    if (TmpI != E && isa<StructType>(*TmpI)) {
      Out << (HasImplicitAddress ? "." : "->");
      Out << "field" << cast<ConstantUInt>(TmpI.getOperand())->getValue();
      I = ++TmpI;
    }
  }

  for (; I != E; ++I)
    if (isa<StructType>(*I)) {
      Out << ".field" << cast<ConstantUInt>(I.getOperand())->getValue();
    } else {
      Out << "[";
      writeOperand(I.getOperand());
      Out << "]";
    }
}

void CWriter::visitLoadInst(LoadInst &I) {
  Out << "*";
  writeOperand(I.getOperand(0));
}

void CWriter::visitStoreInst(StoreInst &I) {
  Out << "*";
  writeOperand(I.getPointerOperand());
  Out << " = ";
  writeOperand(I.getOperand(0));
}

void CWriter::visitGetElementPtrInst(GetElementPtrInst &I) {
  Out << "&";
  printIndexingExpression(I.getPointerOperand(), gep_type_begin(I),
                          gep_type_end(I));
}

void CWriter::visitVANextInst(VANextInst &I) {
  Out << Mang->getValueName(I.getOperand(0));
  Out << ";  va_arg(*(va_list*)&" << Mang->getValueName(&I) << ", ";
  printType(Out, I.getArgType());
  Out << ")";  
}

void CWriter::visitVAArgInst(VAArgInst &I) {
  Out << "0;\n";
  Out << "{ va_list Tmp; va_copy(Tmp, *(va_list*)&";
  writeOperand(I.getOperand(0));
  Out << ");\n  " << Mang->getValueName(&I) << " = va_arg(Tmp, ";
  printType(Out, I.getType());
  Out << ");\n  va_end(Tmp); }";
}

//===----------------------------------------------------------------------===//
//                       External Interface declaration
//===----------------------------------------------------------------------===//

bool CTargetMachine::addPassesToEmitAssembly(PassManager &PM, std::ostream &o) {
  PM.add(createLowerGCPass());
  PM.add(createLowerAllocationsPass());
  PM.add(createLowerInvokePass());
  PM.add(new CBackendNameAllUsedStructs());
  PM.add(new CWriter(o, getIntrinsicLowering()));
  return false;
}

// vim: sw=2
