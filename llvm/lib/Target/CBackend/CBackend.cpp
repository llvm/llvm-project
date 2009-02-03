//===-- CBackend.cpp - Library for converting LLVM code to C --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library converts LLVM code to C code, compilable by GCC and other C
// compilers.
//
//===----------------------------------------------------------------------===//

#include "CTargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/InlineAsm.h"
#include "llvm/Analysis/ConstantsScanner.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Config/config.h"
#include <algorithm>
#include <sstream>
using namespace llvm;

/// CBackendTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int CBackendTargetMachineModule;
int CBackendTargetMachineModule = 0;

// Register the target.
static RegisterTarget<CTargetMachine> X("c", "C backend");

namespace {
  /// CBackendNameAllUsedStructsAndMergeFunctions - This pass inserts names for
  /// any unnamed structure types that are used by the program, and merges
  /// external functions with the same name.
  ///
  class CBackendNameAllUsedStructsAndMergeFunctions : public ModulePass {
  public:
    static char ID;
    CBackendNameAllUsedStructsAndMergeFunctions() 
      : ModulePass(&ID) {}
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<FindUsedTypes>();
    }

    virtual const char *getPassName() const {
      return "C backend type canonicalizer";
    }

    virtual bool runOnModule(Module &M);
  };

  char CBackendNameAllUsedStructsAndMergeFunctions::ID = 0;

  /// CWriter - This class is the main chunk of code that converts an LLVM
  /// module to a C translation unit.
  class CWriter : public FunctionPass, public InstVisitor<CWriter> {
    raw_ostream &Out;
    IntrinsicLowering *IL;
    Mangler *Mang;
    LoopInfo *LI;
    const Module *TheModule;
    const TargetAsmInfo* TAsm;
    const TargetData* TD;
    std::map<const Type *, std::string> TypeNames;
    std::map<const ConstantFP *, unsigned> FPConstantMap;
    std::set<Function*> intrinsicPrototypesAlreadyGenerated;
    std::set<const Argument*> ByValParams;
    unsigned FPCounter;

  public:
    static char ID;
    explicit CWriter(raw_ostream &o)
      : FunctionPass(&ID), Out(o), IL(0), Mang(0), LI(0), 
        TheModule(0), TAsm(0), TD(0) {
      FPCounter = 0;
    }

    virtual const char *getPassName() const { return "C backend"; }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.setPreservesAll();
    }

    virtual bool doInitialization(Module &M);

    bool runOnFunction(Function &F) {
      LI = &getAnalysis<LoopInfo>();

      // Get rid of intrinsics we can't handle.
      lowerIntrinsics(F);

      // Output all floating point constants that cannot be printed accurately.
      printFloatingPointConstants(F);

      printFunction(F);
      return false;
    }

    virtual bool doFinalization(Module &M) {
      // Free memory...
      delete IL;
      delete TD;
      delete Mang;
      FPConstantMap.clear();
      TypeNames.clear();
      ByValParams.clear();
      intrinsicPrototypesAlreadyGenerated.clear();
      return false;
    }

    raw_ostream &printType(raw_ostream &Out, const Type *Ty, 
                            bool isSigned = false,
                            const std::string &VariableName = "",
                            bool IgnoreName = false,
                            const AttrListPtr &PAL = AttrListPtr());
    std::ostream &printType(std::ostream &Out, const Type *Ty, 
                           bool isSigned = false,
                           const std::string &VariableName = "",
                           bool IgnoreName = false,
                           const AttrListPtr &PAL = AttrListPtr());
    raw_ostream &printSimpleType(raw_ostream &Out, const Type *Ty, 
                                  bool isSigned, 
                                  const std::string &NameSoFar = "");
    std::ostream &printSimpleType(std::ostream &Out, const Type *Ty, 
                                 bool isSigned, 
                                 const std::string &NameSoFar = "");

    void printStructReturnPointerFunctionType(raw_ostream &Out,
                                              const AttrListPtr &PAL,
                                              const PointerType *Ty);

    /// writeOperandDeref - Print the result of dereferencing the specified
    /// operand with '*'.  This is equivalent to printing '*' then using
    /// writeOperand, but avoids excess syntax in some cases.
    void writeOperandDeref(Value *Operand) {
      if (isAddressExposed(Operand)) {
        // Already something with an address exposed.
        writeOperandInternal(Operand);
      } else {
        Out << "*(";
        writeOperand(Operand);
        Out << ")";
      }
    }
    
    void writeOperand(Value *Operand, bool Static = false);
    void writeInstComputationInline(Instruction &I);
    void writeOperandInternal(Value *Operand, bool Static = false);
    void writeOperandWithCast(Value* Operand, unsigned Opcode);
    void writeOperandWithCast(Value* Operand, const ICmpInst &I);
    bool writeInstructionCast(const Instruction &I);

    void writeMemoryAccess(Value *Operand, const Type *OperandType,
                           bool IsVolatile, unsigned Alignment);

  private :
    std::string InterpretASMConstraint(InlineAsm::ConstraintInfo& c);

    void lowerIntrinsics(Function &F);

    void printModule(Module *M);
    void printModuleTypes(const TypeSymbolTable &ST);
    void printContainedStructs(const Type *Ty, std::set<const Type *> &);
    void printFloatingPointConstants(Function &F);
    void printFloatingPointConstants(const Constant *C);
    void printFunctionSignature(const Function *F, bool Prototype);

    void printFunction(Function &);
    void printBasicBlock(BasicBlock *BB);
    void printLoop(Loop *L);

    void printCast(unsigned opcode, const Type *SrcTy, const Type *DstTy);
    void printConstant(Constant *CPV, bool Static);
    void printConstantWithCast(Constant *CPV, unsigned Opcode);
    bool printConstExprCast(const ConstantExpr *CE, bool Static);
    void printConstantArray(ConstantArray *CPA, bool Static);
    void printConstantVector(ConstantVector *CV, bool Static);

    /// isAddressExposed - Return true if the specified value's name needs to
    /// have its address taken in order to get a C value of the correct type.
    /// This happens for global variables, byval parameters, and direct allocas.
    bool isAddressExposed(const Value *V) const {
      if (const Argument *A = dyn_cast<Argument>(V))
        return ByValParams.count(A);
      return isa<GlobalVariable>(V) || isDirectAlloca(V);
    }
    
    // isInlinableInst - Attempt to inline instructions into their uses to build
    // trees as much as possible.  To do this, we have to consistently decide
    // what is acceptable to inline, so that variable declarations don't get
    // printed and an extra copy of the expr is not emitted.
    //
    static bool isInlinableInst(const Instruction &I) {
      // Always inline cmp instructions, even if they are shared by multiple
      // expressions.  GCC generates horrible code if we don't.
      if (isa<CmpInst>(I)) 
        return true;

      // Must be an expression, must be used exactly once.  If it is dead, we
      // emit it inline where it would go.
      if (I.getType() == Type::VoidTy || !I.hasOneUse() ||
          isa<TerminatorInst>(I) || isa<CallInst>(I) || isa<PHINode>(I) ||
          isa<LoadInst>(I) || isa<VAArgInst>(I) || isa<InsertElementInst>(I) ||
          isa<InsertValueInst>(I))
        // Don't inline a load across a store or other bad things!
        return false;

      // Must not be used in inline asm, extractelement, or shufflevector.
      if (I.hasOneUse()) {
        const Instruction &User = cast<Instruction>(*I.use_back());
        if (isInlineAsm(User) || isa<ExtractElementInst>(User) ||
            isa<ShuffleVectorInst>(User))
          return false;
      }

      // Only inline instruction it if it's use is in the same BB as the inst.
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
    
    // isInlineAsm - Check if the instruction is a call to an inline asm chunk
    static bool isInlineAsm(const Instruction& I) {
      if (isa<CallInst>(&I) && isa<InlineAsm>(I.getOperand(0)))
        return true;
      return false;
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
    void visitUnreachableInst(UnreachableInst &I);

    void visitPHINode(PHINode &I);
    void visitBinaryOperator(Instruction &I);
    void visitICmpInst(ICmpInst &I);
    void visitFCmpInst(FCmpInst &I);

    void visitCastInst (CastInst &I);
    void visitSelectInst(SelectInst &I);
    void visitCallInst (CallInst &I);
    void visitInlineAsm(CallInst &I);
    bool visitBuiltinCall(CallInst &I, Intrinsic::ID ID, bool &WroteCallee);

    void visitMallocInst(MallocInst &I);
    void visitAllocaInst(AllocaInst &I);
    void visitFreeInst  (FreeInst   &I);
    void visitLoadInst  (LoadInst   &I);
    void visitStoreInst (StoreInst  &I);
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitVAArgInst (VAArgInst &I);
    
    void visitInsertElementInst(InsertElementInst &I);
    void visitExtractElementInst(ExtractElementInst &I);
    void visitShuffleVectorInst(ShuffleVectorInst &SVI);

    void visitInsertValueInst(InsertValueInst &I);
    void visitExtractValueInst(ExtractValueInst &I);

    void visitInstruction(Instruction &I) {
      cerr << "C Writer does not know about " << I;
      abort();
    }

    void outputLValue(Instruction *I) {
      Out << "  " << GetValueName(I) << " = ";
    }

    bool isGotoCodeNecessary(BasicBlock *From, BasicBlock *To);
    void printPHICopiesForSuccessor(BasicBlock *CurBlock,
                                    BasicBlock *Successor, unsigned Indent);
    void printBranchToBlock(BasicBlock *CurBlock, BasicBlock *SuccBlock,
                            unsigned Indent);
    void printGEPExpression(Value *Ptr, gep_type_iterator I,
                            gep_type_iterator E, bool Static);

    std::string GetValueName(const Value *Operand);
  };
}

char CWriter::ID = 0;

/// This method inserts names for any unnamed structure types that are used by
/// the program, and removes names from structure types that are not used by the
/// program.
///
bool CBackendNameAllUsedStructsAndMergeFunctions::runOnModule(Module &M) {
  // Get a set of types that are used by the program...
  std::set<const Type *> UT = getAnalysis<FindUsedTypes>().getTypes();

  // Loop over the module symbol table, removing types from UT that are
  // already named, and removing names for types that are not used.
  //
  TypeSymbolTable &TST = M.getTypeSymbolTable();
  for (TypeSymbolTable::iterator TI = TST.begin(), TE = TST.end();
       TI != TE; ) {
    TypeSymbolTable::iterator I = TI++;
    
    // If this isn't a struct or array type, remove it from our set of types
    // to name. This simplifies emission later.
    if (!isa<StructType>(I->second) && !isa<OpaqueType>(I->second) &&
        !isa<ArrayType>(I->second)) {
      TST.remove(I);
    } else {
      // If this is not used, remove it from the symbol table.
      std::set<const Type *>::iterator UTI = UT.find(I->second);
      if (UTI == UT.end())
        TST.remove(I);
      else
        UT.erase(UTI);    // Only keep one name for this type.
    }
  }

  // UT now contains types that are not named.  Loop over it, naming
  // structure types.
  //
  bool Changed = false;
  unsigned RenameCounter = 0;
  for (std::set<const Type *>::const_iterator I = UT.begin(), E = UT.end();
       I != E; ++I)
    if (isa<StructType>(*I) || isa<ArrayType>(*I)) {
      while (M.addTypeName("unnamed"+utostr(RenameCounter), *I))
        ++RenameCounter;
      Changed = true;
    }
      
      
  // Loop over all external functions and globals.  If we have two with
  // identical names, merge them.
  // FIXME: This code should disappear when we don't allow values with the same
  // names when they have different types!
  std::map<std::string, GlobalValue*> ExtSymbols;
  for (Module::iterator I = M.begin(), E = M.end(); I != E;) {
    Function *GV = I++;
    if (GV->isDeclaration() && GV->hasName()) {
      std::pair<std::map<std::string, GlobalValue*>::iterator, bool> X
        = ExtSymbols.insert(std::make_pair(GV->getName(), GV));
      if (!X.second) {
        // Found a conflict, replace this global with the previous one.
        GlobalValue *OldGV = X.first->second;
        GV->replaceAllUsesWith(ConstantExpr::getBitCast(OldGV, GV->getType()));
        GV->eraseFromParent();
        Changed = true;
      }
    }
  }
  // Do the same for globals.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E;) {
    GlobalVariable *GV = I++;
    if (GV->isDeclaration() && GV->hasName()) {
      std::pair<std::map<std::string, GlobalValue*>::iterator, bool> X
        = ExtSymbols.insert(std::make_pair(GV->getName(), GV));
      if (!X.second) {
        // Found a conflict, replace this global with the previous one.
        GlobalValue *OldGV = X.first->second;
        GV->replaceAllUsesWith(ConstantExpr::getBitCast(OldGV, GV->getType()));
        GV->eraseFromParent();
        Changed = true;
      }
    }
  }
  
  return Changed;
}

/// printStructReturnPointerFunctionType - This is like printType for a struct
/// return type, except, instead of printing the type as void (*)(Struct*, ...)
/// print it as "Struct (*)(...)", for struct return functions.
void CWriter::printStructReturnPointerFunctionType(raw_ostream &Out,
                                                   const AttrListPtr &PAL,
                                                   const PointerType *TheTy) {
  const FunctionType *FTy = cast<FunctionType>(TheTy->getElementType());
  std::stringstream FunctionInnards;
  FunctionInnards << " (*) (";
  bool PrintedType = false;

  FunctionType::param_iterator I = FTy->param_begin(), E = FTy->param_end();
  const Type *RetTy = cast<PointerType>(I->get())->getElementType();
  unsigned Idx = 1;
  for (++I, ++Idx; I != E; ++I, ++Idx) {
    if (PrintedType)
      FunctionInnards << ", ";
    const Type *ArgTy = *I;
    if (PAL.paramHasAttr(Idx, Attribute::ByVal)) {
      assert(isa<PointerType>(ArgTy));
      ArgTy = cast<PointerType>(ArgTy)->getElementType();
    }
    printType(FunctionInnards, ArgTy,
        /*isSigned=*/PAL.paramHasAttr(Idx, Attribute::SExt), "");
    PrintedType = true;
  }
  if (FTy->isVarArg()) {
    if (PrintedType)
      FunctionInnards << ", ...";
  } else if (!PrintedType) {
    FunctionInnards << "void";
  }
  FunctionInnards << ')';
  std::string tstr = FunctionInnards.str();
  printType(Out, RetTy, 
      /*isSigned=*/PAL.paramHasAttr(0, Attribute::SExt), tstr);
}

raw_ostream &
CWriter::printSimpleType(raw_ostream &Out, const Type *Ty, bool isSigned,
                         const std::string &NameSoFar) {
  assert((Ty->isPrimitiveType() || Ty->isInteger() || isa<VectorType>(Ty)) && 
         "Invalid type for printSimpleType");
  switch (Ty->getTypeID()) {
  case Type::VoidTyID:   return Out << "void " << NameSoFar;
  case Type::IntegerTyID: {
    unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
    if (NumBits == 1) 
      return Out << "bool " << NameSoFar;
    else if (NumBits <= 8)
      return Out << (isSigned?"signed":"unsigned") << " char " << NameSoFar;
    else if (NumBits <= 16)
      return Out << (isSigned?"signed":"unsigned") << " short " << NameSoFar;
    else if (NumBits <= 32)
      return Out << (isSigned?"signed":"unsigned") << " int " << NameSoFar;
    else if (NumBits <= 64)
      return Out << (isSigned?"signed":"unsigned") << " long long "<< NameSoFar;
    else { 
      assert(NumBits <= 128 && "Bit widths > 128 not implemented yet");
      return Out << (isSigned?"llvmInt128":"llvmUInt128") << " " << NameSoFar;
    }
  }
  case Type::FloatTyID:  return Out << "float "   << NameSoFar;
  case Type::DoubleTyID: return Out << "double "  << NameSoFar;
  // Lacking emulation of FP80 on PPC, etc., we assume whichever of these is
  // present matches host 'long double'.
  case Type::X86_FP80TyID:
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:  return Out << "long double " << NameSoFar;
      
  case Type::VectorTyID: {
    const VectorType *VTy = cast<VectorType>(Ty);
    return printSimpleType(Out, VTy->getElementType(), isSigned,
                     " __attribute__((vector_size(" +
                     utostr(TD->getTypePaddedSize(VTy)) + " ))) " + NameSoFar);
  }
    
  default:
    cerr << "Unknown primitive type: " << *Ty << "\n";
    abort();
  }
}

std::ostream &
CWriter::printSimpleType(std::ostream &Out, const Type *Ty, bool isSigned,
                         const std::string &NameSoFar) {
  assert((Ty->isPrimitiveType() || Ty->isInteger() || isa<VectorType>(Ty)) && 
         "Invalid type for printSimpleType");
  switch (Ty->getTypeID()) {
  case Type::VoidTyID:   return Out << "void " << NameSoFar;
  case Type::IntegerTyID: {
    unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
    if (NumBits == 1) 
      return Out << "bool " << NameSoFar;
    else if (NumBits <= 8)
      return Out << (isSigned?"signed":"unsigned") << " char " << NameSoFar;
    else if (NumBits <= 16)
      return Out << (isSigned?"signed":"unsigned") << " short " << NameSoFar;
    else if (NumBits <= 32)
      return Out << (isSigned?"signed":"unsigned") << " int " << NameSoFar;
    else if (NumBits <= 64)
      return Out << (isSigned?"signed":"unsigned") << " long long "<< NameSoFar;
    else { 
      assert(NumBits <= 128 && "Bit widths > 128 not implemented yet");
      return Out << (isSigned?"llvmInt128":"llvmUInt128") << " " << NameSoFar;
    }
  }
  case Type::FloatTyID:  return Out << "float "   << NameSoFar;
  case Type::DoubleTyID: return Out << "double "  << NameSoFar;
  // Lacking emulation of FP80 on PPC, etc., we assume whichever of these is
  // present matches host 'long double'.
  case Type::X86_FP80TyID:
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:  return Out << "long double " << NameSoFar;
      
  case Type::VectorTyID: {
    const VectorType *VTy = cast<VectorType>(Ty);
    return printSimpleType(Out, VTy->getElementType(), isSigned,
                     " __attribute__((vector_size(" +
                     utostr(TD->getTypePaddedSize(VTy)) + " ))) " + NameSoFar);
  }
    
  default:
    cerr << "Unknown primitive type: " << *Ty << "\n";
    abort();
  }
}

// Pass the Type* and the variable name and this prints out the variable
// declaration.
//
raw_ostream &CWriter::printType(raw_ostream &Out, const Type *Ty,
                                 bool isSigned, const std::string &NameSoFar,
                                 bool IgnoreName, const AttrListPtr &PAL) {
  if (Ty->isPrimitiveType() || Ty->isInteger() || isa<VectorType>(Ty)) {
    printSimpleType(Out, Ty, isSigned, NameSoFar);
    return Out;
  }

  // Check to see if the type is named.
  if (!IgnoreName || isa<OpaqueType>(Ty)) {
    std::map<const Type *, std::string>::iterator I = TypeNames.find(Ty);
    if (I != TypeNames.end()) return Out << I->second << ' ' << NameSoFar;
  }

  switch (Ty->getTypeID()) {
  case Type::FunctionTyID: {
    const FunctionType *FTy = cast<FunctionType>(Ty);
    std::stringstream FunctionInnards;
    FunctionInnards << " (" << NameSoFar << ") (";
    unsigned Idx = 1;
    for (FunctionType::param_iterator I = FTy->param_begin(),
           E = FTy->param_end(); I != E; ++I) {
      const Type *ArgTy = *I;
      if (PAL.paramHasAttr(Idx, Attribute::ByVal)) {
        assert(isa<PointerType>(ArgTy));
        ArgTy = cast<PointerType>(ArgTy)->getElementType();
      }
      if (I != FTy->param_begin())
        FunctionInnards << ", ";
      printType(FunctionInnards, ArgTy,
        /*isSigned=*/PAL.paramHasAttr(Idx, Attribute::SExt), "");
      ++Idx;
    }
    if (FTy->isVarArg()) {
      if (FTy->getNumParams())
        FunctionInnards << ", ...";
    } else if (!FTy->getNumParams()) {
      FunctionInnards << "void";
    }
    FunctionInnards << ')';
    std::string tstr = FunctionInnards.str();
    printType(Out, FTy->getReturnType(), 
      /*isSigned=*/PAL.paramHasAttr(0, Attribute::SExt), tstr);
    return Out;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<StructType>(Ty);
    Out << NameSoFar + " {\n";
    unsigned Idx = 0;
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      Out << "  ";
      printType(Out, *I, false, "field" + utostr(Idx++));
      Out << ";\n";
    }
    Out << '}';
    if (STy->isPacked())
      Out << " __attribute__ ((packed))";
    return Out;
  }

  case Type::PointerTyID: {
    const PointerType *PTy = cast<PointerType>(Ty);
    std::string ptrName = "*" + NameSoFar;

    if (isa<ArrayType>(PTy->getElementType()) ||
        isa<VectorType>(PTy->getElementType()))
      ptrName = "(" + ptrName + ")";

    if (!PAL.isEmpty())
      // Must be a function ptr cast!
      return printType(Out, PTy->getElementType(), false, ptrName, true, PAL);
    return printType(Out, PTy->getElementType(), false, ptrName);
  }

  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    unsigned NumElements = ATy->getNumElements();
    if (NumElements == 0) NumElements = 1;
    // Arrays are wrapped in structs to allow them to have normal
    // value semantics (avoiding the array "decay").
    Out << NameSoFar << " { ";
    printType(Out, ATy->getElementType(), false,
              "array[" + utostr(NumElements) + "]");
    return Out << "; }";
  }

  case Type::OpaqueTyID: {
    static int Count = 0;
    std::string TyName = "struct opaque_" + itostr(Count++);
    assert(TypeNames.find(Ty) == TypeNames.end());
    TypeNames[Ty] = TyName;
    return Out << TyName << ' ' << NameSoFar;
  }
  default:
    assert(0 && "Unhandled case in getTypeProps!");
    abort();
  }

  return Out;
}

// Pass the Type* and the variable name and this prints out the variable
// declaration.
//
std::ostream &CWriter::printType(std::ostream &Out, const Type *Ty,
                                 bool isSigned, const std::string &NameSoFar,
                                 bool IgnoreName, const AttrListPtr &PAL) {
  if (Ty->isPrimitiveType() || Ty->isInteger() || isa<VectorType>(Ty)) {
    printSimpleType(Out, Ty, isSigned, NameSoFar);
    return Out;
  }

  // Check to see if the type is named.
  if (!IgnoreName || isa<OpaqueType>(Ty)) {
    std::map<const Type *, std::string>::iterator I = TypeNames.find(Ty);
    if (I != TypeNames.end()) return Out << I->second << ' ' << NameSoFar;
  }

  switch (Ty->getTypeID()) {
  case Type::FunctionTyID: {
    const FunctionType *FTy = cast<FunctionType>(Ty);
    std::stringstream FunctionInnards;
    FunctionInnards << " (" << NameSoFar << ") (";
    unsigned Idx = 1;
    for (FunctionType::param_iterator I = FTy->param_begin(),
           E = FTy->param_end(); I != E; ++I) {
      const Type *ArgTy = *I;
      if (PAL.paramHasAttr(Idx, Attribute::ByVal)) {
        assert(isa<PointerType>(ArgTy));
        ArgTy = cast<PointerType>(ArgTy)->getElementType();
      }
      if (I != FTy->param_begin())
        FunctionInnards << ", ";
      printType(FunctionInnards, ArgTy,
        /*isSigned=*/PAL.paramHasAttr(Idx, Attribute::SExt), "");
      ++Idx;
    }
    if (FTy->isVarArg()) {
      if (FTy->getNumParams())
        FunctionInnards << ", ...";
    } else if (!FTy->getNumParams()) {
      FunctionInnards << "void";
    }
    FunctionInnards << ')';
    std::string tstr = FunctionInnards.str();
    printType(Out, FTy->getReturnType(), 
      /*isSigned=*/PAL.paramHasAttr(0, Attribute::SExt), tstr);
    return Out;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<StructType>(Ty);
    Out << NameSoFar + " {\n";
    unsigned Idx = 0;
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      Out << "  ";
      printType(Out, *I, false, "field" + utostr(Idx++));
      Out << ";\n";
    }
    Out << '}';
    if (STy->isPacked())
      Out << " __attribute__ ((packed))";
    return Out;
  }

  case Type::PointerTyID: {
    const PointerType *PTy = cast<PointerType>(Ty);
    std::string ptrName = "*" + NameSoFar;

    if (isa<ArrayType>(PTy->getElementType()) ||
        isa<VectorType>(PTy->getElementType()))
      ptrName = "(" + ptrName + ")";

    if (!PAL.isEmpty())
      // Must be a function ptr cast!
      return printType(Out, PTy->getElementType(), false, ptrName, true, PAL);
    return printType(Out, PTy->getElementType(), false, ptrName);
  }

  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    unsigned NumElements = ATy->getNumElements();
    if (NumElements == 0) NumElements = 1;
    // Arrays are wrapped in structs to allow them to have normal
    // value semantics (avoiding the array "decay").
    Out << NameSoFar << " { ";
    printType(Out, ATy->getElementType(), false,
              "array[" + utostr(NumElements) + "]");
    return Out << "; }";
  }

  case Type::OpaqueTyID: {
    static int Count = 0;
    std::string TyName = "struct opaque_" + itostr(Count++);
    assert(TypeNames.find(Ty) == TypeNames.end());
    TypeNames[Ty] = TyName;
    return Out << TyName << ' ' << NameSoFar;
  }
  default:
    assert(0 && "Unhandled case in getTypeProps!");
    abort();
  }

  return Out;
}

void CWriter::printConstantArray(ConstantArray *CPA, bool Static) {

  // As a special case, print the array as a string if it is an array of
  // ubytes or an array of sbytes with positive values.
  //
  const Type *ETy = CPA->getType()->getElementType();
  bool isString = (ETy == Type::Int8Ty || ETy == Type::Int8Ty);

  // Make sure the last character is a null char, as automatically added by C
  if (isString && (CPA->getNumOperands() == 0 ||
                   !cast<Constant>(*(CPA->op_end()-1))->isNullValue()))
    isString = false;

  if (isString) {
    Out << '\"';
    // Keep track of whether the last number was a hexadecimal escape
    bool LastWasHex = false;

    // Do not include the last character, which we know is null
    for (unsigned i = 0, e = CPA->getNumOperands()-1; i != e; ++i) {
      unsigned char C = cast<ConstantInt>(CPA->getOperand(i))->getZExtValue();

      // Print it out literally if it is a printable character.  The only thing
      // to be careful about is when the last letter output was a hex escape
      // code, in which case we have to be careful not to print out hex digits
      // explicitly (the C compiler thinks it is a continuation of the previous
      // character, sheesh...)
      //
      if (isprint(C) && (!LastWasHex || !isxdigit(C))) {
        LastWasHex = false;
        if (C == '"' || C == '\\')
          Out << "\\" << (char)C;
        else
          Out << (char)C;
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
    Out << '\"';
  } else {
    Out << '{';
    if (CPA->getNumOperands()) {
      Out << ' ';
      printConstant(cast<Constant>(CPA->getOperand(0)), Static);
      for (unsigned i = 1, e = CPA->getNumOperands(); i != e; ++i) {
        Out << ", ";
        printConstant(cast<Constant>(CPA->getOperand(i)), Static);
      }
    }
    Out << " }";
  }
}

void CWriter::printConstantVector(ConstantVector *CP, bool Static) {
  Out << '{';
  if (CP->getNumOperands()) {
    Out << ' ';
    printConstant(cast<Constant>(CP->getOperand(0)), Static);
    for (unsigned i = 1, e = CP->getNumOperands(); i != e; ++i) {
      Out << ", ";
      printConstant(cast<Constant>(CP->getOperand(i)), Static);
    }
  }
  Out << " }";
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
  bool ignored;
  // Do long doubles in hex for now.
  if (CFP->getType() != Type::FloatTy && CFP->getType() != Type::DoubleTy)
    return false;
  APFloat APF = APFloat(CFP->getValueAPF());  // copy
  if (CFP->getType() == Type::FloatTy)
    APF.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &ignored);
#if HAVE_PRINTF_A && ENABLE_CBE_PRINTF_A
  char Buffer[100];
  sprintf(Buffer, "%a", APF.convertToDouble());
  if (!strncmp(Buffer, "0x", 2) ||
      !strncmp(Buffer, "-0x", 3) ||
      !strncmp(Buffer, "+0x", 3))
    return APF.bitwiseIsEqual(APFloat(atof(Buffer)));
  return false;
#else
  std::string StrVal = ftostr(APF);

  while (StrVal[0] == ' ')
    StrVal.erase(StrVal.begin());

  // Check to make sure that the stringized number is not some string like "Inf"
  // or NaN.  Check that the string matches the "[-+]?[0-9]" regex.
  if ((StrVal[0] >= '0' && StrVal[0] <= '9') ||
      ((StrVal[0] == '-' || StrVal[0] == '+') &&
       (StrVal[1] >= '0' && StrVal[1] <= '9')))
    // Reparse stringized version!
    return APF.bitwiseIsEqual(APFloat(atof(StrVal.c_str())));
  return false;
#endif
}

/// Print out the casting for a cast operation. This does the double casting
/// necessary for conversion to the destination type, if necessary. 
/// @brief Print a cast
void CWriter::printCast(unsigned opc, const Type *SrcTy, const Type *DstTy) {
  // Print the destination type cast
  switch (opc) {
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::IntToPtr:
    case Instruction::Trunc:
    case Instruction::BitCast:
    case Instruction::FPExt:
    case Instruction::FPTrunc: // For these the DstTy sign doesn't matter
      Out << '(';
      printType(Out, DstTy);
      Out << ')';
      break;
    case Instruction::ZExt:
    case Instruction::PtrToInt:
    case Instruction::FPToUI: // For these, make sure we get an unsigned dest
      Out << '(';
      printSimpleType(Out, DstTy, false);
      Out << ')';
      break;
    case Instruction::SExt: 
    case Instruction::FPToSI: // For these, make sure we get a signed dest
      Out << '(';
      printSimpleType(Out, DstTy, true);
      Out << ')';
      break;
    default:
      assert(0 && "Invalid cast opcode");
  }

  // Print the source type cast
  switch (opc) {
    case Instruction::UIToFP:
    case Instruction::ZExt:
      Out << '(';
      printSimpleType(Out, SrcTy, false);
      Out << ')';
      break;
    case Instruction::SIToFP:
    case Instruction::SExt:
      Out << '(';
      printSimpleType(Out, SrcTy, true); 
      Out << ')';
      break;
    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
      // Avoid "cast to pointer from integer of different size" warnings
      Out << "(unsigned long)";
      break;
    case Instruction::Trunc:
    case Instruction::BitCast:
    case Instruction::FPExt:
    case Instruction::FPTrunc:
    case Instruction::FPToSI:
    case Instruction::FPToUI:
      break; // These don't need a source cast.
    default:
      assert(0 && "Invalid cast opcode");
      break;
  }
}

// printConstant - The LLVM Constant to C Constant converter.
void CWriter::printConstant(Constant *CPV, bool Static) {
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CPV)) {
    switch (CE->getOpcode()) {
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::BitCast:
      Out << "(";
      printCast(CE->getOpcode(), CE->getOperand(0)->getType(), CE->getType());
      if (CE->getOpcode() == Instruction::SExt &&
          CE->getOperand(0)->getType() == Type::Int1Ty) {
        // Make sure we really sext from bool here by subtracting from 0
        Out << "0-";
      }
      printConstant(CE->getOperand(0), Static);
      if (CE->getType() == Type::Int1Ty &&
          (CE->getOpcode() == Instruction::Trunc ||
           CE->getOpcode() == Instruction::FPToUI ||
           CE->getOpcode() == Instruction::FPToSI ||
           CE->getOpcode() == Instruction::PtrToInt)) {
        // Make sure we really truncate to bool here by anding with 1
        Out << "&1u";
      }
      Out << ')';
      return;

    case Instruction::GetElementPtr:
      Out << "(";
      printGEPExpression(CE->getOperand(0), gep_type_begin(CPV),
                         gep_type_end(CPV), Static);
      Out << ")";
      return;
    case Instruction::Select:
      Out << '(';
      printConstant(CE->getOperand(0), Static);
      Out << '?';
      printConstant(CE->getOperand(1), Static);
      Out << ':';
      printConstant(CE->getOperand(2), Static);
      Out << ')';
      return;
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::SDiv:
    case Instruction::UDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::ICmp:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    {
      Out << '(';
      bool NeedsClosingParens = printConstExprCast(CE, Static); 
      printConstantWithCast(CE->getOperand(0), CE->getOpcode());
      switch (CE->getOpcode()) {
      case Instruction::Add: Out << " + "; break;
      case Instruction::Sub: Out << " - "; break;
      case Instruction::Mul: Out << " * "; break;
      case Instruction::URem:
      case Instruction::SRem: 
      case Instruction::FRem: Out << " % "; break;
      case Instruction::UDiv: 
      case Instruction::SDiv: 
      case Instruction::FDiv: Out << " / "; break;
      case Instruction::And: Out << " & "; break;
      case Instruction::Or:  Out << " | "; break;
      case Instruction::Xor: Out << " ^ "; break;
      case Instruction::Shl: Out << " << "; break;
      case Instruction::LShr:
      case Instruction::AShr: Out << " >> "; break;
      case Instruction::ICmp:
        switch (CE->getPredicate()) {
          case ICmpInst::ICMP_EQ: Out << " == "; break;
          case ICmpInst::ICMP_NE: Out << " != "; break;
          case ICmpInst::ICMP_SLT: 
          case ICmpInst::ICMP_ULT: Out << " < "; break;
          case ICmpInst::ICMP_SLE:
          case ICmpInst::ICMP_ULE: Out << " <= "; break;
          case ICmpInst::ICMP_SGT:
          case ICmpInst::ICMP_UGT: Out << " > "; break;
          case ICmpInst::ICMP_SGE:
          case ICmpInst::ICMP_UGE: Out << " >= "; break;
          default: assert(0 && "Illegal ICmp predicate");
        }
        break;
      default: assert(0 && "Illegal opcode here!");
      }
      printConstantWithCast(CE->getOperand(1), CE->getOpcode());
      if (NeedsClosingParens)
        Out << "))";
      Out << ')';
      return;
    }
    case Instruction::FCmp: {
      Out << '('; 
      bool NeedsClosingParens = printConstExprCast(CE, Static); 
      if (CE->getPredicate() == FCmpInst::FCMP_FALSE)
        Out << "0";
      else if (CE->getPredicate() == FCmpInst::FCMP_TRUE)
        Out << "1";
      else {
        const char* op = 0;
        switch (CE->getPredicate()) {
        default: assert(0 && "Illegal FCmp predicate");
        case FCmpInst::FCMP_ORD: op = "ord"; break;
        case FCmpInst::FCMP_UNO: op = "uno"; break;
        case FCmpInst::FCMP_UEQ: op = "ueq"; break;
        case FCmpInst::FCMP_UNE: op = "une"; break;
        case FCmpInst::FCMP_ULT: op = "ult"; break;
        case FCmpInst::FCMP_ULE: op = "ule"; break;
        case FCmpInst::FCMP_UGT: op = "ugt"; break;
        case FCmpInst::FCMP_UGE: op = "uge"; break;
        case FCmpInst::FCMP_OEQ: op = "oeq"; break;
        case FCmpInst::FCMP_ONE: op = "one"; break;
        case FCmpInst::FCMP_OLT: op = "olt"; break;
        case FCmpInst::FCMP_OLE: op = "ole"; break;
        case FCmpInst::FCMP_OGT: op = "ogt"; break;
        case FCmpInst::FCMP_OGE: op = "oge"; break;
        }
        Out << "llvm_fcmp_" << op << "(";
        printConstantWithCast(CE->getOperand(0), CE->getOpcode());
        Out << ", ";
        printConstantWithCast(CE->getOperand(1), CE->getOpcode());
        Out << ")";
      }
      if (NeedsClosingParens)
        Out << "))";
      Out << ')';
      return;
    }
    default:
      cerr << "CWriter Error: Unhandled constant expression: "
           << *CE << "\n";
      abort();
    }
  } else if (isa<UndefValue>(CPV) && CPV->getType()->isSingleValueType()) {
    Out << "((";
    printType(Out, CPV->getType()); // sign doesn't matter
    Out << ")/*UNDEF*/";
    if (!isa<VectorType>(CPV->getType())) {
      Out << "0)";
    } else {
      Out << "{})";
    }
    return;
  }

  if (ConstantInt *CI = dyn_cast<ConstantInt>(CPV)) {
    const Type* Ty = CI->getType();
    if (Ty == Type::Int1Ty)
      Out << (CI->getZExtValue() ? '1' : '0');
    else if (Ty == Type::Int32Ty)
      Out << CI->getZExtValue() << 'u';
    else if (Ty->getPrimitiveSizeInBits() > 32)
      Out << CI->getZExtValue() << "ull";
    else {
      Out << "((";
      printSimpleType(Out, Ty, false) << ')';
      if (CI->isMinValue(true)) 
        Out << CI->getZExtValue() << 'u';
      else
        Out << CI->getSExtValue();
       Out << ')';
    }
    return;
  } 

  switch (CPV->getType()->getTypeID()) {
  case Type::FloatTyID:
  case Type::DoubleTyID: 
  case Type::X86_FP80TyID:
  case Type::PPC_FP128TyID:
  case Type::FP128TyID: {
    ConstantFP *FPC = cast<ConstantFP>(CPV);
    std::map<const ConstantFP*, unsigned>::iterator I = FPConstantMap.find(FPC);
    if (I != FPConstantMap.end()) {
      // Because of FP precision problems we must load from a stack allocated
      // value that holds the value in hex.
      Out << "(*(" << (FPC->getType() == Type::FloatTy ? "float" : 
                       FPC->getType() == Type::DoubleTy ? "double" :
                       "long double")
          << "*)&FPConstant" << I->second << ')';
    } else {
      double V;
      if (FPC->getType() == Type::FloatTy)
        V = FPC->getValueAPF().convertToFloat();
      else if (FPC->getType() == Type::DoubleTy)
        V = FPC->getValueAPF().convertToDouble();
      else {
        // Long double.  Convert the number to double, discarding precision.
        // This is not awesome, but it at least makes the CBE output somewhat
        // useful.
        APFloat Tmp = FPC->getValueAPF();
        bool LosesInfo;
        Tmp.convert(APFloat::IEEEdouble, APFloat::rmTowardZero, &LosesInfo);
        V = Tmp.convertToDouble();
      }
      
      if (IsNAN(V)) {
        // The value is NaN

        // FIXME the actual NaN bits should be emitted.
        // The prefix for a quiet NaN is 0x7FF8. For a signalling NaN,
        // it's 0x7ff4.
        const unsigned long QuietNaN = 0x7ff8UL;
        //const unsigned long SignalNaN = 0x7ff4UL;

        // We need to grab the first part of the FP #
        char Buffer[100];

        uint64_t ll = DoubleToBits(V);
        sprintf(Buffer, "0x%llx", static_cast<long long>(ll));

        std::string Num(&Buffer[0], &Buffer[6]);
        unsigned long Val = strtoul(Num.c_str(), 0, 16);

        if (FPC->getType() == Type::FloatTy)
          Out << "LLVM_NAN" << (Val == QuietNaN ? "" : "S") << "F(\""
              << Buffer << "\") /*nan*/ ";
        else
          Out << "LLVM_NAN" << (Val == QuietNaN ? "" : "S") << "(\""
              << Buffer << "\") /*nan*/ ";
      } else if (IsInf(V)) {
        // The value is Inf
        if (V < 0) Out << '-';
        Out << "LLVM_INF" << (FPC->getType() == Type::FloatTy ? "F" : "")
            << " /*inf*/ ";
      } else {
        std::string Num;
#if HAVE_PRINTF_A && ENABLE_CBE_PRINTF_A
        // Print out the constant as a floating point number.
        char Buffer[100];
        sprintf(Buffer, "%a", V);
        Num = Buffer;
#else
        Num = ftostr(FPC->getValueAPF());
#endif
       Out << Num;
      }
    }
    break;
  }

  case Type::ArrayTyID:
    // Use C99 compound expression literal initializer syntax.
    if (!Static) {
      Out << "(";
      printType(Out, CPV->getType());
      Out << ")";
    }
    Out << "{ "; // Arrays are wrapped in struct types.
    if (ConstantArray *CA = dyn_cast<ConstantArray>(CPV)) {
      printConstantArray(CA, Static);
    } else {
      assert(isa<ConstantAggregateZero>(CPV) || isa<UndefValue>(CPV));
      const ArrayType *AT = cast<ArrayType>(CPV->getType());
      Out << '{';
      if (AT->getNumElements()) {
        Out << ' ';
        Constant *CZ = Constant::getNullValue(AT->getElementType());
        printConstant(CZ, Static);
        for (unsigned i = 1, e = AT->getNumElements(); i != e; ++i) {
          Out << ", ";
          printConstant(CZ, Static);
        }
      }
      Out << " }";
    }
    Out << " }"; // Arrays are wrapped in struct types.
    break;

  case Type::VectorTyID:
    // Use C99 compound expression literal initializer syntax.
    if (!Static) {
      Out << "(";
      printType(Out, CPV->getType());
      Out << ")";
    }
    if (ConstantVector *CV = dyn_cast<ConstantVector>(CPV)) {
      printConstantVector(CV, Static);
    } else {
      assert(isa<ConstantAggregateZero>(CPV) || isa<UndefValue>(CPV));
      const VectorType *VT = cast<VectorType>(CPV->getType());
      Out << "{ ";
      Constant *CZ = Constant::getNullValue(VT->getElementType());
      printConstant(CZ, Static);
      for (unsigned i = 1, e = VT->getNumElements(); i != e; ++i) {
        Out << ", ";
        printConstant(CZ, Static);
      }
      Out << " }";
    }
    break;

  case Type::StructTyID:
    // Use C99 compound expression literal initializer syntax.
    if (!Static) {
      Out << "(";
      printType(Out, CPV->getType());
      Out << ")";
    }
    if (isa<ConstantAggregateZero>(CPV) || isa<UndefValue>(CPV)) {
      const StructType *ST = cast<StructType>(CPV->getType());
      Out << '{';
      if (ST->getNumElements()) {
        Out << ' ';
        printConstant(Constant::getNullValue(ST->getElementType(0)), Static);
        for (unsigned i = 1, e = ST->getNumElements(); i != e; ++i) {
          Out << ", ";
          printConstant(Constant::getNullValue(ST->getElementType(i)), Static);
        }
      }
      Out << " }";
    } else {
      Out << '{';
      if (CPV->getNumOperands()) {
        Out << ' ';
        printConstant(cast<Constant>(CPV->getOperand(0)), Static);
        for (unsigned i = 1, e = CPV->getNumOperands(); i != e; ++i) {
          Out << ", ";
          printConstant(cast<Constant>(CPV->getOperand(i)), Static);
        }
      }
      Out << " }";
    }
    break;

  case Type::PointerTyID:
    if (isa<ConstantPointerNull>(CPV)) {
      Out << "((";
      printType(Out, CPV->getType()); // sign doesn't matter
      Out << ")/*NULL*/0)";
      break;
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(CPV)) {
      writeOperand(GV, Static);
      break;
    }
    // FALL THROUGH
  default:
    cerr << "Unknown constant type: " << *CPV << "\n";
    abort();
  }
}

// Some constant expressions need to be casted back to the original types
// because their operands were casted to the expected type. This function takes
// care of detecting that case and printing the cast for the ConstantExpr.
bool CWriter::printConstExprCast(const ConstantExpr* CE, bool Static) {
  bool NeedsExplicitCast = false;
  const Type *Ty = CE->getOperand(0)->getType();
  bool TypeIsSigned = false;
  switch (CE->getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    // We need to cast integer arithmetic so that it is always performed
    // as unsigned, to avoid undefined behavior on overflow.
    if (!Ty->isIntOrIntVector()) break;
    // FALL THROUGH
  case Instruction::LShr:
  case Instruction::URem: 
  case Instruction::UDiv: NeedsExplicitCast = true; break;
  case Instruction::AShr:
  case Instruction::SRem: 
  case Instruction::SDiv: NeedsExplicitCast = true; TypeIsSigned = true; break;
  case Instruction::SExt:
    Ty = CE->getType();
    NeedsExplicitCast = true;
    TypeIsSigned = true;
    break;
  case Instruction::ZExt:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    Ty = CE->getType();
    NeedsExplicitCast = true;
    break;
  default: break;
  }
  if (NeedsExplicitCast) {
    Out << "((";
    if (Ty->isInteger() && Ty != Type::Int1Ty)
      printSimpleType(Out, Ty, TypeIsSigned);
    else
      printType(Out, Ty); // not integer, sign doesn't matter
    Out << ")(";
  }
  return NeedsExplicitCast;
}

//  Print a constant assuming that it is the operand for a given Opcode. The
//  opcodes that care about sign need to cast their operands to the expected
//  type before the operation proceeds. This function does the casting.
void CWriter::printConstantWithCast(Constant* CPV, unsigned Opcode) {

  // Extract the operand's type, we'll need it.
  const Type* OpTy = CPV->getType();

  // Indicate whether to do the cast or not.
  bool shouldCast = false;
  bool typeIsSigned = false;

  // Based on the Opcode for which this Constant is being written, determine
  // the new type to which the operand should be casted by setting the value
  // of OpTy. If we change OpTy, also set shouldCast to true so it gets
  // casted below.
  switch (Opcode) {
    default:
      // for most instructions, it doesn't matter
      break; 
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
      // We need to cast integer arithmetic so that it is always performed
      // as unsigned, to avoid undefined behavior on overflow.
      if (!OpTy->isIntOrIntVector()) break;
      // FALL THROUGH
    case Instruction::LShr:
    case Instruction::UDiv:
    case Instruction::URem:
      shouldCast = true;
      break;
    case Instruction::AShr:
    case Instruction::SDiv:
    case Instruction::SRem:
      shouldCast = true;
      typeIsSigned = true;
      break;
  }

  // Write out the casted constant if we should, otherwise just write the
  // operand.
  if (shouldCast) {
    Out << "((";
    printSimpleType(Out, OpTy, typeIsSigned);
    Out << ")";
    printConstant(CPV, false);
    Out << ")";
  } else 
    printConstant(CPV, false);
}

std::string CWriter::GetValueName(const Value *Operand) {
  std::string Name;

  if (!isa<GlobalValue>(Operand) && Operand->getName() != "") {
    std::string VarName;

    Name = Operand->getName();
    VarName.reserve(Name.capacity());

    for (std::string::iterator I = Name.begin(), E = Name.end();
         I != E; ++I) {
      char ch = *I;

      if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') || ch == '_')) {
        char buffer[5];
        sprintf(buffer, "_%x_", ch);
        VarName += buffer;
      } else
        VarName += ch;
    }

    Name = "llvm_cbe_" + VarName;
  } else {
    Name = Mang->getValueName(Operand);
  }

  return Name;
}

/// writeInstComputationInline - Emit the computation for the specified
/// instruction inline, with no destination provided.
void CWriter::writeInstComputationInline(Instruction &I) {
  // If this is a non-trivial bool computation, make sure to truncate down to
  // a 1 bit value.  This is important because we want "add i1 x, y" to return
  // "0" when x and y are true, not "2" for example.
  bool NeedBoolTrunc = false;
  if (I.getType() == Type::Int1Ty && !isa<ICmpInst>(I) && !isa<FCmpInst>(I))
    NeedBoolTrunc = true;
  
  if (NeedBoolTrunc)
    Out << "((";
  
  visit(I);
  
  if (NeedBoolTrunc)
    Out << ")&1)";
}


void CWriter::writeOperandInternal(Value *Operand, bool Static) {
  if (Instruction *I = dyn_cast<Instruction>(Operand))
    // Should we inline this instruction to build a tree?
    if (isInlinableInst(*I) && !isDirectAlloca(I)) {
      Out << '(';
      writeInstComputationInline(*I);
      Out << ')';
      return;
    }

  Constant* CPV = dyn_cast<Constant>(Operand);

  if (CPV && !isa<GlobalValue>(CPV))
    printConstant(CPV, Static);
  else
    Out << GetValueName(Operand);
}

void CWriter::writeOperand(Value *Operand, bool Static) {
  bool isAddressImplicit = isAddressExposed(Operand);
  if (isAddressImplicit)
    Out << "(&";  // Global variables are referenced as their addresses by llvm

  writeOperandInternal(Operand, Static);

  if (isAddressImplicit)
    Out << ')';
}

// Some instructions need to have their result value casted back to the 
// original types because their operands were casted to the expected type. 
// This function takes care of detecting that case and printing the cast 
// for the Instruction.
bool CWriter::writeInstructionCast(const Instruction &I) {
  const Type *Ty = I.getOperand(0)->getType();
  switch (I.getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    // We need to cast integer arithmetic so that it is always performed
    // as unsigned, to avoid undefined behavior on overflow.
    if (!Ty->isIntOrIntVector()) break;
    // FALL THROUGH
  case Instruction::LShr:
  case Instruction::URem: 
  case Instruction::UDiv: 
    Out << "((";
    printSimpleType(Out, Ty, false);
    Out << ")(";
    return true;
  case Instruction::AShr:
  case Instruction::SRem: 
  case Instruction::SDiv: 
    Out << "((";
    printSimpleType(Out, Ty, true);
    Out << ")(";
    return true;
  default: break;
  }
  return false;
}

// Write the operand with a cast to another type based on the Opcode being used.
// This will be used in cases where an instruction has specific type
// requirements (usually signedness) for its operands. 
void CWriter::writeOperandWithCast(Value* Operand, unsigned Opcode) {

  // Extract the operand's type, we'll need it.
  const Type* OpTy = Operand->getType();

  // Indicate whether to do the cast or not.
  bool shouldCast = false;

  // Indicate whether the cast should be to a signed type or not.
  bool castIsSigned = false;

  // Based on the Opcode for which this Operand is being written, determine
  // the new type to which the operand should be casted by setting the value
  // of OpTy. If we change OpTy, also set shouldCast to true.
  switch (Opcode) {
    default:
      // for most instructions, it doesn't matter
      break; 
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
      // We need to cast integer arithmetic so that it is always performed
      // as unsigned, to avoid undefined behavior on overflow.
      if (!OpTy->isIntOrIntVector()) break;
      // FALL THROUGH
    case Instruction::LShr:
    case Instruction::UDiv:
    case Instruction::URem: // Cast to unsigned first
      shouldCast = true;
      castIsSigned = false;
      break;
    case Instruction::GetElementPtr:
    case Instruction::AShr:
    case Instruction::SDiv:
    case Instruction::SRem: // Cast to signed first
      shouldCast = true;
      castIsSigned = true;
      break;
  }

  // Write out the casted operand if we should, otherwise just write the
  // operand.
  if (shouldCast) {
    Out << "((";
    printSimpleType(Out, OpTy, castIsSigned);
    Out << ")";
    writeOperand(Operand);
    Out << ")";
  } else 
    writeOperand(Operand);
}

// Write the operand with a cast to another type based on the icmp predicate 
// being used. 
void CWriter::writeOperandWithCast(Value* Operand, const ICmpInst &Cmp) {
  // This has to do a cast to ensure the operand has the right signedness. 
  // Also, if the operand is a pointer, we make sure to cast to an integer when
  // doing the comparison both for signedness and so that the C compiler doesn't
  // optimize things like "p < NULL" to false (p may contain an integer value
  // f.e.).
  bool shouldCast = Cmp.isRelational();

  // Write out the casted operand if we should, otherwise just write the
  // operand.
  if (!shouldCast) {
    writeOperand(Operand);
    return;
  }
  
  // Should this be a signed comparison?  If so, convert to signed.
  bool castIsSigned = Cmp.isSignedPredicate();

  // If the operand was a pointer, convert to a large integer type.
  const Type* OpTy = Operand->getType();
  if (isa<PointerType>(OpTy))
    OpTy = TD->getIntPtrType();
  
  Out << "((";
  printSimpleType(Out, OpTy, castIsSigned);
  Out << ")";
  writeOperand(Operand);
  Out << ")";
}

// generateCompilerSpecificCode - This is where we add conditional compilation
// directives to cater to specific compilers as need be.
//
static void generateCompilerSpecificCode(raw_ostream& Out,
                                         const TargetData *TD) {
  // Alloca is hard to get, and we don't want to include stdlib.h here.
  Out << "/* get a declaration for alloca */\n"
      << "#if defined(__CYGWIN__) || defined(__MINGW32__)\n"
      << "#define  alloca(x) __builtin_alloca((x))\n"
      << "#define _alloca(x) __builtin_alloca((x))\n"    
      << "#elif defined(__APPLE__)\n"
      << "extern void *__builtin_alloca(unsigned long);\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#define longjmp _longjmp\n"
      << "#define setjmp _setjmp\n"
      << "#elif defined(__sun__)\n"
      << "#if defined(__sparcv9)\n"
      << "extern void *__builtin_alloca(unsigned long);\n"
      << "#else\n"
      << "extern void *__builtin_alloca(unsigned int);\n"
      << "#endif\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#elif defined(_MSC_VER)\n"
      << "#define inline _inline\n"
      << "#define alloca(x) _alloca(x)\n"
      << "#else\n"
      << "#include <alloca.h>\n"
      << "#endif\n\n";

  // We output GCC specific attributes to preserve 'linkonce'ness on globals.
  // If we aren't being compiled with GCC, just drop these attributes.
  Out << "#ifndef __GNUC__  /* Can only support \"linkonce\" vars with GCC */\n"
      << "#define __attribute__(X)\n"
      << "#endif\n\n";

  // On Mac OS X, "external weak" is spelled "__attribute__((weak_import))".
  Out << "#if defined(__GNUC__) && defined(__APPLE_CC__)\n"
      << "#define __EXTERNAL_WEAK__ __attribute__((weak_import))\n"
      << "#elif defined(__GNUC__)\n"
      << "#define __EXTERNAL_WEAK__ __attribute__((weak))\n"
      << "#else\n"
      << "#define __EXTERNAL_WEAK__\n"
      << "#endif\n\n";

  // For now, turn off the weak linkage attribute on Mac OS X. (See above.)
  Out << "#if defined(__GNUC__) && defined(__APPLE_CC__)\n"
      << "#define __ATTRIBUTE_WEAK__\n"
      << "#elif defined(__GNUC__)\n"
      << "#define __ATTRIBUTE_WEAK__ __attribute__((weak))\n"
      << "#else\n"
      << "#define __ATTRIBUTE_WEAK__\n"
      << "#endif\n\n";

  // Add hidden visibility support. FIXME: APPLE_CC?
  Out << "#if defined(__GNUC__)\n"
      << "#define __HIDDEN__ __attribute__((visibility(\"hidden\")))\n"
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
      << "#define LLVM_NAN(NanStr)   __builtin_nan(NanStr)   /* Double */\n"
      << "#define LLVM_NANF(NanStr)  __builtin_nanf(NanStr)  /* Float */\n"
      << "#define LLVM_NANS(NanStr)  __builtin_nans(NanStr)  /* Double */\n"
      << "#define LLVM_NANSF(NanStr) __builtin_nansf(NanStr) /* Float */\n"
      << "#define LLVM_INF           __builtin_inf()         /* Double */\n"
      << "#define LLVM_INFF          __builtin_inff()        /* Float */\n"
      << "#define LLVM_PREFETCH(addr,rw,locality) "
                              "__builtin_prefetch(addr,rw,locality)\n"
      << "#define __ATTRIBUTE_CTOR__ __attribute__((constructor))\n"
      << "#define __ATTRIBUTE_DTOR__ __attribute__((destructor))\n"
      << "#define LLVM_ASM           __asm__\n"
      << "#else\n"
      << "#define LLVM_NAN(NanStr)   ((double)0.0)           /* Double */\n"
      << "#define LLVM_NANF(NanStr)  0.0F                    /* Float */\n"
      << "#define LLVM_NANS(NanStr)  ((double)0.0)           /* Double */\n"
      << "#define LLVM_NANSF(NanStr) 0.0F                    /* Float */\n"
      << "#define LLVM_INF           ((double)0.0)           /* Double */\n"
      << "#define LLVM_INFF          0.0F                    /* Float */\n"
      << "#define LLVM_PREFETCH(addr,rw,locality)            /* PREFETCH */\n"
      << "#define __ATTRIBUTE_CTOR__\n"
      << "#define __ATTRIBUTE_DTOR__\n"
      << "#define LLVM_ASM(X)\n"
      << "#endif\n\n";
  
  Out << "#if __GNUC__ < 4 /* Old GCC's, or compilers not GCC */ \n"
      << "#define __builtin_stack_save() 0   /* not implemented */\n"
      << "#define __builtin_stack_restore(X) /* noop */\n"
      << "#endif\n\n";

  // Output typedefs for 128-bit integers. If these are needed with a
  // 32-bit target or with a C compiler that doesn't support mode(TI),
  // more drastic measures will be needed.
  Out << "#if __GNUC__ && __LP64__ /* 128-bit integer types */\n"
      << "typedef int __attribute__((mode(TI))) llvmInt128;\n"
      << "typedef unsigned __attribute__((mode(TI))) llvmUInt128;\n"
      << "#endif\n\n";

  // Output target-specific code that should be inserted into main.
  Out << "#define CODE_FOR_MAIN() /* Any target-specific code for main()*/\n";
}

/// FindStaticTors - Given a static ctor/dtor list, unpack its contents into
/// the StaticTors set.
static void FindStaticTors(GlobalVariable *GV, std::set<Function*> &StaticTors){
  ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!InitList) return;
  
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(InitList->getOperand(i))){
      if (CS->getNumOperands() != 2) return;  // Not array of 2-element structs.
      
      if (CS->getOperand(1)->isNullValue())
        return;  // Found a null terminator, exit printing.
      Constant *FP = CS->getOperand(1);
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(FP))
        if (CE->isCast())
          FP = CE->getOperand(0);
      if (Function *F = dyn_cast<Function>(FP))
        StaticTors.insert(F);
    }
}

enum SpecialGlobalClass {
  NotSpecial = 0,
  GlobalCtors, GlobalDtors,
  NotPrinted
};

/// getGlobalVariableClass - If this is a global that is specially recognized
/// by LLVM, return a code that indicates how we should handle it.
static SpecialGlobalClass getGlobalVariableClass(const GlobalVariable *GV) {
  // If this is a global ctors/dtors list, handle it now.
  if (GV->hasAppendingLinkage() && GV->use_empty()) {
    if (GV->getName() == "llvm.global_ctors")
      return GlobalCtors;
    else if (GV->getName() == "llvm.global_dtors")
      return GlobalDtors;
  }
  
  // Otherwise, it it is other metadata, don't print it.  This catches things
  // like debug information.
  if (GV->getSection() == "llvm.metadata")
    return NotPrinted;
  
  return NotSpecial;
}


bool CWriter::doInitialization(Module &M) {
  // Initialize
  TheModule = &M;

  TD = new TargetData(&M);
  IL = new IntrinsicLowering(*TD);
  IL->AddPrototypes(M);

  // Ensure that all structure types have names...
  Mang = new Mangler(M);
  Mang->markCharUnacceptable('.');

  // Keep track of which functions are static ctors/dtors so they can have
  // an attribute added to their prototypes.
  std::set<Function*> StaticCtors, StaticDtors;
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    switch (getGlobalVariableClass(I)) {
    default: break;
    case GlobalCtors:
      FindStaticTors(I, StaticCtors);
      break;
    case GlobalDtors:
      FindStaticTors(I, StaticDtors);
      break;
    }
  }
  
  // get declaration for alloca
  Out << "/* Provide Declarations */\n";
  Out << "#include <stdarg.h>\n";      // Varargs support
  Out << "#include <setjmp.h>\n";      // Unwind support
  generateCompilerSpecificCode(Out, TD);

  // Provide a definition for `bool' if not compiling with a C++ compiler.
  Out << "\n"
      << "#ifndef __cplusplus\ntypedef unsigned char bool;\n#endif\n"

      << "\n\n/* Support for floating point constants */\n"
      << "typedef unsigned long long ConstantDoubleTy;\n"
      << "typedef unsigned int        ConstantFloatTy;\n"
      << "typedef struct { unsigned long long f1; unsigned short f2; "
         "unsigned short pad[3]; } ConstantFP80Ty;\n"
      // This is used for both kinds of 128-bit long double; meaning differs.
      << "typedef struct { unsigned long long f1; unsigned long long f2; }"
         " ConstantFP128Ty;\n"
      << "\n\n/* Global Declarations */\n";

  // First output all the declarations for the program, because C requires
  // Functions & globals to be declared before they are used.
  //

  // Loop over the symbol table, emitting all named constants...
  printModuleTypes(M.getTypeSymbolTable());

  // Global variable declarations...
  if (!M.global_empty()) {
    Out << "\n/* External Global Variable Declarations */\n";
    for (Module::global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {

      if (I->hasExternalLinkage() || I->hasExternalWeakLinkage() || 
          I->hasCommonLinkage())
        Out << "extern ";
      else if (I->hasDLLImportLinkage())
        Out << "__declspec(dllimport) ";
      else
        continue; // Internal Global

      // Thread Local Storage
      if (I->isThreadLocal())
        Out << "__thread ";

      printType(Out, I->getType()->getElementType(), false, GetValueName(I));

      if (I->hasExternalWeakLinkage())
         Out << " __EXTERNAL_WEAK__";
      Out << ";\n";
    }
  }

  // Function declarations
  Out << "\n/* Function Declarations */\n";
  Out << "double fmod(double, double);\n";   // Support for FP rem
  Out << "float fmodf(float, float);\n";
  Out << "long double fmodl(long double, long double);\n";
  
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    // Don't print declarations for intrinsic functions.
    if (!I->isIntrinsic() && I->getName() != "setjmp" &&
        I->getName() != "longjmp" && I->getName() != "_setjmp") {
      if (I->hasExternalWeakLinkage())
        Out << "extern ";
      printFunctionSignature(I, true);
      if (I->hasWeakLinkage() || I->hasLinkOnceLinkage()) 
        Out << " __ATTRIBUTE_WEAK__";
      if (I->hasExternalWeakLinkage())
        Out << " __EXTERNAL_WEAK__";
      if (StaticCtors.count(I))
        Out << " __ATTRIBUTE_CTOR__";
      if (StaticDtors.count(I))
        Out << " __ATTRIBUTE_DTOR__";
      if (I->hasHiddenVisibility())
        Out << " __HIDDEN__";
      
      if (I->hasName() && I->getName()[0] == 1)
        Out << " LLVM_ASM(\"" << I->getName().c_str()+1 << "\")";
          
      Out << ";\n";
    }
  }

  // Output the global variable declarations
  if (!M.global_empty()) {
    Out << "\n\n/* Global Variable Declarations */\n";
    for (Module::global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I)
      if (!I->isDeclaration()) {
        // Ignore special globals, such as debug info.
        if (getGlobalVariableClass(I))
          continue;

        if (I->hasLocalLinkage())
          Out << "static ";
        else
          Out << "extern ";

        // Thread Local Storage
        if (I->isThreadLocal())
          Out << "__thread ";

        printType(Out, I->getType()->getElementType(), false, 
                  GetValueName(I));

        if (I->hasLinkOnceLinkage())
          Out << " __attribute__((common))";
        else if (I->hasCommonLinkage())     // FIXME is this right?
          Out << " __ATTRIBUTE_WEAK__";
        else if (I->hasWeakLinkage())
          Out << " __ATTRIBUTE_WEAK__";
        else if (I->hasExternalWeakLinkage())
          Out << " __EXTERNAL_WEAK__";
        if (I->hasHiddenVisibility())
          Out << " __HIDDEN__";
        Out << ";\n";
      }
  }

  // Output the global variable definitions and contents...
  if (!M.global_empty()) {
    Out << "\n\n/* Global Variable Definitions and Initialization */\n";
    for (Module::global_iterator I = M.global_begin(), E = M.global_end(); 
         I != E; ++I)
      if (!I->isDeclaration()) {
        // Ignore special globals, such as debug info.
        if (getGlobalVariableClass(I))
          continue;

        if (I->hasLocalLinkage())
          Out << "static ";
        else if (I->hasDLLImportLinkage())
          Out << "__declspec(dllimport) ";
        else if (I->hasDLLExportLinkage())
          Out << "__declspec(dllexport) ";

        // Thread Local Storage
        if (I->isThreadLocal())
          Out << "__thread ";

        printType(Out, I->getType()->getElementType(), false, 
                  GetValueName(I));
        if (I->hasLinkOnceLinkage())
          Out << " __attribute__((common))";
        else if (I->hasWeakLinkage())
          Out << " __ATTRIBUTE_WEAK__";
        else if (I->hasCommonLinkage())
          Out << " __ATTRIBUTE_WEAK__";

        if (I->hasHiddenVisibility())
          Out << " __HIDDEN__";
        
        // If the initializer is not null, emit the initializer.  If it is null,
        // we try to avoid emitting large amounts of zeros.  The problem with
        // this, however, occurs when the variable has weak linkage.  In this
        // case, the assembler will complain about the variable being both weak
        // and common, so we disable this optimization.
        // FIXME common linkage should avoid this problem.
        if (!I->getInitializer()->isNullValue()) {
          Out << " = " ;
          writeOperand(I->getInitializer(), true);
        } else if (I->hasWeakLinkage()) {
          // We have to specify an initializer, but it doesn't have to be
          // complete.  If the value is an aggregate, print out { 0 }, and let
          // the compiler figure out the rest of the zeros.
          Out << " = " ;
          if (isa<StructType>(I->getInitializer()->getType()) ||
              isa<VectorType>(I->getInitializer()->getType())) {
            Out << "{ 0 }";
          } else if (isa<ArrayType>(I->getInitializer()->getType())) {
            // As with structs and vectors, but with an extra set of braces
            // because arrays are wrapped in structs.
            Out << "{ { 0 } }";
          } else {
            // Just print it out normally.
            writeOperand(I->getInitializer(), true);
          }
        }
        Out << ";\n";
      }
  }

  if (!M.empty())
    Out << "\n\n/* Function Bodies */\n";

  // Emit some helper functions for dealing with FCMP instruction's 
  // predicates
  Out << "static inline int llvm_fcmp_ord(double X, double Y) { ";
  Out << "return X == X && Y == Y; }\n";
  Out << "static inline int llvm_fcmp_uno(double X, double Y) { ";
  Out << "return X != X || Y != Y; }\n";
  Out << "static inline int llvm_fcmp_ueq(double X, double Y) { ";
  Out << "return X == Y || llvm_fcmp_uno(X, Y); }\n";
  Out << "static inline int llvm_fcmp_une(double X, double Y) { ";
  Out << "return X != Y; }\n";
  Out << "static inline int llvm_fcmp_ult(double X, double Y) { ";
  Out << "return X <  Y || llvm_fcmp_uno(X, Y); }\n";
  Out << "static inline int llvm_fcmp_ugt(double X, double Y) { ";
  Out << "return X >  Y || llvm_fcmp_uno(X, Y); }\n";
  Out << "static inline int llvm_fcmp_ule(double X, double Y) { ";
  Out << "return X <= Y || llvm_fcmp_uno(X, Y); }\n";
  Out << "static inline int llvm_fcmp_uge(double X, double Y) { ";
  Out << "return X >= Y || llvm_fcmp_uno(X, Y); }\n";
  Out << "static inline int llvm_fcmp_oeq(double X, double Y) { ";
  Out << "return X == Y ; }\n";
  Out << "static inline int llvm_fcmp_one(double X, double Y) { ";
  Out << "return X != Y && llvm_fcmp_ord(X, Y); }\n";
  Out << "static inline int llvm_fcmp_olt(double X, double Y) { ";
  Out << "return X <  Y ; }\n";
  Out << "static inline int llvm_fcmp_ogt(double X, double Y) { ";
  Out << "return X >  Y ; }\n";
  Out << "static inline int llvm_fcmp_ole(double X, double Y) { ";
  Out << "return X <= Y ; }\n";
  Out << "static inline int llvm_fcmp_oge(double X, double Y) { ";
  Out << "return X >= Y ; }\n";
  return false;
}


/// Output all floating point constants that cannot be printed accurately...
void CWriter::printFloatingPointConstants(Function &F) {
  // Scan the module for floating point constants.  If any FP constant is used
  // in the function, we want to redirect it here so that we do not depend on
  // the precision of the printed form, unless the printed form preserves
  // precision.
  //
  for (constant_iterator I = constant_begin(&F), E = constant_end(&F);
       I != E; ++I)
    printFloatingPointConstants(*I);

  Out << '\n';
}

void CWriter::printFloatingPointConstants(const Constant *C) {
  // If this is a constant expression, recursively check for constant fp values.
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i)
      printFloatingPointConstants(CE->getOperand(i));
    return;
  }
    
  // Otherwise, check for a FP constant that we need to print.
  const ConstantFP *FPC = dyn_cast<ConstantFP>(C);
  if (FPC == 0 ||
      // Do not put in FPConstantMap if safe.
      isFPCSafeToPrint(FPC) ||
      // Already printed this constant?
      FPConstantMap.count(FPC))
    return;

  FPConstantMap[FPC] = FPCounter;  // Number the FP constants
  
  if (FPC->getType() == Type::DoubleTy) {
    double Val = FPC->getValueAPF().convertToDouble();
    uint64_t i = FPC->getValueAPF().bitcastToAPInt().getZExtValue();
    Out << "static const ConstantDoubleTy FPConstant" << FPCounter++
    << " = 0x" << utohexstr(i)
    << "ULL;    /* " << Val << " */\n";
  } else if (FPC->getType() == Type::FloatTy) {
    float Val = FPC->getValueAPF().convertToFloat();
    uint32_t i = (uint32_t)FPC->getValueAPF().bitcastToAPInt().
    getZExtValue();
    Out << "static const ConstantFloatTy FPConstant" << FPCounter++
    << " = 0x" << utohexstr(i)
    << "U;    /* " << Val << " */\n";
  } else if (FPC->getType() == Type::X86_FP80Ty) {
    // api needed to prevent premature destruction
    APInt api = FPC->getValueAPF().bitcastToAPInt();
    const uint64_t *p = api.getRawData();
    Out << "static const ConstantFP80Ty FPConstant" << FPCounter++
    << " = { 0x"
    << utohexstr((uint16_t)p[1] | (p[0] & 0xffffffffffffLL)<<16)
    << "ULL, 0x" << utohexstr((uint16_t)(p[0] >> 48)) << ",{0,0,0}"
    << "}; /* Long double constant */\n";
  } else if (FPC->getType() == Type::PPC_FP128Ty) {
    APInt api = FPC->getValueAPF().bitcastToAPInt();
    const uint64_t *p = api.getRawData();
    Out << "static const ConstantFP128Ty FPConstant" << FPCounter++
    << " = { 0x"
    << utohexstr(p[0]) << ", 0x" << utohexstr(p[1])
    << "}; /* Long double constant */\n";
    
  } else {
    assert(0 && "Unknown float type!");
  }
}



/// printSymbolTable - Run through symbol table looking for type names.  If a
/// type name is found, emit its declaration...
///
void CWriter::printModuleTypes(const TypeSymbolTable &TST) {
  Out << "/* Helper union for bitcasts */\n";
  Out << "typedef union {\n";
  Out << "  unsigned int Int32;\n";
  Out << "  unsigned long long Int64;\n";
  Out << "  float Float;\n";
  Out << "  double Double;\n";
  Out << "} llvmBitCastUnion;\n";

  // We are only interested in the type plane of the symbol table.
  TypeSymbolTable::const_iterator I   = TST.begin();
  TypeSymbolTable::const_iterator End = TST.end();

  // If there are no type names, exit early.
  if (I == End) return;

  // Print out forward declarations for structure types before anything else!
  Out << "/* Structure forward decls */\n";
  for (; I != End; ++I) {
    std::string Name = "struct l_" + Mang->makeNameProper(I->first);
    Out << Name << ";\n";
    TypeNames.insert(std::make_pair(I->second, Name));
  }

  Out << '\n';

  // Now we can print out typedefs.  Above, we guaranteed that this can only be
  // for struct or opaque types.
  Out << "/* Typedefs */\n";
  for (I = TST.begin(); I != End; ++I) {
    std::string Name = "l_" + Mang->makeNameProper(I->first);
    Out << "typedef ";
    printType(Out, I->second, false, Name);
    Out << ";\n";
  }

  Out << '\n';

  // Keep track of which structures have been printed so far...
  std::set<const Type *> StructPrinted;

  // Loop over all structures then push them into the stack so they are
  // printed in the correct order.
  //
  Out << "/* Structure contents */\n";
  for (I = TST.begin(); I != End; ++I)
    if (isa<StructType>(I->second) || isa<ArrayType>(I->second))
      // Only print out used types!
      printContainedStructs(I->second, StructPrinted);
}

// Push the struct onto the stack and recursively push all structs
// this one depends on.
//
// TODO:  Make this work properly with vector types
//
void CWriter::printContainedStructs(const Type *Ty,
                                    std::set<const Type*> &StructPrinted) {
  // Don't walk through pointers.
  if (isa<PointerType>(Ty) || Ty->isPrimitiveType() || Ty->isInteger()) return;
  
  // Print all contained types first.
  for (Type::subtype_iterator I = Ty->subtype_begin(),
       E = Ty->subtype_end(); I != E; ++I)
    printContainedStructs(*I, StructPrinted);
  
  if (isa<StructType>(Ty) || isa<ArrayType>(Ty)) {
    // Check to see if we have already printed this struct.
    if (StructPrinted.insert(Ty).second) {
      // Print structure type out.
      std::string Name = TypeNames[Ty];
      printType(Out, Ty, false, Name, true);
      Out << ";\n\n";
    }
  }
}

void CWriter::printFunctionSignature(const Function *F, bool Prototype) {
  /// isStructReturn - Should this function actually return a struct by-value?
  bool isStructReturn = F->hasStructRetAttr();
  
  if (F->hasLocalLinkage()) Out << "static ";
  if (F->hasDLLImportLinkage()) Out << "__declspec(dllimport) ";
  if (F->hasDLLExportLinkage()) Out << "__declspec(dllexport) ";  
  switch (F->getCallingConv()) {
   case CallingConv::X86_StdCall:
    Out << "__attribute__((stdcall)) ";
    break;
   case CallingConv::X86_FastCall:
    Out << "__attribute__((fastcall)) ";
    break;
  }
  
  // Loop over the arguments, printing them...
  const FunctionType *FT = cast<FunctionType>(F->getFunctionType());
  const AttrListPtr &PAL = F->getAttributes();

  std::stringstream FunctionInnards;

  // Print out the name...
  FunctionInnards << GetValueName(F) << '(';

  bool PrintedArg = false;
  if (!F->isDeclaration()) {
    if (!F->arg_empty()) {
      Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
      unsigned Idx = 1;
      
      // If this is a struct-return function, don't print the hidden
      // struct-return argument.
      if (isStructReturn) {
        assert(I != E && "Invalid struct return function!");
        ++I;
        ++Idx;
      }
      
      std::string ArgName;
      for (; I != E; ++I) {
        if (PrintedArg) FunctionInnards << ", ";
        if (I->hasName() || !Prototype)
          ArgName = GetValueName(I);
        else
          ArgName = "";
        const Type *ArgTy = I->getType();
        if (PAL.paramHasAttr(Idx, Attribute::ByVal)) {
          ArgTy = cast<PointerType>(ArgTy)->getElementType();
          ByValParams.insert(I);
        }
        printType(FunctionInnards, ArgTy,
            /*isSigned=*/PAL.paramHasAttr(Idx, Attribute::SExt),
            ArgName);
        PrintedArg = true;
        ++Idx;
      }
    }
  } else {
    // Loop over the arguments, printing them.
    FunctionType::param_iterator I = FT->param_begin(), E = FT->param_end();
    unsigned Idx = 1;
    
    // If this is a struct-return function, don't print the hidden
    // struct-return argument.
    if (isStructReturn) {
      assert(I != E && "Invalid struct return function!");
      ++I;
      ++Idx;
    }
    
    for (; I != E; ++I) {
      if (PrintedArg) FunctionInnards << ", ";
      const Type *ArgTy = *I;
      if (PAL.paramHasAttr(Idx, Attribute::ByVal)) {
        assert(isa<PointerType>(ArgTy));
        ArgTy = cast<PointerType>(ArgTy)->getElementType();
      }
      printType(FunctionInnards, ArgTy,
             /*isSigned=*/PAL.paramHasAttr(Idx, Attribute::SExt));
      PrintedArg = true;
      ++Idx;
    }
  }

  // Finish printing arguments... if this is a vararg function, print the ...,
  // unless there are no known types, in which case, we just emit ().
  //
  if (FT->isVarArg() && PrintedArg) {
    if (PrintedArg) FunctionInnards << ", ";
    FunctionInnards << "...";  // Output varargs portion of signature!
  } else if (!FT->isVarArg() && !PrintedArg) {
    FunctionInnards << "void"; // ret() -> ret(void) in C.
  }
  FunctionInnards << ')';
  
  // Get the return tpe for the function.
  const Type *RetTy;
  if (!isStructReturn)
    RetTy = F->getReturnType();
  else {
    // If this is a struct-return function, print the struct-return type.
    RetTy = cast<PointerType>(FT->getParamType(0))->getElementType();
  }
    
  // Print out the return type and the signature built above.
  printType(Out, RetTy, 
            /*isSigned=*/PAL.paramHasAttr(0, Attribute::SExt),
            FunctionInnards.str());
}

static inline bool isFPIntBitCast(const Instruction &I) {
  if (!isa<BitCastInst>(I))
    return false;
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DstTy = I.getType();
  return (SrcTy->isFloatingPoint() && DstTy->isInteger()) ||
         (DstTy->isFloatingPoint() && SrcTy->isInteger());
}

void CWriter::printFunction(Function &F) {
  /// isStructReturn - Should this function actually return a struct by-value?
  bool isStructReturn = F.hasStructRetAttr();

  printFunctionSignature(&F, false);
  Out << " {\n";
  
  // If this is a struct return function, handle the result with magic.
  if (isStructReturn) {
    const Type *StructTy =
      cast<PointerType>(F.arg_begin()->getType())->getElementType();
    Out << "  ";
    printType(Out, StructTy, false, "StructReturn");
    Out << ";  /* Struct return temporary */\n";

    Out << "  ";
    printType(Out, F.arg_begin()->getType(), false, 
              GetValueName(F.arg_begin()));
    Out << " = &StructReturn;\n";
  }

  bool PrintedVar = false;
  
  // print local variable information for the function
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    if (const AllocaInst *AI = isDirectAlloca(&*I)) {
      Out << "  ";
      printType(Out, AI->getAllocatedType(), false, GetValueName(AI));
      Out << ";    /* Address-exposed local */\n";
      PrintedVar = true;
    } else if (I->getType() != Type::VoidTy && !isInlinableInst(*I)) {
      Out << "  ";
      printType(Out, I->getType(), false, GetValueName(&*I));
      Out << ";\n";

      if (isa<PHINode>(*I)) {  // Print out PHI node temporaries as well...
        Out << "  ";
        printType(Out, I->getType(), false,
                  GetValueName(&*I)+"__PHI_TEMPORARY");
        Out << ";\n";
      }
      PrintedVar = true;
    }
    // We need a temporary for the BitCast to use so it can pluck a value out
    // of a union to do the BitCast. This is separate from the need for a
    // variable to hold the result of the BitCast. 
    if (isFPIntBitCast(*I)) {
      Out << "  llvmBitCastUnion " << GetValueName(&*I)
          << "__BITCAST_TEMPORARY;\n";
      PrintedVar = true;
    }
  }

  if (PrintedVar)
    Out << '\n';

  if (F.hasExternalLinkage() && F.getName() == "main")
    Out << "  CODE_FOR_MAIN();\n";

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

  if (NeedsLabel) Out << GetValueName(BB) << ":\n";

  // Output all of the instructions in the basic block...
  for (BasicBlock::iterator II = BB->begin(), E = --BB->end(); II != E;
       ++II) {
    if (!isInlinableInst(*II) && !isDirectAlloca(II)) {
      if (II->getType() != Type::VoidTy && !isInlineAsm(*II))
        outputLValue(II);
      else
        Out << "  ";
      writeInstComputationInline(*II);
      Out << ";\n";
    }
  }

  // Don't emit prefix or suffix for the terminator.
  visit(*BB->getTerminator());
}


// Specific Instruction type classes... note that all of the casts are
// necessary because we use the instruction classes as opaque types...
//
void CWriter::visitReturnInst(ReturnInst &I) {
  // If this is a struct return function, return the temporary struct.
  bool isStructReturn = I.getParent()->getParent()->hasStructRetAttr();

  if (isStructReturn) {
    Out << "  return StructReturn;\n";
    return;
  }
  
  // Don't output a void return if this is the last basic block in the function
  if (I.getNumOperands() == 0 &&
      &*--I.getParent()->getParent()->end() == I.getParent() &&
      !I.getParent()->size() == 1) {
    return;
  }

  if (I.getNumOperands() > 1) {
    Out << "  {\n";
    Out << "    ";
    printType(Out, I.getParent()->getParent()->getReturnType());
    Out << "   llvm_cbe_mrv_temp = {\n";
    for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
      Out << "      ";
      writeOperand(I.getOperand(i));
      if (i != e - 1)
        Out << ",";
      Out << "\n";
    }
    Out << "    };\n";
    Out << "    return llvm_cbe_mrv_temp;\n";
    Out << "  }\n";
    return;
  }

  Out << "  return";
  if (I.getNumOperands()) {
    Out << ' ';
    writeOperand(I.getOperand(0));
  }
  Out << ";\n";
}

void CWriter::visitSwitchInst(SwitchInst &SI) {

  Out << "  switch (";
  writeOperand(SI.getOperand(0));
  Out << ") {\n  default:\n";
  printPHICopiesForSuccessor (SI.getParent(), SI.getDefaultDest(), 2);
  printBranchToBlock(SI.getParent(), SI.getDefaultDest(), 2);
  Out << ";\n";
  for (unsigned i = 2, e = SI.getNumOperands(); i != e; i += 2) {
    Out << "  case ";
    writeOperand(SI.getOperand(i));
    Out << ":\n";
    BasicBlock *Succ = cast<BasicBlock>(SI.getOperand(i+1));
    printPHICopiesForSuccessor (SI.getParent(), Succ, 2);
    printBranchToBlock(SI.getParent(), Succ, 2);
    if (Function::iterator(Succ) == next(Function::iterator(SI.getParent())))
      Out << "    break;\n";
  }
  Out << "  }\n";
}

void CWriter::visitUnreachableInst(UnreachableInst &I) {
  Out << "  /*UNREACHABLE*/;\n";
}

bool CWriter::isGotoCodeNecessary(BasicBlock *From, BasicBlock *To) {
  /// FIXME: This should be reenabled, but loop reordering safe!!
  return true;

  if (next(Function::iterator(From)) != Function::iterator(To))
    return true;  // Not the direct successor, we need a goto.

  //isa<SwitchInst>(From->getTerminator())

  if (LI->getLoopFor(From) != LI->getLoopFor(To))
    return true;
  return false;
}

void CWriter::printPHICopiesForSuccessor (BasicBlock *CurBlock,
                                          BasicBlock *Successor,
                                          unsigned Indent) {
  for (BasicBlock::iterator I = Successor->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    // Now we have to do the printing.
    Value *IV = PN->getIncomingValueForBlock(CurBlock);
    if (!isa<UndefValue>(IV)) {
      Out << std::string(Indent, ' ');
      Out << "  " << GetValueName(I) << "__PHI_TEMPORARY = ";
      writeOperand(IV);
      Out << ";   /* for PHI node */\n";
    }
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

  if (I.isConditional()) {
    if (isGotoCodeNecessary(I.getParent(), I.getSuccessor(0))) {
      Out << "  if (";
      writeOperand(I.getCondition());
      Out << ") {\n";

      printPHICopiesForSuccessor (I.getParent(), I.getSuccessor(0), 2);
      printBranchToBlock(I.getParent(), I.getSuccessor(0), 2);

      if (isGotoCodeNecessary(I.getParent(), I.getSuccessor(1))) {
        Out << "  } else {\n";
        printPHICopiesForSuccessor (I.getParent(), I.getSuccessor(1), 2);
        printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
      }
    } else {
      // First goto not necessary, assume second one is...
      Out << "  if (!";
      writeOperand(I.getCondition());
      Out << ") {\n";

      printPHICopiesForSuccessor (I.getParent(), I.getSuccessor(1), 2);
      printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
    }

    Out << "  }\n";
  } else {
    printPHICopiesForSuccessor (I.getParent(), I.getSuccessor(0), 0);
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
  if ((I.getType() == Type::Int8Ty) || (I.getType() == Type::Int16Ty) 
      || (I.getType() == Type::FloatTy)) {
    needsCast = true;
    Out << "((";
    printType(Out, I.getType(), false);
    Out << ")(";
  }

  // If this is a negation operation, print it out as such.  For FP, we don't
  // want to print "-0.0 - X".
  if (BinaryOperator::isNeg(&I)) {
    Out << "-(";
    writeOperand(BinaryOperator::getNegArgument(cast<BinaryOperator>(&I)));
    Out << ")";
  } else if (I.getOpcode() == Instruction::FRem) {
    // Output a call to fmod/fmodf instead of emitting a%b
    if (I.getType() == Type::FloatTy)
      Out << "fmodf(";
    else if (I.getType() == Type::DoubleTy)
      Out << "fmod(";
    else  // all 3 flavors of long double
      Out << "fmodl(";
    writeOperand(I.getOperand(0));
    Out << ", ";
    writeOperand(I.getOperand(1));
    Out << ")";
  } else {

    // Write out the cast of the instruction's value back to the proper type
    // if necessary.
    bool NeedsClosingParens = writeInstructionCast(I);

    // Certain instructions require the operand to be forced to a specific type
    // so we use writeOperandWithCast here instead of writeOperand. Similarly
    // below for operand 1
    writeOperandWithCast(I.getOperand(0), I.getOpcode());

    switch (I.getOpcode()) {
    case Instruction::Add:  Out << " + "; break;
    case Instruction::Sub:  Out << " - "; break;
    case Instruction::Mul:  Out << " * "; break;
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem: Out << " % "; break;
    case Instruction::UDiv:
    case Instruction::SDiv: 
    case Instruction::FDiv: Out << " / "; break;
    case Instruction::And:  Out << " & "; break;
    case Instruction::Or:   Out << " | "; break;
    case Instruction::Xor:  Out << " ^ "; break;
    case Instruction::Shl : Out << " << "; break;
    case Instruction::LShr:
    case Instruction::AShr: Out << " >> "; break;
    default: cerr << "Invalid operator type!" << I; abort();
    }

    writeOperandWithCast(I.getOperand(1), I.getOpcode());
    if (NeedsClosingParens)
      Out << "))";
  }

  if (needsCast) {
    Out << "))";
  }
}

void CWriter::visitICmpInst(ICmpInst &I) {
  // We must cast the results of icmp which might be promoted.
  bool needsCast = false;

  // Write out the cast of the instruction's value back to the proper type
  // if necessary.
  bool NeedsClosingParens = writeInstructionCast(I);

  // Certain icmp predicate require the operand to be forced to a specific type
  // so we use writeOperandWithCast here instead of writeOperand. Similarly
  // below for operand 1
  writeOperandWithCast(I.getOperand(0), I);

  switch (I.getPredicate()) {
  case ICmpInst::ICMP_EQ:  Out << " == "; break;
  case ICmpInst::ICMP_NE:  Out << " != "; break;
  case ICmpInst::ICMP_ULE:
  case ICmpInst::ICMP_SLE: Out << " <= "; break;
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_SGE: Out << " >= "; break;
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLT: Out << " < "; break;
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGT: Out << " > "; break;
  default: cerr << "Invalid icmp predicate!" << I; abort();
  }

  writeOperandWithCast(I.getOperand(1), I);
  if (NeedsClosingParens)
    Out << "))";

  if (needsCast) {
    Out << "))";
  }
}

void CWriter::visitFCmpInst(FCmpInst &I) {
  if (I.getPredicate() == FCmpInst::FCMP_FALSE) {
    Out << "0";
    return;
  }
  if (I.getPredicate() == FCmpInst::FCMP_TRUE) {
    Out << "1";
    return;
  }

  const char* op = 0;
  switch (I.getPredicate()) {
  default: assert(0 && "Illegal FCmp predicate");
  case FCmpInst::FCMP_ORD: op = "ord"; break;
  case FCmpInst::FCMP_UNO: op = "uno"; break;
  case FCmpInst::FCMP_UEQ: op = "ueq"; break;
  case FCmpInst::FCMP_UNE: op = "une"; break;
  case FCmpInst::FCMP_ULT: op = "ult"; break;
  case FCmpInst::FCMP_ULE: op = "ule"; break;
  case FCmpInst::FCMP_UGT: op = "ugt"; break;
  case FCmpInst::FCMP_UGE: op = "uge"; break;
  case FCmpInst::FCMP_OEQ: op = "oeq"; break;
  case FCmpInst::FCMP_ONE: op = "one"; break;
  case FCmpInst::FCMP_OLT: op = "olt"; break;
  case FCmpInst::FCMP_OLE: op = "ole"; break;
  case FCmpInst::FCMP_OGT: op = "ogt"; break;
  case FCmpInst::FCMP_OGE: op = "oge"; break;
  }

  Out << "llvm_fcmp_" << op << "(";
  // Write the first operand
  writeOperand(I.getOperand(0));
  Out << ", ";
  // Write the second operand
  writeOperand(I.getOperand(1));
  Out << ")";
}

static const char * getFloatBitCastField(const Type *Ty) {
  switch (Ty->getTypeID()) {
    default: assert(0 && "Invalid Type");
    case Type::FloatTyID:  return "Float";
    case Type::DoubleTyID: return "Double";
    case Type::IntegerTyID: {
      unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
      if (NumBits <= 32)
        return "Int32";
      else
        return "Int64";
    }
  }
}

void CWriter::visitCastInst(CastInst &I) {
  const Type *DstTy = I.getType();
  const Type *SrcTy = I.getOperand(0)->getType();
  if (isFPIntBitCast(I)) {
    Out << '(';
    // These int<->float and long<->double casts need to be handled specially
    Out << GetValueName(&I) << "__BITCAST_TEMPORARY." 
        << getFloatBitCastField(I.getOperand(0)->getType()) << " = ";
    writeOperand(I.getOperand(0));
    Out << ", " << GetValueName(&I) << "__BITCAST_TEMPORARY."
        << getFloatBitCastField(I.getType());
    Out << ')';
    return;
  }
  
  Out << '(';
  printCast(I.getOpcode(), SrcTy, DstTy);

  // Make a sext from i1 work by subtracting the i1 from 0 (an int).
  if (SrcTy == Type::Int1Ty && I.getOpcode() == Instruction::SExt)
    Out << "0-";
  
  writeOperand(I.getOperand(0));
    
  if (DstTy == Type::Int1Ty && 
      (I.getOpcode() == Instruction::Trunc ||
       I.getOpcode() == Instruction::FPToUI ||
       I.getOpcode() == Instruction::FPToSI ||
       I.getOpcode() == Instruction::PtrToInt)) {
    // Make sure we really get a trunc to bool by anding the operand with 1 
    Out << "&1u";
  }
  Out << ')';
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
  // This is used to keep track of intrinsics that get generated to a lowered
  // function. We must generate the prototypes before the function body which
  // will only be expanded on first use (by the loop below).
  std::vector<Function*> prototypesToGen;

  // Examine all the instructions in this function to find the intrinsics that
  // need to be lowered.
  for (Function::iterator BB = F.begin(), EE = F.end(); BB != EE; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(I++))
        if (Function *F = CI->getCalledFunction())
          switch (F->getIntrinsicID()) {
          case Intrinsic::not_intrinsic:
          case Intrinsic::memory_barrier:
          case Intrinsic::vastart:
          case Intrinsic::vacopy:
          case Intrinsic::vaend:
          case Intrinsic::returnaddress:
          case Intrinsic::frameaddress:
          case Intrinsic::setjmp:
          case Intrinsic::longjmp:
          case Intrinsic::prefetch:
          case Intrinsic::dbg_stoppoint:
          case Intrinsic::powi:
          case Intrinsic::x86_sse_cmp_ss:
          case Intrinsic::x86_sse_cmp_ps:
          case Intrinsic::x86_sse2_cmp_sd:
          case Intrinsic::x86_sse2_cmp_pd:
          case Intrinsic::ppc_altivec_lvsl:
              // We directly implement these intrinsics
            break;
          default:
            // If this is an intrinsic that directly corresponds to a GCC
            // builtin, we handle it.
            const char *BuiltinName = "";
#define GET_GCC_BUILTIN_NAME
#include "llvm/Intrinsics.gen"
#undef GET_GCC_BUILTIN_NAME
            // If we handle it, don't lower it.
            if (BuiltinName[0]) break;
            
            // All other intrinsic calls we must lower.
            Instruction *Before = 0;
            if (CI != &BB->front())
              Before = prior(BasicBlock::iterator(CI));

            IL->LowerIntrinsicCall(CI);
            if (Before) {        // Move iterator to instruction after call
              I = Before; ++I;
            } else {
              I = BB->begin();
            }
            // If the intrinsic got lowered to another call, and that call has
            // a definition then we need to make sure its prototype is emitted
            // before any calls to it.
            if (CallInst *Call = dyn_cast<CallInst>(I))
              if (Function *NewF = Call->getCalledFunction())
                if (!NewF->isDeclaration())
                  prototypesToGen.push_back(NewF);

            break;
          }

  // We may have collected some prototypes to emit in the loop above. 
  // Emit them now, before the function that uses them is emitted. But,
  // be careful not to emit them twice.
  std::vector<Function*>::iterator I = prototypesToGen.begin();
  std::vector<Function*>::iterator E = prototypesToGen.end();
  for ( ; I != E; ++I) {
    if (intrinsicPrototypesAlreadyGenerated.insert(*I).second) {
      Out << '\n';
      printFunctionSignature(*I, true);
      Out << ";\n";
    }
  }
}

void CWriter::visitCallInst(CallInst &I) {
  if (isa<InlineAsm>(I.getOperand(0)))
    return visitInlineAsm(I);

  bool WroteCallee = false;

  // Handle intrinsic function calls first...
  if (Function *F = I.getCalledFunction())
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID())
      if (visitBuiltinCall(I, ID, WroteCallee))
        return;

  Value *Callee = I.getCalledValue();

  const PointerType  *PTy   = cast<PointerType>(Callee->getType());
  const FunctionType *FTy   = cast<FunctionType>(PTy->getElementType());

  // If this is a call to a struct-return function, assign to the first
  // parameter instead of passing it to the call.
  const AttrListPtr &PAL = I.getAttributes();
  bool hasByVal = I.hasByValArgument();
  bool isStructRet = I.hasStructRetAttr();
  if (isStructRet) {
    writeOperandDeref(I.getOperand(1));
    Out << " = ";
  }
  
  if (I.isTailCall()) Out << " /*tail*/ ";
  
  if (!WroteCallee) {
    // If this is an indirect call to a struct return function, we need to cast
    // the pointer. Ditto for indirect calls with byval arguments.
    bool NeedsCast = (hasByVal || isStructRet) && !isa<Function>(Callee);

    // GCC is a real PITA.  It does not permit codegening casts of functions to
    // function pointers if they are in a call (it generates a trap instruction
    // instead!).  We work around this by inserting a cast to void* in between
    // the function and the function pointer cast.  Unfortunately, we can't just
    // form the constant expression here, because the folder will immediately
    // nuke it.
    //
    // Note finally, that this is completely unsafe.  ANSI C does not guarantee
    // that void* and function pointers have the same size. :( To deal with this
    // in the common case, we handle casts where the number of arguments passed
    // match exactly.
    //
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Callee))
      if (CE->isCast())
        if (Function *RF = dyn_cast<Function>(CE->getOperand(0))) {
          NeedsCast = true;
          Callee = RF;
        }
  
    if (NeedsCast) {
      // Ok, just cast the pointer type.
      Out << "((";
      if (isStructRet)
        printStructReturnPointerFunctionType(Out, PAL,
                             cast<PointerType>(I.getCalledValue()->getType()));
      else if (hasByVal)
        printType(Out, I.getCalledValue()->getType(), false, "", true, PAL);
      else
        printType(Out, I.getCalledValue()->getType());
      Out << ")(void*)";
    }
    writeOperand(Callee);
    if (NeedsCast) Out << ')';
  }

  Out << '(';

  unsigned NumDeclaredParams = FTy->getNumParams();

  CallSite::arg_iterator AI = I.op_begin()+1, AE = I.op_end();
  unsigned ArgNo = 0;
  if (isStructRet) {   // Skip struct return argument.
    ++AI;
    ++ArgNo;
  }
      
  bool PrintedArg = false;
  for (; AI != AE; ++AI, ++ArgNo) {
    if (PrintedArg) Out << ", ";
    if (ArgNo < NumDeclaredParams &&
        (*AI)->getType() != FTy->getParamType(ArgNo)) {
      Out << '(';
      printType(Out, FTy->getParamType(ArgNo), 
            /*isSigned=*/PAL.paramHasAttr(ArgNo+1, Attribute::SExt));
      Out << ')';
    }
    // Check if the argument is expected to be passed by value.
    if (I.paramHasAttr(ArgNo+1, Attribute::ByVal))
      writeOperandDeref(*AI);
    else
      writeOperand(*AI);
    PrintedArg = true;
  }
  Out << ')';
}

/// visitBuiltinCall - Handle the call to the specified builtin.  Returns true
/// if the entire call is handled, return false it it wasn't handled, and
/// optionally set 'WroteCallee' if the callee has already been printed out.
bool CWriter::visitBuiltinCall(CallInst &I, Intrinsic::ID ID,
                               bool &WroteCallee) {
  switch (ID) {
  default: {
    // If this is an intrinsic that directly corresponds to a GCC
    // builtin, we emit it here.
    const char *BuiltinName = "";
    Function *F = I.getCalledFunction();
#define GET_GCC_BUILTIN_NAME
#include "llvm/Intrinsics.gen"
#undef GET_GCC_BUILTIN_NAME
    assert(BuiltinName[0] && "Unknown LLVM intrinsic!");
    
    Out << BuiltinName;
    WroteCallee = true;
    return false;
  }
  case Intrinsic::memory_barrier:
    Out << "__sync_synchronize()";
    return true;
  case Intrinsic::vastart:
    Out << "0; ";
      
    Out << "va_start(*(va_list*)";
    writeOperand(I.getOperand(1));
    Out << ", ";
    // Output the last argument to the enclosing function.
    if (I.getParent()->getParent()->arg_empty()) {
      cerr << "The C backend does not currently support zero "
           << "argument varargs functions, such as '"
           << I.getParent()->getParent()->getName() << "'!\n";
      abort();
    }
    writeOperand(--I.getParent()->getParent()->arg_end());
    Out << ')';
    return true;
  case Intrinsic::vaend:
    if (!isa<ConstantPointerNull>(I.getOperand(1))) {
      Out << "0; va_end(*(va_list*)";
      writeOperand(I.getOperand(1));
      Out << ')';
    } else {
      Out << "va_end(*(va_list*)0)";
    }
    return true;
  case Intrinsic::vacopy:
    Out << "0; ";
    Out << "va_copy(*(va_list*)";
    writeOperand(I.getOperand(1));
    Out << ", *(va_list*)";
    writeOperand(I.getOperand(2));
    Out << ')';
    return true;
  case Intrinsic::returnaddress:
    Out << "__builtin_return_address(";
    writeOperand(I.getOperand(1));
    Out << ')';
    return true;
  case Intrinsic::frameaddress:
    Out << "__builtin_frame_address(";
    writeOperand(I.getOperand(1));
    Out << ')';
    return true;
  case Intrinsic::powi:
    Out << "__builtin_powi(";
    writeOperand(I.getOperand(1));
    Out << ", ";
    writeOperand(I.getOperand(2));
    Out << ')';
    return true;
  case Intrinsic::setjmp:
    Out << "setjmp(*(jmp_buf*)";
    writeOperand(I.getOperand(1));
    Out << ')';
    return true;
  case Intrinsic::longjmp:
    Out << "longjmp(*(jmp_buf*)";
    writeOperand(I.getOperand(1));
    Out << ", ";
    writeOperand(I.getOperand(2));
    Out << ')';
    return true;
  case Intrinsic::prefetch:
    Out << "LLVM_PREFETCH((const void *)";
    writeOperand(I.getOperand(1));
    Out << ", ";
    writeOperand(I.getOperand(2));
    Out << ", ";
    writeOperand(I.getOperand(3));
    Out << ")";
    return true;
  case Intrinsic::stacksave:
    // Emit this as: Val = 0; *((void**)&Val) = __builtin_stack_save()
    // to work around GCC bugs (see PR1809).
    Out << "0; *((void**)&" << GetValueName(&I)
        << ") = __builtin_stack_save()";
    return true;
  case Intrinsic::dbg_stoppoint: {
    // If we use writeOperand directly we get a "u" suffix which is rejected
    // by gcc.
    std::stringstream SPIStr;
    DbgStopPointInst &SPI = cast<DbgStopPointInst>(I);
    SPI.getDirectory()->print(SPIStr);
    Out << "\n#line "
        << SPI.getLine()
        << " \"";
    Out << SPIStr.str();
    SPIStr.clear();
    SPI.getFileName()->print(SPIStr);
    Out << SPIStr.str() << "\"\n";
    return true;
  }
  case Intrinsic::x86_sse_cmp_ss:
  case Intrinsic::x86_sse_cmp_ps:
  case Intrinsic::x86_sse2_cmp_sd:
  case Intrinsic::x86_sse2_cmp_pd:
    Out << '(';
    printType(Out, I.getType());
    Out << ')';  
    // Multiple GCC builtins multiplex onto this intrinsic.
    switch (cast<ConstantInt>(I.getOperand(3))->getZExtValue()) {
    default: assert(0 && "Invalid llvm.x86.sse.cmp!");
    case 0: Out << "__builtin_ia32_cmpeq"; break;
    case 1: Out << "__builtin_ia32_cmplt"; break;
    case 2: Out << "__builtin_ia32_cmple"; break;
    case 3: Out << "__builtin_ia32_cmpunord"; break;
    case 4: Out << "__builtin_ia32_cmpneq"; break;
    case 5: Out << "__builtin_ia32_cmpnlt"; break;
    case 6: Out << "__builtin_ia32_cmpnle"; break;
    case 7: Out << "__builtin_ia32_cmpord"; break;
    }
    if (ID == Intrinsic::x86_sse_cmp_ps || ID == Intrinsic::x86_sse2_cmp_pd)
      Out << 'p';
    else
      Out << 's';
    if (ID == Intrinsic::x86_sse_cmp_ss || ID == Intrinsic::x86_sse_cmp_ps)
      Out << 's';
    else
      Out << 'd';
      
    Out << "(";
    writeOperand(I.getOperand(1));
    Out << ", ";
    writeOperand(I.getOperand(2));
    Out << ")";
    return true;
  case Intrinsic::ppc_altivec_lvsl:
    Out << '(';
    printType(Out, I.getType());
    Out << ')';  
    Out << "__builtin_altivec_lvsl(0, (void*)";
    writeOperand(I.getOperand(1));
    Out << ")";
    return true;
  }
}

//This converts the llvm constraint string to something gcc is expecting.
//TODO: work out platform independent constraints and factor those out
//      of the per target tables
//      handle multiple constraint codes
std::string CWriter::InterpretASMConstraint(InlineAsm::ConstraintInfo& c) {

  assert(c.Codes.size() == 1 && "Too many asm constraint codes to handle");

  const char *const *table = 0;
  
  //Grab the translation table from TargetAsmInfo if it exists
  if (!TAsm) {
    std::string E;
    const TargetMachineRegistry::entry* Match = 
      TargetMachineRegistry::getClosestStaticTargetForModule(*TheModule, E);
    if (Match) {
      //Per platform Target Machines don't exist, so create it
      // this must be done only once
      const TargetMachine* TM = Match->CtorFn(*TheModule, "");
      TAsm = TM->getTargetAsmInfo();
    }
  }
  if (TAsm)
    table = TAsm->getAsmCBE();

  //Search the translation table if it exists
  for (int i = 0; table && table[i]; i += 2)
    if (c.Codes[0] == table[i])
      return table[i+1];

  //default is identity
  return c.Codes[0];
}

//TODO: import logic from AsmPrinter.cpp
static std::string gccifyAsm(std::string asmstr) {
  for (std::string::size_type i = 0; i != asmstr.size(); ++i)
    if (asmstr[i] == '\n')
      asmstr.replace(i, 1, "\\n");
    else if (asmstr[i] == '\t')
      asmstr.replace(i, 1, "\\t");
    else if (asmstr[i] == '$') {
      if (asmstr[i + 1] == '{') {
        std::string::size_type a = asmstr.find_first_of(':', i + 1);
        std::string::size_type b = asmstr.find_first_of('}', i + 1);
        std::string n = "%" + 
          asmstr.substr(a + 1, b - a - 1) +
          asmstr.substr(i + 2, a - i - 2);
        asmstr.replace(i, b - i + 1, n);
        i += n.size() - 1;
      } else
        asmstr.replace(i, 1, "%");
    }
    else if (asmstr[i] == '%')//grr
      { asmstr.replace(i, 1, "%%"); ++i;}
  
  return asmstr;
}

//TODO: assumptions about what consume arguments from the call are likely wrong
//      handle communitivity
void CWriter::visitInlineAsm(CallInst &CI) {
  InlineAsm* as = cast<InlineAsm>(CI.getOperand(0));
  std::vector<InlineAsm::ConstraintInfo> Constraints = as->ParseConstraints();
  
  std::vector<std::pair<Value*, int> > ResultVals;
  if (CI.getType() == Type::VoidTy)
    ;
  else if (const StructType *ST = dyn_cast<StructType>(CI.getType())) {
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i)
      ResultVals.push_back(std::make_pair(&CI, (int)i));
  } else {
    ResultVals.push_back(std::make_pair(&CI, -1));
  }
  
  // Fix up the asm string for gcc and emit it.
  Out << "__asm__ volatile (\"" << gccifyAsm(as->getAsmString()) << "\"\n";
  Out << "        :";

  unsigned ValueCount = 0;
  bool IsFirst = true;
  
  // Convert over all the output constraints.
  for (std::vector<InlineAsm::ConstraintInfo>::iterator I = Constraints.begin(),
       E = Constraints.end(); I != E; ++I) {
    
    if (I->Type != InlineAsm::isOutput) {
      ++ValueCount;
      continue;  // Ignore non-output constraints.
    }
    
    assert(I->Codes.size() == 1 && "Too many asm constraint codes to handle");
    std::string C = InterpretASMConstraint(*I);
    if (C.empty()) continue;
    
    if (!IsFirst) {
      Out << ", ";
      IsFirst = false;
    }

    // Unpack the dest.
    Value *DestVal;
    int DestValNo = -1;
    
    if (ValueCount < ResultVals.size()) {
      DestVal = ResultVals[ValueCount].first;
      DestValNo = ResultVals[ValueCount].second;
    } else
      DestVal = CI.getOperand(ValueCount-ResultVals.size()+1);

    if (I->isEarlyClobber)
      C = "&"+C;
      
    Out << "\"=" << C << "\"(" << GetValueName(DestVal);
    if (DestValNo != -1)
      Out << ".field" << DestValNo; // Multiple retvals.
    Out << ")";
    ++ValueCount;
  }
  
  
  // Convert over all the input constraints.
  Out << "\n        :";
  IsFirst = true;
  ValueCount = 0;
  for (std::vector<InlineAsm::ConstraintInfo>::iterator I = Constraints.begin(),
       E = Constraints.end(); I != E; ++I) {
    if (I->Type != InlineAsm::isInput) {
      ++ValueCount;
      continue;  // Ignore non-input constraints.
    }
    
    assert(I->Codes.size() == 1 && "Too many asm constraint codes to handle");
    std::string C = InterpretASMConstraint(*I);
    if (C.empty()) continue;
    
    if (!IsFirst) {
      Out << ", ";
      IsFirst = false;
    }
    
    assert(ValueCount >= ResultVals.size() && "Input can't refer to result");
    Value *SrcVal = CI.getOperand(ValueCount-ResultVals.size()+1);
    
    Out << "\"" << C << "\"(";
    if (!I->isIndirect)
      writeOperand(SrcVal);
    else
      writeOperandDeref(SrcVal);
    Out << ")";
  }
  
  // Convert over the clobber constraints.
  IsFirst = true;
  ValueCount = 0;
  for (std::vector<InlineAsm::ConstraintInfo>::iterator I = Constraints.begin(),
       E = Constraints.end(); I != E; ++I) {
    if (I->Type != InlineAsm::isClobber)
      continue;  // Ignore non-input constraints.

    assert(I->Codes.size() == 1 && "Too many asm constraint codes to handle");
    std::string C = InterpretASMConstraint(*I);
    if (C.empty()) continue;
    
    if (!IsFirst) {
      Out << ", ";
      IsFirst = false;
    }
    
    Out << '\"' << C << '"';
  }
  
  Out << ")";
}

void CWriter::visitMallocInst(MallocInst &I) {
  assert(0 && "lowerallocations pass didn't work!");
}

void CWriter::visitAllocaInst(AllocaInst &I) {
  Out << '(';
  printType(Out, I.getType());
  Out << ") alloca(sizeof(";
  printType(Out, I.getType()->getElementType());
  Out << ')';
  if (I.isArrayAllocation()) {
    Out << " * " ;
    writeOperand(I.getOperand(0));
  }
  Out << ')';
}

void CWriter::visitFreeInst(FreeInst &I) {
  assert(0 && "lowerallocations pass didn't work!");
}

void CWriter::printGEPExpression(Value *Ptr, gep_type_iterator I,
                                 gep_type_iterator E, bool Static) {
  
  // If there are no indices, just print out the pointer.
  if (I == E) {
    writeOperand(Ptr);
    return;
  }
    
  // Find out if the last index is into a vector.  If so, we have to print this
  // specially.  Since vectors can't have elements of indexable type, only the
  // last index could possibly be of a vector element.
  const VectorType *LastIndexIsVector = 0;
  {
    for (gep_type_iterator TmpI = I; TmpI != E; ++TmpI)
      LastIndexIsVector = dyn_cast<VectorType>(*TmpI);
  }
  
  Out << "(";
  
  // If the last index is into a vector, we can't print it as &a[i][j] because
  // we can't index into a vector with j in GCC.  Instead, emit this as
  // (((float*)&a[i])+j)
  if (LastIndexIsVector) {
    Out << "((";
    printType(Out, PointerType::getUnqual(LastIndexIsVector->getElementType()));
    Out << ")(";
  }
  
  Out << '&';

  // If the first index is 0 (very typical) we can do a number of
  // simplifications to clean up the code.
  Value *FirstOp = I.getOperand();
  if (!isa<Constant>(FirstOp) || !cast<Constant>(FirstOp)->isNullValue()) {
    // First index isn't simple, print it the hard way.
    writeOperand(Ptr);
  } else {
    ++I;  // Skip the zero index.

    // Okay, emit the first operand. If Ptr is something that is already address
    // exposed, like a global, avoid emitting (&foo)[0], just emit foo instead.
    if (isAddressExposed(Ptr)) {
      writeOperandInternal(Ptr, Static);
    } else if (I != E && isa<StructType>(*I)) {
      // If we didn't already emit the first operand, see if we can print it as
      // P->f instead of "P[0].f"
      writeOperand(Ptr);
      Out << "->field" << cast<ConstantInt>(I.getOperand())->getZExtValue();
      ++I;  // eat the struct index as well.
    } else {
      // Instead of emitting P[0][1], emit (*P)[1], which is more idiomatic.
      Out << "(*";
      writeOperand(Ptr);
      Out << ")";
    }
  }

  for (; I != E; ++I) {
    if (isa<StructType>(*I)) {
      Out << ".field" << cast<ConstantInt>(I.getOperand())->getZExtValue();
    } else if (isa<ArrayType>(*I)) {
      Out << ".array[";
      writeOperandWithCast(I.getOperand(), Instruction::GetElementPtr);
      Out << ']';
    } else if (!isa<VectorType>(*I)) {
      Out << '[';
      writeOperandWithCast(I.getOperand(), Instruction::GetElementPtr);
      Out << ']';
    } else {
      // If the last index is into a vector, then print it out as "+j)".  This
      // works with the 'LastIndexIsVector' code above.
      if (isa<Constant>(I.getOperand()) &&
          cast<Constant>(I.getOperand())->isNullValue()) {
        Out << "))";  // avoid "+0".
      } else {
        Out << ")+(";
        writeOperandWithCast(I.getOperand(), Instruction::GetElementPtr);
        Out << "))";
      }
    }
  }
  Out << ")";
}

void CWriter::writeMemoryAccess(Value *Operand, const Type *OperandType,
                                bool IsVolatile, unsigned Alignment) {

  bool IsUnaligned = Alignment &&
    Alignment < TD->getABITypeAlignment(OperandType);

  if (!IsUnaligned)
    Out << '*';
  if (IsVolatile || IsUnaligned) {
    Out << "((";
    if (IsUnaligned)
      Out << "struct __attribute__ ((packed, aligned(" << Alignment << "))) {";
    printType(Out, OperandType, false, IsUnaligned ? "data" : "volatile*");
    if (IsUnaligned) {
      Out << "; } ";
      if (IsVolatile) Out << "volatile ";
      Out << "*";
    }
    Out << ")";
  }

  writeOperand(Operand);

  if (IsVolatile || IsUnaligned) {
    Out << ')';
    if (IsUnaligned)
      Out << "->data";
  }
}

void CWriter::visitLoadInst(LoadInst &I) {
  writeMemoryAccess(I.getOperand(0), I.getType(), I.isVolatile(),
                    I.getAlignment());

}

void CWriter::visitStoreInst(StoreInst &I) {
  writeMemoryAccess(I.getPointerOperand(), I.getOperand(0)->getType(),
                    I.isVolatile(), I.getAlignment());
  Out << " = ";
  Value *Operand = I.getOperand(0);
  Constant *BitMask = 0;
  if (const IntegerType* ITy = dyn_cast<IntegerType>(Operand->getType()))
    if (!ITy->isPowerOf2ByteWidth())
      // We have a bit width that doesn't match an even power-of-2 byte
      // size. Consequently we must & the value with the type's bit mask
      BitMask = ConstantInt::get(ITy, ITy->getBitMask());
  if (BitMask)
    Out << "((";
  writeOperand(Operand);
  if (BitMask) {
    Out << ") & ";
    printConstant(BitMask, false);
    Out << ")"; 
  }
}

void CWriter::visitGetElementPtrInst(GetElementPtrInst &I) {
  printGEPExpression(I.getPointerOperand(), gep_type_begin(I),
                     gep_type_end(I), false);
}

void CWriter::visitVAArgInst(VAArgInst &I) {
  Out << "va_arg(*(va_list*)";
  writeOperand(I.getOperand(0));
  Out << ", ";
  printType(Out, I.getType());
  Out << ");\n ";
}

void CWriter::visitInsertElementInst(InsertElementInst &I) {
  const Type *EltTy = I.getType()->getElementType();
  writeOperand(I.getOperand(0));
  Out << ";\n  ";
  Out << "((";
  printType(Out, PointerType::getUnqual(EltTy));
  Out << ")(&" << GetValueName(&I) << "))[";
  writeOperand(I.getOperand(2));
  Out << "] = (";
  writeOperand(I.getOperand(1));
  Out << ")";
}

void CWriter::visitExtractElementInst(ExtractElementInst &I) {
  // We know that our operand is not inlined.
  Out << "((";
  const Type *EltTy = 
    cast<VectorType>(I.getOperand(0)->getType())->getElementType();
  printType(Out, PointerType::getUnqual(EltTy));
  Out << ")(&" << GetValueName(I.getOperand(0)) << "))[";
  writeOperand(I.getOperand(1));
  Out << "]";
}

void CWriter::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  Out << "(";
  printType(Out, SVI.getType());
  Out << "){ ";
  const VectorType *VT = SVI.getType();
  unsigned NumElts = VT->getNumElements();
  const Type *EltTy = VT->getElementType();

  for (unsigned i = 0; i != NumElts; ++i) {
    if (i) Out << ", ";
    int SrcVal = SVI.getMaskValue(i);
    if ((unsigned)SrcVal >= NumElts*2) {
      Out << " 0/*undef*/ ";
    } else {
      Value *Op = SVI.getOperand((unsigned)SrcVal >= NumElts);
      if (isa<Instruction>(Op)) {
        // Do an extractelement of this value from the appropriate input.
        Out << "((";
        printType(Out, PointerType::getUnqual(EltTy));
        Out << ")(&" << GetValueName(Op)
            << "))[" << (SrcVal & (NumElts-1)) << "]";
      } else if (isa<ConstantAggregateZero>(Op) || isa<UndefValue>(Op)) {
        Out << "0";
      } else {
        printConstant(cast<ConstantVector>(Op)->getOperand(SrcVal &
                                                           (NumElts-1)),
                      false);
      }
    }
  }
  Out << "}";
}

void CWriter::visitInsertValueInst(InsertValueInst &IVI) {
  // Start by copying the entire aggregate value into the result variable.
  writeOperand(IVI.getOperand(0));
  Out << ";\n  ";

  // Then do the insert to update the field.
  Out << GetValueName(&IVI);
  for (const unsigned *b = IVI.idx_begin(), *i = b, *e = IVI.idx_end();
       i != e; ++i) {
    const Type *IndexedTy =
      ExtractValueInst::getIndexedType(IVI.getOperand(0)->getType(), b, i+1);
    if (isa<ArrayType>(IndexedTy))
      Out << ".array[" << *i << "]";
    else
      Out << ".field" << *i;
  }
  Out << " = ";
  writeOperand(IVI.getOperand(1));
}

void CWriter::visitExtractValueInst(ExtractValueInst &EVI) {
  Out << "(";
  if (isa<UndefValue>(EVI.getOperand(0))) {
    Out << "(";
    printType(Out, EVI.getType());
    Out << ") 0/*UNDEF*/";
  } else {
    Out << GetValueName(EVI.getOperand(0));
    for (const unsigned *b = EVI.idx_begin(), *i = b, *e = EVI.idx_end();
         i != e; ++i) {
      const Type *IndexedTy =
        ExtractValueInst::getIndexedType(EVI.getOperand(0)->getType(), b, i+1);
      if (isa<ArrayType>(IndexedTy))
        Out << ".array[" << *i << "]";
      else
        Out << ".field" << *i;
    }
  }
  Out << ")";
}

//===----------------------------------------------------------------------===//
//                       External Interface declaration
//===----------------------------------------------------------------------===//

bool CTargetMachine::addPassesToEmitWholeFile(PassManager &PM,
                                              raw_ostream &o,
                                              CodeGenFileType FileType,
                                              bool Fast) {
  if (FileType != TargetMachine::AssemblyFile) return true;

  PM.add(createGCLoweringPass());
  PM.add(createLowerAllocationsPass(true));
  PM.add(createLowerInvokePass());
  PM.add(createCFGSimplificationPass());   // clean up after lower invoke.
  PM.add(new CBackendNameAllUsedStructsAndMergeFunctions());
  PM.add(new CWriter(o));
  PM.add(createGCInfoDeleter());
  return false;
}
