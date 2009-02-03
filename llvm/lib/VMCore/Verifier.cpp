//===-- Verifier.cpp - Implement the Module Verifier -------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the function verifier interface, that can be used for some
// sanity checking of input to the system.
//
// Note that this does not provide full `Java style' security and verifications,
// instead it just tries to ensure that code is well-formed.
//
//  * Both of a binary operator's parameters are of the same type
//  * Verify that the indices of mem access instructions match other operands
//  * Verify that arithmetic and other things are only performed on first-class
//    types.  Verify that shifts & logicals only happen on integrals f.e.
//  * All of the constants in a switch statement are of the correct type
//  * The code is in valid SSA form
//  * It should be illegal to put a label into any other type (like a structure)
//    or to return one. [except constant arrays!]
//  * Only phi nodes can be self referential: 'add i32 %0, %0 ; <int>:0' is bad
//  * PHI nodes must have an entry for each predecessor, with no extras.
//  * PHI nodes must be the first thing in a basic block, all grouped together
//  * PHI nodes must have at least one entry
//  * All basic blocks should only end with terminator insts, not contain them
//  * The entry node to a function must not have predecessors
//  * All Instructions must be embedded into a basic block
//  * Functions cannot take a void-typed parameter
//  * Verify that a function's argument list agrees with it's declared type.
//  * It is illegal to specify a name for a void value.
//  * It is illegal to have a internal global value with no initializer
//  * It is illegal to have a ret instruction that returns a value that does not
//    agree with the function return value type.
//  * Function call argument types match the function prototype
//  * All other things that are tested by asserts spread about the code...
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Verifier.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
#include <sstream>
#include <cstdarg>
using namespace llvm;

namespace {  // Anonymous namespace for class
  struct VISIBILITY_HIDDEN PreVerifier : public FunctionPass {
    static char ID; // Pass ID, replacement for typeid

    PreVerifier() : FunctionPass(&ID) { }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    // Check that the prerequisites for successful DominatorTree construction
    // are satisfied.
    bool runOnFunction(Function &F) {
      bool Broken = false;

      for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
        if (I->empty() || !I->back().isTerminator()) {
          cerr << "Basic Block does not have terminator!\n";
          WriteAsOperand(*cerr, I, true);
          cerr << "\n";
          Broken = true;
        }
      }

      if (Broken)
        abort();

      return false;
    }
  };
}

char PreVerifier::ID = 0;
static RegisterPass<PreVerifier>
PreVer("preverify", "Preliminary module verification");
static const PassInfo *const PreVerifyID = &PreVer;

namespace {
  struct VISIBILITY_HIDDEN
     Verifier : public FunctionPass, InstVisitor<Verifier> {
    static char ID; // Pass ID, replacement for typeid
    bool Broken;          // Is this module found to be broken?
    bool RealPass;        // Are we not being run by a PassManager?
    VerifierFailureAction action;
                          // What to do if verification fails.
    Module *Mod;          // Module we are verifying right now
    DominatorTree *DT; // Dominator Tree, caution can be null!
    std::stringstream msgs;  // A stringstream to collect messages

    /// InstInThisBlock - when verifying a basic block, keep track of all of the
    /// instructions we have seen so far.  This allows us to do efficient
    /// dominance checks for the case when an instruction has an operand that is
    /// an instruction in the same block.
    SmallPtrSet<Instruction*, 16> InstsInThisBlock;

    Verifier()
      : FunctionPass(&ID), 
      Broken(false), RealPass(true), action(AbortProcessAction),
      DT(0), msgs( std::ios::app | std::ios::out ) {}
    explicit Verifier(VerifierFailureAction ctn)
      : FunctionPass(&ID), 
      Broken(false), RealPass(true), action(ctn), DT(0),
      msgs( std::ios::app | std::ios::out ) {}
    explicit Verifier(bool AB)
      : FunctionPass(&ID), 
      Broken(false), RealPass(true),
      action( AB ? AbortProcessAction : PrintMessageAction), DT(0),
      msgs( std::ios::app | std::ios::out ) {}
    explicit Verifier(DominatorTree &dt)
      : FunctionPass(&ID), 
      Broken(false), RealPass(false), action(PrintMessageAction),
      DT(&dt), msgs( std::ios::app | std::ios::out ) {}


    bool doInitialization(Module &M) {
      Mod = &M;
      verifyTypeSymbolTable(M.getTypeSymbolTable());

      // If this is a real pass, in a pass manager, we must abort before
      // returning back to the pass manager, or else the pass manager may try to
      // run other passes on the broken module.
      if (RealPass)
        return abortIfBroken();
      return false;
    }

    bool runOnFunction(Function &F) {
      // Get dominator information if we are being run by PassManager
      if (RealPass) DT = &getAnalysis<DominatorTree>();

      Mod = F.getParent();

      visit(F);
      InstsInThisBlock.clear();

      // If this is a real pass, in a pass manager, we must abort before
      // returning back to the pass manager, or else the pass manager may try to
      // run other passes on the broken module.
      if (RealPass)
        return abortIfBroken();

      return false;
    }

    bool doFinalization(Module &M) {
      // Scan through, checking all of the external function's linkage now...
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
        visitGlobalValue(*I);

        // Check to make sure function prototypes are okay.
        if (I->isDeclaration()) visitFunction(*I);
      }

      for (Module::global_iterator I = M.global_begin(), E = M.global_end(); 
           I != E; ++I)
        visitGlobalVariable(*I);

      for (Module::alias_iterator I = M.alias_begin(), E = M.alias_end(); 
           I != E; ++I)
        visitGlobalAlias(*I);

      // If the module is broken, abort at this time.
      return abortIfBroken();
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequiredID(PreVerifyID);
      if (RealPass)
        AU.addRequired<DominatorTree>();
    }

    /// abortIfBroken - If the module is broken and we are supposed to abort on
    /// this condition, do so.
    ///
    bool abortIfBroken() {
      if (!Broken) return false;
      msgs << "Broken module found, ";
      switch (action) {
      default: assert(0 && "Unknown action");
      case AbortProcessAction:
        msgs << "compilation aborted!\n";
        cerr << msgs.str();
        abort();
      case PrintMessageAction:
        msgs << "verification continues.\n";
        cerr << msgs.str();
        return false;
      case ReturnStatusAction:
        msgs << "compilation terminated.\n";
        return Broken;
      }
    }


    // Verification methods...
    void verifyTypeSymbolTable(TypeSymbolTable &ST);
    void visitGlobalValue(GlobalValue &GV);
    void visitGlobalVariable(GlobalVariable &GV);
    void visitGlobalAlias(GlobalAlias &GA);
    void visitFunction(Function &F);
    void visitBasicBlock(BasicBlock &BB);
    using InstVisitor<Verifier>::visit;
       
    void visit(Instruction &I);
       
    void visitTruncInst(TruncInst &I);
    void visitZExtInst(ZExtInst &I);
    void visitSExtInst(SExtInst &I);
    void visitFPTruncInst(FPTruncInst &I);
    void visitFPExtInst(FPExtInst &I);
    void visitFPToUIInst(FPToUIInst &I);
    void visitFPToSIInst(FPToSIInst &I);
    void visitUIToFPInst(UIToFPInst &I);
    void visitSIToFPInst(SIToFPInst &I);
    void visitIntToPtrInst(IntToPtrInst &I);
    void visitPtrToIntInst(PtrToIntInst &I);
    void visitBitCastInst(BitCastInst &I);
    void visitPHINode(PHINode &PN);
    void visitBinaryOperator(BinaryOperator &B);
    void visitICmpInst(ICmpInst &IC);
    void visitFCmpInst(FCmpInst &FC);
    void visitExtractElementInst(ExtractElementInst &EI);
    void visitInsertElementInst(InsertElementInst &EI);
    void visitShuffleVectorInst(ShuffleVectorInst &EI);
    void visitVAArgInst(VAArgInst &VAA) { visitInstruction(VAA); }
    void visitCallInst(CallInst &CI);
    void visitInvokeInst(InvokeInst &II);
    void visitGetElementPtrInst(GetElementPtrInst &GEP);
    void visitLoadInst(LoadInst &LI);
    void visitStoreInst(StoreInst &SI);
    void visitInstruction(Instruction &I);
    void visitTerminatorInst(TerminatorInst &I);
    void visitReturnInst(ReturnInst &RI);
    void visitSwitchInst(SwitchInst &SI);
    void visitSelectInst(SelectInst &SI);
    void visitUserOp1(Instruction &I);
    void visitUserOp2(Instruction &I) { visitUserOp1(I); }
    void visitIntrinsicFunctionCall(Intrinsic::ID ID, CallInst &CI);
    void visitAllocationInst(AllocationInst &AI);
    void visitExtractValueInst(ExtractValueInst &EVI);
    void visitInsertValueInst(InsertValueInst &IVI);

    void VerifyCallSite(CallSite CS);
    bool PerformTypeCheck(Intrinsic::ID ID, Function *F, const Type *Ty,
                          int VT, unsigned ArgNo, std::string &Suffix);
    void VerifyIntrinsicPrototype(Intrinsic::ID ID, Function *F,
                                  unsigned RetNum, unsigned ParamNum, ...);
    void VerifyAttrs(Attributes Attrs, const Type *Ty,
                     bool isReturnValue, const Value *V);
    void VerifyFunctionAttrs(const FunctionType *FT, const AttrListPtr &Attrs,
                             const Value *V);

    void WriteValue(const Value *V) {
      if (!V) return;
      if (isa<Instruction>(V)) {
        msgs << *V;
      } else {
        WriteAsOperand(msgs, V, true, Mod);
        msgs << "\n";
      }
    }

    void WriteType(const Type *T) {
      if ( !T ) return;
      WriteTypeSymbolic(msgs, T, Mod );
    }


    // CheckFailed - A check failed, so print out the condition and the message
    // that failed.  This provides a nice place to put a breakpoint if you want
    // to see why something is not correct.
    void CheckFailed(const std::string &Message,
                     const Value *V1 = 0, const Value *V2 = 0,
                     const Value *V3 = 0, const Value *V4 = 0) {
      msgs << Message << "\n";
      WriteValue(V1);
      WriteValue(V2);
      WriteValue(V3);
      WriteValue(V4);
      Broken = true;
    }

    void CheckFailed( const std::string& Message, const Value* V1,
                      const Type* T2, const Value* V3 = 0 ) {
      msgs << Message << "\n";
      WriteValue(V1);
      WriteType(T2);
      WriteValue(V3);
      Broken = true;
    }
  };
} // End anonymous namespace

char Verifier::ID = 0;
static RegisterPass<Verifier> X("verify", "Module Verifier");

// Assert - We know that cond should be true, if not print an error message.
#define Assert(C, M) \
  do { if (!(C)) { CheckFailed(M); return; } } while (0)
#define Assert1(C, M, V1) \
  do { if (!(C)) { CheckFailed(M, V1); return; } } while (0)
#define Assert2(C, M, V1, V2) \
  do { if (!(C)) { CheckFailed(M, V1, V2); return; } } while (0)
#define Assert3(C, M, V1, V2, V3) \
  do { if (!(C)) { CheckFailed(M, V1, V2, V3); return; } } while (0)
#define Assert4(C, M, V1, V2, V3, V4) \
  do { if (!(C)) { CheckFailed(M, V1, V2, V3, V4); return; } } while (0)


void Verifier::visit(Instruction &I) {
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
    Assert1(I.getOperand(i) != 0, "Operand is null", &I);
  InstVisitor<Verifier>::visit(I);
}


void Verifier::visitGlobalValue(GlobalValue &GV) {
  Assert1(!GV.isDeclaration() ||
          GV.hasExternalLinkage() ||
          GV.hasDLLImportLinkage() ||
          GV.hasExternalWeakLinkage() ||
          GV.hasGhostLinkage() ||
          (isa<GlobalAlias>(GV) &&
           (GV.hasLocalLinkage() || GV.hasWeakLinkage())),
  "Global is external, but doesn't have external or dllimport or weak linkage!",
          &GV);

  Assert1(!GV.hasDLLImportLinkage() || GV.isDeclaration(),
          "Global is marked as dllimport, but not external", &GV);
  
  Assert1(!GV.hasAppendingLinkage() || isa<GlobalVariable>(GV),
          "Only global variables can have appending linkage!", &GV);

  if (GV.hasAppendingLinkage()) {
    GlobalVariable &GVar = cast<GlobalVariable>(GV);
    Assert1(isa<ArrayType>(GVar.getType()->getElementType()),
            "Only global arrays can have appending linkage!", &GV);
  }
}

void Verifier::visitGlobalVariable(GlobalVariable &GV) {
  if (GV.hasInitializer()) {
    Assert1(GV.getInitializer()->getType() == GV.getType()->getElementType(),
            "Global variable initializer type does not match global "
            "variable type!", &GV);
  } else {
    Assert1(GV.hasExternalLinkage() || GV.hasDLLImportLinkage() ||
            GV.hasExternalWeakLinkage(),
            "invalid linkage type for global declaration", &GV);
  }

  visitGlobalValue(GV);
}

void Verifier::visitGlobalAlias(GlobalAlias &GA) {
  Assert1(!GA.getName().empty(),
          "Alias name cannot be empty!", &GA);
  Assert1(GA.hasExternalLinkage() || GA.hasLocalLinkage() ||
          GA.hasWeakLinkage(),
          "Alias should have external or external weak linkage!", &GA);
  Assert1(GA.getAliasee(),
          "Aliasee cannot be NULL!", &GA);
  Assert1(GA.getType() == GA.getAliasee()->getType(),
          "Alias and aliasee types should match!", &GA);

  if (!isa<GlobalValue>(GA.getAliasee())) {
    const ConstantExpr *CE = dyn_cast<ConstantExpr>(GA.getAliasee());
    Assert1(CE && CE->getOpcode() == Instruction::BitCast &&
            isa<GlobalValue>(CE->getOperand(0)),
            "Aliasee should be either GlobalValue or bitcast of GlobalValue",
            &GA);
  }

  const GlobalValue* Aliasee = GA.resolveAliasedGlobal(/*stopOnWeak*/ false);
  Assert1(Aliasee,
          "Aliasing chain should end with function or global variable", &GA);

  visitGlobalValue(GA);
}

void Verifier::verifyTypeSymbolTable(TypeSymbolTable &ST) {
}

// VerifyAttrs - Check the given parameter attributes for an argument or return
// value of the specified type.  The value V is printed in error messages.
void Verifier::VerifyAttrs(Attributes Attrs, const Type *Ty, 
                           bool isReturnValue, const Value *V) {
  if (Attrs == Attribute::None)
    return;

  if (isReturnValue) {
    Attributes RetI = Attrs & Attribute::ParameterOnly;
    Assert1(!RetI, "Attribute " + Attribute::getAsString(RetI) +
            " does not apply to return values!", V);
  }
  Attributes FnCheckAttr = Attrs & Attribute::FunctionOnly;
  Assert1(!FnCheckAttr, "Attribute " + Attribute::getAsString(FnCheckAttr) +
          " only applies to functions!", V);
  
  for (unsigned i = 0;
       i < array_lengthof(Attribute::MutuallyIncompatible); ++i) {
    Attributes MutI = Attrs & Attribute::MutuallyIncompatible[i];
    Assert1(!(MutI & (MutI - 1)), "Attributes " +
            Attribute::getAsString(MutI) + " are incompatible!", V);
  }

  Attributes TypeI = Attrs & Attribute::typeIncompatible(Ty);
  Assert1(!TypeI, "Wrong type for attribute " +
          Attribute::getAsString(TypeI), V);

  Attributes ByValI = Attrs & Attribute::ByVal;
  if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
    Assert1(!ByValI || PTy->getElementType()->isSized(),
            "Attribute " + Attribute::getAsString(ByValI) +
            " does not support unsized types!", V);
  } else {
    Assert1(!ByValI,
            "Attribute " + Attribute::getAsString(ByValI) +
            " only applies to parameters with pointer type!", V);
  }
}

// VerifyFunctionAttrs - Check parameter attributes against a function type.
// The value V is printed in error messages.
void Verifier::VerifyFunctionAttrs(const FunctionType *FT,
                                   const AttrListPtr &Attrs,
                                   const Value *V) {
  if (Attrs.isEmpty())
    return;

  bool SawNest = false;

  for (unsigned i = 0, e = Attrs.getNumSlots(); i != e; ++i) {
    const AttributeWithIndex &Attr = Attrs.getSlot(i);

    const Type *Ty;
    if (Attr.Index == 0)
      Ty = FT->getReturnType();
    else if (Attr.Index-1 < FT->getNumParams())
      Ty = FT->getParamType(Attr.Index-1);
    else
      break;  // VarArgs attributes, don't verify.
    
    VerifyAttrs(Attr.Attrs, Ty, Attr.Index == 0, V);

    if (Attr.Attrs & Attribute::Nest) {
      Assert1(!SawNest, "More than one parameter has attribute nest!", V);
      SawNest = true;
    }

    if (Attr.Attrs & Attribute::StructRet)
      Assert1(Attr.Index == 1, "Attribute sret not on first parameter!", V);
  }

  Attributes FAttrs = Attrs.getFnAttributes();
  Assert1(!(FAttrs & (~Attribute::FunctionOnly)),
          "Attribute " + Attribute::getAsString(FAttrs) +
          " does not apply to function!", V);
      
  for (unsigned i = 0;
       i < array_lengthof(Attribute::MutuallyIncompatible); ++i) {
    Attributes MutI = FAttrs & Attribute::MutuallyIncompatible[i];
    Assert1(!(MutI & (MutI - 1)), "Attributes " +
            Attribute::getAsString(MutI) + " are incompatible!", V);
  }
}

static bool VerifyAttributeCount(const AttrListPtr &Attrs, unsigned Params) {
  if (Attrs.isEmpty())
    return true;
    
  unsigned LastSlot = Attrs.getNumSlots() - 1;
  unsigned LastIndex = Attrs.getSlot(LastSlot).Index;
  if (LastIndex <= Params
      || (LastIndex == (unsigned)~0
          && (LastSlot == 0 || Attrs.getSlot(LastSlot - 1).Index <= Params)))  
    return true;
    
  return false;
}
// visitFunction - Verify that a function is ok.
//
void Verifier::visitFunction(Function &F) {
  // Check function arguments.
  const FunctionType *FT = F.getFunctionType();
  unsigned NumArgs = F.arg_size();

  Assert2(FT->getNumParams() == NumArgs,
          "# formal arguments must match # of arguments for function type!",
          &F, FT);
  Assert1(F.getReturnType()->isFirstClassType() ||
          F.getReturnType() == Type::VoidTy || 
          isa<StructType>(F.getReturnType()),
          "Functions cannot return aggregate values!", &F);

  Assert1(!F.hasStructRetAttr() || F.getReturnType() == Type::VoidTy,
          "Invalid struct return type!", &F);

  const AttrListPtr &Attrs = F.getAttributes();

  Assert1(VerifyAttributeCount(Attrs, FT->getNumParams()),
          "Attributes after last parameter!", &F);

  // Check function attributes.
  VerifyFunctionAttrs(FT, Attrs, &F);

  // Check that this function meets the restrictions on this calling convention.
  switch (F.getCallingConv()) {
  default:
    break;
  case CallingConv::C:
    break;
  case CallingConv::Fast:
  case CallingConv::Cold:
  case CallingConv::X86_FastCall:
    Assert1(!F.isVarArg(),
            "Varargs functions must have C calling conventions!", &F);
    break;
  }
  
  // Check that the argument values match the function type for this function...
  unsigned i = 0;
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end();
       I != E; ++I, ++i) {
    Assert2(I->getType() == FT->getParamType(i),
            "Argument value does not match function argument type!",
            I, FT->getParamType(i));
    Assert1(I->getType()->isFirstClassType(),
            "Function arguments must have first-class types!", I);
  }

  if (F.isDeclaration()) {
    Assert1(F.hasExternalLinkage() || F.hasDLLImportLinkage() ||
            F.hasExternalWeakLinkage() || F.hasGhostLinkage(),
            "invalid linkage type for function declaration", &F);
  } else {
    // Verify that this function (which has a body) is not named "llvm.*".  It
    // is not legal to define intrinsics.
    if (F.getName().size() >= 5)
      Assert1(F.getName().substr(0, 5) != "llvm.",
              "llvm intrinsics cannot be defined!", &F);
    
    // Check the entry node
    BasicBlock *Entry = &F.getEntryBlock();
    Assert1(pred_begin(Entry) == pred_end(Entry),
            "Entry block to function must not have predecessors!", Entry);
  }
}


// verifyBasicBlock - Verify that a basic block is well formed...
//
void Verifier::visitBasicBlock(BasicBlock &BB) {
  InstsInThisBlock.clear();

  // Ensure that basic blocks have terminators!
  Assert1(BB.getTerminator(), "Basic Block does not have terminator!", &BB);

  // Check constraints that this basic block imposes on all of the PHI nodes in
  // it.
  if (isa<PHINode>(BB.front())) {
    SmallVector<BasicBlock*, 8> Preds(pred_begin(&BB), pred_end(&BB));
    SmallVector<std::pair<BasicBlock*, Value*>, 8> Values;
    std::sort(Preds.begin(), Preds.end());
    PHINode *PN;
    for (BasicBlock::iterator I = BB.begin(); (PN = dyn_cast<PHINode>(I));++I) {

      // Ensure that PHI nodes have at least one entry!
      Assert1(PN->getNumIncomingValues() != 0,
              "PHI nodes must have at least one entry.  If the block is dead, "
              "the PHI should be removed!", PN);
      Assert1(PN->getNumIncomingValues() == Preds.size(),
              "PHINode should have one entry for each predecessor of its "
              "parent basic block!", PN);

      // Get and sort all incoming values in the PHI node...
      Values.clear();
      Values.reserve(PN->getNumIncomingValues());
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        Values.push_back(std::make_pair(PN->getIncomingBlock(i),
                                        PN->getIncomingValue(i)));
      std::sort(Values.begin(), Values.end());

      for (unsigned i = 0, e = Values.size(); i != e; ++i) {
        // Check to make sure that if there is more than one entry for a
        // particular basic block in this PHI node, that the incoming values are
        // all identical.
        //
        Assert4(i == 0 || Values[i].first  != Values[i-1].first ||
                Values[i].second == Values[i-1].second,
                "PHI node has multiple entries for the same basic block with "
                "different incoming values!", PN, Values[i].first,
                Values[i].second, Values[i-1].second);

        // Check to make sure that the predecessors and PHI node entries are
        // matched up.
        Assert3(Values[i].first == Preds[i],
                "PHI node entries do not match predecessors!", PN,
                Values[i].first, Preds[i]);
      }
    }
  }
}

void Verifier::visitTerminatorInst(TerminatorInst &I) {
  // Ensure that terminators only exist at the end of the basic block.
  Assert1(&I == I.getParent()->getTerminator(),
          "Terminator found in the middle of a basic block!", I.getParent());
  visitInstruction(I);
}

void Verifier::visitReturnInst(ReturnInst &RI) {
  Function *F = RI.getParent()->getParent();
  unsigned N = RI.getNumOperands();
  if (F->getReturnType() == Type::VoidTy) 
    Assert2(N == 0,
            "Found return instr that returns non-void in Function of void "
            "return type!", &RI, F->getReturnType());
  else if (N == 1 && F->getReturnType() == RI.getOperand(0)->getType()) {
    // Exactly one return value and it matches the return type. Good.
  } else if (const StructType *STy = dyn_cast<StructType>(F->getReturnType())) {
    // The return type is a struct; check for multiple return values.
    Assert2(STy->getNumElements() == N,
            "Incorrect number of return values in ret instruction!",
            &RI, F->getReturnType());
    for (unsigned i = 0; i != N; ++i)
      Assert2(STy->getElementType(i) == RI.getOperand(i)->getType(),
              "Function return type does not match operand "
              "type of return inst!", &RI, F->getReturnType());
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(F->getReturnType())) {
    // The return type is an array; check for multiple return values.
    Assert2(ATy->getNumElements() == N,
            "Incorrect number of return values in ret instruction!",
            &RI, F->getReturnType());
    for (unsigned i = 0; i != N; ++i)
      Assert2(ATy->getElementType() == RI.getOperand(i)->getType(),
              "Function return type does not match operand "
              "type of return inst!", &RI, F->getReturnType());
  } else {
    CheckFailed("Function return type does not match operand "
                "type of return inst!", &RI, F->getReturnType());
  }
  
  // Check to make sure that the return value has necessary properties for
  // terminators...
  visitTerminatorInst(RI);
}

void Verifier::visitSwitchInst(SwitchInst &SI) {
  // Check to make sure that all of the constants in the switch instruction
  // have the same type as the switched-on value.
  const Type *SwitchTy = SI.getCondition()->getType();
  for (unsigned i = 1, e = SI.getNumCases(); i != e; ++i)
    Assert1(SI.getCaseValue(i)->getType() == SwitchTy,
            "Switch constants must all be same type as switch value!", &SI);

  visitTerminatorInst(SI);
}

void Verifier::visitSelectInst(SelectInst &SI) {
  Assert1(!SelectInst::areInvalidOperands(SI.getOperand(0), SI.getOperand(1),
                                          SI.getOperand(2)),
          "Invalid operands for select instruction!", &SI);

  Assert1(SI.getTrueValue()->getType() == SI.getType(),
          "Select values must have same type as select instruction!", &SI);
  visitInstruction(SI);
}


/// visitUserOp1 - User defined operators shouldn't live beyond the lifetime of
/// a pass, if any exist, it's an error.
///
void Verifier::visitUserOp1(Instruction &I) {
  Assert1(0, "User-defined operators should not live outside of a pass!", &I);
}

void Verifier::visitTruncInst(TruncInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBitSize = DestTy->getPrimitiveSizeInBits();

  Assert1(SrcTy->isIntOrIntVector(), "Trunc only operates on integer", &I);
  Assert1(DestTy->isIntOrIntVector(), "Trunc only produces integer", &I);
  Assert1(isa<VectorType>(SrcTy) == isa<VectorType>(DestTy),
          "trunc source and destination must both be a vector or neither", &I);
  Assert1(SrcBitSize > DestBitSize,"DestTy too big for Trunc", &I);

  visitInstruction(I);
}

void Verifier::visitZExtInst(ZExtInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  // Get the size of the types in bits, we'll need this later
  Assert1(SrcTy->isIntOrIntVector(), "ZExt only operates on integer", &I);
  Assert1(DestTy->isIntOrIntVector(), "ZExt only produces an integer", &I);
  Assert1(isa<VectorType>(SrcTy) == isa<VectorType>(DestTy),
          "zext source and destination must both be a vector or neither", &I);
  unsigned SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBitSize = DestTy->getPrimitiveSizeInBits();

  Assert1(SrcBitSize < DestBitSize,"Type too small for ZExt", &I);

  visitInstruction(I);
}

void Verifier::visitSExtInst(SExtInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBitSize = DestTy->getPrimitiveSizeInBits();

  Assert1(SrcTy->isIntOrIntVector(), "SExt only operates on integer", &I);
  Assert1(DestTy->isIntOrIntVector(), "SExt only produces an integer", &I);
  Assert1(isa<VectorType>(SrcTy) == isa<VectorType>(DestTy),
          "sext source and destination must both be a vector or neither", &I);
  Assert1(SrcBitSize < DestBitSize,"Type too small for SExt", &I);

  visitInstruction(I);
}

void Verifier::visitFPTruncInst(FPTruncInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();
  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBitSize = DestTy->getPrimitiveSizeInBits();

  Assert1(SrcTy->isFPOrFPVector(),"FPTrunc only operates on FP", &I);
  Assert1(DestTy->isFPOrFPVector(),"FPTrunc only produces an FP", &I);
  Assert1(isa<VectorType>(SrcTy) == isa<VectorType>(DestTy),
          "fptrunc source and destination must both be a vector or neither",&I);
  Assert1(SrcBitSize > DestBitSize,"DestTy too big for FPTrunc", &I);

  visitInstruction(I);
}

void Verifier::visitFPExtInst(FPExtInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBitSize = DestTy->getPrimitiveSizeInBits();

  Assert1(SrcTy->isFPOrFPVector(),"FPExt only operates on FP", &I);
  Assert1(DestTy->isFPOrFPVector(),"FPExt only produces an FP", &I);
  Assert1(isa<VectorType>(SrcTy) == isa<VectorType>(DestTy),
          "fpext source and destination must both be a vector or neither", &I);
  Assert1(SrcBitSize < DestBitSize,"DestTy too small for FPExt", &I);

  visitInstruction(I);
}

void Verifier::visitUIToFPInst(UIToFPInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  bool SrcVec = isa<VectorType>(SrcTy);
  bool DstVec = isa<VectorType>(DestTy);

  Assert1(SrcVec == DstVec,
          "UIToFP source and dest must both be vector or scalar", &I);
  Assert1(SrcTy->isIntOrIntVector(),
          "UIToFP source must be integer or integer vector", &I);
  Assert1(DestTy->isFPOrFPVector(),
          "UIToFP result must be FP or FP vector", &I);

  if (SrcVec && DstVec)
    Assert1(cast<VectorType>(SrcTy)->getNumElements() ==
            cast<VectorType>(DestTy)->getNumElements(),
            "UIToFP source and dest vector length mismatch", &I);

  visitInstruction(I);
}

void Verifier::visitSIToFPInst(SIToFPInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  bool SrcVec = SrcTy->getTypeID() == Type::VectorTyID;
  bool DstVec = DestTy->getTypeID() == Type::VectorTyID;

  Assert1(SrcVec == DstVec,
          "SIToFP source and dest must both be vector or scalar", &I);
  Assert1(SrcTy->isIntOrIntVector(),
          "SIToFP source must be integer or integer vector", &I);
  Assert1(DestTy->isFPOrFPVector(),
          "SIToFP result must be FP or FP vector", &I);

  if (SrcVec && DstVec)
    Assert1(cast<VectorType>(SrcTy)->getNumElements() ==
            cast<VectorType>(DestTy)->getNumElements(),
            "SIToFP source and dest vector length mismatch", &I);

  visitInstruction(I);
}

void Verifier::visitFPToUIInst(FPToUIInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  bool SrcVec = isa<VectorType>(SrcTy);
  bool DstVec = isa<VectorType>(DestTy);

  Assert1(SrcVec == DstVec,
          "FPToUI source and dest must both be vector or scalar", &I);
  Assert1(SrcTy->isFPOrFPVector(), "FPToUI source must be FP or FP vector", &I);
  Assert1(DestTy->isIntOrIntVector(),
          "FPToUI result must be integer or integer vector", &I);

  if (SrcVec && DstVec)
    Assert1(cast<VectorType>(SrcTy)->getNumElements() ==
            cast<VectorType>(DestTy)->getNumElements(),
            "FPToUI source and dest vector length mismatch", &I);

  visitInstruction(I);
}

void Verifier::visitFPToSIInst(FPToSIInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  bool SrcVec = isa<VectorType>(SrcTy);
  bool DstVec = isa<VectorType>(DestTy);

  Assert1(SrcVec == DstVec,
          "FPToSI source and dest must both be vector or scalar", &I);
  Assert1(SrcTy->isFPOrFPVector(),
          "FPToSI source must be FP or FP vector", &I);
  Assert1(DestTy->isIntOrIntVector(),
          "FPToSI result must be integer or integer vector", &I);

  if (SrcVec && DstVec)
    Assert1(cast<VectorType>(SrcTy)->getNumElements() ==
            cast<VectorType>(DestTy)->getNumElements(),
            "FPToSI source and dest vector length mismatch", &I);

  visitInstruction(I);
}

void Verifier::visitPtrToIntInst(PtrToIntInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  Assert1(isa<PointerType>(SrcTy), "PtrToInt source must be pointer", &I);
  Assert1(DestTy->isInteger(), "PtrToInt result must be integral", &I);

  visitInstruction(I);
}

void Verifier::visitIntToPtrInst(IntToPtrInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  Assert1(SrcTy->isInteger(), "IntToPtr source must be an integral", &I);
  Assert1(isa<PointerType>(DestTy), "IntToPtr result must be a pointer",&I);

  visitInstruction(I);
}

void Verifier::visitBitCastInst(BitCastInst &I) {
  // Get the source and destination types
  const Type *SrcTy = I.getOperand(0)->getType();
  const Type *DestTy = I.getType();

  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBitSize = DestTy->getPrimitiveSizeInBits();

  // BitCast implies a no-op cast of type only. No bits change.
  // However, you can't cast pointers to anything but pointers.
  Assert1(isa<PointerType>(DestTy) == isa<PointerType>(DestTy),
          "Bitcast requires both operands to be pointer or neither", &I);
  Assert1(SrcBitSize == DestBitSize, "Bitcast requies types of same width", &I);

  // Disallow aggregates.
  Assert1(!SrcTy->isAggregateType(),
          "Bitcast operand must not be aggregate", &I);
  Assert1(!DestTy->isAggregateType(),
          "Bitcast type must not be aggregate", &I);

  visitInstruction(I);
}

/// visitPHINode - Ensure that a PHI node is well formed.
///
void Verifier::visitPHINode(PHINode &PN) {
  // Ensure that the PHI nodes are all grouped together at the top of the block.
  // This can be tested by checking whether the instruction before this is
  // either nonexistent (because this is begin()) or is a PHI node.  If not,
  // then there is some other instruction before a PHI.
  Assert2(&PN == &PN.getParent()->front() || 
          isa<PHINode>(--BasicBlock::iterator(&PN)),
          "PHI nodes not grouped at top of basic block!",
          &PN, PN.getParent());

  // Check that all of the operands of the PHI node have the same type as the
  // result.
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    Assert1(PN.getType() == PN.getIncomingValue(i)->getType(),
            "PHI node operands are not the same type as the result!", &PN);

  // All other PHI node constraints are checked in the visitBasicBlock method.

  visitInstruction(PN);
}

void Verifier::VerifyCallSite(CallSite CS) {
  Instruction *I = CS.getInstruction();

  Assert1(isa<PointerType>(CS.getCalledValue()->getType()),
          "Called function must be a pointer!", I);
  const PointerType *FPTy = cast<PointerType>(CS.getCalledValue()->getType());
  Assert1(isa<FunctionType>(FPTy->getElementType()),
          "Called function is not pointer to function type!", I);

  const FunctionType *FTy = cast<FunctionType>(FPTy->getElementType());

  // Verify that the correct number of arguments are being passed
  if (FTy->isVarArg())
    Assert1(CS.arg_size() >= FTy->getNumParams(),
            "Called function requires more parameters than were provided!",I);
  else
    Assert1(CS.arg_size() == FTy->getNumParams(),
            "Incorrect number of arguments passed to called function!", I);

  // Verify that all arguments to the call match the function type...
  for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
    Assert3(CS.getArgument(i)->getType() == FTy->getParamType(i),
            "Call parameter type does not match function signature!",
            CS.getArgument(i), FTy->getParamType(i), I);

  const AttrListPtr &Attrs = CS.getAttributes();

  Assert1(VerifyAttributeCount(Attrs, CS.arg_size()),
          "Attributes after last parameter!", I);

  // Verify call attributes.
  VerifyFunctionAttrs(FTy, Attrs, I);

  if (FTy->isVarArg())
    // Check attributes on the varargs part.
    for (unsigned Idx = 1 + FTy->getNumParams(); Idx <= CS.arg_size(); ++Idx) {
      Attributes Attr = Attrs.getParamAttributes(Idx);

      VerifyAttrs(Attr, CS.getArgument(Idx-1)->getType(), false, I);

      Attributes VArgI = Attr & Attribute::VarArgsIncompatible;
      Assert1(!VArgI, "Attribute " + Attribute::getAsString(VArgI) +
              " cannot be used for vararg call arguments!", I);
    }

  visitInstruction(*I);
}

void Verifier::visitCallInst(CallInst &CI) {
  VerifyCallSite(&CI);

  if (Function *F = CI.getCalledFunction()) {
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID())
      visitIntrinsicFunctionCall(ID, CI);
  }
}

void Verifier::visitInvokeInst(InvokeInst &II) {
  VerifyCallSite(&II);
}

/// visitBinaryOperator - Check that both arguments to the binary operator are
/// of the same type!
///
void Verifier::visitBinaryOperator(BinaryOperator &B) {
  Assert1(B.getOperand(0)->getType() == B.getOperand(1)->getType(),
          "Both operands to a binary operator are not of the same type!", &B);

  switch (B.getOpcode()) {
  // Check that logical operators are only used with integral operands.
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    Assert1(B.getType()->isInteger() ||
            (isa<VectorType>(B.getType()) && 
             cast<VectorType>(B.getType())->getElementType()->isInteger()),
            "Logical operators only work with integral types!", &B);
    Assert1(B.getType() == B.getOperand(0)->getType(),
            "Logical operators must have same type for operands and result!",
            &B);
    break;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    Assert1(B.getType()->isInteger() ||
            (isa<VectorType>(B.getType()) && 
             cast<VectorType>(B.getType())->getElementType()->isInteger()),
            "Shifts only work with integral types!", &B);
    Assert1(B.getType() == B.getOperand(0)->getType(),
            "Shift return type must be same as operands!", &B);
    /* FALL THROUGH */
  default:
    // Arithmetic operators only work on integer or fp values
    Assert1(B.getType() == B.getOperand(0)->getType(),
            "Arithmetic operators must have same type for operands and result!",
            &B);
    Assert1(B.getType()->isInteger() || B.getType()->isFloatingPoint() ||
            isa<VectorType>(B.getType()),
            "Arithmetic operators must have integer, fp, or vector type!", &B);
    break;
  }

  visitInstruction(B);
}

void Verifier::visitICmpInst(ICmpInst& IC) {
  // Check that the operands are the same type
  const Type* Op0Ty = IC.getOperand(0)->getType();
  const Type* Op1Ty = IC.getOperand(1)->getType();
  Assert1(Op0Ty == Op1Ty,
          "Both operands to ICmp instruction are not of the same type!", &IC);
  // Check that the operands are the right type
  Assert1(Op0Ty->isIntOrIntVector() || isa<PointerType>(Op0Ty),
          "Invalid operand types for ICmp instruction", &IC);
  visitInstruction(IC);
}

void Verifier::visitFCmpInst(FCmpInst& FC) {
  // Check that the operands are the same type
  const Type* Op0Ty = FC.getOperand(0)->getType();
  const Type* Op1Ty = FC.getOperand(1)->getType();
  Assert1(Op0Ty == Op1Ty,
          "Both operands to FCmp instruction are not of the same type!", &FC);
  // Check that the operands are the right type
  Assert1(Op0Ty->isFPOrFPVector(),
          "Invalid operand types for FCmp instruction", &FC);
  visitInstruction(FC);
}

void Verifier::visitExtractElementInst(ExtractElementInst &EI) {
  Assert1(ExtractElementInst::isValidOperands(EI.getOperand(0),
                                              EI.getOperand(1)),
          "Invalid extractelement operands!", &EI);
  visitInstruction(EI);
}

void Verifier::visitInsertElementInst(InsertElementInst &IE) {
  Assert1(InsertElementInst::isValidOperands(IE.getOperand(0),
                                             IE.getOperand(1),
                                             IE.getOperand(2)),
          "Invalid insertelement operands!", &IE);
  visitInstruction(IE);
}

void Verifier::visitShuffleVectorInst(ShuffleVectorInst &SV) {
  Assert1(ShuffleVectorInst::isValidOperands(SV.getOperand(0), SV.getOperand(1),
                                             SV.getOperand(2)),
          "Invalid shufflevector operands!", &SV);

  const VectorType *VTy = dyn_cast<VectorType>(SV.getOperand(0)->getType());
  Assert1(VTy, "Operands are not a vector type", &SV);

  // Check to see if Mask is valid.
  if (const ConstantVector *MV = dyn_cast<ConstantVector>(SV.getOperand(2))) {
    for (unsigned i = 0, e = MV->getNumOperands(); i != e; ++i) {
      if (ConstantInt* CI = dyn_cast<ConstantInt>(MV->getOperand(i))) {
        Assert1(!CI->uge(VTy->getNumElements()*2),
                "Invalid shufflevector shuffle mask!", &SV);
      } else {
        Assert1(isa<UndefValue>(MV->getOperand(i)),
                "Invalid shufflevector shuffle mask!", &SV);
      }
    }
  } else {
    Assert1(isa<UndefValue>(SV.getOperand(2)) || 
            isa<ConstantAggregateZero>(SV.getOperand(2)),
            "Invalid shufflevector shuffle mask!", &SV);
  }

  visitInstruction(SV);
}

void Verifier::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  SmallVector<Value*, 16> Idxs(GEP.idx_begin(), GEP.idx_end());
  const Type *ElTy =
    GetElementPtrInst::getIndexedType(GEP.getOperand(0)->getType(),
                                      Idxs.begin(), Idxs.end());
  Assert1(ElTy, "Invalid indices for GEP pointer type!", &GEP);
  Assert2(isa<PointerType>(GEP.getType()) &&
          cast<PointerType>(GEP.getType())->getElementType() == ElTy,
          "GEP is not of right type for indices!", &GEP, ElTy);
  visitInstruction(GEP);
}

void Verifier::visitLoadInst(LoadInst &LI) {
  const Type *ElTy =
    cast<PointerType>(LI.getOperand(0)->getType())->getElementType();
  Assert2(ElTy == LI.getType(),
          "Load result type does not match pointer operand type!", &LI, ElTy);
  visitInstruction(LI);
}

void Verifier::visitStoreInst(StoreInst &SI) {
  const Type *ElTy =
    cast<PointerType>(SI.getOperand(1)->getType())->getElementType();
  Assert2(ElTy == SI.getOperand(0)->getType(),
          "Stored value type does not match pointer operand type!", &SI, ElTy);
  visitInstruction(SI);
}

void Verifier::visitAllocationInst(AllocationInst &AI) {
  const PointerType *PTy = AI.getType();
  Assert1(PTy->getAddressSpace() == 0, 
          "Allocation instruction pointer not in the generic address space!",
          &AI);
  Assert1(PTy->getElementType()->isSized(), "Cannot allocate unsized type",
          &AI);
  visitInstruction(AI);
}

void Verifier::visitExtractValueInst(ExtractValueInst &EVI) {
  Assert1(ExtractValueInst::getIndexedType(EVI.getAggregateOperand()->getType(),
                                           EVI.idx_begin(), EVI.idx_end()) ==
          EVI.getType(),
          "Invalid ExtractValueInst operands!", &EVI);
  
  visitInstruction(EVI);
}

void Verifier::visitInsertValueInst(InsertValueInst &IVI) {
  Assert1(ExtractValueInst::getIndexedType(IVI.getAggregateOperand()->getType(),
                                           IVI.idx_begin(), IVI.idx_end()) ==
          IVI.getOperand(1)->getType(),
          "Invalid InsertValueInst operands!", &IVI);
  
  visitInstruction(IVI);
}

/// verifyInstruction - Verify that an instruction is well formed.
///
void Verifier::visitInstruction(Instruction &I) {
  BasicBlock *BB = I.getParent();
  Assert1(BB, "Instruction not embedded in basic block!", &I);

  if (!isa<PHINode>(I)) {   // Check that non-phi nodes are not self referential
    for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
         UI != UE; ++UI)
      Assert1(*UI != (User*)&I ||
              !DT->dominates(&BB->getParent()->getEntryBlock(), BB),
              "Only PHI nodes may reference their own value!", &I);
  }
  
  // Verify that if this is a terminator that it is at the end of the block.
  if (isa<TerminatorInst>(I))
    Assert1(BB->getTerminator() == &I, "Terminator not at end of block!", &I);
  

  // Check that void typed values don't have names
  Assert1(I.getType() != Type::VoidTy || !I.hasName(),
          "Instruction has a name, but provides a void value!", &I);

  // Check that the return value of the instruction is either void or a legal
  // value type.
  Assert1(I.getType() == Type::VoidTy || I.getType()->isFirstClassType()
          || ((isa<CallInst>(I) || isa<InvokeInst>(I)) 
              && isa<StructType>(I.getType())),
          "Instruction returns a non-scalar type!", &I);

  // Check that all uses of the instruction, if they are instructions
  // themselves, actually have parent basic blocks.  If the use is not an
  // instruction, it is an error!
  for (User::use_iterator UI = I.use_begin(), UE = I.use_end();
       UI != UE; ++UI) {
    Assert1(isa<Instruction>(*UI), "Use of instruction is not an instruction!",
            *UI);
    Instruction *Used = cast<Instruction>(*UI);
    Assert2(Used->getParent() != 0, "Instruction referencing instruction not"
            " embeded in a basic block!", &I, Used);
  }

  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    Assert1(I.getOperand(i) != 0, "Instruction has null operand!", &I);

    // Check to make sure that only first-class-values are operands to
    // instructions.
    if (!I.getOperand(i)->getType()->isFirstClassType()) {
      Assert1(0, "Instruction operands must be first-class values!", &I);
    }
    
    if (Function *F = dyn_cast<Function>(I.getOperand(i))) {
      // Check to make sure that the "address of" an intrinsic function is never
      // taken.
      Assert1(!F->isIntrinsic() || (i == 0 && isa<CallInst>(I)),
              "Cannot take the address of an intrinsic!", &I);
      Assert1(F->getParent() == Mod, "Referencing function in another module!",
              &I);
    } else if (BasicBlock *OpBB = dyn_cast<BasicBlock>(I.getOperand(i))) {
      Assert1(OpBB->getParent() == BB->getParent(),
              "Referring to a basic block in another function!", &I);
    } else if (Argument *OpArg = dyn_cast<Argument>(I.getOperand(i))) {
      Assert1(OpArg->getParent() == BB->getParent(),
              "Referring to an argument in another function!", &I);
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(I.getOperand(i))) {
      Assert1(GV->getParent() == Mod, "Referencing global in another module!",
              &I);
    } else if (Instruction *Op = dyn_cast<Instruction>(I.getOperand(i))) {
      BasicBlock *OpBlock = Op->getParent();

      // Check that a definition dominates all of its uses.
      if (!isa<PHINode>(I)) {
        // Invoke results are only usable in the normal destination, not in the
        // exceptional destination.
        if (InvokeInst *II = dyn_cast<InvokeInst>(Op)) {
          OpBlock = II->getNormalDest();
          
          Assert2(OpBlock != II->getUnwindDest(),
                  "No uses of invoke possible due to dominance structure!",
                  Op, II);
          
          // If the normal successor of an invoke instruction has multiple
          // predecessors, then the normal edge from the invoke is critical, so
          // the invoke value can only be live if the destination block
          // dominates all of it's predecessors (other than the invoke) or if
          // the invoke value is only used by a phi in the successor.
          if (!OpBlock->getSinglePredecessor() &&
              DT->dominates(&BB->getParent()->getEntryBlock(), BB)) {
            // The first case we allow is if the use is a PHI operand in the
            // normal block, and if that PHI operand corresponds to the invoke's
            // block.
            bool Bad = true;
            if (PHINode *PN = dyn_cast<PHINode>(&I))
              if (PN->getParent() == OpBlock &&
                  PN->getIncomingBlock(i/2) == Op->getParent())
                Bad = false;
            
            // If it is used by something non-phi, then the other case is that
            // 'OpBlock' dominates all of its predecessors other than the
            // invoke.  In this case, the invoke value can still be used.
            if (Bad) {
              Bad = false;
              for (pred_iterator PI = pred_begin(OpBlock),
                   E = pred_end(OpBlock); PI != E; ++PI) {
                if (*PI != II->getParent() && !DT->dominates(OpBlock, *PI)) {
                  Bad = true;
                  break;
                }
              }
            }
            Assert2(!Bad,
                    "Invoke value defined on critical edge but not dead!", &I,
                    Op);
          }
        } else if (OpBlock == BB) {
          // If they are in the same basic block, make sure that the definition
          // comes before the use.
          Assert2(InstsInThisBlock.count(Op) ||
                  !DT->dominates(&BB->getParent()->getEntryBlock(), BB),
                  "Instruction does not dominate all uses!", Op, &I);
        }

        // Definition must dominate use unless use is unreachable!
        Assert2(InstsInThisBlock.count(Op) || DT->dominates(Op, &I) ||
                !DT->dominates(&BB->getParent()->getEntryBlock(), BB),
                "Instruction does not dominate all uses!", Op, &I);
      } else {
        // PHI nodes are more difficult than other nodes because they actually
        // "use" the value in the predecessor basic blocks they correspond to.
        BasicBlock *PredBB = cast<BasicBlock>(I.getOperand(i+1));
        Assert2(DT->dominates(OpBlock, PredBB) ||
                !DT->dominates(&BB->getParent()->getEntryBlock(), PredBB),
                "Instruction does not dominate all uses!", Op, &I);
      }
    } else if (isa<InlineAsm>(I.getOperand(i))) {
      Assert1(i == 0 && (isa<CallInst>(I) || isa<InvokeInst>(I)),
              "Cannot take the address of an inline asm!", &I);
    }
  }
  InstsInThisBlock.insert(&I);
}

// Flags used by TableGen to mark intrinsic parameters with the
// LLVMExtendedElementVectorType and LLVMTruncatedElementVectorType classes.
static const unsigned ExtendedElementVectorType = 0x40000000;
static const unsigned TruncatedElementVectorType = 0x20000000;

/// visitIntrinsicFunction - Allow intrinsics to be verified in different ways.
///
void Verifier::visitIntrinsicFunctionCall(Intrinsic::ID ID, CallInst &CI) {
  Function *IF = CI.getCalledFunction();
  Assert1(IF->isDeclaration(), "Intrinsic functions should never be defined!",
          IF);
  
#define GET_INTRINSIC_VERIFIER
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_VERIFIER
  
  switch (ID) {
  default:
    break;
  case Intrinsic::dbg_declare:		// llvm.dbg.declare
    if (Constant *C = dyn_cast<Constant>(CI.getOperand(1)))
      Assert1(C && !isa<ConstantPointerNull>(C),
              "invalid llvm.dbg.declare intrinsic call", &CI);
    break;
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset:
    Assert1(isa<ConstantInt>(CI.getOperand(4)),
            "alignment argument of memory intrinsics must be a constant int",
            &CI);
    break;
  case Intrinsic::gcroot:
  case Intrinsic::gcwrite:
  case Intrinsic::gcread:
    if (ID == Intrinsic::gcroot) {
      AllocaInst *AI =
        dyn_cast<AllocaInst>(CI.getOperand(1)->stripPointerCasts());
      Assert1(AI && isa<PointerType>(AI->getType()->getElementType()),
              "llvm.gcroot parameter #1 must be a pointer alloca.", &CI);
      Assert1(isa<Constant>(CI.getOperand(2)),
              "llvm.gcroot parameter #2 must be a constant.", &CI);
    }
      
    Assert1(CI.getParent()->getParent()->hasGC(),
            "Enclosing function does not use GC.", &CI);
    break;
  case Intrinsic::init_trampoline:
    Assert1(isa<Function>(CI.getOperand(2)->stripPointerCasts()),
            "llvm.init_trampoline parameter #2 must resolve to a function.",
            &CI);
    break;
  case Intrinsic::prefetch:
    Assert1(isa<ConstantInt>(CI.getOperand(2)) &&
            isa<ConstantInt>(CI.getOperand(3)) &&
            cast<ConstantInt>(CI.getOperand(2))->getZExtValue() < 2 &&
            cast<ConstantInt>(CI.getOperand(3))->getZExtValue() < 4,
            "invalid arguments to llvm.prefetch",
            &CI);
    break;
  case Intrinsic::stackprotector:
    Assert1(isa<AllocaInst>(CI.getOperand(2)->stripPointerCasts()),
            "llvm.stackprotector parameter #2 must resolve to an alloca.",
            &CI);
    break;
  }
}

/// Produce a string to identify an intrinsic parameter or return value.
/// The ArgNo value numbers the return values from 0 to NumRets-1 and the
/// parameters beginning with NumRets.
///
static std::string IntrinsicParam(unsigned ArgNo, unsigned NumRets) {
  if (ArgNo < NumRets) {
    if (NumRets == 1)
      return "Intrinsic result type";
    else
      return "Intrinsic result type #" + utostr(ArgNo);
  } else
    return "Intrinsic parameter #" + utostr(ArgNo - NumRets);
}

bool Verifier::PerformTypeCheck(Intrinsic::ID ID, Function *F, const Type *Ty,
                                int VT, unsigned ArgNo, std::string &Suffix) {
  const FunctionType *FTy = F->getFunctionType();

  unsigned NumElts = 0;
  const Type *EltTy = Ty;
  const VectorType *VTy = dyn_cast<VectorType>(Ty);
  if (VTy) {
    EltTy = VTy->getElementType();
    NumElts = VTy->getNumElements();
  }

  const Type *RetTy = FTy->getReturnType();
  const StructType *ST = dyn_cast<StructType>(RetTy);
  unsigned NumRets = 1;
  if (ST)
    NumRets = ST->getNumElements();

  if (VT < 0) {
    int Match = ~VT;

    // Check flags that indicate a type that is an integral vector type with
    // elements that are larger or smaller than the elements of the matched
    // type.
    if ((Match & (ExtendedElementVectorType |
                  TruncatedElementVectorType)) != 0) {
      const IntegerType *IEltTy = dyn_cast<IntegerType>(EltTy);
      if (!VTy || !IEltTy) {
        CheckFailed(IntrinsicParam(ArgNo, NumRets) + " is not "
                    "an integral vector type.", F);
        return false;
      }
      // Adjust the current Ty (in the opposite direction) rather than
      // the type being matched against.
      if ((Match & ExtendedElementVectorType) != 0) {
        if ((IEltTy->getBitWidth() & 1) != 0) {
          CheckFailed(IntrinsicParam(ArgNo, NumRets) + " vector "
                      "element bit-width is odd.", F);
          return false;
        }
        Ty = VectorType::getTruncatedElementVectorType(VTy);
      } else
        Ty = VectorType::getExtendedElementVectorType(VTy);
      Match &= ~(ExtendedElementVectorType | TruncatedElementVectorType);
    }

    if (Match <= static_cast<int>(NumRets - 1)) {
      if (ST)
        RetTy = ST->getElementType(Match);

      if (Ty != RetTy) {
        CheckFailed(IntrinsicParam(ArgNo, NumRets) + " does not "
                    "match return type.", F);
        return false;
      }
    } else {
      if (Ty != FTy->getParamType(Match - 1)) {
        CheckFailed(IntrinsicParam(ArgNo, NumRets) + " does not "
                    "match parameter %" + utostr(Match - 1) + ".", F);
        return false;
      }
    }
  } else if (VT == MVT::iAny) {
    if (!EltTy->isInteger()) {
      CheckFailed(IntrinsicParam(ArgNo, NumRets) + " is not "
                  "an integer type.", F);
      return false;
    }

    unsigned GotBits = cast<IntegerType>(EltTy)->getBitWidth();
    Suffix += ".";

    if (EltTy != Ty)
      Suffix += "v" + utostr(NumElts);

    Suffix += "i" + utostr(GotBits);;

    // Check some constraints on various intrinsics.
    switch (ID) {
    default: break; // Not everything needs to be checked.
    case Intrinsic::bswap:
      if (GotBits < 16 || GotBits % 16 != 0) {
        CheckFailed("Intrinsic requires even byte width argument", F);
        return false;
      }
      break;
    }
  } else if (VT == MVT::fAny) {
    if (!EltTy->isFloatingPoint()) {
      CheckFailed(IntrinsicParam(ArgNo, NumRets) + " is not "
                  "a floating-point type.", F);
      return false;
    }

    Suffix += ".";

    if (EltTy != Ty)
      Suffix += "v" + utostr(NumElts);

    Suffix += MVT::getMVT(EltTy).getMVTString();
  } else if (VT == MVT::iPTR) {
    if (!isa<PointerType>(Ty)) {
      CheckFailed(IntrinsicParam(ArgNo, NumRets) + " is not a "
                  "pointer and a pointer is required.", F);
      return false;
    }
  } else if (VT == MVT::iPTRAny) {
    // Outside of TableGen, we don't distinguish iPTRAny (to any address space)
    // and iPTR. In the verifier, we can not distinguish which case we have so
    // allow either case to be legal.
    if (const PointerType* PTyp = dyn_cast<PointerType>(Ty)) {
      Suffix += ".p" + utostr(PTyp->getAddressSpace()) + 
        MVT::getMVT(PTyp->getElementType()).getMVTString();
    } else {
      CheckFailed(IntrinsicParam(ArgNo, NumRets) + " is not a "
                  "pointer and a pointer is required.", F);
      return false;
    }
  } else if (MVT((MVT::SimpleValueType)VT).isVector()) {
    MVT VVT = MVT((MVT::SimpleValueType)VT);

    // If this is a vector argument, verify the number and type of elements.
    if (VVT.getVectorElementType() != MVT::getMVT(EltTy)) {
      CheckFailed("Intrinsic prototype has incorrect vector element type!", F);
      return false;
    }

    if (VVT.getVectorNumElements() != NumElts) {
      CheckFailed("Intrinsic prototype has incorrect number of "
                  "vector elements!", F);
      return false;
    }
  } else if (MVT((MVT::SimpleValueType)VT).getTypeForMVT() != EltTy) {
    CheckFailed(IntrinsicParam(ArgNo, NumRets) + " is wrong!", F);
    return false;
  } else if (EltTy != Ty) {
    CheckFailed(IntrinsicParam(ArgNo, NumRets) + " is a vector "
                "and a scalar is required.", F);
    return false;
  }

  return true;
}

/// VerifyIntrinsicPrototype - TableGen emits calls to this function into
/// Intrinsics.gen.  This implements a little state machine that verifies the
/// prototype of intrinsics.
void Verifier::VerifyIntrinsicPrototype(Intrinsic::ID ID, Function *F,
                                        unsigned RetNum,
                                        unsigned ParamNum, ...) {
  va_list VA;
  va_start(VA, ParamNum);
  const FunctionType *FTy = F->getFunctionType();
  
  // For overloaded intrinsics, the Suffix of the function name must match the
  // types of the arguments. This variable keeps track of the expected
  // suffix, to be checked at the end.
  std::string Suffix;

  if (FTy->getNumParams() + FTy->isVarArg() != ParamNum) {
    CheckFailed("Intrinsic prototype has incorrect number of arguments!", F);
    return;
  }

  const Type *Ty = FTy->getReturnType();
  const StructType *ST = dyn_cast<StructType>(Ty);

  // Verify the return types.
  if (ST && ST->getNumElements() != RetNum) {
    CheckFailed("Intrinsic prototype has incorrect number of return types!", F);
    return;
  }

  for (unsigned ArgNo = 0; ArgNo < RetNum; ++ArgNo) {
    int VT = va_arg(VA, int); // An MVT::SimpleValueType when non-negative.

    if (ST) Ty = ST->getElementType(ArgNo);

    if (!PerformTypeCheck(ID, F, Ty, VT, ArgNo, Suffix))
      break;
  }

  // Verify the parameter types.
  for (unsigned ArgNo = 0; ArgNo < ParamNum; ++ArgNo) {
    int VT = va_arg(VA, int); // An MVT::SimpleValueType when non-negative.

    if (VT == MVT::isVoid && ArgNo > 0) {
      if (!FTy->isVarArg())
        CheckFailed("Intrinsic prototype has no '...'!", F);
      break;
    }

    if (!PerformTypeCheck(ID, F, FTy->getParamType(ArgNo), VT, ArgNo + RetNum,
                          Suffix))
      break;
  }

  va_end(VA);

  // For intrinsics without pointer arguments, if we computed a Suffix then the
  // intrinsic is overloaded and we need to make sure that the name of the
  // function is correct. We add the suffix to the name of the intrinsic and
  // compare against the given function name. If they are not the same, the
  // function name is invalid. This ensures that overloading of intrinsics
  // uses a sane and consistent naming convention.  Note that intrinsics with
  // pointer argument may or may not be overloaded so we will check assuming it
  // has a suffix and not.
  if (!Suffix.empty()) {
    std::string Name(Intrinsic::getName(ID));
    if (Name + Suffix != F->getName()) {
      CheckFailed("Overloaded intrinsic has incorrect suffix: '" +
                  F->getName().substr(Name.length()) + "'. It should be '" +
                  Suffix + "'", F);
    }
  }

  // Check parameter attributes.
  Assert1(F->getAttributes() == Intrinsic::getAttributes(ID),
          "Intrinsic has wrong parameter attributes!", F);
}


//===----------------------------------------------------------------------===//
//  Implement the public interfaces to this file...
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createVerifierPass(VerifierFailureAction action) {
  return new Verifier(action);
}


// verifyFunction - Create
bool llvm::verifyFunction(const Function &f, VerifierFailureAction action) {
  Function &F = const_cast<Function&>(f);
  assert(!F.isDeclaration() && "Cannot verify external functions");

  ExistingModuleProvider MP(F.getParent());
  FunctionPassManager FPM(&MP);
  Verifier *V = new Verifier(action);
  FPM.add(V);
  FPM.run(F);
  MP.releaseModule();
  return V->Broken;
}

/// verifyModule - Check a module for errors, printing messages on stderr.
/// Return true if the module is corrupt.
///
bool llvm::verifyModule(const Module &M, VerifierFailureAction action,
                        std::string *ErrorInfo) {
  PassManager PM;
  Verifier *V = new Verifier(action);
  PM.add(V);
  PM.run(const_cast<Module&>(M));
  
  if (ErrorInfo && V->Broken)
    *ErrorInfo = V->msgs.str();
  return V->Broken;
}

// vim: sw=2
