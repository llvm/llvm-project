//===- LevelRaise.cpp - Code to change LLVM to higher level ---------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the 'raising' part of the LevelChange API.  This is
// useful because, in general, it makes the LLVM code terser and easier to
// analyze.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "TransformInternals.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include <algorithm>
using namespace llvm;

// StartInst - This enables the -raise-start-inst=foo option to cause the level
// raising pass to start at instruction "foo", which is immensely useful for
// debugging!
//
static cl::opt<std::string>
StartInst("raise-start-inst", cl::Hidden, cl::value_desc("inst name"),
       cl::desc("Start raise pass at the instruction with the specified name"));

static Statistic<>
NumLoadStorePeepholes("raise", "Number of load/store peepholes");

static Statistic<> 
NumGEPInstFormed("raise", "Number of other getelementptr's formed");

static Statistic<>
NumExprTreesConv("raise", "Number of expression trees converted");

static Statistic<>
NumCastOfCast("raise", "Number of cast-of-self removed");

static Statistic<>
NumDCEorCP("raise", "Number of insts DCEd or constprop'd");

static Statistic<>
NumVarargCallChanges("raise", "Number of vararg call peepholes");

#define PRINT_PEEPHOLE(ID, NUM, I)            \
  DEBUG(std::cerr << "Inst P/H " << ID << "[" << NUM << "] " << I)

#define PRINT_PEEPHOLE1(ID, I1) do { PRINT_PEEPHOLE(ID, 0, I1); } while (0)
#define PRINT_PEEPHOLE2(ID, I1, I2) \
  do { PRINT_PEEPHOLE(ID, 0, I1); PRINT_PEEPHOLE(ID, 1, I2); } while (0)
#define PRINT_PEEPHOLE3(ID, I1, I2, I3) \
  do { PRINT_PEEPHOLE(ID, 0, I1); PRINT_PEEPHOLE(ID, 1, I2); \
       PRINT_PEEPHOLE(ID, 2, I3); } while (0)
#define PRINT_PEEPHOLE4(ID, I1, I2, I3, I4) \
  do { PRINT_PEEPHOLE(ID, 0, I1); PRINT_PEEPHOLE(ID, 1, I2); \
       PRINT_PEEPHOLE(ID, 2, I3); PRINT_PEEPHOLE(ID, 3, I4); } while (0)

namespace {
  struct RPR : public FunctionPass {
    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<TargetData>();
    }

  private:
    bool DoRaisePass(Function &F);
    bool PeepholeOptimize(BasicBlock *BB, BasicBlock::iterator &BI);
  };

  RegisterOpt<RPR> X("raise", "Raise Pointer References");
}


Pass *llvm::createRaisePointerReferencesPass() {
  return new RPR();
}


// isReinterpretingCast - Return true if the cast instruction specified will
// cause the operand to be "reinterpreted".  A value is reinterpreted if the
// cast instruction would cause the underlying bits to change.
//
static inline bool isReinterpretingCast(const CastInst *CI) {
  return!CI->getOperand(0)->getType()->isLosslesslyConvertibleTo(CI->getType());
}


// Peephole optimize the following instructions:
// %t1 = cast ? to x *
// %t2 = add x * %SP, %t1              ;; Constant must be 2nd operand
//
// Into: %t3 = getelementptr {<...>} * %SP, <element indices>
//       %t2 = cast <eltype> * %t3 to {<...>}*
//
static bool HandleCastToPointer(BasicBlock::iterator BI,
                                const PointerType *DestPTy,
                                const TargetData &TD) {
  CastInst &CI = cast<CastInst>(*BI);
  if (CI.use_empty()) return false;

  // Scan all of the uses, looking for any uses that are not add or sub
  // instructions.  If we have non-adds, do not make this transformation.
  //
  bool HasSubUse = false;  // Keep track of any subtracts...
  for (Value::use_iterator I = CI.use_begin(), E = CI.use_end();
       I != E; ++I)
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(*I)) {
      if ((BO->getOpcode() != Instruction::Add &&
           BO->getOpcode() != Instruction::Sub) ||
          // Avoid add sbyte* %X, %X cases...
          BO->getOperand(0) == BO->getOperand(1))
        return false;
      else
        HasSubUse |= BO->getOpcode() == Instruction::Sub;
    } else {
      return false;
    }

  std::vector<Value*> Indices;
  Value *Src = CI.getOperand(0);
  const Type *Result = ConvertibleToGEP(DestPTy, Src, Indices, TD, &BI);
  if (Result == 0) return false;  // Not convertible...

  // Cannot handle subtracts if there is more than one index required...
  if (HasSubUse && Indices.size() != 1) return false;

  PRINT_PEEPHOLE2("cast-add-to-gep:in", *Src, CI);

  // If we have a getelementptr capability... transform all of the 
  // add instruction uses into getelementptr's.
  while (!CI.use_empty()) {
    BinaryOperator *I = cast<BinaryOperator>(*CI.use_begin());
    assert((I->getOpcode() == Instruction::Add ||
            I->getOpcode() == Instruction::Sub) && 
           "Use is not a valid add instruction!");
    
    // Get the value added to the cast result pointer...
    Value *OtherPtr = I->getOperand((I->getOperand(0) == &CI) ? 1 : 0);

    Instruction *GEP = new GetElementPtrInst(OtherPtr, Indices, I->getName());
    PRINT_PEEPHOLE1("cast-add-to-gep:i", *I);

    // If the instruction is actually a subtract, we are guaranteed to only have
    // one index (from code above), so we just need to negate the pointer index
    // long value.
    if (I->getOpcode() == Instruction::Sub) {
      Instruction *Neg = BinaryOperator::createNeg(GEP->getOperand(1), 
                                       GEP->getOperand(1)->getName()+".neg", I);
      GEP->setOperand(1, Neg);
    }

    if (GEP->getType() == I->getType()) {
      // Replace the old add instruction with the shiny new GEP inst
      ReplaceInstWithInst(I, GEP);
    } else {
      // If the type produced by the gep instruction differs from the original
      // add instruction type, insert a cast now.
      //

      // Insert the GEP instruction before the old add instruction...
      I->getParent()->getInstList().insert(I, GEP);

      PRINT_PEEPHOLE1("cast-add-to-gep:o", *GEP);
      GEP = new CastInst(GEP, I->getType());

      // Replace the old add instruction with the shiny new GEP inst
      ReplaceInstWithInst(I, GEP);
    }

    PRINT_PEEPHOLE1("cast-add-to-gep:o", *GEP);
  }
  return true;
}

// Peephole optimize the following instructions:
// %t1 = cast ulong <const int> to {<...>} *
// %t2 = add {<...>} * %SP, %t1              ;; Constant must be 2nd operand
//
//    or
// %t1 = cast {<...>}* %SP to int*
// %t5 = cast ulong <const int> to int*
// %t2 = add int* %t1, %t5                   ;; int is same size as field
//
// Into: %t3 = getelementptr {<...>} * %SP, <element indices>
//       %t2 = cast <eltype> * %t3 to {<...>}*
//
static bool PeepholeOptimizeAddCast(BasicBlock *BB, BasicBlock::iterator &BI,
                                    Value *AddOp1, CastInst *AddOp2,
                                    const TargetData &TD) {
  const CompositeType *CompTy;
  Value *OffsetVal = AddOp2->getOperand(0);
  Value *SrcPtr = 0;  // Of type pointer to struct...

  if ((CompTy = getPointedToComposite(AddOp1->getType()))) {
    SrcPtr = AddOp1;                      // Handle the first case...
  } else if (CastInst *AddOp1c = dyn_cast<CastInst>(AddOp1)) {
    SrcPtr = AddOp1c->getOperand(0);      // Handle the second case...
    CompTy = getPointedToComposite(SrcPtr->getType());
  }

  // Only proceed if we have detected all of our conditions successfully...
  if (!CompTy || !SrcPtr || !OffsetVal->getType()->isInteger())
    return false;

  std::vector<Value*> Indices;
  if (!ConvertibleToGEP(SrcPtr->getType(), OffsetVal, Indices, TD, &BI))
    return false;  // Not convertible... perhaps next time

  if (getPointedToComposite(AddOp1->getType())) {  // case 1
    PRINT_PEEPHOLE2("add-to-gep1:in", *AddOp2, *BI);
  } else {
    PRINT_PEEPHOLE3("add-to-gep2:in", *AddOp1, *AddOp2, *BI);
  }

  GetElementPtrInst *GEP = new GetElementPtrInst(SrcPtr, Indices,
                                                 AddOp2->getName(), BI);

  Instruction *NCI = new CastInst(GEP, AddOp1->getType());
  ReplaceInstWithInst(BB->getInstList(), BI, NCI);
  PRINT_PEEPHOLE2("add-to-gep:out", *GEP, *NCI);
  return true;
}

bool RPR::PeepholeOptimize(BasicBlock *BB, BasicBlock::iterator &BI) {
  Instruction *I = BI;
  const TargetData &TD = getAnalysis<TargetData>();

  if (CastInst *CI = dyn_cast<CastInst>(I)) {
    Value       *Src    = CI->getOperand(0);
    Instruction *SrcI   = dyn_cast<Instruction>(Src); // Nonnull if instr source
    const Type  *DestTy = CI->getType();

    // Peephole optimize the following instruction:
    // %V2 = cast <ty> %V to <ty>
    //
    // Into: <nothing>
    //
    if (DestTy == Src->getType()) {   // Check for a cast to same type as src!!
      PRINT_PEEPHOLE1("cast-of-self-ty", *CI);
      CI->replaceAllUsesWith(Src);
      if (!Src->hasName() && CI->hasName()) {
        std::string Name = CI->getName();
        CI->setName("");
        Src->setName(Name, &BB->getParent()->getSymbolTable());
      }

      // DCE the instruction now, to avoid having the iterative version of DCE
      // have to worry about it.
      //
      BI = BB->getInstList().erase(BI);

      ++NumCastOfCast;
      return true;
    }

    // Check to see if it's a cast of an instruction that does not depend on the
    // specific type of the operands to do it's job.
    if (!isReinterpretingCast(CI)) {
      ValueTypeCache ConvertedTypes;

      // Check to see if we can convert the source of the cast to match the
      // destination type of the cast...
      //
      ConvertedTypes[CI] = CI->getType();  // Make sure the cast doesn't change
      if (ExpressionConvertibleToType(Src, DestTy, ConvertedTypes, TD)) {
        PRINT_PEEPHOLE3("CAST-SRC-EXPR-CONV:in ", *Src, *CI, *BB->getParent());
          
        DEBUG(std::cerr << "\nCONVERTING SRC EXPR TYPE:\n");
        { // ValueMap must be destroyed before function verified!
          ValueMapCache ValueMap;
          Value *E = ConvertExpressionToType(Src, DestTy, ValueMap, TD);

          if (Constant *CPV = dyn_cast<Constant>(E))
            CI->replaceAllUsesWith(CPV);
          
          PRINT_PEEPHOLE1("CAST-SRC-EXPR-CONV:out", *E);
          DEBUG(std::cerr << "DONE CONVERTING SRC EXPR TYPE: \n"
                          << *BB->getParent());
        }

        BI = BB->begin();  // Rescan basic block.  BI might be invalidated.
        ++NumExprTreesConv;
        return true;
      }

      // Check to see if we can convert the users of the cast value to match the
      // source type of the cast...
      //
      ConvertedTypes.clear();
      // Make sure the source doesn't change type
      ConvertedTypes[Src] = Src->getType();
      if (ValueConvertibleToType(CI, Src->getType(), ConvertedTypes, TD)) {
        //PRINT_PEEPHOLE3("CAST-DEST-EXPR-CONV:in ", *Src, *CI,
        //                *BB->getParent());

        DEBUG(std::cerr << "\nCONVERTING EXPR TYPE:\n");
        { // ValueMap must be destroyed before function verified!
          ValueMapCache ValueMap;
          ConvertValueToNewType(CI, Src, ValueMap, TD);  // This will delete CI!
        }

        PRINT_PEEPHOLE1("CAST-DEST-EXPR-CONV:out", *Src);
        DEBUG(std::cerr << "DONE CONVERTING EXPR TYPE: \n\n" <<
              *BB->getParent());

        BI = BB->begin();  // Rescan basic block.  BI might be invalidated.
        ++NumExprTreesConv;
        return true;
      }
    }

    // Otherwise find out it this cast is a cast to a pointer type, which is
    // then added to some other pointer, then loaded or stored through.  If
    // so, convert the add into a getelementptr instruction...
    //
    if (const PointerType *DestPTy = dyn_cast<PointerType>(DestTy)) {
      if (HandleCastToPointer(BI, DestPTy, TD)) {
        BI = BB->begin();  // Rescan basic block.  BI might be invalidated.
        ++NumGEPInstFormed;
        return true;
      }
    }

    // Check to see if we are casting from a structure pointer to a pointer to
    // the first element of the structure... to avoid munching other peepholes,
    // we only let this happen if there are no add uses of the cast.
    //
    // Peephole optimize the following instructions:
    // %t1 = cast {<...>} * %StructPtr to <ty> *
    //
    // Into: %t2 = getelementptr {<...>} * %StructPtr, <0, 0, 0, ...>
    //       %t1 = cast <eltype> * %t1 to <ty> *
    //
    if (const CompositeType *CTy = getPointedToComposite(Src->getType()))
      if (const PointerType *DestPTy = dyn_cast<PointerType>(DestTy)) {

        // Loop over uses of the cast, checking for add instructions.  If an add
        // exists, this is probably a part of a more complex GEP, so we don't
        // want to mess around with the cast.
        //
        bool HasAddUse = false;
        for (Value::use_iterator I = CI->use_begin(), E = CI->use_end();
             I != E; ++I)
          if (isa<Instruction>(*I) &&
              cast<Instruction>(*I)->getOpcode() == Instruction::Add) {
            HasAddUse = true; break;
          }

        // If it doesn't have an add use, check to see if the dest type is
        // losslessly convertible to one of the types in the start of the struct
        // type.
        //
        if (!HasAddUse) {
          const Type *DestPointedTy = DestPTy->getElementType();
          unsigned Depth = 1;
          const CompositeType *CurCTy = CTy;
          const Type *ElTy = 0;

          // Build the index vector, full of all zeros
          std::vector<Value*> Indices;

          Indices.push_back(Constant::getNullValue(Type::UIntTy));
          while (CurCTy && !isa<PointerType>(CurCTy)) {
            if (const StructType *CurSTy = dyn_cast<StructType>(CurCTy)) {
              // Check for a zero element struct type... if we have one, bail.
              if (CurSTy->getNumElements() == 0) break;
            
              // Grab the first element of the struct type, which must lie at
              // offset zero in the struct.
              //
              ElTy = CurSTy->getElementType(0);
            } else {
              ElTy = cast<ArrayType>(CurCTy)->getElementType();
            }

            // Insert a zero to index through this type...
            Indices.push_back(Constant::getNullValue(Type::UIntTy));

            // Did we find what we're looking for?
            if (ElTy->isLosslesslyConvertibleTo(DestPointedTy)) break;
            
            // Nope, go a level deeper.
            ++Depth;
            CurCTy = dyn_cast<CompositeType>(ElTy);
            ElTy = 0;
          }
          
          // Did we find what we were looking for? If so, do the transformation
          if (ElTy) {
            PRINT_PEEPHOLE1("cast-for-first:in", *CI);

            std::string Name = CI->getName(); CI->setName("");

            // Insert the new T cast instruction... stealing old T's name
            GetElementPtrInst *GEP = new GetElementPtrInst(Src, Indices,
                                                           Name, BI);

            // Make the old cast instruction reference the new GEP instead of
            // the old src value.
            //
            CI->setOperand(0, GEP);
            
            PRINT_PEEPHOLE2("cast-for-first:out", *GEP, *CI);
            ++NumGEPInstFormed;
            return true;
          }
        }
      }

  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Value *Val     = SI->getOperand(0);
    Value *Pointer = SI->getPointerOperand();
    
    // Peephole optimize the following instructions:
    // %t = cast <T1>* %P to <T2> * ;; If T1 is losslessly convertible to T2
    // store <T2> %V, <T2>* %t
    //
    // Into: 
    // %t = cast <T2> %V to <T1>
    // store <T1> %t2, <T1>* %P
    //
    // Note: This is not taken care of by expr conversion because there might
    // not be a cast available for the store to convert the incoming value of.
    // This code is basically here to make sure that pointers don't have casts
    // if possible.
    //
    if (CastInst *CI = dyn_cast<CastInst>(Pointer))
      if (Value *CastSrc = CI->getOperand(0)) // CSPT = CastSrcPointerType
        if (const PointerType *CSPT = dyn_cast<PointerType>(CastSrc->getType()))
          // convertible types?
          if (Val->getType()->isLosslesslyConvertibleTo(CSPT->getElementType())) {
            PRINT_PEEPHOLE3("st-src-cast:in ", *Pointer, *Val, *SI);

            // Insert the new T cast instruction... stealing old T's name
            std::string Name(CI->getName()); CI->setName("");
            CastInst *NCI = new CastInst(Val, CSPT->getElementType(),
                                         Name, BI);

            // Replace the old store with a new one!
            ReplaceInstWithInst(BB->getInstList(), BI,
                                SI = new StoreInst(NCI, CastSrc));
            PRINT_PEEPHOLE3("st-src-cast:out", *NCI, *CastSrc, *SI);
            ++NumLoadStorePeepholes;
            return true;
          }

  } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Value *Pointer = LI->getOperand(0);
    const Type *PtrElType =
      cast<PointerType>(Pointer->getType())->getElementType();
    
    // Peephole optimize the following instructions:
    // %Val = cast <T1>* to <T2>*    ;; If T1 is losslessly convertible to T2
    // %t = load <T2>* %P
    //
    // Into: 
    // %t = load <T1>* %P
    // %Val = cast <T1> to <T2>
    //
    // Note: This is not taken care of by expr conversion because there might
    // not be a cast available for the store to convert the incoming value of.
    // This code is basically here to make sure that pointers don't have casts
    // if possible.
    //
    if (CastInst *CI = dyn_cast<CastInst>(Pointer))
      if (Value *CastSrc = CI->getOperand(0)) // CSPT = CastSrcPointerType
        if (const PointerType *CSPT = dyn_cast<PointerType>(CastSrc->getType()))
          // convertible types?
          if (PtrElType->isLosslesslyConvertibleTo(CSPT->getElementType())) {
            PRINT_PEEPHOLE2("load-src-cast:in ", *Pointer, *LI);

            // Create the new load instruction... loading the pre-casted value
            LoadInst *NewLI = new LoadInst(CastSrc, LI->getName(), BI);
            
            // Insert the new T cast instruction... stealing old T's name
            CastInst *NCI = new CastInst(NewLI, LI->getType(), CI->getName());

            // Replace the old store with a new one!
            ReplaceInstWithInst(BB->getInstList(), BI, NCI);
            PRINT_PEEPHOLE3("load-src-cast:out", *NCI, *CastSrc, *NewLI);
            ++NumLoadStorePeepholes;
            return true;
          }

  } else if (I->getOpcode() == Instruction::Add &&
             isa<CastInst>(I->getOperand(1))) {

    if (PeepholeOptimizeAddCast(BB, BI, I->getOperand(0),
                                cast<CastInst>(I->getOperand(1)), TD)) {
      ++NumGEPInstFormed;
      return true;
    }
  } else if (CallInst *CI = dyn_cast<CallInst>(I)) {
    // If we have a call with all varargs arguments, convert the call to use the
    // actual argument types present...
    //
    const PointerType *PTy = cast<PointerType>(CI->getCalledValue()->getType());
    const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());

    // Is the call to a vararg variable with no real parameters?
    if (FTy->isVarArg() && FTy->getNumParams() == 0 &&
        !CI->getCalledFunction()) {
      // If so, insert a new cast instruction, casting it to a function type
      // that matches the current arguments...
      //
      std::vector<const Type *> Params;  // Parameter types...
      for (unsigned i = 1, e = CI->getNumOperands(); i != e; ++i)
        Params.push_back(CI->getOperand(i)->getType());

      FunctionType *NewFT = FunctionType::get(FTy->getReturnType(),
                                              Params, false);
      PointerType *NewPFunTy = PointerType::get(NewFT);

      // Create a new cast, inserting it right before the function call...
      Value *NewCast;
      Constant *ConstantCallSrc = 0;
      if (Constant *CS = dyn_cast<Constant>(CI->getCalledValue()))
        ConstantCallSrc = CS;

      if (ConstantCallSrc)
        NewCast = ConstantExpr::getCast(ConstantCallSrc, NewPFunTy);
      else
        NewCast = new CastInst(CI->getCalledValue(), NewPFunTy,
                               CI->getCalledValue()->getName()+"_c",CI);

      // Create a new call instruction...
      CallInst *NewCall = new CallInst(NewCast,
                           std::vector<Value*>(CI->op_begin()+1, CI->op_end()));
      ++BI;
      ReplaceInstWithInst(CI, NewCall);
      
      ++NumVarargCallChanges;
      return true;
    }

  }

  return false;
}




bool RPR::DoRaisePass(Function &F) {
  bool Changed = false;
  for (Function::iterator BB = F.begin(), BBE = F.end(); BB != BBE; ++BB)
    for (BasicBlock::iterator BI = BB->begin(); BI != BB->end();) {
      DEBUG(std::cerr << "LevelRaising: " << *BI);
      if (dceInstruction(BI) || doConstantPropagation(BI)) {
        Changed = true; 
        ++NumDCEorCP;
        DEBUG(std::cerr << "***\t\t^^-- Dead code eliminated!\n");
      } else if (PeepholeOptimize(BB, BI)) {
        Changed = true;
      } else {
        ++BI;
      }
    }

  return Changed;
}


// runOnFunction - Raise a function representation to a higher level.
bool RPR::runOnFunction(Function &F) {
  DEBUG(std::cerr << "\n\n\nStarting to work on Function '" << F.getName()
                  << "'\n");

  // Insert casts for all incoming pointer pointer values that are treated as
  // arrays...
  //
  bool Changed = false, LocalChange;

  // If the StartInst option was specified, then Peephole optimize that
  // instruction first if it occurs in this function.
  //
  if (!StartInst.empty()) {
    for (Function::iterator BB = F.begin(), BBE = F.end(); BB != BBE; ++BB)
      for (BasicBlock::iterator BI = BB->begin(); BI != BB->end(); ++BI)
        if (BI->getName() == StartInst) {
          bool SavedDebug = DebugFlag;  // Save the DEBUG() controlling flag.
          DebugFlag = true;             // Turn on DEBUG's
          Changed |= PeepholeOptimize(BB, BI);
          DebugFlag = SavedDebug;       // Restore DebugFlag to previous state
        }
  }

  do {
    DEBUG(std::cerr << "Looping: \n" << F);

    // Iterate over the function, refining it, until it converges on a stable
    // state
    LocalChange = false;
    while (DoRaisePass(F)) LocalChange = true;
    Changed |= LocalChange;

  } while (LocalChange);

  return Changed;
}

