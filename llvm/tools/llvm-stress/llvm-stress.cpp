//===-- llvm-stress.cpp - Generate random LL files to stress-test LLVM ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that generates random .ll files to stress-test
// different components in LLVM.
//
//===----------------------------------------------------------------------===//
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Constants.h"
#include "llvm/Instruction.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/PassNameParser.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ToolOutputFile.h"
#include <memory>
#include <sstream>
#include <set>
#include <vector>
#include <algorithm>
using namespace llvm;

static cl::opt<unsigned> SeedCL("seed",
  cl::desc("Seed used for randomness"), cl::init(0));
static cl::opt<unsigned> SizeCL("size",
  cl::desc("The estimated size of the generated function (# of instrs)"),
  cl::init(100));
static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool> GenHalfFloat("generate-half-float",
  cl::desc("Generate half-length floating-point values"), cl::init(false));
static cl::opt<bool> GenX86FP80("generate-x86-fp80",
  cl::desc("Generate 80-bit X86 floating-point values"), cl::init(false));
static cl::opt<bool> GenFP128("generate-fp128",
  cl::desc("Generate 128-bit floating-point values"), cl::init(false));
static cl::opt<bool> GenPPCFP128("generate-ppc-fp128",
  cl::desc("Generate 128-bit PPC floating-point values"), cl::init(false));
static cl::opt<bool> GenX86MMX("generate-x86-mmx",
  cl::desc("Generate X86 MMX floating-point values"), cl::init(false));

/// A utility class to provide a pseudo-random number generator which is
/// the same across all platforms. This is somewhat close to the libc
/// implementation. Note: This is not a cryptographically secure pseudorandom
/// number generator.
class Random {
public:
  /// C'tor
  Random(unsigned _seed):Seed(_seed) {}

  /// Return a random integer, up to a
  /// maximum of 2**19 - 1.
  uint32_t Rand() {
    uint32_t Val = Seed + 0x000b07a1;
    Seed = (Val * 0x3c7c0ac1);
    // Only lowest 19 bits are random-ish.
    return Seed & 0x7ffff;
  }

  /// Return a random 32 bit integer.
  uint32_t Rand32() {
    uint32_t Val = Rand();
    Val &= 0xffff;
    return Val | (Rand() << 16);
  }

  /// Return a random 64 bit integer.
  uint64_t Rand64() {
    uint64_t Val = Rand32();
    return Val | (uint64_t(Rand32()) << 32);
  }

  /// Rand operator for STL algorithms.
  ptrdiff_t operator()(ptrdiff_t y) {
    return  Rand64() % y;
  }

private:
  unsigned Seed;
};

/// Generate an empty function with a default argument list.
Function *GenEmptyFunction(Module *M) {
  // Type Definitions
  std::vector<Type*> ArgsTy;
  // Define a few arguments
  LLVMContext &Context = M->getContext();
  ArgsTy.push_back(PointerType::get(IntegerType::getInt8Ty(Context), 0));
  ArgsTy.push_back(PointerType::get(IntegerType::getInt32Ty(Context), 0));
  ArgsTy.push_back(PointerType::get(IntegerType::getInt64Ty(Context), 0));
  ArgsTy.push_back(IntegerType::getInt32Ty(Context));
  ArgsTy.push_back(IntegerType::getInt64Ty(Context));
  ArgsTy.push_back(IntegerType::getInt8Ty(Context));

  FunctionType *FuncTy = FunctionType::get(Type::getVoidTy(Context), ArgsTy, 0);
  // Pick a unique name to describe the input parameters
  std::stringstream ss;
  ss<<"autogen_SD"<<SeedCL;
  Function *Func = Function::Create(FuncTy, GlobalValue::ExternalLinkage,
                                    ss.str(), M);

  Func->setCallingConv(CallingConv::C);
  return Func;
}

/// A base class, implementing utilities needed for
/// modifying and adding new random instructions.
struct Modifier {
  /// Used to store the randomly generated values.
  typedef std::vector<Value*> PieceTable;

public:
  /// C'tor
  Modifier(BasicBlock *Block, PieceTable *PT, Random *R):
    BB(Block),PT(PT),Ran(R),Context(BB->getContext()) {}

  /// virtual D'tor to silence warnings.
  virtual ~Modifier() {}

  /// Add a new instruction.
  virtual void Act() = 0;
  /// Add N new instructions,
  virtual void ActN(unsigned n) {
    for (unsigned i=0; i<n; ++i)
      Act();
  }

protected:
  /// Return a random value from the list of known values.
  Value *getRandomVal() {
    assert(PT->size());
    return PT->at(Ran->Rand() % PT->size());
  }

  Constant *getRandomConstant(Type *Tp) {
    if (Tp->isIntegerTy()) {
      if (Ran->Rand() & 1)
        return ConstantInt::getAllOnesValue(Tp);
      return ConstantInt::getNullValue(Tp);
    } else if (Tp->isFloatingPointTy()) {
      if (Ran->Rand() & 1)
        return ConstantFP::getAllOnesValue(Tp);
      return ConstantFP::getNullValue(Tp);
    }
    return UndefValue::get(Tp);
  }

  /// Return a random value with a known type.
  Value *getRandomValue(Type *Tp) {
    unsigned index = Ran->Rand();
    for (unsigned i=0; i<PT->size(); ++i) {
      Value *V = PT->at((index + i) % PT->size());
      if (V->getType() == Tp)
        return V;
    }

    // If the requested type was not found, generate a constant value.
    if (Tp->isIntegerTy()) {
      if (Ran->Rand() & 1)
        return ConstantInt::getAllOnesValue(Tp);
      return ConstantInt::getNullValue(Tp);
    } else if (Tp->isFloatingPointTy()) {
      if (Ran->Rand() & 1)
        return ConstantFP::getAllOnesValue(Tp);
      return ConstantFP::getNullValue(Tp);
    } else if (Tp->isVectorTy()) {
      VectorType *VTp = cast<VectorType>(Tp);

      std::vector<Constant*> TempValues;
      TempValues.reserve(VTp->getNumElements());
      for (unsigned i = 0; i < VTp->getNumElements(); ++i)
        TempValues.push_back(getRandomConstant(VTp->getScalarType()));

      ArrayRef<Constant*> VectorValue(TempValues);
      return ConstantVector::get(VectorValue);
    }

    return UndefValue::get(Tp);
  }

  /// Return a random value of any pointer type.
  Value *getRandomPointerValue() {
    unsigned index = Ran->Rand();
    for (unsigned i=0; i<PT->size(); ++i) {
      Value *V = PT->at((index + i) % PT->size());
      if (V->getType()->isPointerTy())
        return V;
    }
    return UndefValue::get(pickPointerType());
  }

  /// Return a random value of any vector type.
  Value *getRandomVectorValue() {
    unsigned index = Ran->Rand();
    for (unsigned i=0; i<PT->size(); ++i) {
      Value *V = PT->at((index + i) % PT->size());
      if (V->getType()->isVectorTy())
        return V;
    }
    return UndefValue::get(pickVectorType());
  }

  /// Pick a random type.
  Type *pickType() {
    return (Ran->Rand() & 1 ? pickVectorType() : pickScalarType());
  }

  /// Pick a random pointer type.
  Type *pickPointerType() {
    Type *Ty = pickType();
    return PointerType::get(Ty, 0);
  }

  /// Pick a random vector type.
  Type *pickVectorType(unsigned len = (unsigned)-1) {
    // Pick a random vector width in the range 2**0 to 2**4.
    // by adding two randoms we are generating a normal-like distribution
    // around 2**3.
    unsigned width = 1<<((Ran->Rand() % 3) + (Ran->Rand() % 3));
    Type *Ty;

    // Vectors of x86mmx are illegal; keep trying till we get something else.
    do {
      Ty = pickScalarType();
    } while (Ty->isX86_MMXTy());

    if (len != (unsigned)-1)
      width = len;
    return VectorType::get(Ty, width);
  }

  /// Pick a random scalar type.
  Type *pickScalarType() {
    Type *t = 0;
    do {
      switch (Ran->Rand() % 30) {
      case 0: t = Type::getInt1Ty(Context); break;
      case 1: t = Type::getInt8Ty(Context); break;
      case 2: t = Type::getInt16Ty(Context); break;
      case 3: case 4:
      case 5: t = Type::getFloatTy(Context); break;
      case 6: case 7:
      case 8: t = Type::getDoubleTy(Context); break;
      case 9: case 10:
      case 11: t = Type::getInt32Ty(Context); break;
      case 12: case 13:
      case 14: t = Type::getInt64Ty(Context); break;
      case 15: case 16:
      case 17: if (GenHalfFloat) t = Type::getHalfTy(Context); break;
      case 18: case 19:
      case 20: if (GenX86FP80) t = Type::getX86_FP80Ty(Context); break;
      case 21: case 22:
      case 23: if (GenFP128) t = Type::getFP128Ty(Context); break;
      case 24: case 25:
      case 26: if (GenPPCFP128) t = Type::getPPC_FP128Ty(Context); break;
      case 27: case 28:
      case 29: if (GenX86MMX) t = Type::getX86_MMXTy(Context); break;
      default: llvm_unreachable("Invalid scalar value");
      }
    } while (t == 0);

    return t;
  }

  /// Basic block to populate
  BasicBlock *BB;
  /// Value table
  PieceTable *PT;
  /// Random number generator
  Random *Ran;
  /// Context
  LLVMContext &Context;
};

struct LoadModifier: public Modifier {
  LoadModifier(BasicBlock *BB, PieceTable *PT, Random *R):Modifier(BB, PT, R) {}
  virtual void Act() {
    // Try to use predefined pointers. If non exist, use undef pointer value;
    Value *Ptr = getRandomPointerValue();
    Value *V = new LoadInst(Ptr, "L", BB->getTerminator());
    PT->push_back(V);
  }
};

struct StoreModifier: public Modifier {
  StoreModifier(BasicBlock *BB, PieceTable *PT, Random *R):Modifier(BB, PT, R) {}
  virtual void Act() {
    // Try to use predefined pointers. If non exist, use undef pointer value;
    Value *Ptr = getRandomPointerValue();
    Type  *Tp = Ptr->getType();
    Value *Val = getRandomValue(Tp->getContainedType(0));
    Type  *ValTy = Val->getType();

    // Do not store vectors of i1s because they are unsupported
    // by the codegen.
    if (ValTy->isVectorTy() && ValTy->getScalarSizeInBits() == 1)
      return;

    new StoreInst(Val, Ptr, BB->getTerminator());
  }
};

struct BinModifier: public Modifier {
  BinModifier(BasicBlock *BB, PieceTable *PT, Random *R):Modifier(BB, PT, R) {}

  virtual void Act() {
    Value *Val0 = getRandomVal();
    Value *Val1 = getRandomValue(Val0->getType());

    // Don't handle pointer types.
    if (Val0->getType()->isPointerTy() ||
        Val1->getType()->isPointerTy())
      return;

    // Don't handle i1 types.
    if (Val0->getType()->getScalarSizeInBits() == 1)
      return;


    bool isFloat = Val0->getType()->getScalarType()->isFloatingPointTy();
    Instruction* Term = BB->getTerminator();
    unsigned R = Ran->Rand() % (isFloat ? 7 : 13);
    Instruction::BinaryOps Op;

    switch (R) {
    default: llvm_unreachable("Invalid BinOp");
    case 0:{Op = (isFloat?Instruction::FAdd : Instruction::Add); break; }
    case 1:{Op = (isFloat?Instruction::FSub : Instruction::Sub); break; }
    case 2:{Op = (isFloat?Instruction::FMul : Instruction::Mul); break; }
    case 3:{Op = (isFloat?Instruction::FDiv : Instruction::SDiv); break; }
    case 4:{Op = (isFloat?Instruction::FDiv : Instruction::UDiv); break; }
    case 5:{Op = (isFloat?Instruction::FRem : Instruction::SRem); break; }
    case 6:{Op = (isFloat?Instruction::FRem : Instruction::URem); break; }
    case 7: {Op = Instruction::Shl;  break; }
    case 8: {Op = Instruction::LShr; break; }
    case 9: {Op = Instruction::AShr; break; }
    case 10:{Op = Instruction::And;  break; }
    case 11:{Op = Instruction::Or;   break; }
    case 12:{Op = Instruction::Xor;  break; }
    }

    PT->push_back(BinaryOperator::Create(Op, Val0, Val1, "B", Term));
  }
};

/// Generate constant values.
struct ConstModifier: public Modifier {
  ConstModifier(BasicBlock *BB, PieceTable *PT, Random *R):Modifier(BB, PT, R) {}
  virtual void Act() {
    Type *Ty = pickType();

    if (Ty->isVectorTy()) {
      switch (Ran->Rand() % 2) {
      case 0: if (Ty->getScalarType()->isIntegerTy())
                return PT->push_back(ConstantVector::getAllOnesValue(Ty));
      case 1: if (Ty->getScalarType()->isIntegerTy())
                return PT->push_back(ConstantVector::getNullValue(Ty));
      }
    }

    if (Ty->isFloatingPointTy()) {
      // Generate 128 random bits, the size of the (currently)
      // largest floating-point types.
      uint64_t RandomBits[2];
      for (unsigned i = 0; i < 2; ++i)
        RandomBits[i] = Ran->Rand64();

      APInt RandomInt(Ty->getPrimitiveSizeInBits(), makeArrayRef(RandomBits));

      bool isIEEE = !Ty->isX86_FP80Ty() && !Ty->isPPC_FP128Ty();
      APFloat RandomFloat(RandomInt, isIEEE);

      if (Ran->Rand() & 1)
        return PT->push_back(ConstantFP::getNullValue(Ty));
      return PT->push_back(ConstantFP::get(Ty->getContext(), RandomFloat));
    }

    if (Ty->isIntegerTy()) {
      switch (Ran->Rand() % 7) {
      case 0: if (Ty->isIntegerTy())
                return PT->push_back(ConstantInt::get(Ty,
                  APInt::getAllOnesValue(Ty->getPrimitiveSizeInBits())));
      case 1: if (Ty->isIntegerTy())
                return PT->push_back(ConstantInt::get(Ty,
                  APInt::getNullValue(Ty->getPrimitiveSizeInBits())));
      case 2: case 3: case 4: case 5:
      case 6: if (Ty->isIntegerTy())
                PT->push_back(ConstantInt::get(Ty, Ran->Rand()));
      }
    }

  }
};

struct AllocaModifier: public Modifier {
  AllocaModifier(BasicBlock *BB, PieceTable *PT, Random *R):Modifier(BB, PT, R){}

  virtual void Act() {
    Type *Tp = pickType();
    PT->push_back(new AllocaInst(Tp, "A", BB->getFirstNonPHI()));
  }
};

struct ExtractElementModifier: public Modifier {
  ExtractElementModifier(BasicBlock *BB, PieceTable *PT, Random *R):
    Modifier(BB, PT, R) {}

  virtual void Act() {
    Value *Val0 = getRandomVectorValue();
    Value *V = ExtractElementInst::Create(Val0,
             ConstantInt::get(Type::getInt32Ty(BB->getContext()),
             Ran->Rand() % cast<VectorType>(Val0->getType())->getNumElements()),
             "E", BB->getTerminator());
    return PT->push_back(V);
  }
};

struct ShuffModifier: public Modifier {
  ShuffModifier(BasicBlock *BB, PieceTable *PT, Random *R):Modifier(BB, PT, R) {}
  virtual void Act() {

    Value *Val0 = getRandomVectorValue();
    Value *Val1 = getRandomValue(Val0->getType());

    unsigned Width = cast<VectorType>(Val0->getType())->getNumElements();
    std::vector<Constant*> Idxs;

    Type *I32 = Type::getInt32Ty(BB->getContext());
    for (unsigned i=0; i<Width; ++i) {
      Constant *CI = ConstantInt::get(I32, Ran->Rand() % (Width*2));
      // Pick some undef values.
      if (!(Ran->Rand() % 5))
        CI = UndefValue::get(I32);
      Idxs.push_back(CI);
    }

    Constant *Mask = ConstantVector::get(Idxs);

    Value *V = new ShuffleVectorInst(Val0, Val1, Mask, "Shuff",
                                     BB->getTerminator());
    PT->push_back(V);
  }
};

struct InsertElementModifier: public Modifier {
  InsertElementModifier(BasicBlock *BB, PieceTable *PT, Random *R):
    Modifier(BB, PT, R) {}

  virtual void Act() {
    Value *Val0 = getRandomVectorValue();
    Value *Val1 = getRandomValue(Val0->getType()->getScalarType());

    Value *V = InsertElementInst::Create(Val0, Val1,
              ConstantInt::get(Type::getInt32Ty(BB->getContext()),
              Ran->Rand() % cast<VectorType>(Val0->getType())->getNumElements()),
              "I",  BB->getTerminator());
    return PT->push_back(V);
  }

};

struct CastModifier: public Modifier {
  CastModifier(BasicBlock *BB, PieceTable *PT, Random *R):Modifier(BB, PT, R) {}
  virtual void Act() {

    Value *V = getRandomVal();
    Type *VTy = V->getType();
    Type *DestTy = pickScalarType();

    // Handle vector casts vectors.
    if (VTy->isVectorTy()) {
      VectorType *VecTy = cast<VectorType>(VTy);
      DestTy = pickVectorType(VecTy->getNumElements());
    }

    // no need to cast.
    if (VTy == DestTy) return;

    // Pointers:
    if (VTy->isPointerTy()) {
      if (!DestTy->isPointerTy())
        DestTy = PointerType::get(DestTy, 0);
      return PT->push_back(
        new BitCastInst(V, DestTy, "PC", BB->getTerminator()));
    }

    unsigned VSize = VTy->getScalarType()->getPrimitiveSizeInBits();
    unsigned DestSize = DestTy->getScalarType()->getPrimitiveSizeInBits();

    // Generate lots of bitcasts.
    if ((Ran->Rand() & 1) && VSize == DestSize) {
      return PT->push_back(
        new BitCastInst(V, DestTy, "BC", BB->getTerminator()));
    }

    // Both types are integers:
    if (VTy->getScalarType()->isIntegerTy() &&
        DestTy->getScalarType()->isIntegerTy()) {
      if (VSize > DestSize) {
        return PT->push_back(
          new TruncInst(V, DestTy, "Tr", BB->getTerminator()));
      } else {
        assert(VSize < DestSize && "Different int types with the same size?");
        if (Ran->Rand() & 1)
          return PT->push_back(
            new ZExtInst(V, DestTy, "ZE", BB->getTerminator()));
        return PT->push_back(new SExtInst(V, DestTy, "Se", BB->getTerminator()));
      }
    }

    // Fp to int.
    if (VTy->getScalarType()->isFloatingPointTy() &&
        DestTy->getScalarType()->isIntegerTy()) {
      if (Ran->Rand() & 1)
        return PT->push_back(
          new FPToSIInst(V, DestTy, "FC", BB->getTerminator()));
      return PT->push_back(new FPToUIInst(V, DestTy, "FC", BB->getTerminator()));
    }

    // Int to fp.
    if (VTy->getScalarType()->isIntegerTy() &&
        DestTy->getScalarType()->isFloatingPointTy()) {
      if (Ran->Rand() & 1)
        return PT->push_back(
          new SIToFPInst(V, DestTy, "FC", BB->getTerminator()));
      return PT->push_back(new UIToFPInst(V, DestTy, "FC", BB->getTerminator()));

    }

    // Both floats.
    if (VTy->getScalarType()->isFloatingPointTy() &&
        DestTy->getScalarType()->isFloatingPointTy()) {
      if (VSize > DestSize) {
        return PT->push_back(
          new FPTruncInst(V, DestTy, "Tr", BB->getTerminator()));
      } else if (VSize < DestSize) {
        return PT->push_back(
          new FPExtInst(V, DestTy, "ZE", BB->getTerminator()));
      }
      // If VSize == DestSize, then the two types must be fp128 and ppc_fp128,
      // for which there is no defined conversion. So do nothing.
    }
  }

};

struct SelectModifier: public Modifier {
  SelectModifier(BasicBlock *BB, PieceTable *PT, Random *R):
    Modifier(BB, PT, R) {}

  virtual void Act() {
    // Try a bunch of different select configuration until a valid one is found.
      Value *Val0 = getRandomVal();
      Value *Val1 = getRandomValue(Val0->getType());

      Type *CondTy = Type::getInt1Ty(Context);

      // If the value type is a vector, and we allow vector select, then in 50%
      // of the cases generate a vector select.
      if (Val0->getType()->isVectorTy() && (Ran->Rand() % 1)) {
        unsigned NumElem = cast<VectorType>(Val0->getType())->getNumElements();
        CondTy = VectorType::get(CondTy, NumElem);
      }

      Value *Cond = getRandomValue(CondTy);
      Value *V = SelectInst::Create(Cond, Val0, Val1, "Sl", BB->getTerminator());
      return PT->push_back(V);
  }
};


struct CmpModifier: public Modifier {
  CmpModifier(BasicBlock *BB, PieceTable *PT, Random *R):Modifier(BB, PT, R) {}
  virtual void Act() {

    Value *Val0 = getRandomVal();
    Value *Val1 = getRandomValue(Val0->getType());

    if (Val0->getType()->isPointerTy()) return;
    bool fp = Val0->getType()->getScalarType()->isFloatingPointTy();

    int op;
    if (fp) {
      op = Ran->Rand() %
      (CmpInst::LAST_FCMP_PREDICATE - CmpInst::FIRST_FCMP_PREDICATE) +
       CmpInst::FIRST_FCMP_PREDICATE;
    } else {
      op = Ran->Rand() %
      (CmpInst::LAST_ICMP_PREDICATE - CmpInst::FIRST_ICMP_PREDICATE) +
       CmpInst::FIRST_ICMP_PREDICATE;
    }

    Value *V = CmpInst::Create(fp ? Instruction::FCmp : Instruction::ICmp,
                               op, Val0, Val1, "Cmp", BB->getTerminator());
    return PT->push_back(V);
  }
};

void FillFunction(Function *F, Random &R) {
  // Create a legal entry block.
  BasicBlock *BB = BasicBlock::Create(F->getContext(), "BB", F);
  ReturnInst::Create(F->getContext(), BB);

  // Create the value table.
  Modifier::PieceTable PT;

  // Consider arguments as legal values.
  for (Function::arg_iterator it = F->arg_begin(), e = F->arg_end();
       it != e; ++it)
    PT.push_back(it);

  // List of modifiers which add new random instructions.
  std::vector<Modifier*> Modifiers;
  std::auto_ptr<Modifier> LM(new LoadModifier(BB, &PT, &R));
  std::auto_ptr<Modifier> SM(new StoreModifier(BB, &PT, &R));
  std::auto_ptr<Modifier> EE(new ExtractElementModifier(BB, &PT, &R));
  std::auto_ptr<Modifier> SHM(new ShuffModifier(BB, &PT, &R));
  std::auto_ptr<Modifier> IE(new InsertElementModifier(BB, &PT, &R));
  std::auto_ptr<Modifier> BM(new BinModifier(BB, &PT, &R));
  std::auto_ptr<Modifier> CM(new CastModifier(BB, &PT, &R));
  std::auto_ptr<Modifier> SLM(new SelectModifier(BB, &PT, &R));
  std::auto_ptr<Modifier> PM(new CmpModifier(BB, &PT, &R));
  Modifiers.push_back(LM.get());
  Modifiers.push_back(SM.get());
  Modifiers.push_back(EE.get());
  Modifiers.push_back(SHM.get());
  Modifiers.push_back(IE.get());
  Modifiers.push_back(BM.get());
  Modifiers.push_back(CM.get());
  Modifiers.push_back(SLM.get());
  Modifiers.push_back(PM.get());

  // Generate the random instructions
  AllocaModifier AM(BB, &PT, &R); AM.ActN(5); // Throw in a few allocas
  ConstModifier COM(BB, &PT, &R);  COM.ActN(40); // Throw in a few constants

  for (unsigned i=0; i< SizeCL / Modifiers.size(); ++i)
    for (std::vector<Modifier*>::iterator it = Modifiers.begin(),
         e = Modifiers.end(); it != e; ++it) {
      (*it)->Act();
    }

  SM->ActN(5); // Throw in a few stores.
}

void IntroduceControlFlow(Function *F, Random &R) {
  std::vector<Instruction*> BoolInst;
  for (BasicBlock::iterator it = F->begin()->begin(),
       e = F->begin()->end(); it != e; ++it) {
    if (it->getType() == IntegerType::getInt1Ty(F->getContext()))
      BoolInst.push_back(it);
  }

  std::random_shuffle(BoolInst.begin(), BoolInst.end(), R);

  for (std::vector<Instruction*>::iterator it = BoolInst.begin(),
       e = BoolInst.end(); it != e; ++it) {
    Instruction *Instr = *it;
    BasicBlock *Curr = Instr->getParent();
    BasicBlock::iterator Loc= Instr;
    BasicBlock *Next = Curr->splitBasicBlock(Loc, "CF");
    Instr->moveBefore(Curr->getTerminator());
    if (Curr != &F->getEntryBlock()) {
      BranchInst::Create(Curr, Next, Instr, Curr->getTerminator());
      Curr->getTerminator()->eraseFromParent();
    }
  }
}

int main(int argc, char **argv) {
  // Init LLVM, call llvm_shutdown() on exit, parse args, etc.
  llvm::PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "llvm codegen stress-tester\n");
  llvm_shutdown_obj Y;

  std::auto_ptr<Module> M(new Module("/tmp/autogen.bc", getGlobalContext()));
  Function *F = GenEmptyFunction(M.get());

  // Pick an initial seed value
  Random R(SeedCL);
  // Generate lots of random instructions inside a single basic block.
  FillFunction(F, R);
  // Break the basic block into many loops.
  IntroduceControlFlow(F, R);

  // Figure out what stream we are supposed to write to...
  OwningPtr<tool_output_file> Out;
  // Default to standard output.
  if (OutputFilename.empty())
    OutputFilename = "-";

  std::string ErrorInfo;
  Out.reset(new tool_output_file(OutputFilename.c_str(), ErrorInfo,
                                 raw_fd_ostream::F_Binary));
  if (!ErrorInfo.empty()) {
    errs() << ErrorInfo << '\n';
    return 1;
  }

  PassManager Passes;
  Passes.add(createVerifierPass());
  Passes.add(createPrintModulePass(&Out->os()));
  Passes.run(*M.get());
  Out->keep();

  return 0;
}
