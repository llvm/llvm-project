//===-- IntrinsicLowering.cpp - Intrinsic Lowering default implementation -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IntrinsicLowering class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Support/Streams.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

template <class ArgIt>
static void EnsureFunctionExists(Module &M, const char *Name,
                                 ArgIt ArgBegin, ArgIt ArgEnd,
                                 const Type *RetTy) {
  // Insert a correctly-typed definition now.
  std::vector<const Type *> ParamTys;
  for (ArgIt I = ArgBegin; I != ArgEnd; ++I)
    ParamTys.push_back(I->getType());
  M.getOrInsertFunction(Name, FunctionType::get(RetTy, ParamTys, false));
}

static void EnsureFPIntrinsicsExist(Module &M, Module::iterator I,
                                 const char *FName,
                                 const char *DName, const char *LDName) {
  // Insert definitions for all the floating point types.
  switch((int)I->arg_begin()->getType()->getTypeID()) {
  case Type::FloatTyID:
    EnsureFunctionExists(M, FName, I->arg_begin(), I->arg_end(),
                         Type::FloatTy);
  case Type::DoubleTyID:
    EnsureFunctionExists(M, DName, I->arg_begin(), I->arg_end(),
                         Type::DoubleTy);
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
    EnsureFunctionExists(M, LDName, I->arg_begin(), I->arg_end(),
                         I->arg_begin()->getType());
  }
}

/// ReplaceCallWith - This function is used when we want to lower an intrinsic
/// call to a call of an external function.  This handles hard cases such as
/// when there was already a prototype for the external function, and if that
/// prototype doesn't match the arguments we expect to pass in.
template <class ArgIt>
static CallInst *ReplaceCallWith(const char *NewFn, CallInst *CI,
                                 ArgIt ArgBegin, ArgIt ArgEnd,
                                 const Type *RetTy, Constant *&FCache) {
  if (!FCache) {
    // If we haven't already looked up this function, check to see if the
    // program already contains a function with this name.
    Module *M = CI->getParent()->getParent()->getParent();
    // Get or insert the definition now.
    std::vector<const Type *> ParamTys;
    for (ArgIt I = ArgBegin; I != ArgEnd; ++I)
      ParamTys.push_back((*I)->getType());
    FCache = M->getOrInsertFunction(NewFn,
                                    FunctionType::get(RetTy, ParamTys, false));
  }

  SmallVector<Value *, 8> Args(ArgBegin, ArgEnd);
  CallInst *NewCI = CallInst::Create(FCache, Args.begin(), Args.end(),
                                     CI->getName(), CI);
  if (!CI->use_empty())
    CI->replaceAllUsesWith(NewCI);
  return NewCI;
}

void IntrinsicLowering::AddPrototypes(Module &M) {
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->isDeclaration() && !I->use_empty())
      switch (I->getIntrinsicID()) {
      default: break;
      case Intrinsic::setjmp:
        EnsureFunctionExists(M, "setjmp", I->arg_begin(), I->arg_end(),
                             Type::Int32Ty);
        break;
      case Intrinsic::longjmp:
        EnsureFunctionExists(M, "longjmp", I->arg_begin(), I->arg_end(),
                             Type::VoidTy);
        break;
      case Intrinsic::siglongjmp:
        EnsureFunctionExists(M, "abort", I->arg_end(), I->arg_end(),
                             Type::VoidTy);
        break;
      case Intrinsic::memcpy:
        M.getOrInsertFunction("memcpy", PointerType::getUnqual(Type::Int8Ty),
                              PointerType::getUnqual(Type::Int8Ty), 
                              PointerType::getUnqual(Type::Int8Ty), 
                              TD.getIntPtrType(), (Type *)0);
        break;
      case Intrinsic::memmove:
        M.getOrInsertFunction("memmove", PointerType::getUnqual(Type::Int8Ty),
                              PointerType::getUnqual(Type::Int8Ty), 
                              PointerType::getUnqual(Type::Int8Ty), 
                              TD.getIntPtrType(), (Type *)0);
        break;
      case Intrinsic::memset:
        M.getOrInsertFunction("memset", PointerType::getUnqual(Type::Int8Ty),
                              PointerType::getUnqual(Type::Int8Ty), 
                              Type::Int32Ty, 
                              TD.getIntPtrType(), (Type *)0);
        break;
      case Intrinsic::sqrt:
        EnsureFPIntrinsicsExist(M, I, "sqrtf", "sqrt", "sqrtl");
        break;
      case Intrinsic::sin:
        EnsureFPIntrinsicsExist(M, I, "sinf", "sin", "sinl");
        break;
      case Intrinsic::cos:
        EnsureFPIntrinsicsExist(M, I, "cosf", "cos", "cosl");
        break;
      case Intrinsic::pow:
        EnsureFPIntrinsicsExist(M, I, "powf", "pow", "powl");
        break;
      case Intrinsic::log:
        EnsureFPIntrinsicsExist(M, I, "logf", "log", "logl");
        break;
      case Intrinsic::log2:
        EnsureFPIntrinsicsExist(M, I, "log2f", "log2", "log2l");
        break;
      case Intrinsic::log10:
        EnsureFPIntrinsicsExist(M, I, "log10f", "log10", "log10l");
        break;
      case Intrinsic::exp:
        EnsureFPIntrinsicsExist(M, I, "expf", "exp", "expl");
        break;
      case Intrinsic::exp2:
        EnsureFPIntrinsicsExist(M, I, "exp2f", "exp2", "exp2l");
        break;
      }
}

/// LowerBSWAP - Emit the code to lower bswap of V before the specified
/// instruction IP.
static Value *LowerBSWAP(Value *V, Instruction *IP) {
  assert(V->getType()->isInteger() && "Can't bswap a non-integer type!");

  unsigned BitSize = V->getType()->getPrimitiveSizeInBits();
  
  switch(BitSize) {
  default: assert(0 && "Unhandled type size of value to byteswap!");
  case 16: {
    Value *Tmp1 = BinaryOperator::CreateShl(V,
                                ConstantInt::get(V->getType(),8),"bswap.2",IP);
    Value *Tmp2 = BinaryOperator::CreateLShr(V,
                                ConstantInt::get(V->getType(),8),"bswap.1",IP);
    V = BinaryOperator::CreateOr(Tmp1, Tmp2, "bswap.i16", IP);
    break;
  }
  case 32: {
    Value *Tmp4 = BinaryOperator::CreateShl(V,
                              ConstantInt::get(V->getType(),24),"bswap.4", IP);
    Value *Tmp3 = BinaryOperator::CreateShl(V,
                              ConstantInt::get(V->getType(),8),"bswap.3",IP);
    Value *Tmp2 = BinaryOperator::CreateLShr(V,
                              ConstantInt::get(V->getType(),8),"bswap.2",IP);
    Value *Tmp1 = BinaryOperator::CreateLShr(V,
                              ConstantInt::get(V->getType(),24),"bswap.1", IP);
    Tmp3 = BinaryOperator::CreateAnd(Tmp3, 
                                     ConstantInt::get(Type::Int32Ty, 0xFF0000),
                                     "bswap.and3", IP);
    Tmp2 = BinaryOperator::CreateAnd(Tmp2, 
                                     ConstantInt::get(Type::Int32Ty, 0xFF00),
                                     "bswap.and2", IP);
    Tmp4 = BinaryOperator::CreateOr(Tmp4, Tmp3, "bswap.or1", IP);
    Tmp2 = BinaryOperator::CreateOr(Tmp2, Tmp1, "bswap.or2", IP);
    V = BinaryOperator::CreateOr(Tmp4, Tmp2, "bswap.i32", IP);
    break;
  }
  case 64: {
    Value *Tmp8 = BinaryOperator::CreateShl(V,
                              ConstantInt::get(V->getType(),56),"bswap.8", IP);
    Value *Tmp7 = BinaryOperator::CreateShl(V,
                              ConstantInt::get(V->getType(),40),"bswap.7", IP);
    Value *Tmp6 = BinaryOperator::CreateShl(V,
                              ConstantInt::get(V->getType(),24),"bswap.6", IP);
    Value *Tmp5 = BinaryOperator::CreateShl(V,
                              ConstantInt::get(V->getType(),8),"bswap.5", IP);
    Value* Tmp4 = BinaryOperator::CreateLShr(V,
                              ConstantInt::get(V->getType(),8),"bswap.4", IP);
    Value* Tmp3 = BinaryOperator::CreateLShr(V,
                              ConstantInt::get(V->getType(),24),"bswap.3", IP);
    Value* Tmp2 = BinaryOperator::CreateLShr(V,
                              ConstantInt::get(V->getType(),40),"bswap.2", IP);
    Value* Tmp1 = BinaryOperator::CreateLShr(V,
                              ConstantInt::get(V->getType(),56),"bswap.1", IP);
    Tmp7 = BinaryOperator::CreateAnd(Tmp7,
                             ConstantInt::get(Type::Int64Ty, 
                               0xFF000000000000ULL),
                             "bswap.and7", IP);
    Tmp6 = BinaryOperator::CreateAnd(Tmp6,
                             ConstantInt::get(Type::Int64Ty, 0xFF0000000000ULL),
                             "bswap.and6", IP);
    Tmp5 = BinaryOperator::CreateAnd(Tmp5,
                             ConstantInt::get(Type::Int64Ty, 0xFF00000000ULL),
                             "bswap.and5", IP);
    Tmp4 = BinaryOperator::CreateAnd(Tmp4,
                             ConstantInt::get(Type::Int64Ty, 0xFF000000ULL),
                             "bswap.and4", IP);
    Tmp3 = BinaryOperator::CreateAnd(Tmp3,
                             ConstantInt::get(Type::Int64Ty, 0xFF0000ULL),
                             "bswap.and3", IP);
    Tmp2 = BinaryOperator::CreateAnd(Tmp2,
                             ConstantInt::get(Type::Int64Ty, 0xFF00ULL),
                             "bswap.and2", IP);
    Tmp8 = BinaryOperator::CreateOr(Tmp8, Tmp7, "bswap.or1", IP);
    Tmp6 = BinaryOperator::CreateOr(Tmp6, Tmp5, "bswap.or2", IP);
    Tmp4 = BinaryOperator::CreateOr(Tmp4, Tmp3, "bswap.or3", IP);
    Tmp2 = BinaryOperator::CreateOr(Tmp2, Tmp1, "bswap.or4", IP);
    Tmp8 = BinaryOperator::CreateOr(Tmp8, Tmp6, "bswap.or5", IP);
    Tmp4 = BinaryOperator::CreateOr(Tmp4, Tmp2, "bswap.or6", IP);
    V = BinaryOperator::CreateOr(Tmp8, Tmp4, "bswap.i64", IP);
    break;
  }
  }
  return V;
}

/// LowerCTPOP - Emit the code to lower ctpop of V before the specified
/// instruction IP.
static Value *LowerCTPOP(Value *V, Instruction *IP) {
  assert(V->getType()->isInteger() && "Can't ctpop a non-integer type!");

  static const uint64_t MaskValues[6] = {
    0x5555555555555555ULL, 0x3333333333333333ULL,
    0x0F0F0F0F0F0F0F0FULL, 0x00FF00FF00FF00FFULL,
    0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL
  };

  unsigned BitSize = V->getType()->getPrimitiveSizeInBits();
  unsigned WordSize = (BitSize + 63) / 64;
  Value *Count = ConstantInt::get(V->getType(), 0);

  for (unsigned n = 0; n < WordSize; ++n) {
    Value *PartValue = V;
    for (unsigned i = 1, ct = 0; i < (BitSize>64 ? 64 : BitSize); 
         i <<= 1, ++ct) {
      Value *MaskCst = ConstantInt::get(V->getType(), MaskValues[ct]);
      Value *LHS = BinaryOperator::CreateAnd(
                     PartValue, MaskCst, "cppop.and1", IP);
      Value *VShift = BinaryOperator::CreateLShr(PartValue,
                        ConstantInt::get(V->getType(), i), "ctpop.sh", IP);
      Value *RHS = BinaryOperator::CreateAnd(VShift, MaskCst, "cppop.and2", IP);
      PartValue = BinaryOperator::CreateAdd(LHS, RHS, "ctpop.step", IP);
    }
    Count = BinaryOperator::CreateAdd(PartValue, Count, "ctpop.part", IP);
    if (BitSize > 64) {
      V = BinaryOperator::CreateLShr(V, ConstantInt::get(V->getType(), 64), 
                                     "ctpop.part.sh", IP);
      BitSize -= 64;
    }
  }

  return Count;
}

/// LowerCTLZ - Emit the code to lower ctlz of V before the specified
/// instruction IP.
static Value *LowerCTLZ(Value *V, Instruction *IP) {

  unsigned BitSize = V->getType()->getPrimitiveSizeInBits();
  for (unsigned i = 1; i < BitSize; i <<= 1) {
    Value *ShVal = ConstantInt::get(V->getType(), i);
    ShVal = BinaryOperator::CreateLShr(V, ShVal, "ctlz.sh", IP);
    V = BinaryOperator::CreateOr(V, ShVal, "ctlz.step", IP);
  }

  V = BinaryOperator::CreateNot(V, "", IP);
  return LowerCTPOP(V, IP);
}

/// Convert the llvm.part.select.iX.iY intrinsic. This intrinsic takes 
/// three integer arguments. The first argument is the Value from which the
/// bits will be selected. It may be of any bit width. The second and third
/// arguments specify a range of bits to select with the second argument 
/// specifying the low bit and the third argument specifying the high bit. Both
/// must be type i32. The result is the corresponding selected bits from the
/// Value in the same width as the Value (first argument). If the low bit index
/// is higher than the high bit index then the inverse selection is done and 
/// the bits are returned in inverse order. 
/// @brief Lowering of llvm.part.select intrinsic.
static Instruction *LowerPartSelect(CallInst *CI) {
  // Make sure we're dealing with a part select intrinsic here
  Function *F = CI->getCalledFunction();
  const FunctionType *FT = F->getFunctionType();
  if (!F->isDeclaration() || !FT->getReturnType()->isInteger() ||
      FT->getNumParams() != 3 || !FT->getParamType(0)->isInteger() ||
      !FT->getParamType(1)->isInteger() || !FT->getParamType(2)->isInteger())
    return CI;

  // Get the intrinsic implementation function by converting all the . to _
  // in the intrinsic's function name and then reconstructing the function
  // declaration.
  std::string Name(F->getName());
  for (unsigned i = 4; i < Name.length(); ++i)
    if (Name[i] == '.')
      Name[i] = '_';
  Module* M = F->getParent();
  F = cast<Function>(M->getOrInsertFunction(Name, FT));
  F->setLinkage(GlobalValue::WeakLinkage);

  // If we haven't defined the impl function yet, do so now
  if (F->isDeclaration()) {

    // Get the arguments to the function
    Function::arg_iterator args = F->arg_begin();
    Value* Val = args++; Val->setName("Val");
    Value* Lo = args++; Lo->setName("Lo");
    Value* Hi = args++; Hi->setName("High");

    // We want to select a range of bits here such that [Hi, Lo] is shifted
    // down to the low bits. However, it is quite possible that Hi is smaller
    // than Lo in which case the bits have to be reversed. 
    
    // Create the blocks we will need for the two cases (forward, reverse)
    BasicBlock* CurBB   = BasicBlock::Create("entry", F);
    BasicBlock *RevSize = BasicBlock::Create("revsize", CurBB->getParent());
    BasicBlock *FwdSize = BasicBlock::Create("fwdsize", CurBB->getParent());
    BasicBlock *Compute = BasicBlock::Create("compute", CurBB->getParent());
    BasicBlock *Reverse = BasicBlock::Create("reverse", CurBB->getParent());
    BasicBlock *RsltBlk = BasicBlock::Create("result",  CurBB->getParent());

    // Cast Hi and Lo to the size of Val so the widths are all the same
    if (Hi->getType() != Val->getType())
      Hi = CastInst::CreateIntegerCast(Hi, Val->getType(), false, 
                                         "tmp", CurBB);
    if (Lo->getType() != Val->getType())
      Lo = CastInst::CreateIntegerCast(Lo, Val->getType(), false, 
                                          "tmp", CurBB);

    // Compute a few things that both cases will need, up front.
    Constant* Zero = ConstantInt::get(Val->getType(), 0);
    Constant* One = ConstantInt::get(Val->getType(), 1);
    Constant* AllOnes = ConstantInt::getAllOnesValue(Val->getType());

    // Compare the Hi and Lo bit positions. This is used to determine 
    // which case we have (forward or reverse)
    ICmpInst *Cmp = new ICmpInst(ICmpInst::ICMP_ULT, Hi, Lo, "less",CurBB);
    BranchInst::Create(RevSize, FwdSize, Cmp, CurBB);

    // First, copmute the number of bits in the forward case.
    Instruction* FBitSize = 
      BinaryOperator::CreateSub(Hi, Lo,"fbits", FwdSize);
    BranchInst::Create(Compute, FwdSize);

    // Second, compute the number of bits in the reverse case.
    Instruction* RBitSize = 
      BinaryOperator::CreateSub(Lo, Hi, "rbits", RevSize);
    BranchInst::Create(Compute, RevSize);

    // Now, compute the bit range. Start by getting the bitsize and the shift
    // amount (either Hi or Lo) from PHI nodes. Then we compute a mask for 
    // the number of bits we want in the range. We shift the bits down to the 
    // least significant bits, apply the mask to zero out unwanted high bits, 
    // and we have computed the "forward" result. It may still need to be 
    // reversed.

    // Get the BitSize from one of the two subtractions
    PHINode *BitSize = PHINode::Create(Val->getType(), "bits", Compute);
    BitSize->reserveOperandSpace(2);
    BitSize->addIncoming(FBitSize, FwdSize);
    BitSize->addIncoming(RBitSize, RevSize);

    // Get the ShiftAmount as the smaller of Hi/Lo
    PHINode *ShiftAmt = PHINode::Create(Val->getType(), "shiftamt", Compute);
    ShiftAmt->reserveOperandSpace(2);
    ShiftAmt->addIncoming(Lo, FwdSize);
    ShiftAmt->addIncoming(Hi, RevSize);

    // Increment the bit size
    Instruction *BitSizePlusOne = 
      BinaryOperator::CreateAdd(BitSize, One, "bits", Compute);

    // Create a Mask to zero out the high order bits.
    Instruction* Mask = 
      BinaryOperator::CreateShl(AllOnes, BitSizePlusOne, "mask", Compute);
    Mask = BinaryOperator::CreateNot(Mask, "mask", Compute);

    // Shift the bits down and apply the mask
    Instruction* FRes = 
      BinaryOperator::CreateLShr(Val, ShiftAmt, "fres", Compute);
    FRes = BinaryOperator::CreateAnd(FRes, Mask, "fres", Compute);
    BranchInst::Create(Reverse, RsltBlk, Cmp, Compute);

    // In the Reverse block we have the mask already in FRes but we must reverse
    // it by shifting FRes bits right and putting them in RRes by shifting them 
    // in from left.

    // First set up our loop counters
    PHINode *Count = PHINode::Create(Val->getType(), "count", Reverse);
    Count->reserveOperandSpace(2);
    Count->addIncoming(BitSizePlusOne, Compute);

    // Next, get the value that we are shifting.
    PHINode *BitsToShift = PHINode::Create(Val->getType(), "val", Reverse);
    BitsToShift->reserveOperandSpace(2);
    BitsToShift->addIncoming(FRes, Compute);

    // Finally, get the result of the last computation
    PHINode *RRes = PHINode::Create(Val->getType(), "rres", Reverse);
    RRes->reserveOperandSpace(2);
    RRes->addIncoming(Zero, Compute);

    // Decrement the counter
    Instruction *Decr = BinaryOperator::CreateSub(Count, One, "decr", Reverse);
    Count->addIncoming(Decr, Reverse);

    // Compute the Bit that we want to move
    Instruction *Bit = 
      BinaryOperator::CreateAnd(BitsToShift, One, "bit", Reverse);

    // Compute the new value for next iteration.
    Instruction *NewVal = 
      BinaryOperator::CreateLShr(BitsToShift, One, "rshift", Reverse);
    BitsToShift->addIncoming(NewVal, Reverse);

    // Shift the bit into the low bits of the result.
    Instruction *NewRes = 
      BinaryOperator::CreateShl(RRes, One, "lshift", Reverse);
    NewRes = BinaryOperator::CreateOr(NewRes, Bit, "addbit", Reverse);
    RRes->addIncoming(NewRes, Reverse);
    
    // Terminate loop if we've moved all the bits.
    ICmpInst *Cond = 
      new ICmpInst(ICmpInst::ICMP_EQ, Decr, Zero, "cond", Reverse);
    BranchInst::Create(RsltBlk, Reverse, Cond, Reverse);

    // Finally, in the result block, select one of the two results with a PHI
    // node and return the result;
    CurBB = RsltBlk;
    PHINode *BitSelect = PHINode::Create(Val->getType(), "part_select", CurBB);
    BitSelect->reserveOperandSpace(2);
    BitSelect->addIncoming(FRes, Compute);
    BitSelect->addIncoming(NewRes, Reverse);
    ReturnInst::Create(BitSelect, CurBB);
  }

  // Return a call to the implementation function
  Value *Args[] = {
    CI->getOperand(1),
    CI->getOperand(2),
    CI->getOperand(3)
  };
  return CallInst::Create(F, Args, array_endof(Args), CI->getName(), CI);
}

/// Convert the llvm.part.set.iX.iY.iZ intrinsic. This intrinsic takes 
/// four integer arguments (iAny %Value, iAny %Replacement, i32 %Low, i32 %High)
/// The first two arguments can be any bit width. The result is the same width
/// as %Value. The operation replaces bits between %Low and %High with the value
/// in %Replacement. If %Replacement is not the same width, it is truncated or
/// zero extended as appropriate to fit the bits being replaced. If %Low is
/// greater than %High then the inverse set of bits are replaced.
/// @brief Lowering of llvm.bit.part.set intrinsic.
static Instruction *LowerPartSet(CallInst *CI) {
  // Make sure we're dealing with a part select intrinsic here
  Function *F = CI->getCalledFunction();
  const FunctionType *FT = F->getFunctionType();
  if (!F->isDeclaration() || !FT->getReturnType()->isInteger() ||
      FT->getNumParams() != 4 || !FT->getParamType(0)->isInteger() ||
      !FT->getParamType(1)->isInteger() || !FT->getParamType(2)->isInteger() ||
      !FT->getParamType(3)->isInteger())
    return CI;

  // Get the intrinsic implementation function by converting all the . to _
  // in the intrinsic's function name and then reconstructing the function
  // declaration.
  std::string Name(F->getName());
  for (unsigned i = 4; i < Name.length(); ++i)
    if (Name[i] == '.')
      Name[i] = '_';
  Module* M = F->getParent();
  F = cast<Function>(M->getOrInsertFunction(Name, FT));
  F->setLinkage(GlobalValue::WeakLinkage);

  // If we haven't defined the impl function yet, do so now
  if (F->isDeclaration()) {
    // Get the arguments for the function.
    Function::arg_iterator args = F->arg_begin();
    Value* Val = args++; Val->setName("Val");
    Value* Rep = args++; Rep->setName("Rep");
    Value* Lo  = args++; Lo->setName("Lo");
    Value* Hi  = args++; Hi->setName("Hi");

    // Get some types we need
    const IntegerType* ValTy = cast<IntegerType>(Val->getType());
    const IntegerType* RepTy = cast<IntegerType>(Rep->getType());
    uint32_t ValBits = ValTy->getBitWidth();
    uint32_t RepBits = RepTy->getBitWidth();

    // Constant Definitions
    ConstantInt* RepBitWidth = ConstantInt::get(Type::Int32Ty, RepBits);
    ConstantInt* RepMask = ConstantInt::getAllOnesValue(RepTy);
    ConstantInt* ValMask = ConstantInt::getAllOnesValue(ValTy);
    ConstantInt* One = ConstantInt::get(Type::Int32Ty, 1);
    ConstantInt* ValOne = ConstantInt::get(ValTy, 1);
    ConstantInt* Zero = ConstantInt::get(Type::Int32Ty, 0);
    ConstantInt* ValZero = ConstantInt::get(ValTy, 0);

    // Basic blocks we fill in below.
    BasicBlock* entry = BasicBlock::Create("entry", F, 0);
    BasicBlock* large = BasicBlock::Create("large", F, 0);
    BasicBlock* small = BasicBlock::Create("small", F, 0);
    BasicBlock* reverse = BasicBlock::Create("reverse", F, 0);
    BasicBlock* result = BasicBlock::Create("result", F, 0);

    // BASIC BLOCK: entry
    // First, get the number of bits that we're placing as an i32
    ICmpInst* is_forward = 
      new ICmpInst(ICmpInst::ICMP_ULT, Lo, Hi, "", entry);
    SelectInst* Hi_pn = SelectInst::Create(is_forward, Hi, Lo, "", entry);
    SelectInst* Lo_pn = SelectInst::Create(is_forward, Lo, Hi, "", entry);
    BinaryOperator* NumBits = BinaryOperator::CreateSub(Hi_pn, Lo_pn, "",entry);
    NumBits = BinaryOperator::CreateAdd(NumBits, One, "", entry);
    // Now, convert Lo and Hi to ValTy bit width
    if (ValBits > 32) {
      Lo = new ZExtInst(Lo_pn, ValTy, "", entry);
    } else if (ValBits < 32) {
      Lo = new TruncInst(Lo_pn, ValTy, "", entry);
    } else {
      Lo = Lo_pn;
    }
    // Determine if the replacement bits are larger than the number of bits we
    // are replacing and deal with it.
    ICmpInst* is_large = 
      new ICmpInst(ICmpInst::ICMP_ULT, NumBits, RepBitWidth, "", entry);
    BranchInst::Create(large, small, is_large, entry);

    // BASIC BLOCK: large
    Instruction* MaskBits = 
      BinaryOperator::CreateSub(RepBitWidth, NumBits, "", large);
    MaskBits = CastInst::CreateIntegerCast(MaskBits, RepMask->getType(), 
                                           false, "", large);
    BinaryOperator* Mask1 = 
      BinaryOperator::CreateLShr(RepMask, MaskBits, "", large);
    BinaryOperator* Rep2 = BinaryOperator::CreateAnd(Mask1, Rep, "", large);
    BranchInst::Create(small, large);

    // BASIC BLOCK: small
    PHINode* Rep3 = PHINode::Create(RepTy, "", small);
    Rep3->reserveOperandSpace(2);
    Rep3->addIncoming(Rep2, large);
    Rep3->addIncoming(Rep, entry);
    Value* Rep4 = Rep3;
    if (ValBits > RepBits)
      Rep4 = new ZExtInst(Rep3, ValTy, "", small);
    else if (ValBits < RepBits)
      Rep4 = new TruncInst(Rep3, ValTy, "", small);
    BranchInst::Create(result, reverse, is_forward, small);

    // BASIC BLOCK: reverse (reverses the bits of the replacement)
    // Set up our loop counter as a PHI so we can decrement on each iteration.
    // We will loop for the number of bits in the replacement value.
    PHINode *Count = PHINode::Create(Type::Int32Ty, "count", reverse);
    Count->reserveOperandSpace(2);
    Count->addIncoming(NumBits, small);

    // Get the value that we are shifting bits out of as a PHI because
    // we'll change this with each iteration.
    PHINode *BitsToShift = PHINode::Create(Val->getType(), "val", reverse);
    BitsToShift->reserveOperandSpace(2);
    BitsToShift->addIncoming(Rep4, small);

    // Get the result of the last computation or zero on first iteration
    PHINode *RRes = PHINode::Create(Val->getType(), "rres", reverse);
    RRes->reserveOperandSpace(2);
    RRes->addIncoming(ValZero, small);

    // Decrement the loop counter by one
    Instruction *Decr = BinaryOperator::CreateSub(Count, One, "", reverse);
    Count->addIncoming(Decr, reverse);

    // Get the bit that we want to move into the result
    Value *Bit = BinaryOperator::CreateAnd(BitsToShift, ValOne, "", reverse);

    // Compute the new value of the bits to shift for the next iteration.
    Value *NewVal = BinaryOperator::CreateLShr(BitsToShift, ValOne,"", reverse);
    BitsToShift->addIncoming(NewVal, reverse);

    // Shift the bit we extracted into the low bit of the result.
    Instruction *NewRes = BinaryOperator::CreateShl(RRes, ValOne, "", reverse);
    NewRes = BinaryOperator::CreateOr(NewRes, Bit, "", reverse);
    RRes->addIncoming(NewRes, reverse);
    
    // Terminate loop if we've moved all the bits.
    ICmpInst *Cond = new ICmpInst(ICmpInst::ICMP_EQ, Decr, Zero, "", reverse);
    BranchInst::Create(result, reverse, Cond, reverse);

    // BASIC BLOCK: result
    PHINode *Rplcmnt = PHINode::Create(Val->getType(), "", result);
    Rplcmnt->reserveOperandSpace(2);
    Rplcmnt->addIncoming(NewRes, reverse);
    Rplcmnt->addIncoming(Rep4, small);
    Value* t0   = CastInst::CreateIntegerCast(NumBits,ValTy,false,"",result);
    Value* t1   = BinaryOperator::CreateShl(ValMask, Lo, "", result);
    Value* t2   = BinaryOperator::CreateNot(t1, "", result);
    Value* t3   = BinaryOperator::CreateShl(t1, t0, "", result);
    Value* t4   = BinaryOperator::CreateOr(t2, t3, "", result);
    Value* t5   = BinaryOperator::CreateAnd(t4, Val, "", result);
    Value* t6   = BinaryOperator::CreateShl(Rplcmnt, Lo, "", result);
    Value* Rslt = BinaryOperator::CreateOr(t5, t6, "part_set", result);
    ReturnInst::Create(Rslt, result);
  }

  // Return a call to the implementation function
  Value *Args[] = {
    CI->getOperand(1),
    CI->getOperand(2),
    CI->getOperand(3),
    CI->getOperand(4)
  };
  return CallInst::Create(F, Args, array_endof(Args), CI->getName(), CI);
}

static void ReplaceFPIntrinsicWithCall(CallInst *CI, Constant *FCache,
                                       Constant *DCache, Constant *LDCache,
                                       const char *Fname, const char *Dname,
                                       const char *LDname) {
  switch (CI->getOperand(1)->getType()->getTypeID()) {
  default: assert(0 && "Invalid type in intrinsic"); abort();
  case Type::FloatTyID:
    ReplaceCallWith(Fname, CI, CI->op_begin()+1, CI->op_end(),
                  Type::FloatTy, FCache);
    break;
  case Type::DoubleTyID:
    ReplaceCallWith(Dname, CI, CI->op_begin()+1, CI->op_end(),
                  Type::DoubleTy, DCache);
    break;
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
    ReplaceCallWith(LDname, CI, CI->op_begin()+1, CI->op_end(),
                  CI->getOperand(1)->getType(), LDCache);
    break;
  }
}

void IntrinsicLowering::LowerIntrinsicCall(CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  assert(Callee && "Cannot lower an indirect call!");

  switch (Callee->getIntrinsicID()) {
  case Intrinsic::not_intrinsic:
    cerr << "Cannot lower a call to a non-intrinsic function '"
         << Callee->getName() << "'!\n";
    abort();
  default:
    cerr << "Error: Code generator does not support intrinsic function '"
         << Callee->getName() << "'!\n";
    abort();

    // The setjmp/longjmp intrinsics should only exist in the code if it was
    // never optimized (ie, right out of the CFE), or if it has been hacked on
    // by the lowerinvoke pass.  In both cases, the right thing to do is to
    // convert the call to an explicit setjmp or longjmp call.
  case Intrinsic::setjmp: {
    static Constant *SetjmpFCache = 0;
    Value *V = ReplaceCallWith("setjmp", CI, CI->op_begin()+1, CI->op_end(),
                               Type::Int32Ty, SetjmpFCache);
    if (CI->getType() != Type::VoidTy)
      CI->replaceAllUsesWith(V);
    break;
  }
  case Intrinsic::sigsetjmp:
     if (CI->getType() != Type::VoidTy)
       CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
     break;

  case Intrinsic::longjmp: {
    static Constant *LongjmpFCache = 0;
    ReplaceCallWith("longjmp", CI, CI->op_begin()+1, CI->op_end(),
                    Type::VoidTy, LongjmpFCache);
    break;
  }

  case Intrinsic::siglongjmp: {
    // Insert the call to abort
    static Constant *AbortFCache = 0;
    ReplaceCallWith("abort", CI, CI->op_end(), CI->op_end(), 
                    Type::VoidTy, AbortFCache);
    break;
  }
  case Intrinsic::ctpop:
    CI->replaceAllUsesWith(LowerCTPOP(CI->getOperand(1), CI));
    break;

  case Intrinsic::bswap:
    CI->replaceAllUsesWith(LowerBSWAP(CI->getOperand(1), CI));
    break;
    
  case Intrinsic::ctlz:
    CI->replaceAllUsesWith(LowerCTLZ(CI->getOperand(1), CI));
    break;

  case Intrinsic::cttz: {
    // cttz(x) -> ctpop(~X & (X-1))
    Value *Src = CI->getOperand(1);
    Value *NotSrc = BinaryOperator::CreateNot(Src, Src->getName()+".not", CI);
    Value *SrcM1 = ConstantInt::get(Src->getType(), 1);
    SrcM1 = BinaryOperator::CreateSub(Src, SrcM1, "", CI);
    Src = LowerCTPOP(BinaryOperator::CreateAnd(NotSrc, SrcM1, "", CI), CI);
    CI->replaceAllUsesWith(Src);
    break;
  }

  case Intrinsic::part_select:
    CI->replaceAllUsesWith(LowerPartSelect(CI));
    break;

  case Intrinsic::part_set:
    CI->replaceAllUsesWith(LowerPartSet(CI));
    break;

  case Intrinsic::stacksave:
  case Intrinsic::stackrestore: {
    static bool Warned = false;
    if (!Warned)
      cerr << "WARNING: this target does not support the llvm.stack"
           << (Callee->getIntrinsicID() == Intrinsic::stacksave ?
               "save" : "restore") << " intrinsic.\n";
    Warned = true;
    if (Callee->getIntrinsicID() == Intrinsic::stacksave)
      CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
    break;
  }
    
  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
    cerr << "WARNING: this target does not support the llvm."
         << (Callee->getIntrinsicID() == Intrinsic::returnaddress ?
             "return" : "frame") << "address intrinsic.\n";
    CI->replaceAllUsesWith(ConstantPointerNull::get(
                                            cast<PointerType>(CI->getType())));
    break;

  case Intrinsic::prefetch:
    break;    // Simply strip out prefetches on unsupported architectures

  case Intrinsic::pcmarker:
    break;    // Simply strip out pcmarker on unsupported architectures
  case Intrinsic::readcyclecounter: {
    cerr << "WARNING: this target does not support the llvm.readcyclecoun"
         << "ter intrinsic.  It is being lowered to a constant 0\n";
    CI->replaceAllUsesWith(ConstantInt::get(Type::Int64Ty, 0));
    break;
  }

  case Intrinsic::dbg_stoppoint:
  case Intrinsic::dbg_region_start:
  case Intrinsic::dbg_region_end:
  case Intrinsic::dbg_func_start:
  case Intrinsic::dbg_declare:
    break;    // Simply strip out debugging intrinsics

  case Intrinsic::eh_exception:
  case Intrinsic::eh_selector_i32:
  case Intrinsic::eh_selector_i64:
    CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
    break;

  case Intrinsic::eh_typeid_for_i32:
  case Intrinsic::eh_typeid_for_i64:
    // Return something different to eh_selector.
    CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 1));
    break;

  case Intrinsic::var_annotation:
    break;   // Strip out annotate intrinsic
    
  case Intrinsic::memcpy: {
    static Constant *MemcpyFCache = 0;
    Value *Size = CI->getOperand(3);
    const Type *IntPtr = TD.getIntPtrType();
    if (Size->getType()->getPrimitiveSizeInBits() <
        IntPtr->getPrimitiveSizeInBits())
      Size = new ZExtInst(Size, IntPtr, "", CI);
    else if (Size->getType()->getPrimitiveSizeInBits() >
             IntPtr->getPrimitiveSizeInBits())
      Size = new TruncInst(Size, IntPtr, "", CI);
    Value *Ops[3];
    Ops[0] = CI->getOperand(1);
    Ops[1] = CI->getOperand(2);
    Ops[2] = Size;
    ReplaceCallWith("memcpy", CI, Ops, Ops+3, CI->getOperand(1)->getType(),
                    MemcpyFCache);
    break;
  }
  case Intrinsic::memmove: {
    static Constant *MemmoveFCache = 0;
    Value *Size = CI->getOperand(3);
    const Type *IntPtr = TD.getIntPtrType();
    if (Size->getType()->getPrimitiveSizeInBits() <
        IntPtr->getPrimitiveSizeInBits())
      Size = new ZExtInst(Size, IntPtr, "", CI);
    else if (Size->getType()->getPrimitiveSizeInBits() >
             IntPtr->getPrimitiveSizeInBits())
      Size = new TruncInst(Size, IntPtr, "", CI);
    Value *Ops[3];
    Ops[0] = CI->getOperand(1);
    Ops[1] = CI->getOperand(2);
    Ops[2] = Size;
    ReplaceCallWith("memmove", CI, Ops, Ops+3, CI->getOperand(1)->getType(),
                    MemmoveFCache);
    break;
  }
  case Intrinsic::memset: {
    static Constant *MemsetFCache = 0;
    Value *Size = CI->getOperand(3);
    const Type *IntPtr = TD.getIntPtrType();
    if (Size->getType()->getPrimitiveSizeInBits() <
        IntPtr->getPrimitiveSizeInBits())
      Size = new ZExtInst(Size, IntPtr, "", CI);
    else if (Size->getType()->getPrimitiveSizeInBits() >
             IntPtr->getPrimitiveSizeInBits())
      Size = new TruncInst(Size, IntPtr, "", CI);
    Value *Ops[3];
    Ops[0] = CI->getOperand(1);
    // Extend the amount to i32.
    Ops[1] = new ZExtInst(CI->getOperand(2), Type::Int32Ty, "", CI);
    Ops[2] = Size;
    ReplaceCallWith("memset", CI, Ops, Ops+3, CI->getOperand(1)->getType(),
                    MemsetFCache);
    break;
  }
  case Intrinsic::sqrt: {
    static Constant *sqrtFCache = 0;
    static Constant *sqrtDCache = 0;
    static Constant *sqrtLDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, sqrtFCache, sqrtDCache, sqrtLDCache,
                               "sqrtf", "sqrt", "sqrtl");
    break;
  }
  case Intrinsic::log: {
    static Constant *logFCache = 0;
    static Constant *logDCache = 0;
    static Constant *logLDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, logFCache, logDCache, logLDCache,
                               "logf", "log", "logl");
    break;
  }
  case Intrinsic::log2: {
    static Constant *log2FCache = 0;
    static Constant *log2DCache = 0;
    static Constant *log2LDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, log2FCache, log2DCache, log2LDCache,
                               "log2f", "log2", "log2l");
    break;
  }
  case Intrinsic::log10: {
    static Constant *log10FCache = 0;
    static Constant *log10DCache = 0;
    static Constant *log10LDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, log10FCache, log10DCache, log10LDCache,
                               "log10f", "log10", "log10l");
    break;
  }
  case Intrinsic::exp: {
    static Constant *expFCache = 0;
    static Constant *expDCache = 0;
    static Constant *expLDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, expFCache, expDCache, expLDCache,
                               "expf", "exp", "expl");
    break;
  }
  case Intrinsic::exp2: {
    static Constant *exp2FCache = 0;
    static Constant *exp2DCache = 0;
    static Constant *exp2LDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, exp2FCache, exp2DCache, exp2LDCache,
                               "exp2f", "exp2", "exp2l");
    break;
  }
  case Intrinsic::pow: {
    static Constant *powFCache = 0;
    static Constant *powDCache = 0;
    static Constant *powLDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, powFCache, powDCache, powLDCache,
                               "powf", "pow", "powl");
    break;
  }
  case Intrinsic::flt_rounds:
     // Lower to "round to the nearest"
     if (CI->getType() != Type::VoidTy)
       CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 1));
     break;
  }

  assert(CI->use_empty() &&
         "Lowering should have eliminated any uses of the intrinsic call!");
  CI->eraseFromParent();
}
