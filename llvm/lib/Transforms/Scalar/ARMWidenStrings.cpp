// ARMWidenStrings.cpp - Widen strings to word boundaries to speed up
// programs that use simple strcpy's with constant strings as source
// and stack allocated array for destination.

#define DEBUG_TYPE "arm-widen-strings"

#include "llvm/Transforms/Scalar/ARMWidenStrings.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

cl::opt<bool> DisableARMWidenStrings("disable-arm-widen-strings",
                                     cl::init(false));

namespace {

class ARMWidenStrings {
public:
  /*
  Max number of bytes that memcpy allows for lowering to load/stores before it
  uses library function (__aeabi_memcpy).  This is the same value returned by
  ARMSubtarget::getMaxInlineSizeThreshold which I would have called in place of
  the constant int but can't get access to the subtarget info class from the
  midend.
  */
  const unsigned int MemcpyInliningLimit = 64;

  bool run(Function &F);
};

static bool IsCharArray(Type *t) {
  const unsigned int CHAR_BIT_SIZE = 8;
  return t && t->isArrayTy() && t->getArrayElementType()->isIntegerTy() &&
         t->getArrayElementType()->getIntegerBitWidth() == CHAR_BIT_SIZE;
}

bool ARMWidenStrings::run(Function &F) {
  if (DisableARMWidenStrings) {
    return false;
  }

  LLVM_DEBUG(dbgs() << "Running ARMWidenStrings on module " << F.getName()
                    << "\n");

  for (Function::iterator b = F.begin(); b != F.end(); ++b) {
    for (BasicBlock::iterator i = b->begin(); i != b->end(); ++i) {
      CallInst *CI = dyn_cast<CallInst>(i);
      if (!CI) {
        continue;
      }

      Function *CallMemcpy = CI->getCalledFunction();
      // find out if the current call instruction is a call to llvm memcpy
      // intrinsics
      if (CallMemcpy == NULL || !CallMemcpy->isIntrinsic() ||
          CallMemcpy->getIntrinsicID() != Intrinsic::memcpy) {
        continue;
      }

      LLVM_DEBUG(dbgs() << "Found call to strcpy/memcpy:\n" << *CI << "\n");

      auto *Alloca = dyn_cast<AllocaInst>(CI->getArgOperand(0));
      auto *SourceVar = dyn_cast<GlobalVariable>(CI->getArgOperand(1));
      auto *BytesToCopy = dyn_cast<ConstantInt>(CI->getArgOperand(2));
      auto *IsVolatile = dyn_cast<ConstantInt>(CI->getArgOperand(3));

      if (!BytesToCopy) {
        LLVM_DEBUG(dbgs() << "Number of bytes to copy is null\n");
        continue;
      }

      uint64_t NumBytesToCopy = BytesToCopy->getZExtValue();

      if (!Alloca) {
        LLVM_DEBUG(dbgs() << "Destination isn't a Alloca\n");
        continue;
      }

      if (!SourceVar) {
        LLVM_DEBUG(dbgs() << "Source isn't a global constant variable\n");
        continue;
      }

      if (!IsVolatile || IsVolatile->isOne()) {
        LLVM_DEBUG(
            dbgs() << "Not widening strings for this memcpy because it's "
                      "a volatile operations\n");
        continue;
      }

      if (NumBytesToCopy % 4 == 0) {
        LLVM_DEBUG(dbgs() << "Bytes to copy in strcpy/memcpy is already word "
                             "aligned so nothing to do here.\n");
        continue;
      }

      if (!SourceVar->hasInitializer() || !SourceVar->isConstant() ||
          !SourceVar->hasLocalLinkage() || !SourceVar->hasGlobalUnnamedAddr()) {
        LLVM_DEBUG(dbgs() << "Source is not constant global, thus it's "
                             "mutable therefore it's not safe to pad\n");
        continue;
      }

      ConstantDataArray *SourceDataArray =
          dyn_cast<ConstantDataArray>(SourceVar->getInitializer());
      if (!SourceDataArray || !IsCharArray(SourceDataArray->getType())) {
        LLVM_DEBUG(dbgs() << "Source isn't a constant data array\n");
        continue;
      }

      if (!Alloca->isStaticAlloca()) {
        LLVM_DEBUG(dbgs() << "Destination allocation isn't a static "
                             "constant which is locally allocated in this "
                             "function, so skipping.\n");
        continue;
      }

      // Make sure destination is definitley a char array.
      if (!IsCharArray(Alloca->getAllocatedType())) {
        LLVM_DEBUG(dbgs() << "Destination doesn't look like a constant char (8 "
                             "bits) array\n");
        continue;
      }
      LLVM_DEBUG(dbgs() << "With Alloca: " << *Alloca << "\n");

      uint64_t DZSize = Alloca->getAllocatedType()->getArrayNumElements();
      uint64_t SZSize = SourceDataArray->getType()->getNumElements();

      // For safety purposes lets add a constraint and only padd when
      // num bytes to copy == destination array size == source string
      // which is a constant
      LLVM_DEBUG(dbgs() << "Number of bytes to copy is: " << NumBytesToCopy
                        << "\n");
      LLVM_DEBUG(dbgs() << "Size of destination array is: " << DZSize << "\n");
      LLVM_DEBUG(dbgs() << "Size of source array is: " << SZSize << "\n");
      if (NumBytesToCopy != DZSize || DZSize != SZSize) {
        LLVM_DEBUG(dbgs() << "Size of number of bytes to copy, destination "
                             "array and source string don't match, so "
                             "skipping\n");
        continue;
      }
      LLVM_DEBUG(dbgs() << "Going to widen.\n");
      unsigned int NumBytesToPad = 4 - (NumBytesToCopy % 4);
      LLVM_DEBUG(dbgs() << "Number of bytes to pad by is " << NumBytesToPad
                        << "\n");
      unsigned int TotalBytes = NumBytesToCopy + NumBytesToPad;

      if (TotalBytes > MemcpyInliningLimit) {
        LLVM_DEBUG(
            dbgs() << "Not going to pad because total number of bytes is "
                   << TotalBytes
                   << "  which be greater than the inlining "
                      "limit for memcpy which is "
                   << MemcpyInliningLimit << "\n");
        continue;
      }

      // update destination char array to be word aligned (memcpy(X,...,...))
      IRBuilder<> BuildAlloca(Alloca);
      AllocaInst *NewAlloca = cast<AllocaInst>(BuildAlloca.CreateAlloca(
          ArrayType::get(Alloca->getAllocatedType()->getArrayElementType(),
                         NumBytesToCopy + NumBytesToPad)));
      NewAlloca->takeName(Alloca);
      NewAlloca->setAlignment(Alloca->getAlign());
      Alloca->replaceAllUsesWith(NewAlloca);

      LLVM_DEBUG(dbgs() << "Updating users of destination stack object to use "
                        << "new size\n");

      // update source to be word aligned (memcpy(...,X,...))
      // create replacement string with padded null bytes.
      StringRef Data = SourceDataArray->getRawDataValues();
      std::vector<uint8_t> StrData(Data.begin(), Data.end());
      for (unsigned int p = 0; p < NumBytesToPad; p++)
        StrData.push_back('\0');
      auto Arr = ArrayRef(StrData.data(), TotalBytes);

      // create new padded version of global variable string.
      Constant *SourceReplace = ConstantDataArray::get(F.getContext(), Arr);
      GlobalVariable *NewGV = new GlobalVariable(
          *F.getParent(), SourceReplace->getType(), true,
          SourceVar->getLinkage(), SourceReplace, SourceReplace->getName());

      // copy any other attributes from original global variable string
      // e.g. unamed_addr
      NewGV->copyAttributesFrom(SourceVar);
      NewGV->takeName(SourceVar);

      // replace intrinsic source.
      CI->setArgOperand(1, NewGV);

      // Update number of bytes to copy (memcpy(...,...,X))
      CI->setArgOperand(2,
                        ConstantInt::get(BytesToCopy->getType(), TotalBytes));
      LLVM_DEBUG(dbgs() << "Padded dest/source and increased number of bytes:\n"
                        << *CI << "\n"
                        << *NewAlloca << "\n");
    }
  }
  return true;
}

} // end of anonymous namespace

PreservedAnalyses ARMWidenStringsPass::run(Function &F,
                                           FunctionAnalysisManager &AM) {
  if (!ARMWidenStrings().run(F))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
