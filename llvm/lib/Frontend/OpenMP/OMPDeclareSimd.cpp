#include "llvm/Frontend/OpenMP/OMPDeclareSimd.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace omp;

std::string mangleVectorParameters(ArrayRef<llvm::omp::DeclareSimdAttrTy> ParamAttrs);

/// Return type size in bits for `Ty` using DL.
/// If scalable, return known-min as a conservative approximation.
static unsigned getTypeSizeInBits(llvm::Type *Ty, const llvm::DataLayout &DL) {
  if (!Ty)
    return 0;
  llvm::TypeSize TS = DL.getTypeSizeInBits(Ty);

  if (TS.isScalable())
    return (unsigned)TS.getKnownMinValue();
  return (unsigned)TS.getFixedValue();
}

/// Returns size in *bits* of the Characteristic Data Type (CDT).
static unsigned evaluateCDTSize(const llvm::Function *Fn,
                                ArrayRef<llvm::omp::DeclareSimdAttrTy> ParamAttrs) {
  const llvm::DataLayout &DL = Fn->getParent()->getDataLayout();

  llvm::Type *RetTy = Fn->getReturnType();
  llvm::Type *CDT = nullptr;

  // Non-void return => CDT = return type
  if (RetTy && !RetTy->isVoidTy()) {
    CDT = RetTy;
  } else {
    // First "Vector" param (ParamAttrs aligned with function params)
    // If ParamAttrs is shorter than the parameter list, treat missing as Vector
    // (matches the idea "default Kind is Vector").
    unsigned NumParams = Fn->getFunctionType()->getNumParams();
    for (unsigned I = 0; I < NumParams; ++I) {
      bool IsVector =
          (I < ParamAttrs.size()) ? ParamAttrs[I].Kind == llvm::omp::DeclareSimdKindTy::Vector : true;
      if (!IsVector)
        continue;
      CDT = Fn->getFunctionType()->getParamType(I);
      break;
    }
  }

  llvm::Type *IntTy = llvm::Type::getInt32Ty(Fn->getContext());
  if (!CDT || CDT->isStructTy() || CDT->isArrayTy())
    CDT = IntTy;

  return getTypeSizeInBits(CDT, DL);
}

static void emitX86DeclareSimdFunction(llvm::Function *Fn,
                                       const llvm::APSInt &VLENVal,
                                       ArrayRef<llvm::omp::DeclareSimdAttrTy> ParamAttrs,
                                       DeclareSimdBranch Branch) {
  struct ISADataTy {
    char ISA;
    unsigned VecRegSize;
  };
  ISADataTy ISAData[] = {
      {
          'b', 128
      }, // SSE
      {
          'c', 256
      }, // AVX
      {
          'd', 256
      }, // AVX2
      {
          'e', 512
      }, // AVX512
  };
  llvm::SmallVector<char, 2> Masked;
  switch (Branch) {
  case DeclareSimdBranch::Undefined:
    Masked.push_back('N');
    Masked.push_back('M');
    break;
  case DeclareSimdBranch::Notinbranch:
    Masked.push_back('N');
    break;
  case DeclareSimdBranch::Inbranch:
    Masked.push_back('M');
    break;
  }
  for (char Mask : Masked) {
    for (const ISADataTy &Data : ISAData) {
      SmallString<256> Buffer;
      llvm::raw_svector_ostream Out(Buffer);
      Out << "_ZGV" << Data.ISA << Mask;
      if (!VLENVal) {
        unsigned NumElts = evaluateCDTSize(Fn, ParamAttrs);
        assert(NumElts && "Non-zero simdlen/cdtsize expected");
        Out << llvm::APSInt::getUnsigned(Data.VecRegSize / NumElts);
      } else {
        Out << VLENVal;
      }
      Out << llvm::omp::mangleVectorParameters(ParamAttrs);
      Out << '_' << Fn->getName();
      Fn->addFnAttr(Out.str());
    }
  }
}

namespace llvm {

namespace omp {

/// Mangle the parameter part of the vector function name according to
/// their OpenMP classification. The mangling function is defined in
/// section 4.5 of the AAVFABI(2021Q1).
std::string mangleVectorParameters(ArrayRef<llvm::omp::DeclareSimdAttrTy> ParamAttrs) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  for (const auto &ParamAttr : ParamAttrs) {
    switch (ParamAttr.Kind) {
    case llvm::omp::DeclareSimdKindTy::Linear:
      Out << 'l';
      break;
    case llvm::omp::DeclareSimdKindTy::LinearRef:
      Out << 'R';
      break;
    case llvm::omp::DeclareSimdKindTy::LinearUVal:
      Out << 'U';
      break;
    case llvm::omp::DeclareSimdKindTy::LinearVal:
      Out << 'L';
      break;
    case llvm::omp::DeclareSimdKindTy::Uniform:
      Out << 'u';
      break;
    case llvm::omp::DeclareSimdKindTy::Vector:
      Out << 'v';
      break;
    }
    if (ParamAttr.HasVarStride)
      Out << "s" << ParamAttr.StrideOrArg;
    else if (ParamAttr.Kind == llvm::omp::DeclareSimdKindTy::Linear || ParamAttr.Kind == llvm::omp::DeclareSimdKindTy::LinearRef ||
             ParamAttr.Kind == llvm::omp::DeclareSimdKindTy::LinearUVal || ParamAttr.Kind == llvm::omp::DeclareSimdKindTy::LinearVal) {
      // Don't print the step value if it is not present or if it is
      // equal to 1.
      if (ParamAttr.StrideOrArg < 0)
        Out << 'n' << -ParamAttr.StrideOrArg;
      else if (ParamAttr.StrideOrArg != 1)
        Out << ParamAttr.StrideOrArg;
    }

    if (!!ParamAttr.Alignment)
      Out << 'a' << ParamAttr.Alignment;
  }

  return std::string(Out.str());
}

void emitDeclareSimdFunction(llvm::Function *Fn,
                             const llvm::APSInt &VLENVal,
                             ArrayRef<DeclareSimdAttrTy> ParamAttrs,
                             DeclareSimdBranch Branch) {
  Module *M = Fn->getParent();
  const llvm::Triple &Triple = M->getTargetTriple();

  if (Triple.isX86())
    emitX86DeclareSimdFunction(Fn, VLENVal, ParamAttrs, Branch);
}

}; // end namespace omp
}; // end namespace llvm