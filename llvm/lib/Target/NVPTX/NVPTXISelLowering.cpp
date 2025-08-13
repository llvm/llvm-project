//===-- NVPTXISelLowering.cpp - NVPTX DAG Lowering Implementation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that NVPTX uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "NVPTXISelLowering.h"
#include "MCTargetDesc/NVPTXBaseInfo.h"
#include "NVPTX.h"
#include "NVPTXSubtarget.h"
#include "NVPTXTargetMachine.h"
#include "NVPTXTargetObjectFile.h"
#include "NVPTXUtilities.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetCallingConv.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/NVPTXAddrSpace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#define DEBUG_TYPE "nvptx-lower"

using namespace llvm;

static cl::opt<bool> sched4reg(
    "nvptx-sched4reg",
    cl::desc("NVPTX Specific: schedule for register pressue"), cl::init(false));

static cl::opt<unsigned> FMAContractLevelOpt(
    "nvptx-fma-level", cl::Hidden,
    cl::desc("NVPTX Specific: FMA contraction (0: don't do it"
             " 1: do it  2: do it aggressively"),
    cl::init(2));

static cl::opt<NVPTX::DivPrecisionLevel> UsePrecDivF32(
    "nvptx-prec-divf32", cl::Hidden,
    cl::desc(
        "NVPTX Specific: Override the precision of the lowering for f32 fdiv"),
    cl::values(
        clEnumValN(NVPTX::DivPrecisionLevel::Approx, "0", "Use div.approx"),
        clEnumValN(NVPTX::DivPrecisionLevel::Full, "1", "Use div.full"),
        clEnumValN(NVPTX::DivPrecisionLevel::IEEE754, "2",
                   "Use IEEE Compliant F32 div.rnd if available (default)"),
        clEnumValN(NVPTX::DivPrecisionLevel::IEEE754_NoFTZ, "3",
                   "Use IEEE Compliant F32 div.rnd if available, no FTZ")),
    cl::init(NVPTX::DivPrecisionLevel::IEEE754));

static cl::opt<bool> UsePrecSqrtF32(
    "nvptx-prec-sqrtf32", cl::Hidden,
    cl::desc("NVPTX Specific: 0 use sqrt.approx, 1 use sqrt.rn."),
    cl::init(true));

/// Whereas CUDA's implementation (see libdevice) uses ex2.approx for exp2(), it
/// does NOT use lg2.approx for log2, so this is disabled by default.
static cl::opt<bool> UseApproxLog2F32(
    "nvptx-approx-log2f32",
    cl::desc("NVPTX Specific: whether to use lg2.approx for log2"),
    cl::init(false));

static cl::opt<bool> ForceMinByValParamAlign(
    "nvptx-force-min-byval-param-align", cl::Hidden,
    cl::desc("NVPTX Specific: force 4-byte minimal alignment for byval"
             " params of device functions."),
    cl::init(false));

NVPTX::DivPrecisionLevel
NVPTXTargetLowering::getDivF32Level(const MachineFunction &MF,
                                    const SDNode &N) const {
  // If nvptx-prec-div32=N is used on the command-line, always honor it
  if (UsePrecDivF32.getNumOccurrences() > 0)
    return UsePrecDivF32;

  // Otherwise, use div.approx if fast math is enabled
  if (allowUnsafeFPMath(MF))
    return NVPTX::DivPrecisionLevel::Approx;

  const SDNodeFlags Flags = N.getFlags();
  if (Flags.hasApproximateFuncs())
    return NVPTX::DivPrecisionLevel::Approx;

  return NVPTX::DivPrecisionLevel::IEEE754;
}

bool NVPTXTargetLowering::usePrecSqrtF32(const MachineFunction &MF,
                                         const SDNode *N) const {
  // If nvptx-prec-sqrtf32 is used on the command-line, always honor it
  if (UsePrecSqrtF32.getNumOccurrences() > 0)
    return UsePrecSqrtF32;

  // Otherwise, use sqrt.approx if fast math is enabled
  if (allowUnsafeFPMath(MF))
    return false;

  if (N) {
    const SDNodeFlags Flags = N->getFlags();
    if (Flags.hasApproximateFuncs())
      return false;
  }

  return true;
}

bool NVPTXTargetLowering::useF32FTZ(const MachineFunction &MF) const {
  return MF.getDenormalMode(APFloat::IEEEsingle()).Output ==
         DenormalMode::PreserveSign;
}

static bool IsPTXVectorType(MVT VT) {
  switch (VT.SimpleTy) {
  default:
    return false;
  case MVT::v2i1:
  case MVT::v4i1:
  case MVT::v2i8:
  case MVT::v4i8:
  case MVT::v8i8:  // <2 x i8x4>
  case MVT::v16i8: // <4 x i8x4>
  case MVT::v2i16:
  case MVT::v4i16:
  case MVT::v8i16: // <4 x i16x2>
  case MVT::v2i32:
  case MVT::v4i32:
  case MVT::v2i64:
  case MVT::v2f16:
  case MVT::v4f16:
  case MVT::v8f16: // <4 x f16x2>
  case MVT::v2bf16:
  case MVT::v4bf16:
  case MVT::v8bf16: // <4 x bf16x2>
  case MVT::v2f32:
  case MVT::v4f32:
  case MVT::v2f64:
  case MVT::v4i64:
  case MVT::v4f64:
  case MVT::v8i32:
  case MVT::v8f32:
  case MVT::v16f16:  // <8 x f16x2>
  case MVT::v16bf16: // <8 x bf16x2>
  case MVT::v16i16:  // <8 x i16x2>
  case MVT::v32i8:   // <8 x i8x4>
    return true;
  }
}

// When legalizing vector loads/stores, this function is called, which does two
// things:
// 1. Determines Whether the vector is something we want to custom lower,
// std::nullopt is returned if we do not want to custom lower it.
// 2. If we do want to handle it, returns two parameters:
//    - unsigned int NumElts - The number of elements in the final vector
//    - EVT EltVT - The type of the elements in the final vector
static std::optional<std::pair<unsigned int, MVT>>
getVectorLoweringShape(EVT VectorEVT, bool CanLowerTo256Bit) {
  if (!VectorEVT.isSimple())
    return std::nullopt;
  const MVT VectorVT = VectorEVT.getSimpleVT();

  if (!VectorVT.isVector()) {
    if (VectorVT == MVT::i128 || VectorVT == MVT::f128)
      return {{2, MVT::i64}};
    return std::nullopt;
  }

  const MVT EltVT = VectorVT.getVectorElementType();
  const unsigned NumElts = VectorVT.getVectorNumElements();

  // The size of the PTX virtual register that holds a packed type.
  unsigned PackRegSize;

  // We only handle "native" vector sizes for now, e.g. <4 x double> is not
  // legal.  We can (and should) split that into 2 stores of <2 x double> here
  // but I'm leaving that as a TODO for now.
  switch (VectorVT.SimpleTy) {
  default:
    return std::nullopt;
  case MVT::v4i64:
  case MVT::v4f64:
  case MVT::v8i32:
    // This is a "native" vector type iff the address space is global
    // and the target supports 256-bit loads/stores
    if (!CanLowerTo256Bit)
      return std::nullopt;
    LLVM_FALLTHROUGH;
  case MVT::v2i8:
  case MVT::v2i32:
  case MVT::v2i64:
  case MVT::v2f64:
  case MVT::v4i32:
    // This is a "native" vector type
    return std::pair(NumElts, EltVT);
  case MVT::v16f16:  // <8 x f16x2>
  case MVT::v16bf16: // <8 x bf16x2>
  case MVT::v16i16:  // <8 x i16x2>
  case MVT::v32i8:   // <8 x i8x4>
    // This can be upsized into a "native" vector type iff the address space is
    // global and the target supports 256-bit loads/stores.
    if (!CanLowerTo256Bit)
      return std::nullopt;
    LLVM_FALLTHROUGH;
  case MVT::v2i16:  // <1 x i16x2>
  case MVT::v2f16:  // <1 x f16x2>
  case MVT::v2bf16: // <1 x bf16x2>
  case MVT::v4i8:   // <1 x i8x4>
  case MVT::v4i16:  // <2 x i16x2>
  case MVT::v4f16:  // <2 x f16x2>
  case MVT::v4bf16: // <2 x bf16x2>
  case MVT::v8i8:   // <2 x i8x4>
  case MVT::v8f16:  // <4 x f16x2>
  case MVT::v8bf16: // <4 x bf16x2>
  case MVT::v8i16:  // <4 x i16x2>
  case MVT::v16i8:  // <4 x i8x4>
    PackRegSize = 32;
    break;
  case MVT::v8f32: // <4 x f32x2>
    if (!CanLowerTo256Bit)
      return std::nullopt;
    LLVM_FALLTHROUGH;
  case MVT::v2f32: // <1 x f32x2>
  case MVT::v4f32: // <2 x f32x2>
    PackRegSize = 64;
    break;
  }

  // If we reach here, then we can pack 2 or more elements into a single 32-bit
  // or 64-bit PTX register and treat the vector as a new vector containing
  // packed elements.

  // Number of elements to pack in one word.
  const unsigned NPerReg = PackRegSize / EltVT.getSizeInBits();

  return std::pair(NumElts / NPerReg, MVT::getVectorVT(EltVT, NPerReg));
}

/// ComputePTXValueVTs - For the given Type \p Ty, returns the set of primitive
/// EVTs that compose it.  Unlike ComputeValueVTs, this will break apart vectors
/// into their primitive components.
/// NOTE: This is a band-aid for code that expects ComputeValueVTs to return the
/// same number of types as the Ins/Outs arrays in LowerFormalArguments,
/// LowerCall, and LowerReturn.
static void ComputePTXValueVTs(const TargetLowering &TLI, const DataLayout &DL,
                               Type *Ty, SmallVectorImpl<EVT> &ValueVTs,
                               SmallVectorImpl<uint64_t> *Offsets = nullptr,
                               uint64_t StartingOffset = 0) {
  SmallVector<EVT, 16> TempVTs;
  SmallVector<uint64_t, 16> TempOffsets;

  // Special case for i128 - decompose to (i64, i64)
  if (Ty->isIntegerTy(128) || Ty->isFP128Ty()) {
    ValueVTs.append({MVT::i64, MVT::i64});

    if (Offsets)
      Offsets->append({StartingOffset + 0, StartingOffset + 8});

    return;
  }

  // Given a struct type, recursively traverse the elements with custom ComputePTXValueVTs.
  if (StructType *STy = dyn_cast<StructType>(Ty)) {
    auto const *SL = DL.getStructLayout(STy);
    auto ElementNum = 0;
    for(auto *EI : STy->elements()) {
      ComputePTXValueVTs(TLI, DL, EI, ValueVTs, Offsets,
                         StartingOffset + SL->getElementOffset(ElementNum));
      ++ElementNum;
    }
    return;
  }

  // Given an array type, recursively traverse the elements with custom ComputePTXValueVTs.
  if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    Type *EltTy = ATy->getElementType();
    uint64_t EltSize = DL.getTypeAllocSize(EltTy);
    for (int I : llvm::seq<int>(ATy->getNumElements()))
      ComputePTXValueVTs(TLI, DL, EltTy, ValueVTs, Offsets, StartingOffset + I * EltSize);
    return;
  }

  // Will split structs and arrays into member types, but will not split vector
  // types. We do that manually below.
  ComputeValueVTs(TLI, DL, Ty, TempVTs, &TempOffsets, StartingOffset);

  for (auto [VT, Off] : zip(TempVTs, TempOffsets)) {
    // Split vectors into individual elements that fit into registers.
    if (VT.isVector()) {
      unsigned NumElts = VT.getVectorNumElements();
      EVT EltVT = VT.getVectorElementType();
      // Below we must maintain power-of-2 sized vectors because
      // TargetLoweringBase::getVectorTypeBreakdown() which is invoked in
      // ComputePTXValueVTs() cannot currently break down non-power-of-2 sized
      // vectors.

      // If the element type belongs to one of the supported packed vector types
      // then we can pack multiples of this element into a single register.
      if (VT == MVT::v2i8) {
        // We can pack 2 i8s into a single 16-bit register. We only do this for
        // loads and stores, which is why we have a separate case for it.
        EltVT = MVT::v2i8;
        NumElts = 1;
      } else if (VT == MVT::v3i8) {
        // We can also pack 3 i8s into 32-bit register, leaving the 4th
        // element undefined.
        EltVT = MVT::v4i8;
        NumElts = 1;
      } else if (NumElts > 1 && isPowerOf2_32(NumElts)) {
        // Handle default packed types.
        for (MVT PackedVT : NVPTX::packed_types()) {
          const auto NumEltsPerReg = PackedVT.getVectorNumElements();
          if (NumElts % NumEltsPerReg == 0 &&
              EltVT == PackedVT.getVectorElementType()) {
            EltVT = PackedVT;
            NumElts /= NumEltsPerReg;
            break;
          }
        }
      }

      for (unsigned J : seq(NumElts)) {
        ValueVTs.push_back(EltVT);
        if (Offsets)
          Offsets->push_back(Off + J * EltVT.getStoreSize());
      }
    } else {
      ValueVTs.push_back(VT);
      if (Offsets)
        Offsets->push_back(Off);
    }
  }
}

// We return an EVT that can hold N VTs
// If the VT is a vector, the resulting EVT is a flat vector with the same
// element type as VT's element type.
static EVT getVectorizedVT(EVT VT, unsigned N, LLVMContext &C) {
  if (N == 1)
    return VT;

  return VT.isVector() ? EVT::getVectorVT(C, VT.getScalarType(),
                                          VT.getVectorNumElements() * N)
                       : EVT::getVectorVT(C, VT, N);
}

static SDValue getExtractVectorizedValue(SDValue V, unsigned I, EVT VT,
                                         const SDLoc &dl, SelectionDAG &DAG) {
  if (V.getValueType() == VT) {
    assert(I == 0 && "Index must be 0 for scalar value");
    return V;
  }

  if (!VT.isVector())
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, VT, V,
                       DAG.getVectorIdxConstant(I, dl));

  return DAG.getNode(
      ISD::EXTRACT_SUBVECTOR, dl, VT, V,
      DAG.getVectorIdxConstant(I * VT.getVectorNumElements(), dl));
}

template <typename T>
static inline SDValue getBuildVectorizedValue(unsigned N, const SDLoc &dl,
                                              SelectionDAG &DAG, T GetElement) {
  if (N == 1)
    return GetElement(0);

  SmallVector<SDValue, 8> Values;
  for (const unsigned I : llvm::seq(N)) {
    SDValue Val = GetElement(I);
    if (Val.getValueType().isVector())
      DAG.ExtractVectorElements(Val, Values);
    else
      Values.push_back(Val);
  }

  EVT VT = EVT::getVectorVT(*DAG.getContext(), Values[0].getValueType(),
                            Values.size());
  return DAG.getBuildVector(VT, dl, Values);
}

/// PromoteScalarIntegerPTX
/// Used to make sure the arguments/returns are suitable for passing
/// and promote them to a larger size if they're not.
///
/// The promoted type is placed in \p PromoteVT if the function returns true.
static EVT promoteScalarIntegerPTX(const EVT VT) {
  if (VT.isScalarInteger()) {
    switch (PowerOf2Ceil(VT.getFixedSizeInBits())) {
    default:
      llvm_unreachable(
          "Promotion is not suitable for scalars of size larger than 64-bits");
    case 1:
      return MVT::i1;
    case 2:
    case 4:
    case 8:
      return MVT::i8;
    case 16:
      return MVT::i16;
    case 32:
      return MVT::i32;
    case 64:
      return MVT::i64;
    }
  }
  return VT;
}

// Check whether we can merge loads/stores of some of the pieces of a
// flattened function parameter or return value into a single vector
// load/store.
//
// The flattened parameter is represented as a list of EVTs and
// offsets, and the whole structure is aligned to ParamAlignment. This
// function determines whether we can load/store pieces of the
// parameter starting at index Idx using a single vectorized op of
// size AccessSize. If so, it returns the number of param pieces
// covered by the vector op. Otherwise, it returns 1.
template <typename T>
static unsigned canMergeParamLoadStoresStartingAt(
    unsigned Idx, uint32_t AccessSize, const SmallVectorImpl<EVT> &ValueVTs,
    const SmallVectorImpl<T> &Offsets, Align ParamAlignment) {

  // Can't vectorize if param alignment is not sufficient.
  if (ParamAlignment < AccessSize)
    return 1;
  // Can't vectorize if offset is not aligned.
  if (Offsets[Idx] & (AccessSize - 1))
    return 1;

  EVT EltVT = ValueVTs[Idx];
  unsigned EltSize = EltVT.getStoreSize();

  // Element is too large to vectorize.
  if (EltSize >= AccessSize)
    return 1;

  unsigned NumElts = AccessSize / EltSize;
  // Can't vectorize if AccessBytes if not a multiple of EltSize.
  if (AccessSize != EltSize * NumElts)
    return 1;

  // We don't have enough elements to vectorize.
  if (Idx + NumElts > ValueVTs.size())
    return 1;

  // PTX ISA can only deal with 2- and 4-element vector ops.
  if (NumElts != 4 && NumElts != 2)
    return 1;

  for (unsigned j = Idx + 1; j < Idx + NumElts; ++j) {
    // Types do not match.
    if (ValueVTs[j] != EltVT)
      return 1;

    // Elements are not contiguous.
    if (Offsets[j] - Offsets[j - 1] != EltSize)
      return 1;
  }
  // OK. We can vectorize ValueVTs[i..i+NumElts)
  return NumElts;
}

// Computes whether and how we can vectorize the loads/stores of a
// flattened function parameter or return value.
//
// The flattened parameter is represented as the list of ValueVTs and
// Offsets, and is aligned to ParamAlignment bytes. We return a vector
// of the same size as ValueVTs indicating how each piece should be
// loaded/stored (i.e. as a scalar, or as part of a vector
// load/store).
template <typename T>
static SmallVector<unsigned, 16>
VectorizePTXValueVTs(const SmallVectorImpl<EVT> &ValueVTs,
                     const SmallVectorImpl<T> &Offsets, Align ParamAlignment,
                     bool IsVAArg = false) {
  // Set vector size to match ValueVTs and mark all elements as
  // scalars by default.

  if (IsVAArg)
    return SmallVector<unsigned>(ValueVTs.size(), 1);

  SmallVector<unsigned, 16> VectorInfo;

  const auto GetNumElts = [&](unsigned I) -> unsigned {
    for (const unsigned AccessSize : {16, 8, 4, 2}) {
      const unsigned NumElts = canMergeParamLoadStoresStartingAt(
          I, AccessSize, ValueVTs, Offsets, ParamAlignment);
      assert((NumElts == 1 || NumElts == 2 || NumElts == 4) &&
             "Unexpected vectorization size");
      if (NumElts != 1)
        return NumElts;
    }
    return 1;
  };

  // Check what we can vectorize using 128/64/32-bit accesses.
  for (unsigned I = 0, E = ValueVTs.size(); I != E;) {
    const unsigned NumElts = GetNumElts(I);
    VectorInfo.push_back(NumElts);
    I += NumElts;
  }
  assert(std::accumulate(VectorInfo.begin(), VectorInfo.end(), 0u) ==
         ValueVTs.size());
  return VectorInfo;
}

// NVPTXTargetLowering Constructor.
NVPTXTargetLowering::NVPTXTargetLowering(const NVPTXTargetMachine &TM,
                                         const NVPTXSubtarget &STI)
    : TargetLowering(TM), nvTM(&TM), STI(STI), GlobalUniqueCallSite(0) {
  // always lower memset, memcpy, and memmove intrinsics to load/store
  // instructions, rather
  // then generating calls to memset, mempcy or memmove.
  MaxStoresPerMemset = MaxStoresPerMemsetOptSize = (unsigned)0xFFFFFFFF;
  MaxStoresPerMemcpy = MaxStoresPerMemcpyOptSize = (unsigned) 0xFFFFFFFF;
  MaxStoresPerMemmove = MaxStoresPerMemmoveOptSize = (unsigned) 0xFFFFFFFF;

  setBooleanContents(ZeroOrNegativeOneBooleanContent);
  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent);

  // Jump is Expensive. Don't create extra control flow for 'and', 'or'
  // condition branches.
  setJumpIsExpensive(true);

  // Wide divides are _very_ slow. Try to reduce the width of the divide if
  // possible.
  addBypassSlowDiv(64, 32);

  // By default, use the Source scheduling
  if (sched4reg)
    setSchedulingPreference(Sched::RegPressure);
  else
    setSchedulingPreference(Sched::Source);

  auto setFP16OperationAction = [&](unsigned Op, MVT VT, LegalizeAction Action,
                                    LegalizeAction NoF16Action) {
    bool IsOpSupported = STI.allowFP16Math();
    switch (Op) {
    // Several FP16 instructions are available on sm_80 only.
    case ISD::FMINNUM:
    case ISD::FMAXNUM:
    case ISD::FMAXNUM_IEEE:
    case ISD::FMINNUM_IEEE:
    case ISD::FMAXIMUM:
    case ISD::FMINIMUM:
      IsOpSupported &= STI.getSmVersion() >= 80 && STI.getPTXVersion() >= 70;
      break;
    case ISD::FEXP2:
      IsOpSupported &= STI.getSmVersion() >= 75 && STI.getPTXVersion() >= 70;
      break;
    }
    setOperationAction(Op, VT, IsOpSupported ? Action : NoF16Action);
  };

  auto setBF16OperationAction = [&](unsigned Op, MVT VT, LegalizeAction Action,
                                    LegalizeAction NoBF16Action) {
    bool IsOpSupported = STI.hasNativeBF16Support(Op);
    setOperationAction(
        Op, VT, IsOpSupported ? Action : NoBF16Action);
  };

  auto setI16x2OperationAction = [&](unsigned Op, MVT VT, LegalizeAction Action,
                                     LegalizeAction NoI16x2Action) {
    bool IsOpSupported = false;
    // instructions are available on sm_90 only
    switch (Op) {
    case ISD::ADD:
    case ISD::SMAX:
    case ISD::SMIN:
    case ISD::UMIN:
    case ISD::UMAX:
      IsOpSupported = STI.getSmVersion() >= 90 && STI.getPTXVersion() >= 80;
      break;
    }
    setOperationAction(Op, VT, IsOpSupported ? Action : NoI16x2Action);
  };

  addRegisterClass(MVT::i1, &NVPTX::B1RegClass);
  addRegisterClass(MVT::i16, &NVPTX::B16RegClass);
  addRegisterClass(MVT::v2i16, &NVPTX::B32RegClass);
  addRegisterClass(MVT::v4i8, &NVPTX::B32RegClass);
  addRegisterClass(MVT::i32, &NVPTX::B32RegClass);
  addRegisterClass(MVT::i64, &NVPTX::B64RegClass);
  addRegisterClass(MVT::f32, &NVPTX::B32RegClass);
  addRegisterClass(MVT::f64, &NVPTX::B64RegClass);
  addRegisterClass(MVT::f16, &NVPTX::B16RegClass);
  addRegisterClass(MVT::v2f16, &NVPTX::B32RegClass);
  addRegisterClass(MVT::bf16, &NVPTX::B16RegClass);
  addRegisterClass(MVT::v2bf16, &NVPTX::B32RegClass);
  addRegisterClass(MVT::v2f32, &NVPTX::B64RegClass);

  // Conversion to/from FP16/FP16x2 is always legal.
  setOperationAction(ISD::BUILD_VECTOR, MVT::v2f16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2f16, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v2f16, Expand);

  setOperationAction(ISD::READCYCLECOUNTER, MVT::i64, Legal);
  if (STI.getSmVersion() >= 30 && STI.getPTXVersion() > 31)
    setOperationAction(ISD::READSTEADYCOUNTER, MVT::i64, Legal);

  setFP16OperationAction(ISD::SETCC, MVT::f16, Legal, Promote);
  setFP16OperationAction(ISD::SETCC, MVT::v2f16, Legal, Expand);

  // Conversion to/from BFP16/BFP16x2 is always legal.
  setOperationAction(ISD::BUILD_VECTOR, MVT::v2bf16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2bf16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2bf16, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v2bf16, Expand);

  setBF16OperationAction(ISD::SETCC, MVT::v2bf16, Legal, Expand);
  setBF16OperationAction(ISD::SETCC, MVT::bf16, Legal, Promote);
  if (getOperationAction(ISD::SETCC, MVT::bf16) == Promote)
    AddPromotedToType(ISD::SETCC, MVT::bf16, MVT::f32);

  // Conversion to/from i16/i16x2 is always legal.
  setOperationAction(ISD::BUILD_VECTOR, MVT::v2i16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2i16, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v2i16, Expand);

  setOperationAction(ISD::BUILD_VECTOR, MVT::v4i8, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4i8, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4i8, Custom);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v4i8, Custom);

  // No support for these operations with v2f32.
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2f32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v2f32, Expand);
  // Need custom lowering in case the index is dynamic.
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f32, Custom);

  // Custom conversions to/from v2i8.
  setOperationAction(ISD::BITCAST, MVT::v2i8, Custom);

  // Only logical ops can be done on v4i8 directly, others must be done
  // elementwise.
  setOperationAction(
      {ISD::ABS,         ISD::ADD,        ISD::ADDC,        ISD::ADDE,
       ISD::BITREVERSE,  ISD::CTLZ,       ISD::CTPOP,       ISD::CTTZ,
       ISD::FP_TO_SINT,  ISD::FP_TO_UINT, ISD::FSHL,        ISD::FSHR,
       ISD::MUL,         ISD::MULHS,      ISD::MULHU,       ISD::PARITY,
       ISD::ROTL,        ISD::ROTR,       ISD::SADDO,       ISD::SADDO_CARRY,
       ISD::SADDSAT,     ISD::SDIV,       ISD::SDIVREM,     ISD::SELECT_CC,
       ISD::SETCC,       ISD::SHL,        ISD::SINT_TO_FP,  ISD::SMAX,
       ISD::SMIN,        ISD::SMULO,      ISD::SMUL_LOHI,   ISD::SRA,
       ISD::SREM,        ISD::SRL,        ISD::SSHLSAT,     ISD::SSUBO,
       ISD::SSUBO_CARRY, ISD::SSUBSAT,    ISD::SUB,         ISD::SUBC,
       ISD::SUBE,        ISD::UADDO,      ISD::UADDO_CARRY, ISD::UADDSAT,
       ISD::UDIV,        ISD::UDIVREM,    ISD::UINT_TO_FP,  ISD::UMAX,
       ISD::UMIN,        ISD::UMULO,      ISD::UMUL_LOHI,   ISD::UREM,
       ISD::USHLSAT,     ISD::USUBO,      ISD::USUBO_CARRY, ISD::VSELECT,
       ISD::USUBSAT},
      MVT::v4i8, Expand);

  // Operations not directly supported by NVPTX.
  for (MVT VT : {MVT::bf16, MVT::f16, MVT::v2bf16, MVT::v2f16, MVT::f32,
                 MVT::v2f32, MVT::f64, MVT::i1, MVT::i8, MVT::i16, MVT::v2i16,
                 MVT::v4i8, MVT::i32, MVT::i64}) {
    setOperationAction(ISD::SELECT_CC, VT, Expand);
    setOperationAction(ISD::BR_CC, VT, Expand);
  }

  // Not directly supported. TLI would attempt to expand operations like
  // FMINIMUM(v2f32) using invalid SETCC and VSELECT nodes.
  setOperationAction(ISD::VSELECT, MVT::v2f32, Expand);

  // Some SIGN_EXTEND_INREG can be done using cvt instruction.
  // For others we will expand to a SHL/SRA pair.
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i64, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8 , Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i16, Expand);

  setOperationAction(ISD::SHL_PARTS, MVT::i32  , Custom);
  setOperationAction(ISD::SRA_PARTS, MVT::i32  , Custom);
  setOperationAction(ISD::SRL_PARTS, MVT::i32  , Custom);
  setOperationAction(ISD::SHL_PARTS, MVT::i64  , Custom);
  setOperationAction(ISD::SRA_PARTS, MVT::i64  , Custom);
  setOperationAction(ISD::SRL_PARTS, MVT::i64  , Custom);

  setOperationAction(ISD::BITREVERSE, MVT::i32, Legal);
  setOperationAction(ISD::BITREVERSE, MVT::i64, Legal);

  setOperationAction({ISD::ROTL, ISD::ROTR},
                     {MVT::i8, MVT::i16, MVT::v2i16, MVT::i32, MVT::i64},
                     Expand);

  if (STI.hasHWROT32()) {
    setOperationAction({ISD::FSHL, ISD::FSHR}, MVT::i32, Legal);
    setOperationAction({ISD::ROTL, ISD::ROTR, ISD::FSHL, ISD::FSHR}, MVT::i64,
                       Custom);
  }

  setOperationAction(ISD::BSWAP, MVT::i16, Expand);

  setOperationAction(ISD::BR_JT, MVT::Other, Custom);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);

  // We want to legalize constant related memmove and memcopy
  // intrinsics.
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);

  // Turn FP extload into load/fpextend
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, MVT::f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, MVT::bf16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::bf16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2f32, MVT::v2f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2f64, MVT::v2f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2f32, MVT::v2bf16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2f64, MVT::v2bf16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2f64, MVT::v2f32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f32, MVT::v4f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f64, MVT::v4f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f32, MVT::v4bf16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f64, MVT::v4bf16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f64, MVT::v4f32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v8f32, MVT::v8f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v8f64, MVT::v8f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v8f32, MVT::v8bf16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v8f64, MVT::v8bf16, Expand);
  // Turn FP truncstore into trunc + store.
  // FIXME: vector types should also be expanded
  setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  setTruncStoreAction(MVT::f64, MVT::f16, Expand);
  setTruncStoreAction(MVT::f32, MVT::bf16, Expand);
  setTruncStoreAction(MVT::f64, MVT::bf16, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  setTruncStoreAction(MVT::v2f32, MVT::v2f16, Expand);
  setTruncStoreAction(MVT::v2f32, MVT::v2bf16, Expand);

  // PTX does not support load / store predicate registers
  setOperationAction(ISD::LOAD, MVT::i1, Custom);
  setOperationAction(ISD::STORE, MVT::i1, Custom);

  for (MVT VT : MVT::integer_valuetypes()) {
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
    setTruncStoreAction(VT, MVT::i1, Expand);
  }

  setCondCodeAction({ISD::SETNE, ISD::SETEQ, ISD::SETUGE, ISD::SETULE,
                     ISD::SETUGT, ISD::SETULT, ISD::SETGT, ISD::SETLT,
                     ISD::SETGE, ISD::SETLE},
                    MVT::i1, Expand);

  // expand extload of vector of integers.
  setLoadExtAction({ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD}, MVT::v2i16,
                   MVT::v2i8, Expand);
  setTruncStoreAction(MVT::v2i16, MVT::v2i8, Expand);

  // This is legal in NVPTX
  setOperationAction(ISD::ConstantFP, MVT::f64, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f16, Legal);
  setOperationAction(ISD::ConstantFP, MVT::bf16, Legal);

  setOperationAction(ISD::DYNAMIC_STACKALLOC, {MVT::i32, MVT::i64}, Custom);
  setOperationAction({ISD::STACKRESTORE, ISD::STACKSAVE}, MVT::Other, Custom);

  // TRAP can be lowered to PTX trap
  setOperationAction(ISD::TRAP, MVT::Other, Legal);
  // DEBUGTRAP can be lowered to PTX brkpt
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Legal);

  // Register custom handling for vector loads/stores
  for (MVT VT : MVT::fixedlen_vector_valuetypes())
    if (IsPTXVectorType(VT))
      setOperationAction({ISD::LOAD, ISD::STORE, ISD::INTRINSIC_W_CHAIN}, VT,
                         Custom);

  setOperationAction({ISD::LOAD, ISD::STORE, ISD::INTRINSIC_W_CHAIN},
                     {MVT::i128, MVT::f128}, Custom);

  // Support varargs.
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAARG, MVT::Other, Custom);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  // Custom handling for i8 intrinsics
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i8, Custom);

  setOperationAction({ISD::ABS, ISD::SMIN, ISD::SMAX, ISD::UMIN, ISD::UMAX},
                     {MVT::i16, MVT::i32, MVT::i64}, Legal);

  setOperationAction({ISD::CTPOP, ISD::CTLZ, ISD::CTLZ_ZERO_UNDEF}, MVT::i16,
                     Promote);
  setOperationAction({ISD::CTPOP, ISD::CTLZ}, MVT::i32, Legal);
  setOperationAction({ISD::CTPOP, ISD::CTLZ}, MVT::i64, Custom);

  setI16x2OperationAction(ISD::ABS, MVT::v2i16, Legal, Custom);
  setI16x2OperationAction(ISD::SMIN, MVT::v2i16, Legal, Custom);
  setI16x2OperationAction(ISD::SMAX, MVT::v2i16, Legal, Custom);
  setI16x2OperationAction(ISD::UMIN, MVT::v2i16, Legal, Custom);
  setI16x2OperationAction(ISD::UMAX, MVT::v2i16, Legal, Custom);
  setI16x2OperationAction(ISD::CTPOP, MVT::v2i16, Legal, Expand);
  setI16x2OperationAction(ISD::CTLZ, MVT::v2i16, Legal, Expand);

  setI16x2OperationAction(ISD::ADD, MVT::v2i16, Legal, Custom);
  setI16x2OperationAction(ISD::SUB, MVT::v2i16, Legal, Custom);
  setI16x2OperationAction(ISD::MUL, MVT::v2i16, Legal, Custom);
  setI16x2OperationAction(ISD::SHL, MVT::v2i16, Legal, Custom);
  setI16x2OperationAction(ISD::SREM, MVT::v2i16, Legal, Custom);
  setI16x2OperationAction(ISD::UREM, MVT::v2i16, Legal, Custom);

  // Other arithmetic and logic ops are unsupported.
  setOperationAction({ISD::SDIV, ISD::UDIV, ISD::SRA, ISD::SRL, ISD::MULHS,
                      ISD::MULHU, ISD::FP_TO_SINT, ISD::FP_TO_UINT,
                      ISD::SINT_TO_FP, ISD::UINT_TO_FP, ISD::SETCC},
                     MVT::v2i16, Expand);

  setOperationAction(ISD::ADDC, MVT::i32, Legal);
  setOperationAction(ISD::ADDE, MVT::i32, Legal);
  setOperationAction(ISD::SUBC, MVT::i32, Legal);
  setOperationAction(ISD::SUBE, MVT::i32, Legal);
  if (STI.getPTXVersion() >= 43) {
    setOperationAction(ISD::ADDC, MVT::i64, Legal);
    setOperationAction(ISD::ADDE, MVT::i64, Legal);
    setOperationAction(ISD::SUBC, MVT::i64, Legal);
    setOperationAction(ISD::SUBE, MVT::i64, Legal);
  }

  setOperationAction(ISD::CTTZ, MVT::i16, Expand);
  setOperationAction(ISD::CTTZ, MVT::v2i16, Expand);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ, MVT::i64, Expand);

  // PTX does not directly support SELP of i1, so promote to i32 first
  setOperationAction(ISD::SELECT, MVT::i1, Custom);

  // PTX cannot multiply two i64s in a single instruction.
  setOperationAction(ISD::SMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);

  // We have some custom DAG combine patterns for these nodes
  setTargetDAGCombine({ISD::ADD, ISD::AND, ISD::EXTRACT_VECTOR_ELT, ISD::FADD,
                       ISD::MUL, ISD::SHL, ISD::SREM, ISD::UREM, ISD::VSELECT,
                       ISD::BUILD_VECTOR, ISD::ADDRSPACECAST, ISD::LOAD,
                       ISD::STORE, ISD::ZERO_EXTEND, ISD::SIGN_EXTEND});

  // setcc for f16x2 and bf16x2 needs special handling to prevent
  // legalizer's attempt to scalarize it due to v2i1 not being legal.
  if (STI.allowFP16Math() || STI.hasBF16Math())
    setTargetDAGCombine(ISD::SETCC);

  // Vector reduction operations. These may be turned into shuffle or tree
  // reductions depending on what instructions are available for each type.
  for (MVT VT : MVT::fixedlen_vector_valuetypes()) {
    MVT EltVT = VT.getVectorElementType();
    if (EltVT == MVT::f32 || EltVT == MVT::f64) {
      setOperationAction({ISD::VECREDUCE_FMAX, ISD::VECREDUCE_FMIN,
                          ISD::VECREDUCE_FMAXIMUM, ISD::VECREDUCE_FMINIMUM},
                         VT, Custom);
    }
  }

  // Promote fp16 arithmetic if fp16 hardware isn't available or the
  // user passed --nvptx-no-fp16-math. The flag is useful because,
  // although sm_53+ GPUs have some sort of FP16 support in
  // hardware, only sm_53 and sm_60 have full implementation. Others
  // only have token amount of hardware and are likely to run faster
  // by using fp32 units instead.
  for (const auto &Op : {ISD::FADD, ISD::FMUL, ISD::FSUB, ISD::FMA}) {
    setFP16OperationAction(Op, MVT::f16, Legal, Promote);
    setFP16OperationAction(Op, MVT::v2f16, Legal, Expand);
    setBF16OperationAction(Op, MVT::v2bf16, Legal, Expand);
    // bf16 must be promoted to f32.
    setBF16OperationAction(Op, MVT::bf16, Legal, Promote);
    if (getOperationAction(Op, MVT::bf16) == Promote)
      AddPromotedToType(Op, MVT::bf16, MVT::f32);
    setOperationAction(Op, MVT::v2f32,
                       STI.hasF32x2Instructions() ? Legal : Expand);
  }

  // On SM80, we select add/mul/sub as fma to avoid promotion to float
  for (const auto &Op : {ISD::FADD, ISD::FMUL, ISD::FSUB}) {
    for (const auto &VT : {MVT::bf16, MVT::v2bf16}) {
      if (!STI.hasNativeBF16Support(Op) && STI.hasNativeBF16Support(ISD::FMA)) {
        setOperationAction(Op, VT, Custom);
      }
    }
  }

  // f16/f16x2 neg was introduced in PTX 60, SM_53.
  const bool IsFP16FP16x2NegAvailable = STI.getSmVersion() >= 53 &&
                                        STI.getPTXVersion() >= 60 &&
                                        STI.allowFP16Math();
  for (const auto &VT : {MVT::f16, MVT::v2f16})
    setOperationAction(ISD::FNEG, VT,
                       IsFP16FP16x2NegAvailable ? Legal : Expand);

  setBF16OperationAction(ISD::FNEG, MVT::bf16, Legal, Expand);
  setBF16OperationAction(ISD::FNEG, MVT::v2bf16, Legal, Expand);
  setOperationAction(ISD::FNEG, MVT::v2f32, Expand);
  // (would be) Library functions.

  // These map to conversion instructions for scalar FP types.
  for (const auto &Op : {ISD::FCEIL, ISD::FFLOOR, ISD::FNEARBYINT, ISD::FRINT,
                         ISD::FROUNDEVEN, ISD::FTRUNC}) {
    setOperationAction(Op, MVT::f16, Legal);
    setOperationAction(Op, MVT::f32, Legal);
    setOperationAction(Op, MVT::f64, Legal);
    setOperationAction(Op, MVT::v2f16, Expand);
    setOperationAction(Op, MVT::v2bf16, Expand);
    setOperationAction(Op, MVT::v2f32, Expand);
    setBF16OperationAction(Op, MVT::bf16, Legal, Promote);
    if (getOperationAction(Op, MVT::bf16) == Promote)
      AddPromotedToType(Op, MVT::bf16, MVT::f32);
  }

  if (STI.getSmVersion() < 80 || STI.getPTXVersion() < 71) {
    setOperationAction(ISD::BF16_TO_FP, MVT::f32, Expand);
  }
  if (STI.getSmVersion() < 90 || STI.getPTXVersion() < 78) {
    for (MVT VT : {MVT::bf16, MVT::f32, MVT::f64}) {
      setOperationAction(ISD::FP_EXTEND, VT, Custom);
      setOperationAction(ISD::FP_ROUND, VT, Custom);
    }
  }

  // Expand v2f32 = fp_extend
  setOperationAction(ISD::FP_EXTEND, MVT::v2f32, Expand);
  // Expand v2[b]f16 = fp_round v2f32
  setOperationAction(ISD::FP_ROUND, {MVT::v2bf16, MVT::v2f16}, Expand);

  // sm_80 only has conversions between f32 and bf16. Custom lower all other
  // bf16 conversions.
  if (STI.getSmVersion() < 90 || STI.getPTXVersion() < 78) {
    for (MVT VT : {MVT::i1, MVT::i16, MVT::i32, MVT::i64}) {
      setOperationAction(
          {ISD::SINT_TO_FP, ISD::UINT_TO_FP, ISD::FP_TO_SINT, ISD::FP_TO_UINT},
          VT, Custom);
    }
    setOperationAction(
        {ISD::SINT_TO_FP, ISD::UINT_TO_FP, ISD::FP_TO_SINT, ISD::FP_TO_UINT},
        MVT::bf16, Custom);
  }

  setOperationAction(ISD::FROUND, MVT::f16, Promote);
  setOperationAction(ISD::FROUND, MVT::v2f16, Expand);
  setOperationAction(ISD::FROUND, MVT::v2bf16, Expand);
  setOperationAction(ISD::FROUND, MVT::f32, Custom);
  setOperationAction(ISD::FROUND, MVT::f64, Custom);
  setOperationAction(ISD::FROUND, MVT::bf16, Promote);
  AddPromotedToType(ISD::FROUND, MVT::bf16, MVT::f32);

  // 'Expand' implements FCOPYSIGN without calling an external library.
  setOperationAction(ISD::FCOPYSIGN, MVT::f16, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::v2f16, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::bf16, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::v2bf16, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Custom);
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Custom);

  // These map to corresponding instructions for f32/f64. f16 must be
  // promoted to f32. v2f16 is expanded to f16, which is then promoted
  // to f32.
  for (const auto &Op :
       {ISD::FDIV, ISD::FREM, ISD::FSQRT, ISD::FSIN, ISD::FCOS, ISD::FTANH}) {
    setOperationAction(Op, MVT::f16, Promote);
    setOperationAction(Op, MVT::f32, Legal);
    // only div/rem/sqrt are legal for f64
    if (Op == ISD::FDIV || Op == ISD::FREM || Op == ISD::FSQRT) {
      setOperationAction(Op, MVT::f64, Legal);
    }
    setOperationAction(Op, {MVT::v2f16, MVT::v2bf16, MVT::v2f32}, Expand);
    setOperationAction(Op, MVT::bf16, Promote);
    AddPromotedToType(Op, MVT::bf16, MVT::f32);
  }
  setOperationAction(ISD::FREM, {MVT::f32, MVT::f64}, Custom);

  setOperationAction(ISD::FABS, {MVT::f32, MVT::f64}, Legal);
  setOperationAction(ISD::FABS, MVT::v2f32, Expand);
  if (STI.getPTXVersion() >= 65) {
    setFP16OperationAction(ISD::FABS, MVT::f16, Legal, Promote);
    setFP16OperationAction(ISD::FABS, MVT::v2f16, Legal, Expand);
  } else {
    setOperationAction(ISD::FABS, MVT::f16, Promote);
    setOperationAction(ISD::FABS, MVT::v2f16, Expand);
  }
  setBF16OperationAction(ISD::FABS, MVT::v2bf16, Legal, Expand);
  setBF16OperationAction(ISD::FABS, MVT::bf16, Legal, Promote);
  if (getOperationAction(ISD::FABS, MVT::bf16) == Promote)
    AddPromotedToType(ISD::FABS, MVT::bf16, MVT::f32);

  for (const auto &Op : {ISD::FMINNUM, ISD::FMAXNUM}) {
    setOperationAction(Op, MVT::f32, Legal);
    setOperationAction(Op, MVT::f64, Legal);
    setFP16OperationAction(Op, MVT::f16, Legal, Promote);
    setFP16OperationAction(Op, MVT::v2f16, Legal, Expand);
    setBF16OperationAction(Op, MVT::v2bf16, Legal, Expand);
    setBF16OperationAction(Op, MVT::bf16, Legal, Promote);
    if (getOperationAction(Op, MVT::bf16) == Promote)
      AddPromotedToType(Op, MVT::bf16, MVT::f32);
    setOperationAction(Op, MVT::v2f32, Expand);
  }
  bool SupportsF32MinMaxNaN =
      STI.getSmVersion() >= 80 && STI.getPTXVersion() >= 70;
  for (const auto &Op : {ISD::FMINIMUM, ISD::FMAXIMUM}) {
    setOperationAction(Op, MVT::f32, SupportsF32MinMaxNaN ? Legal : Expand);
    setFP16OperationAction(Op, MVT::f16, Legal, Expand);
    setFP16OperationAction(Op, MVT::v2f16, Legal, Expand);
    setBF16OperationAction(Op, MVT::bf16, Legal, Expand);
    setBF16OperationAction(Op, MVT::v2bf16, Legal, Expand);
    setOperationAction(Op, MVT::v2f32, Expand);
  }

  // Custom lowering for inline asm with 128-bit operands
  setOperationAction(ISD::CopyToReg, MVT::i128, Custom);
  setOperationAction(ISD::CopyFromReg, MVT::i128, Custom);

  // FEXP2 support:
  // - f32
  // - f16/f16x2 (sm_70+, PTX 7.0+)
  // - bf16/bf16x2 (sm_90+, PTX 7.8+)
  // When f16/bf16 types aren't supported, they are promoted/expanded to f32.
  setOperationAction(ISD::FEXP2, MVT::f32, Legal);
  setOperationAction(ISD::FEXP2, MVT::v2f32, Expand);
  setFP16OperationAction(ISD::FEXP2, MVT::f16, Legal, Promote);
  setFP16OperationAction(ISD::FEXP2, MVT::v2f16, Legal, Expand);
  setBF16OperationAction(ISD::FEXP2, MVT::bf16, Legal, Promote);
  setBF16OperationAction(ISD::FEXP2, MVT::v2bf16, Legal, Expand);

  // FLOG2 supports f32 only
  // f16/bf16 types aren't supported, but they are promoted/expanded to f32.
  if (UseApproxLog2F32) {
    setOperationAction(ISD::FLOG2, MVT::f32, Legal);
    setOperationPromotedToType(ISD::FLOG2, MVT::f16, MVT::f32);
    setOperationPromotedToType(ISD::FLOG2, MVT::bf16, MVT::f32);
    setOperationAction(ISD::FLOG2, {MVT::v2f16, MVT::v2bf16, MVT::v2f32},
                       Expand);
  }

  setOperationAction(ISD::ADDRSPACECAST, {MVT::i32, MVT::i64}, Custom);

  setOperationAction(ISD::ATOMIC_LOAD_SUB, {MVT::i32, MVT::i64}, Expand);
  // No FPOW or FREM in PTX.

  // Now deduce the information based on the above mentioned
  // actions
  computeRegisterProperties(STI.getRegisterInfo());

  // PTX support for 16-bit CAS is emulated. Only use 32+
  setMinCmpXchgSizeInBits(STI.getMinCmpXchgSizeInBits());
  setMaxAtomicSizeInBitsSupported(64);
  setMaxDivRemBitWidthSupported(64);

  // Custom lowering for tcgen05.ld vector operands
  setOperationAction(ISD::INTRINSIC_W_CHAIN,
                     {MVT::v2i32, MVT::v4i32, MVT::v8i32, MVT::v16i32,
                      MVT::v32i32, MVT::v64i32, MVT::v128i32},
                     Custom);

  // Custom lowering for tcgen05.st vector operands
  setOperationAction(ISD::INTRINSIC_VOID,
                     {MVT::v2i32, MVT::v4i32, MVT::v8i32, MVT::v16i32,
                      MVT::v32i32, MVT::v64i32, MVT::v128i32},
                     Custom);

  // Enable custom lowering for the following:
  //   * MVT::i128 - clusterlaunchcontrol
  //   * MVT::i32 - prmt
  //   * MVT::Other - internal.addrspace.wrap
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, {MVT::i32, MVT::i128, MVT::Other},
                     Custom);
}

const char *NVPTXTargetLowering::getTargetNodeName(unsigned Opcode) const {

#define MAKE_CASE(V)                                                           \
  case V:                                                                      \
    return #V;

  switch ((NVPTXISD::NodeType)Opcode) {
  case NVPTXISD::FIRST_NUMBER:
    break;

    MAKE_CASE(NVPTXISD::RET_GLUE)
    MAKE_CASE(NVPTXISD::DeclareArrayParam)
    MAKE_CASE(NVPTXISD::DeclareScalarParam)
    MAKE_CASE(NVPTXISD::CALL)
    MAKE_CASE(NVPTXISD::MoveParam)
    MAKE_CASE(NVPTXISD::UNPACK_VECTOR)
    MAKE_CASE(NVPTXISD::BUILD_VECTOR)
    MAKE_CASE(NVPTXISD::CallPrototype)
    MAKE_CASE(NVPTXISD::ProxyReg)
    MAKE_CASE(NVPTXISD::LoadV2)
    MAKE_CASE(NVPTXISD::LoadV4)
    MAKE_CASE(NVPTXISD::LoadV8)
    MAKE_CASE(NVPTXISD::LDUV2)
    MAKE_CASE(NVPTXISD::LDUV4)
    MAKE_CASE(NVPTXISD::StoreV2)
    MAKE_CASE(NVPTXISD::StoreV4)
    MAKE_CASE(NVPTXISD::StoreV8)
    MAKE_CASE(NVPTXISD::FSHL_CLAMP)
    MAKE_CASE(NVPTXISD::FSHR_CLAMP)
    MAKE_CASE(NVPTXISD::BFI)
    MAKE_CASE(NVPTXISD::PRMT)
    MAKE_CASE(NVPTXISD::FCOPYSIGN)
    MAKE_CASE(NVPTXISD::FMAXNUM3)
    MAKE_CASE(NVPTXISD::FMINNUM3)
    MAKE_CASE(NVPTXISD::FMAXIMUM3)
    MAKE_CASE(NVPTXISD::FMINIMUM3)
    MAKE_CASE(NVPTXISD::DYNAMIC_STACKALLOC)
    MAKE_CASE(NVPTXISD::STACKRESTORE)
    MAKE_CASE(NVPTXISD::STACKSAVE)
    MAKE_CASE(NVPTXISD::SETP_F16X2)
    MAKE_CASE(NVPTXISD::SETP_BF16X2)
    MAKE_CASE(NVPTXISD::MUL_WIDE_SIGNED)
    MAKE_CASE(NVPTXISD::MUL_WIDE_UNSIGNED)
    MAKE_CASE(NVPTXISD::BrxEnd)
    MAKE_CASE(NVPTXISD::BrxItem)
    MAKE_CASE(NVPTXISD::BrxStart)
    MAKE_CASE(NVPTXISD::CLUSTERLAUNCHCONTROL_QUERY_CANCEL_IS_CANCELED)
    MAKE_CASE(NVPTXISD::CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID_X)
    MAKE_CASE(NVPTXISD::CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID_Y)
    MAKE_CASE(NVPTXISD::CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID_Z)
  }
  return nullptr;

#undef MAKE_CASE
}

TargetLoweringBase::LegalizeTypeAction
NVPTXTargetLowering::getPreferredVectorAction(MVT VT) const {
  if (!VT.isScalableVector() && VT.getVectorNumElements() != 1 &&
      VT.getScalarType() == MVT::i1)
    return TypeSplitVector;
  return TargetLoweringBase::getPreferredVectorAction(VT);
}

SDValue NVPTXTargetLowering::getSqrtEstimate(SDValue Operand, SelectionDAG &DAG,
                                             int Enabled, int &ExtraSteps,
                                             bool &UseOneConst,
                                             bool Reciprocal) const {
  if (!(Enabled == ReciprocalEstimate::Enabled ||
        (Enabled == ReciprocalEstimate::Unspecified &&
         !usePrecSqrtF32(DAG.getMachineFunction()))))
    return SDValue();

  if (ExtraSteps == ReciprocalEstimate::Unspecified)
    ExtraSteps = 0;

  SDLoc DL(Operand);
  EVT VT = Operand.getValueType();
  bool Ftz = useF32FTZ(DAG.getMachineFunction());

  auto MakeIntrinsicCall = [&](Intrinsic::ID IID) {
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT,
                       DAG.getConstant(IID, DL, MVT::i32), Operand);
  };

  // The sqrt and rsqrt refinement processes assume we always start out with an
  // approximation of the rsqrt.  Therefore, if we're going to do any refinement
  // (i.e. ExtraSteps > 0), we must return an rsqrt.  But if we're *not* doing
  // any refinement, we must return a regular sqrt.
  if (Reciprocal || ExtraSteps > 0) {
    if (VT == MVT::f32)
      return MakeIntrinsicCall(Ftz ? Intrinsic::nvvm_rsqrt_approx_ftz_f
                                   : Intrinsic::nvvm_rsqrt_approx_f);
    else if (VT == MVT::f64)
      return MakeIntrinsicCall(Intrinsic::nvvm_rsqrt_approx_d);
    else
      return SDValue();
  } else {
    if (VT == MVT::f32)
      return MakeIntrinsicCall(Ftz ? Intrinsic::nvvm_sqrt_approx_ftz_f
                                   : Intrinsic::nvvm_sqrt_approx_f);
    else {
      // There's no sqrt.approx.f64 instruction, so we emit
      // reciprocal(rsqrt(x)).  This is faster than
      // select(x == 0, 0, x * rsqrt(x)).  (In fact, it's faster than plain
      // x * rsqrt(x).)
      return DAG.getNode(
          ISD::INTRINSIC_WO_CHAIN, DL, VT,
          DAG.getConstant(Intrinsic::nvvm_rcp_approx_ftz_d, DL, MVT::i32),
          MakeIntrinsicCall(Intrinsic::nvvm_rsqrt_approx_d));
    }
  }
}

std::string NVPTXTargetLowering::getPrototype(
    const DataLayout &DL, Type *RetTy, const ArgListTy &Args,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    std::optional<unsigned> FirstVAArg, const CallBase &CB,
    unsigned UniqueCallSite) const {
  auto PtrVT = getPointerTy(DL);

  std::string Prototype;
  raw_string_ostream O(Prototype);
  O << "prototype_" << UniqueCallSite << " : .callprototype ";

  if (RetTy->isVoidTy()) {
    O << "()";
  } else {
    O << "(";
    if (shouldPassAsArray(RetTy)) {
      const Align RetAlign = getArgumentAlignment(&CB, RetTy, 0, DL);
      O << ".param .align " << RetAlign.value() << " .b8 _["
        << DL.getTypeAllocSize(RetTy) << "]";
    } else if (RetTy->isFloatingPointTy() || RetTy->isIntegerTy()) {
      unsigned size = 0;
      if (auto *ITy = dyn_cast<IntegerType>(RetTy)) {
        size = ITy->getBitWidth();
      } else {
        assert(RetTy->isFloatingPointTy() &&
               "Floating point type expected here");
        size = RetTy->getPrimitiveSizeInBits();
      }
      // PTX ABI requires all scalar return values to be at least 32
      // bits in size.  fp16 normally uses .b16 as its storage type in
      // PTX, so its size must be adjusted here, too.
      size = promoteScalarArgumentSize(size);

      O << ".param .b" << size << " _";
    } else if (isa<PointerType>(RetTy)) {
      O << ".param .b" << PtrVT.getSizeInBits() << " _";
    } else {
      llvm_unreachable("Unknown return type");
    }
    O << ") ";
  }
  O << "_ (";

  bool first = true;

  const unsigned NumArgs = FirstVAArg.value_or(Args.size());
  auto AllOuts = ArrayRef(Outs);
  for (const unsigned I : llvm::seq(NumArgs)) {
    const auto ArgOuts =
        AllOuts.take_while([I](auto O) { return O.OrigArgIndex == I; });
    AllOuts = AllOuts.drop_front(ArgOuts.size());

    Type *Ty = Args[I].Ty;
    if (!first) {
      O << ", ";
    }
    first = false;

    if (ArgOuts[0].Flags.isByVal()) {
      // Indirect calls need strict ABI alignment so we disable optimizations by
      // not providing a function to optimize.
      Type *ETy = Args[I].IndirectType;
      Align InitialAlign = ArgOuts[0].Flags.getNonZeroByValAlign();
      Align ParamByValAlign =
          getFunctionByValParamAlign(/*F=*/nullptr, ETy, InitialAlign, DL);

      O << ".param .align " << ParamByValAlign.value() << " .b8 _["
        << ArgOuts[0].Flags.getByValSize() << "]";
    } else {
      if (shouldPassAsArray(Ty)) {
        Align ParamAlign =
            getArgumentAlignment(&CB, Ty, I + AttributeList::FirstArgIndex, DL);
        O << ".param .align " << ParamAlign.value() << " .b8 _["
          << DL.getTypeAllocSize(Ty) << "]";
        continue;
      }
      // i8 types in IR will be i16 types in SDAG
      assert((getValueType(DL, Ty) == ArgOuts[0].VT ||
              (getValueType(DL, Ty) == MVT::i8 && ArgOuts[0].VT == MVT::i16)) &&
             "type mismatch between callee prototype and arguments");
      // scalar type
      unsigned sz = 0;
      if (auto *ITy = dyn_cast<IntegerType>(Ty)) {
        sz = promoteScalarArgumentSize(ITy->getBitWidth());
      } else if (isa<PointerType>(Ty)) {
        sz = PtrVT.getSizeInBits();
      } else {
        sz = Ty->getPrimitiveSizeInBits();
      }
      O << ".param .b" << sz << " _";
    }
  }

  if (FirstVAArg)
    O << (first ? "" : ",") << " .param .align "
      << STI.getMaxRequiredAlignment() << " .b8 _[]";
  O << ")";
  if (shouldEmitPTXNoReturn(&CB, *nvTM))
    O << " .noreturn";
  O << ";";

  return Prototype;
}

Align NVPTXTargetLowering::getFunctionArgumentAlignment(
    const Function *F, Type *Ty, unsigned Idx, const DataLayout &DL) const {
  return getAlign(*F, Idx).value_or(getFunctionParamOptimizedAlign(F, Ty, DL));
}

Align NVPTXTargetLowering::getArgumentAlignment(const CallBase *CB, Type *Ty,
                                                unsigned Idx,
                                                const DataLayout &DL) const {
  if (!CB) {
    // CallSite is zero, fallback to ABI type alignment
    return DL.getABITypeAlign(Ty);
  }

  const Function *DirectCallee = CB->getCalledFunction();

  if (!DirectCallee) {
    // We don't have a direct function symbol, but that may be because of
    // constant cast instructions in the call.

    // With bitcast'd call targets, the instruction will be the call
    if (const auto *CI = dyn_cast<CallInst>(CB)) {
      // Check if we have call alignment metadata
      if (MaybeAlign StackAlign = getAlign(*CI, Idx))
        return StackAlign.value();
    }
    DirectCallee = getMaybeBitcastedCallee(CB);
  }

  // Check for function alignment information if we found that the
  // ultimate target is a Function
  if (DirectCallee)
    return getFunctionArgumentAlignment(DirectCallee, Ty, Idx, DL);

  // Call is indirect, fall back to the ABI type alignment
  return DL.getABITypeAlign(Ty);
}

static bool shouldConvertToIndirectCall(const CallBase *CB,
                                        const GlobalAddressSDNode *Func) {
  if (!Func)
    return false;
  if (auto *CalleeFunc = dyn_cast<Function>(Func->getGlobal()))
    return CB->getFunctionType() != CalleeFunc->getFunctionType();
  return false;
}

static MachinePointerInfo refinePtrAS(SDValue &Ptr, SelectionDAG &DAG,
                                      const DataLayout &DL,
                                      const TargetLowering &TL) {
  if (Ptr->getOpcode() == ISD::FrameIndex) {
    auto Ty = TL.getPointerTy(DL, ADDRESS_SPACE_LOCAL);
    Ptr = DAG.getAddrSpaceCast(SDLoc(), Ty, Ptr, ADDRESS_SPACE_GENERIC,
                               ADDRESS_SPACE_LOCAL);

    return MachinePointerInfo(ADDRESS_SPACE_LOCAL);
  }

  // Peel of an addrspacecast to generic and load directly from the specific
  // address space.
  if (Ptr->getOpcode() == ISD::ADDRSPACECAST) {
    const auto *ASC = cast<AddrSpaceCastSDNode>(Ptr);
    if (ASC->getDestAddressSpace() == ADDRESS_SPACE_GENERIC) {
      Ptr = ASC->getOperand(0);
      return MachinePointerInfo(ASC->getSrcAddressSpace());
    }
  }

  return MachinePointerInfo();
}

static ISD::NodeType getExtOpcode(const ISD::ArgFlagsTy &Flags) {
  if (Flags.isSExt())
    return ISD::SIGN_EXTEND;
  if (Flags.isZExt())
    return ISD::ZERO_EXTEND;
  return ISD::ANY_EXTEND;
}

static SDValue correctParamType(SDValue V, EVT ExpectedVT,
                                ISD::ArgFlagsTy Flags, SelectionDAG &DAG,
                                SDLoc dl) {
  const EVT ActualVT = V.getValueType();
  assert((ActualVT == ExpectedVT ||
          (ExpectedVT.isInteger() && ActualVT.isInteger())) &&
         "Non-integer argument type size mismatch");
  if (ExpectedVT.bitsGT(ActualVT))
    return DAG.getNode(getExtOpcode(Flags), dl, ExpectedVT, V);
  if (ExpectedVT.bitsLT(ActualVT))
    return DAG.getNode(ISD::TRUNCATE, dl, ExpectedVT, V);

  return V;
}

SDValue NVPTXTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                       SmallVectorImpl<SDValue> &InVals) const {

  if (CLI.IsVarArg && (STI.getPTXVersion() < 60 || STI.getSmVersion() < 30))
    report_fatal_error(
        "Support for variadic functions (unsized array parameter) introduced "
        "in PTX ISA version 6.0 and requires target sm_30.");

  SelectionDAG &DAG = CLI.DAG;
  SDLoc dl = CLI.DL;
  const SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Callee = CLI.Callee;
  ArgListTy &Args = CLI.getArgs();
  Type *RetTy = CLI.RetTy;
  const CallBase *CB = CLI.CB;
  const DataLayout &DL = DAG.getDataLayout();
  LLVMContext &Ctx = *DAG.getContext();

  const auto GetI32 = [&](const unsigned I) {
    return DAG.getConstant(I, dl, MVT::i32);
  };

  const unsigned UniqueCallSite = GlobalUniqueCallSite++;
  const SDValue CallChain = CLI.Chain;
  const SDValue StartChain =
      DAG.getCALLSEQ_START(CallChain, UniqueCallSite, 0, dl);
  SDValue DeclareGlue = StartChain.getValue(1);

  SmallVector<SDValue, 16> CallPrereqs{StartChain};

  const auto MakeDeclareScalarParam = [&](SDValue Symbol, unsigned Size) {
    // PTX ABI requires integral types to be at least 32 bits in size. FP16 is
    // loaded/stored using i16, so it's handled here as well.
    const unsigned SizeBits = promoteScalarArgumentSize(Size * 8);
    SDValue Declare =
        DAG.getNode(NVPTXISD::DeclareScalarParam, dl, {MVT::Other, MVT::Glue},
                    {StartChain, Symbol, GetI32(SizeBits), DeclareGlue});
    CallPrereqs.push_back(Declare);
    DeclareGlue = Declare.getValue(1);
    return Declare;
  };

  const auto MakeDeclareArrayParam = [&](SDValue Symbol, Align Align,
                                         unsigned Size) {
    SDValue Declare = DAG.getNode(
        NVPTXISD::DeclareArrayParam, dl, {MVT::Other, MVT::Glue},
        {StartChain, Symbol, GetI32(Align.value()), GetI32(Size), DeclareGlue});
    CallPrereqs.push_back(Declare);
    DeclareGlue = Declare.getValue(1);
    return Declare;
  };

  // Variadic arguments.
  //
  // Normally, for each argument, we declare a param scalar or a param
  // byte array in the .param space, and store the argument value to that
  // param scalar or array starting at offset 0.
  //
  // In the case of the first variadic argument, we declare a vararg byte array
  // with size 0. The exact size of this array isn't known at this point, so
  // it'll be patched later. All the variadic arguments will be stored to this
  // array at a certain offset (which gets tracked by 'VAOffset'). The offset is
  // initially set to 0, so it can be used for non-variadic arguments (which use
  // 0 offset) to simplify the code.
  //
  // After all vararg is processed, 'VAOffset' holds the size of the
  // vararg byte array.
  assert((CLI.IsVarArg || CLI.Args.size() == CLI.NumFixedArgs) &&
         "Non-VarArg function with extra arguments");

  const unsigned FirstVAArg = CLI.NumFixedArgs; // position of first variadic
  unsigned VAOffset = 0; // current offset in the param array

  const SDValue VADeclareParam =
      CLI.Args.size() > FirstVAArg
          ? MakeDeclareArrayParam(getCallParamSymbol(DAG, FirstVAArg, MVT::i32),
                                  Align(STI.getMaxRequiredAlignment()), 0)
          : SDValue();

  // Args.size() and Outs.size() need not match.
  // Outs.size() will be larger
  //   * if there is an aggregate argument with multiple fields (each field
  //     showing up separately in Outs)
  //   * if there is a vector argument with more than typical vector-length
  //     elements (generally if more than 4) where each vector element is
  //     individually present in Outs.
  // So a different index should be used for indexing into Outs/OutVals.
  // See similar issue in LowerFormalArguments.
  auto AllOuts = ArrayRef(CLI.Outs);
  auto AllOutVals = ArrayRef(CLI.OutVals);
  assert(AllOuts.size() == AllOutVals.size() &&
         "Outs and OutVals must be the same size");
  // Declare the .params or .reg need to pass values
  // to the function
  for (const auto E : llvm::enumerate(Args)) {
    const auto ArgI = E.index();
    const auto Arg = E.value();
    const auto ArgOuts =
        AllOuts.take_while([&](auto O) { return O.OrigArgIndex == ArgI; });
    const auto ArgOutVals = AllOutVals.take_front(ArgOuts.size());
    AllOuts = AllOuts.drop_front(ArgOuts.size());
    AllOutVals = AllOutVals.drop_front(ArgOuts.size());

    const bool IsVAArg = (ArgI >= FirstVAArg);
    const bool IsByVal = Arg.IsByVal;

    const SDValue ParamSymbol =
        getCallParamSymbol(DAG, IsVAArg ? FirstVAArg : ArgI, MVT::i32);

    assert((!IsByVal || Arg.IndirectType) &&
           "byval arg must have indirect type");
    Type *ETy = (IsByVal ? Arg.IndirectType : Arg.Ty);

    const Align ArgAlign = [&]() {
      if (IsByVal) {
        // The ByValAlign in the Outs[OIdx].Flags is always set at this point,
        // so we don't need to worry whether it's naturally aligned or not.
        // See TargetLowering::LowerCallTo().
        const Align InitialAlign = ArgOuts[0].Flags.getNonZeroByValAlign();
        return getFunctionByValParamAlign(CB->getCalledFunction(), ETy,
                                          InitialAlign, DL);
      }
      return getArgumentAlignment(CB, Arg.Ty, ArgI + 1, DL);
    }();

    const unsigned TySize = DL.getTypeAllocSize(ETy);
    assert((!IsByVal || TySize == ArgOuts[0].Flags.getByValSize()) &&
           "type size mismatch");

    const SDValue ArgDeclare = [&]() {
      if (IsVAArg)
        return VADeclareParam;

      if (IsByVal || shouldPassAsArray(Arg.Ty))
        return MakeDeclareArrayParam(ParamSymbol, ArgAlign, TySize);

      assert(ArgOuts.size() == 1 && "We must pass only one value as non-array");
      assert((ArgOuts[0].VT.isInteger() || ArgOuts[0].VT.isFloatingPoint()) &&
             "Only int and float types are supported as non-array arguments");

      return MakeDeclareScalarParam(ParamSymbol, TySize);
    }();

    if (IsByVal) {
      assert(ArgOutVals.size() == 1 && "We must pass only one value as byval");
      SDValue SrcPtr = ArgOutVals[0];
      const auto PointerInfo = refinePtrAS(SrcPtr, DAG, DL, *this);
      const Align BaseSrcAlign = ArgOuts[0].Flags.getNonZeroByValAlign();

      if (IsVAArg)
        VAOffset = alignTo(VAOffset, ArgAlign);

      SmallVector<EVT, 4> ValueVTs, MemVTs;
      SmallVector<TypeSize, 4> Offsets;
      ComputeValueVTs(*this, DL, ETy, ValueVTs, &MemVTs, &Offsets);

      unsigned J = 0;
      const auto VI = VectorizePTXValueVTs(MemVTs, Offsets, ArgAlign, IsVAArg);
      for (const unsigned NumElts : VI) {
        EVT LoadVT = getVectorizedVT(MemVTs[J], NumElts, Ctx);
        Align SrcAlign = commonAlignment(BaseSrcAlign, Offsets[J]);
        SDValue SrcAddr = DAG.getObjectPtrOffset(dl, SrcPtr, Offsets[J]);
        SDValue SrcLoad =
            DAG.getLoad(LoadVT, dl, CallChain, SrcAddr, PointerInfo, SrcAlign);

        TypeSize ParamOffset = Offsets[J].getWithIncrement(VAOffset);
        Align ParamAlign = commonAlignment(ArgAlign, ParamOffset);
        SDValue ParamAddr =
            DAG.getObjectPtrOffset(dl, ParamSymbol, ParamOffset);
        SDValue StoreParam =
            DAG.getStore(ArgDeclare, dl, SrcLoad, ParamAddr,
                         MachinePointerInfo(ADDRESS_SPACE_PARAM), ParamAlign);
        CallPrereqs.push_back(StoreParam);

        J += NumElts;
      }
      if (IsVAArg)
        VAOffset += TySize;
    } else {
      SmallVector<EVT, 16> VTs;
      SmallVector<uint64_t, 16> Offsets;
      ComputePTXValueVTs(*this, DL, Arg.Ty, VTs, &Offsets, VAOffset);
      assert(VTs.size() == Offsets.size() && "Size mismatch");
      assert(VTs.size() == ArgOuts.size() && "Size mismatch");

      // PTX Interoperability Guide 3.3(A): [Integer] Values shorter
      // than 32-bits are sign extended or zero extended, depending on
      // whether they are signed or unsigned types. This case applies
      // only to scalar parameters and not to aggregate values.
      const bool ExtendIntegerParam =
          Arg.Ty->isIntegerTy() && DL.getTypeAllocSizeInBits(Arg.Ty) < 32;

      const auto GetStoredValue = [&](const unsigned I) {
        SDValue StVal = ArgOutVals[I];
        assert(promoteScalarIntegerPTX(StVal.getValueType()) ==
                   StVal.getValueType() &&
               "OutVal type should always be legal");

        const EVT VTI = promoteScalarIntegerPTX(VTs[I]);
        const EVT StoreVT =
            ExtendIntegerParam ? MVT::i32 : (VTI == MVT::i1 ? MVT::i8 : VTI);

        return correctParamType(StVal, StoreVT, ArgOuts[I].Flags, DAG, dl);
      };

      unsigned J = 0;
      const auto VI = VectorizePTXValueVTs(VTs, Offsets, ArgAlign, IsVAArg);
      for (const unsigned NumElts : VI) {
        const EVT EltVT = promoteScalarIntegerPTX(VTs[J]);

        unsigned Offset;
        if (IsVAArg) {
          // TODO: We may need to support vector types that can be passed
          // as scalars in variadic arguments.
          assert(NumElts == 1 &&
                 "Vectorization should be disabled for vaargs.");

          // Align each part of the variadic argument to their type.
          VAOffset = alignTo(VAOffset, DAG.getEVTAlign(EltVT));
          Offset = VAOffset;

          const EVT TheStoreType = ExtendIntegerParam ? MVT::i32 : EltVT;
          VAOffset += DL.getTypeAllocSize(TheStoreType.getTypeForEVT(Ctx));
        } else {
          assert(VAOffset == 0 && "VAOffset must be 0 for non-VA args");
          Offset = Offsets[J];
        }

        SDValue Ptr =
            DAG.getObjectPtrOffset(dl, ParamSymbol, TypeSize::getFixed(Offset));

        const MaybeAlign CurrentAlign = ExtendIntegerParam
                                            ? MaybeAlign(std::nullopt)
                                            : commonAlignment(ArgAlign, Offset);

        SDValue Val =
            getBuildVectorizedValue(NumElts, dl, DAG, [&](unsigned K) {
              return GetStoredValue(J + K);
            });

        SDValue StoreParam =
            DAG.getStore(ArgDeclare, dl, Val, Ptr,
                         MachinePointerInfo(ADDRESS_SPACE_PARAM), CurrentAlign);
        CallPrereqs.push_back(StoreParam);

        J += NumElts;
      }
    }
  }

  // Handle Result
  if (!Ins.empty()) {
    const SDValue RetSymbol = DAG.getExternalSymbol("retval0", MVT::i32);
    const unsigned ResultSize = DL.getTypeAllocSize(RetTy);
    if (shouldPassAsArray(RetTy)) {
      const Align RetAlign = getArgumentAlignment(CB, RetTy, 0, DL);
      MakeDeclareArrayParam(RetSymbol, RetAlign, ResultSize);
    } else {
      MakeDeclareScalarParam(RetSymbol, ResultSize);
    }
  }

  // Set the size of the vararg param byte array if the callee is a variadic
  // function and the variadic part is not empty.
  if (VADeclareParam) {
    SDValue DeclareParamOps[] = {VADeclareParam.getOperand(0),
                                 VADeclareParam.getOperand(1),
                                 VADeclareParam.getOperand(2), GetI32(VAOffset),
                                 VADeclareParam.getOperand(4)};
    DAG.MorphNodeTo(VADeclareParam.getNode(), VADeclareParam.getOpcode(),
                    VADeclareParam->getVTList(), DeclareParamOps);
  }

  const auto *Func = dyn_cast<GlobalAddressSDNode>(Callee.getNode());
  // If the type of the callsite does not match that of the function, convert
  // the callsite to an indirect call.
  const bool ConvertToIndirectCall = shouldConvertToIndirectCall(CB, Func);

  // Both indirect calls and libcalls have nullptr Func. In order to distinguish
  // between them we must rely on the call site value which is valid for
  // indirect calls but is always null for libcalls.
  const bool IsIndirectCall = (!Func && CB) || ConvertToIndirectCall;

  if (isa<ExternalSymbolSDNode>(Callee)) {
    Function* CalleeFunc = nullptr;

    // Try to find the callee in the current module.
    Callee = DAG.getSymbolFunctionGlobalAddress(Callee, &CalleeFunc);
    assert(CalleeFunc != nullptr && "Libcall callee must be set.");

    // Set the "libcall callee" attribute to indicate that the function
    // must always have a declaration.
    CalleeFunc->addFnAttr("nvptx-libcall-callee", "true");
  }

  if (IsIndirectCall) {
    // This is indirect function call case : PTX requires a prototype of the
    // form
    // proto_0 : .callprototype(.param .b32 _) _ (.param .b32 _);
    // to be emitted, and the label has to used as the last arg of call
    // instruction.
    // The prototype is embedded in a string and put as the operand for a
    // CallPrototype SDNode which will print out to the value of the string.
    const bool HasVAArgs = CLI.IsVarArg && (CLI.Args.size() > CLI.NumFixedArgs);
    std::string Proto =
        getPrototype(DL, RetTy, Args, CLI.Outs,
                     HasVAArgs ? std::optional(FirstVAArg) : std::nullopt, *CB,
                     UniqueCallSite);
    const char *ProtoStr = nvTM->getStrPool().save(Proto).data();
    const SDValue PrototypeDeclare = DAG.getNode(
        NVPTXISD::CallPrototype, dl, MVT::Other,
        {StartChain, DAG.getTargetExternalSymbol(ProtoStr, MVT::i32)});
    CallPrereqs.push_back(PrototypeDeclare);
  }

  const unsigned Proto = IsIndirectCall ? UniqueCallSite : 0;
  const unsigned NumArgs =
      std::min<unsigned>(CLI.NumFixedArgs + 1, Args.size());
  /// CALL(Chain, IsConvergent, IsIndirectCall/IsUniform, NumReturns,
  ///      NumParams, Callee, Proto)
  const SDValue CallToken = DAG.getTokenFactor(dl, CallPrereqs);
  const SDValue Call = DAG.getNode(
      NVPTXISD::CALL, dl, MVT::Other,
      {CallToken, GetI32(CLI.IsConvergent), GetI32(IsIndirectCall),
       GetI32(Ins.empty() ? 0 : 1), GetI32(NumArgs), Callee, GetI32(Proto)});

  SmallVector<SDValue, 16> LoadChains{Call};
  SmallVector<SDValue, 16> ProxyRegOps;
  if (!Ins.empty()) {
    SmallVector<EVT, 16> VTs;
    SmallVector<uint64_t, 16> Offsets;
    ComputePTXValueVTs(*this, DL, RetTy, VTs, &Offsets);
    assert(VTs.size() == Ins.size() && "Bad value decomposition");

    const Align RetAlign = getArgumentAlignment(CB, RetTy, 0, DL);
    const SDValue RetSymbol = DAG.getExternalSymbol("retval0", MVT::i32);

    // PTX Interoperability Guide 3.3(A): [Integer] Values shorter than
    // 32-bits are sign extended or zero extended, depending on whether
    // they are signed or unsigned types.
    const bool ExtendIntegerRetVal =
        RetTy->isIntegerTy() && DL.getTypeAllocSizeInBits(RetTy) < 32;

    unsigned I = 0;
    const auto VI = VectorizePTXValueVTs(VTs, Offsets, RetAlign);
    for (const unsigned NumElts : VI) {
      const MaybeAlign CurrentAlign =
          ExtendIntegerRetVal ? MaybeAlign(std::nullopt)
                              : commonAlignment(RetAlign, Offsets[I]);

      const EVT VTI = promoteScalarIntegerPTX(VTs[I]);
      const EVT LoadVT =
          ExtendIntegerRetVal ? MVT::i32 : (VTI == MVT::i1 ? MVT::i8 : VTI);
      const EVT VecVT = getVectorizedVT(LoadVT, NumElts, Ctx);
      SDValue Ptr =
          DAG.getObjectPtrOffset(dl, RetSymbol, TypeSize::getFixed(Offsets[I]));

      SDValue R =
          DAG.getLoad(VecVT, dl, Call, Ptr,
                      MachinePointerInfo(ADDRESS_SPACE_PARAM), CurrentAlign);

      LoadChains.push_back(R.getValue(1));
      for (const unsigned J : llvm::seq(NumElts))
        ProxyRegOps.push_back(getExtractVectorizedValue(R, J, LoadVT, dl, DAG));
      I += NumElts;
    }
  }

  const SDValue EndToken = DAG.getTokenFactor(dl, LoadChains);
  const SDValue CallEnd = DAG.getCALLSEQ_END(EndToken, UniqueCallSite,
                                             UniqueCallSite + 1, SDValue(), dl);

  // Append ProxyReg instructions to the chain to make sure that `callseq_end`
  // will not get lost. Otherwise, during libcalls expansion, the nodes can become
  // dangling.
  for (const auto [I, Reg] : llvm::enumerate(ProxyRegOps)) {
    SDValue Proxy =
        DAG.getNode(NVPTXISD::ProxyReg, dl, Reg.getValueType(), {CallEnd, Reg});
    SDValue Ret = correctParamType(Proxy, Ins[I].VT, Ins[I].Flags, DAG, dl);
    InVals.push_back(Ret);
  }

  // set IsTailCall to false for now, until we figure out how to express
  // tail call optimization in PTX
  CLI.IsTailCall = false;
  return CallEnd;
}

SDValue NVPTXTargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                                     SelectionDAG &DAG) const {

  if (STI.getPTXVersion() < 73 || STI.getSmVersion() < 52) {
    const Function &Fn = DAG.getMachineFunction().getFunction();

    DAG.getContext()->diagnose(DiagnosticInfoUnsupported(
        Fn,
        "Support for dynamic alloca introduced in PTX ISA version 7.3 and "
        "requires target sm_52.",
        SDLoc(Op).getDebugLoc()));
    auto Ops = {DAG.getConstant(0, SDLoc(), Op.getValueType()),
                Op.getOperand(0)};
    return DAG.getMergeValues(Ops, SDLoc());
  }

  SDLoc DL(Op.getNode());
  SDValue Chain = Op.getOperand(0);
  SDValue Size = Op.getOperand(1);
  uint64_t Align = Op.getConstantOperandVal(2);

  // The alignment on a ISD::DYNAMIC_STACKALLOC node may be 0 to indicate that
  // the default stack alignment should be used.
  if (Align == 0)
    Align = DAG.getSubtarget().getFrameLowering()->getStackAlign().value();

  // The size for ptx alloca instruction is 64-bit for m64 and 32-bit for m32.
  const MVT LocalVT = getPointerTy(DAG.getDataLayout(), ADDRESS_SPACE_LOCAL);

  SDValue Alloc =
      DAG.getNode(NVPTXISD::DYNAMIC_STACKALLOC, DL, {LocalVT, MVT::Other},
                  {Chain, DAG.getZExtOrTrunc(Size, DL, LocalVT),
                   DAG.getTargetConstant(Align, DL, MVT::i32)});

  SDValue ASC = DAG.getAddrSpaceCast(
      DL, Op.getValueType(), Alloc, ADDRESS_SPACE_LOCAL, ADDRESS_SPACE_GENERIC);

  return DAG.getMergeValues({ASC, SDValue(Alloc.getNode(), 1)}, DL);
}

SDValue NVPTXTargetLowering::LowerSTACKRESTORE(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Op.getNode());
  if (STI.getPTXVersion() < 73 || STI.getSmVersion() < 52) {
    const Function &Fn = DAG.getMachineFunction().getFunction();

    DAG.getContext()->diagnose(DiagnosticInfoUnsupported(
        Fn,
        "Support for stackrestore requires PTX ISA version >= 7.3 and target "
        ">= sm_52.",
        DL.getDebugLoc()));
    return Op.getOperand(0);
  }

  const MVT LocalVT = getPointerTy(DAG.getDataLayout(), ADDRESS_SPACE_LOCAL);
  SDValue Chain = Op.getOperand(0);
  SDValue Ptr = Op.getOperand(1);
  SDValue ASC = DAG.getAddrSpaceCast(DL, LocalVT, Ptr, ADDRESS_SPACE_GENERIC,
                                     ADDRESS_SPACE_LOCAL);
  return DAG.getNode(NVPTXISD::STACKRESTORE, DL, MVT::Other, {Chain, ASC});
}

SDValue NVPTXTargetLowering::LowerSTACKSAVE(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op.getNode());
  if (STI.getPTXVersion() < 73 || STI.getSmVersion() < 52) {
    const Function &Fn = DAG.getMachineFunction().getFunction();

    DAG.getContext()->diagnose(DiagnosticInfoUnsupported(
        Fn,
        "Support for stacksave requires PTX ISA version >= 7.3 and target >= "
        "sm_52.",
        DL.getDebugLoc()));
    auto Ops = {DAG.getConstant(0, DL, Op.getValueType()), Op.getOperand(0)};
    return DAG.getMergeValues(Ops, DL);
  }

  const MVT LocalVT = getPointerTy(DAG.getDataLayout(), ADDRESS_SPACE_LOCAL);
  SDValue Chain = Op.getOperand(0);
  SDValue SS =
      DAG.getNode(NVPTXISD::STACKSAVE, DL, {LocalVT, MVT::Other}, Chain);
  SDValue ASC = DAG.getAddrSpaceCast(
      DL, Op.getValueType(), SS, ADDRESS_SPACE_LOCAL, ADDRESS_SPACE_GENERIC);
  return DAG.getMergeValues({ASC, SDValue(SS.getNode(), 1)}, DL);
}

// By default CONCAT_VECTORS is lowered by ExpandVectorBuildThroughStack()
// (see LegalizeDAG.cpp). This is slow and uses local memory.
// We use extract/insert/build vector just as what LegalizeOp() does in llvm 2.5
SDValue
NVPTXTargetLowering::LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  SDLoc dl(Node);
  SmallVector<SDValue, 8> Ops;
  unsigned NumOperands = Node->getNumOperands();
  for (unsigned i = 0; i < NumOperands; ++i) {
    SDValue SubOp = Node->getOperand(i);
    EVT VVT = SubOp.getNode()->getValueType(0);
    EVT EltVT = VVT.getVectorElementType();
    unsigned NumSubElem = VVT.getVectorNumElements();
    for (unsigned j = 0; j < NumSubElem; ++j) {
      Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, SubOp,
                                DAG.getIntPtrConstant(j, dl)));
    }
  }
  return DAG.getBuildVector(Node->getValueType(0), dl, Ops);
}

static SDValue getPRMT(SDValue A, SDValue B, SDValue Selector, SDLoc DL,
                       SelectionDAG &DAG,
                       unsigned Mode = NVPTX::PTXPrmtMode::NONE) {
  assert(A.getValueType() == MVT::i32 && B.getValueType() == MVT::i32 &&
         Selector.getValueType() == MVT::i32 && "PRMT must have i32 operands");
  return DAG.getNode(NVPTXISD::PRMT, DL, MVT::i32,
                     {A, B, Selector, DAG.getConstant(Mode, DL, MVT::i32)});
}

static SDValue getPRMT(SDValue A, SDValue B, uint64_t Selector, SDLoc DL,
                       SelectionDAG &DAG,
                       unsigned Mode = NVPTX::PTXPrmtMode::NONE) {
  return getPRMT(A, B, DAG.getConstant(Selector, DL, MVT::i32), DL, DAG, Mode);
}

/// Reduces the elements using the scalar operations provided. The operations
/// are sorted descending in number of inputs they take. The flags on the
/// original reduction operation will be propagated to each scalar operation.
/// Nearby elements are grouped in tree reduction, unlike the shuffle reduction
/// used in ExpandReductions and SelectionDAG.
static SDValue buildTreeReduction(
    const SmallVector<SDValue> &Elements, EVT EltTy,
    ArrayRef<std::pair<unsigned /*NodeType*/, unsigned /*NumInputs*/>> Ops,
    const SDLoc &DL, const SDNodeFlags Flags, SelectionDAG &DAG) {
  // Build the reduction tree at each level, starting with all the elements.
  SmallVector<SDValue> Level = Elements;

  unsigned OpIdx = 0;
  while (Level.size() > 1) {
    // Try to reduce this level using the current operator.
    const auto [Op, NumInputs] = Ops[OpIdx];

    // Build the next level by partially reducing all elements.
    SmallVector<SDValue> ReducedLevel;
    unsigned I = 0, E = Level.size();
    for (; I + NumInputs <= E; I += NumInputs) {
      // Reduce elements in groups of [NumInputs], as much as possible.
      ReducedLevel.push_back(DAG.getNode(
          Op, DL, EltTy, ArrayRef<SDValue>(Level).slice(I, NumInputs), Flags));
    }

    if (I < E) {
      // Handle leftover elements.

      if (ReducedLevel.empty()) {
        // We didn't reduce anything at this level. We need to pick a smaller
        // operator.
        ++OpIdx;
        assert(OpIdx < Ops.size() && "no smaller operators for reduction");
        continue;
      }

      // We reduced some things but there's still more left, meaning the
      // operator's number of inputs doesn't evenly divide this level size. Move
      // these elements to the next level.
      for (; I < E; ++I)
        ReducedLevel.push_back(Level[I]);
    }

    // Process the next level.
    Level = ReducedLevel;
  }

  return *Level.begin();
}

// Get scalar reduction opcode
static ISD::NodeType getScalarOpcodeForReduction(unsigned ReductionOpcode) {
  switch (ReductionOpcode) {
  case ISD::VECREDUCE_FMAX:
    return ISD::FMAXNUM;
  case ISD::VECREDUCE_FMIN:
    return ISD::FMINNUM;
  case ISD::VECREDUCE_FMAXIMUM:
    return ISD::FMAXIMUM;
  case ISD::VECREDUCE_FMINIMUM:
    return ISD::FMINIMUM;
  default:
    llvm_unreachable("unhandled reduction opcode");
  }
}

/// Get 3-input scalar reduction opcode
static std::optional<NVPTXISD::NodeType>
getScalar3OpcodeForReduction(unsigned ReductionOpcode) {
  switch (ReductionOpcode) {
  case ISD::VECREDUCE_FMAX:
    return NVPTXISD::FMAXNUM3;
  case ISD::VECREDUCE_FMIN:
    return NVPTXISD::FMINNUM3;
  case ISD::VECREDUCE_FMAXIMUM:
    return NVPTXISD::FMAXIMUM3;
  case ISD::VECREDUCE_FMINIMUM:
    return NVPTXISD::FMINIMUM3;
  default:
    return std::nullopt;
  }
}

/// Lower reductions to either a sequence of operations or a tree if
/// reassociations are allowed. This method will use larger operations like
/// max3/min3 when the target supports them.
SDValue NVPTXTargetLowering::LowerVECREDUCE(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const SDNodeFlags Flags = Op->getFlags();
  SDValue Vector = Op.getOperand(0);

  const unsigned Opcode = Op->getOpcode();
  const EVT EltTy = Vector.getValueType().getVectorElementType();

  // Whether we can use 3-input min/max when expanding the reduction.
  const bool CanUseMinMax3 =
      EltTy == MVT::f32 && STI.getSmVersion() >= 100 &&
      STI.getPTXVersion() >= 88 &&
      (Opcode == ISD::VECREDUCE_FMAX || Opcode == ISD::VECREDUCE_FMIN ||
       Opcode == ISD::VECREDUCE_FMAXIMUM || Opcode == ISD::VECREDUCE_FMINIMUM);

  // A list of SDNode opcodes with equivalent semantics, sorted descending by
  // number of inputs they take.
  SmallVector<std::pair<unsigned /*Op*/, unsigned /*NumIn*/>, 2> ScalarOps;

  if (auto Opcode3Elem = getScalar3OpcodeForReduction(Opcode);
      CanUseMinMax3 && Opcode3Elem)
    ScalarOps.push_back({*Opcode3Elem, 3});
  ScalarOps.push_back({getScalarOpcodeForReduction(Opcode), 2});

  SmallVector<SDValue> Elements;
  DAG.ExtractVectorElements(Vector, Elements);

  return buildTreeReduction(Elements, EltTy, ScalarOps, DL, Flags, DAG);
}

SDValue NVPTXTargetLowering::LowerBITCAST(SDValue Op, SelectionDAG &DAG) const {
  // Handle bitcasting from v2i8 without hitting the default promotion
  // strategy which goes through stack memory.
  EVT FromVT = Op->getOperand(0)->getValueType(0);
  if (FromVT != MVT::v2i8) {
    return Op;
  }

  // Pack vector elements into i16 and bitcast to final type
  SDLoc DL(Op);
  SDValue Vec0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i8,
                             Op->getOperand(0), DAG.getIntPtrConstant(0, DL));
  SDValue Vec1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i8,
                             Op->getOperand(0), DAG.getIntPtrConstant(1, DL));
  SDValue Extend0 = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i16, Vec0);
  SDValue Extend1 = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i16, Vec1);
  SDValue Const8 = DAG.getConstant(8, DL, MVT::i16);
  SDValue AsInt = DAG.getNode(
      ISD::OR, DL, MVT::i16,
      {Extend0, DAG.getNode(ISD::SHL, DL, MVT::i16, {Extend1, Const8})});
  EVT ToVT = Op->getValueType(0);
  return DAG.getBitcast(ToVT, AsInt);
}

// We can init constant f16x2/v2i16/v4i8 with a single .b32 move.  Normally it
// would get lowered as two constant loads and vector-packing move.
// Instead we want just a constant move:
//        mov.b32         %r2, 0x40003C00
SDValue NVPTXTargetLowering::LowerBUILD_VECTOR(SDValue Op,
                                               SelectionDAG &DAG) const {
  EVT VT = Op->getValueType(0);
  if (!(NVPTX::isPackedVectorTy(VT) && VT.is32BitVector()))
    return Op;
  SDLoc DL(Op);

  if (!llvm::all_of(Op->ops(), [](SDValue Operand) {
        return Operand->isUndef() || isa<ConstantSDNode>(Operand) ||
               isa<ConstantFPSDNode>(Operand);
      })) {
    if (VT != MVT::v4i8)
      return Op;
    // Lower non-const v4i8 vector as byte-wise constructed i32, which allows us
    // to optimize calculation of constant parts.
    auto GetPRMT = [&](const SDValue Left, const SDValue Right, bool Cast,
                       uint64_t SelectionValue) -> SDValue {
      SDValue L = Left;
      SDValue R = Right;
      if (Cast) {
        L = DAG.getAnyExtOrTrunc(L, DL, MVT::i32);
        R = DAG.getAnyExtOrTrunc(R, DL, MVT::i32);
      }
      return getPRMT(L, R, SelectionValue, DL, DAG);
    };
    auto PRMT__10 = GetPRMT(Op->getOperand(0), Op->getOperand(1), true, 0x3340);
    auto PRMT__32 = GetPRMT(Op->getOperand(2), Op->getOperand(3), true, 0x3340);
    auto PRMT3210 = GetPRMT(PRMT__10, PRMT__32, false, 0x5410);
    return DAG.getBitcast(VT, PRMT3210);
  }

  // Get value or the Nth operand as an APInt(32). Undef values treated as 0.
  auto GetOperand = [](SDValue Op, int N) -> APInt {
    const SDValue &Operand = Op->getOperand(N);
    EVT VT = Op->getValueType(0);
    if (Operand->isUndef())
      return APInt(32, 0);
    APInt Value;
    if (VT == MVT::v2f16 || VT == MVT::v2bf16)
      Value = cast<ConstantFPSDNode>(Operand)->getValueAPF().bitcastToAPInt();
    else if (VT == MVT::v2i16 || VT == MVT::v4i8)
      Value = Operand->getAsAPIntVal();
    else
      llvm_unreachable("Unsupported type");
    // i8 values are carried around as i16, so we need to zero out upper bits,
    // so they do not get in the way of combining individual byte values
    if (VT == MVT::v4i8)
      Value = Value.trunc(8);
    return Value.zext(32);
  };

  // Construct a 32-bit constant by shifting into place smaller values
  // (elements of the vector type VT).
  // For example, if VT has 2 elements, then N == 2:
  //   ShiftAmount = 32 / N = 16
  //   Value |= Op0 (b16) << 0
  //   Value |= Op1 (b16) << 16
  // If N == 4:
  //   ShiftAmount = 32 / N = 8
  //   Value |= Op0 (b8) << 0
  //   Value |= Op1 (b8) << 8
  //   Value |= Op2 (b8) << 16
  //   Value |= Op3 (b8) << 24
  // ...etc
  APInt Value(32, 0);
  const unsigned NumElements = VT.getVectorNumElements();
  assert(32 % NumElements == 0 && "must evenly divide bit length");
  const unsigned ShiftAmount = 32 / NumElements;
  for (unsigned ElementNo : seq(NumElements))
    Value |= GetOperand(Op, ElementNo).shl(ElementNo * ShiftAmount);
  SDValue Const = DAG.getConstant(Value, DL, MVT::i32);
  return DAG.getNode(ISD::BITCAST, DL, Op->getValueType(0), Const);
}

SDValue NVPTXTargetLowering::LowerEXTRACT_VECTOR_ELT(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDValue Index = Op->getOperand(1);
  SDValue Vector = Op->getOperand(0);
  SDLoc DL(Op);
  EVT VectorVT = Vector.getValueType();

  if (VectorVT == MVT::v4i8) {
    SDValue Selector = DAG.getNode(ISD::OR, DL, MVT::i32,
                                   DAG.getZExtOrTrunc(Index, DL, MVT::i32),
                                   DAG.getConstant(0x7770, DL, MVT::i32));
    SDValue PRMT = getPRMT(DAG.getBitcast(MVT::i32, Vector),
                           DAG.getConstant(0, DL, MVT::i32), Selector, DL, DAG);
    SDValue Ext = DAG.getAnyExtOrTrunc(PRMT, DL, Op->getValueType(0));
    SDNodeFlags Flags;
    Flags.setNoSignedWrap(Ext.getScalarValueSizeInBits() > 8);
    Flags.setNoUnsignedWrap(Ext.getScalarValueSizeInBits() >= 8);
    Ext->setFlags(Flags);
    return Ext;
  }

  // Constant index will be matched by tablegen.
  if (isa<ConstantSDNode>(Index.getNode()))
    return Op;

  // Extract individual elements and select one of them.
  assert(NVPTX::isPackedVectorTy(VectorVT) &&
         VectorVT.getVectorNumElements() == 2 && "Unexpected vector type.");
  EVT EltVT = VectorVT.getVectorElementType();

  SDLoc dl(Op.getNode());
  SDValue E0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, Vector,
                           DAG.getIntPtrConstant(0, dl));
  SDValue E1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, Vector,
                           DAG.getIntPtrConstant(1, dl));
  return DAG.getSelectCC(dl, Index, DAG.getIntPtrConstant(0, dl), E0, E1,
                         ISD::CondCode::SETEQ);
}

SDValue NVPTXTargetLowering::LowerINSERT_VECTOR_ELT(SDValue Op,
                                                    SelectionDAG &DAG) const {
  SDValue Vector = Op->getOperand(0);
  EVT VectorVT = Vector.getValueType();

  if (VectorVT != MVT::v4i8)
    return Op;
  SDLoc DL(Op);
  SDValue Value = Op->getOperand(1);
  if (Value->isUndef())
    return Vector;

  SDValue Index = Op->getOperand(2);

  SDValue BFI =
      DAG.getNode(NVPTXISD::BFI, DL, MVT::i32,
                  {DAG.getZExtOrTrunc(Value, DL, MVT::i32), Vector,
                   DAG.getNode(ISD::MUL, DL, MVT::i32,
                               DAG.getZExtOrTrunc(Index, DL, MVT::i32),
                               DAG.getConstant(8, DL, MVT::i32)),
                   DAG.getConstant(8, DL, MVT::i32)});
  return DAG.getNode(ISD::BITCAST, DL, Op->getValueType(0), BFI);
}

SDValue NVPTXTargetLowering::LowerVECTOR_SHUFFLE(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDValue V1 = Op.getOperand(0);
  EVT VectorVT = V1.getValueType();
  if (VectorVT != MVT::v4i8 || Op.getValueType() != MVT::v4i8)
    return Op;

  // Lower shuffle to PRMT instruction.
  const ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op.getNode());
  SDValue V2 = Op.getOperand(1);
  uint32_t Selector = 0;
  for (auto I : llvm::enumerate(SVN->getMask())) {
    if (I.value() != -1) // -1 is a placeholder for undef.
      Selector |= (I.value() << (I.index() * 4));
  }

  SDLoc DL(Op);
  SDValue PRMT = getPRMT(DAG.getBitcast(MVT::i32, V1),
                         DAG.getBitcast(MVT::i32, V2), Selector, DL, DAG);
  return DAG.getBitcast(Op.getValueType(), PRMT);
}
/// LowerShiftRightParts - Lower SRL_PARTS, SRA_PARTS, which
/// 1) returns two i32 values and take a 2 x i32 value to shift plus a shift
///    amount, or
/// 2) returns two i64 values and take a 2 x i64 value to shift plus a shift
///    amount.
SDValue NVPTXTargetLowering::LowerShiftRightParts(SDValue Op,
                                                  SelectionDAG &DAG) const {
  assert(Op.getNumOperands() == 3 && "Not a double-shift!");
  assert(Op.getOpcode() == ISD::SRA_PARTS || Op.getOpcode() == ISD::SRL_PARTS);

  EVT VT = Op.getValueType();
  unsigned VTBits = VT.getSizeInBits();
  SDLoc dl(Op);
  SDValue ShOpLo = Op.getOperand(0);
  SDValue ShOpHi = Op.getOperand(1);
  SDValue ShAmt  = Op.getOperand(2);
  unsigned Opc = (Op.getOpcode() == ISD::SRA_PARTS) ? ISD::SRA : ISD::SRL;

  if (VTBits == 32 && STI.getSmVersion() >= 35) {
    // For 32bit and sm35, we can use the funnel shift 'shf' instruction.
    // {dHi, dLo} = {aHi, aLo} >> Amt
    //   dHi = aHi >> Amt
    //   dLo = shf.r.clamp aLo, aHi, Amt

    SDValue Hi = DAG.getNode(Opc, dl, VT, ShOpHi, ShAmt);
    SDValue Lo =
        DAG.getNode(NVPTXISD::FSHR_CLAMP, dl, VT, ShOpHi, ShOpLo, ShAmt);

    SDValue Ops[2] = { Lo, Hi };
    return DAG.getMergeValues(Ops, dl);
  }
  else {
    // {dHi, dLo} = {aHi, aLo} >> Amt
    // - if (Amt>=size) then
    //      dLo = aHi >> (Amt-size)
    //      dHi = aHi >> Amt (this is either all 0 or all 1)
    //   else
    //      dLo = (aLo >>logic Amt) | (aHi << (size-Amt))
    //      dHi = aHi >> Amt

    SDValue RevShAmt = DAG.getNode(ISD::SUB, dl, MVT::i32,
                                   DAG.getConstant(VTBits, dl, MVT::i32),
                                   ShAmt);
    SDValue Tmp1 = DAG.getNode(ISD::SRL, dl, VT, ShOpLo, ShAmt);
    SDValue ExtraShAmt = DAG.getNode(ISD::SUB, dl, MVT::i32, ShAmt,
                                     DAG.getConstant(VTBits, dl, MVT::i32));
    SDValue Tmp2 = DAG.getNode(ISD::SHL, dl, VT, ShOpHi, RevShAmt);
    SDValue FalseVal = DAG.getNode(ISD::OR, dl, VT, Tmp1, Tmp2);
    SDValue TrueVal = DAG.getNode(Opc, dl, VT, ShOpHi, ExtraShAmt);

    SDValue Cmp = DAG.getSetCC(dl, MVT::i1, ShAmt,
                               DAG.getConstant(VTBits, dl, MVT::i32),
                               ISD::SETGE);
    SDValue Hi = DAG.getNode(Opc, dl, VT, ShOpHi, ShAmt);
    SDValue Lo = DAG.getNode(ISD::SELECT, dl, VT, Cmp, TrueVal, FalseVal);

    SDValue Ops[2] = { Lo, Hi };
    return DAG.getMergeValues(Ops, dl);
  }
}

/// LowerShiftLeftParts - Lower SHL_PARTS, which
/// 1) returns two i32 values and take a 2 x i32 value to shift plus a shift
///    amount, or
/// 2) returns two i64 values and take a 2 x i64 value to shift plus a shift
///    amount.
SDValue NVPTXTargetLowering::LowerShiftLeftParts(SDValue Op,
                                                 SelectionDAG &DAG) const {
  assert(Op.getNumOperands() == 3 && "Not a double-shift!");
  assert(Op.getOpcode() == ISD::SHL_PARTS);

  EVT VT = Op.getValueType();
  unsigned VTBits = VT.getSizeInBits();
  SDLoc dl(Op);
  SDValue ShOpLo = Op.getOperand(0);
  SDValue ShOpHi = Op.getOperand(1);
  SDValue ShAmt  = Op.getOperand(2);

  if (VTBits == 32 && STI.getSmVersion() >= 35) {
    // For 32bit and sm35, we can use the funnel shift 'shf' instruction.
    // {dHi, dLo} = {aHi, aLo} << Amt
    //   dHi = shf.l.clamp aLo, aHi, Amt
    //   dLo = aLo << Amt

    SDValue Hi =
        DAG.getNode(NVPTXISD::FSHL_CLAMP, dl, VT, ShOpHi, ShOpLo, ShAmt);
    SDValue Lo = DAG.getNode(ISD::SHL, dl, VT, ShOpLo, ShAmt);

    SDValue Ops[2] = { Lo, Hi };
    return DAG.getMergeValues(Ops, dl);
  }
  else {
    // {dHi, dLo} = {aHi, aLo} << Amt
    // - if (Amt>=size) then
    //      dLo = aLo << Amt (all 0)
    //      dLo = aLo << (Amt-size)
    //   else
    //      dLo = aLo << Amt
    //      dHi = (aHi << Amt) | (aLo >> (size-Amt))

    SDValue RevShAmt = DAG.getNode(ISD::SUB, dl, MVT::i32,
                                   DAG.getConstant(VTBits, dl, MVT::i32),
                                   ShAmt);
    SDValue Tmp1 = DAG.getNode(ISD::SHL, dl, VT, ShOpHi, ShAmt);
    SDValue ExtraShAmt = DAG.getNode(ISD::SUB, dl, MVT::i32, ShAmt,
                                     DAG.getConstant(VTBits, dl, MVT::i32));
    SDValue Tmp2 = DAG.getNode(ISD::SRL, dl, VT, ShOpLo, RevShAmt);
    SDValue FalseVal = DAG.getNode(ISD::OR, dl, VT, Tmp1, Tmp2);
    SDValue TrueVal = DAG.getNode(ISD::SHL, dl, VT, ShOpLo, ExtraShAmt);

    SDValue Cmp = DAG.getSetCC(dl, MVT::i1, ShAmt,
                               DAG.getConstant(VTBits, dl, MVT::i32),
                               ISD::SETGE);
    SDValue Lo = DAG.getNode(ISD::SHL, dl, VT, ShOpLo, ShAmt);
    SDValue Hi = DAG.getNode(ISD::SELECT, dl, VT, Cmp, TrueVal, FalseVal);

    SDValue Ops[2] = { Lo, Hi };
    return DAG.getMergeValues(Ops, dl);
  }
}

/// If the types match, convert the generic copysign to the NVPTXISD version,
/// otherwise bail ensuring that mismatched cases are properly expaned.
SDValue NVPTXTargetLowering::LowerFCOPYSIGN(SDValue Op,
                                            SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  SDValue In1 = Op.getOperand(0);
  SDValue In2 = Op.getOperand(1);
  EVT SrcVT = In2.getValueType();

  if (!SrcVT.bitsEq(VT))
    return SDValue();

  return DAG.getNode(NVPTXISD::FCOPYSIGN, DL, VT, In1, In2);
}

SDValue NVPTXTargetLowering::LowerFROUND(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();

  if (VT == MVT::f32)
    return LowerFROUND32(Op, DAG);

  if (VT == MVT::f64)
    return LowerFROUND64(Op, DAG);

  llvm_unreachable("unhandled type");
}

// This is the the rounding method used in CUDA libdevice in C like code:
// float roundf(float A)
// {
//   float RoundedA = (float) (int) ( A > 0 ? (A + 0.5f) : (A - 0.5f));
//   RoundedA = abs(A) > 0x1.0p23 ? A : RoundedA;
//   return abs(A) < 0.5 ? (float)(int)A : RoundedA;
// }
SDValue NVPTXTargetLowering::LowerFROUND32(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue A = Op.getOperand(0);
  EVT VT = Op.getValueType();

  SDValue AbsA = DAG.getNode(ISD::FABS, SL, VT, A);

  // RoundedA = (float) (int) ( A > 0 ? (A + 0.5f) : (A - 0.5f))
  SDValue Bitcast  = DAG.getNode(ISD::BITCAST, SL, MVT::i32, A);
  const unsigned SignBitMask = 0x80000000;
  SDValue Sign = DAG.getNode(ISD::AND, SL, MVT::i32, Bitcast,
                             DAG.getConstant(SignBitMask, SL, MVT::i32));
  const unsigned PointFiveInBits = 0x3F000000;
  SDValue PointFiveWithSignRaw =
      DAG.getNode(ISD::OR, SL, MVT::i32, Sign,
                  DAG.getConstant(PointFiveInBits, SL, MVT::i32));
  SDValue PointFiveWithSign =
      DAG.getNode(ISD::BITCAST, SL, VT, PointFiveWithSignRaw);
  SDValue AdjustedA = DAG.getNode(ISD::FADD, SL, VT, A, PointFiveWithSign);
  SDValue RoundedA = DAG.getNode(ISD::FTRUNC, SL, VT, AdjustedA);

  // RoundedA = abs(A) > 0x1.0p23 ? A : RoundedA;
  EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
  SDValue IsLarge =
      DAG.getSetCC(SL, SetCCVT, AbsA, DAG.getConstantFP(pow(2.0, 23.0), SL, VT),
                   ISD::SETOGT);
  RoundedA = DAG.getNode(ISD::SELECT, SL, VT, IsLarge, A, RoundedA);

  // return abs(A) < 0.5 ? (float)(int)A : RoundedA;
  SDValue IsSmall =DAG.getSetCC(SL, SetCCVT, AbsA,
                                DAG.getConstantFP(0.5, SL, VT), ISD::SETOLT);
  SDValue RoundedAForSmallA = DAG.getNode(ISD::FTRUNC, SL, VT, A);
  return DAG.getNode(ISD::SELECT, SL, VT, IsSmall, RoundedAForSmallA, RoundedA);
}

// The implementation of round(double) is similar to that of round(float) in
// that they both separate the value range into three regions and use a method
// specific to the region to round the values. However, round(double) first
// calculates the round of the absolute value and then adds the sign back while
// round(float) directly rounds the value with sign.
SDValue NVPTXTargetLowering::LowerFROUND64(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue A = Op.getOperand(0);
  EVT VT = Op.getValueType();

  SDValue AbsA = DAG.getNode(ISD::FABS, SL, VT, A);

  // double RoundedA = (double) (int) (abs(A) + 0.5f);
  SDValue AdjustedA = DAG.getNode(ISD::FADD, SL, VT, AbsA,
                                  DAG.getConstantFP(0.5, SL, VT));
  SDValue RoundedA = DAG.getNode(ISD::FTRUNC, SL, VT, AdjustedA);

  // RoundedA = abs(A) < 0.5 ? (double)0 : RoundedA;
  EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
  SDValue IsSmall =DAG.getSetCC(SL, SetCCVT, AbsA,
                                DAG.getConstantFP(0.5, SL, VT), ISD::SETOLT);
  RoundedA = DAG.getNode(ISD::SELECT, SL, VT, IsSmall,
                         DAG.getConstantFP(0, SL, VT),
                         RoundedA);

  // Add sign to rounded_A
  RoundedA = DAG.getNode(ISD::FCOPYSIGN, SL, VT, RoundedA, A);
  DAG.getNode(ISD::FTRUNC, SL, VT, A);

  // RoundedA = abs(A) > 0x1.0p52 ? A : RoundedA;
  SDValue IsLarge =
      DAG.getSetCC(SL, SetCCVT, AbsA, DAG.getConstantFP(pow(2.0, 52.0), SL, VT),
                   ISD::SETOGT);
  return DAG.getNode(ISD::SELECT, SL, VT, IsLarge, A, RoundedA);
}

static SDValue PromoteBinOpToF32(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  EVT NVT = MVT::f32;
  if (VT.isVector()) {
    NVT = EVT::getVectorVT(*DAG.getContext(), NVT, VT.getVectorElementCount());
  }
  SDLoc DL(N);
  SDValue Tmp0 = DAG.getFPExtendOrRound(N->getOperand(0), DL, NVT);
  SDValue Tmp1 = DAG.getFPExtendOrRound(N->getOperand(1), DL, NVT);
  SDValue Res = DAG.getNode(N->getOpcode(), DL, NVT, Tmp0, Tmp1, N->getFlags());
  return DAG.getFPExtendOrRound(Res, DL, VT);
}

SDValue NVPTXTargetLowering::PromoteBinOpIfF32FTZ(SDValue Op,
                                                  SelectionDAG &DAG) const {
  if (useF32FTZ(DAG.getMachineFunction())) {
    return PromoteBinOpToF32(Op.getNode(), DAG);
  }
  return Op;
}

SDValue NVPTXTargetLowering::LowerINT_TO_FP(SDValue Op,
                                            SelectionDAG &DAG) const {
  assert(STI.getSmVersion() < 90 || STI.getPTXVersion() < 78);

  if (Op.getValueType() == MVT::bf16) {
    SDLoc Loc(Op);
    return DAG.getNode(
        ISD::FP_ROUND, Loc, MVT::bf16,
        DAG.getNode(Op.getOpcode(), Loc, MVT::f32, Op.getOperand(0)),
        DAG.getIntPtrConstant(0, Loc, /*isTarget=*/true));
  }

  // Everything else is considered legal.
  return Op;
}

SDValue NVPTXTargetLowering::LowerFP_TO_INT(SDValue Op,
                                            SelectionDAG &DAG) const {
  assert(STI.getSmVersion() < 90 || STI.getPTXVersion() < 78);

  if (Op.getOperand(0).getValueType() == MVT::bf16) {
    SDLoc Loc(Op);
    return DAG.getNode(
        Op.getOpcode(), Loc, Op.getValueType(),
        DAG.getNode(ISD::FP_EXTEND, Loc, MVT::f32, Op.getOperand(0)));
  }

  // Everything else is considered legal.
  return Op;
}

SDValue NVPTXTargetLowering::LowerFP_ROUND(SDValue Op,
                                           SelectionDAG &DAG) const {
  EVT NarrowVT = Op.getValueType();
  SDValue Wide = Op.getOperand(0);
  EVT WideVT = Wide.getValueType();
  if (NarrowVT.getScalarType() == MVT::bf16) {
    const TargetLowering *TLI = STI.getTargetLowering();
    if (STI.getSmVersion() < 80 || STI.getPTXVersion() < 70) {
      return TLI->expandFP_ROUND(Op.getNode(), DAG);
    }
    if (STI.getSmVersion() < 90 || STI.getPTXVersion() < 78) {
      // This combination was the first to support f32 -> bf16.
      if (STI.getSmVersion() >= 80 && STI.getPTXVersion() >= 70) {
        if (WideVT.getScalarType() == MVT::f32) {
          return Op;
        }
        if (WideVT.getScalarType() == MVT::f64) {
          SDLoc Loc(Op);
          // Round-inexact-to-odd f64 to f32, then do the final rounding using
          // the hardware f32 -> bf16 instruction.
          SDValue rod = TLI->expandRoundInexactToOdd(
              WideVT.isVector() ? WideVT.changeVectorElementType(MVT::f32)
                                : MVT::f32,
              Wide, Loc, DAG);
          return DAG.getFPExtendOrRound(rod, Loc, NarrowVT);
        }
      }
      return TLI->expandFP_ROUND(Op.getNode(), DAG);
    }
  }

  // Everything else is considered legal.
  return Op;
}

SDValue NVPTXTargetLowering::LowerFP_EXTEND(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDValue Narrow = Op.getOperand(0);
  EVT NarrowVT = Narrow.getValueType();
  EVT WideVT = Op.getValueType();
  if (NarrowVT.getScalarType() == MVT::bf16) {
    if (WideVT.getScalarType() == MVT::f32 &&
        (STI.getSmVersion() < 80 || STI.getPTXVersion() < 71)) {
      SDLoc Loc(Op);
      return DAG.getNode(ISD::BF16_TO_FP, Loc, WideVT, Narrow);
    }
    if (WideVT.getScalarType() == MVT::f64 &&
        (STI.getSmVersion() < 90 || STI.getPTXVersion() < 78)) {
      EVT F32 = NarrowVT.isVector() ? NarrowVT.changeVectorElementType(MVT::f32)
                                    : MVT::f32;
      SDLoc Loc(Op);
      if (STI.getSmVersion() >= 80 && STI.getPTXVersion() >= 71) {
        Op = DAG.getNode(ISD::FP_EXTEND, Loc, F32, Narrow);
      } else {
        Op = DAG.getNode(ISD::BF16_TO_FP, Loc, F32, Narrow);
      }
      return DAG.getNode(ISD::FP_EXTEND, Loc, WideVT, Op);
    }
  }

  // Everything else is considered legal.
  return Op;
}

static SDValue LowerVectorArith(SDValue Op, SelectionDAG &DAG) {
  SDLoc DL(Op);
  if (Op.getValueType() != MVT::v2i16)
    return Op;
  EVT EltVT = Op.getValueType().getVectorElementType();
  SmallVector<SDValue> VecElements;
  for (int I = 0, E = Op.getValueType().getVectorNumElements(); I < E; I++) {
    SmallVector<SDValue> ScalarArgs;
    llvm::transform(Op->ops(), std::back_inserter(ScalarArgs),
                    [&](const SDUse &O) {
                      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT,
                                         O.get(), DAG.getIntPtrConstant(I, DL));
                    });
    VecElements.push_back(DAG.getNode(Op.getOpcode(), DL, EltVT, ScalarArgs));
  }
  SDValue V =
      DAG.getNode(ISD::BUILD_VECTOR, DL, Op.getValueType(), VecElements);
  return V;
}

static SDValue LowerTcgen05St(SDValue Op, SelectionDAG &DAG) {
  SDNode *N = Op.getNode();
  SDLoc DL(N);
  SmallVector<SDValue, 32> Ops;

  // split the vector argument
  for (size_t I = 0; I < N->getNumOperands(); I++) {
    SDValue Val = N->getOperand(I);
    EVT ValVT = Val.getValueType();
    if (ValVT.isVector()) {
      EVT EltVT = ValVT.getVectorElementType();
      for (unsigned J = 0, NElts = ValVT.getVectorNumElements(); J < NElts; J++)
        Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, Val,
                                  DAG.getIntPtrConstant(J, DL)));
    } else
      Ops.push_back(Val);
  }

  MemIntrinsicSDNode *MemSD = cast<MemIntrinsicSDNode>(N);
  SDValue Tcgen05StNode =
      DAG.getMemIntrinsicNode(ISD::INTRINSIC_VOID, DL, N->getVTList(), Ops,
                              MemSD->getMemoryVT(), MemSD->getMemOperand());

  return Tcgen05StNode;
}

static SDValue LowerIntrinsicVoid(SDValue Op, SelectionDAG &DAG) {
  SDNode *N = Op.getNode();
  SDValue Intrin = N->getOperand(1);

  // Get the intrinsic ID
  unsigned IntrinNo = cast<ConstantSDNode>(Intrin.getNode())->getZExtValue();
  switch (IntrinNo) {
  default:
    break;
  case Intrinsic::nvvm_tcgen05_st_16x64b_x1:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x2:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x4:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x8:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x16:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x32:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x128:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x1:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x2:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x4:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x8:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x16:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x32:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x64:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x1:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x2:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x4:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x8:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x16:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x32:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x1:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x2:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x4:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x8:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x16:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x32:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x64:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x128:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x1:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x2:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x4:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x8:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x16:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x32:
  case Intrinsic::nvvm_tcgen05_st_16x64b_x64:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x64:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x128:
    return LowerTcgen05St(Op, DAG);
  }
  return Op;
}

static SDValue LowerClusterLaunchControlQueryCancel(SDValue Op,
                                                    SelectionDAG &DAG) {

  SDNode *N = Op.getNode();
  if (N->getOperand(1).getValueType() != MVT::i128) {
    // return, if the operand is already lowered
    return SDValue();
  }

  unsigned IID =
      cast<ConstantSDNode>(N->getOperand(0).getNode())->getZExtValue();
  auto Opcode = [&]() {
    switch (IID) {
    case Intrinsic::nvvm_clusterlaunchcontrol_query_cancel_is_canceled:
      return NVPTXISD::CLUSTERLAUNCHCONTROL_QUERY_CANCEL_IS_CANCELED;
    case Intrinsic::nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_x:
      return NVPTXISD::CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID_X;
    case Intrinsic::nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_y:
      return NVPTXISD::CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID_Y;
    case Intrinsic::nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_z:
      return NVPTXISD::CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID_Z;
    default:
      llvm_unreachable("unsupported/unhandled intrinsic");
    }
  }();

  SDLoc DL(N);
  SDValue TryCancelResponse = N->getOperand(1);
  SDValue Cast = DAG.getNode(ISD::BITCAST, DL, MVT::v2i64, TryCancelResponse);
  SDValue TryCancelResponse0 =
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i64, Cast,
                  DAG.getIntPtrConstant(0, DL));
  SDValue TryCancelResponse1 =
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i64, Cast,
                  DAG.getIntPtrConstant(1, DL));

  return DAG.getNode(Opcode, DL, N->getVTList(),
                     {TryCancelResponse0, TryCancelResponse1});
}

static SDValue lowerPrmtIntrinsic(SDValue Op, SelectionDAG &DAG) {
  const unsigned Mode = [&]() {
    switch (Op->getConstantOperandVal(0)) {
    case Intrinsic::nvvm_prmt:
      return NVPTX::PTXPrmtMode::NONE;
    case Intrinsic::nvvm_prmt_b4e:
      return NVPTX::PTXPrmtMode::B4E;
    case Intrinsic::nvvm_prmt_ecl:
      return NVPTX::PTXPrmtMode::ECL;
    case Intrinsic::nvvm_prmt_ecr:
      return NVPTX::PTXPrmtMode::ECR;
    case Intrinsic::nvvm_prmt_f4e:
      return NVPTX::PTXPrmtMode::F4E;
    case Intrinsic::nvvm_prmt_rc16:
      return NVPTX::PTXPrmtMode::RC16;
    case Intrinsic::nvvm_prmt_rc8:
      return NVPTX::PTXPrmtMode::RC8;
    default:
      llvm_unreachable("unsupported/unhandled intrinsic");
    }
  }();
  SDLoc DL(Op);
  SDValue A = Op->getOperand(1);
  SDValue B = Op.getNumOperands() == 4 ? Op.getOperand(2)
                                       : DAG.getConstant(0, DL, MVT::i32);
  SDValue Selector = (Op->op_end() - 1)->get();
  return getPRMT(A, B, Selector, DL, DAG, Mode);
}
static SDValue lowerIntrinsicWOChain(SDValue Op, SelectionDAG &DAG) {
  switch (Op->getConstantOperandVal(0)) {
  default:
    return Op;
  case Intrinsic::nvvm_prmt:
  case Intrinsic::nvvm_prmt_b4e:
  case Intrinsic::nvvm_prmt_ecl:
  case Intrinsic::nvvm_prmt_ecr:
  case Intrinsic::nvvm_prmt_f4e:
  case Intrinsic::nvvm_prmt_rc16:
  case Intrinsic::nvvm_prmt_rc8:
    return lowerPrmtIntrinsic(Op, DAG);
  case Intrinsic::nvvm_internal_addrspace_wrap:
    return Op.getOperand(1);
  case Intrinsic::nvvm_clusterlaunchcontrol_query_cancel_is_canceled:
  case Intrinsic::nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_x:
  case Intrinsic::nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_y:
  case Intrinsic::nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_z:
    return LowerClusterLaunchControlQueryCancel(Op, DAG);
  }
}

// In PTX 64-bit CTLZ and CTPOP are supported, but they return a 32-bit value.
// Lower these into a node returning the correct type which is zero-extended
// back to the correct size.
static SDValue lowerCTLZCTPOP(SDValue Op, SelectionDAG &DAG) {
  SDValue V = Op->getOperand(0);
  assert(V.getValueType() == MVT::i64 &&
         "Unexpected CTLZ/CTPOP type to legalize");

  SDLoc DL(Op);
  SDValue CT = DAG.getNode(Op->getOpcode(), DL, MVT::i32, V);
  return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, CT, SDNodeFlags::NonNeg);
}

static SDValue expandFSH64(SDValue A, SDValue B, SDValue ShiftAmount, SDLoc DL,
                           unsigned Opcode, SelectionDAG &DAG) {
  assert(A.getValueType() == MVT::i64 && B.getValueType() == MVT::i64);

  const auto *AmtConst = dyn_cast<ConstantSDNode>(ShiftAmount);
  if (!AmtConst)
    return SDValue();
  const auto Amt = AmtConst->getZExtValue() & 63;

  SDValue UnpackA =
      DAG.getNode(NVPTXISD::UNPACK_VECTOR, DL, {MVT::i32, MVT::i32}, A);
  SDValue UnpackB =
      DAG.getNode(NVPTXISD::UNPACK_VECTOR, DL, {MVT::i32, MVT::i32}, B);

  // Arch is Little endiain: 0 = low bits, 1 = high bits
  SDValue ALo = UnpackA.getValue(0);
  SDValue AHi = UnpackA.getValue(1);
  SDValue BLo = UnpackB.getValue(0);
  SDValue BHi = UnpackB.getValue(1);

  // The bitfeild consists of { AHi : ALo : BHi : BLo }
  //
  // * FSHL, Amt <  32 - The window will contain { AHi : ALo : BHi }
  // * FSHL, Amt >= 32 - The window will contain { ALo : BHi : BLo }
  // * FSHR, Amt <  32 - The window will contain { ALo : BHi : BLo }
  // * FSHR, Amt >= 32 - The window will contain { AHi : ALo : BHi }
  //
  // Note that Amt = 0 and Amt = 32 are special cases where 32-bit funnel shifts
  // are not needed at all. Amt = 0 is a no-op producing either A or B depending
  // on the direction. Amt = 32 can be implemented by a packing and unpacking
  // move to select and arrange the 32bit values. For simplicity, these cases
  // are not handled here explicitly and instead we rely on DAGCombiner to
  // remove the no-op funnel shifts we insert.
  auto [High, Mid, Low] = ((Opcode == ISD::FSHL) == (Amt < 32))
                              ? std::make_tuple(AHi, ALo, BHi)
                              : std::make_tuple(ALo, BHi, BLo);

  SDValue NewAmt = DAG.getConstant(Amt & 31, DL, MVT::i32);
  SDValue RHi = DAG.getNode(Opcode, DL, MVT::i32, {High, Mid, NewAmt});
  SDValue RLo = DAG.getNode(Opcode, DL, MVT::i32, {Mid, Low, NewAmt});

  return DAG.getNode(NVPTXISD::BUILD_VECTOR, DL, MVT::i64, {RLo, RHi});
}

static SDValue lowerFSH(SDValue Op, SelectionDAG &DAG) {
  return expandFSH64(Op->getOperand(0), Op->getOperand(1), Op->getOperand(2),
                     SDLoc(Op), Op->getOpcode(), DAG);
}

static SDValue lowerROT(SDValue Op, SelectionDAG &DAG) {
  unsigned Opcode = Op->getOpcode() == ISD::ROTL ? ISD::FSHL : ISD::FSHR;
  return expandFSH64(Op->getOperand(0), Op->getOperand(0), Op->getOperand(1),
                     SDLoc(Op), Opcode, DAG);
}

static SDValue lowerFREM(SDValue Op, SelectionDAG &DAG,
                         bool AllowUnsafeFPMath) {
  // Lower (frem x, y) into (sub x, (mul (ftrunc (div x, y)) y)),
  // i.e. "poor man's fmod()". When y is infinite, x is returned. This matches
  // the semantics of LLVM's frem.
  SDLoc DL(Op);
  SDValue X = Op->getOperand(0);
  SDValue Y = Op->getOperand(1);
  EVT Ty = Op.getValueType();
  SDNodeFlags Flags = Op->getFlags();

  SDValue Div = DAG.getNode(ISD::FDIV, DL, Ty, X, Y, Flags);
  SDValue Trunc = DAG.getNode(ISD::FTRUNC, DL, Ty, Div, Flags);
  SDValue Mul = DAG.getNode(ISD::FMUL, DL, Ty, Trunc, Y,
                            Flags | SDNodeFlags::AllowContract);
  SDValue Sub = DAG.getNode(ISD::FSUB, DL, Ty, X, Mul,
                            Flags | SDNodeFlags::AllowContract);

  if (AllowUnsafeFPMath || Flags.hasNoInfs())
    return Sub;

  // If Y is infinite, return X
  SDValue AbsY = DAG.getNode(ISD::FABS, DL, Ty, Y);
  SDValue Inf =
      DAG.getConstantFP(APFloat::getInf(Ty.getFltSemantics()), DL, Ty);
  SDValue IsInf = DAG.getSetCC(DL, MVT::i1, AbsY, Inf, ISD::SETEQ);
  return DAG.getSelect(DL, Ty, IsInf, X, Sub);
}

static SDValue lowerSELECT(SDValue Op, SelectionDAG &DAG) {
  assert(Op.getValueType() == MVT::i1 && "Custom lowering enabled only for i1");

  SDValue Cond = Op->getOperand(0);
  SDValue TrueVal = Op->getOperand(1);
  SDValue FalseVal = Op->getOperand(2);
  SDLoc DL(Op);

  // If both operands are truncated, we push the select through the truncates.
  if (TrueVal.getOpcode() == ISD::TRUNCATE &&
      FalseVal.getOpcode() == ISD::TRUNCATE) {
    TrueVal = TrueVal.getOperand(0);
    FalseVal = FalseVal.getOperand(0);

    EVT VT = TrueVal.getSimpleValueType().bitsLE(FalseVal.getSimpleValueType())
                 ? TrueVal.getValueType()
                 : FalseVal.getValueType();
    TrueVal = DAG.getAnyExtOrTrunc(TrueVal, DL, VT);
    FalseVal = DAG.getAnyExtOrTrunc(FalseVal, DL, VT);
    SDValue Select = DAG.getSelect(DL, VT, Cond, TrueVal, FalseVal);
    return DAG.getNode(ISD::TRUNCATE, DL, MVT::i1, Select);
  }

  // Otherwise, expand the select into a series of logical operations. These
  // often can be folded into other operations either by us or ptxas.
  TrueVal = DAG.getFreeze(TrueVal);
  FalseVal = DAG.getFreeze(FalseVal);
  SDValue And1 = DAG.getNode(ISD::AND, DL, MVT::i1, Cond, TrueVal);
  SDValue NotCond = DAG.getNOT(DL, Cond, MVT::i1);
  SDValue And2 = DAG.getNode(ISD::AND, DL, MVT::i1, NotCond, FalseVal);
  SDValue Or = DAG.getNode(ISD::OR, DL, MVT::i1, And1, And2);
  return Or;
}

SDValue
NVPTXTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::RETURNADDR:
    return SDValue();
  case ISD::FRAMEADDR:
    return SDValue();
  case ISD::ADDRSPACECAST:
    return LowerADDRSPACECAST(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN:
    return Op;
  case ISD::INTRINSIC_WO_CHAIN:
    return lowerIntrinsicWOChain(Op, DAG);
  case ISD::INTRINSIC_VOID:
    return LowerIntrinsicVoid(Op, DAG);
  case ISD::BUILD_VECTOR:
    return LowerBUILD_VECTOR(Op, DAG);
  case ISD::BITCAST:
    return LowerBITCAST(Op, DAG);
  case ISD::EXTRACT_SUBVECTOR:
    return Op;
  case ISD::EXTRACT_VECTOR_ELT:
    return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:
    return LowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::VECTOR_SHUFFLE:
    return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::CONCAT_VECTORS:
    return LowerCONCAT_VECTORS(Op, DAG);
  case ISD::VECREDUCE_FMAX:
  case ISD::VECREDUCE_FMIN:
  case ISD::VECREDUCE_FMAXIMUM:
  case ISD::VECREDUCE_FMINIMUM:
    return LowerVECREDUCE(Op, DAG);
  case ISD::STORE:
    return LowerSTORE(Op, DAG);
  case ISD::LOAD:
    return LowerLOAD(Op, DAG);
  case ISD::SHL_PARTS:
    return LowerShiftLeftParts(Op, DAG);
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS:
    return LowerShiftRightParts(Op, DAG);
  case ISD::SELECT:
    return lowerSELECT(Op, DAG);
  case ISD::FROUND:
    return LowerFROUND(Op, DAG);
  case ISD::FCOPYSIGN:
    return LowerFCOPYSIGN(Op, DAG);
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    return LowerINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    return LowerFP_TO_INT(Op, DAG);
  case ISD::FP_ROUND:
    return LowerFP_ROUND(Op, DAG);
  case ISD::FP_EXTEND:
    return LowerFP_EXTEND(Op, DAG);
  case ISD::BR_JT:
    return LowerBR_JT(Op, DAG);
  case ISD::VAARG:
    return LowerVAARG(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG);
  case ISD::FSHL:
  case ISD::FSHR:
    return lowerFSH(Op, DAG);
  case ISD::ROTL:
  case ISD::ROTR:
    return lowerROT(Op, DAG);
  case ISD::ABS:
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::SHL:
  case ISD::SREM:
  case ISD::UREM:
    return LowerVectorArith(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC:
    return LowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::STACKRESTORE:
    return LowerSTACKRESTORE(Op, DAG);
  case ISD::STACKSAVE:
    return LowerSTACKSAVE(Op, DAG);
  case ISD::CopyToReg:
    return LowerCopyToReg_128(Op, DAG);
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
    // Used only for bf16 on SM80, where we select fma for non-ftz operation
    return PromoteBinOpIfF32FTZ(Op, DAG);
  case ISD::CTPOP:
  case ISD::CTLZ:
    return lowerCTLZCTPOP(Op, DAG);
  case ISD::FREM:
    return lowerFREM(Op, DAG, allowUnsafeFPMath(DAG.getMachineFunction()));

  default:
    llvm_unreachable("Custom lowering not defined for operation");
  }
}

SDValue NVPTXTargetLowering::LowerBR_JT(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  const auto *JT = cast<JumpTableSDNode>(Op.getOperand(1));
  SDValue Index = Op.getOperand(2);

  unsigned JId = JT->getIndex();
  MachineJumpTableInfo *MJTI = DAG.getMachineFunction().getJumpTableInfo();
  ArrayRef<MachineBasicBlock *> MBBs = MJTI->getJumpTables()[JId].MBBs;

  SDValue IdV = DAG.getConstant(JId, DL, MVT::i32);

  // Generate BrxStart node
  SDVTList VTs = DAG.getVTList(MVT::Other, MVT::Glue);
  Chain = DAG.getNode(NVPTXISD::BrxStart, DL, VTs, Chain, IdV);

  // Generate BrxItem nodes
  assert(!MBBs.empty());
  for (MachineBasicBlock *MBB : MBBs.drop_back())
    Chain = DAG.getNode(NVPTXISD::BrxItem, DL, VTs, Chain.getValue(0),
                        DAG.getBasicBlock(MBB), Chain.getValue(1));

  // Generate BrxEnd nodes
  SDValue EndOps[] = {Chain.getValue(0), DAG.getBasicBlock(MBBs.back()), Index,
                      IdV, Chain.getValue(1)};
  SDValue BrxEnd = DAG.getNode(NVPTXISD::BrxEnd, DL, VTs, EndOps);

  return BrxEnd;
}

// This will prevent AsmPrinter from trying to print the jump tables itself.
unsigned NVPTXTargetLowering::getJumpTableEncoding() const {
  return MachineJumpTableInfo::EK_Inline;
}

SDValue NVPTXTargetLowering::LowerADDRSPACECAST(SDValue Op,
                                                SelectionDAG &DAG) const {
  AddrSpaceCastSDNode *N = cast<AddrSpaceCastSDNode>(Op.getNode());
  unsigned SrcAS = N->getSrcAddressSpace();
  unsigned DestAS = N->getDestAddressSpace();
  if (SrcAS != llvm::ADDRESS_SPACE_GENERIC &&
      DestAS != llvm::ADDRESS_SPACE_GENERIC) {
    // Shared and SharedCluster can be converted to each other through generic
    // space
    if ((SrcAS == llvm::ADDRESS_SPACE_SHARED &&
         DestAS == llvm::ADDRESS_SPACE_SHARED_CLUSTER) ||
        (SrcAS == llvm::ADDRESS_SPACE_SHARED_CLUSTER &&
         DestAS == llvm::ADDRESS_SPACE_SHARED)) {
      SDLoc DL(Op.getNode());
      const MVT GenerictVT =
          getPointerTy(DAG.getDataLayout(), ADDRESS_SPACE_GENERIC);
      SDValue GenericConversion = DAG.getAddrSpaceCast(
          DL, GenerictVT, Op.getOperand(0), SrcAS, ADDRESS_SPACE_GENERIC);
      SDValue SharedClusterConversion =
          DAG.getAddrSpaceCast(DL, Op.getValueType(), GenericConversion,
                               ADDRESS_SPACE_GENERIC, DestAS);
      return SharedClusterConversion;
    }

    return DAG.getUNDEF(Op.getValueType());
  }

  return Op;
}

// This function is almost a copy of SelectionDAG::expandVAArg().
// The only diff is that this one produces loads from local address space.
SDValue NVPTXTargetLowering::LowerVAARG(SDValue Op, SelectionDAG &DAG) const {
  const TargetLowering *TLI = STI.getTargetLowering();
  SDLoc DL(Op);

  SDNode *Node = Op.getNode();
  const Value *V = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  EVT VT = Node->getValueType(0);
  auto *Ty = VT.getTypeForEVT(*DAG.getContext());
  SDValue Tmp1 = Node->getOperand(0);
  SDValue Tmp2 = Node->getOperand(1);
  const MaybeAlign MA(Node->getConstantOperandVal(3));

  SDValue VAListLoad = DAG.getLoad(TLI->getPointerTy(DAG.getDataLayout()), DL,
                                   Tmp1, Tmp2, MachinePointerInfo(V));
  SDValue VAList = VAListLoad;

  if (MA && *MA > TLI->getMinStackArgumentAlignment()) {
    VAList = DAG.getNode(
        ISD::ADD, DL, VAList.getValueType(), VAList,
        DAG.getConstant(MA->value() - 1, DL, VAList.getValueType()));

    VAList = DAG.getNode(ISD::AND, DL, VAList.getValueType(), VAList,
                         DAG.getSignedConstant(-(int64_t)MA->value(), DL,
                                               VAList.getValueType()));
  }

  // Increment the pointer, VAList, to the next vaarg
  Tmp1 = DAG.getNode(ISD::ADD, DL, VAList.getValueType(), VAList,
                     DAG.getConstant(DAG.getDataLayout().getTypeAllocSize(Ty),
                                     DL, VAList.getValueType()));

  // Store the incremented VAList to the legalized pointer
  Tmp1 = DAG.getStore(VAListLoad.getValue(1), DL, Tmp1, Tmp2,
                      MachinePointerInfo(V));

  const Value *SrcV = Constant::getNullValue(
      PointerType::get(*DAG.getContext(), ADDRESS_SPACE_LOCAL));

  // Load the actual argument out of the pointer VAList
  return DAG.getLoad(VT, DL, Tmp1, VAList, MachinePointerInfo(SrcV));
}

SDValue NVPTXTargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  const TargetLowering *TLI = STI.getTargetLowering();
  SDLoc DL(Op);
  EVT PtrVT = TLI->getPointerTy(DAG.getDataLayout());

  // Store the address of unsized array <function>_vararg[] in the ap object.
  SDValue VAReg = getParamSymbol(DAG, /* vararg */ -1, PtrVT);

  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), DL, VAReg, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

static void replaceLoadVector(SDNode *N, SelectionDAG &DAG,
                              SmallVectorImpl<SDValue> &Results,
                              const NVPTXSubtarget &STI);

SDValue NVPTXTargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  if (Op.getValueType() == MVT::i1)
    return LowerLOADi1(Op, DAG);

  EVT VT = Op.getValueType();

  if (NVPTX::isPackedVectorTy(VT)) {
    // v2f32/v2f16/v2bf16/v2i16/v4i8 are legal, so we can't rely on legalizer to
    // handle unaligned loads and have to handle it here.
    LoadSDNode *Load = cast<LoadSDNode>(Op);
    EVT MemVT = Load->getMemoryVT();
    if (!allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                        MemVT, *Load->getMemOperand())) {
      SDValue Ops[2];
      std::tie(Ops[0], Ops[1]) = expandUnalignedLoad(Load, DAG);
      return DAG.getMergeValues(Ops, SDLoc(Op));
    }
  }

  return SDValue();
}

// v = ld i1* addr
//   =>
// v1 = ld i8* addr (-> i16)
// v = trunc i16 to i1
SDValue NVPTXTargetLowering::LowerLOADi1(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  LoadSDNode *LD = cast<LoadSDNode>(Node);
  SDLoc dl(Node);
  assert(LD->getExtensionType() == ISD::NON_EXTLOAD);
  assert(Node->getValueType(0) == MVT::i1 &&
         "Custom lowering for i1 load only");
  SDValue newLD = DAG.getExtLoad(ISD::ZEXTLOAD, dl, MVT::i16, LD->getChain(),
                                 LD->getBasePtr(), LD->getPointerInfo(),
                                 MVT::i8, LD->getAlign(),
                                 LD->getMemOperand()->getFlags());
  SDValue result = DAG.getNode(ISD::TRUNCATE, dl, MVT::i1, newLD);
  // The legalizer (the caller) is expecting two values from the legalized
  // load, so we build a MergeValues node for it. See ExpandUnalignedLoad()
  // in LegalizeDAG.cpp which also uses MergeValues.
  SDValue Ops[] = { result, LD->getChain() };
  return DAG.getMergeValues(Ops, dl);
}

SDValue NVPTXTargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  StoreSDNode *Store = cast<StoreSDNode>(Op);
  EVT VT = Store->getMemoryVT();

  if (VT == MVT::i1)
    return LowerSTOREi1(Op, DAG);

  // v2f32/v2f16/v2bf16/v2i16/v4i8 are legal, so we can't rely on legalizer to
  // handle unaligned stores and have to handle it here.
  if (NVPTX::isPackedVectorTy(VT) &&
      !allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                      VT, *Store->getMemOperand()))
    return expandUnalignedStore(Store, DAG);

  // v2f16/v2bf16/v2i16 don't need special handling.
  if (NVPTX::isPackedVectorTy(VT) && VT.is32BitVector())
    return SDValue();

  // Lower store of any other vector type, including v2f32 as we want to break
  // it apart since this is not a widely-supported type.
  return LowerSTOREVector(Op, DAG);
}

SDValue
NVPTXTargetLowering::LowerSTOREVector(SDValue Op, SelectionDAG &DAG) const {
  MemSDNode *N = cast<MemSDNode>(Op.getNode());
  SDValue Val = N->getOperand(1);
  SDLoc DL(N);
  const EVT ValVT = Val.getValueType();
  const EVT MemVT = N->getMemoryVT();

  // If we're truncating as part of the store, avoid lowering to a StoreV node.
  // TODO: consider relaxing this restriction.
  if (ValVT != MemVT)
    return SDValue();

  const auto NumEltsAndEltVT = getVectorLoweringShape(
      ValVT, STI.has256BitVectorLoadStore(N->getAddressSpace()));
  if (!NumEltsAndEltVT)
    return SDValue();
  const auto [NumElts, EltVT] = NumEltsAndEltVT.value();

  const DataLayout &TD = DAG.getDataLayout();

  Align Alignment = N->getAlign();
  Align PrefAlign = TD.getPrefTypeAlign(ValVT.getTypeForEVT(*DAG.getContext()));
  if (Alignment < PrefAlign) {
    // This store is not sufficiently aligned, so bail out and let this vector
    // store be scalarized.  Note that we may still be able to emit smaller
    // vector stores.  For example, if we are storing a <4 x float> with an
    // alignment of 8, this check will fail but the legalizer will try again
    // with 2 x <2 x float>, which will succeed with an alignment of 8.
    return SDValue();
  }

  unsigned Opcode;
  switch (NumElts) {
  default:
    return SDValue();
  case 2:
    Opcode = NVPTXISD::StoreV2;
    break;
  case 4:
    Opcode = NVPTXISD::StoreV4;
    break;
  case 8:
    Opcode = NVPTXISD::StoreV8;
    break;
  }

  SmallVector<SDValue, 8> Ops;

  // First is the chain
  Ops.push_back(N->getOperand(0));

  // Then the split values
  if (EltVT.isVector()) {
    assert(EVT(EltVT.getVectorElementType()) == ValVT.getVectorElementType());
    assert(NumElts * EltVT.getVectorNumElements() ==
           ValVT.getVectorNumElements());
    // Combine individual elements into v2[i,f,bf]16/v4i8 subvectors to be
    // stored as b32s
    const unsigned NumEltsPerSubVector = EltVT.getVectorNumElements();
    for (const unsigned I : llvm::seq(NumElts)) {
      SmallVector<SDValue, 4> SubVectorElts;
      DAG.ExtractVectorElements(Val, SubVectorElts, I * NumEltsPerSubVector,
                                NumEltsPerSubVector);
      Ops.push_back(DAG.getBuildVector(EltVT, DL, SubVectorElts));
    }
  } else {
    SDValue V = DAG.getBitcast(MVT::getVectorVT(EltVT, NumElts), Val);
    for (const unsigned I : llvm::seq(NumElts)) {
      SDValue ExtVal = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, V,
                                   DAG.getIntPtrConstant(I, DL));

      // Since StoreV2 is a target node, we cannot rely on DAG type
      // legalization. Therefore, we must ensure the type is legal.  For i1 and
      // i8, we set the stored type to i16 and propagate the "real" type as the
      // memory type.
      if (EltVT.getSizeInBits() < 16)
        ExtVal = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, ExtVal);
      Ops.push_back(ExtVal);
    }
  }

  // Then any remaining arguments
  Ops.append(N->op_begin() + 2, N->op_end());

  SDValue NewSt =
      DAG.getMemIntrinsicNode(Opcode, DL, DAG.getVTList(MVT::Other), Ops,
                              N->getMemoryVT(), N->getMemOperand());

  // return DCI.CombineTo(N, NewSt, true);
  return NewSt;
}

// st i1 v, addr
//    =>
// v1 = zxt v to i16
// st.u8 i16, addr
SDValue NVPTXTargetLowering::LowerSTOREi1(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  SDLoc dl(Node);
  StoreSDNode *ST = cast<StoreSDNode>(Node);
  SDValue Tmp1 = ST->getChain();
  SDValue Tmp2 = ST->getBasePtr();
  SDValue Tmp3 = ST->getValue();
  assert(Tmp3.getValueType() == MVT::i1 && "Custom lowering for i1 store only");
  Tmp3 = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i16, Tmp3);
  SDValue Result =
      DAG.getTruncStore(Tmp1, dl, Tmp3, Tmp2, ST->getPointerInfo(), MVT::i8,
                        ST->getAlign(), ST->getMemOperand()->getFlags());
  return Result;
}

SDValue NVPTXTargetLowering::LowerCopyToReg_128(SDValue Op,
                                                SelectionDAG &DAG) const {
  // Change the CopyToReg to take in two 64-bit operands instead of a 128-bit
  // operand so that it can pass the legalization.

  assert(Op.getOperand(1).getValueType() == MVT::i128 &&
         "Custom lowering for 128-bit CopyToReg only");

  SDNode *Node = Op.getNode();
  SDLoc DL(Node);

  SDValue Cast = DAG.getBitcast(MVT::v2i64, Op->getOperand(2));
  SDValue Lo = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i64, Cast,
                           DAG.getIntPtrConstant(0, DL));
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i64, Cast,
                           DAG.getIntPtrConstant(1, DL));

  SmallVector<SDValue, 5> NewOps(Op->getNumOperands() + 1);
  SmallVector<EVT, 3> ResultsType(Node->values());

  NewOps[0] = Op->getOperand(0); // Chain
  NewOps[1] = Op->getOperand(1); // Dst Reg
  NewOps[2] = Lo;                // Lower 64-bit
  NewOps[3] = Hi;                // Higher 64-bit
  if (Op.getNumOperands() == 4)
    NewOps[4] = Op->getOperand(3); // Glue if exists

  return DAG.getNode(ISD::CopyToReg, DL, ResultsType, NewOps);
}

unsigned NVPTXTargetLowering::getNumRegisters(
    LLVMContext &Context, EVT VT,
    std::optional<MVT> RegisterVT = std::nullopt) const {
  if (VT == MVT::i128 && RegisterVT == MVT::i128)
    return 1;
  return TargetLoweringBase::getNumRegisters(Context, VT, RegisterVT);
}

bool NVPTXTargetLowering::splitValueIntoRegisterParts(
    SelectionDAG &DAG, const SDLoc &DL, SDValue Val, SDValue *Parts,
    unsigned NumParts, MVT PartVT, std::optional<CallingConv::ID> CC) const {
  if (Val.getValueType() == MVT::i128 && NumParts == 1) {
    Parts[0] = Val;
    return true;
  }
  return false;
}

// This creates target external symbol for a function parameter.
// Name of the symbol is composed from its index and the function name.
// Negative index corresponds to special parameter (unsized array) used for
// passing variable arguments.
SDValue NVPTXTargetLowering::getParamSymbol(SelectionDAG &DAG, int I,
                                            EVT T) const {
  StringRef SavedStr = nvTM->getStrPool().save(
      getParamName(&DAG.getMachineFunction().getFunction(), I));
  return DAG.getExternalSymbol(SavedStr.data(), T);
}

SDValue NVPTXTargetLowering::getCallParamSymbol(SelectionDAG &DAG, int I,
                                                EVT T) const {
  const StringRef SavedStr = nvTM->getStrPool().save("param" + Twine(I));
  return DAG.getExternalSymbol(SavedStr.data(), T);
}

SDValue NVPTXTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  const DataLayout &DL = DAG.getDataLayout();
  auto PtrVT = getPointerTy(DAG.getDataLayout());

  const Function &F = DAG.getMachineFunction().getFunction();

  SDValue Root = DAG.getRoot();
  SmallVector<SDValue, 16> OutChains;

  // argTypes.size() (or theArgs.size()) and Ins.size() need not match.
  // Ins.size() will be larger
  //   * if there is an aggregate argument with multiple fields (each field
  //     showing up separately in Ins)
  //   * if there is a vector argument with more than typical vector-length
  //     elements (generally if more than 4) where each vector element is
  //     individually present in Ins.
  // So a different index should be used for indexing into Ins.
  // See similar issue in LowerCall.

  auto AllIns = ArrayRef(Ins);
  for (const auto &Arg : F.args()) {
    const auto ArgIns = AllIns.take_while(
        [&](auto I) { return I.OrigArgIndex == Arg.getArgNo(); });
    AllIns = AllIns.drop_front(ArgIns.size());

    Type *Ty = Arg.getType();

    if (ArgIns.empty())
      report_fatal_error("Empty parameter types are not supported");

    if (Arg.use_empty()) {
      // argument is dead
      for (const auto &In : ArgIns) {
        assert(!In.Used && "Arg.use_empty() is true but Arg is used?");
        InVals.push_back(DAG.getUNDEF(In.VT));
      }
      continue;
    }

    SDValue ArgSymbol = getParamSymbol(DAG, Arg.getArgNo(), PtrVT);

    // In the following cases, assign a node order of "i+1"
    // to newly created nodes. The SDNodes for params have to
    // appear in the same order as their order of appearance
    // in the original function. "i+1" holds that order.
    if (Arg.hasByValAttr()) {
      // Param has ByVal attribute
      // Return MoveParam(param symbol).
      // Ideally, the param symbol can be returned directly,
      // but when SDNode builder decides to use it in a CopyToReg(),
      // machine instruction fails because TargetExternalSymbol
      // (not lowered) is target dependent, and CopyToReg assumes
      // the source is lowered.
      assert(ArgIns.size() == 1 && "ByVal argument must be a pointer");
      const auto &ByvalIn = ArgIns[0];
      assert(getValueType(DL, Ty) == ByvalIn.VT &&
             "Ins type did not match function type");
      assert(ByvalIn.VT == PtrVT && "ByVal argument must be a pointer");

      SDValue P;
      if (isKernelFunction(F)) {
        P = ArgSymbol;
        P.getNode()->setIROrder(Arg.getArgNo() + 1);
      } else {
        P = DAG.getNode(NVPTXISD::MoveParam, dl, ByvalIn.VT, ArgSymbol);
        P.getNode()->setIROrder(Arg.getArgNo() + 1);
        P = DAG.getAddrSpaceCast(dl, ByvalIn.VT, P, ADDRESS_SPACE_LOCAL,
                                 ADDRESS_SPACE_GENERIC);
      }
      InVals.push_back(P);
    } else {
      SmallVector<EVT, 16> VTs;
      SmallVector<uint64_t, 16> Offsets;
      ComputePTXValueVTs(*this, DL, Ty, VTs, &Offsets, 0);
      assert(VTs.size() == ArgIns.size() && "Size mismatch");
      assert(VTs.size() == Offsets.size() && "Size mismatch");

      const Align ArgAlign = getFunctionArgumentAlignment(
          &F, Ty, Arg.getArgNo() + AttributeList::FirstArgIndex, DL);

      unsigned I = 0;
      const auto VI = VectorizePTXValueVTs(VTs, Offsets, ArgAlign);
      for (const unsigned NumElts : VI) {
        // i1 is loaded/stored as i8
        const EVT LoadVT = VTs[I] == MVT::i1 ? MVT::i8 : VTs[I];
        const EVT VecVT = getVectorizedVT(LoadVT, NumElts, *DAG.getContext());

        SDValue VecAddr = DAG.getObjectPtrOffset(
            dl, ArgSymbol, TypeSize::getFixed(Offsets[I]));

        const Align PartAlign = commonAlignment(ArgAlign, Offsets[I]);
        SDValue P =
            DAG.getLoad(VecVT, dl, Root, VecAddr,
                        MachinePointerInfo(ADDRESS_SPACE_PARAM), PartAlign,
                        MachineMemOperand::MODereferenceable |
                            MachineMemOperand::MOInvariant);
        P.getNode()->setIROrder(Arg.getArgNo() + 1);
        for (const unsigned J : llvm::seq(NumElts)) {
          SDValue Elt = getExtractVectorizedValue(P, J, LoadVT, dl, DAG);

          Elt = correctParamType(Elt, ArgIns[I + J].VT, ArgIns[I + J].Flags,
                                 DAG, dl);
          InVals.push_back(Elt);
        }
        I += NumElts;
      }
    }
  }

  if (!OutChains.empty())
    DAG.setRoot(DAG.getTokenFactor(dl, OutChains));

  return Chain;
}

SDValue
NVPTXTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                 bool isVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 const SDLoc &dl, SelectionDAG &DAG) const {
  const Function &F = DAG.getMachineFunction().getFunction();
  Type *RetTy = F.getReturnType();

  if (RetTy->isVoidTy()) {
    assert(OutVals.empty() && Outs.empty() && "Return value expected for void");
    return DAG.getNode(NVPTXISD::RET_GLUE, dl, MVT::Other, Chain);
  }

  const DataLayout &DL = DAG.getDataLayout();

  const SDValue RetSymbol = DAG.getExternalSymbol("func_retval0", MVT::i32);
  const auto RetAlign = getFunctionParamOptimizedAlign(&F, RetTy, DL);

  // PTX Interoperability Guide 3.3(A): [Integer] Values shorter than
  // 32-bits are sign extended or zero extended, depending on whether
  // they are signed or unsigned types.
  const bool ExtendIntegerRetVal =
      RetTy->isIntegerTy() && DL.getTypeAllocSizeInBits(RetTy) < 32;

  SmallVector<EVT, 16> VTs;
  SmallVector<uint64_t, 16> Offsets;
  ComputePTXValueVTs(*this, DL, RetTy, VTs, &Offsets);
  assert(VTs.size() == OutVals.size() && "Bad return value decomposition");

  const auto GetRetVal = [&](unsigned I) -> SDValue {
    SDValue RetVal = OutVals[I];
    assert(promoteScalarIntegerPTX(RetVal.getValueType()) ==
               RetVal.getValueType() &&
           "OutVal type should always be legal");

    const EVT VTI = promoteScalarIntegerPTX(VTs[I]);
    const EVT StoreVT =
        ExtendIntegerRetVal ? MVT::i32 : (VTI == MVT::i1 ? MVT::i8 : VTI);
    return correctParamType(RetVal, StoreVT, Outs[I].Flags, DAG, dl);
  };

  unsigned I = 0;
  const auto VI = VectorizePTXValueVTs(VTs, Offsets, RetAlign);
  for (const unsigned NumElts : VI) {
    const MaybeAlign CurrentAlign = ExtendIntegerRetVal
                                        ? MaybeAlign(std::nullopt)
                                        : commonAlignment(RetAlign, Offsets[I]);

    SDValue Val = getBuildVectorizedValue(
        NumElts, dl, DAG, [&](unsigned K) { return GetRetVal(I + K); });

    SDValue Ptr =
        DAG.getObjectPtrOffset(dl, RetSymbol, TypeSize::getFixed(Offsets[I]));

    Chain = DAG.getStore(Chain, dl, Val, Ptr,
                         MachinePointerInfo(ADDRESS_SPACE_PARAM), CurrentAlign);

    I += NumElts;
  }

  return DAG.getNode(NVPTXISD::RET_GLUE, dl, MVT::Other, Chain);
}

void NVPTXTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, StringRef Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  if (Constraint.size() > 1)
    return;
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

// llvm.ptx.memcpy.const and llvm.ptx.memmove.const need to be modeled as
// TgtMemIntrinsic
// because we need the information that is only available in the "Value" type
// of destination
// pointer. In particular, the address space information.
bool NVPTXTargetLowering::getTgtMemIntrinsic(
    IntrinsicInfo &Info, const CallInst &I,
    MachineFunction &MF, unsigned Intrinsic) const {
  switch (Intrinsic) {
  default:
    return false;
  case Intrinsic::nvvm_match_all_sync_i32p:
  case Intrinsic::nvvm_match_all_sync_i64p:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    // memVT is bogus. These intrinsics have IntrInaccessibleMemOnly attribute
    // in order to model data exchange with other threads, but perform no real
    // memory accesses.
    Info.memVT = MVT::i1;

    // Our result depends on both our and other thread's arguments.
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore;
    return true;
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_row_stride: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v8f16;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;
  }
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_s8_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_s8_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_u8_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_u8_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_s8_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_s8_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_u8_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_u8_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_bf16_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_bf16_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_bf16_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_bf16_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_s8_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_s8_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_u8_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_u8_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_s8_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_s8_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_u8_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_u8_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_bf16_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_bf16_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_bf16_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_bf16_row_stride: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v2i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(8);
    return true;
  }

  case Intrinsic::nvvm_wmma_m32n8k16_load_a_s8_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_s8_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_u8_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_u8_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_s8_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_s8_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_u8_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_u8_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_bf16_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_bf16_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_bf16_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_bf16_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k8_load_a_tf32_col:
  case Intrinsic::nvvm_wmma_m16n16k8_load_a_tf32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k8_load_a_tf32_row:
  case Intrinsic::nvvm_wmma_m16n16k8_load_a_tf32_row_stride:

  case Intrinsic::nvvm_wmma_m8n32k16_load_b_s8_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_s8_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_u8_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_u8_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_s8_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_s8_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_u8_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_u8_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_bf16_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_bf16_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_bf16_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_bf16_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k8_load_b_tf32_col:
  case Intrinsic::nvvm_wmma_m16n16k8_load_b_tf32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k8_load_b_tf32_row:
  case Intrinsic::nvvm_wmma_m16n16k8_load_b_tf32_row_stride:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_trans_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m16n16_x2_trans_b8:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m16n16_x2_trans_b8x16_b4x16_p64:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m16n16_x2_trans_b8x16_b6x16_p32:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n16_x4_b8x16_b4x16_p64:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n16_x4_b8x16_b6x16_p32: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v4i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;
  }

  case Intrinsic::nvvm_wmma_m32n8k16_load_b_s8_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_s8_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_u8_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_u8_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_s8_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_s8_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_u8_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_u8_row:

  case Intrinsic::nvvm_wmma_m8n32k16_load_a_s8_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_s8_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_u8_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_u8_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_s8_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_s8_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_u8_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_u8_row:
  case Intrinsic::nvvm_wmma_m8n8k128_load_a_b1_row:
  case Intrinsic::nvvm_wmma_m8n8k128_load_a_b1_row_stride:
  case Intrinsic::nvvm_wmma_m8n8k128_load_b_b1_col:
  case Intrinsic::nvvm_wmma_m8n8k128_load_b_b1_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_a_s4_row:
  case Intrinsic::nvvm_wmma_m8n8k32_load_a_s4_row_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_a_u4_row_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_a_u4_row:
  case Intrinsic::nvvm_wmma_m8n8k32_load_b_s4_col:
  case Intrinsic::nvvm_wmma_m8n8k32_load_b_s4_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_b_u4_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_b_u4_col:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x1_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x1_trans_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n16_x1_b8x16_b4x16_p64:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n16_x1_b8x16_b6x16_p32: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(4);
    return true;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_row_stride: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v4f16;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k8_load_c_f32_col:
  case Intrinsic::nvvm_wmma_m16n16k8_load_c_f32_row:
  case Intrinsic::nvvm_wmma_m16n16k8_load_c_f32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k8_load_c_f32_row_stride: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v8f32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;
  }

  case Intrinsic::nvvm_wmma_m32n8k16_load_a_bf16_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_bf16_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_bf16_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_bf16_row_stride:

  case Intrinsic::nvvm_wmma_m8n32k16_load_b_bf16_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_bf16_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_bf16_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_bf16_row_stride:

  case Intrinsic::nvvm_wmma_m16n16k16_load_c_s32_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_s32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_s32_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_s32_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_s32_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_s32_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_s32_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_s32_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_s32_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_s32_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_s32_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_s32_row_stride: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v8i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;
  }

  case Intrinsic::nvvm_wmma_m8n8k128_load_c_s32_col:
  case Intrinsic::nvvm_wmma_m8n8k128_load_c_s32_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k128_load_c_s32_row:
  case Intrinsic::nvvm_wmma_m8n8k128_load_c_s32_row_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_c_s32_col:
  case Intrinsic::nvvm_wmma_m8n8k32_load_c_s32_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_c_s32_row:
  case Intrinsic::nvvm_wmma_m8n8k32_load_c_s32_row_stride:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x2_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m16n16_x1_trans_b8:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m16n16_x1_trans_b8x16_b4x16_p64:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m16n16_x1_trans_b8x16_b6x16_p32:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n16_x2_b8x16_b4x16_p64:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n16_x2_b8x16_b6x16_p32: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v2i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(8);
    return true;
  }

  case Intrinsic::nvvm_wmma_m8n8k4_load_a_f64_col:
  case Intrinsic::nvvm_wmma_m8n8k4_load_a_f64_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k4_load_a_f64_row:
  case Intrinsic::nvvm_wmma_m8n8k4_load_a_f64_row_stride:

  case Intrinsic::nvvm_wmma_m8n8k4_load_b_f64_col:
  case Intrinsic::nvvm_wmma_m8n8k4_load_b_f64_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k4_load_b_f64_row:
  case Intrinsic::nvvm_wmma_m8n8k4_load_b_f64_row_stride: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::f64;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(8);
    return true;
  }

  case Intrinsic::nvvm_wmma_m8n8k4_load_c_f64_col:
  case Intrinsic::nvvm_wmma_m8n8k4_load_c_f64_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k4_load_c_f64_row:
  case Intrinsic::nvvm_wmma_m8n8k4_load_c_f64_row_stride: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v2f64;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_col:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_row:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_col:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_row:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_col:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_row:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_row_stride: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v4f16;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align = Align(16);
    return true;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_col:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_row:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_col:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_row:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_col:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_row:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k8_store_d_f32_col:
  case Intrinsic::nvvm_wmma_m16n16k8_store_d_f32_row:
  case Intrinsic::nvvm_wmma_m16n16k8_store_d_f32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k8_store_d_f32_row_stride: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v8f32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align = Align(16);
    return true;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_store_d_s32_col:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_s32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_s32_row:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_s32_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_s32_col:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_s32_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_s32_row:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_s32_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_s32_col:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_s32_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_s32_row:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_s32_row_stride: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v8i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align = Align(16);
    return true;
  }

  case Intrinsic::nvvm_wmma_m8n8k128_store_d_s32_col:
  case Intrinsic::nvvm_wmma_m8n8k128_store_d_s32_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k128_store_d_s32_row:
  case Intrinsic::nvvm_wmma_m8n8k128_store_d_s32_row_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_store_d_s32_col:
  case Intrinsic::nvvm_wmma_m8n8k32_store_d_s32_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_store_d_s32_row:
  case Intrinsic::nvvm_wmma_m8n8k32_store_d_s32_row_stride:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x2_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x2_trans_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m16n8_x2_trans_b8: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v2i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align = Align(8);
    return true;
  }

  case Intrinsic::nvvm_wmma_m8n8k4_store_d_f64_col:
  case Intrinsic::nvvm_wmma_m8n8k4_store_d_f64_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k4_store_d_f64_row:
  case Intrinsic::nvvm_wmma_m8n8k4_store_d_f64_row_stride: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v2f64;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align = Align(16);
    return true;
  }

  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x1_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x1_trans_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m16n8_x1_trans_b8: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align = Align(4);
    return true;
  }

  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x4_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x4_trans_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m16n8_x4_trans_b8: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v4i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align = Align(16);
    return true;
  }

  case Intrinsic::nvvm_atomic_add_gen_f_cta:
  case Intrinsic::nvvm_atomic_add_gen_f_sys:
  case Intrinsic::nvvm_atomic_add_gen_i_cta:
  case Intrinsic::nvvm_atomic_add_gen_i_sys:
  case Intrinsic::nvvm_atomic_and_gen_i_cta:
  case Intrinsic::nvvm_atomic_and_gen_i_sys:
  case Intrinsic::nvvm_atomic_cas_gen_i_cta:
  case Intrinsic::nvvm_atomic_cas_gen_i_sys:
  case Intrinsic::nvvm_atomic_dec_gen_i_cta:
  case Intrinsic::nvvm_atomic_dec_gen_i_sys:
  case Intrinsic::nvvm_atomic_inc_gen_i_cta:
  case Intrinsic::nvvm_atomic_inc_gen_i_sys:
  case Intrinsic::nvvm_atomic_max_gen_i_cta:
  case Intrinsic::nvvm_atomic_max_gen_i_sys:
  case Intrinsic::nvvm_atomic_min_gen_i_cta:
  case Intrinsic::nvvm_atomic_min_gen_i_sys:
  case Intrinsic::nvvm_atomic_or_gen_i_cta:
  case Intrinsic::nvvm_atomic_or_gen_i_sys:
  case Intrinsic::nvvm_atomic_exch_gen_i_cta:
  case Intrinsic::nvvm_atomic_exch_gen_i_sys:
  case Intrinsic::nvvm_atomic_xor_gen_i_cta:
  case Intrinsic::nvvm_atomic_xor_gen_i_sys: {
    auto &DL = I.getDataLayout();
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = getValueType(DL, I.getType());
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_prefetch_tensormap: {
    auto &DL = I.getDataLayout();
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = getPointerTy(DL);
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags =
        MachineMemOperand::MOLoad | MachineMemOperand::MODereferenceable;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_ldu_global_i:
  case Intrinsic::nvvm_ldu_global_f:
  case Intrinsic::nvvm_ldu_global_p: {
    auto &DL = I.getDataLayout();
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    if (Intrinsic == Intrinsic::nvvm_ldu_global_i)
      Info.memVT = getValueType(DL, I.getType());
    else if(Intrinsic == Intrinsic::nvvm_ldu_global_p)
      Info.memVT = getPointerTy(DL);
    else
      Info.memVT = getValueType(DL, I.getType());
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = cast<ConstantInt>(I.getArgOperand(1))->getMaybeAlignValue();

    return true;
  }
  case Intrinsic::nvvm_tex_1d_v4f32_s32:
  case Intrinsic::nvvm_tex_1d_v4f32_f32:
  case Intrinsic::nvvm_tex_1d_level_v4f32_f32:
  case Intrinsic::nvvm_tex_1d_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_1d_array_v4f32_s32:
  case Intrinsic::nvvm_tex_1d_array_v4f32_f32:
  case Intrinsic::nvvm_tex_1d_array_level_v4f32_f32:
  case Intrinsic::nvvm_tex_1d_array_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_2d_v4f32_s32:
  case Intrinsic::nvvm_tex_2d_v4f32_f32:
  case Intrinsic::nvvm_tex_2d_level_v4f32_f32:
  case Intrinsic::nvvm_tex_2d_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_2d_array_v4f32_s32:
  case Intrinsic::nvvm_tex_2d_array_v4f32_f32:
  case Intrinsic::nvvm_tex_2d_array_level_v4f32_f32:
  case Intrinsic::nvvm_tex_2d_array_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_3d_v4f32_s32:
  case Intrinsic::nvvm_tex_3d_v4f32_f32:
  case Intrinsic::nvvm_tex_3d_level_v4f32_f32:
  case Intrinsic::nvvm_tex_3d_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_cube_v4f32_f32:
  case Intrinsic::nvvm_tex_cube_level_v4f32_f32:
  case Intrinsic::nvvm_tex_cube_array_v4f32_f32:
  case Intrinsic::nvvm_tex_cube_array_level_v4f32_f32:
  case Intrinsic::nvvm_tld4_r_2d_v4f32_f32:
  case Intrinsic::nvvm_tld4_g_2d_v4f32_f32:
  case Intrinsic::nvvm_tld4_b_2d_v4f32_f32:
  case Intrinsic::nvvm_tld4_a_2d_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_1d_v4f32_s32:
  case Intrinsic::nvvm_tex_unified_1d_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_1d_level_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_1d_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_1d_array_v4f32_s32:
  case Intrinsic::nvvm_tex_unified_1d_array_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_1d_array_level_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_1d_array_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_2d_v4f32_s32:
  case Intrinsic::nvvm_tex_unified_2d_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_2d_level_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_2d_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_2d_array_v4f32_s32:
  case Intrinsic::nvvm_tex_unified_2d_array_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_2d_array_level_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_2d_array_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_3d_v4f32_s32:
  case Intrinsic::nvvm_tex_unified_3d_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_3d_level_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_3d_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_cube_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_cube_level_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_cube_array_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_cube_array_level_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_cube_grad_v4f32_f32:
  case Intrinsic::nvvm_tex_unified_cube_array_grad_v4f32_f32:
  case Intrinsic::nvvm_tld4_unified_r_2d_v4f32_f32:
  case Intrinsic::nvvm_tld4_unified_g_2d_v4f32_f32:
  case Intrinsic::nvvm_tld4_unified_b_2d_v4f32_f32:
  case Intrinsic::nvvm_tld4_unified_a_2d_v4f32_f32:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v4f32;
    Info.ptrVal = nullptr;
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;

  case Intrinsic::nvvm_tex_1d_v4s32_s32:
  case Intrinsic::nvvm_tex_1d_v4s32_f32:
  case Intrinsic::nvvm_tex_1d_level_v4s32_f32:
  case Intrinsic::nvvm_tex_1d_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_1d_array_v4s32_s32:
  case Intrinsic::nvvm_tex_1d_array_v4s32_f32:
  case Intrinsic::nvvm_tex_1d_array_level_v4s32_f32:
  case Intrinsic::nvvm_tex_1d_array_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_2d_v4s32_s32:
  case Intrinsic::nvvm_tex_2d_v4s32_f32:
  case Intrinsic::nvvm_tex_2d_level_v4s32_f32:
  case Intrinsic::nvvm_tex_2d_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_2d_array_v4s32_s32:
  case Intrinsic::nvvm_tex_2d_array_v4s32_f32:
  case Intrinsic::nvvm_tex_2d_array_level_v4s32_f32:
  case Intrinsic::nvvm_tex_2d_array_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_3d_v4s32_s32:
  case Intrinsic::nvvm_tex_3d_v4s32_f32:
  case Intrinsic::nvvm_tex_3d_level_v4s32_f32:
  case Intrinsic::nvvm_tex_3d_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_cube_v4s32_f32:
  case Intrinsic::nvvm_tex_cube_level_v4s32_f32:
  case Intrinsic::nvvm_tex_cube_array_v4s32_f32:
  case Intrinsic::nvvm_tex_cube_array_level_v4s32_f32:
  case Intrinsic::nvvm_tex_cube_v4u32_f32:
  case Intrinsic::nvvm_tex_cube_level_v4u32_f32:
  case Intrinsic::nvvm_tex_cube_array_v4u32_f32:
  case Intrinsic::nvvm_tex_cube_array_level_v4u32_f32:
  case Intrinsic::nvvm_tex_1d_v4u32_s32:
  case Intrinsic::nvvm_tex_1d_v4u32_f32:
  case Intrinsic::nvvm_tex_1d_level_v4u32_f32:
  case Intrinsic::nvvm_tex_1d_grad_v4u32_f32:
  case Intrinsic::nvvm_tex_1d_array_v4u32_s32:
  case Intrinsic::nvvm_tex_1d_array_v4u32_f32:
  case Intrinsic::nvvm_tex_1d_array_level_v4u32_f32:
  case Intrinsic::nvvm_tex_1d_array_grad_v4u32_f32:
  case Intrinsic::nvvm_tex_2d_v4u32_s32:
  case Intrinsic::nvvm_tex_2d_v4u32_f32:
  case Intrinsic::nvvm_tex_2d_level_v4u32_f32:
  case Intrinsic::nvvm_tex_2d_grad_v4u32_f32:
  case Intrinsic::nvvm_tex_2d_array_v4u32_s32:
  case Intrinsic::nvvm_tex_2d_array_v4u32_f32:
  case Intrinsic::nvvm_tex_2d_array_level_v4u32_f32:
  case Intrinsic::nvvm_tex_2d_array_grad_v4u32_f32:
  case Intrinsic::nvvm_tex_3d_v4u32_s32:
  case Intrinsic::nvvm_tex_3d_v4u32_f32:
  case Intrinsic::nvvm_tex_3d_level_v4u32_f32:
  case Intrinsic::nvvm_tex_3d_grad_v4u32_f32:
  case Intrinsic::nvvm_tld4_r_2d_v4s32_f32:
  case Intrinsic::nvvm_tld4_g_2d_v4s32_f32:
  case Intrinsic::nvvm_tld4_b_2d_v4s32_f32:
  case Intrinsic::nvvm_tld4_a_2d_v4s32_f32:
  case Intrinsic::nvvm_tld4_r_2d_v4u32_f32:
  case Intrinsic::nvvm_tld4_g_2d_v4u32_f32:
  case Intrinsic::nvvm_tld4_b_2d_v4u32_f32:
  case Intrinsic::nvvm_tld4_a_2d_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_1d_v4s32_s32:
  case Intrinsic::nvvm_tex_unified_1d_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_1d_level_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_1d_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_1d_array_v4s32_s32:
  case Intrinsic::nvvm_tex_unified_1d_array_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_1d_array_level_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_1d_array_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_2d_v4s32_s32:
  case Intrinsic::nvvm_tex_unified_2d_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_2d_level_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_2d_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_2d_array_v4s32_s32:
  case Intrinsic::nvvm_tex_unified_2d_array_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_2d_array_level_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_2d_array_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_3d_v4s32_s32:
  case Intrinsic::nvvm_tex_unified_3d_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_3d_level_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_3d_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_1d_v4u32_s32:
  case Intrinsic::nvvm_tex_unified_1d_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_1d_level_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_1d_grad_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_1d_array_v4u32_s32:
  case Intrinsic::nvvm_tex_unified_1d_array_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_1d_array_level_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_1d_array_grad_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_2d_v4u32_s32:
  case Intrinsic::nvvm_tex_unified_2d_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_2d_level_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_2d_grad_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_2d_array_v4u32_s32:
  case Intrinsic::nvvm_tex_unified_2d_array_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_2d_array_level_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_2d_array_grad_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_3d_v4u32_s32:
  case Intrinsic::nvvm_tex_unified_3d_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_3d_level_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_3d_grad_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_cube_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_cube_level_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_cube_array_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_cube_array_level_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_cube_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_cube_level_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_cube_array_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_cube_array_level_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_cube_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_cube_grad_v4u32_f32:
  case Intrinsic::nvvm_tex_unified_cube_array_grad_v4s32_f32:
  case Intrinsic::nvvm_tex_unified_cube_array_grad_v4u32_f32:
  case Intrinsic::nvvm_tld4_unified_r_2d_v4s32_f32:
  case Intrinsic::nvvm_tld4_unified_g_2d_v4s32_f32:
  case Intrinsic::nvvm_tld4_unified_b_2d_v4s32_f32:
  case Intrinsic::nvvm_tld4_unified_a_2d_v4s32_f32:
  case Intrinsic::nvvm_tld4_unified_r_2d_v4u32_f32:
  case Intrinsic::nvvm_tld4_unified_g_2d_v4u32_f32:
  case Intrinsic::nvvm_tld4_unified_b_2d_v4u32_f32:
  case Intrinsic::nvvm_tld4_unified_a_2d_v4u32_f32:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v4i32;
    Info.ptrVal = nullptr;
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;

  case Intrinsic::nvvm_suld_1d_i8_clamp:
  case Intrinsic::nvvm_suld_1d_v2i8_clamp:
  case Intrinsic::nvvm_suld_1d_v4i8_clamp:
  case Intrinsic::nvvm_suld_1d_array_i8_clamp:
  case Intrinsic::nvvm_suld_1d_array_v2i8_clamp:
  case Intrinsic::nvvm_suld_1d_array_v4i8_clamp:
  case Intrinsic::nvvm_suld_2d_i8_clamp:
  case Intrinsic::nvvm_suld_2d_v2i8_clamp:
  case Intrinsic::nvvm_suld_2d_v4i8_clamp:
  case Intrinsic::nvvm_suld_2d_array_i8_clamp:
  case Intrinsic::nvvm_suld_2d_array_v2i8_clamp:
  case Intrinsic::nvvm_suld_2d_array_v4i8_clamp:
  case Intrinsic::nvvm_suld_3d_i8_clamp:
  case Intrinsic::nvvm_suld_3d_v2i8_clamp:
  case Intrinsic::nvvm_suld_3d_v4i8_clamp:
  case Intrinsic::nvvm_suld_1d_i8_trap:
  case Intrinsic::nvvm_suld_1d_v2i8_trap:
  case Intrinsic::nvvm_suld_1d_v4i8_trap:
  case Intrinsic::nvvm_suld_1d_array_i8_trap:
  case Intrinsic::nvvm_suld_1d_array_v2i8_trap:
  case Intrinsic::nvvm_suld_1d_array_v4i8_trap:
  case Intrinsic::nvvm_suld_2d_i8_trap:
  case Intrinsic::nvvm_suld_2d_v2i8_trap:
  case Intrinsic::nvvm_suld_2d_v4i8_trap:
  case Intrinsic::nvvm_suld_2d_array_i8_trap:
  case Intrinsic::nvvm_suld_2d_array_v2i8_trap:
  case Intrinsic::nvvm_suld_2d_array_v4i8_trap:
  case Intrinsic::nvvm_suld_3d_i8_trap:
  case Intrinsic::nvvm_suld_3d_v2i8_trap:
  case Intrinsic::nvvm_suld_3d_v4i8_trap:
  case Intrinsic::nvvm_suld_1d_i8_zero:
  case Intrinsic::nvvm_suld_1d_v2i8_zero:
  case Intrinsic::nvvm_suld_1d_v4i8_zero:
  case Intrinsic::nvvm_suld_1d_array_i8_zero:
  case Intrinsic::nvvm_suld_1d_array_v2i8_zero:
  case Intrinsic::nvvm_suld_1d_array_v4i8_zero:
  case Intrinsic::nvvm_suld_2d_i8_zero:
  case Intrinsic::nvvm_suld_2d_v2i8_zero:
  case Intrinsic::nvvm_suld_2d_v4i8_zero:
  case Intrinsic::nvvm_suld_2d_array_i8_zero:
  case Intrinsic::nvvm_suld_2d_array_v2i8_zero:
  case Intrinsic::nvvm_suld_2d_array_v4i8_zero:
  case Intrinsic::nvvm_suld_3d_i8_zero:
  case Intrinsic::nvvm_suld_3d_v2i8_zero:
  case Intrinsic::nvvm_suld_3d_v4i8_zero:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i8;
    Info.ptrVal = nullptr;
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;

  case Intrinsic::nvvm_suld_1d_i16_clamp:
  case Intrinsic::nvvm_suld_1d_v2i16_clamp:
  case Intrinsic::nvvm_suld_1d_v4i16_clamp:
  case Intrinsic::nvvm_suld_1d_array_i16_clamp:
  case Intrinsic::nvvm_suld_1d_array_v2i16_clamp:
  case Intrinsic::nvvm_suld_1d_array_v4i16_clamp:
  case Intrinsic::nvvm_suld_2d_i16_clamp:
  case Intrinsic::nvvm_suld_2d_v2i16_clamp:
  case Intrinsic::nvvm_suld_2d_v4i16_clamp:
  case Intrinsic::nvvm_suld_2d_array_i16_clamp:
  case Intrinsic::nvvm_suld_2d_array_v2i16_clamp:
  case Intrinsic::nvvm_suld_2d_array_v4i16_clamp:
  case Intrinsic::nvvm_suld_3d_i16_clamp:
  case Intrinsic::nvvm_suld_3d_v2i16_clamp:
  case Intrinsic::nvvm_suld_3d_v4i16_clamp:
  case Intrinsic::nvvm_suld_1d_i16_trap:
  case Intrinsic::nvvm_suld_1d_v2i16_trap:
  case Intrinsic::nvvm_suld_1d_v4i16_trap:
  case Intrinsic::nvvm_suld_1d_array_i16_trap:
  case Intrinsic::nvvm_suld_1d_array_v2i16_trap:
  case Intrinsic::nvvm_suld_1d_array_v4i16_trap:
  case Intrinsic::nvvm_suld_2d_i16_trap:
  case Intrinsic::nvvm_suld_2d_v2i16_trap:
  case Intrinsic::nvvm_suld_2d_v4i16_trap:
  case Intrinsic::nvvm_suld_2d_array_i16_trap:
  case Intrinsic::nvvm_suld_2d_array_v2i16_trap:
  case Intrinsic::nvvm_suld_2d_array_v4i16_trap:
  case Intrinsic::nvvm_suld_3d_i16_trap:
  case Intrinsic::nvvm_suld_3d_v2i16_trap:
  case Intrinsic::nvvm_suld_3d_v4i16_trap:
  case Intrinsic::nvvm_suld_1d_i16_zero:
  case Intrinsic::nvvm_suld_1d_v2i16_zero:
  case Intrinsic::nvvm_suld_1d_v4i16_zero:
  case Intrinsic::nvvm_suld_1d_array_i16_zero:
  case Intrinsic::nvvm_suld_1d_array_v2i16_zero:
  case Intrinsic::nvvm_suld_1d_array_v4i16_zero:
  case Intrinsic::nvvm_suld_2d_i16_zero:
  case Intrinsic::nvvm_suld_2d_v2i16_zero:
  case Intrinsic::nvvm_suld_2d_v4i16_zero:
  case Intrinsic::nvvm_suld_2d_array_i16_zero:
  case Intrinsic::nvvm_suld_2d_array_v2i16_zero:
  case Intrinsic::nvvm_suld_2d_array_v4i16_zero:
  case Intrinsic::nvvm_suld_3d_i16_zero:
  case Intrinsic::nvvm_suld_3d_v2i16_zero:
  case Intrinsic::nvvm_suld_3d_v4i16_zero:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i16;
    Info.ptrVal = nullptr;
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;

  case Intrinsic::nvvm_suld_1d_i32_clamp:
  case Intrinsic::nvvm_suld_1d_v2i32_clamp:
  case Intrinsic::nvvm_suld_1d_v4i32_clamp:
  case Intrinsic::nvvm_suld_1d_array_i32_clamp:
  case Intrinsic::nvvm_suld_1d_array_v2i32_clamp:
  case Intrinsic::nvvm_suld_1d_array_v4i32_clamp:
  case Intrinsic::nvvm_suld_2d_i32_clamp:
  case Intrinsic::nvvm_suld_2d_v2i32_clamp:
  case Intrinsic::nvvm_suld_2d_v4i32_clamp:
  case Intrinsic::nvvm_suld_2d_array_i32_clamp:
  case Intrinsic::nvvm_suld_2d_array_v2i32_clamp:
  case Intrinsic::nvvm_suld_2d_array_v4i32_clamp:
  case Intrinsic::nvvm_suld_3d_i32_clamp:
  case Intrinsic::nvvm_suld_3d_v2i32_clamp:
  case Intrinsic::nvvm_suld_3d_v4i32_clamp:
  case Intrinsic::nvvm_suld_1d_i32_trap:
  case Intrinsic::nvvm_suld_1d_v2i32_trap:
  case Intrinsic::nvvm_suld_1d_v4i32_trap:
  case Intrinsic::nvvm_suld_1d_array_i32_trap:
  case Intrinsic::nvvm_suld_1d_array_v2i32_trap:
  case Intrinsic::nvvm_suld_1d_array_v4i32_trap:
  case Intrinsic::nvvm_suld_2d_i32_trap:
  case Intrinsic::nvvm_suld_2d_v2i32_trap:
  case Intrinsic::nvvm_suld_2d_v4i32_trap:
  case Intrinsic::nvvm_suld_2d_array_i32_trap:
  case Intrinsic::nvvm_suld_2d_array_v2i32_trap:
  case Intrinsic::nvvm_suld_2d_array_v4i32_trap:
  case Intrinsic::nvvm_suld_3d_i32_trap:
  case Intrinsic::nvvm_suld_3d_v2i32_trap:
  case Intrinsic::nvvm_suld_3d_v4i32_trap:
  case Intrinsic::nvvm_suld_1d_i32_zero:
  case Intrinsic::nvvm_suld_1d_v2i32_zero:
  case Intrinsic::nvvm_suld_1d_v4i32_zero:
  case Intrinsic::nvvm_suld_1d_array_i32_zero:
  case Intrinsic::nvvm_suld_1d_array_v2i32_zero:
  case Intrinsic::nvvm_suld_1d_array_v4i32_zero:
  case Intrinsic::nvvm_suld_2d_i32_zero:
  case Intrinsic::nvvm_suld_2d_v2i32_zero:
  case Intrinsic::nvvm_suld_2d_v4i32_zero:
  case Intrinsic::nvvm_suld_2d_array_i32_zero:
  case Intrinsic::nvvm_suld_2d_array_v2i32_zero:
  case Intrinsic::nvvm_suld_2d_array_v4i32_zero:
  case Intrinsic::nvvm_suld_3d_i32_zero:
  case Intrinsic::nvvm_suld_3d_v2i32_zero:
  case Intrinsic::nvvm_suld_3d_v4i32_zero:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i32;
    Info.ptrVal = nullptr;
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;

  case Intrinsic::nvvm_suld_1d_i64_clamp:
  case Intrinsic::nvvm_suld_1d_v2i64_clamp:
  case Intrinsic::nvvm_suld_1d_array_i64_clamp:
  case Intrinsic::nvvm_suld_1d_array_v2i64_clamp:
  case Intrinsic::nvvm_suld_2d_i64_clamp:
  case Intrinsic::nvvm_suld_2d_v2i64_clamp:
  case Intrinsic::nvvm_suld_2d_array_i64_clamp:
  case Intrinsic::nvvm_suld_2d_array_v2i64_clamp:
  case Intrinsic::nvvm_suld_3d_i64_clamp:
  case Intrinsic::nvvm_suld_3d_v2i64_clamp:
  case Intrinsic::nvvm_suld_1d_i64_trap:
  case Intrinsic::nvvm_suld_1d_v2i64_trap:
  case Intrinsic::nvvm_suld_1d_array_i64_trap:
  case Intrinsic::nvvm_suld_1d_array_v2i64_trap:
  case Intrinsic::nvvm_suld_2d_i64_trap:
  case Intrinsic::nvvm_suld_2d_v2i64_trap:
  case Intrinsic::nvvm_suld_2d_array_i64_trap:
  case Intrinsic::nvvm_suld_2d_array_v2i64_trap:
  case Intrinsic::nvvm_suld_3d_i64_trap:
  case Intrinsic::nvvm_suld_3d_v2i64_trap:
  case Intrinsic::nvvm_suld_1d_i64_zero:
  case Intrinsic::nvvm_suld_1d_v2i64_zero:
  case Intrinsic::nvvm_suld_1d_array_i64_zero:
  case Intrinsic::nvvm_suld_1d_array_v2i64_zero:
  case Intrinsic::nvvm_suld_2d_i64_zero:
  case Intrinsic::nvvm_suld_2d_v2i64_zero:
  case Intrinsic::nvvm_suld_2d_array_i64_zero:
  case Intrinsic::nvvm_suld_2d_array_v2i64_zero:
  case Intrinsic::nvvm_suld_3d_i64_zero:
  case Intrinsic::nvvm_suld_3d_v2i64_zero:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i64;
    Info.ptrVal = nullptr;
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align = Align(16);
    return true;

  case Intrinsic::nvvm_tcgen05_ld_16x64b_x1:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x1:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x1: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v1i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_ld_16x64b_x2:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x1:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x2:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x2: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v2i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_ld_16x64b_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x2:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x1:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x4: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v4i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_ld_16x64b_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x2:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x8: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v8i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_ld_16x64b_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x4:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x16: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v16i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_ld_16x64b_x32:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x8:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x32:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x32: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v32i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_ld_16x64b_x64:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x32:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x16:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x64:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x64: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v64i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_ld_16x64b_x128:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x64:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x32:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x128:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x128: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::v128i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOLoad;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_st_16x64b_x1:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x1:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x1: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_st_16x64b_x2:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x1:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x2:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x2: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v2i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_st_16x64b_x4:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x2:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x1:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x4:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x4: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v4i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_st_16x64b_x8:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x4:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x2:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x8:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x8: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v8i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_st_16x64b_x16:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x8:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x4:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x16:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x16: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v16i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_st_16x64b_x32:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x16:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x8:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x32:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x32: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v32i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_st_16x64b_x64:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x32:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x16:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x64:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x64: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v64i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align.reset();
    return true;
  }

  case Intrinsic::nvvm_tcgen05_st_16x64b_x128:
  case Intrinsic::nvvm_tcgen05_st_16x128b_x64:
  case Intrinsic::nvvm_tcgen05_st_16x256b_x32:
  case Intrinsic::nvvm_tcgen05_st_32x32b_x128:
  case Intrinsic::nvvm_tcgen05_st_16x32bx2_x128: {
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::v128i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.flags = MachineMemOperand::MOStore;
    Info.align.reset();
    return true;
  }
  }
  return false;
}

/// getFunctionParamOptimizedAlign - since function arguments are passed via
/// .param space, we may want to increase their alignment in a way that
/// ensures that we can effectively vectorize their loads & stores. We can
/// increase alignment only if the function has internal or has private
/// linkage as for other linkage types callers may already rely on default
/// alignment. To allow using 128-bit vectorized loads/stores, this function
/// ensures that alignment is 16 or greater.
Align NVPTXTargetLowering::getFunctionParamOptimizedAlign(
    const Function *F, Type *ArgTy, const DataLayout &DL) const {
  // Capping the alignment to 128 bytes as that is the maximum alignment
  // supported by PTX.
  const Align ABITypeAlign = std::min(Align(128), DL.getABITypeAlign(ArgTy));

  // If a function has linkage different from internal or private, we
  // must use default ABI alignment as external users rely on it. Same
  // for a function that may be called from a function pointer.
  if (!F || !F->hasLocalLinkage() ||
      F->hasAddressTaken(/*Users=*/nullptr,
                         /*IgnoreCallbackUses=*/false,
                         /*IgnoreAssumeLikeCalls=*/true,
                         /*IgnoreLLVMUsed=*/true))
    return ABITypeAlign;

  assert(!isKernelFunction(*F) && "Expect kernels to have non-local linkage");
  return std::max(Align(16), ABITypeAlign);
}

/// Helper for computing alignment of a device function byval parameter.
Align NVPTXTargetLowering::getFunctionByValParamAlign(
    const Function *F, Type *ArgTy, Align InitialAlign,
    const DataLayout &DL) const {
  Align ArgAlign = InitialAlign;
  // Try to increase alignment to enhance vectorization options.
  if (F)
    ArgAlign = std::max(ArgAlign, getFunctionParamOptimizedAlign(F, ArgTy, DL));

  // Old ptx versions have a bug. When PTX code takes address of
  // byval parameter with alignment < 4, ptxas generates code to
  // spill argument into memory. Alas on sm_50+ ptxas generates
  // SASS code that fails with misaligned access. To work around
  // the problem, make sure that we align byval parameters by at
  // least 4. This bug seems to be fixed at least starting from
  // ptxas > 9.0.
  // TODO: remove this after verifying the bug is not reproduced
  // on non-deprecated ptxas versions.
  if (ForceMinByValParamAlign)
    ArgAlign = std::max(ArgAlign, Align(4));

  return ArgAlign;
}

// Helper for getting a function parameter name. Name is composed from
// its index and the function name. Negative index corresponds to special
// parameter (unsized array) used for passing variable arguments.
std::string NVPTXTargetLowering::getParamName(const Function *F,
                                              int Idx) const {
  std::string ParamName;
  raw_string_ostream ParamStr(ParamName);

  ParamStr << getTargetMachine().getSymbol(F)->getName();
  if (Idx < 0)
    ParamStr << "_vararg";
  else
    ParamStr << "_param_" << Idx;

  return ParamName;
}

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
/// Used to guide target specific optimizations, like loop strength reduction
/// (LoopStrengthReduce.cpp) and memory optimization for address mode
/// (CodeGenPrepare.cpp)
bool NVPTXTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                                const AddrMode &AM, Type *Ty,
                                                unsigned AS, Instruction *I) const {
  // AddrMode - This represents an addressing mode of:
  //    BaseGV + BaseOffs + BaseReg + Scale*ScaleReg
  //
  // The legal address modes are
  // - [avar]
  // - [areg]
  // - [areg+immoff]
  // - [immAddr]

  // immoff must fit in a signed 32-bit int
  if (!APInt(64, AM.BaseOffs).isSignedIntN(32))
    return false;

  if (AM.BaseGV)
    return !AM.BaseOffs && !AM.HasBaseReg && !AM.Scale;

  switch (AM.Scale) {
  case 0: // "r", "r+i" or "i" is allowed
    break;
  case 1:
    if (AM.HasBaseReg) // "r+r+i" or "r+r" is not allowed.
      return false;
    // Otherwise we have r+i.
    break;
  default:
    // No scale > 1 is allowed
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
//                         NVPTX Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
NVPTXTargetLowering::ConstraintType
NVPTXTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'b':
    case 'r':
    case 'h':
    case 'c':
    case 'l':
    case 'f':
    case 'd':
    case 'q':
    case '0':
    case 'N':
      return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass *>
NVPTXTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                  StringRef Constraint,
                                                  MVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'b':
      return std::make_pair(0U, &NVPTX::B1RegClass);
    case 'c':
    case 'h':
      return std::make_pair(0U, &NVPTX::B16RegClass);
    case 'r':
    case 'f':
      return std::make_pair(0U, &NVPTX::B32RegClass);
    case 'l':
    case 'N':
    case 'd':
      return std::make_pair(0U, &NVPTX::B64RegClass);
    case 'q': {
      if (STI.getSmVersion() < 70)
        report_fatal_error("Inline asm with 128 bit operands is only "
                           "supported for sm_70 and higher!");
      return std::make_pair(0U, &NVPTX::B128RegClass);
    }
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

//===----------------------------------------------------------------------===//
//                         NVPTX DAG Combining
//===----------------------------------------------------------------------===//

bool NVPTXTargetLowering::allowFMA(MachineFunction &MF,
                                   CodeGenOptLevel OptLevel) const {
  // Always honor command-line argument
  if (FMAContractLevelOpt.getNumOccurrences() > 0)
    return FMAContractLevelOpt > 0;

  // Do not contract if we're not optimizing the code.
  if (OptLevel == CodeGenOptLevel::None)
    return false;

  // Honor TargetOptions flags that explicitly say fusion is okay.
  if (MF.getTarget().Options.AllowFPOpFusion == FPOpFusion::Fast)
    return true;

  return allowUnsafeFPMath(MF);
}

bool NVPTXTargetLowering::allowUnsafeFPMath(const MachineFunction &MF) const {
  // Honor TargetOptions flags that explicitly say unsafe math is okay.
  if (MF.getTarget().Options.UnsafeFPMath)
    return true;

  // Allow unsafe math if unsafe-fp-math attribute explicitly says so.
  const Function &F = MF.getFunction();
  return F.getFnAttribute("unsafe-fp-math").getValueAsBool();
}

static bool isConstZero(const SDValue &Operand) {
  const auto *Const = dyn_cast<ConstantSDNode>(Operand);
  return Const && Const->getZExtValue() == 0;
}

/// PerformADDCombineWithOperands - Try DAG combinations for an ADD with
/// operands N0 and N1.  This is a helper for PerformADDCombine that is
/// called with the default operands, and if that fails, with commuted
/// operands.
static SDValue
PerformADDCombineWithOperands(SDNode *N, SDValue N0, SDValue N1,
                              TargetLowering::DAGCombinerInfo &DCI) {
  EVT VT = N0.getValueType();

  // Since integer multiply-add costs the same as integer multiply
  // but is more costly than integer add, do the fusion only when
  // the mul is only used in the add.
  // TODO: this may not be true for later architectures, consider relaxing this
  if (!N0.getNode()->hasOneUse())
    return SDValue();

  // fold (add (select cond, 0, (mul a, b)), c)
  //   -> (select cond, c, (add (mul a, b), c))
  //
  if (N0.getOpcode() == ISD::SELECT) {
    unsigned ZeroOpNum;
    if (isConstZero(N0->getOperand(1)))
      ZeroOpNum = 1;
    else if (isConstZero(N0->getOperand(2)))
      ZeroOpNum = 2;
    else
      return SDValue();

    SDValue M = N0->getOperand((ZeroOpNum == 1) ? 2 : 1);
    if (M->getOpcode() != ISD::MUL || !M.getNode()->hasOneUse())
      return SDValue();

    SDLoc DL(N);
    SDValue Mul =
        DCI.DAG.getNode(ISD::MUL, DL, VT, M->getOperand(0), M->getOperand(1));
    SDValue MAD = DCI.DAG.getNode(ISD::ADD, DL, VT, Mul, N1);
    return DCI.DAG.getSelect(SDLoc(N), VT, N0->getOperand(0),
                             ((ZeroOpNum == 1) ? N1 : MAD),
                             ((ZeroOpNum == 1) ? MAD : N1));
  }

  return SDValue();
}

static SDValue
PerformFADDCombineWithOperands(SDNode *N, SDValue N0, SDValue N1,
                               TargetLowering::DAGCombinerInfo &DCI,
                               CodeGenOptLevel OptLevel) {
  EVT VT = N0.getValueType();
  if (N0.getOpcode() == ISD::FMUL) {
    const auto *TLI = static_cast<const NVPTXTargetLowering *>(
        &DCI.DAG.getTargetLoweringInfo());
    if (!(TLI->allowFMA(DCI.DAG.getMachineFunction(), OptLevel) ||
          (N->getFlags().hasAllowContract() &&
           N0->getFlags().hasAllowContract())))
      return SDValue();

    // For floating point:
    // Do the fusion only when the mul has less than 5 uses and all
    // are add.
    // The heuristic is that if a use is not an add, then that use
    // cannot be fused into fma, therefore mul is still needed anyway.
    // If there are more than 4 uses, even if they are all add, fusing
    // them will increase register pressue.
    //
    int numUses = 0;
    int nonAddCount = 0;
    for (const SDNode *User : N0.getNode()->users()) {
      numUses++;
      if (User->getOpcode() != ISD::FADD)
        ++nonAddCount;
      if (numUses >= 5)
        return SDValue();
    }
    if (nonAddCount) {
      int orderNo = N->getIROrder();
      int orderNo2 = N0.getNode()->getIROrder();
      // simple heuristics here for considering potential register
      // pressure, the logics here is that the differnce are used
      // to measure the distance between def and use, the longer distance
      // more likely cause register pressure.
      if (orderNo - orderNo2 < 500)
        return SDValue();

      // Now, check if at least one of the FMUL's operands is live beyond the
      // node N, which guarantees that the FMA will not increase register
      // pressure at node N.
      bool opIsLive = false;
      const SDNode *left = N0.getOperand(0).getNode();
      const SDNode *right = N0.getOperand(1).getNode();

      if (isa<ConstantSDNode>(left) || isa<ConstantSDNode>(right))
        opIsLive = true;

      if (!opIsLive)
        for (const SDNode *User : left->users()) {
          int orderNo3 = User->getIROrder();
          if (orderNo3 > orderNo) {
            opIsLive = true;
            break;
          }
        }

      if (!opIsLive)
        for (const SDNode *User : right->users()) {
          int orderNo3 = User->getIROrder();
          if (orderNo3 > orderNo) {
            opIsLive = true;
            break;
          }
        }

      if (!opIsLive)
        return SDValue();
    }

    return DCI.DAG.getNode(ISD::FMA, SDLoc(N), VT, N0.getOperand(0),
                           N0.getOperand(1), N1);
  }

  return SDValue();
}

/// Fold unpacking movs into a load by increasing the number of return values.
///
/// ex:
/// L: v2f16,ch = load <p>
/// a: f16 = extractelt L:0, 0
/// b: f16 = extractelt L:0, 1
/// use(a, b)
///
/// ...is turned into...
///
/// L: f16,f16,ch = LoadV2 <p>
/// use(L:0, L:1)
static SDValue
combineUnpackingMovIntoLoad(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  // Don't run this optimization before the legalizer
  if (!DCI.isAfterLegalizeDAG())
    return SDValue();

  EVT ElementVT = N->getValueType(0);
  // Avoid non-packed types and v4i8
  if (!NVPTX::isPackedVectorTy(ElementVT) || ElementVT == MVT::v4i8)
    return SDValue();

  SmallVector<SDNode *> DeadCopyToRegs;

  // Check whether all outputs are either used by an extractelt or are
  // glue/chain nodes
  if (!all_of(N->uses(), [&](SDUse &U) {
        // Skip glue, chain nodes
        if (U.getValueType() == MVT::Glue || U.getValueType() == MVT::Other)
          return true;
        if (U.getUser()->getOpcode() == ISD::EXTRACT_VECTOR_ELT) {
          if (N->getOpcode() != ISD::LOAD)
            return true;
          // Since this is an ISD::LOAD, check all extractelts are used. If
          // any are not used, we don't want to defeat another optimization that
          // will narrow the load.
          //
          // For example:
          //
          // L: v2f16,ch = load <p>
          // e0: f16 = extractelt L:0, 0
          // e1: f16 = extractelt L:0, 1        <-- unused
          // store e0
          //
          // Can be optimized by DAGCombiner to:
          //
          // L: f16,ch = load <p>
          // store L:0
          return !U.getUser()->use_empty();
        }

        // Otherwise, this use prevents us from splitting a value.
        return false;
      }))
    return SDValue();

  auto *LD = cast<MemSDNode>(N);
  SDLoc DL(LD);

  // the new opcode after we double the number of operands
  NVPTXISD::NodeType Opcode;
  SmallVector<SDValue> Operands(LD->ops());
  unsigned OldNumOutputs; // non-glue, non-chain outputs
  switch (LD->getOpcode()) {
  case ISD::LOAD:
    OldNumOutputs = 1;
    // Any packed type is legal, so the legalizer will not have lowered
    // ISD::LOAD -> NVPTXISD::Load (unless it's under-aligned). We have to do it
    // here.
    Opcode = NVPTXISD::LoadV2;
    Operands.push_back(DCI.DAG.getIntPtrConstant(
        cast<LoadSDNode>(LD)->getExtensionType(), DL));
    break;
  case NVPTXISD::LoadV2:
    OldNumOutputs = 2;
    Opcode = NVPTXISD::LoadV4;
    break;
  case NVPTXISD::LoadV4:
    // V8 is only supported for f32. Don't forget, we're not changing the load
    // size here. This is already a 256-bit load.
    if (ElementVT != MVT::v2f32)
      return SDValue();
    OldNumOutputs = 4;
    Opcode = NVPTXISD::LoadV8;
    break;
  case NVPTXISD::LoadV8:
    // PTX doesn't support the next doubling of outputs
    return SDValue();
  }

  // the non-glue, non-chain outputs in the new load
  const unsigned NewNumOutputs = OldNumOutputs * 2;
  SmallVector<EVT> NewVTs(NewNumOutputs, ElementVT.getVectorElementType());
  // add remaining chain and glue values
  NewVTs.append(LD->value_begin() + OldNumOutputs, LD->value_end());

  // Create the new load
  SDValue NewLoad = DCI.DAG.getMemIntrinsicNode(
      Opcode, DL, DCI.DAG.getVTList(NewVTs), Operands, LD->getMemoryVT(),
      LD->getMemOperand());

  // Now we use a combination of BUILD_VECTORs and a MERGE_VALUES node to keep
  // the outputs the same. These nodes will be optimized away in later
  // DAGCombiner iterations.
  SmallVector<SDValue> Results;
  for (unsigned I : seq(OldNumOutputs))
    Results.push_back(DCI.DAG.getBuildVector(
        ElementVT, DL, {NewLoad.getValue(I * 2), NewLoad.getValue(I * 2 + 1)}));
  // Add remaining chain and glue nodes
  for (unsigned I : seq(NewLoad->getNumValues() - NewNumOutputs))
    Results.push_back(NewLoad.getValue(NewNumOutputs + I));

  return DCI.DAG.getMergeValues(Results, DL);
}

/// Fold packing movs into a store.
///
/// ex:
/// v1: v2f16 = BUILD_VECTOR a:f16, b:f16
/// v2: v2f16 = BUILD_VECTOR c:f16, d:f16
/// StoreV2 v1, v2
///
/// ...is turned into...
///
/// StoreV4 a, b, c, d
static SDValue combinePackingMovIntoStore(SDNode *N,
                                          TargetLowering::DAGCombinerInfo &DCI,
                                          unsigned Front, unsigned Back) {
  // We want to run this as late as possible since other optimizations may
  // eliminate the BUILD_VECTORs.
  if (!DCI.isAfterLegalizeDAG())
    return SDValue();

  // Get the type of the operands being stored.
  EVT ElementVT = N->getOperand(Front).getValueType();

  // Avoid non-packed types and v4i8
  if (!NVPTX::isPackedVectorTy(ElementVT) || ElementVT == MVT::v4i8)
    return SDValue();

  auto *ST = cast<MemSDNode>(N);

  // The new opcode after we double the number of operands.
  NVPTXISD::NodeType Opcode;
  switch (N->getOpcode()) {
  case ISD::STORE:
    // Any packed type is legal, so the legalizer will not have lowered
    // ISD::STORE -> NVPTXISD::Store (unless it's under-aligned). We have to do
    // it here.
    Opcode = NVPTXISD::StoreV2;
    break;
  case NVPTXISD::StoreV2:
    Opcode = NVPTXISD::StoreV4;
    break;
  case NVPTXISD::StoreV4:
    // V8 is only supported for f32. Don't forget, we're not changing the store
    // size here. This is already a 256-bit store.
    if (ElementVT != MVT::v2f32)
      return SDValue();
    Opcode = NVPTXISD::StoreV8;
    break;
  case NVPTXISD::StoreV8:
    // PTX doesn't support the next doubling of operands
    return SDValue();
  default:
    llvm_unreachable("Unhandled store opcode");
  }

  // Scan the operands and if they're all BUILD_VECTORs, we'll have gathered
  // their elements.
  SmallVector<SDValue, 4> Operands(N->ops().take_front(Front));
  for (SDValue BV : N->ops().drop_front(Front).drop_back(Back)) {
    if (BV.getOpcode() != ISD::BUILD_VECTOR)
      return SDValue();

    // If the operand has multiple uses, this optimization can increase register
    // pressure.
    if (!BV.hasOneUse())
      return SDValue();

    // DAGCombiner visits nodes bottom-up. Check the BUILD_VECTOR operands for
    // any signs they may be folded by some other pattern or rule.
    for (SDValue Op : BV->ops()) {
      // Peek through bitcasts
      if (Op.getOpcode() == ISD::BITCAST)
        Op = Op.getOperand(0);

      // This may be folded into a PRMT.
      if (Op.getValueType() == MVT::i16 && Op.getOpcode() == ISD::TRUNCATE &&
          Op->getOperand(0).getValueType() == MVT::i32)
        return SDValue();

      // This may be folded into cvt.bf16x2
      if (Op.getOpcode() == ISD::FP_ROUND)
        return SDValue();
    }
    Operands.append({BV.getOperand(0), BV.getOperand(1)});
  }
  Operands.append(N->op_end() - Back, N->op_end());

  // Now we replace the store
  return DCI.DAG.getMemIntrinsicNode(Opcode, SDLoc(N), N->getVTList(), Operands,
                                     ST->getMemoryVT(), ST->getMemOperand());
}

static SDValue PerformStoreCombine(SDNode *N,
                                   TargetLowering::DAGCombinerInfo &DCI) {
  return combinePackingMovIntoStore(N, DCI, 1, 2);
}

/// PerformADDCombine - Target-specific dag combine xforms for ISD::ADD.
///
static SDValue PerformADDCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 CodeGenOptLevel OptLevel) {
  if (OptLevel == CodeGenOptLevel::None)
    return SDValue();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // Skip non-integer, non-scalar case
  EVT VT = N0.getValueType();
  if (VT.isVector() || VT != MVT::i32)
    return SDValue();

  // First try with the default operand order.
  if (SDValue Result = PerformADDCombineWithOperands(N, N0, N1, DCI))
    return Result;

  // If that didn't work, try again with the operands commuted.
  return PerformADDCombineWithOperands(N, N1, N0, DCI);
}

/// PerformFADDCombine - Target-specific dag combine xforms for ISD::FADD.
///
static SDValue PerformFADDCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 CodeGenOptLevel OptLevel) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  EVT VT = N0.getValueType();
  if (VT.isVector() || !(VT == MVT::f32 || VT == MVT::f64))
    return SDValue();

  // First try with the default operand order.
  if (SDValue Result = PerformFADDCombineWithOperands(N, N0, N1, DCI, OptLevel))
    return Result;

  // If that didn't work, try again with the operands commuted.
  return PerformFADDCombineWithOperands(N, N1, N0, DCI, OptLevel);
}

static SDValue PerformANDCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI) {
  // The type legalizer turns a vector load of i8 values into a zextload to i16
  // registers, optionally ANY_EXTENDs it (if target type is integer),
  // and ANDs off the high 8 bits. Since we turn this load into a
  // target-specific DAG node, the DAG combiner fails to eliminate these AND
  // nodes. Do that here.
  SDValue Val = N->getOperand(0);
  SDValue Mask = N->getOperand(1);

  if (isa<ConstantSDNode>(Val)) {
    std::swap(Val, Mask);
  }

  SDValue AExt;

  // Generally, we will see zextload -> IMOV16rr -> ANY_EXTEND -> and
  if (Val.getOpcode() == ISD::ANY_EXTEND) {
    AExt = Val;
    Val = Val->getOperand(0);
  }

  if (Val->getOpcode() == NVPTXISD::LoadV2 ||
      Val->getOpcode() == NVPTXISD::LoadV4) {
    ConstantSDNode *MaskCnst = dyn_cast<ConstantSDNode>(Mask);
    if (!MaskCnst) {
      // Not an AND with a constant
      return SDValue();
    }

    uint64_t MaskVal = MaskCnst->getZExtValue();
    if (MaskVal != 0xff) {
      // Not an AND that chops off top 8 bits
      return SDValue();
    }

    MemSDNode *Mem = dyn_cast<MemSDNode>(Val);
    if (!Mem) {
      // Not a MemSDNode?!?
      return SDValue();
    }

    EVT MemVT = Mem->getMemoryVT();
    if (MemVT != MVT::v2i8 && MemVT != MVT::v4i8) {
      // We only handle the i8 case
      return SDValue();
    }

    unsigned ExtType = Val->getConstantOperandVal(Val->getNumOperands() - 1);
    if (ExtType == ISD::SEXTLOAD) {
      // If for some reason the load is a sextload, the and is needed to zero
      // out the high 8 bits
      return SDValue();
    }

    bool AddTo = false;
    if (AExt.getNode() != nullptr) {
      // Re-insert the ext as a zext.
      Val = DCI.DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N),
                            AExt.getValueType(), Val);
      AddTo = true;
    }

    // If we get here, the AND is unnecessary.  Just replace it with the load
    DCI.CombineTo(N, Val, AddTo);
  }

  return SDValue();
}

static SDValue PerformREMCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 CodeGenOptLevel OptLevel) {
  assert(N->getOpcode() == ISD::SREM || N->getOpcode() == ISD::UREM);

  // Don't do anything at less than -O2.
  if (OptLevel < CodeGenOptLevel::Default)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  bool IsSigned = N->getOpcode() == ISD::SREM;
  unsigned DivOpc = IsSigned ? ISD::SDIV : ISD::UDIV;

  const SDValue &Num = N->getOperand(0);
  const SDValue &Den = N->getOperand(1);

  for (const SDNode *U : Num->users()) {
    if (U->getOpcode() == DivOpc && U->getOperand(0) == Num &&
        U->getOperand(1) == Den) {
      // Num % Den -> Num - (Num / Den) * Den
      return DAG.getNode(ISD::SUB, DL, VT, Num,
                         DAG.getNode(ISD::MUL, DL, VT,
                                     DAG.getNode(DivOpc, DL, VT, Num, Den),
                                     Den));
    }
  }
  return SDValue();
}

// (sign_extend|zero_extend (mul|shl) x, y) -> (mul.wide x, y)
static SDValue combineMulWide(SDNode *N, TargetLowering::DAGCombinerInfo &DCI,
                              CodeGenOptLevel OptLevel) {
  if (OptLevel == CodeGenOptLevel::None)
    return SDValue();

  SDValue Op = N->getOperand(0);
  if (!Op.hasOneUse())
    return SDValue();
  EVT ToVT = N->getValueType(0);
  EVT FromVT = Op.getValueType();
  if (!((ToVT == MVT::i32 && FromVT == MVT::i16) ||
        (ToVT == MVT::i64 && FromVT == MVT::i32)))
    return SDValue();
  if (!(Op.getOpcode() == ISD::MUL ||
        (Op.getOpcode() == ISD::SHL && isa<ConstantSDNode>(Op.getOperand(1)))))
    return SDValue();

  SDLoc DL(N);
  unsigned ExtOpcode = N->getOpcode();
  unsigned Opcode = 0;
  if (ExtOpcode == ISD::SIGN_EXTEND && Op->getFlags().hasNoSignedWrap())
    Opcode = NVPTXISD::MUL_WIDE_SIGNED;
  else if (ExtOpcode == ISD::ZERO_EXTEND && Op->getFlags().hasNoUnsignedWrap())
    Opcode = NVPTXISD::MUL_WIDE_UNSIGNED;
  else
    return SDValue();
  SDValue RHS = Op.getOperand(1);
  if (Op.getOpcode() == ISD::SHL) {
    const auto ShiftAmt = Op.getConstantOperandVal(1);
    const auto MulVal = APInt(ToVT.getSizeInBits(), 1) << ShiftAmt;
    RHS = DCI.DAG.getConstant(MulVal, DL, ToVT);
  }
  return DCI.DAG.getNode(Opcode, DL, ToVT, Op.getOperand(0), RHS);
}

enum OperandSignedness {
  Signed = 0,
  Unsigned,
  Unknown
};

/// IsMulWideOperandDemotable - Checks if the provided DAG node is an operand
/// that can be demoted to \p OptSize bits without loss of information. The
/// signedness of the operand, if determinable, is placed in \p S.
static bool IsMulWideOperandDemotable(SDValue Op,
                                      unsigned OptSize,
                                      OperandSignedness &S) {
  S = Unknown;

  if (Op.getOpcode() == ISD::SIGN_EXTEND ||
      Op.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    EVT OrigVT = Op.getOperand(0).getValueType();
    if (OrigVT.getFixedSizeInBits() <= OptSize) {
      S = Signed;
      return true;
    }
  } else if (Op.getOpcode() == ISD::ZERO_EXTEND) {
    EVT OrigVT = Op.getOperand(0).getValueType();
    if (OrigVT.getFixedSizeInBits() <= OptSize) {
      S = Unsigned;
      return true;
    }
  }

  return false;
}

/// AreMulWideOperandsDemotable - Checks if the given LHS and RHS operands can
/// be demoted to \p OptSize bits without loss of information. If the operands
/// contain a constant, it should appear as the RHS operand. The signedness of
/// the operands is placed in \p IsSigned.
static bool AreMulWideOperandsDemotable(SDValue LHS, SDValue RHS,
                                        unsigned OptSize,
                                        bool &IsSigned) {
  OperandSignedness LHSSign;

  // The LHS operand must be a demotable op
  if (!IsMulWideOperandDemotable(LHS, OptSize, LHSSign))
    return false;

  // We should have been able to determine the signedness from the LHS
  if (LHSSign == Unknown)
    return false;

  IsSigned = (LHSSign == Signed);

  // The RHS can be a demotable op or a constant
  if (ConstantSDNode *CI = dyn_cast<ConstantSDNode>(RHS)) {
    const APInt &Val = CI->getAPIntValue();
    if (LHSSign == Unsigned) {
      return Val.isIntN(OptSize);
    } else {
      return Val.isSignedIntN(OptSize);
    }
  } else {
    OperandSignedness RHSSign;
    if (!IsMulWideOperandDemotable(RHS, OptSize, RHSSign))
      return false;

    return LHSSign == RHSSign;
  }
}

/// TryMULWIDECombine - Attempt to replace a multiply of M bits with a multiply
/// of M/2 bits that produces an M-bit result (i.e. mul.wide). This transform
/// works on both multiply DAG nodes and SHL DAG nodes with a constant shift
/// amount.
static SDValue TryMULWIDECombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI) {
  EVT MulType = N->getValueType(0);
  if (MulType != MVT::i32 && MulType != MVT::i64) {
    return SDValue();
  }

  SDLoc DL(N);
  unsigned OptSize = MulType.getSizeInBits() >> 1;
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  // Canonicalize the multiply so the constant (if any) is on the right
  if (N->getOpcode() == ISD::MUL) {
    if (isa<ConstantSDNode>(LHS)) {
      std::swap(LHS, RHS);
    }
  }

  // If we have a SHL, determine the actual multiply amount
  if (N->getOpcode() == ISD::SHL) {
    ConstantSDNode *ShlRHS = dyn_cast<ConstantSDNode>(RHS);
    if (!ShlRHS) {
      return SDValue();
    }

    APInt ShiftAmt = ShlRHS->getAPIntValue();
    unsigned BitWidth = MulType.getSizeInBits();
    if (ShiftAmt.sge(0) && ShiftAmt.slt(BitWidth)) {
      APInt MulVal = APInt(BitWidth, 1) << ShiftAmt;
      RHS = DCI.DAG.getConstant(MulVal, DL, MulType);
    } else {
      return SDValue();
    }
  }

  bool Signed;
  // Verify that our operands are demotable
  if (!AreMulWideOperandsDemotable(LHS, RHS, OptSize, Signed)) {
    return SDValue();
  }

  EVT DemotedVT;
  if (MulType == MVT::i32) {
    DemotedVT = MVT::i16;
  } else {
    DemotedVT = MVT::i32;
  }

  // Truncate the operands to the correct size. Note that these are just for
  // type consistency and will (likely) be eliminated in later phases.
  SDValue TruncLHS =
    DCI.DAG.getNode(ISD::TRUNCATE, DL, DemotedVT, LHS);
  SDValue TruncRHS =
    DCI.DAG.getNode(ISD::TRUNCATE, DL, DemotedVT, RHS);

  unsigned Opc;
  if (Signed) {
    Opc = NVPTXISD::MUL_WIDE_SIGNED;
  } else {
    Opc = NVPTXISD::MUL_WIDE_UNSIGNED;
  }

  return DCI.DAG.getNode(Opc, DL, MulType, TruncLHS, TruncRHS);
}

static bool isConstOne(const SDValue &Operand) {
  const auto *Const = dyn_cast<ConstantSDNode>(Operand);
  return Const && Const->getZExtValue() == 1;
}

static SDValue matchMADConstOnePattern(SDValue Add) {
  if (Add->getOpcode() != ISD::ADD)
    return SDValue();

  if (isConstOne(Add->getOperand(0)))
    return Add->getOperand(1);

  if (isConstOne(Add->getOperand(1)))
    return Add->getOperand(0);

  return SDValue();
}

static SDValue combineMADConstOne(SDValue X, SDValue Add, EVT VT, SDLoc DL,
                                  TargetLowering::DAGCombinerInfo &DCI) {

  if (SDValue Y = matchMADConstOnePattern(Add)) {
    SDValue Mul = DCI.DAG.getNode(ISD::MUL, DL, VT, X, Y);
    return DCI.DAG.getNode(ISD::ADD, DL, VT, Mul, X);
  }

  return SDValue();
}

static SDValue combineMulSelectConstOne(SDValue X, SDValue Select, EVT VT,
                                        SDLoc DL,
                                        TargetLowering::DAGCombinerInfo &DCI) {
  if (Select->getOpcode() != ISD::SELECT)
    return SDValue();

  SDValue Cond = Select->getOperand(0);

  unsigned ConstOpNo;
  if (isConstOne(Select->getOperand(1)))
    ConstOpNo = 1;
  else if (isConstOne(Select->getOperand(2)))
    ConstOpNo = 2;
  else
    return SDValue();

  SDValue Y = Select->getOperand((ConstOpNo == 1) ? 2 : 1);

  // Do not combine if the resulting sequence is not obviously profitable.
  if (!matchMADConstOnePattern(Y))
    return SDValue();

  SDValue NewMul = DCI.DAG.getNode(ISD::MUL, DL, VT, X, Y);

  return DCI.DAG.getNode(ISD::SELECT, DL, VT, Cond,
                         (ConstOpNo == 1) ? X : NewMul,
                         (ConstOpNo == 1) ? NewMul : X);
}

static SDValue
PerformMULCombineWithOperands(SDNode *N, SDValue N0, SDValue N1,
                              TargetLowering::DAGCombinerInfo &DCI) {

  EVT VT = N0.getValueType();
  if (VT.isVector())
    return SDValue();

  if (VT != MVT::i16 && VT != MVT::i32 && VT != MVT::i64)
    return SDValue();

  SDLoc DL(N);

  // (mul x, (add y, 1)) -> (add (mul x, y), x)
  if (SDValue Res = combineMADConstOne(N0, N1, VT, DL, DCI))
    return Res;
  if (SDValue Res = combineMADConstOne(N1, N0, VT, DL, DCI))
    return Res;

  // (mul x, (select y, 1)) -> (select (mul x, y), x)
  if (SDValue Res = combineMulSelectConstOne(N0, N1, VT, DL, DCI))
    return Res;
  if (SDValue Res = combineMulSelectConstOne(N1, N0, VT, DL, DCI))
    return Res;

  return SDValue();
}

/// PerformMULCombine - Runs PTX-specific DAG combine patterns on MUL nodes.
static SDValue PerformMULCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 CodeGenOptLevel OptLevel) {
  if (OptLevel == CodeGenOptLevel::None)
    return SDValue();

  if (SDValue Ret = TryMULWIDECombine(N, DCI))
    return Ret;

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  return PerformMULCombineWithOperands(N, N0, N1, DCI);
}

/// PerformSHLCombine - Runs PTX-specific DAG combine patterns on SHL nodes.
static SDValue PerformSHLCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 CodeGenOptLevel OptLevel) {
  if (OptLevel > CodeGenOptLevel::None) {
    // Try mul.wide combining at OptLevel > 0
    if (SDValue Ret = TryMULWIDECombine(N, DCI))
      return Ret;
  }

  return SDValue();
}

static SDValue PerformSETCCCombine(SDNode *N,
                                   TargetLowering::DAGCombinerInfo &DCI,
                                   unsigned int SmVersion) {
  EVT CCType = N->getValueType(0);
  SDValue A = N->getOperand(0);
  SDValue B = N->getOperand(1);

  EVT AType = A.getValueType();
  if (!(CCType == MVT::v2i1 && (AType == MVT::v2f16 || AType == MVT::v2bf16)))
    return SDValue();

  if (A.getValueType() == MVT::v2bf16 && SmVersion < 90)
    return SDValue();

  SDLoc DL(N);
  // setp.f16x2 returns two scalar predicates, which we need to
  // convert back to v2i1. The returned result will be scalarized by
  // the legalizer, but the comparison will remain a single vector
  // instruction.
  SDValue CCNode = DCI.DAG.getNode(
      A.getValueType() == MVT::v2f16 ? NVPTXISD::SETP_F16X2
                                     : NVPTXISD::SETP_BF16X2,
      DL, DCI.DAG.getVTList(MVT::i1, MVT::i1), {A, B, N->getOperand(2)});
  return DCI.DAG.getNode(ISD::BUILD_VECTOR, DL, CCType, CCNode.getValue(0),
                         CCNode.getValue(1));
}

static SDValue PerformEXTRACTCombine(SDNode *N,
                                     TargetLowering::DAGCombinerInfo &DCI) {
  SDValue Vector = N->getOperand(0);
  if (Vector->getOpcode() == ISD::FREEZE)
    Vector = Vector->getOperand(0);
  SDLoc DL(N);
  EVT VectorVT = Vector.getValueType();
  if (Vector->getOpcode() == ISD::LOAD && VectorVT.isSimple() &&
      IsPTXVectorType(VectorVT.getSimpleVT()))
    return SDValue(); // Native vector loads already combine nicely w/
                      // extract_vector_elt.
  // Don't mess with singletons or packed types (v2f32, v2*16, v4i8 and v8i8),
  // we already handle them OK.
  if (VectorVT.getVectorNumElements() == 1 ||
      NVPTX::isPackedVectorTy(VectorVT) || VectorVT == MVT::v8i8)
    return SDValue();

  // Don't mess with undef values as sra may be simplified to 0, not undef.
  if (Vector->isUndef() || ISD::allOperandsUndef(Vector.getNode()))
    return SDValue();

  uint64_t VectorBits = VectorVT.getSizeInBits();
  // We only handle the types we can extract in-register.
  if (!(VectorBits == 16 || VectorBits == 32 || VectorBits == 64))
    return SDValue();

  ConstantSDNode *Index = dyn_cast<ConstantSDNode>(N->getOperand(1));
  // Index == 0 is handled by generic DAG combiner.
  if (!Index || Index->getZExtValue() == 0)
    return SDValue();

  MVT IVT = MVT::getIntegerVT(VectorBits);
  EVT EltVT = VectorVT.getVectorElementType();
  EVT EltIVT = EltVT.changeTypeToInteger();
  uint64_t EltBits = EltVT.getScalarSizeInBits();

  SDValue Result = DCI.DAG.getNode(
      ISD::TRUNCATE, DL, EltIVT,
      DCI.DAG.getNode(
          ISD::SRA, DL, IVT, DCI.DAG.getNode(ISD::BITCAST, DL, IVT, Vector),
          DCI.DAG.getConstant(Index->getZExtValue() * EltBits, DL, IVT)));

  // If element has non-integer type, bitcast it back to the expected type.
  if (EltVT != EltIVT)
    Result = DCI.DAG.getNode(ISD::BITCAST, DL, EltVT, Result);
  // Past legalizer, we may need to extent i8 -> i16 to match the register type.
  if (EltVT != N->getValueType(0))
    Result = DCI.DAG.getNode(ISD::ANY_EXTEND, DL, N->getValueType(0), Result);

  return Result;
}

static SDValue PerformVSELECTCombine(SDNode *N,
                                     TargetLowering::DAGCombinerInfo &DCI) {
  SDValue VA = N->getOperand(1);
  EVT VectorVT = VA.getValueType();
  if (VectorVT != MVT::v4i8)
    return SDValue();

  // We need to split vselect into individual per-element operations Because we
  // use BFE/BFI instruction for byte extraction/insertion, we do end up with
  // 32-bit values, so we may as well do comparison as i32 to avoid conversions
  // to/from i16 normally used for i8 values.
  SmallVector<SDValue, 4> E;
  SDLoc DL(N);
  SDValue VCond = N->getOperand(0);
  SDValue VB = N->getOperand(2);
  for (int I = 0; I < 4; ++I) {
    SDValue C = DCI.DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i1, VCond,
                                DCI.DAG.getConstant(I, DL, MVT::i32));
    SDValue EA = DCI.DAG.getAnyExtOrTrunc(
        DCI.DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i8, VA,
                        DCI.DAG.getConstant(I, DL, MVT::i32)),
        DL, MVT::i32);
    SDValue EB = DCI.DAG.getAnyExtOrTrunc(
        DCI.DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i8, VB,
                        DCI.DAG.getConstant(I, DL, MVT::i32)),
        DL, MVT::i32);
    E.push_back(DCI.DAG.getAnyExtOrTrunc(
        DCI.DAG.getNode(ISD::SELECT, DL, MVT::i32, C, EA, EB), DL, MVT::i8));
  }
  return DCI.DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v4i8, E);
}

static SDValue
PerformBUILD_VECTORCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  auto VT = N->getValueType(0);
  if (!DCI.isAfterLegalizeDAG() ||
      // only process v2*16 types
      !(NVPTX::isPackedVectorTy(VT) && VT.is32BitVector() &&
        VT.getVectorNumElements() == 2))
    return SDValue();

  auto Op0 = N->getOperand(0);
  auto Op1 = N->getOperand(1);

  // Start out by assuming we want to take the lower 2 bytes of each i32
  // operand.
  uint64_t Op0Bytes = 0x10;
  uint64_t Op1Bytes = 0x54;

  std::pair<SDValue *, uint64_t *> OpData[2] = {{&Op0, &Op0Bytes},
                                                {&Op1, &Op1Bytes}};

  // Check that each operand is an i16, truncated from an i32 operand. We'll
  // select individual bytes from those original operands. Optionally, fold in a
  // shift right of that original operand.
  for (auto &[Op, OpBytes] : OpData) {
    // Eat up any bitcast
    if (Op->getOpcode() == ISD::BITCAST)
      *Op = Op->getOperand(0);

    if (!(Op->getValueType() == MVT::i16 && Op->getOpcode() == ISD::TRUNCATE &&
          Op->getOperand(0).getValueType() == MVT::i32))
      return SDValue();

    // If the truncate has multiple uses, this optimization can increase
    // register pressure
    if (!Op->hasOneUse())
      return SDValue();

    *Op = Op->getOperand(0);

    // Optionally, fold in a shift-right of the original operand and let permute
    // pick the two higher bytes of the original value directly.
    if (Op->getOpcode() == ISD::SRL && isa<ConstantSDNode>(Op->getOperand(1))) {
      if (cast<ConstantSDNode>(Op->getOperand(1))->getZExtValue() == 16) {
        // Shift the PRMT byte selector to pick upper bytes from each respective
        // value, instead of the lower ones: 0x10 -> 0x32, 0x54 -> 0x76
        assert((*OpBytes == 0x10 || *OpBytes == 0x54) &&
               "PRMT selector values out of range");
        *OpBytes += 0x22;
        *Op = Op->getOperand(0);
      }
    }
  }

  SDLoc DL(N);
  auto &DAG = DCI.DAG;

  auto PRMT =
      getPRMT(DAG.getBitcast(MVT::i32, Op0), DAG.getBitcast(MVT::i32, Op1),
              (Op1Bytes << 8) | Op0Bytes, DL, DAG);
  return DAG.getBitcast(VT, PRMT);
}

static SDValue combineADDRSPACECAST(SDNode *N,
                                    TargetLowering::DAGCombinerInfo &DCI) {
  auto *ASCN1 = cast<AddrSpaceCastSDNode>(N);

  if (auto *ASCN2 = dyn_cast<AddrSpaceCastSDNode>(ASCN1->getOperand(0))) {
    assert(ASCN2->getDestAddressSpace() == ASCN1->getSrcAddressSpace());

    // Fold asc[B -> A](asc[A -> B](x)) -> x
    if (ASCN1->getDestAddressSpace() == ASCN2->getSrcAddressSpace())
      return ASCN2->getOperand(0);
  }

  return SDValue();
}

// Given a constant selector value and a prmt mode, return the selector value
// normalized to the generic prmt mode. See the PTX ISA documentation for more
// details:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
static APInt getPRMTSelector(const APInt &Selector, unsigned Mode) {
  assert(Selector.getBitWidth() == 32 && "PRMT must have i32 operands");

  if (Mode == NVPTX::PTXPrmtMode::NONE)
    return Selector;

  const unsigned V = Selector.trunc(2).getZExtValue();

  const auto GetSelector = [](unsigned S0, unsigned S1, unsigned S2,
                              unsigned S3) {
    return APInt(32, S0 | (S1 << 4) | (S2 << 8) | (S3 << 12));
  };

  switch (Mode) {
  case NVPTX::PTXPrmtMode::F4E:
    return GetSelector(V, V + 1, V + 2, V + 3);
  case NVPTX::PTXPrmtMode::B4E:
    return GetSelector(V, (V - 1) & 7, (V - 2) & 7, (V - 3) & 7);
  case NVPTX::PTXPrmtMode::RC8:
    return GetSelector(V, V, V, V);
  case NVPTX::PTXPrmtMode::ECL:
    return GetSelector(V, std::max(V, 1U), std::max(V, 2U), 3U);
  case NVPTX::PTXPrmtMode::ECR:
    return GetSelector(0, std::min(V, 1U), std::min(V, 2U), V);
  case NVPTX::PTXPrmtMode::RC16: {
    unsigned V1 = (V & 1) << 1;
    return GetSelector(V1, V1 + 1, V1, V1 + 1);
  }
  default:
    llvm_unreachable("Invalid PRMT mode");
  }
}

static APInt computePRMT(APInt A, APInt B, APInt Selector, unsigned Mode) {
  assert(A.getBitWidth() == 32 && B.getBitWidth() == 32 &&
         Selector.getBitWidth() == 32 && "PRMT must have i32 operands");
  // {b, a} = {{b7, b6, b5, b4}, {b3, b2, b1, b0}}
  APInt BitField = B.concat(A);
  APInt SelectorVal = getPRMTSelector(Selector, Mode);
  APInt Result(32, 0);
  for (unsigned I : llvm::seq(4U)) {
    APInt Sel = SelectorVal.extractBits(4, I * 4);
    unsigned Idx = Sel.getLoBits(3).getZExtValue();
    unsigned Sign = Sel.getHiBits(1).getZExtValue();
    APInt Byte = BitField.extractBits(8, Idx * 8);
    if (Sign)
      Byte = Byte.ashr(8);
    Result.insertBits(Byte, I * 8);
  }
  return Result;
}

static SDValue combinePRMT(SDNode *N, TargetLowering::DAGCombinerInfo &DCI,
                           CodeGenOptLevel OptLevel) {
  if (OptLevel == CodeGenOptLevel::None)
    return SDValue();

  // Constant fold PRMT
  if (isa<ConstantSDNode>(N->getOperand(0)) &&
      isa<ConstantSDNode>(N->getOperand(1)) &&
      isa<ConstantSDNode>(N->getOperand(2)))
    return DCI.DAG.getConstant(computePRMT(N->getConstantOperandAPInt(0),
                                           N->getConstantOperandAPInt(1),
                                           N->getConstantOperandAPInt(2),
                                           N->getConstantOperandVal(3)),
                               SDLoc(N), N->getValueType(0));
  return SDValue();
}

// During call lowering we wrap the return values in a ProxyReg node which
// depend on the chain value produced by the completed call. This ensures that
// the full call is emitted in cases where libcalls are used to legalize
// operations. To improve the functioning of other DAG combines we pull all
// operations we can through one of these nodes, ensuring that the ProxyReg
// directly wraps a load. That is:
//
//  (ProxyReg (zext (load retval0)))  =>  (zext (ProxyReg (load retval0)))
//
static SDValue sinkProxyReg(SDValue R, SDValue Chain,
                            TargetLowering::DAGCombinerInfo &DCI) {
  switch (R.getOpcode()) {
  case ISD::TRUNCATE:
  case ISD::ANY_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::BITCAST: {
    if (SDValue V = sinkProxyReg(R.getOperand(0), Chain, DCI))
      return DCI.DAG.getNode(R.getOpcode(), SDLoc(R), R.getValueType(), V);
    return SDValue();
  }
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
  case ISD::OR: {
    if (SDValue A = sinkProxyReg(R.getOperand(0), Chain, DCI))
      if (SDValue B = sinkProxyReg(R.getOperand(1), Chain, DCI))
        return DCI.DAG.getNode(R.getOpcode(), SDLoc(R), R.getValueType(), A, B);
    return SDValue();
  }
  case ISD::Constant:
    return R;
  case ISD::LOAD:
  case NVPTXISD::LoadV2:
  case NVPTXISD::LoadV4: {
    return DCI.DAG.getNode(NVPTXISD::ProxyReg, SDLoc(R), R.getValueType(),
                           {Chain, R});
  }
  case ISD::BUILD_VECTOR: {
    if (DCI.isBeforeLegalize())
      return SDValue();

    SmallVector<SDValue, 16> Ops;
    for (auto &Op : R->ops()) {
      SDValue V = sinkProxyReg(Op, Chain, DCI);
      if (!V)
        return SDValue();
      Ops.push_back(V);
    }
    return DCI.DAG.getNode(ISD::BUILD_VECTOR, SDLoc(R), R.getValueType(), Ops);
  }
  case ISD::EXTRACT_VECTOR_ELT: {
    if (DCI.isBeforeLegalize())
      return SDValue();

    if (SDValue V = sinkProxyReg(R.getOperand(0), Chain, DCI))
      return DCI.DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SDLoc(R),
                             R.getValueType(), V, R.getOperand(1));
    return SDValue();
  }
  default:
    return SDValue();
  }
}

static SDValue combineProxyReg(SDNode *N,
                               TargetLowering::DAGCombinerInfo &DCI) {

  SDValue Chain = N->getOperand(0);
  SDValue Reg = N->getOperand(1);

  // If the ProxyReg is not wrapping a load, try to pull the operations through
  // the ProxyReg.
  if (Reg.getOpcode() != ISD::LOAD) {
    if (SDValue V = sinkProxyReg(Reg, Chain, DCI))
      return V;
  }

  return SDValue();
}

SDValue NVPTXTargetLowering::PerformDAGCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  CodeGenOptLevel OptLevel = getTargetMachine().getOptLevel();
  switch (N->getOpcode()) {
  default:
    break;
  case ISD::ADD:
    return PerformADDCombine(N, DCI, OptLevel);
  case ISD::ADDRSPACECAST:
    return combineADDRSPACECAST(N, DCI);
  case ISD::AND:
    return PerformANDCombine(N, DCI);
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
    return combineMulWide(N, DCI, OptLevel);
  case ISD::BUILD_VECTOR:
    return PerformBUILD_VECTORCombine(N, DCI);
  case ISD::EXTRACT_VECTOR_ELT:
    return PerformEXTRACTCombine(N, DCI);
  case ISD::FADD:
    return PerformFADDCombine(N, DCI, OptLevel);
  case ISD::LOAD:
  case NVPTXISD::LoadV2:
  case NVPTXISD::LoadV4:
    return combineUnpackingMovIntoLoad(N, DCI);
  case ISD::MUL:
    return PerformMULCombine(N, DCI, OptLevel);
  case NVPTXISD::PRMT:
    return combinePRMT(N, DCI, OptLevel);
  case NVPTXISD::ProxyReg:
    return combineProxyReg(N, DCI);
  case ISD::SETCC:
    return PerformSETCCCombine(N, DCI, STI.getSmVersion());
  case ISD::SHL:
    return PerformSHLCombine(N, DCI, OptLevel);
  case ISD::SREM:
  case ISD::UREM:
    return PerformREMCombine(N, DCI, OptLevel);
  case ISD::STORE:
  case NVPTXISD::StoreV2:
  case NVPTXISD::StoreV4:
    return PerformStoreCombine(N, DCI);
  case ISD::VSELECT:
    return PerformVSELECTCombine(N, DCI);
  }
  return SDValue();
}

static void ReplaceBITCAST(SDNode *Node, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &Results) {
  // Handle bitcasting to v2i8 without hitting the default promotion
  // strategy which goes through stack memory.
  SDValue Op(Node, 0);
  EVT ToVT = Op->getValueType(0);
  if (ToVT != MVT::v2i8) {
    return;
  }

  // Bitcast to i16 and unpack elements into a vector
  SDLoc DL(Node);
  SDValue AsInt = DAG.getBitcast(MVT::i16, Op->getOperand(0));
  SDValue Vec0 = DAG.getNode(ISD::TRUNCATE, DL, MVT::i8, AsInt);
  SDValue Const8 = DAG.getConstant(8, DL, MVT::i16);
  SDValue Vec1 =
      DAG.getNode(ISD::TRUNCATE, DL, MVT::i8,
                  DAG.getNode(ISD::SRL, DL, MVT::i16, {AsInt, Const8}));
  Results.push_back(
      DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i8, {Vec0, Vec1}));
}

/// ReplaceVectorLoad - Convert vector loads into multi-output scalar loads.
static void replaceLoadVector(SDNode *N, SelectionDAG &DAG,
                              SmallVectorImpl<SDValue> &Results,
                              const NVPTXSubtarget &STI) {
  LoadSDNode *LD = cast<LoadSDNode>(N);
  const EVT ResVT = LD->getValueType(0);
  const EVT MemVT = LD->getMemoryVT();

  // If we're doing sign/zero extension as part of the load, avoid lowering to
  // a LoadV node. TODO: consider relaxing this restriction.
  if (ResVT != MemVT)
    return;

  const auto NumEltsAndEltVT = getVectorLoweringShape(
      ResVT, STI.has256BitVectorLoadStore(LD->getAddressSpace()));
  if (!NumEltsAndEltVT)
    return;
  const auto [NumElts, EltVT] = NumEltsAndEltVT.value();

  Align Alignment = LD->getAlign();
  const auto &TD = DAG.getDataLayout();
  Align PrefAlign = TD.getPrefTypeAlign(MemVT.getTypeForEVT(*DAG.getContext()));
  if (Alignment < PrefAlign) {
    // This load is not sufficiently aligned, so bail out and let this vector
    // load be scalarized.  Note that we may still be able to emit smaller
    // vector loads.  For example, if we are loading a <4 x float> with an
    // alignment of 8, this check will fail but the legalizer will try again
    // with 2 x <2 x float>, which will succeed with an alignment of 8.
    return;
  }

  // Since LoadV2 is a target node, we cannot rely on DAG type legalization.
  // Therefore, we must ensure the type is legal.  For i1 and i8, we set the
  // loaded type to i16 and propagate the "real" type as the memory type.
  const MVT LoadEltVT = (EltVT.getSizeInBits() < 16) ? MVT::i16 : EltVT;

  unsigned Opcode;
  switch (NumElts) {
  default:
    return;
  case 2:
    Opcode = NVPTXISD::LoadV2;
    break;
  case 4:
    Opcode = NVPTXISD::LoadV4;
    break;
  case 8:
    Opcode = NVPTXISD::LoadV8;
    break;
  }
  auto ListVTs = SmallVector<EVT, 9>(NumElts, LoadEltVT);
  ListVTs.push_back(MVT::Other);
  SDVTList LdResVTs = DAG.getVTList(ListVTs);

  SDLoc DL(LD);

  // Copy regular operands
  SmallVector<SDValue, 8> OtherOps(LD->ops());

  // The select routine does not have access to the LoadSDNode instance, so
  // pass along the extension information
  OtherOps.push_back(DAG.getIntPtrConstant(LD->getExtensionType(), DL));

  SDValue NewLD = DAG.getMemIntrinsicNode(Opcode, DL, LdResVTs, OtherOps,
                                          LD->getMemoryVT(),
                                          LD->getMemOperand());

  SmallVector<SDValue> ScalarRes;
  if (EltVT.isVector()) {
    assert(EVT(EltVT.getVectorElementType()) == ResVT.getVectorElementType());
    assert(NumElts * EltVT.getVectorNumElements() ==
           ResVT.getVectorNumElements());
    // Generate EXTRACT_VECTOR_ELTs to split v2[i,f,bf]16/v4i8 subvectors back
    // into individual elements.
    for (const unsigned I : llvm::seq(NumElts)) {
      SDValue SubVector = NewLD.getValue(I);
      DAG.ExtractVectorElements(SubVector, ScalarRes);
    }
  } else {
    for (const unsigned I : llvm::seq(NumElts)) {
      SDValue Res = NewLD.getValue(I);
      if (LoadEltVT != EltVT)
        Res = DAG.getNode(ISD::TRUNCATE, DL, EltVT, Res);
      ScalarRes.push_back(Res);
    }
  }

  SDValue LoadChain = NewLD.getValue(NumElts);

  const MVT BuildVecVT =
      MVT::getVectorVT(EltVT.getScalarType(), ScalarRes.size());
  SDValue BuildVec = DAG.getBuildVector(BuildVecVT, DL, ScalarRes);
  SDValue LoadValue = DAG.getBitcast(ResVT, BuildVec);

  Results.append({LoadValue, LoadChain});
}

// Lower vector return type of tcgen05.ld intrinsics
static void ReplaceTcgen05Ld(SDNode *N, SelectionDAG &DAG,
                             SmallVectorImpl<SDValue> &Results,
                             bool hasOffset = false) {
  SDLoc DL(N);
  EVT ResVT = N->getValueType(0);
  if (!ResVT.isVector())
    return; // already legalized.

  const unsigned NumElts = ResVT.getVectorNumElements();

  // Create the return type of the instructions
  SmallVector<EVT, 5> ListVTs;
  for (unsigned i = 0; i < NumElts; ++i)
    ListVTs.push_back(MVT::i32);

  ListVTs.push_back(N->getValueType(1)); // Chain

  SDVTList ResVTs = DAG.getVTList(ListVTs);

  SmallVector<SDValue, 8> Ops{N->getOperand(0), N->getOperand(1),
                              N->getOperand(2)};

  if (hasOffset) {
    Ops.push_back(N->getOperand(3)); // offset
    Ops.push_back(N->getOperand(4)); // Pack flag
  } else
    Ops.push_back(N->getOperand(3)); // Pack flag

  MemIntrinsicSDNode *MemSD = cast<MemIntrinsicSDNode>(N);
  SDValue NewNode =
      DAG.getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, ResVTs, Ops,
                              MemSD->getMemoryVT(), MemSD->getMemOperand());

  // split the vector result
  SmallVector<SDValue, 4> ScalarRes;
  for (unsigned i = 0; i < NumElts; ++i) {
    SDValue Res = NewNode.getValue(i);
    ScalarRes.push_back(Res);
  }

  SDValue Chain = NewNode.getValue(NumElts);
  SDValue BuildVector = DAG.getNode(ISD::BUILD_VECTOR, DL, ResVT, ScalarRes);
  Results.push_back(BuildVector); // Build Vector
  Results.push_back(Chain);       // Chain
}

static void ReplaceINTRINSIC_W_CHAIN(SDNode *N, SelectionDAG &DAG,
                                     SmallVectorImpl<SDValue> &Results) {
  SDValue Chain = N->getOperand(0);
  SDValue Intrin = N->getOperand(1);
  SDLoc DL(N);

  // Get the intrinsic ID
  unsigned IntrinNo = Intrin.getNode()->getAsZExtVal();
  switch (IntrinNo) {
  default:
    return;
  case Intrinsic::nvvm_ldu_global_i:
  case Intrinsic::nvvm_ldu_global_f:
  case Intrinsic::nvvm_ldu_global_p: {
    EVT ResVT = N->getValueType(0);

    if (ResVT.isVector()) {
      // Vector LDG/LDU

      unsigned NumElts = ResVT.getVectorNumElements();
      EVT EltVT = ResVT.getVectorElementType();

      // Since LDU/LDG are target nodes, we cannot rely on DAG type
      // legalization.
      // Therefore, we must ensure the type is legal.  For i1 and i8, we set the
      // loaded type to i16 and propagate the "real" type as the memory type.
      bool NeedTrunc = false;
      if (EltVT.getSizeInBits() < 16) {
        EltVT = MVT::i16;
        NeedTrunc = true;
      }

      unsigned Opcode = 0;
      SDVTList LdResVTs;

      switch (NumElts) {
      default:
        return;
      case 2:
        Opcode = NVPTXISD::LDUV2;
        LdResVTs = DAG.getVTList(EltVT, EltVT, MVT::Other);
        break;
      case 4: {
        Opcode = NVPTXISD::LDUV4;
        EVT ListVTs[] = { EltVT, EltVT, EltVT, EltVT, MVT::Other };
        LdResVTs = DAG.getVTList(ListVTs);
        break;
      }
      }

      SmallVector<SDValue, 8> OtherOps;

      // Copy regular operands

      OtherOps.push_back(Chain); // Chain
                                 // Skip operand 1 (intrinsic ID)
      // Others
      OtherOps.append(N->op_begin() + 2, N->op_end());

      MemIntrinsicSDNode *MemSD = cast<MemIntrinsicSDNode>(N);

      SDValue NewLD = DAG.getMemIntrinsicNode(Opcode, DL, LdResVTs, OtherOps,
                                              MemSD->getMemoryVT(),
                                              MemSD->getMemOperand());

      SmallVector<SDValue, 4> ScalarRes;

      for (unsigned i = 0; i < NumElts; ++i) {
        SDValue Res = NewLD.getValue(i);
        if (NeedTrunc)
          Res =
              DAG.getNode(ISD::TRUNCATE, DL, ResVT.getVectorElementType(), Res);
        ScalarRes.push_back(Res);
      }

      SDValue LoadChain = NewLD.getValue(NumElts);

      SDValue BuildVec =
          DAG.getBuildVector(ResVT, DL, ScalarRes);

      Results.push_back(BuildVec);
      Results.push_back(LoadChain);
    } else {
      // i8 LDG/LDU
      assert(ResVT.isSimple() && ResVT.getSimpleVT().SimpleTy == MVT::i8 &&
             "Custom handling of non-i8 ldu/ldg?");

      // Just copy all operands as-is
      SmallVector<SDValue, 4> Ops(N->ops());

      // Force output to i16
      SDVTList LdResVTs = DAG.getVTList(MVT::i16, MVT::Other);

      MemIntrinsicSDNode *MemSD = cast<MemIntrinsicSDNode>(N);

      // We make sure the memory type is i8, which will be used during isel
      // to select the proper instruction.
      SDValue NewLD =
          DAG.getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, LdResVTs, Ops,
                                  MVT::i8, MemSD->getMemOperand());

      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i8,
                                    NewLD.getValue(0)));
      Results.push_back(NewLD.getValue(1));
    }
    return;
  }

  case Intrinsic::nvvm_tcgen05_ld_16x64b_x2:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x32:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x64:
  case Intrinsic::nvvm_tcgen05_ld_16x64b_x128:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x2:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x4:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x8:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x16:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x32:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x64:
  case Intrinsic::nvvm_tcgen05_ld_32x32b_x128:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x1:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x2:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x32:
  case Intrinsic::nvvm_tcgen05_ld_16x128b_x64:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x1:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x2:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x256b_x32:
    return ReplaceTcgen05Ld(N, DAG, Results);

  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x2:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x4:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x8:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x16:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x32:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x64:
  case Intrinsic::nvvm_tcgen05_ld_16x32bx2_x128:
    return ReplaceTcgen05Ld(N, DAG, Results, /* Offset */ true);
  }
}

static void ReplaceCopyFromReg_128(SDNode *N, SelectionDAG &DAG,
                                   SmallVectorImpl<SDValue> &Results) {
  // Change the CopyFromReg to output 2 64-bit results instead of a 128-bit
  // result so that it can pass the legalization
  SDLoc DL(N);
  SDValue Chain = N->getOperand(0);
  SDValue Reg = N->getOperand(1);
  SDValue Glue = N->getOperand(2);

  assert(Reg.getValueType() == MVT::i128 &&
         "Custom lowering for CopyFromReg with 128-bit reg only");
  SmallVector<EVT, 4> ResultsType = {MVT::i64, MVT::i64, N->getValueType(1),
                                     N->getValueType(2)};
  SmallVector<SDValue, 3> NewOps = {Chain, Reg, Glue};

  SDValue NewValue = DAG.getNode(ISD::CopyFromReg, DL, ResultsType, NewOps);
  SDValue Pair = DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i128,
                             {NewValue.getValue(0), NewValue.getValue(1)});

  Results.push_back(Pair);
  Results.push_back(NewValue.getValue(2));
  Results.push_back(NewValue.getValue(3));
}

static void replaceProxyReg(SDNode *N, SelectionDAG &DAG,
                            const TargetLowering &TLI,
                            SmallVectorImpl<SDValue> &Results) {
  SDValue Chain = N->getOperand(0);
  SDValue Reg = N->getOperand(1);

  MVT VT = TLI.getRegisterType(*DAG.getContext(), Reg.getValueType());

  SDValue NewReg = DAG.getAnyExtOrTrunc(Reg, SDLoc(N), VT);
  SDValue NewProxy =
      DAG.getNode(NVPTXISD::ProxyReg, SDLoc(N), VT, {Chain, NewReg});
  SDValue Res = DAG.getAnyExtOrTrunc(NewProxy, SDLoc(N), N->getValueType(0));

  Results.push_back(Res);
}

void NVPTXTargetLowering::ReplaceNodeResults(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  default:
    report_fatal_error("Unhandled custom legalization");
  case ISD::BITCAST:
    ReplaceBITCAST(N, DAG, Results);
    return;
  case ISD::LOAD:
    replaceLoadVector(N, DAG, Results, STI);
    return;
  case ISD::INTRINSIC_W_CHAIN:
    ReplaceINTRINSIC_W_CHAIN(N, DAG, Results);
    return;
  case ISD::CopyFromReg:
    ReplaceCopyFromReg_128(N, DAG, Results);
    return;
  case NVPTXISD::ProxyReg:
    replaceProxyReg(N, DAG, *this, Results);
    return;
  }
}

NVPTXTargetLowering::AtomicExpansionKind
NVPTXTargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const {
  Type *Ty = AI->getValOperand()->getType();

  if (AI->isFloatingPointOperation()) {
    if (AI->getOperation() == AtomicRMWInst::BinOp::FAdd) {
      if (Ty->isHalfTy() && STI.getSmVersion() >= 70 &&
          STI.getPTXVersion() >= 63)
        return AtomicExpansionKind::None;
      if (Ty->isBFloatTy() && STI.getSmVersion() >= 90 &&
          STI.getPTXVersion() >= 78)
        return AtomicExpansionKind::None;
      if (Ty->isFloatTy())
        return AtomicExpansionKind::None;
      if (Ty->isDoubleTy() && STI.hasAtomAddF64())
        return AtomicExpansionKind::None;
    }
    return AtomicExpansionKind::CmpXChg;
  }

  assert(Ty->isIntegerTy() && "Ty should be integer at this point");
  auto ITy = cast<llvm::IntegerType>(Ty);

  switch (AI->getOperation()) {
  default:
    return AtomicExpansionKind::CmpXChg;
  case AtomicRMWInst::BinOp::And:
  case AtomicRMWInst::BinOp::Or:
  case AtomicRMWInst::BinOp::Xor:
  case AtomicRMWInst::BinOp::Xchg:
    switch (ITy->getBitWidth()) {
    case 8:
    case 16:
      return AtomicExpansionKind::CmpXChg;
    case 32:
      return AtomicExpansionKind::None;
    case 64:
      if (STI.hasAtomBitwise64())
        return AtomicExpansionKind::None;
      return AtomicExpansionKind::CmpXChg;
    default:
      llvm_unreachable("unsupported width encountered");
    }
  case AtomicRMWInst::BinOp::Add:
  case AtomicRMWInst::BinOp::Sub:
  case AtomicRMWInst::BinOp::Max:
  case AtomicRMWInst::BinOp::Min:
  case AtomicRMWInst::BinOp::UMax:
  case AtomicRMWInst::BinOp::UMin:
    switch (ITy->getBitWidth()) {
    case 8:
    case 16:
      return AtomicExpansionKind::CmpXChg;
    case 32:
      return AtomicExpansionKind::None;
    case 64:
      if (STI.hasAtomMinMax64())
        return AtomicExpansionKind::None;
      return AtomicExpansionKind::CmpXChg;
    default:
      llvm_unreachable("unsupported width encountered");
    }
  case AtomicRMWInst::BinOp::UIncWrap:
  case AtomicRMWInst::BinOp::UDecWrap:
    switch (ITy->getBitWidth()) {
    case 32:
      return AtomicExpansionKind::None;
    case 8:
    case 16:
    case 64:
      return AtomicExpansionKind::CmpXChg;
    default:
      llvm_unreachable("unsupported width encountered");
    }
  }

  return AtomicExpansionKind::CmpXChg;
}

bool NVPTXTargetLowering::shouldInsertFencesForAtomic(
    const Instruction *I) const {
  auto *CI = dyn_cast<AtomicCmpXchgInst>(I);
  // When CAS bitwidth is not supported on the hardware, the CAS is emulated
  // using a retry loop that uses a higher-bitwidth monotonic CAS. We enforce
  // the memory order using explicit fences around the retry loop.
  // The memory order of natively supported CAS operations can be enforced
  // by lowering to an atom.cas with the right memory synchronizing effect.
  // However, atom.cas only supports relaxed, acquire, release and acq_rel.
  // So we also use explicit fences for enforcing memory order for
  // seq_cast CAS with natively-supported bitwidths.
  return CI &&
         (cast<IntegerType>(CI->getCompareOperand()->getType())->getBitWidth() <
              STI.getMinCmpXchgSizeInBits() ||
          CI->getMergedOrdering() == AtomicOrdering::SequentiallyConsistent);
}

AtomicOrdering NVPTXTargetLowering::atomicOperationOrderAfterFenceSplit(
    const Instruction *I) const {
  auto *CI = dyn_cast<AtomicCmpXchgInst>(I);
  bool BitwidthSupportedAndIsSeqCst =
      CI && CI->getMergedOrdering() == AtomicOrdering::SequentiallyConsistent &&
      cast<IntegerType>(CI->getCompareOperand()->getType())->getBitWidth() >=
          STI.getMinCmpXchgSizeInBits();
  return BitwidthSupportedAndIsSeqCst ? AtomicOrdering::Acquire
                                      : AtomicOrdering::Monotonic;
}

Instruction *NVPTXTargetLowering::emitLeadingFence(IRBuilderBase &Builder,
                                                   Instruction *Inst,
                                                   AtomicOrdering Ord) const {
  if (!isa<AtomicCmpXchgInst>(Inst))
    return TargetLoweringBase::emitLeadingFence(Builder, Inst, Ord);

  // Specialize for cmpxchg
  // Emit a fence.sc leading fence for cmpxchg seq_cst which are not emulated
  SyncScope::ID SSID = cast<AtomicCmpXchgInst>(Inst)->getSyncScopeID();
  if (isReleaseOrStronger(Ord))
    return Builder.CreateFence(Ord == AtomicOrdering::SequentiallyConsistent
                                   ? Ord
                                   : AtomicOrdering::Release,
                               SSID);

  return nullptr;
}

Instruction *NVPTXTargetLowering::emitTrailingFence(IRBuilderBase &Builder,
                                                    Instruction *Inst,
                                                    AtomicOrdering Ord) const {
  // Specialize for cmpxchg
  if (!isa<AtomicCmpXchgInst>(Inst))
    return TargetLoweringBase::emitTrailingFence(Builder, Inst, Ord);

  auto *CI = cast<AtomicCmpXchgInst>(Inst);
  auto CASWidth =
      cast<IntegerType>(CI->getCompareOperand()->getType())->getBitWidth();
  SyncScope::ID SSID = CI->getSyncScopeID();
  // Do not emit a trailing fence for cmpxchg seq_cst which are not emulated
  if (isAcquireOrStronger(Ord) &&
      (Ord != AtomicOrdering::SequentiallyConsistent ||
       CASWidth < STI.getMinCmpXchgSizeInBits()))
    return Builder.CreateFence(AtomicOrdering::Acquire, SSID);

  return nullptr;
}

// Rather than default to SINT when both UINT and SINT are custom, we only
// change the opcode when UINT is not legal and SINT is. UINT is preferred when
// both are custom since unsigned CVT instructions can lead to slightly better
// SASS code with fewer instructions.
unsigned NVPTXTargetLowering::getPreferredFPToIntOpcode(unsigned Op, EVT FromVT,
                                                        EVT ToVT) const {
  if (isOperationLegal(Op, ToVT))
    return Op;
  switch (Op) {
  case ISD::FP_TO_UINT:
    if (isOperationLegal(ISD::FP_TO_SINT, ToVT))
      return ISD::FP_TO_SINT;
    break;
  case ISD::STRICT_FP_TO_UINT:
    if (isOperationLegal(ISD::STRICT_FP_TO_SINT, ToVT))
      return ISD::STRICT_FP_TO_SINT;
    break;
  case ISD::VP_FP_TO_UINT:
    if (isOperationLegal(ISD::VP_FP_TO_SINT, ToVT))
      return ISD::VP_FP_TO_SINT;
    break;
  default:
    break;
  }
  return Op;
}

// Pin NVPTXTargetObjectFile's vtables to this file.
NVPTXTargetObjectFile::~NVPTXTargetObjectFile() = default;

MCSection *NVPTXTargetObjectFile::SelectSectionForGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {
  return getDataSection();
}

static void computeKnownBitsForPRMT(const SDValue Op, KnownBits &Known,
                                    const SelectionDAG &DAG, unsigned Depth) {
  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  ConstantSDNode *Selector = dyn_cast<ConstantSDNode>(Op.getOperand(2));
  unsigned Mode = Op.getConstantOperandVal(3);

  if (!Selector)
    return;

  KnownBits AKnown = DAG.computeKnownBits(A, Depth);
  KnownBits BKnown = DAG.computeKnownBits(B, Depth);

  // {b, a} = {{b7, b6, b5, b4}, {b3, b2, b1, b0}}
  assert(AKnown.getBitWidth() == 32 && BKnown.getBitWidth() == 32 &&
         "PRMT must have i32 operands");
  assert(Known.getBitWidth() == 32 && "PRMT must have i32 result");
  KnownBits BitField = BKnown.concat(AKnown);

  APInt SelectorVal = getPRMTSelector(Selector->getAPIntValue(), Mode);
  for (unsigned I : llvm::seq(4)) {
    APInt Sel = SelectorVal.extractBits(4, I * 4);
    unsigned Idx = Sel.getLoBits(3).getZExtValue();
    unsigned Sign = Sel.getHiBits(1).getZExtValue();
    KnownBits Byte = BitField.extractBits(8, Idx * 8);
    if (Sign)
      Byte = KnownBits::ashr(Byte, 8);
    Known.insertBits(Byte, I * 8);
  }
}

void NVPTXTargetLowering::computeKnownBitsForTargetNode(
    const SDValue Op, KnownBits &Known, const APInt &DemandedElts,
    const SelectionDAG &DAG, unsigned Depth) const {
  Known.resetAll();

  switch (Op.getOpcode()) {
  case NVPTXISD::PRMT:
    computeKnownBitsForPRMT(Op, Known, DAG, Depth);
    break;
  default:
    break;
  }
}

static std::pair<APInt, APInt> getPRMTDemandedBits(const APInt &SelectorVal,
                                                   const APInt &DemandedBits) {
  APInt DemandedLHS = APInt(32, 0);
  APInt DemandedRHS = APInt(32, 0);

  for (unsigned I : llvm::seq(4)) {
    if (DemandedBits.extractBits(8, I * 8).isZero())
      continue;

    APInt Sel = SelectorVal.extractBits(4, I * 4);
    unsigned Idx = Sel.getLoBits(3).getZExtValue();
    unsigned Sign = Sel.getHiBits(1).getZExtValue();

    APInt &Src = Idx < 4 ? DemandedLHS : DemandedRHS;
    unsigned ByteStart = (Idx % 4) * 8;
    if (Sign)
      Src.setBit(ByteStart + 7);
    else
      Src.setBits(ByteStart, ByteStart + 8);
  }

  return {DemandedLHS, DemandedRHS};
}

// Replace undef with 0 as this is easier for other optimizations such as
// known bits.
static SDValue canonicalizePRMTInput(SDValue Op, SelectionDAG &DAG) {
  if (!Op)
    return SDValue();
  if (Op.isUndef())
    return DAG.getConstant(0, SDLoc(), MVT::i32);
  return Op;
}

static SDValue simplifyDemandedBitsForPRMT(SDValue PRMT,
                                           const APInt &DemandedBits,
                                           SelectionDAG &DAG,
                                           const TargetLowering &TLI,
                                           unsigned Depth) {
  assert(PRMT.getOpcode() == NVPTXISD::PRMT);
  SDValue Op0 = PRMT.getOperand(0);
  SDValue Op1 = PRMT.getOperand(1);
  auto *SelectorConst = dyn_cast<ConstantSDNode>(PRMT.getOperand(2));
  if (!SelectorConst)
    return SDValue();

  unsigned Mode = PRMT.getConstantOperandVal(3);
  const APInt Selector = getPRMTSelector(SelectorConst->getAPIntValue(), Mode);

  // Try to simplify the PRMT to one of the inputs if the used bytes are all
  // from the same input in the correct order.
  const unsigned LeadingBytes = DemandedBits.countLeadingZeros() / 8;
  const unsigned SelBits = (4 - LeadingBytes) * 4;
  if (Selector.getLoBits(SelBits) == APInt(32, 0x3210).getLoBits(SelBits))
    return Op0;
  if (Selector.getLoBits(SelBits) == APInt(32, 0x7654).getLoBits(SelBits))
    return Op1;

  auto [DemandedLHS, DemandedRHS] = getPRMTDemandedBits(Selector, DemandedBits);

  // Attempt to avoid multi-use ops if we don't need anything from them.
  SDValue DemandedOp0 =
      TLI.SimplifyMultipleUseDemandedBits(Op0, DemandedLHS, DAG, Depth + 1);
  SDValue DemandedOp1 =
      TLI.SimplifyMultipleUseDemandedBits(Op1, DemandedRHS, DAG, Depth + 1);

  DemandedOp0 = canonicalizePRMTInput(DemandedOp0, DAG);
  DemandedOp1 = canonicalizePRMTInput(DemandedOp1, DAG);
  if ((DemandedOp0 && DemandedOp0 != Op0) ||
      (DemandedOp1 && DemandedOp1 != Op1)) {
    Op0 = DemandedOp0 ? DemandedOp0 : Op0;
    Op1 = DemandedOp1 ? DemandedOp1 : Op1;
    return getPRMT(Op0, Op1, Selector.getZExtValue(), SDLoc(PRMT), DAG);
  }

  return SDValue();
}

bool NVPTXTargetLowering::SimplifyDemandedBitsForTargetNode(
    SDValue Op, const APInt &DemandedBits, const APInt &DemandedElts,
    KnownBits &Known, TargetLoweringOpt &TLO, unsigned Depth) const {
  Known.resetAll();

  switch (Op.getOpcode()) {
  case NVPTXISD::PRMT:
    if (SDValue Result = simplifyDemandedBitsForPRMT(Op, DemandedBits, TLO.DAG,
                                                     *this, Depth)) {
      TLO.CombineTo(Op, Result);
      return true;
    }
    break;
  default:
    break;
  }

  computeKnownBitsForTargetNode(Op, Known, DemandedElts, TLO.DAG, Depth);
  return false;
}
