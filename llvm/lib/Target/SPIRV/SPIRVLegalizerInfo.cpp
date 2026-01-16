//===- SPIRVLegalizerInfo.cpp --- SPIR-V Legalization Rules ------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the targeting of the Machinelegalizer class for SPIR-V.
//
//===----------------------------------------------------------------------===//

#include "SPIRVLegalizerInfo.h"
#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::LegalizeActions;
using namespace llvm::LegalityPredicates;

#define DEBUG_TYPE "spirv-legalizer"

LegalityPredicate typeOfExtendedScalars(unsigned TypeIdx, bool IsExtendedInts) {
  return [IsExtendedInts, TypeIdx](const LegalityQuery &Query) {
    const LLT Ty = Query.Types[TypeIdx];
    return IsExtendedInts && Ty.isValid() && Ty.isScalar();
  };
}

SPIRVLegalizerInfo::SPIRVLegalizerInfo(const SPIRVSubtarget &ST) {
  using namespace TargetOpcode;

  this->ST = &ST;
  GR = ST.getSPIRVGlobalRegistry();

  const LLT s1 = LLT::scalar(1);
  const LLT s8 = LLT::scalar(8);
  const LLT s16 = LLT::scalar(16);
  const LLT s32 = LLT::scalar(32);
  const LLT s64 = LLT::scalar(64);
  const LLT s128 = LLT::scalar(128);

  const LLT v16s64 = LLT::fixed_vector(16, 64);
  const LLT v16s32 = LLT::fixed_vector(16, 32);
  const LLT v16s16 = LLT::fixed_vector(16, 16);
  const LLT v16s8 = LLT::fixed_vector(16, 8);
  const LLT v16s1 = LLT::fixed_vector(16, 1);

  const LLT v8s64 = LLT::fixed_vector(8, 64);
  const LLT v8s32 = LLT::fixed_vector(8, 32);
  const LLT v8s16 = LLT::fixed_vector(8, 16);
  const LLT v8s8 = LLT::fixed_vector(8, 8);
  const LLT v8s1 = LLT::fixed_vector(8, 1);

  const LLT v4s64 = LLT::fixed_vector(4, 64);
  const LLT v4s32 = LLT::fixed_vector(4, 32);
  const LLT v4s16 = LLT::fixed_vector(4, 16);
  const LLT v4s8 = LLT::fixed_vector(4, 8);
  const LLT v4s1 = LLT::fixed_vector(4, 1);

  const LLT v3s64 = LLT::fixed_vector(3, 64);
  const LLT v3s32 = LLT::fixed_vector(3, 32);
  const LLT v3s16 = LLT::fixed_vector(3, 16);
  const LLT v3s8 = LLT::fixed_vector(3, 8);
  const LLT v3s1 = LLT::fixed_vector(3, 1);

  const LLT v2s64 = LLT::fixed_vector(2, 64);
  const LLT v2s32 = LLT::fixed_vector(2, 32);
  const LLT v2s16 = LLT::fixed_vector(2, 16);
  const LLT v2s8 = LLT::fixed_vector(2, 8);
  const LLT v2s1 = LLT::fixed_vector(2, 1);

  const unsigned PSize = ST.getPointerSize();
  const LLT p0 = LLT::pointer(0, PSize); // Function
  const LLT p1 = LLT::pointer(1, PSize); // CrossWorkgroup
  const LLT p2 = LLT::pointer(2, PSize); // UniformConstant
  const LLT p3 = LLT::pointer(3, PSize); // Workgroup
  const LLT p4 = LLT::pointer(4, PSize); // Generic
  const LLT p5 =
      LLT::pointer(5, PSize); // Input, SPV_INTEL_usm_storage_classes (Device)
  const LLT p6 = LLT::pointer(6, PSize); // SPV_INTEL_usm_storage_classes (Host)
  const LLT p7 = LLT::pointer(7, PSize); // Input
  const LLT p8 = LLT::pointer(8, PSize); // Output
  const LLT p9 =
      LLT::pointer(9, PSize); // CodeSectionINTEL, SPV_INTEL_function_pointers
  const LLT p10 = LLT::pointer(10, PSize); // Private
  const LLT p11 = LLT::pointer(11, PSize); // StorageBuffer
  const LLT p12 = LLT::pointer(12, PSize); // Uniform
  const LLT p13 = LLT::pointer(13, PSize); // PushConstant

  // TODO: remove copy-pasting here by using concatenation in some way.
  auto allPtrsScalarsAndVectors = {
      p0,    p1,    p2,    p3,    p4,    p5,     p6,     p7,    p8,
      p9,    p10,   p11,   p12,   p13,   s1,     s8,     s16,   s32,
      s64,   v2s1,  v2s8,  v2s16, v2s32, v2s64,  v3s1,   v3s8,  v3s16,
      v3s32, v3s64, v4s1,  v4s8,  v4s16, v4s32,  v4s64,  v8s1,  v8s8,
      v8s16, v8s32, v8s64, v16s1, v16s8, v16s16, v16s32, v16s64};

  auto allVectors = {v2s1,  v2s8,   v2s16,  v2s32, v2s64, v3s1,  v3s8,
                     v3s16, v3s32,  v3s64,  v4s1,  v4s8,  v4s16, v4s32,
                     v4s64, v8s1,   v8s8,   v8s16, v8s32, v8s64, v16s1,
                     v16s8, v16s16, v16s32, v16s64};

  auto allShaderVectors = {v2s1, v2s8, v2s16, v2s32, v2s64,
                           v3s1, v3s8, v3s16, v3s32, v3s64,
                           v4s1, v4s8, v4s16, v4s32, v4s64};

  auto allScalars = {s1, s8, s16, s32, s64};

  auto allScalarsAndVectors = {
      s1,    s8,    s16,   s32,   s64,    s128,   v2s1,  v2s8,
      v2s16, v2s32, v2s64, v3s1,  v3s8,   v3s16,  v3s32, v3s64,
      v4s1,  v4s8,  v4s16, v4s32, v4s64,  v8s1,   v8s8,  v8s16,
      v8s32, v8s64, v16s1, v16s8, v16s16, v16s32, v16s64};

  auto allIntScalarsAndVectors = {
      s8,    s16,   s32,   s64,   s128,   v2s8,   v2s16, v2s32, v2s64,
      v3s8,  v3s16, v3s32, v3s64, v4s8,   v4s16,  v4s32, v4s64, v8s8,
      v8s16, v8s32, v8s64, v16s8, v16s16, v16s32, v16s64};

  auto allBoolScalarsAndVectors = {s1, v2s1, v3s1, v4s1, v8s1, v16s1};

  auto allIntScalars = {s8, s16, s32, s64, s128};

  auto allFloatScalarsAndF16Vector2AndVector4s = {s16, s32, s64, v2s16, v4s16};

  auto allFloatScalarsAndVectors = {
      s16,   s32,   s64,   v2s16, v2s32, v2s64, v3s16,  v3s32,  v3s64,
      v4s16, v4s32, v4s64, v8s16, v8s32, v8s64, v16s16, v16s32, v16s64};

  auto allFloatAndIntScalarsAndPtrs = {s8, s16, s32, s64, p0,  p1,
                                       p2, p3,  p4,  p5,  p6,  p7,
                                       p8, p9,  p10, p11, p12, p13};

  auto allPtrs = {p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13};

  auto &allowedVectorTypes = ST.isShader() ? allShaderVectors : allVectors;

  bool IsExtendedInts =
      ST.canUseExtension(
          SPIRV::Extension::SPV_ALTERA_arbitrary_precision_integers) ||
      ST.canUseExtension(SPIRV::Extension::SPV_KHR_bit_instructions) ||
      ST.canUseExtension(SPIRV::Extension::SPV_INTEL_int4);
  auto extendedScalarsAndVectors =
      [IsExtendedInts](const LegalityQuery &Query) {
        const LLT Ty = Query.Types[0];
        return IsExtendedInts && Ty.isValid() && !Ty.isPointerOrPointerVector();
      };
  auto extendedScalarsAndVectorsProduct = [IsExtendedInts](
                                              const LegalityQuery &Query) {
    const LLT Ty1 = Query.Types[0], Ty2 = Query.Types[1];
    return IsExtendedInts && Ty1.isValid() && Ty2.isValid() &&
           !Ty1.isPointerOrPointerVector() && !Ty2.isPointerOrPointerVector();
  };
  auto extendedPtrsScalarsAndVectors =
      [IsExtendedInts](const LegalityQuery &Query) {
        const LLT Ty = Query.Types[0];
        return IsExtendedInts && Ty.isValid();
      };

  // The universal validation rules in the SPIR-V specification state that
  // vector sizes are typically limited to 2, 3, or 4. However, larger vector
  // sizes (8 and 16) are enabled when the Kernel capability is present. For
  // shader execution models, vector sizes are strictly limited to 4. In
  // non-shader contexts, vector sizes of 8 and 16 are also permitted, but
  // arbitrary sizes (e.g., 6 or 11) are not.
  uint32_t MaxVectorSize = ST.isShader() ? 4 : 16;
  LLVM_DEBUG(dbgs() << "MaxVectorSize: " << MaxVectorSize << "\n");

  for (auto Opc : getTypeFoldingSupportedOpcodes()) {
    switch (Opc) {
    case G_EXTRACT_VECTOR_ELT:
    case G_UREM:
    case G_SREM:
    case G_UDIV:
    case G_SDIV:
    case G_FREM:
      break;
    default:
      getActionDefinitionsBuilder(Opc)
          .customFor(allScalars)
          .customFor(allowedVectorTypes)
          .moreElementsToNextPow2(0)
          .fewerElementsIf(vectorElementCountIsGreaterThan(0, MaxVectorSize),
                           LegalizeMutations::changeElementCountTo(
                               0, ElementCount::getFixed(MaxVectorSize)))
          .custom();
      break;
    }
  }

  getActionDefinitionsBuilder({G_UREM, G_SREM, G_SDIV, G_UDIV, G_FREM})
      .customFor(allScalars)
      .customFor(allowedVectorTypes)
      .scalarizeIf(numElementsNotPow2(0), 0)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, MaxVectorSize),
                       LegalizeMutations::changeElementCountTo(
                           0, ElementCount::getFixed(MaxVectorSize)))
      .custom();

  getActionDefinitionsBuilder({G_FMA, G_STRICT_FMA})
      .legalFor(allScalars)
      .legalFor(allowedVectorTypes)
      .moreElementsToNextPow2(0)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, MaxVectorSize),
                       LegalizeMutations::changeElementCountTo(
                           0, ElementCount::getFixed(MaxVectorSize)))
      .alwaysLegal();

  getActionDefinitionsBuilder(G_INTRINSIC_W_SIDE_EFFECTS).custom();

  getActionDefinitionsBuilder(G_SHUFFLE_VECTOR)
      .legalForCartesianProduct(allowedVectorTypes, allowedVectorTypes)
      .moreElementsToNextPow2(0)
      .lowerIf(vectorElementCountIsGreaterThan(0, MaxVectorSize))
      .moreElementsToNextPow2(1)
      .lowerIf(vectorElementCountIsGreaterThan(1, MaxVectorSize));

  getActionDefinitionsBuilder(G_EXTRACT_VECTOR_ELT)
      .moreElementsToNextPow2(1)
      .fewerElementsIf(vectorElementCountIsGreaterThan(1, MaxVectorSize),
                       LegalizeMutations::changeElementCountTo(
                           1, ElementCount::getFixed(MaxVectorSize)))
      .custom();

  getActionDefinitionsBuilder(G_INSERT_VECTOR_ELT)
      .moreElementsToNextPow2(0)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, MaxVectorSize),
                       LegalizeMutations::changeElementCountTo(
                           0, ElementCount::getFixed(MaxVectorSize)))
      .custom();

  // Illegal G_UNMERGE_VALUES instructions should be handled
  // during the combine phase.
  getActionDefinitionsBuilder(G_BUILD_VECTOR)
      .legalIf(vectorElementCountIsLessThanOrEqualTo(0, MaxVectorSize));

  // When entering the legalizer, there should be no G_BITCAST instructions.
  // They should all be calls to the `spv_bitcast` intrinsic. The call to
  // the intrinsic will be converted to a G_BITCAST during legalization if
  // the vectors are not legal. After using the rules to legalize a G_BITCAST,
  // we turn it back into a call to the intrinsic with a custom rule to avoid
  // potential machine verifier failures.
  getActionDefinitionsBuilder(G_BITCAST)
      .moreElementsToNextPow2(0)
      .moreElementsToNextPow2(1)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, MaxVectorSize),
                       LegalizeMutations::changeElementCountTo(
                           0, ElementCount::getFixed(MaxVectorSize)))
      .lowerIf(vectorElementCountIsGreaterThan(1, MaxVectorSize))
      .custom();

  // If the result is still illegal, the combiner should be able to remove it.
  getActionDefinitionsBuilder(G_CONCAT_VECTORS)
      .legalForCartesianProduct(allowedVectorTypes, allowedVectorTypes);

  getActionDefinitionsBuilder(G_SPLAT_VECTOR)
      .legalFor(allowedVectorTypes)
      .moreElementsToNextPow2(0)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, MaxVectorSize),
                       LegalizeMutations::changeElementSizeTo(0, MaxVectorSize))
      .alwaysLegal();

  // Vector Reduction Operations
  getActionDefinitionsBuilder(
      {G_VECREDUCE_SMIN, G_VECREDUCE_SMAX, G_VECREDUCE_UMIN, G_VECREDUCE_UMAX,
       G_VECREDUCE_ADD, G_VECREDUCE_MUL, G_VECREDUCE_FMUL, G_VECREDUCE_FMIN,
       G_VECREDUCE_FMAX, G_VECREDUCE_FMINIMUM, G_VECREDUCE_FMAXIMUM,
       G_VECREDUCE_OR, G_VECREDUCE_AND, G_VECREDUCE_XOR})
      .legalFor(allowedVectorTypes)
      .scalarize(1)
      .lower();

  getActionDefinitionsBuilder({G_VECREDUCE_SEQ_FADD, G_VECREDUCE_SEQ_FMUL})
      .scalarize(2)
      .lower();

  // Illegal G_UNMERGE_VALUES instructions should be handled
  // during the combine phase.
  getActionDefinitionsBuilder(G_UNMERGE_VALUES)
      .legalIf(vectorElementCountIsLessThanOrEqualTo(1, MaxVectorSize));

  getActionDefinitionsBuilder({G_MEMCPY, G_MEMMOVE})
      .unsupportedIf(LegalityPredicates::any(typeIs(0, p9), typeIs(1, p9)))
      .legalIf(all(typeInSet(0, allPtrs), typeInSet(1, allPtrs)));

  getActionDefinitionsBuilder(G_MEMSET)
      .unsupportedIf(typeIs(0, p9))
      .legalIf(all(typeInSet(0, allPtrs), typeInSet(1, allIntScalars)));

  getActionDefinitionsBuilder(G_ADDRSPACE_CAST)
      .unsupportedIf(
          LegalityPredicates::any(all(typeIs(0, p9), typeIsNot(1, p9)),
                                  all(typeIsNot(0, p9), typeIs(1, p9))))
      .legalForCartesianProduct(allPtrs, allPtrs);

  // Should we be legalizing bad scalar sizes like s5 here instead
  // of handling them in the instruction selector?
  getActionDefinitionsBuilder({G_LOAD, G_STORE})
      .unsupportedIf(typeIs(1, p9))
      .legalForCartesianProduct(allowedVectorTypes, allPtrs)
      .legalForCartesianProduct(allPtrs, allPtrs)
      .legalIf(isScalar(0))
      .custom();

  getActionDefinitionsBuilder({G_SMIN, G_SMAX, G_UMIN, G_UMAX, G_ABS,
                               G_BITREVERSE, G_SADDSAT, G_UADDSAT, G_SSUBSAT,
                               G_USUBSAT, G_SCMP, G_UCMP})
      .legalFor(allIntScalarsAndVectors)
      .legalIf(extendedScalarsAndVectors);

  getActionDefinitionsBuilder(G_STRICT_FLDEXP)
      .legalForCartesianProduct(allFloatScalarsAndVectors, allIntScalars);

  getActionDefinitionsBuilder({G_FPTOSI, G_FPTOUI})
      .legalForCartesianProduct(allIntScalarsAndVectors,
                                allFloatScalarsAndVectors);

  getActionDefinitionsBuilder({G_FPTOSI_SAT, G_FPTOUI_SAT})
      .legalForCartesianProduct(allIntScalarsAndVectors,
                                allFloatScalarsAndVectors);

  getActionDefinitionsBuilder({G_SITOFP, G_UITOFP})
      .legalForCartesianProduct(allFloatScalarsAndVectors,
                                allScalarsAndVectors);

  getActionDefinitionsBuilder(G_CTPOP)
      .legalForCartesianProduct(allIntScalarsAndVectors)
      .legalIf(extendedScalarsAndVectorsProduct);

  // Extensions.
  getActionDefinitionsBuilder({G_TRUNC, G_ZEXT, G_SEXT, G_ANYEXT})
      .legalForCartesianProduct(allScalarsAndVectors)
      .legalIf(extendedScalarsAndVectorsProduct);

  getActionDefinitionsBuilder(G_PHI)
      .legalFor(allPtrsScalarsAndVectors)
      .legalIf(extendedPtrsScalarsAndVectors);

  getActionDefinitionsBuilder(G_BITCAST).legalIf(
      all(typeInSet(0, allPtrsScalarsAndVectors),
          typeInSet(1, allPtrsScalarsAndVectors)));

  getActionDefinitionsBuilder({G_IMPLICIT_DEF, G_FREEZE})
      .legalFor({s1, s128})
      .legalFor(allFloatAndIntScalarsAndPtrs)
      .legalFor(allowedVectorTypes)
      .moreElementsToNextPow2(0)
      .fewerElementsIf(vectorElementCountIsGreaterThan(0, MaxVectorSize),
                       LegalizeMutations::changeElementCountTo(
                           0, ElementCount::getFixed(MaxVectorSize)));

  getActionDefinitionsBuilder({G_STACKSAVE, G_STACKRESTORE}).alwaysLegal();

  getActionDefinitionsBuilder(G_INTTOPTR)
      .legalForCartesianProduct(allPtrs, allIntScalars)
      .legalIf(
          all(typeInSet(0, allPtrs), typeOfExtendedScalars(1, IsExtendedInts)));
  getActionDefinitionsBuilder(G_PTRTOINT)
      .legalForCartesianProduct(allIntScalars, allPtrs)
      .legalIf(
          all(typeOfExtendedScalars(0, IsExtendedInts), typeInSet(1, allPtrs)));
  getActionDefinitionsBuilder(G_PTR_ADD)
      .legalForCartesianProduct(allPtrs, allIntScalars)
      .legalIf(
          all(typeInSet(0, allPtrs), typeOfExtendedScalars(1, IsExtendedInts)));

  // ST.canDirectlyComparePointers() for pointer args is supported in
  // legalizeCustom().
  getActionDefinitionsBuilder(G_ICMP)
      .unsupportedIf(LegalityPredicates::any(
          all(typeIs(0, p9), typeInSet(1, allPtrs), typeIsNot(1, p9)),
          all(typeInSet(0, allPtrs), typeIsNot(0, p9), typeIs(1, p9))))
      .customIf(all(typeInSet(0, allBoolScalarsAndVectors),
                    typeInSet(1, allPtrsScalarsAndVectors)));

  getActionDefinitionsBuilder(G_FCMP).legalIf(
      all(typeInSet(0, allBoolScalarsAndVectors),
          typeInSet(1, allFloatScalarsAndVectors)));

  getActionDefinitionsBuilder({G_ATOMICRMW_OR, G_ATOMICRMW_ADD, G_ATOMICRMW_AND,
                               G_ATOMICRMW_MAX, G_ATOMICRMW_MIN,
                               G_ATOMICRMW_SUB, G_ATOMICRMW_XOR,
                               G_ATOMICRMW_UMAX, G_ATOMICRMW_UMIN})
      .legalForCartesianProduct(allIntScalars, allPtrs);

  getActionDefinitionsBuilder(
      {G_ATOMICRMW_FADD, G_ATOMICRMW_FSUB, G_ATOMICRMW_FMIN, G_ATOMICRMW_FMAX})
      .legalForCartesianProduct(allFloatScalarsAndF16Vector2AndVector4s,
                                allPtrs);

  getActionDefinitionsBuilder(G_ATOMICRMW_XCHG)
      .legalForCartesianProduct(allFloatAndIntScalarsAndPtrs, allPtrs);

  getActionDefinitionsBuilder(G_ATOMIC_CMPXCHG_WITH_SUCCESS).lower();
  // TODO: add proper legalization rules.
  getActionDefinitionsBuilder(G_ATOMIC_CMPXCHG).alwaysLegal();

  getActionDefinitionsBuilder(
      {G_UADDO, G_SADDO, G_USUBO, G_SSUBO, G_UMULO, G_SMULO})
      .alwaysLegal();

  getActionDefinitionsBuilder({G_LROUND, G_LLROUND})
      .legalForCartesianProduct(allFloatScalarsAndVectors,
                                allIntScalarsAndVectors);

  // FP conversions.
  getActionDefinitionsBuilder({G_FPTRUNC, G_FPEXT})
      .legalForCartesianProduct(allFloatScalarsAndVectors);

  // Pointer-handling.
  getActionDefinitionsBuilder(G_FRAME_INDEX).legalFor({p0});

  getActionDefinitionsBuilder(G_GLOBAL_VALUE).legalFor(allPtrs);

  // Control-flow. In some cases (e.g. constants) s1 may be promoted to s32.
  getActionDefinitionsBuilder(G_BRCOND).legalFor({s1, s32});

  getActionDefinitionsBuilder(G_FFREXP).legalForCartesianProduct(
      allFloatScalarsAndVectors, {s32, v2s32, v3s32, v4s32, v8s32, v16s32});

  // TODO: Review the target OpenCL and GLSL Extended Instruction Set specs to
  // tighten these requirements. Many of these math functions are only legal on
  // specific bitwidths, so they are not selectable for
  // allFloatScalarsAndVectors.
  getActionDefinitionsBuilder({G_STRICT_FSQRT,
                               G_FPOW,
                               G_FEXP,
                               G_FMODF,
                               G_FEXP2,
                               G_FLOG,
                               G_FLOG2,
                               G_FLOG10,
                               G_FABS,
                               G_FMINNUM,
                               G_FMAXNUM,
                               G_FCEIL,
                               G_FCOS,
                               G_FSIN,
                               G_FTAN,
                               G_FACOS,
                               G_FASIN,
                               G_FATAN,
                               G_FATAN2,
                               G_FCOSH,
                               G_FSINH,
                               G_FTANH,
                               G_FSQRT,
                               G_FFLOOR,
                               G_FRINT,
                               G_FNEARBYINT,
                               G_INTRINSIC_ROUND,
                               G_INTRINSIC_TRUNC,
                               G_FMINIMUM,
                               G_FMAXIMUM,
                               G_INTRINSIC_ROUNDEVEN})
      .legalFor(allFloatScalarsAndVectors);

  getActionDefinitionsBuilder(G_FCOPYSIGN)
      .legalForCartesianProduct(allFloatScalarsAndVectors,
                                allFloatScalarsAndVectors);

  getActionDefinitionsBuilder(G_FPOWI).legalForCartesianProduct(
      allFloatScalarsAndVectors, allIntScalarsAndVectors);

  if (ST.canUseExtInstSet(SPIRV::InstructionSet::OpenCL_std)) {
    getActionDefinitionsBuilder(
        {G_CTTZ, G_CTTZ_ZERO_UNDEF, G_CTLZ, G_CTLZ_ZERO_UNDEF})
        .legalForCartesianProduct(allIntScalarsAndVectors,
                                  allIntScalarsAndVectors);

    // Struct return types become a single scalar, so cannot easily legalize.
    getActionDefinitionsBuilder({G_SMULH, G_UMULH}).alwaysLegal();
  }

  getActionDefinitionsBuilder(G_IS_FPCLASS).custom();

  getLegacyLegalizerInfo().computeTables();
  verify(*ST.getInstrInfo());
}

static bool legalizeExtractVectorElt(LegalizerHelper &Helper, MachineInstr &MI,
                                     SPIRVGlobalRegistry *GR) {
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  Register IdxReg = MI.getOperand(2).getReg();

  MIRBuilder
      .buildIntrinsic(Intrinsic::spv_extractelt, ArrayRef<Register>{DstReg})
      .addUse(SrcReg)
      .addUse(IdxReg);
  MI.eraseFromParent();
  return true;
}

static bool legalizeInsertVectorElt(LegalizerHelper &Helper, MachineInstr &MI,
                                    SPIRVGlobalRegistry *GR) {
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  Register ValReg = MI.getOperand(2).getReg();
  Register IdxReg = MI.getOperand(3).getReg();

  MIRBuilder
      .buildIntrinsic(Intrinsic::spv_insertelt, ArrayRef<Register>{DstReg})
      .addUse(SrcReg)
      .addUse(ValReg)
      .addUse(IdxReg);
  MI.eraseFromParent();
  return true;
}

static Register convertPtrToInt(Register Reg, LLT ConvTy, SPIRVType *SpvType,
                                LegalizerHelper &Helper,
                                MachineRegisterInfo &MRI,
                                SPIRVGlobalRegistry *GR) {
  Register ConvReg = MRI.createGenericVirtualRegister(ConvTy);
  MRI.setRegClass(ConvReg, GR->getRegClass(SpvType));
  GR->assignSPIRVTypeToVReg(SpvType, ConvReg, Helper.MIRBuilder.getMF());
  Helper.MIRBuilder.buildInstr(TargetOpcode::G_PTRTOINT)
      .addDef(ConvReg)
      .addUse(Reg);
  return ConvReg;
}

static bool needsVectorLegalization(const LLT &Ty, const SPIRVSubtarget &ST) {
  if (!Ty.isVector())
    return false;
  unsigned NumElements = Ty.getNumElements();
  unsigned MaxVectorSize = ST.isShader() ? 4 : 16;
  return (NumElements > 4 && !isPowerOf2_32(NumElements)) ||
         NumElements > MaxVectorSize;
}

static bool legalizeLoad(LegalizerHelper &Helper, MachineInstr &MI,
                         SPIRVGlobalRegistry *GR) {
  MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  Register DstReg = MI.getOperand(0).getReg();
  Register PtrReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);

  if (!DstTy.isVector())
    return true;

  const SPIRVSubtarget &ST = MI.getMF()->getSubtarget<SPIRVSubtarget>();
  if (!needsVectorLegalization(DstTy, ST))
    return true;

  SmallVector<Register, 8> SplitRegs;
  LLT EltTy = DstTy.getElementType();
  unsigned NumElts = DstTy.getNumElements();

  LLT PtrTy = MRI.getType(PtrReg);
  auto Zero = MIRBuilder.buildConstant(LLT::scalar(32), 0);

  for (unsigned i = 0; i < NumElts; ++i) {
    auto Idx = MIRBuilder.buildConstant(LLT::scalar(32), i);
    Register EltPtr = MRI.createGenericVirtualRegister(PtrTy);

    MIRBuilder.buildIntrinsic(Intrinsic::spv_gep, ArrayRef<Register>{EltPtr})
        .addImm(1) // InBounds
        .addUse(PtrReg)
        .addUse(Zero.getReg(0))
        .addUse(Idx.getReg(0));

    MachinePointerInfo EltPtrInfo;
    Align EltAlign = Align(1);
    if (!MI.memoperands_empty()) {
      MachineMemOperand *MMO = *MI.memoperands_begin();
      EltPtrInfo =
          MMO->getPointerInfo().getWithOffset(i * EltTy.getSizeInBytes());
      EltAlign = commonAlignment(MMO->getAlign(), i * EltTy.getSizeInBytes());
    }

    Register EltReg = MRI.createGenericVirtualRegister(EltTy);
    MIRBuilder.buildLoad(EltReg, EltPtr, EltPtrInfo, EltAlign);
    SplitRegs.push_back(EltReg);
  }

  MIRBuilder.buildBuildVector(DstReg, SplitRegs);
  MI.eraseFromParent();
  return true;
}

static bool legalizeStore(LegalizerHelper &Helper, MachineInstr &MI,
                          SPIRVGlobalRegistry *GR) {
  MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  Register ValReg = MI.getOperand(0).getReg();
  Register PtrReg = MI.getOperand(1).getReg();
  LLT ValTy = MRI.getType(ValReg);

  assert(ValTy.isVector() && "Expected vector store");

  SmallVector<Register, 8> SplitRegs;
  LLT EltTy = ValTy.getElementType();
  unsigned NumElts = ValTy.getNumElements();

  for (unsigned i = 0; i < NumElts; ++i)
    SplitRegs.push_back(MRI.createGenericVirtualRegister(EltTy));

  MIRBuilder.buildUnmerge(SplitRegs, ValReg);

  LLT PtrTy = MRI.getType(PtrReg);
  auto Zero = MIRBuilder.buildConstant(LLT::scalar(32), 0);

  for (unsigned i = 0; i < NumElts; ++i) {
    auto Idx = MIRBuilder.buildConstant(LLT::scalar(32), i);
    Register EltPtr = MRI.createGenericVirtualRegister(PtrTy);

    MIRBuilder.buildIntrinsic(Intrinsic::spv_gep, ArrayRef<Register>{EltPtr})
        .addImm(1) // InBounds
        .addUse(PtrReg)
        .addUse(Zero.getReg(0))
        .addUse(Idx.getReg(0));

    MachinePointerInfo EltPtrInfo;
    Align EltAlign = Align(1);
    if (!MI.memoperands_empty()) {
      MachineMemOperand *MMO = *MI.memoperands_begin();
      EltPtrInfo =
          MMO->getPointerInfo().getWithOffset(i * EltTy.getSizeInBytes());
      EltAlign = commonAlignment(MMO->getAlign(), i * EltTy.getSizeInBytes());
    }

    MIRBuilder.buildStore(SplitRegs[i], EltPtr, EltPtrInfo, EltAlign);
  }

  MI.eraseFromParent();
  return true;
}

bool SPIRVLegalizerInfo::legalizeCustom(
    LegalizerHelper &Helper, MachineInstr &MI,
    LostDebugLocObserver &LocObserver) const {
  MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  switch (MI.getOpcode()) {
  default:
    // TODO: implement legalization for other opcodes.
    return true;
  case TargetOpcode::G_BITCAST:
    return legalizeBitcast(Helper, MI);
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
    return legalizeExtractVectorElt(Helper, MI, GR);
  case TargetOpcode::G_INSERT_VECTOR_ELT:
    return legalizeInsertVectorElt(Helper, MI, GR);
  case TargetOpcode::G_INTRINSIC:
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS:
    return legalizeIntrinsic(Helper, MI);
  case TargetOpcode::G_IS_FPCLASS:
    return legalizeIsFPClass(Helper, MI, LocObserver);
  case TargetOpcode::G_ICMP: {
    assert(GR->getSPIRVTypeForVReg(MI.getOperand(0).getReg()));
    auto &Op0 = MI.getOperand(2);
    auto &Op1 = MI.getOperand(3);
    Register Reg0 = Op0.getReg();
    Register Reg1 = Op1.getReg();
    CmpInst::Predicate Cond =
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
    if ((!ST->canDirectlyComparePointers() ||
         (Cond != CmpInst::ICMP_EQ && Cond != CmpInst::ICMP_NE)) &&
        MRI.getType(Reg0).isPointer() && MRI.getType(Reg1).isPointer()) {
      LLT ConvT = LLT::scalar(ST->getPointerSize());
      Type *LLVMTy = IntegerType::get(MI.getMF()->getFunction().getContext(),
                                      ST->getPointerSize());
      SPIRVType *SpirvTy = GR->getOrCreateSPIRVType(
          LLVMTy, Helper.MIRBuilder, SPIRV::AccessQualifier::ReadWrite, true);
      Op0.setReg(convertPtrToInt(Reg0, ConvT, SpirvTy, Helper, MRI, GR));
      Op1.setReg(convertPtrToInt(Reg1, ConvT, SpirvTy, Helper, MRI, GR));
    }
    return true;
  }
  case TargetOpcode::G_LOAD:
    return legalizeLoad(Helper, MI, GR);
  case TargetOpcode::G_STORE:
    return legalizeStore(Helper, MI, GR);
  }
}

static MachineInstrBuilder
createStackTemporaryForVector(LegalizerHelper &Helper, SPIRVGlobalRegistry *GR,
                              Register SrcReg, LLT SrcTy,
                              MachinePointerInfo &PtrInfo, Align &VecAlign) {
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  MachineRegisterInfo &MRI = *MIRBuilder.getMRI();

  VecAlign = Helper.getStackTemporaryAlignment(SrcTy);
  auto StackTemp = Helper.createStackTemporary(
      TypeSize::getFixed(SrcTy.getSizeInBytes()), VecAlign, PtrInfo);

  // Set the type of StackTemp to a pointer to an array of the element type.
  SPIRVType *SpvSrcTy = GR->getSPIRVTypeForVReg(SrcReg);
  SPIRVType *EltSpvTy = GR->getScalarOrVectorComponentType(SpvSrcTy);
  const Type *LLVMEltTy = GR->getTypeForSPIRVType(EltSpvTy);
  const Type *LLVMArrTy =
      ArrayType::get(const_cast<Type *>(LLVMEltTy), SrcTy.getNumElements());
  SPIRVType *ArrSpvTy = GR->getOrCreateSPIRVType(
      LLVMArrTy, MIRBuilder, SPIRV::AccessQualifier::ReadWrite, true);
  SPIRVType *PtrToArrSpvTy = GR->getOrCreateSPIRVPointerType(
      ArrSpvTy, MIRBuilder, SPIRV::StorageClass::Function);

  Register StackReg = StackTemp.getReg(0);
  MRI.setRegClass(StackReg, GR->getRegClass(PtrToArrSpvTy));
  GR->assignSPIRVTypeToVReg(PtrToArrSpvTy, StackReg, MIRBuilder.getMF());

  return StackTemp;
}

static bool legalizeSpvBitcast(LegalizerHelper &Helper, MachineInstr &MI,
                               SPIRVGlobalRegistry *GR) {
  LLVM_DEBUG(dbgs() << "Found a bitcast instruction\n");
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  MachineRegisterInfo &MRI = *MIRBuilder.getMRI();
  const SPIRVSubtarget &ST = MI.getMF()->getSubtarget<SPIRVSubtarget>();

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(2).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);

  // If an spv_bitcast needs to be legalized, we convert it to G_BITCAST to
  // allow using the generic legalization rules.
  if (needsVectorLegalization(DstTy, ST) ||
      needsVectorLegalization(SrcTy, ST)) {
    LLVM_DEBUG(dbgs() << "Replacing with a G_BITCAST\n");
    MIRBuilder.buildBitcast(DstReg, SrcReg);
    MI.eraseFromParent();
  }
  return true;
}

static bool legalizeSpvInsertElt(LegalizerHelper &Helper, MachineInstr &MI,
                                 SPIRVGlobalRegistry *GR) {
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  MachineRegisterInfo &MRI = *MIRBuilder.getMRI();
  const SPIRVSubtarget &ST = MI.getMF()->getSubtarget<SPIRVSubtarget>();

  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);

  if (needsVectorLegalization(DstTy, ST)) {
    Register SrcReg = MI.getOperand(2).getReg();
    Register ValReg = MI.getOperand(3).getReg();
    LLT SrcTy = MRI.getType(SrcReg);
    MachineOperand &IdxOperand = MI.getOperand(4);

    if (getImm(IdxOperand, &MRI)) {
      uint64_t IdxVal = foldImm(IdxOperand, &MRI);
      if (IdxVal < SrcTy.getNumElements()) {
        SmallVector<Register, 8> Regs;
        SPIRVType *ElementType =
            GR->getScalarOrVectorComponentType(GR->getSPIRVTypeForVReg(DstReg));
        LLT ElementLLTTy = GR->getRegType(ElementType);
        for (unsigned I = 0, E = SrcTy.getNumElements(); I < E; ++I) {
          Register Reg = MRI.createGenericVirtualRegister(ElementLLTTy);
          MRI.setRegClass(Reg, GR->getRegClass(ElementType));
          GR->assignSPIRVTypeToVReg(ElementType, Reg, *MI.getMF());
          Regs.push_back(Reg);
        }
        MIRBuilder.buildUnmerge(Regs, SrcReg);
        Regs[IdxVal] = ValReg;
        MIRBuilder.buildBuildVector(DstReg, Regs);
        MI.eraseFromParent();
        return true;
      }
    }

    LLT EltTy = SrcTy.getElementType();
    Align VecAlign;
    MachinePointerInfo PtrInfo;
    auto StackTemp = createStackTemporaryForVector(Helper, GR, SrcReg, SrcTy,
                                                   PtrInfo, VecAlign);

    MIRBuilder.buildStore(SrcReg, StackTemp, PtrInfo, VecAlign);

    Register IdxReg = IdxOperand.getReg();
    LLT PtrTy = MRI.getType(StackTemp.getReg(0));
    Register EltPtr = MRI.createGenericVirtualRegister(PtrTy);
    auto Zero = MIRBuilder.buildConstant(LLT::scalar(32), 0);

    MIRBuilder.buildIntrinsic(Intrinsic::spv_gep, ArrayRef<Register>{EltPtr})
        .addImm(1) // InBounds
        .addUse(StackTemp.getReg(0))
        .addUse(Zero.getReg(0))
        .addUse(IdxReg);

    MachinePointerInfo EltPtrInfo = MachinePointerInfo(PtrTy.getAddressSpace());
    Align EltAlign = Helper.getStackTemporaryAlignment(EltTy);
    MIRBuilder.buildStore(ValReg, EltPtr, EltPtrInfo, EltAlign);

    MIRBuilder.buildLoad(DstReg, StackTemp, PtrInfo, VecAlign);
    MI.eraseFromParent();
    return true;
  }
  return true;
}

static bool legalizeSpvExtractElt(LegalizerHelper &Helper, MachineInstr &MI,
                                  SPIRVGlobalRegistry *GR) {
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  MachineRegisterInfo &MRI = *MIRBuilder.getMRI();
  const SPIRVSubtarget &ST = MI.getMF()->getSubtarget<SPIRVSubtarget>();

  Register SrcReg = MI.getOperand(2).getReg();
  LLT SrcTy = MRI.getType(SrcReg);

  if (needsVectorLegalization(SrcTy, ST)) {
    Register DstReg = MI.getOperand(0).getReg();
    MachineOperand &IdxOperand = MI.getOperand(3);

    if (getImm(IdxOperand, &MRI)) {
      uint64_t IdxVal = foldImm(IdxOperand, &MRI);
      if (IdxVal < SrcTy.getNumElements()) {
        LLT DstTy = MRI.getType(DstReg);
        SmallVector<Register, 8> Regs;
        SPIRVType *DstSpvTy = GR->getSPIRVTypeForVReg(DstReg);
        for (unsigned I = 0, E = SrcTy.getNumElements(); I < E; ++I) {
          if (I == IdxVal) {
            Regs.push_back(DstReg);
          } else {
            Register Reg = MRI.createGenericVirtualRegister(DstTy);
            MRI.setRegClass(Reg, GR->getRegClass(DstSpvTy));
            GR->assignSPIRVTypeToVReg(DstSpvTy, Reg, *MI.getMF());
            Regs.push_back(Reg);
          }
        }
        MIRBuilder.buildUnmerge(Regs, SrcReg);
        MI.eraseFromParent();
        return true;
      }
    }

    LLT EltTy = SrcTy.getElementType();
    Align VecAlign;
    MachinePointerInfo PtrInfo;
    auto StackTemp = createStackTemporaryForVector(Helper, GR, SrcReg, SrcTy,
                                                   PtrInfo, VecAlign);

    MIRBuilder.buildStore(SrcReg, StackTemp, PtrInfo, VecAlign);

    Register IdxReg = IdxOperand.getReg();
    LLT PtrTy = MRI.getType(StackTemp.getReg(0));
    Register EltPtr = MRI.createGenericVirtualRegister(PtrTy);
    auto Zero = MIRBuilder.buildConstant(LLT::scalar(32), 0);

    MIRBuilder.buildIntrinsic(Intrinsic::spv_gep, ArrayRef<Register>{EltPtr})
        .addImm(1) // InBounds
        .addUse(StackTemp.getReg(0))
        .addUse(Zero.getReg(0))
        .addUse(IdxReg);

    MachinePointerInfo EltPtrInfo = MachinePointerInfo(PtrTy.getAddressSpace());
    Align EltAlign = Helper.getStackTemporaryAlignment(EltTy);
    MIRBuilder.buildLoad(DstReg, EltPtr, EltPtrInfo, EltAlign);

    MI.eraseFromParent();
    return true;
  }
  return true;
}

static bool legalizeSpvConstComposite(LegalizerHelper &Helper, MachineInstr &MI,
                                      SPIRVGlobalRegistry *GR) {
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  MachineRegisterInfo &MRI = *MIRBuilder.getMRI();
  const SPIRVSubtarget &ST = MI.getMF()->getSubtarget<SPIRVSubtarget>();

  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);

  if (!needsVectorLegalization(DstTy, ST))
    return true;

  SmallVector<Register, 8> SrcRegs;
  if (MI.getNumOperands() == 2) {
    // The "null" case: no values are attached.
    LLT EltTy = DstTy.getElementType();
    auto Zero = MIRBuilder.buildConstant(EltTy, 0);
    SPIRVType *SpvDstTy = GR->getSPIRVTypeForVReg(DstReg);
    SPIRVType *SpvEltTy = GR->getScalarOrVectorComponentType(SpvDstTy);
    GR->assignSPIRVTypeToVReg(SpvEltTy, Zero.getReg(0), MIRBuilder.getMF());
    for (unsigned i = 0; i < DstTy.getNumElements(); ++i)
      SrcRegs.push_back(Zero.getReg(0));
  } else {
    for (unsigned i = 2; i < MI.getNumOperands(); ++i) {
      SrcRegs.push_back(MI.getOperand(i).getReg());
    }
  }
  MIRBuilder.buildBuildVector(DstReg, SrcRegs);
  MI.eraseFromParent();
  return true;
}

bool SPIRVLegalizerInfo::legalizeIntrinsic(LegalizerHelper &Helper,
                                           MachineInstr &MI) const {
  LLVM_DEBUG(dbgs() << "legalizeIntrinsic: " << MI);
  auto IntrinsicID = cast<GIntrinsic>(MI).getIntrinsicID();
  switch (IntrinsicID) {
  case Intrinsic::spv_bitcast:
    return legalizeSpvBitcast(Helper, MI, GR);
  case Intrinsic::spv_insertelt:
    return legalizeSpvInsertElt(Helper, MI, GR);
  case Intrinsic::spv_extractelt:
    return legalizeSpvExtractElt(Helper, MI, GR);
  case Intrinsic::spv_const_composite:
    return legalizeSpvConstComposite(Helper, MI, GR);
  }
  return true;
}

bool SPIRVLegalizerInfo::legalizeBitcast(LegalizerHelper &Helper,
                                         MachineInstr &MI) const {
  // Once the G_BITCAST is using vectors that are allowed, we turn it back into
  // an spv_bitcast to avoid verifier problems when the register types are the
  // same for the source and the result. Note that the SPIR-V types associated
  // with the bitcast can be different even if the register types are the same.
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  SmallVector<Register, 1> DstRegs = {DstReg};
  MIRBuilder.buildIntrinsic(Intrinsic::spv_bitcast, DstRegs).addUse(SrcReg);
  MI.eraseFromParent();
  return true;
}

// Note this code was copied from LegalizerHelper::lowerISFPCLASS and adjusted
// to ensure that all instructions created during the lowering have SPIR-V types
// assigned to them.
bool SPIRVLegalizerInfo::legalizeIsFPClass(
    LegalizerHelper &Helper, MachineInstr &MI,
    LostDebugLocObserver &LocObserver) const {
  auto [DstReg, DstTy, SrcReg, SrcTy] = MI.getFirst2RegLLTs();
  FPClassTest Mask = static_cast<FPClassTest>(MI.getOperand(2).getImm());

  auto &MIRBuilder = Helper.MIRBuilder;
  auto &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  Type *LLVMDstTy =
      IntegerType::get(MIRBuilder.getContext(), DstTy.getScalarSizeInBits());
  if (DstTy.isVector())
    LLVMDstTy = VectorType::get(LLVMDstTy, DstTy.getElementCount());
  SPIRVType *SPIRVDstTy = GR->getOrCreateSPIRVType(
      LLVMDstTy, MIRBuilder, SPIRV::AccessQualifier::ReadWrite,
      /*EmitIR*/ true);

  unsigned BitSize = SrcTy.getScalarSizeInBits();
  const fltSemantics &Semantics = getFltSemanticForLLT(SrcTy.getScalarType());

  LLT IntTy = LLT::scalar(BitSize);
  Type *LLVMIntTy = IntegerType::get(MIRBuilder.getContext(), BitSize);
  if (SrcTy.isVector()) {
    IntTy = LLT::vector(SrcTy.getElementCount(), IntTy);
    LLVMIntTy = VectorType::get(LLVMIntTy, SrcTy.getElementCount());
  }
  SPIRVType *SPIRVIntTy = GR->getOrCreateSPIRVType(
      LLVMIntTy, MIRBuilder, SPIRV::AccessQualifier::ReadWrite,
      /*EmitIR*/ true);

  // Clang doesn't support capture of structured bindings:
  LLT DstTyCopy = DstTy;
  const auto assignSPIRVTy = [&](MachineInstrBuilder &&MI) {
    // Assign this MI's (assumed only) destination to one of the two types we
    // expect: either the G_IS_FPCLASS's destination type, or the integer type
    // bitcast from the source type.
    LLT MITy = MRI.getType(MI.getReg(0));
    assert((MITy == IntTy || MITy == DstTyCopy) &&
           "Unexpected LLT type while lowering G_IS_FPCLASS");
    auto *SPVTy = MITy == IntTy ? SPIRVIntTy : SPIRVDstTy;
    GR->assignSPIRVTypeToVReg(SPVTy, MI.getReg(0), MF);
    return MI;
  };

  // Helper to build and assign a constant in one go
  const auto buildSPIRVConstant = [&](LLT Ty, auto &&C) -> MachineInstrBuilder {
    if (!Ty.isFixedVector())
      return assignSPIRVTy(MIRBuilder.buildConstant(Ty, C));
    auto ScalarC = MIRBuilder.buildConstant(Ty.getScalarType(), C);
    assert((Ty == IntTy || Ty == DstTyCopy) &&
           "Unexpected LLT type while lowering constant for G_IS_FPCLASS");
    SPIRVType *VecEltTy = GR->getOrCreateSPIRVType(
        (Ty == IntTy ? LLVMIntTy : LLVMDstTy)->getScalarType(), MIRBuilder,
        SPIRV::AccessQualifier::ReadWrite,
        /*EmitIR*/ true);
    GR->assignSPIRVTypeToVReg(VecEltTy, ScalarC.getReg(0), MF);
    return assignSPIRVTy(MIRBuilder.buildSplatBuildVector(Ty, ScalarC));
  };

  if (Mask == fcNone) {
    MIRBuilder.buildCopy(DstReg, buildSPIRVConstant(DstTy, 0));
    MI.eraseFromParent();
    return true;
  }
  if (Mask == fcAllFlags) {
    MIRBuilder.buildCopy(DstReg, buildSPIRVConstant(DstTy, 1));
    MI.eraseFromParent();
    return true;
  }

  // Note that rather than creating a COPY here (between a floating-point and
  // integer type of the same size) we create a SPIR-V bitcast immediately. We
  // can't create a G_BITCAST because the LLTs are the same, and we can't seem
  // to correctly lower COPYs to SPIR-V bitcasts at this moment.
  Register ResVReg = MRI.createGenericVirtualRegister(IntTy);
  MRI.setRegClass(ResVReg, GR->getRegClass(SPIRVIntTy));
  GR->assignSPIRVTypeToVReg(SPIRVIntTy, ResVReg, Helper.MIRBuilder.getMF());
  auto AsInt = MIRBuilder.buildInstr(SPIRV::OpBitcast)
                   .addDef(ResVReg)
                   .addUse(GR->getSPIRVTypeID(SPIRVIntTy))
                   .addUse(SrcReg);
  AsInt = assignSPIRVTy(std::move(AsInt));

  // Various masks.
  APInt SignBit = APInt::getSignMask(BitSize);
  APInt ValueMask = APInt::getSignedMaxValue(BitSize);     // All bits but sign.
  APInt Inf = APFloat::getInf(Semantics).bitcastToAPInt(); // Exp and int bit.
  APInt ExpMask = Inf;
  APInt AllOneMantissa = APFloat::getLargest(Semantics).bitcastToAPInt() & ~Inf;
  APInt QNaNBitMask =
      APInt::getOneBitSet(BitSize, AllOneMantissa.getActiveBits() - 1);
  APInt InversionMask = APInt::getAllOnes(DstTy.getScalarSizeInBits());

  auto SignBitC = buildSPIRVConstant(IntTy, SignBit);
  auto ValueMaskC = buildSPIRVConstant(IntTy, ValueMask);
  auto InfC = buildSPIRVConstant(IntTy, Inf);
  auto ExpMaskC = buildSPIRVConstant(IntTy, ExpMask);
  auto ZeroC = buildSPIRVConstant(IntTy, 0);

  auto Abs = assignSPIRVTy(MIRBuilder.buildAnd(IntTy, AsInt, ValueMaskC));
  auto Sign = assignSPIRVTy(
      MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_NE, DstTy, AsInt, Abs));

  auto Res = buildSPIRVConstant(DstTy, 0);

  const auto appendToRes = [&](MachineInstrBuilder &&ToAppend) {
    Res = assignSPIRVTy(
        MIRBuilder.buildOr(DstTyCopy, Res, assignSPIRVTy(std::move(ToAppend))));
  };

  // Tests that involve more than one class should be processed first.
  if ((Mask & fcFinite) == fcFinite) {
    // finite(V) ==> abs(V) u< exp_mask
    appendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy, Abs,
                                     ExpMaskC));
    Mask &= ~fcFinite;
  } else if ((Mask & fcFinite) == fcPosFinite) {
    // finite(V) && V > 0 ==> V u< exp_mask
    appendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy, AsInt,
                                     ExpMaskC));
    Mask &= ~fcPosFinite;
  } else if ((Mask & fcFinite) == fcNegFinite) {
    // finite(V) && V < 0 ==> abs(V) u< exp_mask && signbit == 1
    auto Cmp = assignSPIRVTy(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT,
                                                  DstTy, Abs, ExpMaskC));
    appendToRes(MIRBuilder.buildAnd(DstTy, Cmp, Sign));
    Mask &= ~fcNegFinite;
  }

  if (FPClassTest PartialCheck = Mask & (fcZero | fcSubnormal)) {
    // fcZero | fcSubnormal => test all exponent bits are 0
    // TODO: Handle sign bit specific cases
    // TODO: Handle inverted case
    if (PartialCheck == (fcZero | fcSubnormal)) {
      auto ExpBits = assignSPIRVTy(MIRBuilder.buildAnd(IntTy, AsInt, ExpMaskC));
      appendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy,
                                       ExpBits, ZeroC));
      Mask &= ~PartialCheck;
    }
  }

  // Check for individual classes.
  if (FPClassTest PartialCheck = Mask & fcZero) {
    if (PartialCheck == fcPosZero)
      appendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy,
                                       AsInt, ZeroC));
    else if (PartialCheck == fcZero)
      appendToRes(
          MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy, Abs, ZeroC));
    else // fcNegZero
      appendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy,
                                       AsInt, SignBitC));
  }

  if (FPClassTest PartialCheck = Mask & fcSubnormal) {
    // issubnormal(V) ==> unsigned(abs(V) - 1) u< (all mantissa bits set)
    // issubnormal(V) && V>0 ==> unsigned(V - 1) u< (all mantissa bits set)
    auto V = (PartialCheck == fcPosSubnormal) ? AsInt : Abs;
    auto OneC = buildSPIRVConstant(IntTy, 1);
    auto VMinusOne = MIRBuilder.buildSub(IntTy, V, OneC);
    auto SubnormalRes = assignSPIRVTy(
        MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy, VMinusOne,
                             buildSPIRVConstant(IntTy, AllOneMantissa)));
    if (PartialCheck == fcNegSubnormal)
      SubnormalRes = MIRBuilder.buildAnd(DstTy, SubnormalRes, Sign);
    appendToRes(std::move(SubnormalRes));
  }

  if (FPClassTest PartialCheck = Mask & fcInf) {
    if (PartialCheck == fcPosInf)
      appendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy,
                                       AsInt, InfC));
    else if (PartialCheck == fcInf)
      appendToRes(
          MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy, Abs, InfC));
    else { // fcNegInf
      APInt NegInf = APFloat::getInf(Semantics, true).bitcastToAPInt();
      auto NegInfC = buildSPIRVConstant(IntTy, NegInf);
      appendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_EQ, DstTy,
                                       AsInt, NegInfC));
    }
  }

  if (FPClassTest PartialCheck = Mask & fcNan) {
    auto InfWithQnanBitC =
        buildSPIRVConstant(IntTy, std::move(Inf) | QNaNBitMask);
    if (PartialCheck == fcNan) {
      // isnan(V) ==> abs(V) u> int(inf)
      appendToRes(
          MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_UGT, DstTy, Abs, InfC));
    } else if (PartialCheck == fcQNan) {
      // isquiet(V) ==> abs(V) u>= (unsigned(Inf) | quiet_bit)
      appendToRes(MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_UGE, DstTy, Abs,
                                       InfWithQnanBitC));
    } else { // fcSNan
      // issignaling(V) ==> abs(V) u> unsigned(Inf) &&
      //                    abs(V) u< (unsigned(Inf) | quiet_bit)
      auto IsNan = assignSPIRVTy(
          MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_UGT, DstTy, Abs, InfC));
      auto IsNotQnan = assignSPIRVTy(MIRBuilder.buildICmp(
          CmpInst::Predicate::ICMP_ULT, DstTy, Abs, InfWithQnanBitC));
      appendToRes(MIRBuilder.buildAnd(DstTy, IsNan, IsNotQnan));
    }
  }

  if (FPClassTest PartialCheck = Mask & fcNormal) {
    // isnormal(V) ==> (0 u< exp u< max_exp) ==> (unsigned(exp-1) u<
    // (max_exp-1))
    APInt ExpLSB = ExpMask & ~(ExpMask.shl(1));
    auto ExpMinusOne = assignSPIRVTy(
        MIRBuilder.buildSub(IntTy, Abs, buildSPIRVConstant(IntTy, ExpLSB)));
    APInt MaxExpMinusOne = std::move(ExpMask) - ExpLSB;
    auto NormalRes = assignSPIRVTy(
        MIRBuilder.buildICmp(CmpInst::Predicate::ICMP_ULT, DstTy, ExpMinusOne,
                             buildSPIRVConstant(IntTy, MaxExpMinusOne)));
    if (PartialCheck == fcNegNormal)
      NormalRes = MIRBuilder.buildAnd(DstTy, NormalRes, Sign);
    else if (PartialCheck == fcPosNormal) {
      auto PosSign = assignSPIRVTy(MIRBuilder.buildXor(
          DstTy, Sign, buildSPIRVConstant(DstTy, InversionMask)));
      NormalRes = MIRBuilder.buildAnd(DstTy, NormalRes, PosSign);
    }
    appendToRes(std::move(NormalRes));
  }

  MIRBuilder.buildCopy(DstReg, Res);
  MI.eraseFromParent();
  return true;
}
