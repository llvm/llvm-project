//===- AMDGPUBufferUtils.cpp - Buffer related helper functions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUBufferUtils.h"
#include "GCNSubtarget.h"
#include "SIDefines.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

bool AMDGPU::isRawBufferAccessDefinitelyOOB(const GCNSubtarget &ST, Value *Rsrc,
                                            Value *OffsetV, Value *SOffsetV,
                                            Value *AuxV, uint64_t Payload,
                                            const DataLayout &DL) {
  using OOBModel = GCNSubtarget::AMDGPUBufferOOBModel;
  std::optional<OOBModel> Model = ST.getBufferOOBModel();
  std::optional<unsigned> NumRecordsWidth =
      ST.getBufferResourceNumRecordsWidth();
  if (!Model || !NumRecordsWidth)
    return false;

  auto *AuxC = dyn_cast<ConstantInt>(AuxV);
  if (!AuxC)
    return false;

  uint64_t Aux = AuxC->getZExtValue();
  if (Aux & AMDGPU::CPol::VOLATILE)
    return false;

  // The resource descriptor must also be a compile-time constant.
  const APInt *StrideAP, *NumRecAP, *FlagsAP;
  if (!match(Rsrc, m_Intrinsic<Intrinsic::amdgcn_make_buffer_rsrc>(
                       m_Value(), m_APInt(StrideAP), m_APInt(NumRecAP),
                       m_APInt(FlagsAP))))
    return false;

  // Don't reason about a num_records that does not fit the hardware field.
  if (NumRecAP->getActiveBits() > *NumRecordsWidth)
    return false;

  uint64_t Offset = computeKnownBits(OffsetV, DL).getMinValue().getZExtValue();
  uint64_t SOffset =
      computeKnownBits(SOffsetV, DL).getMinValue().getZExtValue();

  uint64_t StrideRaw = StrideAP->getZExtValue();
  uint64_t Stride = StrideRaw & 0x3fff;
  uint64_t Flags = FlagsAP->getZExtValue();
  uint64_t NumRecordsMask = maskTrailingOnes<uint64_t>(*NumRecordsWidth);
  uint64_t NumRecords =
      NumRecAP->zextOrTrunc(64).getZExtValue() & NumRecordsMask;

  switch (*Model) {
  case OOBModel::NoSelect:
  case OOBModel::Select2Bit: {
    bool AddTid = (Flags >> 23) & 1;
    bool Swizzle =
        ST.isGFX11Plus() ? (StrideRaw >> 14) != 0 : (StrideRaw >> 15) != 0;
    if (AddTid || Swizzle)
      return false;

    // Saturating uint64 subtract: clamp at 0 if SOffset >= NumRecords.
    uint64_t Bound = NumRecords > SOffset ? NumRecords - SOffset : 0;
    if (*Model == OOBModel::Select2Bit) {
      unsigned OOBSelect = (Flags >> 28) & 3;
      return OOBSelect == 3 && Offset + Payload > Bound;
    }
    // Nonzero stride drops soffset out of the no select model equation.
    return Offset + Payload > (Stride == 0 ? Bound : NumRecords);
  }
  case OOBModel::Select1Bit: {
    bool IsBuffer = ((Flags >> 2) & 3) == 0;
    bool Swizzle = Flags & 1;
    bool StrideScaled = (StrideRaw >> 14) != 0;
    bool OOBCheckDisabled = NumRecords == NumRecordsMask || NumRecords == 0;
    if (!IsBuffer || Swizzle || StrideScaled || OOBCheckDisabled)
      return false;

    unsigned OOBSelect = (Flags >> 1) & 1;
    // Here SOffset is added directly into the offset used for bound check.
    uint64_t Addr = Offset + SOffset;
    return Addr + Payload > NumRecords ||
           (OOBSelect == 1 && Addr + Payload > Stride);
  }
  }
  return false;
}
