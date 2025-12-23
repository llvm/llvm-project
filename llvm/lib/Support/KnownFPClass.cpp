//===- llvm/Support/KnownFPClass.h - Stores known fplcass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a class for representing known fpclasses used by
// computeKnownFPClass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/KnownFPClass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

KnownFPClass::KnownFPClass(const APFloat &C)
    : KnownFPClasses(C.classify()), SignBit(C.isNegative()) {}

/// Return true if it's possible to assume IEEE treatment of input denormals in
/// \p F for \p Val.
static bool inputDenormalIsIEEE(DenormalMode Mode) {
  return Mode.Input == DenormalMode::IEEE;
}

static bool inputDenormalIsIEEEOrPosZero(DenormalMode Mode) {
  return Mode.Input == DenormalMode::IEEE ||
         Mode.Input == DenormalMode::PositiveZero;
}

bool KnownFPClass::isKnownNeverLogicalZero(DenormalMode Mode) const {
  return isKnownNeverZero() &&
         (isKnownNeverSubnormal() || inputDenormalIsIEEE(Mode));
}

bool KnownFPClass::isKnownNeverLogicalNegZero(DenormalMode Mode) const {
  return isKnownNeverNegZero() &&
         (isKnownNeverNegSubnormal() || inputDenormalIsIEEEOrPosZero(Mode));
}

bool KnownFPClass::isKnownNeverLogicalPosZero(DenormalMode Mode) const {
  if (!isKnownNeverPosZero())
    return false;

  // If we know there are no denormals, nothing can be flushed to zero.
  if (isKnownNeverSubnormal())
    return true;

  switch (Mode.Input) {
  case DenormalMode::IEEE:
    return true;
  case DenormalMode::PreserveSign:
    // Negative subnormal won't flush to +0
    return isKnownNeverPosSubnormal();
  case DenormalMode::PositiveZero:
  default:
    // Both positive and negative subnormal could flush to +0
    return false;
  }

  llvm_unreachable("covered switch over denormal mode");
}

void KnownFPClass::propagateDenormal(const KnownFPClass &Src,
                                     DenormalMode Mode) {
  KnownFPClasses = Src.KnownFPClasses;
  // If we aren't assuming the source can't be a zero, we don't have to check if
  // a denormal input could be flushed.
  if (!Src.isKnownNeverPosZero() && !Src.isKnownNeverNegZero())
    return;

  // If we know the input can't be a denormal, it can't be flushed to 0.
  if (Src.isKnownNeverSubnormal())
    return;

  if (!Src.isKnownNeverPosSubnormal() && Mode != DenormalMode::getIEEE())
    KnownFPClasses |= fcPosZero;

  if (!Src.isKnownNeverNegSubnormal() && Mode != DenormalMode::getIEEE()) {
    if (Mode != DenormalMode::getPositiveZero())
      KnownFPClasses |= fcNegZero;

    if (Mode.Input == DenormalMode::PositiveZero ||
        Mode.Output == DenormalMode::PositiveZero ||
        Mode.Input == DenormalMode::Dynamic ||
        Mode.Output == DenormalMode::Dynamic)
      KnownFPClasses |= fcPosZero;
  }
}

KnownFPClass KnownFPClass::canonicalize(const KnownFPClass &KnownSrc,
                                        DenormalMode DenormMode) {
  KnownFPClass Known;

  // This is essentially a stronger form of
  // propagateCanonicalizingSrc. Other "canonicalizing" operations don't
  // actually have an IR canonicalization guarantee.

  // Canonicalize may flush denormals to zero, so we have to consider the
  // denormal mode to preserve known-not-0 knowledge.
  Known.KnownFPClasses = KnownSrc.KnownFPClasses | fcZero | fcQNan;

  // Stronger version of propagateNaN
  // Canonicalize is guaranteed to quiet signaling nans.
  if (KnownSrc.isKnownNeverNaN())
    Known.knownNot(fcNan);
  else
    Known.knownNot(fcSNan);

  // FIXME: Missing check of IEEE like types.

  // If the parent function flushes denormals, the canonical output cannot be a
  // denormal.
  if (DenormMode == DenormalMode::getIEEE()) {
    if (KnownSrc.isKnownNever(fcPosZero))
      Known.knownNot(fcPosZero);
    if (KnownSrc.isKnownNever(fcNegZero))
      Known.knownNot(fcNegZero);
    return Known;
  }

  if (DenormMode.inputsAreZero() || DenormMode.outputsAreZero())
    Known.knownNot(fcSubnormal);

  if (DenormMode == DenormalMode::getPreserveSign()) {
    if (KnownSrc.isKnownNever(fcPosZero | fcPosSubnormal))
      Known.knownNot(fcPosZero);
    if (KnownSrc.isKnownNever(fcNegZero | fcNegSubnormal))
      Known.knownNot(fcNegZero);
    return Known;
  }

  if (DenormMode.Input == DenormalMode::PositiveZero ||
      (DenormMode.Output == DenormalMode::PositiveZero &&
       DenormMode.Input == DenormalMode::IEEE))
    Known.knownNot(fcNegZero);

  return Known;
}

KnownFPClass KnownFPClass::exp(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;
  Known.knownNot(fcNegative);

  if (KnownSrc.isKnownNeverNaN()) {
    Known.knownNot(fcNan);
    Known.signBitMustBeZero();
  }

  if (KnownSrc.cannotBeOrderedLessThanZero()) {
    // If the source is positive this cannot underflow.
    Known.knownNot(fcPosZero);

    // Cannot introduce denormal values.
    Known.knownNot(fcPosSubnormal);
  }

  // If the source is negative, this cannot overflow to infinity.
  if (KnownSrc.cannotBeOrderedGreaterThanZero())
    Known.knownNot(fcPosInf);

  return Known;
}

void KnownFPClass::propagateCanonicalizingSrc(const KnownFPClass &Src,
                                              DenormalMode Mode) {
  propagateDenormal(Src, Mode);
  propagateNaN(Src, /*PreserveSign=*/true);
}
