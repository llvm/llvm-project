//===--- ModRef.cpp - Memory effect modeling --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements ModRef and MemoryEffects misc functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ModRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

raw_ostream &llvm::operator<<(raw_ostream &OS, ModRefInfo MR) {
  switch (MR) {
  case ModRefInfo::NoModRef:
    OS << "NoModRef";
    break;
  case ModRefInfo::Ref:
    OS << "Ref";
    break;
  case ModRefInfo::Mod:
    OS << "Mod";
    break;
  case ModRefInfo::ModRef:
    OS << "ModRef";
    break;
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, MemoryEffects ME) {
  interleaveComma(MemoryEffects::locations(), OS, [&](IRMemLocation Loc) {
    switch (Loc) {
    case IRMemLocation::ArgMem:
      OS << "ArgMem: ";
      break;
    case IRMemLocation::InaccessibleMem:
      OS << "InaccessibleMem: ";
      break;
    case IRMemLocation::Other:
      OS << "Other: ";
      break;
    }
    OS << ME.getModRef(Loc);
  });
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, CaptureComponents CC) {
  if (capturesNothing(CC)) {
    OS << "none";
    return OS;
  }

  ListSeparator LS;
  if (capturesAddressIsNullOnly(CC))
    OS << LS << "address_is_null";
  else if (capturesAddress(CC))
    OS << LS << "address";
  if (capturesReadProvenanceOnly(CC))
    OS << LS << "read_provenance";
  if (capturesFullProvenance(CC))
    OS << LS << "provenance";

  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, CaptureInfo CI) {
  ListSeparator LS;
  CaptureComponents Other = CI.getOtherComponents();
  CaptureComponents Ret = CI.getRetComponents();

  OS << "captures(";
  if (!capturesNothing(Other) || Other == Ret)
    OS << LS << Other;
  if (Other != Ret)
    OS << LS << "ret: " << Ret;
  OS << ")";
  return OS;
}
