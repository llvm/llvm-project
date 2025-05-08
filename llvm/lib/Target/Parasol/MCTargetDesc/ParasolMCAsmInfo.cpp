//===-- ParasolMCAsmInfo.cpp - Parasol Asm Properties ---------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the ParasolMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "ParasolMCAsmInfo.h"

#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void ParasolMCAsmInfo::anchor() {}

ParasolMCAsmInfo::ParasolMCAsmInfo(const Triple &TheTriple) {
  // This architecture is little endian only
  IsLittleEndian = true;

  AlignmentIsInBytes = false;
  Data16bitsDirective = "\t.hword\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = "\t.dword\t";

  PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = ".L";

  LabelSuffix = ":";

  CommentString = ";";

  ZeroDirective = "\t.zero\t";

  UseAssignmentForEHBegin = true;

  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
  DwarfRegNumForCFI = true;
}
