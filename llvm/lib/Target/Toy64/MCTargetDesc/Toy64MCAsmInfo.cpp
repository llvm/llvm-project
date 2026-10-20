//===-- Toy64MCAsmInfo.cpp - Toy64 Assembly Properties --------------------===//

#include "Toy64MCAsmInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void Toy64MCAsmInfo::anchor() {}

Toy64MCAsmInfo::Toy64MCAsmInfo(const Triple &TT,
                               const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
  CodePointerSize = 8;
  CalleeSaveStackSlotSize = 8;

  IsLittleEndian = true;

  CommentString = "#";
}