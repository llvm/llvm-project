//===-- TargetInstrInfo.cpp - Target Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  TargetInstrInfo
//
// Methods that depend on CodeGen are implemented in
// TargetInstrInfoImpl.cpp. Invoking them without linking libCodeGen raises a
// link error.
// ===----------------------------------------------------------------------===//

TargetInstrInfo::~TargetInstrInfo() {
}

const TargetRegisterClass*
TargetInstrInfo::getRegClass(const MCInstrDesc &MCID, unsigned OpNum,
                             const TargetRegisterInfo *TRI,
                             const MachineFunction &MF) const {
  if (OpNum >= MCID.getNumOperands())
    return 0;

  short RegClass = MCID.OpInfo[OpNum].RegClass;
  if (MCID.OpInfo[OpNum].isLookupPtrRegClass())
    return TRI->getPointerRegClass(MF, RegClass);

  // Instructions like INSERT_SUBREG do not have fixed register classes.
  if (RegClass < 0)
    return 0;

  // Otherwise just look it up normally.
  return TRI->getRegClass(RegClass);
}

/// insertNoop - Insert a noop into the instruction stream at the specified
/// point.
void TargetInstrInfo::insertNoop(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI) const {
  llvm_unreachable("Target didn't implement insertNoop!");
}

/// Measure the specified inline asm to determine an approximation of its
/// length.
/// Comments (which run till the next SeparatorString or newline) do not
/// count as an instruction.
/// Any other non-whitespace text is considered an instruction, with
/// multiple instructions separated by SeparatorString or newlines.
/// Variable-length instructions are not handled here; this function
/// may be overloaded in the target code to do that.
unsigned TargetInstrInfo::getInlineAsmLength(const char *Str,
                                             const MCAsmInfo &MAI) const {


  // Count the number of instructions in the asm.
  bool atInsnStart = true;
  unsigned Length = 0;
  for (; *Str; ++Str) {
    if (*Str == '\n' || strncmp(Str, MAI.getSeparatorString(),
                                strlen(MAI.getSeparatorString())) == 0)
      atInsnStart = true;
    if (atInsnStart && !std::isspace(*Str)) {
      Length += MAI.getMaxInstLength();
      atInsnStart = false;
    }
    if (atInsnStart && strncmp(Str, MAI.getCommentString(),
                               strlen(MAI.getCommentString())) == 0)
      atInsnStart = false;
  }

  return Length;
}
