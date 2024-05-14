#ifndef LLVM_LIB_TARGET_INARCH_MCTARGETDESC_INARCHINFO_H
#define LLVM_LIB_TARGET_INARCH_MCTARGETDESC_INARCHINFO_H

#include "llvm/MC/MCInstrDesc.h"

namespace llvm {

namespace InArchCC {
enum CondCode {
  EQ,
  NE,
  LE,
  GT,
  LEU,
  GTU,
  INVALID,
};

CondCode getOppositeBranchCondition(CondCode);

enum BRCondCode {
  BREQ = 0x0,
};
} // end namespace InArchCC

namespace InArchOp {
enum OperandType : unsigned {
  OPERAND_SIMM16 = MCOI::OPERAND_FIRST_TARGET,
};
} // namespace SimOp

} // end namespace llvm

#endif