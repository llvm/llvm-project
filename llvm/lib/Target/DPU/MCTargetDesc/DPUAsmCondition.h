//===-- DPUAsmCondition.h - DPU Assembler Condition Representation ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DPUASMCONDITION_H
#define LLVM_DPUASMCONDITION_H

#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm {
namespace DPUAsmCondition {
enum Condition {
  Carry = 0,
  Equal = 1,
  Even = 2,
  ExtendedGreaterThanSigned = 3,
  ExtendedGreaterThanUnsigned = 4,
  ExtendedLessOrEqualSigned = 5,
  ExtendedLessOrEqualUnsigned = 6,
  ExtendedNotZero = 7,
  ExtendedZero = 8,
  False = 9,
  GreaterOrEqualSigned = 10,
  GreaterOrEqualUnsigned = 11,
  GreaterThanSigned = 12,
  GreaterThanUnsigned = 13,
  Large = 14,
  LessOrEqualSigned = 15,
  LessOrEqualUnsigned = 16,
  LessThanSigned = 17,
  LessThanUnsigned = 18,
  Maximum = 19,
  Negative = 20,
  NotCarry = 21,
  NotCarry10 = 22,
  NotCarry11 = 23,
  NotCarry12 = 24,
  NotCarry13 = 25,
  NotCarry14 = 26,
  NotCarry5 = 27,
  NotCarry6 = 28,
  NotCarry7 = 29,
  NotCarry8 = 30,
  NotCarry9 = 31,
  NotEqual = 32,
  NotMaximum = 33,
  NotOverflow = 34,
  NotShift32 = 35,
  NotZero = 36,
  Odd = 37,
  Overflow = 38,
  PositiveOrNull = 39,
  Shift32 = 40,
  Small = 41,
  SourceEven = 42,
  SourceNegative = 43,
  SourceNotZero = 44,
  SourceOdd = 45,
  SourcePositiveOrNull = 46,
  SourceZero = 47,
  True = 48,
  Zero = 49,
  NR_CONDITIONS = 50
};

enum ConditionClass {
  AcquireCC = 0,
  AddCC = 1,
  Add_nzCC = 2,
  BootCC = 3,
  ConstCC_ge0 = 4,
  ConstCC_geu = 5,
  ConstCC_zero = 6,
  CountCC = 7,
  Count_nzCC = 8,
  DivCC = 9,
  Div_nzCC = 10,
  Ext_sub_setCC = 11,
  FalseCC = 12,
  Imm_shiftCC = 13,
  Imm_shift_nzCC = 14,
  LogCC = 15,
  Log_nzCC = 16,
  Log_setCC = 17,
  MulCC = 18,
  Mul_nzCC = 19,
  NoCC = 20,
  ReleaseCC = 21,
  ShiftCC = 22,
  Shift_nzCC = 23,
  SubCC = 24,
  Sub_nzCC = 25,
  Sub_setCC = 26,
  TrueCC = 27,
  True_falseCC = 28,
  NR_CONDITION_CLASSES = 29
};

bool fromString(const std::string &string, Condition &Cond);

StringRef toString(Condition Cond);

bool isInConditionClass(Condition Cond, ConditionClass CondClass);

int64_t getEncoding(Condition Cond, ConditionClass CondClass);

int64_t getDecoding(uint64_t Cond, ConditionClass CondClass);

ConditionClass findConditionClassForInstruction(unsigned InstOpcode);

const unsigned int nrEncodingValue = 64;
} // namespace DPUAsmCondition
} // namespace llvm

#endif // LLVM_DPUASMCONDITION_H
