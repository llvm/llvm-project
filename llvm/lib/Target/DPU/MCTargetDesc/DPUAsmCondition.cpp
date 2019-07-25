//===-- DPUAsmCondition.cpp - DPU Assembler Condition Representation ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DPUAsmCondition.h"
#include "assert.h"
#include <set>

#define GET_INSTRINFO_ENUM
#include "DPUGenInstrInfo.inc"

namespace llvm {
namespace DPUAsmCondition {
const std::string ConditionStrings[] = {
    {"c"},     {"eq"},   {"e"},     {"xgts"}, {"xgtu"}, {"xles"}, {"xleu"},
    {"xnz"},   {"xz"},   {"false"}, {"ges"},  {"geu"},  {"gts"},  {"gtu"},
    {"large"}, {"les"},  {"leu"},   {"lts"},  {"ltu"},  {"max"},  {"mi"},
    {"nc"},    {"nc10"}, {"nc11"},  {"nc12"}, {"nc13"}, {"nc14"}, {"nc5"},
    {"nc6"},   {"nc7"},  {"nc8"},   {"nc9"},  {"neq"},  {"nmax"}, {"nov"},
    {"nsh32"}, {"nz"},   {"o"},     {"ov"},   {"pl"},   {"sh32"}, {"small"},
    {"se"},    {"smi"},  {"snz"},   {"so"},   {"spl"},  {"sz"},   {"true"},
    {"z"},
};

const std::set<Condition> ConditionClassSets[] = {
    {Condition::NotZero, Condition::True, Condition::Zero},
    {Condition::Carry,
     Condition::ExtendedNotZero,
     Condition::ExtendedZero,
     Condition::False,
     Condition::Negative,
     Condition::NotCarry,
     Condition::NotCarry10,
     Condition::NotCarry11,
     Condition::NotCarry12,
     Condition::NotCarry13,
     Condition::NotCarry14,
     Condition::NotCarry5,
     Condition::NotCarry6,
     Condition::NotCarry7,
     Condition::NotCarry8,
     Condition::NotCarry9,
     Condition::NotOverflow,
     Condition::NotZero,
     Condition::Overflow,
     Condition::PositiveOrNull,
     Condition::SourceNegative,
     Condition::SourceNotZero,
     Condition::SourcePositiveOrNull,
     Condition::SourceZero,
     Condition::True,
     Condition::Zero},
    {Condition::Carry,
     Condition::ExtendedNotZero,
     Condition::ExtendedZero,
     Condition::Negative,
     Condition::NotCarry,
     Condition::NotCarry10,
     Condition::NotCarry11,
     Condition::NotCarry12,
     Condition::NotCarry13,
     Condition::NotCarry14,
     Condition::NotCarry5,
     Condition::NotCarry6,
     Condition::NotCarry7,
     Condition::NotCarry8,
     Condition::NotCarry9,
     Condition::NotOverflow,
     Condition::NotZero,
     Condition::Overflow,
     Condition::PositiveOrNull,
     Condition::SourceNegative,
     Condition::SourceNotZero,
     Condition::SourcePositiveOrNull,
     Condition::SourceZero,
     Condition::True,
     Condition::Zero},
    {Condition::ExtendedNotZero, Condition::ExtendedZero, Condition::False,
     Condition::NotZero, Condition::SourceNegative, Condition::SourceNotZero,
     Condition::SourcePositiveOrNull, Condition::SourceZero, Condition::True,
     Condition::Zero},
    {Condition::PositiveOrNull},
    {Condition::GreaterOrEqualUnsigned},
    {Condition::Zero},
    {Condition::ExtendedNotZero, Condition::ExtendedZero, Condition::False,
     Condition::Maximum, Condition::NotMaximum, Condition::NotZero,
     Condition::SourceNegative, Condition::SourceNotZero,
     Condition::SourcePositiveOrNull, Condition::SourceZero, Condition::True,
     Condition::Zero},
    {Condition::ExtendedNotZero, Condition::ExtendedZero, Condition::Maximum,
     Condition::NotMaximum, Condition::NotZero, Condition::SourceNegative,
     Condition::SourceNotZero, Condition::SourcePositiveOrNull,
     Condition::SourceZero, Condition::True, Condition::Zero},
    {Condition::False, Condition::SourceNegative, Condition::SourceNotZero,
     Condition::SourcePositiveOrNull, Condition::SourceZero, Condition::True},
    {Condition::SourceNegative, Condition::SourceNotZero,
     Condition::SourcePositiveOrNull, Condition::SourceZero, Condition::True},
    {Condition::Carry,
     Condition::Equal,
     Condition::ExtendedGreaterThanSigned,
     Condition::ExtendedGreaterThanUnsigned,
     Condition::ExtendedLessOrEqualSigned,
     Condition::ExtendedLessOrEqualUnsigned,
     Condition::ExtendedNotZero,
     Condition::ExtendedZero,
     Condition::GreaterOrEqualSigned,
     Condition::GreaterOrEqualUnsigned,
     Condition::GreaterThanSigned,
     Condition::GreaterThanUnsigned,
     Condition::LessOrEqualSigned,
     Condition::LessOrEqualUnsigned,
     Condition::LessThanSigned,
     Condition::LessThanUnsigned,
     Condition::Negative,
     Condition::NotCarry,
     Condition::NotEqual,
     Condition::NotOverflow,
     Condition::NotZero,
     Condition::Overflow,
     Condition::PositiveOrNull,
     Condition::SourceNegative,
     Condition::SourceNotZero,
     Condition::SourcePositiveOrNull,
     Condition::SourceZero,
     Condition::True,
     Condition::Zero},
    {Condition::False},
    {Condition::Even, Condition::ExtendedNotZero, Condition::ExtendedZero,
     Condition::False, Condition::Negative, Condition::NotZero, Condition::Odd,
     Condition::PositiveOrNull, Condition::SourceEven,
     Condition::SourceNegative, Condition::SourceNotZero, Condition::SourceOdd,
     Condition::SourcePositiveOrNull, Condition::SourceZero, Condition::True,
     Condition::Zero},
    {Condition::Even, Condition::ExtendedNotZero, Condition::ExtendedZero,
     Condition::Negative, Condition::NotZero, Condition::Odd,
     Condition::PositiveOrNull, Condition::SourceEven,
     Condition::SourceNegative, Condition::SourceNotZero, Condition::SourceOdd,
     Condition::SourcePositiveOrNull, Condition::SourceZero, Condition::True,
     Condition::Zero},
    {Condition::ExtendedNotZero, Condition::ExtendedZero, Condition::False,
     Condition::Negative, Condition::NotZero, Condition::PositiveOrNull,
     Condition::SourceNegative, Condition::SourceNotZero,
     Condition::SourcePositiveOrNull, Condition::SourceZero, Condition::True,
     Condition::Zero},
    {Condition::ExtendedNotZero, Condition::ExtendedZero, Condition::Negative,
     Condition::NotZero, Condition::PositiveOrNull, Condition::SourceNegative,
     Condition::SourceNotZero, Condition::SourcePositiveOrNull,
     Condition::SourceZero, Condition::True, Condition::Zero},
    {Condition::ExtendedNotZero, Condition::ExtendedZero, Condition::NotZero,
     Condition::Zero},
    {Condition::ExtendedNotZero, Condition::ExtendedZero, Condition::False,
     Condition::Large, Condition::NotZero, Condition::Small,
     Condition::SourceNegative, Condition::SourceNotZero,
     Condition::SourcePositiveOrNull, Condition::SourceZero, Condition::True,
     Condition::Zero},
    {Condition::ExtendedNotZero, Condition::ExtendedZero, Condition::Large,
     Condition::NotZero, Condition::Small, Condition::SourceNegative,
     Condition::SourceNotZero, Condition::SourcePositiveOrNull,
     Condition::SourceZero, Condition::True, Condition::Zero},
    {},
    {Condition::NotZero},
    {Condition::Even, Condition::ExtendedNotZero, Condition::ExtendedZero,
     Condition::False, Condition::Negative, Condition::NotShift32,
     Condition::NotZero, Condition::Odd, Condition::PositiveOrNull,
     Condition::Shift32, Condition::SourceEven, Condition::SourceNegative,
     Condition::SourceNotZero, Condition::SourceOdd,
     Condition::SourcePositiveOrNull, Condition::SourceZero, Condition::True,
     Condition::Zero},
    {Condition::Even, Condition::ExtendedNotZero, Condition::ExtendedZero,
     Condition::Negative, Condition::NotShift32, Condition::NotZero,
     Condition::Odd, Condition::PositiveOrNull, Condition::Shift32,
     Condition::SourceEven, Condition::SourceNegative, Condition::SourceNotZero,
     Condition::SourceOdd, Condition::SourcePositiveOrNull,
     Condition::SourceZero, Condition::True, Condition::Zero},
    {Condition::Carry,
     Condition::Equal,
     Condition::ExtendedGreaterThanSigned,
     Condition::ExtendedGreaterThanUnsigned,
     Condition::ExtendedLessOrEqualSigned,
     Condition::ExtendedLessOrEqualUnsigned,
     Condition::ExtendedNotZero,
     Condition::ExtendedZero,
     Condition::False,
     Condition::GreaterOrEqualSigned,
     Condition::GreaterOrEqualUnsigned,
     Condition::GreaterThanSigned,
     Condition::GreaterThanUnsigned,
     Condition::LessOrEqualSigned,
     Condition::LessOrEqualUnsigned,
     Condition::LessThanSigned,
     Condition::LessThanUnsigned,
     Condition::Negative,
     Condition::NotCarry,
     Condition::NotEqual,
     Condition::NotOverflow,
     Condition::NotZero,
     Condition::Overflow,
     Condition::PositiveOrNull,
     Condition::SourceNegative,
     Condition::SourceNotZero,
     Condition::SourcePositiveOrNull,
     Condition::SourceZero,
     Condition::True,
     Condition::Zero},
    {Condition::Carry,
     Condition::Equal,
     Condition::ExtendedGreaterThanSigned,
     Condition::ExtendedGreaterThanUnsigned,
     Condition::ExtendedLessOrEqualSigned,
     Condition::ExtendedLessOrEqualUnsigned,
     Condition::ExtendedNotZero,
     Condition::ExtendedZero,
     Condition::GreaterOrEqualSigned,
     Condition::GreaterOrEqualUnsigned,
     Condition::GreaterThanSigned,
     Condition::GreaterThanUnsigned,
     Condition::LessOrEqualSigned,
     Condition::LessOrEqualUnsigned,
     Condition::LessThanSigned,
     Condition::LessThanUnsigned,
     Condition::Negative,
     Condition::NotCarry,
     Condition::NotEqual,
     Condition::NotOverflow,
     Condition::NotZero,
     Condition::Overflow,
     Condition::PositiveOrNull,
     Condition::SourceNegative,
     Condition::SourceNotZero,
     Condition::SourcePositiveOrNull,
     Condition::SourceZero,
     Condition::True,
     Condition::Zero},
    {Condition::Equal, Condition::ExtendedNotZero, Condition::ExtendedZero,
     Condition::NotEqual, Condition::NotZero, Condition::Zero},
    {Condition::True},
    {Condition::False, Condition::True},
};

const int64_t ConditionEncodings[NR_CONDITION_CLASSES][NR_CONDITIONS] = {
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,
    },
    {
        20, 0, 0, 0, 0,  0,  0,  5,  4,  0,  0,  0,  0,  0,  0,  0, 0,
        0,  0, 0, 9, 21, 27, 28, 29, 30, 31, 22, 23, 24, 25, 26, 0, 0,
        19, 0, 3, 0, 18, 8,  0,  0,  0,  15, 13, 0,  14, 12, 1,  2,
    },
    {
        20, 0, 0, 0, 0,  0,  0,  5,  4,  0,  0,  0,  0,  0,  0,  0, 0,
        0,  0, 0, 9, 21, 27, 28, 29, 30, 31, 22, 23, 24, 25, 26, 0, 0,
        19, 0, 3, 0, 18, 8,  0,  0,  0,  15, 13, 0,  14, 12, 1,  2,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 5, 4, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 3, 0, 0, 0, 0, 0, 0, 15, 13, 0, 14, 12, 1, 2,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 5, 4, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 8, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0, 9,
        0, 0, 3, 0, 0, 0, 0, 0, 0, 15, 13, 0, 14, 12, 1, 2,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 5, 4, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 8, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0, 9,
        0, 0, 3, 0, 0, 0, 0, 0, 0, 15, 13, 0, 14, 12, 1, 2,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 0, 14, 12, 1, 0,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 0, 14, 12, 1, 0,
    },
    {
        53, 6,  0, 61, 63, 60, 62, 11, 10, 0,  55, 53, 57, 59, 0,  56, 58,
        54, 52, 0, 41, 52, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  0,
        51, 0,  7, 0,  50, 40, 0,  0,  0,  47, 45, 0,  46, 44, 33, 6,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    },
    {
        0, 0, 24, 0,  0, 0, 0, 5, 4,  0,  0,  0,  0,  0,  0, 0, 0,
        0, 0, 0,  9,  0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,
        0, 0, 3,  25, 0, 8, 0, 0, 30, 15, 13, 31, 14, 12, 1, 2,
    },
    {
        0, 0, 24, 0,  0, 0, 0, 5, 4,  0,  0,  0,  0,  0,  0, 0, 0,
        0, 0, 0,  9,  0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,
        0, 0, 3,  25, 0, 8, 0, 0, 30, 15, 13, 31, 14, 12, 1, 2,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 5, 4, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 0, 9, 0, 0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 3, 0, 0, 8, 0, 0, 0, 15, 13, 0, 14, 12, 1, 2,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 5, 4, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 0, 9, 0, 0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0, 0,
        0, 0, 3, 0, 0, 8, 0, 0, 0, 15, 13, 0, 14, 12, 1, 2,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 11, 10, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 7, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 6,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 5,  4, 0,  0,  0, 0,  0,  31, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0, 0,  0,  0,  0, 0,
        0, 0, 3, 0, 0, 0, 0, 30, 0, 15, 13, 0, 14, 12, 1,  2,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 5,  4, 0,  0,  0, 0,  0,  31, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0, 0,  0,  0,  0, 0,
        0, 0, 3, 0, 0, 0, 0, 30, 0, 15, 13, 0, 14, 12, 1,  2,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    },
    {
        0, 0,  24, 0,  0, 0, 0,  5, 4,  0,  0,  0,  0,  0,  0, 0, 0,
        0, 0,  0,  9,  0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0, 0, 0,
        0, 28, 3,  25, 0, 8, 29, 0, 30, 15, 13, 31, 14, 12, 1, 2,
    },
    {
        0, 0,  24, 0,  0, 0, 0,  5, 4,  0,  0,  0,  0,  0,  0, 0, 0,
        0, 0,  0,  9,  0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0, 0, 0,
        0, 28, 3,  25, 0, 8, 29, 0, 30, 15, 13, 31, 14, 12, 1, 2,
    },
    {
        21, 2,  0, 29, 31, 28, 30, 5, 4, 0,  23, 21, 25, 27, 0, 24, 26,
        22, 20, 0, 9,  20, 0,  0,  0, 0, 0,  0,  0,  0,  0,  0, 3,  0,
        19, 0,  3, 0,  18, 8,  0,  0, 0, 15, 13, 0,  14, 12, 1, 2,
    },
    {
        21, 2,  0, 29, 31, 28, 30, 5, 4, 0,  23, 21, 25, 27, 0, 24, 26,
        22, 20, 0, 9,  20, 0,  0,  0, 0, 0,  0,  0,  0,  0,  0, 3,  0,
        19, 0,  3, 0,  18, 8,  0,  0, 0, 15, 13, 0,  14, 12, 1, 2,
    },
    {
        0, 6, 0, 0, 0, 0, 0, 11, 10, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 7, 0,
        0, 0, 7, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 6,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    },
};
const int64_t
    ConditionDecodings[NR_CONDITION_CLASSES][DPUAsmCondition::nrEncodingValue] =
        {
            {Condition::NR_CONDITIONS, Condition::True,
             Condition::Zero,          Condition::NotZero,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS},
            {Condition::False,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Overflow,
             Condition::NotOverflow,
             Condition::Carry,
             Condition::NotCarry,
             Condition::NotCarry5,
             Condition::NotCarry6,
             Condition::NotCarry7,
             Condition::NotCarry8,
             Condition::NotCarry9,
             Condition::NotCarry10,
             Condition::NotCarry11,
             Condition::NotCarry12,
             Condition::NotCarry13,
             Condition::NotCarry14,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Overflow,
             Condition::NotOverflow,
             Condition::Carry,
             Condition::NotCarry,
             Condition::NotCarry5,
             Condition::NotCarry6,
             Condition::NotCarry7,
             Condition::NotCarry8,
             Condition::NotCarry9,
             Condition::NotCarry10,
             Condition::NotCarry11,
             Condition::NotCarry12,
             Condition::NotCarry13,
             Condition::NotCarry14,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::False,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::PositiveOrNull, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,  Condition::NR_CONDITIONS},
            {Condition::GreaterOrEqualUnsigned, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,          Condition::NR_CONDITIONS},
            {Condition::Zero,          Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS},
            {Condition::False,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Maximum,
             Condition::NotMaximum,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Maximum,
             Condition::NotMaximum,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::False,
             Condition::True,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS,        Condition::True,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::SourceZero,           Condition::SourceNotZero,
             Condition::SourcePositiveOrNull, Condition::SourceNegative,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,        Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Zero,
             Condition::NotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Overflow,
             Condition::NotOverflow,
             Condition::LessThanUnsigned,
             Condition::GreaterOrEqualUnsigned,
             Condition::LessThanSigned,
             Condition::GreaterOrEqualSigned,
             Condition::LessOrEqualSigned,
             Condition::GreaterThanSigned,
             Condition::LessOrEqualUnsigned,
             Condition::GreaterThanUnsigned,
             Condition::ExtendedLessOrEqualSigned,
             Condition::ExtendedGreaterThanSigned,
             Condition::ExtendedLessOrEqualUnsigned,
             Condition::ExtendedGreaterThanUnsigned},
            {Condition::False,         Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS},
            {Condition::False,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Even,
             Condition::Odd,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceEven,
             Condition::SourceOdd,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Even,
             Condition::Odd,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceEven,
             Condition::SourceOdd,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::False,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::Zero,          Condition::NotZero,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::ExtendedZero,  Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS},
            {Condition::False,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Small,
             Condition::Large,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Small,
             Condition::Large,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS},
            {Condition::NotZero,       Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS},
            {Condition::False,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Even,
             Condition::Odd,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NotShift32,
             Condition::Shift32,
             Condition::SourceEven,
             Condition::SourceOdd,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Even,
             Condition::Odd,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NotShift32,
             Condition::Shift32,
             Condition::SourceEven,
             Condition::SourceOdd,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::False,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Overflow,
             Condition::NotOverflow,
             Condition::LessThanUnsigned,
             Condition::GreaterOrEqualUnsigned,
             Condition::LessThanSigned,
             Condition::GreaterOrEqualSigned,
             Condition::LessOrEqualSigned,
             Condition::GreaterThanSigned,
             Condition::LessOrEqualUnsigned,
             Condition::GreaterThanUnsigned,
             Condition::ExtendedLessOrEqualSigned,
             Condition::ExtendedGreaterThanSigned,
             Condition::ExtendedLessOrEqualUnsigned,
             Condition::ExtendedGreaterThanUnsigned,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS,
             Condition::True,
             Condition::Zero,
             Condition::NotZero,
             Condition::ExtendedZero,
             Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::PositiveOrNull,
             Condition::Negative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::SourceZero,
             Condition::SourceNotZero,
             Condition::SourcePositiveOrNull,
             Condition::SourceNegative,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::Overflow,
             Condition::NotOverflow,
             Condition::LessThanUnsigned,
             Condition::GreaterOrEqualUnsigned,
             Condition::LessThanSigned,
             Condition::GreaterOrEqualSigned,
             Condition::LessOrEqualSigned,
             Condition::GreaterThanSigned,
             Condition::LessOrEqualUnsigned,
             Condition::GreaterThanUnsigned,
             Condition::ExtendedLessOrEqualSigned,
             Condition::ExtendedGreaterThanSigned,
             Condition::ExtendedLessOrEqualUnsigned,
             Condition::ExtendedGreaterThanUnsigned,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::Zero,          Condition::NotZero,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::ExtendedZero,  Condition::ExtendedNotZero,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS},
            {Condition::NR_CONDITIONS, Condition::True,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS},
            {Condition::False,         Condition::True,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS,
             Condition::NR_CONDITIONS, Condition::NR_CONDITIONS},
};

bool fromString(const std::string &string, Condition &Cond) {
  const std::string *cond = std::find(std::begin(ConditionStrings),
                                      std::end(ConditionStrings), string);

  if (cond == std::end(ConditionStrings)) {
    return true;
  }

  Cond = static_cast<Condition>(std::distance(ConditionStrings, cond));

  return false;
}

StringRef toString(Condition Cond) {
  assert((Cond < Condition::NR_CONDITIONS) && "invalid condition value");
  return StringRef(ConditionStrings[Cond]);
}

bool isInConditionClass(Condition Cond, ConditionClass CondClass) {
  assert((CondClass < ConditionClass::NR_CONDITION_CLASSES) &&
         "invalid condition class value");
  std::set<Condition> Conditions = ConditionClassSets[CondClass];
  return Conditions.find(Cond) != Conditions.end();
}

int64_t getEncoding(Condition Cond, ConditionClass CondClass) {
  assert((CondClass < ConditionClass::NR_CONDITION_CLASSES) &&
         "invalid condition class value");
  assert((Cond < Condition::NR_CONDITIONS) && "invalid condition value");

  return ConditionEncodings[CondClass][Cond];
}

int64_t getDecoding(uint64_t Cond, ConditionClass CondClass) {
  assert((CondClass < ConditionClass::NR_CONDITION_CLASSES) &&
         "invalid condition class value");
  assert((Cond < DPUAsmCondition::nrEncodingValue) &&
         "invalid condition value");

  return ConditionDecodings[CondClass][Cond];
}

ConditionClass findConditionClassForInstruction(unsigned InstOpcode) {
  switch (InstOpcode) {
  default:
    llvm_unreachable("unknown instruction");
  case DPU::SUBC_Urrrc:
    return ConditionClass::Ext_sub_setCC;
  case DPU::NANDrric:
    return ConditionClass::Log_setCC;
  case DPU::ANDzrif:
    return ConditionClass::FalseCC;
  case DPU::XORzrrci:
    return ConditionClass::Log_nzCC;
  case DPU::MUL_SH_SLzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::OR_Srric:
    return ConditionClass::Log_setCC;
  case DPU::NAND_Srric:
    return ConditionClass::Log_setCC;
  case DPU::SUB_Srirci:
    return ConditionClass::Sub_nzCC;
  case DPU::AND_Urrrci:
    return ConditionClass::Log_nzCC;
  case DPU::ANDN_Srrici:
    return ConditionClass::Log_nzCC;
  case DPU::LSR1Xrrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::ORNrrif:
    return ConditionClass::FalseCC;
  case DPU::CMPB4zrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UL_UH_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSR1zrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::SUBCrirci:
    return ConditionClass::Sub_nzCC;
  case DPU::NXORrrif:
    return ConditionClass::FalseCC;
  case DPU::MUL_SH_ULrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::NORzric:
    return ConditionClass::Log_setCC;
  case DPU::NAND_Srrici:
    return ConditionClass::Log_nzCC;
  case DPU::LSLXrrrc:
    return ConditionClass::Log_setCC;
  case DPU::CLOrrc:
    return ConditionClass::Log_setCC;
  case DPU::CLO_Srrci:
    return ConditionClass::Count_nzCC;
  case DPU::ROLzrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SH_ULzrrc:
    return ConditionClass::Log_setCC;
  case DPU::NOR_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSR1Xzric:
    return ConditionClass::Log_setCC;
  case DPU::CLSrrci:
    return ConditionClass::Count_nzCC;
  case DPU::LSRzrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ASRrrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSLzrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::AND_Srrici:
    return ConditionClass::Log_nzCC;
  case DPU::SUB_Srirc:
    return ConditionClass::Sub_setCC;
  case DPU::ORN_Srrici:
    return ConditionClass::Log_nzCC;
  case DPU::CLO_Urrc:
    return ConditionClass::Log_setCC;
  case DPU::LSLXzrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::AND_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::ADDCzrrc:
    return ConditionClass::Log_setCC;
  case DPU::NXORzrrc:
    return ConditionClass::Log_setCC;
  case DPU::NOR_Urrrci:
    return ConditionClass::Log_nzCC;
  case DPU::ROLrric:
    return ConditionClass::Log_setCC;
  case DPU::LSR1_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSL1rrrc:
    return ConditionClass::Log_setCC;
  case DPU::SATS_Urrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSL1rric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SH_UH_Srrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::ROR_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::CLOzrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_SHrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::SUBrrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::ROL_Srric:
    return ConditionClass::Log_setCC;
  case DPU::CMPB4_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::CLO_Urrci:
    return ConditionClass::Count_nzCC;
  case DPU::SUBrirci:
    return ConditionClass::Sub_nzCC;
  case DPU::CLZ_Urrc:
    return ConditionClass::Log_setCC;
  case DPU::CLZrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORNrrrc:
    return ConditionClass::Log_setCC;
  case DPU::HASH_Srrrci:
    return ConditionClass::Log_nzCC;
  case DPU::NANDrrrc:
    return ConditionClass::Log_setCC;
  case DPU::CAO_Srrc:
    return ConditionClass::Log_setCC;
  case DPU::ASR_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSR_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSR1rrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ROLrrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::CLZ_Srrc:
    return ConditionClass::Log_setCC;
  case DPU::ADD_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORNrrrci:
    return ConditionClass::Log_nzCC;
  case DPU::RSUBC_Srrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::LSL1zrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::ANDN_Srric:
    return ConditionClass::Log_setCC;
  case DPU::NORrrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSL_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::EXTUHzrc:
    return ConditionClass::Log_setCC;
  case DPU::EXTSHzrci:
    return ConditionClass::Log_nzCC;
  case DPU::HASH_Urrif:
    return ConditionClass::FalseCC;
  case DPU::MUL_UH_ULzrrc:
    return ConditionClass::Log_setCC;
  case DPU::ASR_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ADDC_Urrici:
    return ConditionClass::Add_nzCC;
  case DPU::CLR_RUNrici:
    return ConditionClass::BootCC;
  case DPU::ORrrici:
    return ConditionClass::Log_nzCC;
  case DPU::ADDzric:
    return ConditionClass::Log_setCC;
  case DPU::LSLX_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::CMPB4rrrci:
    return ConditionClass::Log_nzCC;
  case DPU::EXTSB_Srrc:
    return ConditionClass::Log_setCC;
  case DPU::NANDzrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSL1_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::MUL_SH_SHrrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_SH_Srrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::XORzric:
    return ConditionClass::Log_setCC;
  case DPU::RSUB_Srrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::NAND_Srrrci:
    return ConditionClass::Log_nzCC;
  case DPU::ROR_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::ORNzrici:
    return ConditionClass::Log_nzCC;
  case DPU::TIME_CFG_Srrci:
    return ConditionClass::TrueCC;
  case DPU::EXTSHrrci:
    return ConditionClass::Log_nzCC;
  case DPU::NXOR_Srrici:
    return ConditionClass::Log_nzCC;
  case DPU::MUL_SL_SHrrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UL_UL_Urrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::ANDzrici:
    return ConditionClass::Log_nzCC;
  case DPU::ASR_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSRX_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ASRzrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::MUL_UL_UHzrrc:
    return ConditionClass::Log_setCC;
  case DPU::SUBC_Urirf:
    return ConditionClass::FalseCC;
  case DPU::EXTUBrrci:
    return ConditionClass::Log_nzCC;
  case DPU::SUBC_Srirci:
    return ConditionClass::Sub_nzCC;
  case DPU::ROL_ADD_Urrrici:
    return ConditionClass::Div_nzCC;
  case DPU::CMPB4rrrc:
    return ConditionClass::Log_setCC;
  case DPU::ADDC_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SH_UL_Srrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::MUL_SH_SH_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSL_SUBrrrici:
    return ConditionClass::Div_nzCC;
  case DPU::CLOrrci:
    return ConditionClass::Count_nzCC;
  case DPU::ASRzric:
    return ConditionClass::Log_setCC;
  case DPU::SUBzirc:
    return ConditionClass::Sub_setCC;
  case DPU::LSR1_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::CLZrrci:
    return ConditionClass::Count_nzCC;
  case DPU::ROL_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSRXzrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSRXzric:
    return ConditionClass::Log_setCC;
  case DPU::EXTSBrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSRX_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::CLZ_Srrci:
    return ConditionClass::Count_nzCC;
  case DPU::LSRXrrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::SUBC_Srirc:
    return ConditionClass::Sub_setCC;
  case DPU::NXOR_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::BOOTrici:
    return ConditionClass::BootCC;
  case DPU::RORrrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::CMPB4zrrci:
    return ConditionClass::Log_nzCC;
  case DPU::HASH_Urric:
    return ConditionClass::Log_setCC;
  case DPU::LSR1X_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSL1Xrrrc:
    return ConditionClass::Log_setCC;
  case DPU::MOVErici:
    return ConditionClass::Log_nzCC;
  case DPU::CLS_Srrc:
    return ConditionClass::Log_setCC;
  case DPU::ORN_Srrif:
    return ConditionClass::FalseCC;
  case DPU::RORrrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::MUL_SH_SLzrrc:
    return ConditionClass::Log_setCC;
  case DPU::XOR_Srric:
    return ConditionClass::Log_setCC;
  case DPU::LSRX_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSR1Xzrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::NAND_Urrif:
    return ConditionClass::FalseCC;
  case DPU::ASR_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::CLSzrci:
    return ConditionClass::Count_nzCC;
  case DPU::LSR1Xzrrc:
    return ConditionClass::Log_setCC;
  case DPU::NXORrrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSR_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::ASRrrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::CLZzrci:
    return ConditionClass::Count_nzCC;
  case DPU::MUL_UH_UHrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::MUL_SL_SLrrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSR_Urric:
    return ConditionClass::Log_setCC;
  case DPU::SUBzric:
    return ConditionClass::Ext_sub_setCC;
  case DPU::AND_Srric:
    return ConditionClass::Log_setCC;
  case DPU::ORzrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_ULrrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSL1zrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::EXTUH_Urrc:
    return ConditionClass::Log_setCC;
  case DPU::SUBzirci:
    return ConditionClass::Sub_nzCC;
  case DPU::SUBCzirci:
    return ConditionClass::Sub_nzCC;
  case DPU::ADDCzrif:
    return ConditionClass::FalseCC;
  case DPU::XOR_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORrric:
    return ConditionClass::Log_setCC;
  case DPU::NANDzric:
    return ConditionClass::Log_setCC;
  case DPU::LSL_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSR1X_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::AND_Urric:
    return ConditionClass::Log_setCC;
  case DPU::LSL1X_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::ADDzrrc:
    return ConditionClass::Log_setCC;
  case DPU::NXORzrici:
    return ConditionClass::Log_nzCC;
  case DPU::ADDC_Urrrci:
    return ConditionClass::Add_nzCC;
  case DPU::LSL_ADD_Srrrici:
    return ConditionClass::Div_nzCC;
  case DPU::ADDC_Srrrci:
    return ConditionClass::Add_nzCC;
  case DPU::RORrrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UL_ULzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::MUL_SH_UHzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::CAOrrci:
    return ConditionClass::Count_nzCC;
  case DPU::NANDzrrc:
    return ConditionClass::Log_setCC;
  case DPU::ROL_ADDzrrici:
    return ConditionClass::Div_nzCC;
  case DPU::MUL_STEPrrrici:
    return ConditionClass::BootCC;
  case DPU::MUL_UH_UHzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::XORrrif:
    return ConditionClass::FalseCC;
  case DPU::LSL1X_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::NAND_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::EXTSB_Srrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSR1X_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::HASHrric:
    return ConditionClass::Log_setCC;
  case DPU::ORN_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSLX_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::MUL_UH_UL_Urrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::LSRXrrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_SL_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::SATS_Urrc:
    return ConditionClass::Log_setCC;
  case DPU::ORrrif:
    return ConditionClass::FalseCC;
  case DPU::MUL_SL_ULzrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSR1Xrrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ADDzrrci:
    return ConditionClass::Add_nzCC;
  case DPU::LSR_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SH_SH_Srrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::EXTSH_Srrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UL_ULzrrc:
    return ConditionClass::Log_setCC;
  case DPU::SATSzrc:
    return ConditionClass::Log_setCC;
  case DPU::AND_Srrrci:
    return ConditionClass::Log_nzCC;
  case DPU::HASHzric:
    return ConditionClass::Log_setCC;
  case DPU::LSLrrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORNrrici:
    return ConditionClass::Log_nzCC;
  case DPU::LSR1rrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSR1rrrc:
    return ConditionClass::Log_setCC;
  case DPU::NOR_Srrif:
    return ConditionClass::FalseCC;
  case DPU::CLS_Urrci:
    return ConditionClass::Count_nzCC;
  case DPU::EXTUH_Urrci:
    return ConditionClass::Log_nzCC;
  case DPU::NXORrrici:
    return ConditionClass::Log_nzCC;
  case DPU::SUB_Srrif:
    return ConditionClass::FalseCC;
  case DPU::EXTUB_Urrc:
    return ConditionClass::Log_setCC;
  case DPU::ADDC_Srrici:
    return ConditionClass::Add_nzCC;
  case DPU::XOR_Srrif:
    return ConditionClass::FalseCC;
  case DPU::LSL_SUB_Srrrici:
    return ConditionClass::Div_nzCC;
  case DPU::SUBrirc:
    return ConditionClass::Sub_setCC;
  case DPU::MUL_SH_SL_Srrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::SUBCrric:
    return ConditionClass::Ext_sub_setCC;
  case DPU::NANDrrrci:
    return ConditionClass::Log_nzCC;
  case DPU::HASHrrrc:
    return ConditionClass::Log_setCC;
  case DPU::OR_Srrrci:
    return ConditionClass::Log_nzCC;
  case DPU::SUBCzric:
    return ConditionClass::Ext_sub_setCC;
  case DPU::ORN_Srrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSR1_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::SUB_Urirc:
    return ConditionClass::Sub_setCC;
  case DPU::SUBCzrici:
    return ConditionClass::Sub_nzCC;
  case DPU::LSLXzric:
    return ConditionClass::Log_setCC;
  case DPU::SUBC_Urric:
    return ConditionClass::Ext_sub_setCC;
  case DPU::ADD_Urrif:
    return ConditionClass::FalseCC;
  case DPU::NANDrrici:
    return ConditionClass::Log_nzCC;
  case DPU::ANDzric:
    return ConditionClass::Log_setCC;
  case DPU::ACQUIRErici:
    return ConditionClass::AcquireCC;
  case DPU::ANDN_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::SUBCzrrc:
    return ConditionClass::Ext_sub_setCC;
  case DPU::LSR1X_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ADDC_Srrif:
    return ConditionClass::FalseCC;
  case DPU::EXTUHrrc:
    return ConditionClass::Log_setCC;
  case DPU::RSUBCrrrc:
    return ConditionClass::Sub_setCC;
  case DPU::SUB_Urrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::RORrric:
    return ConditionClass::Log_setCC;
  case DPU::HASHrrif:
    return ConditionClass::FalseCC;
  case DPU::ANDN_Srrif:
    return ConditionClass::FalseCC;
  case DPU::RSUBzrrc:
    return ConditionClass::Sub_setCC;
  case DPU::ROR_Srric:
    return ConditionClass::Log_setCC;
  case DPU::LSRrrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::ASRrrrc:
    return ConditionClass::Log_setCC;
  case DPU::MOVDrrci:
    return ConditionClass::True_falseCC;
  case DPU::LSLzrrc:
    return ConditionClass::Log_setCC;
  case DPU::TIME_Urci:
    return ConditionClass::TrueCC;
  case DPU::MUL_SH_UHzrrc:
    return ConditionClass::Log_setCC;
  case DPU::NXOR_Urrrci:
    return ConditionClass::Log_nzCC;
  case DPU::MUL_UH_ULrrrc:
    return ConditionClass::Log_setCC;
  case DPU::ASR_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSLX_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::HASHrrici:
    return ConditionClass::Log_nzCC;
  case DPU::CLS_Srrci:
    return ConditionClass::Count_nzCC;
  case DPU::NANDrrif:
    return ConditionClass::FalseCC;
  case DPU::ORN_Urrici:
    return ConditionClass::Log_nzCC;
  case DPU::SUBC_Urirc:
    return ConditionClass::Sub_setCC;
  case DPU::LSL1_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ADDC_Urric:
    return ConditionClass::Log_setCC;
  case DPU::OR_Urrif:
    return ConditionClass::FalseCC;
  case DPU::LSR1X_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSRX_Srric:
    return ConditionClass::Log_setCC;
  case DPU::CMPB4_Urrrci:
    return ConditionClass::Log_nzCC;
  case DPU::ANDNzrici:
    return ConditionClass::Log_nzCC;
  case DPU::ROLrrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::NAND_Urrici:
    return ConditionClass::Log_nzCC;
  case DPU::LSL_ADDzrrici:
    return ConditionClass::Div_nzCC;
  case DPU::RSUBCzrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::LSR1X_Srric:
    return ConditionClass::Log_setCC;
  case DPU::SUBCrrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::LSL1zrrc:
    return ConditionClass::Log_setCC;
  case DPU::NAND_Urrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSL_SUB_Urrrici:
    return ConditionClass::Div_nzCC;
  case DPU::ROL_Urric:
    return ConditionClass::Log_setCC;
  case DPU::LSL1X_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::HASH_Srrif:
    return ConditionClass::FalseCC;
  case DPU::OR_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_UHrrrc:
    return ConditionClass::Log_setCC;
  case DPU::SUBC_Urrici:
    return ConditionClass::Sub_nzCC;
  case DPU::ADDC_Srric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_UHzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::ORN_Srric:
    return ConditionClass::Log_setCC;
  case DPU::HASH_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::SUB_Urrrc:
    return ConditionClass::Ext_sub_setCC;
  case DPU::ROL_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::SUB_Srirf:
    return ConditionClass::FalseCC;
  case DPU::ROL_ADDrrrici:
    return ConditionClass::Div_nzCC;
  case DPU::SUBCzrif:
    return ConditionClass::FalseCC;
  case DPU::ANDNzrif:
    return ConditionClass::FalseCC;
  case DPU::LSR1_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSR_ADD_Srrrici:
    return ConditionClass::Div_nzCC;
  case DPU::OR_Urrici:
    return ConditionClass::Log_nzCC;
  case DPU::HASH_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSR_Srric:
    return ConditionClass::Log_setCC;
  case DPU::NORrrif:
    return ConditionClass::FalseCC;
  case DPU::LSL_ADDrrrici:
    return ConditionClass::Div_nzCC;
  case DPU::LSL_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::ANDzrrci:
    return ConditionClass::Log_nzCC;
  case DPU::MUL_UH_ULrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::LSRzrrc:
    return ConditionClass::Log_setCC;
  case DPU::NXORzric:
    return ConditionClass::Log_setCC;
  case DPU::RORzric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UL_ULrrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSR1_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ANDrrrci:
    return ConditionClass::Log_nzCC;
  case DPU::NOR_Srric:
    return ConditionClass::Log_setCC;
  case DPU::XOR_Urrrci:
    return ConditionClass::Log_nzCC;
  case DPU::AND_Urrici:
    return ConditionClass::Log_nzCC;
  case DPU::SUBCzirf:
    return ConditionClass::FalseCC;
  case DPU::SUB_Urirci:
    return ConditionClass::Sub_nzCC;
  case DPU::MUL_SL_SLzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::MUL_SL_UH_Srrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::XORrrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSL1Xzrrc:
    return ConditionClass::Log_setCC;
  case DPU::ADD_Srric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UL_UL_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::NAND_Srrif:
    return ConditionClass::FalseCC;
  case DPU::ANDNzrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORzrif:
    return ConditionClass::FalseCC;
  case DPU::LSRX_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::EXTUHzrci:
    return ConditionClass::Log_nzCC;
  case DPU::SUB_Urirf:
    return ConditionClass::FalseCC;
  case DPU::DIV_STEPrrrici:
    return ConditionClass::DivCC;
  case DPU::EXTSHrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORzrici:
    return ConditionClass::Log_nzCC;
  case DPU::CAO_Srrci:
    return ConditionClass::Count_nzCC;
  case DPU::LSRXzrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ADDCzric:
    return ConditionClass::Log_setCC;
  case DPU::NORzrif:
    return ConditionClass::FalseCC;
  case DPU::READ_RUNrici:
    return ConditionClass::BootCC;
  case DPU::LSL_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSR1zrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSR1X_Urric:
    return ConditionClass::Log_setCC;
  case DPU::TIME_CFGzrci:
    return ConditionClass::TrueCC;
  case DPU::LSLrrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSRXrric:
    return ConditionClass::Log_setCC;
  case DPU::TIMErci:
    return ConditionClass::TrueCC;
  case DPU::LSLX_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ANDNrric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SH_UHrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::HASH_Srric:
    return ConditionClass::Log_setCC;
  case DPU::EXTSHzrc:
    return ConditionClass::Log_setCC;
  case DPU::ORNzric:
    return ConditionClass::Log_setCC;
  case DPU::LSL1X_Srric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_SHzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::NXOR_Urrif:
    return ConditionClass::FalseCC;
  case DPU::ORN_Urrrci:
    return ConditionClass::Log_nzCC;
  case DPU::ANDN_Urrici:
    return ConditionClass::Log_nzCC;
  case DPU::SUBC_Srirf:
    return ConditionClass::FalseCC;
  case DPU::SUBCrrrc:
    return ConditionClass::Ext_sub_setCC;
  case DPU::ADDCrrrci:
    return ConditionClass::Add_nzCC;
  case DPU::MUL_SL_SLzrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SH_ULrrrc:
    return ConditionClass::Log_setCC;
  case DPU::NAND_Urric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UH_UHrrrc:
    return ConditionClass::Log_setCC;
  case DPU::OR_Urrrci:
    return ConditionClass::Log_nzCC;
  case DPU::HASHzrrci:
    return ConditionClass::Log_nzCC;
  case DPU::NORrrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORzrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSR1X_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::NXORrrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSRXzrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::RSUB_Urrrc:
    return ConditionClass::Sub_setCC;
  case DPU::MUL_SL_UH_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORNzrrci:
    return ConditionClass::Log_nzCC;
  case DPU::NOR_Urric:
    return ConditionClass::Log_setCC;
  case DPU::ADDC_Urrif:
    return ConditionClass::FalseCC;
  case DPU::ADD_Urrici:
    return ConditionClass::Add_nzCC;
  case DPU::SWAPDrrci:
    return ConditionClass::True_falseCC;
  case DPU::RSUBC_Urrrc:
    return ConditionClass::Sub_setCC;
  case DPU::XOR_Srrici:
    return ConditionClass::Log_nzCC;
  case DPU::LSL1Xrrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::ADD_Srrici:
    return ConditionClass::Add_nzCC;
  case DPU::ROL_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::NORzrrci:
    return ConditionClass::Log_nzCC;
  case DPU::SUBCrirc:
    return ConditionClass::Sub_setCC;
  case DPU::LSLzric:
    return ConditionClass::Log_setCC;
  case DPU::LSR_ADDrrrici:
    return ConditionClass::Div_nzCC;
  case DPU::LSR1Xrrrc:
    return ConditionClass::Log_setCC;
  case DPU::ROLrrrc:
    return ConditionClass::Log_setCC;
  case DPU::XOR_Urrici:
    return ConditionClass::Log_nzCC;
  case DPU::SUBCrirf:
    return ConditionClass::FalseCC;
  case DPU::SUBC_Srrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::XORrrrci:
    return ConditionClass::Log_nzCC;
  case DPU::MUL_SH_SL_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSR1Xrric:
    return ConditionClass::Log_setCC;
  case DPU::NXORzrif:
    return ConditionClass::FalseCC;
  case DPU::EXTUBzrci:
    return ConditionClass::Log_nzCC;
  case DPU::SUBrric:
    return ConditionClass::Ext_sub_setCC;
  case DPU::CLZzrc:
    return ConditionClass::Log_setCC;
  case DPU::ORNzrif:
    return ConditionClass::FalseCC;
  case DPU::LSL1Xzrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::MUL_SH_SLrrrc:
    return ConditionClass::Log_setCC;
  case DPU::EXTUHrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSR_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::SUBC_Srrrc:
    return ConditionClass::Ext_sub_setCC;
  case DPU::RSUBCzrrc:
    return ConditionClass::Sub_setCC;
  case DPU::CLSrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SH_UH_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::CAOrrc:
    return ConditionClass::Log_setCC;
  case DPU::ROL_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ASR_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSLXzrrc:
    return ConditionClass::Log_setCC;
  case DPU::CLZ_Urrci:
    return ConditionClass::Count_nzCC;
  case DPU::ORNzrrc:
    return ConditionClass::Log_setCC;
  case DPU::RSUBzrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::ANDrrici:
    return ConditionClass::Log_nzCC;
  case DPU::HASHrrrci:
    return ConditionClass::Log_nzCC;
  case DPU::NXOR_Srric:
    return ConditionClass::Log_setCC;
  case DPU::XORzrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSL1_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SH_SHzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::LSL1X_Urric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_ULrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::LSLrric:
    return ConditionClass::Log_setCC;
  case DPU::SUBzirf:
    return ConditionClass::FalseCC;
  case DPU::MUL_SL_SL_Srrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::SATSrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSLXzrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSR1_Srric:
    return ConditionClass::Log_setCC;
  case DPU::EXTSBrrc:
    return ConditionClass::Log_setCC;
  case DPU::XOR_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::ADDCrrici:
    return ConditionClass::Add_nzCC;
  case DPU::SUBC_Srrici:
    return ConditionClass::Sub_nzCC;
  case DPU::SUBC_Srrif:
    return ConditionClass::FalseCC;
  case DPU::ANDN_Urrif:
    return ConditionClass::FalseCC;
  case DPU::CLSzrc:
    return ConditionClass::Log_setCC;
  case DPU::ADDCzrrci:
    return ConditionClass::Add_nzCC;
  case DPU::LSL1_Urric:
    return ConditionClass::Log_setCC;
  case DPU::EXTUBrrc:
    return ConditionClass::Log_setCC;
  case DPU::SATS_Srrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSRX_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSL1_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::CAOzrc:
    return ConditionClass::Log_setCC;
  case DPU::NOR_Urrici:
    return ConditionClass::Log_nzCC;
  case DPU::EXTSBzrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSR_ADDzrrici:
    return ConditionClass::Div_nzCC;
  case DPU::MUL_UL_UHzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::ROL_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::CLS_Urrc:
    return ConditionClass::Log_setCC;
  case DPU::SUBCrrif:
    return ConditionClass::FalseCC;
  case DPU::MUL_SH_ULzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::AND_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSLX_Srric:
    return ConditionClass::Log_setCC;
  case DPU::LSR_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SH_UHrrrc:
    return ConditionClass::Log_setCC;
  case DPU::ANDNzric:
    return ConditionClass::Log_setCC;
  case DPU::RSUBrrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::LSL1X_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::RSUBC_Urrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::LSLXrrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::SATS_Srrc:
    return ConditionClass::Log_setCC;
  case DPU::HASH_Urrici:
    return ConditionClass::Log_nzCC;
  case DPU::LSRX_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::SUB_Srrici:
    return ConditionClass::Sub_nzCC;
  case DPU::OR_Srrici:
    return ConditionClass::Log_nzCC;
  case DPU::MUL_UH_UL_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::ANDNrrici:
    return ConditionClass::Log_nzCC;
  case DPU::ANDN_Urric:
    return ConditionClass::Log_setCC;
  case DPU::LSLzrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::MUL_UH_UHzrrc:
    return ConditionClass::Log_setCC;
  case DPU::TIME_Srci:
    return ConditionClass::TrueCC;
  case DPU::CAO_Urrc:
    return ConditionClass::Log_setCC;
  case DPU::RORzrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ROR_Urrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSR_ADD_Urrrici:
    return ConditionClass::Div_nzCC;
  case DPU::ROLzric:
    return ConditionClass::Log_setCC;
  case DPU::OR_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ADDCrric:
    return ConditionClass::Log_setCC;
  case DPU::SUB_Srrrc:
    return ConditionClass::Ext_sub_setCC;
  case DPU::MUL_SL_UL_Srrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::CMPB4_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ADD_Srrrci:
    return ConditionClass::Add_nzCC;
  case DPU::TIME_CFGrrci:
    return ConditionClass::TrueCC;
  case DPU::MUL_UL_UHrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::ROR_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::EXTSH_Srrci:
    return ConditionClass::Log_nzCC;
  case DPU::ANDzrrc:
    return ConditionClass::Log_setCC;
  case DPU::SUBC_Urrif:
    return ConditionClass::FalseCC;
  case DPU::RESUMErici:
    return ConditionClass::BootCC;
  case DPU::LSL1zric:
    return ConditionClass::Log_setCC;
  case DPU::NXOR_Urric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_SH_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ADDrrrci:
    return ConditionClass::Add_nzCC;
  case DPU::ROR_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSLrrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSL1X_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ADD_Srrif:
    return ConditionClass::FalseCC;
  case DPU::LSLXrric:
    return ConditionClass::Log_setCC;
  case DPU::ANDrrif:
    return ConditionClass::FalseCC;
  case DPU::ROLzrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSLXrrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSRX_Urric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SH_UL_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::SUBC_Urrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::LSLX_Urric:
    return ConditionClass::Log_setCC;
  case DPU::SUBCzrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::ANDN_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORN_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::XOR_Srrrci:
    return ConditionClass::Log_nzCC;
  case DPU::NOR_Srrrci:
    return ConditionClass::Log_nzCC;
  case DPU::SUBrrici:
    return ConditionClass::Sub_nzCC;
  case DPU::SUB_Urric:
    return ConditionClass::Ext_sub_setCC;
  case DPU::LSL1Xrric:
    return ConditionClass::Log_setCC;
  case DPU::OR_Srrif:
    return ConditionClass::FalseCC;
  case DPU::HASH_Urrrci:
    return ConditionClass::Log_nzCC;
  case DPU::MUL_SH_SHrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::LSL1rrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ORN_Urric:
    return ConditionClass::Log_setCC;
  case DPU::RSUBC_Srrrc:
    return ConditionClass::Sub_setCC;
  case DPU::AND_Srrif:
    return ConditionClass::FalseCC;
  case DPU::ADDC_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::NXOR_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::ANDNrrif:
    return ConditionClass::FalseCC;
  case DPU::XOR_Urrif:
    return ConditionClass::FalseCC;
  case DPU::ASRzrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::NXORrric:
    return ConditionClass::Log_setCC;
  case DPU::LSRrric:
    return ConditionClass::Log_setCC;
  case DPU::SUBC_Urirci:
    return ConditionClass::Sub_nzCC;
  case DPU::LSR1Xzrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::ANDrrrc:
    return ConditionClass::Log_setCC;
  case DPU::TIMEzci:
    return ConditionClass::TrueCC;
  case DPU::NANDzrif:
    return ConditionClass::FalseCC;
  case DPU::ADDrrici:
    return ConditionClass::Add_nzCC;
  case DPU::XORzrif:
    return ConditionClass::FalseCC;
  case DPU::ADDCrrif:
    return ConditionClass::FalseCC;
  case DPU::MUL_UL_UHrrrc:
    return ConditionClass::Log_setCC;
  case DPU::SUBzrici:
    return ConditionClass::Sub_nzCC;
  case DPU::ANDN_Urrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSR1rric:
    return ConditionClass::Log_setCC;
  case DPU::NEGrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::HASHzrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSL1Xzric:
    return ConditionClass::Log_setCC;
  case DPU::NOR_Urrif:
    return ConditionClass::FalseCC;
  case DPU::SUBCrrici:
    return ConditionClass::Sub_nzCC;
  case DPU::CLOzrci:
    return ConditionClass::Count_nzCC;
  case DPU::ASRrric:
    return ConditionClass::Log_setCC;
  case DPU::SUB_Srrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::NXOR_Srrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSR_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSLX_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::CAOzrci:
    return ConditionClass::Count_nzCC;
  case DPU::CMPB4_Srrrci:
    return ConditionClass::Log_nzCC;
  case DPU::RSUBrrrc:
    return ConditionClass::Sub_setCC;
  case DPU::STOPci:
    return ConditionClass::BootCC;
  case DPU::RELEASErici:
    return ConditionClass::ReleaseCC;
  case DPU::EXTUBzrc:
    return ConditionClass::Log_setCC;
  case DPU::LSRrrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::NORrric:
    return ConditionClass::Log_setCC;
  case DPU::ADDCzrici:
    return ConditionClass::Add_nzCC;
  case DPU::CAO_Urrci:
    return ConditionClass::Count_nzCC;
  case DPU::RSUBCrrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::ADD_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::RSUB_Srrrc:
    return ConditionClass::Sub_setCC;
  case DPU::MUL_UL_UH_Urrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::MUL_SH_SLrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::ASR_Urric:
    return ConditionClass::Log_setCC;
  case DPU::ADD_Urrrci:
    return ConditionClass::Add_nzCC;
  case DPU::LSRzric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UL_ULrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::XORzrici:
    return ConditionClass::Log_nzCC;
  case DPU::NOR_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::NORzrici:
    return ConditionClass::Log_nzCC;
  case DPU::NXOR_Urrici:
    return ConditionClass::Log_nzCC;
  case DPU::ANDNrrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORNrric:
    return ConditionClass::Log_setCC;
  case DPU::NXOR_Srrif:
    return ConditionClass::FalseCC;
  case DPU::LSL_ADD_Urrrici:
    return ConditionClass::Div_nzCC;
  case DPU::CLO_Srrc:
    return ConditionClass::Log_setCC;
  case DPU::ADDrrif:
    return ConditionClass::FalseCC;
  case DPU::ANDN_Srrrci:
    return ConditionClass::Log_nzCC;
  case DPU::SATSrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSL_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::ROL_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSL1Xzrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::SUBrrif:
    return ConditionClass::FalseCC;
  case DPU::MOVErrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSR1_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::OR_Urric:
    return ConditionClass::Log_setCC;
  case DPU::ANDNzrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSL1Xrrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::TIME_CFG_Urrci:
    return ConditionClass::TrueCC;
  case DPU::MUL_SH_SHzrrc:
    return ConditionClass::Log_setCC;
  case DPU::RORzrrc:
    return ConditionClass::Log_setCC;
  case DPU::ORN_Urrif:
    return ConditionClass::FalseCC;
  case DPU::LSL1_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::ASR_Srric:
    return ConditionClass::Log_setCC;
  case DPU::SUBzrif:
    return ConditionClass::FalseCC;
  case DPU::ADD_Urric:
    return ConditionClass::Log_setCC;
  case DPU::SUB_Urrici:
    return ConditionClass::Sub_nzCC;
  case DPU::ROR_Urric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_UHrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::NAND_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::XORrrici:
    return ConditionClass::Log_nzCC;
  case DPU::ASRzrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_SLrrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::LSL1_Urrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::SUBrirf:
    return ConditionClass::FalseCC;
  case DPU::HASHzrif:
    return ConditionClass::FalseCC;
  case DPU::ANDNrrrci:
    return ConditionClass::Log_nzCC;
  case DPU::ORzric:
    return ConditionClass::Log_setCC;
  case DPU::NXORzrrci:
    return ConditionClass::Log_nzCC;
  case DPU::ROLzrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSL_SUBzrrici:
    return ConditionClass::Div_nzCC;
  case DPU::ADDCrrrc:
    return ConditionClass::Log_setCC;
  case DPU::NOR_Srrici:
    return ConditionClass::Log_nzCC;
  case DPU::SUB_Urrif:
    return ConditionClass::FalseCC;
  case DPU::SUBC_Srric:
    return ConditionClass::Ext_sub_setCC;
  case DPU::ADDrrrc:
    return ConditionClass::Log_setCC;
  case DPU::NORzrrc:
    return ConditionClass::Log_setCC;
  case DPU::HASH_Srrici:
    return ConditionClass::Log_nzCC;
  case DPU::LSL_Srrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSL1rrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::MUL_SL_UHzrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_SL_SHzrrc:
    return ConditionClass::Log_setCC;
  case DPU::LSR1zric:
    return ConditionClass::Log_setCC;
  case DPU::LSRzrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSLX_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::SUBzrrc:
    return ConditionClass::Ext_sub_setCC;
  case DPU::LSL_Urric:
    return ConditionClass::Log_setCC;
  case DPU::ADDzrici:
    return ConditionClass::Add_nzCC;
  case DPU::LSRrrrc:
    return ConditionClass::Log_setCC;
  case DPU::NORrrici:
    return ConditionClass::Log_nzCC;
  case DPU::MUL_SL_ULzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::EXTSBzrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UH_ULzrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::SUB_Srric:
    return ConditionClass::Ext_sub_setCC;
  case DPU::LSL_Srric:
    return ConditionClass::Log_setCC;
  case DPU::ROR_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::EXTUB_Urrci:
    return ConditionClass::Log_nzCC;
  case DPU::XOR_Urric:
    return ConditionClass::Log_setCC;
  case DPU::ADDzrif:
    return ConditionClass::FalseCC;
  case DPU::ANDrric:
    return ConditionClass::Log_setCC;
  case DPU::ADDrric:
    return ConditionClass::Log_setCC;
  case DPU::HASHzrici:
    return ConditionClass::Log_nzCC;
  case DPU::ORrrrc:
    return ConditionClass::Log_setCC;
  case DPU::NANDzrici:
    return ConditionClass::Log_nzCC;
  case DPU::SUBrrrc:
    return ConditionClass::Ext_sub_setCC;
  case DPU::ORrrrci:
    return ConditionClass::Log_nzCC;
  case DPU::LSL1X_Srrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::MUL_SL_UL_Srrrc:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UH_UH_Urrrc:
    return ConditionClass::Log_setCC;
  case DPU::AND_Urrif:
    return ConditionClass::FalseCC;
  case DPU::LSRXrrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::LSL1_Srric:
    return ConditionClass::Log_setCC;
  case DPU::SUBzrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::LSR1zrici:
    return ConditionClass::Imm_shift_nzCC;
  case DPU::LSR1_Urric:
    return ConditionClass::Log_setCC;
  case DPU::SUBCzirc:
    return ConditionClass::Sub_setCC;
  case DPU::ROL_ADD_Srrrici:
    return ConditionClass::Div_nzCC;
  case DPU::RORzrrci:
    return ConditionClass::Shift_nzCC;
  case DPU::XORrric:
    return ConditionClass::Log_setCC;
  case DPU::MUL_UH_UH_Urrrci:
    return ConditionClass::Mul_nzCC;
  case DPU::RSUB_Urrrci:
    return ConditionClass::Sub_nzCC;
  case DPU::SATSzrci:
    return ConditionClass::Log_nzCC;
  }
}
} // namespace DPUAsmCondition
} // namespace llvm
