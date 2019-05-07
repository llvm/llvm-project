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
		BufferIn10 = 0,
		BufferIn11 = 1,
		BufferIn12 = 2,
		BufferIn13 = 3,
		BufferIn14 = 4,
		BufferIn5 = 5,
		BufferIn6 = 6,
		BufferIn7 = 7,
		BufferIn8 = 8,
		BufferIn9 = 9,
		Carry = 10,
		Equal = 11,
		Even = 12,
		ExtendedGreaterThanSigned = 13,
		ExtendedGreaterThanUnsigned = 14,
		ExtendedLessOrEqualSigned = 15,
		ExtendedLessOrEqualUnsigned = 16,
		ExtendedNotZero = 17,
		ExtendedZero = 18,
		False = 19,
		GreaterOrEqualSigned = 20,
		GreaterOrEqualUnsigned = 21,
		GreaterThanSigned = 22,
		GreaterThanUnsigned = 23,
		Large = 24,
		LessOrEqualSigned = 25,
		LessOrEqualUnsigned = 26,
		LessThanSigned = 27,
		LessThanUnsigned = 28,
		Maximum = 29,
		Negative = 30,
		NotCarry = 31,
		NotEqual = 32,
		NotMaximum = 33,
		NotOverflow = 34,
		NotShift32 = 35,
		NotZero = 36,
		Odd = 37,
		Overflow = 38,
		PositiveOrNull = 39,
		SetCarry = 40,
		SetEqual = 41,
		SetExtendedGreaterThanSigned = 42,
		SetExtendedGreaterThanUnsigned = 43,
		SetExtendedLessOrEqualSigned = 44,
		SetExtendedLessOrEqualUnsigned = 45,
		SetExtendedNotZero = 46,
		SetExtendedZero = 47,
		SetGreaterOrEqualSigned = 48,
		SetGreaterOrEqualUnsigned = 49,
		SetGreaterThanSigned = 50,
		SetGreaterThanUnsigned = 51,
		SetLessOrEqualSigned = 52,
		SetLessOrEqualUnsigned = 53,
		SetLessThanSigned = 54,
		SetLessThanUnsigned = 55,
		SetNegative = 56,
		SetNotCarry = 57,
		SetNotEqual = 58,
		SetNotOverflow = 59,
		SetNotZero = 60,
		SetOverflow = 61,
		SetPositiveOrNull = 62,
		SetSourceNegative = 63,
		SetSourceNotZero = 64,
		SetSourcePositiveOrNull = 65,
		SetSourceZero = 66,
		SetTrue = 67,
		SetZero = 68,
		Shift32 = 69,
		Small = 70,
		SourceEven = 71,
		SourceNegative = 72,
		SourceNotZero = 73,
		SourceOdd = 74,
		SourcePositiveOrNull = 75,
		SourceZero = 76,
		True = 77,
		Zero = 78,
		NR_CONDITIONS = 79
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

	bool fromString(const std::string &string, Condition& Cond);

	StringRef toString(Condition Cond);

	bool isInConditionClass(Condition Cond, ConditionClass CondClass);

	int64_t getEncoding(Condition Cond, ConditionClass CondClass);

	int64_t getDecoding(uint64_t Cond, ConditionClass CondClass);

	ConditionClass findConditionClassForInstruction(unsigned InstOpcode);
}
}

#endif //LLVM_DPUASMCONDITION_H
