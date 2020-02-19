 //===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <Accelerate/Accelerate.h>

int main()
{
	vFloat valueFL = {1.25,0,0.25,0};
	vDouble valueDL = {1.25,2.25};
	int16_t valueI16[8] = {1,0,4,0,0,1,0,4};
	int32_t valueI32[4] = {1,0,4,0};
	vUInt8 valueU8 = {1,0,4,0,0,1,0,4};
	vUInt16 valueU16 = {1,0,4,0,0,1,0,4};
	vUInt32 valueU32 = {1,2,3,4};
	vSInt8 valueS8 = {1,0,4,0,0,1,0,4};
	vSInt16 valueS16 = {1,0,4,0,0,1,0,4};
	vSInt32 valueS32 = {4,3,2,1};
	vBool32 valueBool32 = {false,true,false,true};
	return 0; // Set break point at this line.
}
