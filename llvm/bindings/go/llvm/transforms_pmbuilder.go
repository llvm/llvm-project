//===- transforms_pmbuilder.go - Bindings for PassManagerBuilder ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the PassManagerBuilder class.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Transforms/PassManagerBuilder.h"
*/
import "C"

type PassManagerBuilder struct {
	C C.LLVMPassManagerBuilderRef
}

func NewPassManagerBuilder() (pmb PassManagerBuilder) {
	pmb.C = C.LLVMPassManagerBuilderCreate()
	return
}

func (pmb PassManagerBuilder) SetOptLevel(level int) {
	C.LLVMPassManagerBuilderSetOptLevel(pmb.C, C.uint(level))
}

func (pmb PassManagerBuilder) SetSizeLevel(level int) {
	C.LLVMPassManagerBuilderSetSizeLevel(pmb.C, C.uint(level))
}

func (pmb PassManagerBuilder) Populate(pm PassManager) {
	C.LLVMPassManagerBuilderPopulateModulePassManager(pmb.C, pm.C)
}

func (pmb PassManagerBuilder) PopulateFunc(pm PassManager) {
	C.LLVMPassManagerBuilderPopulateFunctionPassManager(pmb.C, pm.C)
}

func (pmb PassManagerBuilder) Dispose() {
	C.LLVMPassManagerBuilderDispose(pmb.C)
}

func (pmb PassManagerBuilder) SetDisableUnitAtATime(val bool) {
	C.LLVMPassManagerBuilderSetDisableUnitAtATime(pmb.C, boolToLLVMBool(val))
}

func (pmb PassManagerBuilder) SetDisableUnrollLoops(val bool) {
	C.LLVMPassManagerBuilderSetDisableUnrollLoops(pmb.C, boolToLLVMBool(val))
}

func (pmb PassManagerBuilder) SetDisableSimplifyLibCalls(val bool) {
	C.LLVMPassManagerBuilderSetDisableSimplifyLibCalls(pmb.C, boolToLLVMBool(val))
}

func (pmb PassManagerBuilder) UseInlinerWithThreshold(threshold uint) {
	C.LLVMPassManagerBuilderUseInlinerWithThreshold(pmb.C, C.uint(threshold))
}
