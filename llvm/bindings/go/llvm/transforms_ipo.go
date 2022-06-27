//===- transforms_ipo.go - Bindings for ipo -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the ipo component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Transforms/IPO.h"
*/
import "C"

// helpers
func boolToUnsigned(b bool) C.unsigned {
	if b {
		return 1
	}
	return 0
}

func (pm PassManager) AddConstantMergePass()         { C.LLVMAddConstantMergePass(pm.C) }
func (pm PassManager) AddDeadArgEliminationPass()    { C.LLVMAddDeadArgEliminationPass(pm.C) }
func (pm PassManager) AddFunctionAttrsPass()         { C.LLVMAddFunctionAttrsPass(pm.C) }
func (pm PassManager) AddFunctionInliningPass()      { C.LLVMAddFunctionInliningPass(pm.C) }
func (pm PassManager) AddGlobalDCEPass()             { C.LLVMAddGlobalDCEPass(pm.C) }
func (pm PassManager) AddGlobalOptimizerPass()       { C.LLVMAddGlobalOptimizerPass(pm.C) }
func (pm PassManager) AddPruneEHPass()               { C.LLVMAddPruneEHPass(pm.C) }
func (pm PassManager) AddIPSCCPPass()                { C.LLVMAddIPSCCPPass(pm.C) }
func (pm PassManager) AddInternalizePass(allButMain bool) {
	C.LLVMAddInternalizePass(pm.C, boolToUnsigned(allButMain))
}
func (pm PassManager) AddStripDeadPrototypesPass() { C.LLVMAddStripDeadPrototypesPass(pm.C) }
