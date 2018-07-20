//===- transforms_scalar.go - Bindings for scalaropts ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the scalaropts component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Transforms/Scalar.h"
#include "llvm-c/Transforms/Utils.h"
*/
import "C"

func (pm PassManager) AddAggressiveDCEPass()           { C.LLVMAddAggressiveDCEPass(pm.C) }
func (pm PassManager) AddCFGSimplificationPass()       { C.LLVMAddCFGSimplificationPass(pm.C) }
func (pm PassManager) AddDeadStoreEliminationPass()    { C.LLVMAddDeadStoreEliminationPass(pm.C) }
func (pm PassManager) AddGVNPass()                     { C.LLVMAddGVNPass(pm.C) }
func (pm PassManager) AddIndVarSimplifyPass()          { C.LLVMAddIndVarSimplifyPass(pm.C) }
func (pm PassManager) AddInstructionCombiningPass()    { C.LLVMAddInstructionCombiningPass(pm.C) }
func (pm PassManager) AddJumpThreadingPass()           { C.LLVMAddJumpThreadingPass(pm.C) }
func (pm PassManager) AddLICMPass()                    { C.LLVMAddLICMPass(pm.C) }
func (pm PassManager) AddLoopDeletionPass()            { C.LLVMAddLoopDeletionPass(pm.C) }
func (pm PassManager) AddLoopRotatePass()              { C.LLVMAddLoopRotatePass(pm.C) }
func (pm PassManager) AddLoopUnrollPass()              { C.LLVMAddLoopUnrollPass(pm.C) }
func (pm PassManager) AddLoopUnswitchPass()            { C.LLVMAddLoopUnswitchPass(pm.C) }
func (pm PassManager) AddMemCpyOptPass()               { C.LLVMAddMemCpyOptPass(pm.C) }
func (pm PassManager) AddPromoteMemoryToRegisterPass() { C.LLVMAddPromoteMemoryToRegisterPass(pm.C) }
func (pm PassManager) AddReassociatePass()             { C.LLVMAddReassociatePass(pm.C) }
func (pm PassManager) AddSCCPPass()                    { C.LLVMAddSCCPPass(pm.C) }
func (pm PassManager) AddScalarReplAggregatesPass()    { C.LLVMAddScalarReplAggregatesPass(pm.C) }
func (pm PassManager) AddScalarReplAggregatesPassWithThreshold(threshold int) {
	C.LLVMAddScalarReplAggregatesPassWithThreshold(pm.C, C.int(threshold))
}
func (pm PassManager) AddSimplifyLibCallsPass()       { C.LLVMAddSimplifyLibCallsPass(pm.C) }
func (pm PassManager) AddTailCallEliminationPass()    { C.LLVMAddTailCallEliminationPass(pm.C) }
func (pm PassManager) AddConstantPropagationPass()    { C.LLVMAddConstantPropagationPass(pm.C) }
func (pm PassManager) AddDemoteMemoryToRegisterPass() { C.LLVMAddDemoteMemoryToRegisterPass(pm.C) }
func (pm PassManager) AddVerifierPass()               { C.LLVMAddVerifierPass(pm.C) }
