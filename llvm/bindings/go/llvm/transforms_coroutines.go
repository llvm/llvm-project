//===- transforms_coroutines.go - Bindings for coroutines -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the coroutines component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Transforms/Coroutines.h"
*/
import "C"

func (pm PassManager) AddCoroEarlyPass()      { C.LLVMAddCoroEarlyPass(pm.C) }
func (pm PassManager) AddCoroSplitPass()      { C.LLVMAddCoroSplitPass(pm.C) }
func (pm PassManager) AddCoroElidePass()      { C.LLVMAddCoroElidePass(pm.C) }
func (pm PassManager) AddCoroCleanupPass()    { C.LLVMAddCoroCleanupPass(pm.C) }
