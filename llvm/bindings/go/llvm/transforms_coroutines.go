//===- transforms_coroutines.go - Bindings for coroutines -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

func (pm PassManager) AddCoroEarlyPass()   { C.LLVMAddCoroEarlyPass(pm.C) }
func (pm PassManager) AddCoroSplitPass()   { C.LLVMAddCoroSplitPass(pm.C) }
func (pm PassManager) AddCoroElidePass()   { C.LLVMAddCoroElidePass(pm.C) }
func (pm PassManager) AddCoroCleanupPass() { C.LLVMAddCoroCleanupPass(pm.C) }
