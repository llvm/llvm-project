//===- transforms_instrumentation.go - Bindings for instrumentation -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the instrumentation component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "InstrumentationBindings.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

func (pm PassManager) AddAddressSanitizerFunctionPass() {
	C.LLVMAddAddressSanitizerFunctionPass(pm.C)
}

func (pm PassManager) AddAddressSanitizerModulePass() {
	C.LLVMAddAddressSanitizerModulePass(pm.C)
}

func (pm PassManager) AddThreadSanitizerPass() {
	C.LLVMAddThreadSanitizerPass(pm.C)
}

func (pm PassManager) AddMemorySanitizerLegacyPassPass() {
	C.LLVMAddMemorySanitizerLegacyPassPass(pm.C)
}

func (pm PassManager) AddDataFlowSanitizerPass(abilist []string) {
	abiliststrs := make([]*C.char, len(abilist))
	for i, arg := range abilist {
		abiliststrs[i] = C.CString(arg)
		defer C.free(unsafe.Pointer(abiliststrs[i]))
	}
	C.LLVMAddDataFlowSanitizerPass(pm.C, C.int(len(abilist)), &abiliststrs[0])
}
