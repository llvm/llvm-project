//===- analysis.go - Bindings for analysis --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the analysis component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Analysis.h" // If you are getting an error here read bindings/go/README.txt
#include "llvm-c/Core.h"
#include <stdlib.h>
*/
import "C"
import "errors"

type VerifierFailureAction C.LLVMVerifierFailureAction

const (
	// verifier will print to stderr and abort()
	AbortProcessAction VerifierFailureAction = C.LLVMAbortProcessAction
	// verifier will print to stderr and return 1
	PrintMessageAction VerifierFailureAction = C.LLVMPrintMessageAction
	// verifier will just return 1
	ReturnStatusAction VerifierFailureAction = C.LLVMReturnStatusAction
)

// Verifies that a module is valid, taking the specified action if not.
// Optionally returns a human-readable description of any invalid constructs.
func VerifyModule(m Module, a VerifierFailureAction) error {
	var cmsg *C.char
	broken := C.LLVMVerifyModule(m.C, C.LLVMVerifierFailureAction(a), &cmsg)

	// C++'s verifyModule means isModuleBroken, so it returns false if
	// there are no errors
	if broken != 0 {
		err := errors.New(C.GoString(cmsg))
		C.LLVMDisposeMessage(cmsg)
		return err
	}
	return nil
}

var verifyFunctionError = errors.New("Function is broken")

// Verifies that a single function is valid, taking the specified action.
// Useful for debugging.
func VerifyFunction(f Value, a VerifierFailureAction) error {
	broken := C.LLVMVerifyFunction(f.C, C.LLVMVerifierFailureAction(a))

	// C++'s verifyFunction means isFunctionBroken, so it returns false if
	// there are no errors
	if broken != 0 {
		return verifyFunctionError
	}
	return nil
}

// Open up a ghostview window that displays the CFG of the current function.
// Useful for debugging.
func ViewFunctionCFG(f Value)     { C.LLVMViewFunctionCFG(f.C) }
func ViewFunctionCFGOnly(f Value) { C.LLVMViewFunctionCFGOnly(f.C) }
