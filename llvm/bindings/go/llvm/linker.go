//===- linker.go - Bindings for linker ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the linker component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Linker.h"
#include <stdlib.h>
*/
import "C"
import "errors"

func LinkModules(Dest, Src Module) error {
	failed := C.LLVMLinkModules2(Dest.C, Src.C)
	if failed != 0 {
		err := errors.New("Linking failed")
		return err
	}
	return nil
}
