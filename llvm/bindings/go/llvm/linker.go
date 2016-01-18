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
	var cmsg *C.char
	failed := C.LLVMLinkModules(Dest.C, Src.C, C.LLVMLinkerDestroySource, &cmsg)
	if failed != 0 {
		err := errors.New(C.GoString(cmsg))
		C.LLVMDisposeMessage(cmsg)
		return err
	}
	return nil
}
