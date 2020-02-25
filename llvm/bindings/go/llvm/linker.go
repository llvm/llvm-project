//===- linker.go - Bindings for linker ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the linker component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Core.h"
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
