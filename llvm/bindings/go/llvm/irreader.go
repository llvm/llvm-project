//===- irreader.go - Bindings for irreader --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the irreader component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/IRReader.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"unsafe"
)

// ParseIR parses the textual IR given in the memory buffer and returns a new
// LLVM module in this context.
func (c *Context) ParseIR(buf MemoryBuffer) (Module, error) {
	var m Module
	var errmsg *C.char
	if C.LLVMParseIRInContext(c.C, buf.C, &m.C, &errmsg) != 0 {
		err := errors.New(C.GoString(errmsg))
		C.free(unsafe.Pointer(errmsg))
		return Module{}, err
	}
	return m, nil
}
