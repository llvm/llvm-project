//===- string_test.go - test Stringer implementation for Type -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests the Stringer interface for the Type type.
//
//===----------------------------------------------------------------------===//

package llvm

import (
	"testing"
)

func TestStringRecursiveType(t *testing.T) {
	ctx := NewContext()
	defer ctx.Dispose()
	s := ctx.StructCreateNamed("recursive")
	s.StructSetBody([]Type{s, s}, false)
	if str := s.String(); str != "%recursive: StructType(%recursive, %recursive)" {
		t.Errorf("incorrect string result %q", str)
	}
}
