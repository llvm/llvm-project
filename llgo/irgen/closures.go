//===- closures.go - IR generation for closures ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for closures.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llgo/third_party/gotools/go/types"
)

// makeClosure creates a closure from a function pointer and
// a set of bindings. The bindings are addresses of captured
// variables.
func (fr *frame) makeClosure(fn *govalue, bindings []*govalue) *govalue {
	govalues := append([]*govalue{fn}, bindings...)
	fields := make([]*types.Var, len(govalues))
	for i, v := range govalues {
		field := types.NewField(0, nil, "_", v.Type(), false)
		fields[i] = field
	}
	block := fr.createTypeMalloc(types.NewStruct(fields, nil))
	for i, v := range govalues {
		addressPtr := fr.builder.CreateStructGEP(block, i, "")
		fr.builder.CreateStore(v.value, addressPtr)
	}
	closure := fr.builder.CreateBitCast(block, fn.value.Type(), "")
	return newValue(closure, fn.Type())
}
