//===- call.go - IR generation for calls ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for calls.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llvm/bindings/go/llvm"
)

// createCall emits the code for a function call,
// taking into account receivers, and panic/defer.
func (fr *frame) createCall(fn *govalue, chain llvm.Value, argValues []*govalue) []*govalue {
	fntyp := fn.Type().Underlying().(*types.Signature)
	typinfo := fr.types.getSignatureInfo(fntyp)

	args := make([]llvm.Value, len(argValues))
	for i, arg := range argValues {
		args[i] = arg.value
	}
	var results []llvm.Value
	if fr.unwindBlock.IsNil() {
		results = typinfo.call(fr.types.ctx, fr.allocaBuilder, fr.builder, fn.value, chain, args)
	} else {
		contbb := llvm.AddBasicBlock(fr.function, "")
		results = typinfo.invoke(fr.types.ctx, fr.allocaBuilder, fr.builder, fn.value, chain, args, contbb, fr.unwindBlock)
	}

	resultValues := make([]*govalue, len(results))
	for i, res := range results {
		resultValues[i] = newValue(res, fntyp.Results().At(i).Type())
	}
	return resultValues
}
