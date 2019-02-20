//===- utils.go - misc utils ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements misellaneous utilities for IR generation.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llvm/bindings/go/llvm"
)

func (fr *frame) loadOrNull(cond, ptr llvm.Value, ty types.Type) *govalue {
	startbb := fr.builder.GetInsertBlock()
	loadbb := llvm.AddBasicBlock(fr.function, "")
	contbb := llvm.AddBasicBlock(fr.function, "")
	fr.builder.CreateCondBr(cond, loadbb, contbb)

	fr.builder.SetInsertPointAtEnd(loadbb)
	llty := fr.types.ToLLVM(ty)
	typedptr := fr.builder.CreateBitCast(ptr, llvm.PointerType(llty, 0), "")
	loadedval := fr.builder.CreateLoad(typedptr, "")
	fr.builder.CreateBr(contbb)

	fr.builder.SetInsertPointAtEnd(contbb)
	llv := fr.builder.CreatePHI(llty, "")
	llv.AddIncoming(
		[]llvm.Value{llvm.ConstNull(llty), loadedval},
		[]llvm.BasicBlock{startbb, loadbb},
	)
	return newValue(llv, ty)
}
