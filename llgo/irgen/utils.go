//===- utils.go - misc utils ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
