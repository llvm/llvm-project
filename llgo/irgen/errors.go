//===- errors.go - IR generation for run-time panics ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for triggering run-time panics.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llvm/bindings/go/llvm"
)

const (
	// From go-runtime-error.c
	gccgoRuntimeErrorSLICE_INDEX_OUT_OF_BOUNDS  = 0
	gccgoRuntimeErrorARRAY_INDEX_OUT_OF_BOUNDS  = 1
	gccgoRuntimeErrorSTRING_INDEX_OUT_OF_BOUNDS = 2
	gccgoRuntimeErrorSLICE_SLICE_OUT_OF_BOUNDS  = 3
	gccgoRuntimeErrorARRAY_SLICE_OUT_OF_BOUNDS  = 4
	gccgoRuntimeErrorSTRING_SLICE_OUT_OF_BOUNDS = 5
	gccgoRuntimeErrorNIL_DEREFERENCE            = 6
	gccgoRuntimeErrorMAKE_SLICE_OUT_OF_BOUNDS   = 7
	gccgoRuntimeErrorMAKE_MAP_OUT_OF_BOUNDS     = 8
	gccgoRuntimeErrorMAKE_CHAN_OUT_OF_BOUNDS    = 9
	gccgoRuntimeErrorDIVISION_BY_ZERO           = 10
	gccgoRuntimeErrorCount                      = 11
)

func (fr *frame) setBranchWeightMetadata(br llvm.Value, trueweight, falseweight uint64) {
	mdprof := llvm.MDKindID("prof")

	mdnode := llvm.GlobalContext().MDNode([]llvm.Metadata{
		llvm.GlobalContext().MDString("branch_weights"),
		llvm.ConstInt(llvm.Int32Type(), trueweight, false).ConstantAsMetadata(),
		llvm.ConstInt(llvm.Int32Type(), falseweight, false).ConstantAsMetadata(),
	})

	br.SetMetadata(mdprof, mdnode)
}

func (fr *frame) condBrRuntimeError(cond llvm.Value, errcode uint64) {
	if cond.IsNull() {
		return
	}

	errorbb := fr.runtimeErrorBlocks[errcode]
	newbb := errorbb.C == nil
	if newbb {
		errorbb = llvm.AddBasicBlock(fr.function, "")
		fr.runtimeErrorBlocks[errcode] = errorbb
	}

	contbb := llvm.AddBasicBlock(fr.function, "")

	br := fr.builder.CreateCondBr(cond, errorbb, contbb)
	fr.setBranchWeightMetadata(br, 1, 1000)

	if newbb {
		fr.builder.SetInsertPointAtEnd(errorbb)
		fr.runtime.runtimeError.call(fr, llvm.ConstInt(llvm.Int32Type(), errcode, false))
		fr.builder.CreateUnreachable()
	}

	fr.builder.SetInsertPointAtEnd(contbb)
}
