//===- slice.go - IR generation for slices --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for slices.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llvm/bindings/go/llvm"
)

// makeSlice allocates a new slice with the optional length and capacity,
// initialising its contents to their zero values.
func (fr *frame) makeSlice(sliceType types.Type, length, capacity *govalue) *govalue {
	length = fr.convert(length, types.Typ[types.Uintptr])
	capacity = fr.convert(capacity, types.Typ[types.Uintptr])
	runtimeType := fr.types.ToRuntime(sliceType)
	llslice := fr.runtime.makeSlice.call(fr, runtimeType, length.value, capacity.value)
	return newValue(llslice[0], sliceType)
}

func (fr *frame) slice(x llvm.Value, xtyp types.Type, low, high, max llvm.Value) llvm.Value {
	if !low.IsNil() {
		low = fr.createZExtOrTrunc(low, fr.types.inttype, "")
	} else {
		low = llvm.ConstNull(fr.types.inttype)
	}
	if !high.IsNil() {
		high = fr.createZExtOrTrunc(high, fr.types.inttype, "")
	}
	if !max.IsNil() {
		max = fr.createZExtOrTrunc(max, fr.types.inttype, "")
	}

	var arrayptr, arraylen, arraycap llvm.Value
	var elemtyp types.Type
	var errcode uint64
	switch typ := xtyp.Underlying().(type) {
	case *types.Pointer: // *array
		errcode = gccgoRuntimeErrorARRAY_SLICE_OUT_OF_BOUNDS
		arraytyp := typ.Elem().Underlying().(*types.Array)
		elemtyp = arraytyp.Elem()
		arrayptr = x
		arrayptr = fr.builder.CreateBitCast(arrayptr, llvm.PointerType(llvm.Int8Type(), 0), "")
		arraylen = llvm.ConstInt(fr.llvmtypes.inttype, uint64(arraytyp.Len()), false)
		arraycap = arraylen
	case *types.Slice:
		errcode = gccgoRuntimeErrorSLICE_SLICE_OUT_OF_BOUNDS
		elemtyp = typ.Elem()
		arrayptr = fr.builder.CreateExtractValue(x, 0, "")
		arraylen = fr.builder.CreateExtractValue(x, 1, "")
		arraycap = fr.builder.CreateExtractValue(x, 2, "")
	case *types.Basic:
		if high.IsNil() {
			high = llvm.ConstAllOnes(fr.types.inttype) // -1
		}
		result := fr.runtime.stringSlice.call(fr, x, low, high)
		return result[0]
	default:
		panic("unimplemented")
	}
	if high.IsNil() {
		high = arraylen
	}
	if max.IsNil() {
		max = arraycap
	}

	// Bounds checking: 0 <= low <= high <= max <= cap
	zero := llvm.ConstNull(fr.types.inttype)
	l0 := fr.builder.CreateICmp(llvm.IntSLT, low, zero, "")
	hl := fr.builder.CreateICmp(llvm.IntSLT, high, low, "")
	mh := fr.builder.CreateICmp(llvm.IntSLT, max, high, "")
	cm := fr.builder.CreateICmp(llvm.IntSLT, arraycap, max, "")

	cond := fr.builder.CreateOr(l0, hl, "")
	cond = fr.builder.CreateOr(cond, mh, "")
	cond = fr.builder.CreateOr(cond, cm, "")

	fr.condBrRuntimeError(cond, errcode)

	slicelen := fr.builder.CreateSub(high, low, "")
	slicecap := fr.builder.CreateSub(max, low, "")

	elemsize := llvm.ConstInt(fr.llvmtypes.inttype, uint64(fr.llvmtypes.Sizeof(elemtyp)), false)
	offset := fr.builder.CreateMul(low, elemsize, "")

	sliceptr := fr.builder.CreateInBoundsGEP(arrayptr, []llvm.Value{offset}, "")

	llslicetyp := fr.llvmtypes.sliceBackendType().ToLLVM(fr.llvmtypes.ctx)
	sliceValue := llvm.Undef(llslicetyp)
	sliceValue = fr.builder.CreateInsertValue(sliceValue, sliceptr, 0, "")
	sliceValue = fr.builder.CreateInsertValue(sliceValue, slicelen, 1, "")
	sliceValue = fr.builder.CreateInsertValue(sliceValue, slicecap, 2, "")

	return sliceValue
}
