//===- builtins.go - IR generation for builtins ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for the built-in functions.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llvm/bindings/go/llvm"
)

func (fr *frame) callCap(arg *govalue) *govalue {
	var v llvm.Value
	switch typ := arg.Type().Underlying().(type) {
	case *types.Array:
		v = llvm.ConstInt(fr.llvmtypes.inttype, uint64(typ.Len()), false)
	case *types.Pointer:
		atyp := typ.Elem().Underlying().(*types.Array)
		v = llvm.ConstInt(fr.llvmtypes.inttype, uint64(atyp.Len()), false)
	case *types.Slice:
		v = fr.builder.CreateExtractValue(arg.value, 2, "")
	case *types.Chan:
		v = fr.runtime.chanCap.call(fr, arg.value)[0]
	}
	return newValue(v, types.Typ[types.Int])
}

func (fr *frame) callLen(arg *govalue) *govalue {
	var lenvalue llvm.Value
	switch typ := arg.Type().Underlying().(type) {
	case *types.Array:
		lenvalue = llvm.ConstInt(fr.llvmtypes.inttype, uint64(typ.Len()), false)
	case *types.Pointer:
		atyp := typ.Elem().Underlying().(*types.Array)
		lenvalue = llvm.ConstInt(fr.llvmtypes.inttype, uint64(atyp.Len()), false)
	case *types.Slice:
		lenvalue = fr.builder.CreateExtractValue(arg.value, 1, "")
	case *types.Map:
		lenvalue = fr.runtime.mapLen.call(fr, arg.value)[0]
	case *types.Basic:
		if isString(typ) {
			lenvalue = fr.builder.CreateExtractValue(arg.value, 1, "")
		}
	case *types.Chan:
		lenvalue = fr.runtime.chanLen.call(fr, arg.value)[0]
	}
	return newValue(lenvalue, types.Typ[types.Int])
}

// callAppend takes two slices of the same type, and yields
// the result of appending the second to the first.
func (fr *frame) callAppend(a, b *govalue) *govalue {
	bptr := fr.builder.CreateExtractValue(b.value, 0, "")
	blen := fr.builder.CreateExtractValue(b.value, 1, "")
	elemsizeInt64 := fr.types.Sizeof(a.Type().Underlying().(*types.Slice).Elem())
	elemsize := llvm.ConstInt(fr.target.IntPtrType(), uint64(elemsizeInt64), false)
	result := fr.runtime.append.call(fr, a.value, bptr, blen, elemsize)[0]
	return newValue(result, a.Type())
}

// callCopy takes two slices a and b of the same type, and
// yields the result of calling "copy(a, b)".
func (fr *frame) callCopy(dest, source *govalue) *govalue {
	aptr := fr.builder.CreateExtractValue(dest.value, 0, "")
	alen := fr.builder.CreateExtractValue(dest.value, 1, "")
	bptr := fr.builder.CreateExtractValue(source.value, 0, "")
	blen := fr.builder.CreateExtractValue(source.value, 1, "")
	aless := fr.builder.CreateICmp(llvm.IntULT, alen, blen, "")
	minlen := fr.builder.CreateSelect(aless, alen, blen, "")
	elemsizeInt64 := fr.types.Sizeof(dest.Type().Underlying().(*types.Slice).Elem())
	elemsize := llvm.ConstInt(fr.types.inttype, uint64(elemsizeInt64), false)
	bytes := fr.builder.CreateMul(minlen, elemsize, "")
	fr.runtime.copy.call(fr, aptr, bptr, bytes)
	return newValue(minlen, types.Typ[types.Int])
}

func (fr *frame) callRecover(isDeferredRecover bool) *govalue {
	startbb := fr.builder.GetInsertBlock()
	recoverbb := llvm.AddBasicBlock(fr.function, "")
	contbb := llvm.AddBasicBlock(fr.function, "")
	canRecover := fr.builder.CreateTrunc(fr.canRecover, llvm.Int1Type(), "")
	fr.builder.CreateCondBr(canRecover, recoverbb, contbb)

	fr.builder.SetInsertPointAtEnd(recoverbb)
	var recovered llvm.Value
	if isDeferredRecover {
		recovered = fr.runtime.deferredRecover.call(fr)[0]
	} else {
		recovered = fr.runtime.recover.call(fr)[0]
	}
	recoverbb = fr.builder.GetInsertBlock()
	fr.builder.CreateBr(contbb)

	fr.builder.SetInsertPointAtEnd(contbb)
	eface := types.NewInterface(nil, nil)
	llv := fr.builder.CreatePHI(fr.types.ToLLVM(eface), "")
	llv.AddIncoming(
		[]llvm.Value{llvm.ConstNull(llv.Type()), recovered},
		[]llvm.BasicBlock{startbb, recoverbb},
	)
	return newValue(llv, eface)
}

func (fr *frame) callPanic(arg *govalue, term bool) {
	fr.runtime.panic.call(fr, arg.value)
	if term {
		fr.builder.CreateUnreachable()
	}
}
