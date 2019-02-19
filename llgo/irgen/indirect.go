//===- indirect.go - IR generation for thunks -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for thunks required by the "defer" and
// "go" builtins.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llgo/third_party/gotools/go/ssa"
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llvm/bindings/go/llvm"
)

// createThunk creates a thunk from a
// given function and arguments, suitable for use with
// "defer" and "go".
func (fr *frame) createThunk(call ssa.CallInstruction) (thunk llvm.Value, arg llvm.Value) {
	seenarg := make(map[ssa.Value]bool)
	var args []ssa.Value
	var argtypes []*types.Var

	packArg := func(arg ssa.Value) {
		switch arg.(type) {
		case *ssa.Builtin, *ssa.Function, *ssa.Const, *ssa.Global:
			// Do nothing: we can generate these in the thunk
		default:
			if !seenarg[arg] {
				seenarg[arg] = true
				args = append(args, arg)
				field := types.NewField(0, nil, "_", arg.Type(), true)
				argtypes = append(argtypes, field)
			}
		}
	}

	packArg(call.Common().Value)
	for _, arg := range call.Common().Args {
		packArg(arg)
	}

	var isRecoverCall bool
	i8ptr := llvm.PointerType(llvm.Int8Type(), 0)
	var structllptr llvm.Type
	if len(args) == 0 {
		if builtin, ok := call.Common().Value.(*ssa.Builtin); ok {
			isRecoverCall = builtin.Name() == "recover"
		}
		if isRecoverCall {
			// When creating a thunk for recover(), we must pass fr.canRecover.
			arg = fr.builder.CreateZExt(fr.canRecover, fr.target.IntPtrType(), "")
			arg = fr.builder.CreateIntToPtr(arg, i8ptr, "")
		} else {
			arg = llvm.ConstPointerNull(i8ptr)
		}
	} else {
		structtype := types.NewStruct(argtypes, nil)
		arg = fr.createTypeMalloc(structtype)
		structllptr = arg.Type()
		for i, ssaarg := range args {
			argptr := fr.builder.CreateStructGEP(arg, i, "")
			fr.builder.CreateStore(fr.llvmvalue(ssaarg), argptr)
		}
		arg = fr.builder.CreateBitCast(arg, i8ptr, "")
	}

	thunkfntype := llvm.FunctionType(llvm.VoidType(), []llvm.Type{i8ptr}, false)
	thunkfn := llvm.AddFunction(fr.module.Module, "", thunkfntype)
	thunkfn.SetLinkage(llvm.InternalLinkage)
	fr.addCommonFunctionAttrs(thunkfn)

	thunkfr := newFrame(fr.unit, thunkfn)
	defer thunkfr.dispose()

	prologuebb := llvm.AddBasicBlock(thunkfn, "prologue")
	thunkfr.builder.SetInsertPointAtEnd(prologuebb)

	if isRecoverCall {
		thunkarg := thunkfn.Param(0)
		thunkarg = thunkfr.builder.CreatePtrToInt(thunkarg, fr.target.IntPtrType(), "")
		thunkfr.canRecover = thunkfr.builder.CreateTrunc(thunkarg, llvm.Int1Type(), "")
	} else if len(args) > 0 {
		thunkarg := thunkfn.Param(0)
		thunkarg = thunkfr.builder.CreateBitCast(thunkarg, structllptr, "")
		for i, ssaarg := range args {
			thunkargptr := thunkfr.builder.CreateStructGEP(thunkarg, i, "")
			thunkarg := thunkfr.builder.CreateLoad(thunkargptr, "")
			thunkfr.env[ssaarg] = newValue(thunkarg, ssaarg.Type())
		}
	}

	_, isDefer := call.(*ssa.Defer)

	entrybb := llvm.AddBasicBlock(thunkfn, "entry")
	br := thunkfr.builder.CreateBr(entrybb)
	thunkfr.allocaBuilder.SetInsertPointBefore(br)

	thunkfr.builder.SetInsertPointAtEnd(entrybb)
	var exitbb llvm.BasicBlock
	if isDefer {
		exitbb = llvm.AddBasicBlock(thunkfn, "exit")
		thunkfr.runtime.setDeferRetaddr.call(thunkfr, llvm.BlockAddress(thunkfn, exitbb))
	}
	if isDefer && isRecoverCall {
		thunkfr.callRecover(true)
	} else {
		thunkfr.callInstruction(call)
	}
	if isDefer {
		thunkfr.builder.CreateBr(exitbb)
		thunkfr.builder.SetInsertPointAtEnd(exitbb)
	}
	thunkfr.builder.CreateRetVoid()

	thunk = fr.builder.CreateBitCast(thunkfn, i8ptr, "")
	return
}
