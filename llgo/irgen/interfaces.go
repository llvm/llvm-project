//===- interfaces.go - IR generation for interfaces -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for dealing with interface values.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llvm/bindings/go/llvm"
)

// interfaceMethod returns a function and receiver pointer for the specified
// interface and method pair.
func (fr *frame) interfaceMethod(lliface llvm.Value, ifacety types.Type, method *types.Func) (fn, recv *govalue) {
	llitab := fr.builder.CreateExtractValue(lliface, 0, "")
	recv = newValue(fr.builder.CreateExtractValue(lliface, 1, ""), types.Typ[types.UnsafePointer])
	methodset := fr.types.MethodSet(ifacety)
	// TODO(axw) cache ordered method index
	index := -1
	for i, m := range orderedMethodSet(methodset) {
		if m.Obj() == method {
			index = i
			break
		}
	}
	if index == -1 {
		panic("could not find method index")
	}
	llitab = fr.builder.CreateBitCast(llitab, llvm.PointerType(llvm.PointerType(llvm.Int8Type(), 0), 0), "")
	// Skip runtime type pointer.
	llifnptr := fr.builder.CreateGEP(llitab, []llvm.Value{
		llvm.ConstInt(llvm.Int32Type(), uint64(index+1), false),
	}, "")

	llifn := fr.builder.CreateLoad(llifnptr, "")
	// Replace receiver type with unsafe.Pointer.
	recvparam := types.NewParam(0, nil, "", types.Typ[types.UnsafePointer])
	sig := method.Type().(*types.Signature)
	sig = types.NewSignature(nil, recvparam, sig.Params(), sig.Results(), sig.Variadic())
	fn = newValue(llifn, sig)
	return
}

// compareInterfaces emits code to compare two interfaces for
// equality.
func (fr *frame) compareInterfaces(a, b *govalue) *govalue {
	aNull := a.value.IsNull()
	bNull := b.value.IsNull()
	if aNull && bNull {
		return newValue(boolLLVMValue(true), types.Typ[types.Bool])
	}

	compare := fr.runtime.emptyInterfaceCompare
	aI := a.Type().Underlying().(*types.Interface).NumMethods() > 0
	bI := b.Type().Underlying().(*types.Interface).NumMethods() > 0
	switch {
	case aI && bI:
		compare = fr.runtime.interfaceCompare
	case aI:
		a = fr.convertI2E(a)
	case bI:
		b = fr.convertI2E(b)
	}

	result := compare.call(fr, a.value, b.value)[0]
	result = fr.builder.CreateIsNull(result, "")
	result = fr.builder.CreateZExt(result, llvm.Int8Type(), "")
	return newValue(result, types.Typ[types.Bool])
}

func (fr *frame) makeInterface(llv llvm.Value, vty types.Type, iface types.Type) *govalue {
	if _, ok := vty.Underlying().(*types.Pointer); !ok {
		ptr := fr.createTypeMalloc(vty)
		fr.builder.CreateStore(llv, ptr)
		llv = ptr
	}
	return fr.makeInterfaceFromPointer(llv, vty, iface)
}

func (fr *frame) makeInterfaceFromPointer(vptr llvm.Value, vty types.Type, iface types.Type) *govalue {
	i8ptr := llvm.PointerType(llvm.Int8Type(), 0)
	llv := fr.builder.CreateBitCast(vptr, i8ptr, "")
	value := llvm.Undef(fr.types.ToLLVM(iface))
	itab := fr.types.getItabPointer(vty, iface.Underlying().(*types.Interface))
	value = fr.builder.CreateInsertValue(value, itab, 0, "")
	value = fr.builder.CreateInsertValue(value, llv, 1, "")
	return newValue(value, iface)
}

// Reads the type descriptor from the given interface type.
func (fr *frame) getInterfaceTypeDescriptor(v *govalue) llvm.Value {
	isempty := v.Type().Underlying().(*types.Interface).NumMethods() == 0
	itab := fr.builder.CreateExtractValue(v.value, 0, "")
	if isempty {
		return itab
	} else {
		itabnonnull := fr.builder.CreateIsNotNull(itab, "")
		return fr.loadOrNull(itabnonnull, itab, types.Typ[types.UnsafePointer]).value
	}
}

// Reads the value from the given interface type, assuming that the
// interface holds a value of the correct type.
func (fr *frame) getInterfaceValue(v *govalue, ty types.Type) *govalue {
	val := fr.builder.CreateExtractValue(v.value, 1, "")
	if _, ok := ty.Underlying().(*types.Pointer); !ok {
		typedval := fr.builder.CreateBitCast(val, llvm.PointerType(fr.types.ToLLVM(ty), 0), "")
		val = fr.builder.CreateLoad(typedval, "")
	}
	return newValue(val, ty)
}

// If cond is true, reads the value from the given interface type, otherwise
// returns a nil value.
func (fr *frame) getInterfaceValueOrNull(cond llvm.Value, v *govalue, ty types.Type) *govalue {
	val := fr.builder.CreateExtractValue(v.value, 1, "")
	if _, ok := ty.Underlying().(*types.Pointer); ok {
		val = fr.builder.CreateSelect(cond, val, llvm.ConstNull(val.Type()), "")
	} else {
		val = fr.loadOrNull(cond, val, ty).value
	}
	return newValue(val, ty)
}

func (fr *frame) interfaceTypeCheck(val *govalue, ty types.Type) (v *govalue, okval *govalue) {
	tytd := fr.types.ToRuntime(ty)
	if _, ok := ty.Underlying().(*types.Interface); ok {
		var result []llvm.Value
		if val.Type().Underlying().(*types.Interface).NumMethods() > 0 {
			result = fr.runtime.ifaceI2I2.call(fr, tytd, val.value)
		} else {
			result = fr.runtime.ifaceE2I2.call(fr, tytd, val.value)
		}
		v = newValue(result[0], ty)
		okval = newValue(result[1], types.Typ[types.Bool])
	} else {
		valtd := fr.getInterfaceTypeDescriptor(val)
		tyequal := fr.runtime.typeDescriptorsEqual.call(fr, valtd, tytd)[0]
		okval = newValue(tyequal, types.Typ[types.Bool])
		tyequal = fr.builder.CreateTrunc(tyequal, llvm.Int1Type(), "")

		v = fr.getInterfaceValueOrNull(tyequal, val, ty)
	}
	return
}

func (fr *frame) interfaceTypeAssert(val *govalue, ty types.Type) *govalue {
	if _, ok := ty.Underlying().(*types.Interface); ok {
		return fr.changeInterface(val, ty, true)
	} else {
		valtytd := fr.types.ToRuntime(val.Type())
		valtd := fr.getInterfaceTypeDescriptor(val)
		tytd := fr.types.ToRuntime(ty)
		fr.runtime.checkInterfaceType.call(fr, valtd, tytd, valtytd)

		return fr.getInterfaceValue(val, ty)
	}
}

// convertI2E converts a non-empty interface value to an empty interface.
func (fr *frame) convertI2E(v *govalue) *govalue {
	td := fr.getInterfaceTypeDescriptor(v)
	val := fr.builder.CreateExtractValue(v.value, 1, "")

	typ := types.NewInterface(nil, nil)
	intf := llvm.Undef(fr.types.ToLLVM(typ))
	intf = fr.builder.CreateInsertValue(intf, td, 0, "")
	intf = fr.builder.CreateInsertValue(intf, val, 1, "")
	return newValue(intf, typ)
}

func (fr *frame) changeInterface(v *govalue, ty types.Type, assert bool) *govalue {
	td := fr.getInterfaceTypeDescriptor(v)
	tytd := fr.types.ToRuntime(ty)
	var itab llvm.Value
	if assert {
		itab = fr.runtime.assertInterface.call(fr, tytd, td)[0]
	} else {
		itab = fr.runtime.convertInterface.call(fr, tytd, td)[0]
	}
	val := fr.builder.CreateExtractValue(v.value, 1, "")

	intf := llvm.Undef(fr.types.ToLLVM(ty))
	intf = fr.builder.CreateInsertValue(intf, itab, 0, "")
	intf = fr.builder.CreateInsertValue(intf, val, 1, "")
	return newValue(intf, ty)
}
