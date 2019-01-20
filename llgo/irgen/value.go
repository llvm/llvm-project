//===- value.go - govalue and operations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the govalue type, which combines an LLVM value with its Go
// type, and implements various basic operations on govalues.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"fmt"
	"go/token"

	"llvm.org/llgo/third_party/gotools/go/exact"
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llvm/bindings/go/llvm"
)

// govalue contains an LLVM value and a Go type,
// representing the result of a Go expression.
type govalue struct {
	value llvm.Value
	typ   types.Type
}

func (v *govalue) String() string {
	return fmt.Sprintf("[llgo.govalue typ:%s value:%v]", v.typ, v.value)
}

// Create a new dynamic value from a (LLVM Value, Type) pair.
func newValue(v llvm.Value, t types.Type) *govalue {
	return &govalue{v, t}
}

// TODO(axw) remove this, use .typ directly
func (v *govalue) Type() types.Type {
	return v.typ
}

// newValueFromConst converts a constant value to an LLVM value.
func (fr *frame) newValueFromConst(v exact.Value, typ types.Type) *govalue {
	switch {
	case v == nil:
		llvmtyp := fr.types.ToLLVM(typ)
		return newValue(llvm.ConstNull(llvmtyp), typ)

	case isString(typ):
		if isUntyped(typ) {
			typ = types.Typ[types.String]
		}
		llvmtyp := fr.types.ToLLVM(typ)
		strval := exact.StringVal(v)
		strlen := len(strval)
		i8ptr := llvm.PointerType(llvm.Int8Type(), 0)
		var ptr llvm.Value
		if strlen > 0 {
			init := llvm.ConstString(strval, false)
			ptr = llvm.AddGlobal(fr.module.Module, init.Type(), "")
			ptr.SetInitializer(init)
			ptr.SetLinkage(llvm.InternalLinkage)
			ptr = llvm.ConstBitCast(ptr, i8ptr)
		} else {
			ptr = llvm.ConstNull(i8ptr)
		}
		len_ := llvm.ConstInt(fr.types.inttype, uint64(strlen), false)
		llvmvalue := llvm.Undef(llvmtyp)
		llvmvalue = llvm.ConstInsertValue(llvmvalue, ptr, []uint32{0})
		llvmvalue = llvm.ConstInsertValue(llvmvalue, len_, []uint32{1})
		return newValue(llvmvalue, typ)

	case isInteger(typ):
		if isUntyped(typ) {
			typ = types.Typ[types.Int]
		}
		llvmtyp := fr.types.ToLLVM(typ)
		var llvmvalue llvm.Value
		if isUnsigned(typ) {
			v, _ := exact.Uint64Val(v)
			llvmvalue = llvm.ConstInt(llvmtyp, v, false)
		} else {
			v, _ := exact.Int64Val(v)
			llvmvalue = llvm.ConstInt(llvmtyp, uint64(v), true)
		}
		return newValue(llvmvalue, typ)

	case isBoolean(typ):
		if isUntyped(typ) {
			typ = types.Typ[types.Bool]
		}
		return newValue(boolLLVMValue(exact.BoolVal(v)), typ)

	case isFloat(typ):
		if isUntyped(typ) {
			typ = types.Typ[types.Float64]
		}
		llvmtyp := fr.types.ToLLVM(typ)
		floatval, _ := exact.Float64Val(v)
		llvmvalue := llvm.ConstFloat(llvmtyp, floatval)
		return newValue(llvmvalue, typ)

	case typ == types.Typ[types.UnsafePointer]:
		llvmtyp := fr.types.ToLLVM(typ)
		v, _ := exact.Uint64Val(v)
		llvmvalue := llvm.ConstInt(fr.types.inttype, v, false)
		llvmvalue = llvm.ConstIntToPtr(llvmvalue, llvmtyp)
		return newValue(llvmvalue, typ)

	case isComplex(typ):
		if isUntyped(typ) {
			typ = types.Typ[types.Complex128]
		}
		llvmtyp := fr.types.ToLLVM(typ)
		floattyp := llvmtyp.StructElementTypes()[0]
		llvmvalue := llvm.ConstNull(llvmtyp)
		realv := exact.Real(v)
		imagv := exact.Imag(v)
		realfloatval, _ := exact.Float64Val(realv)
		imagfloatval, _ := exact.Float64Val(imagv)
		llvmre := llvm.ConstFloat(floattyp, realfloatval)
		llvmim := llvm.ConstFloat(floattyp, imagfloatval)
		llvmvalue = llvm.ConstInsertValue(llvmvalue, llvmre, []uint32{0})
		llvmvalue = llvm.ConstInsertValue(llvmvalue, llvmim, []uint32{1})
		return newValue(llvmvalue, typ)
	}

	// Special case for string -> [](byte|rune)
	if u, ok := typ.Underlying().(*types.Slice); ok && isInteger(u.Elem()) {
		if v.Kind() == exact.String {
			strval := fr.newValueFromConst(v, types.Typ[types.String])
			return fr.convert(strval, typ)
		}
	}

	panic(fmt.Sprintf("unhandled: t=%s(%T), v=%v(%T)", typ, typ, v, v))
}

func (fr *frame) binaryOp(lhs *govalue, op token.Token, rhs *govalue) *govalue {
	if op == token.NEQ {
		result := fr.binaryOp(lhs, token.EQL, rhs)
		return fr.unaryOp(result, token.NOT)
	}

	var result llvm.Value
	b := fr.builder

	switch typ := lhs.typ.Underlying().(type) {
	case *types.Struct:
		// TODO(axw) use runtime equality algorithm (will be suitably inlined).
		// For now, we use compare all fields unconditionally and bitwise AND
		// to avoid branching (i.e. so we don't create additional blocks).
		value := newValue(boolLLVMValue(true), types.Typ[types.Bool])
		for i := 0; i < typ.NumFields(); i++ {
			t := typ.Field(i).Type()
			lhs := newValue(b.CreateExtractValue(lhs.value, i, ""), t)
			rhs := newValue(b.CreateExtractValue(rhs.value, i, ""), t)
			value = fr.binaryOp(value, token.AND, fr.binaryOp(lhs, token.EQL, rhs))
		}
		return value

	case *types.Array:
		// TODO(pcc): as above.
		value := newValue(boolLLVMValue(true), types.Typ[types.Bool])
		t := typ.Elem()
		for i := int64(0); i < typ.Len(); i++ {
			lhs := newValue(b.CreateExtractValue(lhs.value, int(i), ""), t)
			rhs := newValue(b.CreateExtractValue(rhs.value, int(i), ""), t)
			value = fr.binaryOp(value, token.AND, fr.binaryOp(lhs, token.EQL, rhs))
		}
		return value

	case *types.Slice:
		// []T == nil or nil == []T
		lhsptr := b.CreateExtractValue(lhs.value, 0, "")
		rhsptr := b.CreateExtractValue(rhs.value, 0, "")
		isnil := b.CreateICmp(llvm.IntEQ, lhsptr, rhsptr, "")
		isnil = b.CreateZExt(isnil, llvm.Int8Type(), "")
		return newValue(isnil, types.Typ[types.Bool])

	case *types.Signature:
		// func == nil or nil == func
		isnil := b.CreateICmp(llvm.IntEQ, lhs.value, rhs.value, "")
		isnil = b.CreateZExt(isnil, llvm.Int8Type(), "")
		return newValue(isnil, types.Typ[types.Bool])

	case *types.Interface:
		return fr.compareInterfaces(lhs, rhs)
	}

	// Strings.
	if isString(lhs.typ) {
		if isString(rhs.typ) {
			switch op {
			case token.ADD:
				return fr.concatenateStrings(lhs, rhs)
			case token.EQL, token.LSS, token.GTR, token.LEQ, token.GEQ:
				return fr.compareStrings(lhs, rhs, op)
			default:
				panic(fmt.Sprint("Unimplemented operator: ", op))
			}
		}
		panic("unimplemented")
	}

	// Complex numbers.
	if isComplex(lhs.typ) {
		// XXX Should we represent complex numbers as vectors?
		lhsval := lhs.value
		rhsval := rhs.value
		a_ := b.CreateExtractValue(lhsval, 0, "")
		b_ := b.CreateExtractValue(lhsval, 1, "")
		c_ := b.CreateExtractValue(rhsval, 0, "")
		d_ := b.CreateExtractValue(rhsval, 1, "")
		switch op {
		case token.QUO:
			// (a+bi)/(c+di) = (ac+bd)/(c**2+d**2) + (bc-ad)/(c**2+d**2)i
			ac := b.CreateFMul(a_, c_, "")
			bd := b.CreateFMul(b_, d_, "")
			bc := b.CreateFMul(b_, c_, "")
			ad := b.CreateFMul(a_, d_, "")
			cpow2 := b.CreateFMul(c_, c_, "")
			dpow2 := b.CreateFMul(d_, d_, "")
			denom := b.CreateFAdd(cpow2, dpow2, "")
			realnumer := b.CreateFAdd(ac, bd, "")
			imagnumer := b.CreateFSub(bc, ad, "")
			real_ := b.CreateFDiv(realnumer, denom, "")
			imag_ := b.CreateFDiv(imagnumer, denom, "")
			lhsval = b.CreateInsertValue(lhsval, real_, 0, "")
			result = b.CreateInsertValue(lhsval, imag_, 1, "")
		case token.MUL:
			// (a+bi)(c+di) = (ac-bd)+(bc+ad)i
			ac := b.CreateFMul(a_, c_, "")
			bd := b.CreateFMul(b_, d_, "")
			bc := b.CreateFMul(b_, c_, "")
			ad := b.CreateFMul(a_, d_, "")
			real_ := b.CreateFSub(ac, bd, "")
			imag_ := b.CreateFAdd(bc, ad, "")
			lhsval = b.CreateInsertValue(lhsval, real_, 0, "")
			result = b.CreateInsertValue(lhsval, imag_, 1, "")
		case token.ADD:
			real_ := b.CreateFAdd(a_, c_, "")
			imag_ := b.CreateFAdd(b_, d_, "")
			lhsval = b.CreateInsertValue(lhsval, real_, 0, "")
			result = b.CreateInsertValue(lhsval, imag_, 1, "")
		case token.SUB:
			real_ := b.CreateFSub(a_, c_, "")
			imag_ := b.CreateFSub(b_, d_, "")
			lhsval = b.CreateInsertValue(lhsval, real_, 0, "")
			result = b.CreateInsertValue(lhsval, imag_, 1, "")
		case token.EQL:
			realeq := b.CreateFCmp(llvm.FloatOEQ, a_, c_, "")
			imageq := b.CreateFCmp(llvm.FloatOEQ, b_, d_, "")
			result = b.CreateAnd(realeq, imageq, "")
			result = b.CreateZExt(result, llvm.Int8Type(), "")
			return newValue(result, types.Typ[types.Bool])
		default:
			panic(fmt.Errorf("unhandled operator: %v", op))
		}
		return newValue(result, lhs.typ)
	}

	// Floats and integers.
	// TODO determine the NaN rules.

	switch op {
	case token.MUL:
		if isFloat(lhs.typ) {
			result = b.CreateFMul(lhs.value, rhs.value, "")
		} else {
			result = b.CreateMul(lhs.value, rhs.value, "")
		}
		return newValue(result, lhs.typ)
	case token.QUO:
		switch {
		case isFloat(lhs.typ):
			result = b.CreateFDiv(lhs.value, rhs.value, "")
		case !isUnsigned(lhs.typ):
			result = b.CreateSDiv(lhs.value, rhs.value, "")
		default:
			result = b.CreateUDiv(lhs.value, rhs.value, "")
		}
		return newValue(result, lhs.typ)
	case token.REM:
		switch {
		case isFloat(lhs.typ):
			result = b.CreateFRem(lhs.value, rhs.value, "")
		case !isUnsigned(lhs.typ):
			result = b.CreateSRem(lhs.value, rhs.value, "")
		default:
			result = b.CreateURem(lhs.value, rhs.value, "")
		}
		return newValue(result, lhs.typ)
	case token.ADD:
		if isFloat(lhs.typ) {
			result = b.CreateFAdd(lhs.value, rhs.value, "")
		} else {
			result = b.CreateAdd(lhs.value, rhs.value, "")
		}
		return newValue(result, lhs.typ)
	case token.SUB:
		if isFloat(lhs.typ) {
			result = b.CreateFSub(lhs.value, rhs.value, "")
		} else {
			result = b.CreateSub(lhs.value, rhs.value, "")
		}
		return newValue(result, lhs.typ)
	case token.SHL, token.SHR:
		return fr.shift(lhs, rhs, op)
	case token.EQL:
		if isFloat(lhs.typ) {
			result = b.CreateFCmp(llvm.FloatOEQ, lhs.value, rhs.value, "")
		} else {
			result = b.CreateICmp(llvm.IntEQ, lhs.value, rhs.value, "")
		}
		result = b.CreateZExt(result, llvm.Int8Type(), "")
		return newValue(result, types.Typ[types.Bool])
	case token.LSS:
		switch {
		case isFloat(lhs.typ):
			result = b.CreateFCmp(llvm.FloatOLT, lhs.value, rhs.value, "")
		case !isUnsigned(lhs.typ):
			result = b.CreateICmp(llvm.IntSLT, lhs.value, rhs.value, "")
		default:
			result = b.CreateICmp(llvm.IntULT, lhs.value, rhs.value, "")
		}
		result = b.CreateZExt(result, llvm.Int8Type(), "")
		return newValue(result, types.Typ[types.Bool])
	case token.LEQ:
		switch {
		case isFloat(lhs.typ):
			result = b.CreateFCmp(llvm.FloatOLE, lhs.value, rhs.value, "")
		case !isUnsigned(lhs.typ):
			result = b.CreateICmp(llvm.IntSLE, lhs.value, rhs.value, "")
		default:
			result = b.CreateICmp(llvm.IntULE, lhs.value, rhs.value, "")
		}
		result = b.CreateZExt(result, llvm.Int8Type(), "")
		return newValue(result, types.Typ[types.Bool])
	case token.GTR:
		switch {
		case isFloat(lhs.typ):
			result = b.CreateFCmp(llvm.FloatOGT, lhs.value, rhs.value, "")
		case !isUnsigned(lhs.typ):
			result = b.CreateICmp(llvm.IntSGT, lhs.value, rhs.value, "")
		default:
			result = b.CreateICmp(llvm.IntUGT, lhs.value, rhs.value, "")
		}
		result = b.CreateZExt(result, llvm.Int8Type(), "")
		return newValue(result, types.Typ[types.Bool])
	case token.GEQ:
		switch {
		case isFloat(lhs.typ):
			result = b.CreateFCmp(llvm.FloatOGE, lhs.value, rhs.value, "")
		case !isUnsigned(lhs.typ):
			result = b.CreateICmp(llvm.IntSGE, lhs.value, rhs.value, "")
		default:
			result = b.CreateICmp(llvm.IntUGE, lhs.value, rhs.value, "")
		}
		result = b.CreateZExt(result, llvm.Int8Type(), "")
		return newValue(result, types.Typ[types.Bool])
	case token.AND: // a & b
		result = b.CreateAnd(lhs.value, rhs.value, "")
		return newValue(result, lhs.typ)
	case token.AND_NOT: // a &^ b
		rhsval := rhs.value
		rhsval = b.CreateXor(rhsval, llvm.ConstAllOnes(rhsval.Type()), "")
		result = b.CreateAnd(lhs.value, rhsval, "")
		return newValue(result, lhs.typ)
	case token.OR: // a | b
		result = b.CreateOr(lhs.value, rhs.value, "")
		return newValue(result, lhs.typ)
	case token.XOR: // a ^ b
		result = b.CreateXor(lhs.value, rhs.value, "")
		return newValue(result, lhs.typ)
	default:
		panic(fmt.Sprint("Unimplemented operator: ", op))
	}
	panic("unreachable")
}

func (fr *frame) shift(lhs *govalue, rhs *govalue, op token.Token) *govalue {
	rhs = fr.convert(rhs, lhs.Type())
	lhsval := lhs.value
	bits := rhs.value
	unsigned := isUnsigned(lhs.Type())
	// Shifting >= width of lhs yields undefined behaviour, so we must select.
	max := llvm.ConstInt(bits.Type(), uint64(lhsval.Type().IntTypeWidth()-1), false)
	var result llvm.Value
	lessEqualWidth := fr.builder.CreateICmp(llvm.IntULE, bits, max, "")
	if !unsigned && op == token.SHR {
		bits := fr.builder.CreateSelect(lessEqualWidth, bits, max, "")
		result = fr.builder.CreateAShr(lhsval, bits, "")
	} else {
		if op == token.SHL {
			result = fr.builder.CreateShl(lhsval, bits, "")
		} else {
			result = fr.builder.CreateLShr(lhsval, bits, "")
		}
		zero := llvm.ConstNull(lhsval.Type())
		result = fr.builder.CreateSelect(lessEqualWidth, result, zero, "")
	}
	return newValue(result, lhs.typ)
}

func (fr *frame) unaryOp(v *govalue, op token.Token) *govalue {
	switch op {
	case token.SUB:
		var value llvm.Value
		if isComplex(v.typ) {
			realv := fr.builder.CreateExtractValue(v.value, 0, "")
			imagv := fr.builder.CreateExtractValue(v.value, 1, "")
			negzero := llvm.ConstFloatFromString(realv.Type(), "-0")
			realv = fr.builder.CreateFSub(negzero, realv, "")
			imagv = fr.builder.CreateFSub(negzero, imagv, "")
			value = llvm.Undef(v.value.Type())
			value = fr.builder.CreateInsertValue(value, realv, 0, "")
			value = fr.builder.CreateInsertValue(value, imagv, 1, "")
		} else if isFloat(v.typ) {
			negzero := llvm.ConstFloatFromString(fr.types.ToLLVM(v.Type()), "-0")
			value = fr.builder.CreateFSub(negzero, v.value, "")
		} else {
			value = fr.builder.CreateNeg(v.value, "")
		}
		return newValue(value, v.typ)
	case token.ADD:
		return v // No-op
	case token.NOT:
		value := fr.builder.CreateXor(v.value, boolLLVMValue(true), "")
		return newValue(value, v.typ)
	case token.XOR:
		lhs := v.value
		rhs := llvm.ConstAllOnes(lhs.Type())
		value := fr.builder.CreateXor(lhs, rhs, "")
		return newValue(value, v.typ)
	default:
		panic(fmt.Sprintf("Unhandled operator: %s", op))
	}
}

func (fr *frame) convert(v *govalue, dsttyp types.Type) *govalue {
	b := fr.builder

	// If it's a stack allocated value, we'll want to compare the
	// value type, not the pointer type.
	srctyp := v.typ

	// Get the underlying type, if any.
	origdsttyp := dsttyp
	dsttyp = dsttyp.Underlying()
	srctyp = srctyp.Underlying()

	// Identical (underlying) types? Just swap in the destination type.
	if types.Identical(srctyp, dsttyp) {
		return newValue(v.value, origdsttyp)
	}

	// Both pointer types with identical underlying types? Same as above.
	if srctyp, ok := srctyp.(*types.Pointer); ok {
		if dsttyp, ok := dsttyp.(*types.Pointer); ok {
			srctyp := srctyp.Elem().Underlying()
			dsttyp := dsttyp.Elem().Underlying()
			if types.Identical(srctyp, dsttyp) {
				return newValue(v.value, origdsttyp)
			}
		}
	}

	// string ->
	if isString(srctyp) {
		// (untyped) string -> string
		// XXX should untyped strings be able to escape go/types?
		if isString(dsttyp) {
			return newValue(v.value, origdsttyp)
		}

		// string -> []byte
		if isSlice(dsttyp, types.Byte) {
			sliceValue := fr.runtime.stringToByteArray.callOnly(fr, v.value)[0]
			return newValue(sliceValue, origdsttyp)
		}

		// string -> []rune
		if isSlice(dsttyp, types.Rune) {
			return fr.stringToRuneSlice(v)
		}
	}

	// []byte -> string
	if isSlice(srctyp, types.Byte) && isString(dsttyp) {
		data := fr.builder.CreateExtractValue(v.value, 0, "")
		len := fr.builder.CreateExtractValue(v.value, 1, "")
		stringValue := fr.runtime.byteArrayToString.callOnly(fr, data, len)[0]
		return newValue(stringValue, dsttyp)
	}

	// []rune -> string
	if isSlice(srctyp, types.Rune) && isString(dsttyp) {
		return fr.runeSliceToString(v)
	}

	// rune -> string
	if isString(dsttyp) && isInteger(srctyp) {
		return fr.runeToString(v)
	}

	// Unsafe pointer conversions.
	llvm_type := fr.types.ToLLVM(dsttyp)
	if dsttyp == types.Typ[types.UnsafePointer] { // X -> unsafe.Pointer
		if _, isptr := srctyp.(*types.Pointer); isptr {
			return newValue(v.value, origdsttyp)
		} else if srctyp == types.Typ[types.Uintptr] {
			value := b.CreateIntToPtr(v.value, llvm_type, "")
			return newValue(value, origdsttyp)
		}
	} else if srctyp == types.Typ[types.UnsafePointer] { // unsafe.Pointer -> X
		if _, isptr := dsttyp.(*types.Pointer); isptr {
			return newValue(v.value, origdsttyp)
		} else if dsttyp == types.Typ[types.Uintptr] {
			value := b.CreatePtrToInt(v.value, llvm_type, "")
			return newValue(value, origdsttyp)
		}
	}

	lv := v.value
	srcType := lv.Type()
	switch srcType.TypeKind() {
	case llvm.IntegerTypeKind:
		switch llvm_type.TypeKind() {
		case llvm.IntegerTypeKind:
			srcBits := srcType.IntTypeWidth()
			dstBits := llvm_type.IntTypeWidth()
			delta := srcBits - dstBits
			switch {
			case delta < 0:
				if !isUnsigned(srctyp) {
					lv = b.CreateSExt(lv, llvm_type, "")
				} else {
					lv = b.CreateZExt(lv, llvm_type, "")
				}
			case delta > 0:
				lv = b.CreateTrunc(lv, llvm_type, "")
			}
			return newValue(lv, origdsttyp)
		case llvm.FloatTypeKind, llvm.DoubleTypeKind:
			if !isUnsigned(v.Type()) {
				lv = b.CreateSIToFP(lv, llvm_type, "")
			} else {
				lv = b.CreateUIToFP(lv, llvm_type, "")
			}
			return newValue(lv, origdsttyp)
		}
	case llvm.DoubleTypeKind:
		switch llvm_type.TypeKind() {
		case llvm.FloatTypeKind:
			lv = b.CreateFPTrunc(lv, llvm_type, "")
			return newValue(lv, origdsttyp)
		case llvm.IntegerTypeKind:
			if !isUnsigned(dsttyp) {
				lv = b.CreateFPToSI(lv, llvm_type, "")
			} else {
				lv = b.CreateFPToUI(lv, llvm_type, "")
			}
			return newValue(lv, origdsttyp)
		}
	case llvm.FloatTypeKind:
		switch llvm_type.TypeKind() {
		case llvm.DoubleTypeKind:
			lv = b.CreateFPExt(lv, llvm_type, "")
			return newValue(lv, origdsttyp)
		case llvm.IntegerTypeKind:
			if !isUnsigned(dsttyp) {
				lv = b.CreateFPToSI(lv, llvm_type, "")
			} else {
				lv = b.CreateFPToUI(lv, llvm_type, "")
			}
			return newValue(lv, origdsttyp)
		}
	}

	// Complex -> complex. Complexes are only convertible to other
	// complexes, contant conversions aside. So we can just check the
	// source type here; given that the types are not identical
	// (checked above), we can assume the destination type is the alternate
	// complex type.
	if isComplex(srctyp) {
		var fpcast func(llvm.Builder, llvm.Value, llvm.Type, string) llvm.Value
		var fptype llvm.Type
		if srctyp == types.Typ[types.Complex64] {
			fpcast = (llvm.Builder).CreateFPExt
			fptype = llvm.DoubleType()
		} else {
			fpcast = (llvm.Builder).CreateFPTrunc
			fptype = llvm.FloatType()
		}
		if fpcast != nil {
			realv := b.CreateExtractValue(lv, 0, "")
			imagv := b.CreateExtractValue(lv, 1, "")
			realv = fpcast(b, realv, fptype, "")
			imagv = fpcast(b, imagv, fptype, "")
			lv = llvm.Undef(fr.types.ToLLVM(dsttyp))
			lv = b.CreateInsertValue(lv, realv, 0, "")
			lv = b.CreateInsertValue(lv, imagv, 1, "")
			return newValue(lv, origdsttyp)
		}
	}
	panic(fmt.Sprintf("unimplemented conversion: %s (%s) -> %s", v.typ, lv.Type(), origdsttyp))
}

// extractRealValue extracts the real component of a complex number.
func (fr *frame) extractRealValue(v *govalue) *govalue {
	component := fr.builder.CreateExtractValue(v.value, 0, "")
	if component.Type().TypeKind() == llvm.FloatTypeKind {
		return newValue(component, types.Typ[types.Float32])
	}
	return newValue(component, types.Typ[types.Float64])
}

// extractRealValue extracts the imaginary component of a complex number.
func (fr *frame) extractImagValue(v *govalue) *govalue {
	component := fr.builder.CreateExtractValue(v.value, 1, "")
	if component.Type().TypeKind() == llvm.FloatTypeKind {
		return newValue(component, types.Typ[types.Float32])
	}
	return newValue(component, types.Typ[types.Float64])
}

func boolLLVMValue(v bool) (lv llvm.Value) {
	if v {
		return llvm.ConstInt(llvm.Int8Type(), 1, false)
	}
	return llvm.ConstNull(llvm.Int8Type())
}
