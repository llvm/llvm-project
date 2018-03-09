//===- println.go - IR generation for print and println -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for the print and println built-in
// functions.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"fmt"

	"llvm.org/llgo/third_party/gotools/go/types"
)

func (fr *frame) printValues(println_ bool, values ...*govalue) {
	for i, value := range values {
		llvm_value := value.value

		typ := value.Type().Underlying()
		if name, isname := typ.(*types.Named); isname {
			typ = name.Underlying()
		}

		if println_ && i > 0 {
			fr.runtime.printSpace.call(fr)
		}
		switch typ := typ.(type) {
		case *types.Basic:
			switch typ.Kind() {
			case types.Uint8, types.Uint16, types.Uint32, types.Uintptr, types.Uint, types.Uint64:
				i64 := fr.llvmtypes.ctx.Int64Type()
				zext := fr.builder.CreateZExt(llvm_value, i64, "")
				fr.runtime.printUint64.call(fr, zext)

			case types.Int, types.Int8, types.Int16, types.Int32, types.Int64:
				i64 := fr.llvmtypes.ctx.Int64Type()
				sext := fr.builder.CreateSExt(llvm_value, i64, "")
				fr.runtime.printInt64.call(fr, sext)

			case types.Float32:
				llvm_value = fr.builder.CreateFPExt(llvm_value, fr.llvmtypes.ctx.DoubleType(), "")
				fallthrough
			case types.Float64:
				fr.runtime.printDouble.call(fr, llvm_value)

			case types.Complex64:
				llvm_value = fr.convert(value, types.Typ[types.Complex128]).value
				fallthrough
			case types.Complex128:
				fr.runtime.printComplex.call(fr, llvm_value)

			case types.String, types.UntypedString:
				fr.runtime.printString.call(fr, llvm_value)

			case types.Bool:
				fr.runtime.printBool.call(fr, llvm_value)

			case types.UnsafePointer:
				fr.runtime.printPointer.call(fr, llvm_value)

			default:
				panic(fmt.Sprint("Unhandled Basic Kind: ", typ.Kind))
			}

		case *types.Interface:
			if typ.Empty() {
				fr.runtime.printEmptyInterface.call(fr, llvm_value)
			} else {
				fr.runtime.printInterface.call(fr, llvm_value)
			}

		case *types.Slice:
			fr.runtime.printSlice.call(fr, llvm_value)

		case *types.Pointer, *types.Map, *types.Chan, *types.Signature:
			fr.runtime.printPointer.call(fr, llvm_value)

		default:
			panic(fmt.Sprintf("Unhandled type kind: %s (%T)", typ, typ))
		}
	}
	if println_ {
		fr.runtime.printNl.call(fr)
	}
}
