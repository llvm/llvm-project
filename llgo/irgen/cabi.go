//===- cabi.go - C ABI abstraction layer ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an abstraction layer for the platform's C ABI (currently
// supports only Linux/x86_64).
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llvm/bindings/go/llvm"
)

type abiArgInfo int

const (
	AIK_Direct = abiArgInfo(iota)
	AIK_Indirect
)

type backendType interface {
	ToLLVM(llvm.Context) llvm.Type
}

type ptrBType struct {
}

func (t ptrBType) ToLLVM(c llvm.Context) llvm.Type {
	return llvm.PointerType(c.Int8Type(), 0)
}

type intBType struct {
	width  int
	signed bool
}

func (t intBType) ToLLVM(c llvm.Context) llvm.Type {
	return c.IntType(t.width * 8)
}

type floatBType struct {
	isDouble bool
}

func (t floatBType) ToLLVM(c llvm.Context) llvm.Type {
	if t.isDouble {
		return c.DoubleType()
	} else {
		return c.FloatType()
	}
}

type structBType struct {
	fields []backendType
}

func (t structBType) ToLLVM(c llvm.Context) llvm.Type {
	var lfields []llvm.Type
	for _, f := range t.fields {
		lfields = append(lfields, f.ToLLVM(c))
	}
	return c.StructType(lfields, false)
}

type arrayBType struct {
	length uint64
	elem   backendType
}

func (t arrayBType) ToLLVM(c llvm.Context) llvm.Type {
	return llvm.ArrayType(t.elem.ToLLVM(c), int(t.length))
}

// align returns the smallest y >= x such that y % a == 0.
func align(x, a int64) int64 {
	y := x + a - 1
	return y - y%a
}

func (tm *llvmTypeMap) sizeofStruct(fields ...types.Type) int64 {
	var o int64
	for _, f := range fields {
		a := tm.Alignof(f)
		o = align(o, a)
		o += tm.Sizeof(f)
	}
	return o
}

// This decides whether the x86_64 classification algorithm produces MEMORY for
// the given type. Given the subset of types that Go supports, this is exactly
// equivalent to testing the type's size.  See in particular the first step of
// the algorithm and its footnote.
func (tm *llvmTypeMap) classify(t ...types.Type) abiArgInfo {
	if tm.sizeofStruct(t...) > 16 {
		return AIK_Indirect
	}
	return AIK_Direct
}

func (tm *llvmTypeMap) sliceBackendType() backendType {
	i8ptr := &ptrBType{}
	uintptr := &intBType{tm.target.PointerSize(), false}
	return &structBType{[]backendType{i8ptr, uintptr, uintptr}}
}

func (tm *llvmTypeMap) getBackendType(t types.Type) backendType {
	switch t := t.(type) {
	case *types.Named:
		return tm.getBackendType(t.Underlying())

	case *types.Basic:
		switch t.Kind() {
		case types.Bool, types.Uint8:
			return &intBType{1, false}
		case types.Int8:
			return &intBType{1, true}
		case types.Uint16:
			return &intBType{2, false}
		case types.Int16:
			return &intBType{2, true}
		case types.Uint32:
			return &intBType{4, false}
		case types.Int32:
			return &intBType{4, true}
		case types.Uint64:
			return &intBType{8, false}
		case types.Int64:
			return &intBType{8, true}
		case types.Uint, types.Uintptr:
			return &intBType{tm.target.PointerSize(), false}
		case types.Int:
			return &intBType{tm.target.PointerSize(), true}
		case types.Float32:
			return &floatBType{false}
		case types.Float64:
			return &floatBType{true}
		case types.UnsafePointer:
			return &ptrBType{}
		case types.Complex64:
			f32 := &floatBType{false}
			return &structBType{[]backendType{f32, f32}}
		case types.Complex128:
			f64 := &floatBType{true}
			return &structBType{[]backendType{f64, f64}}
		case types.String:
			return &structBType{[]backendType{&ptrBType{}, &intBType{tm.target.PointerSize(), false}}}
		}

	case *types.Struct:
		var fields []backendType
		for i := 0; i != t.NumFields(); i++ {
			f := t.Field(i)
			fields = append(fields, tm.getBackendType(f.Type()))
		}
		return &structBType{fields}

	case *types.Pointer, *types.Signature, *types.Map, *types.Chan:
		return &ptrBType{}

	case *types.Interface:
		i8ptr := &ptrBType{}
		return &structBType{[]backendType{i8ptr, i8ptr}}

	case *types.Slice:
		return tm.sliceBackendType()

	case *types.Array:
		return &arrayBType{uint64(t.Len()), tm.getBackendType(t.Elem())}
	}

	panic("unhandled type: " + t.String())
}

type offsetedType struct {
	typ    backendType
	offset uint64
}

func (tm *llvmTypeMap) getBackendOffsets(bt backendType) (offsets []offsetedType) {
	switch bt := bt.(type) {
	case *structBType:
		t := bt.ToLLVM(tm.ctx)
		for i, f := range bt.fields {
			offset := tm.target.ElementOffset(t, i)
			fieldOffsets := tm.getBackendOffsets(f)
			for _, fo := range fieldOffsets {
				offsets = append(offsets, offsetedType{fo.typ, offset + fo.offset})
			}
		}

	case *arrayBType:
		size := tm.target.TypeAllocSize(bt.elem.ToLLVM(tm.ctx))
		fieldOffsets := tm.getBackendOffsets(bt.elem)
		for i := uint64(0); i != bt.length; i++ {
			for _, fo := range fieldOffsets {
				offsets = append(offsets, offsetedType{fo.typ, i*size + fo.offset})
			}
		}

	default:
		offsets = []offsetedType{offsetedType{bt, 0}}
	}

	return
}

func (tm *llvmTypeMap) classifyEightbyte(offsets []offsetedType, numInt, numSSE *int) llvm.Type {
	if len(offsets) == 1 {
		if _, ok := offsets[0].typ.(*floatBType); ok {
			*numSSE++
		} else {
			*numInt++
		}
		return offsets[0].typ.ToLLVM(tm.ctx)
	}
	// This implements classification for the basic types and step 4 of the
	// classification algorithm. At this point, the only two possible
	// classifications are SSE (floats) and INTEGER (everything else).
	sse := true
	for _, ot := range offsets {
		if _, ok := ot.typ.(*floatBType); !ok {
			sse = false
			break
		}
	}
	if sse {
		// This can only be (float, float), which uses an SSE vector.
		*numSSE++
		return llvm.VectorType(tm.ctx.FloatType(), 2)
	} else {
		*numInt++
		width := offsets[len(offsets)-1].offset + tm.target.TypeAllocSize(offsets[len(offsets)-1].typ.ToLLVM(tm.ctx)) - offsets[0].offset
		return tm.ctx.IntType(int(width) * 8)
	}
}

func (tm *llvmTypeMap) expandType(argTypes []llvm.Type, argAttrs []llvm.Attribute, bt backendType) ([]llvm.Type, []llvm.Attribute, int, int) {
	var numInt, numSSE int
	var argAttr llvm.Attribute

	switch bt := bt.(type) {
	case *structBType, *arrayBType:
		noneAttr := tm.ctx.CreateEnumAttribute(0, 0)
		bo := tm.getBackendOffsets(bt)
		sp := 0
		for sp != len(bo) && bo[sp].offset < 8 {
			sp++
		}
		eb1 := bo[0:sp]
		eb2 := bo[sp:]
		if len(eb2) > 0 {
			argTypes = append(argTypes, tm.classifyEightbyte(eb1, &numInt, &numSSE), tm.classifyEightbyte(eb2, &numInt, &numSSE))
			argAttrs = append(argAttrs, noneAttr, noneAttr)
		} else {
			argTypes = append(argTypes, tm.classifyEightbyte(eb1, &numInt, &numSSE))
			argAttrs = append(argAttrs, noneAttr)
		}

		return argTypes, argAttrs, numInt, numSSE

	case *intBType:
		if bt.width < 4 {
			var argAttrKind uint
			if bt.signed {
				argAttrKind = llvm.AttributeKindID("signext")
			} else {
				argAttrKind = llvm.AttributeKindID("zeroext")
			}
			argAttr = tm.ctx.CreateEnumAttribute(argAttrKind, 0)
		}
	}

	argTypes = append(argTypes, tm.classifyEightbyte([]offsetedType{{bt, 0}}, &numInt, &numSSE))
	argAttrs = append(argAttrs, argAttr)

	return argTypes, argAttrs, numInt, numSSE
}

type argInfo interface {
	// Emit instructions to builder to ABI encode val and store result to args.
	encode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, args []llvm.Value, val llvm.Value)

	// Emit instructions to builder to ABI decode and return the resulting Value.
	decode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder) llvm.Value
}

type retInfo interface {
	// Prepare args to receive a value. allocaBuilder refers to a builder in the entry block.
	prepare(ctx llvm.Context, allocaBuilder llvm.Builder, args []llvm.Value)

	// Emit instructions to builder to ABI decode the return value(s), if any. call is the
	// call instruction. Must be called after prepare().
	decode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, call llvm.Value) []llvm.Value

	// Emit instructions to builder to ABI encode the return value(s), if any, and return.
	encode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, vals []llvm.Value)
}

type directArgInfo struct {
	argOffset int
	argTypes  []llvm.Type
	valType   llvm.Type
}

func directEncode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, argTypes []llvm.Type, args []llvm.Value, val llvm.Value) {
	valType := val.Type()

	switch len(argTypes) {
	case 0:
		// do nothing

	case 1:
		if argTypes[0].C == valType.C {
			args[0] = val
			return
		}
		alloca := allocaBuilder.CreateAlloca(valType, "")
		bitcast := builder.CreateBitCast(alloca, llvm.PointerType(argTypes[0], 0), "")
		builder.CreateStore(val, alloca)
		args[0] = builder.CreateLoad(bitcast, "")

	case 2:
		encodeType := llvm.StructType(argTypes, false)
		alloca := allocaBuilder.CreateAlloca(valType, "")
		bitcast := builder.CreateBitCast(alloca, llvm.PointerType(encodeType, 0), "")
		builder.CreateStore(val, alloca)
		args[0] = builder.CreateLoad(builder.CreateStructGEP(bitcast, 0, ""), "")
		args[1] = builder.CreateLoad(builder.CreateStructGEP(bitcast, 1, ""), "")

	default:
		panic("unexpected argTypes size")
	}
}

func (ai *directArgInfo) encode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, args []llvm.Value, val llvm.Value) {
	directEncode(ctx, allocaBuilder, builder, ai.argTypes, args[ai.argOffset:ai.argOffset+len(ai.argTypes)], val)
}

func directDecode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, valType llvm.Type, args []llvm.Value) llvm.Value {
	var alloca llvm.Value

	switch len(args) {
	case 0:
		return llvm.ConstNull(ctx.StructType(nil, false))

	case 1:
		if args[0].Type().C == valType.C {
			return args[0]
		}
		alloca = allocaBuilder.CreateAlloca(valType, "")
		bitcast := builder.CreateBitCast(alloca, llvm.PointerType(args[0].Type(), 0), "")
		builder.CreateStore(args[0], bitcast)

	case 2:
		alloca = allocaBuilder.CreateAlloca(valType, "")
		var argTypes []llvm.Type
		for _, a := range args {
			argTypes = append(argTypes, a.Type())
		}
		encodeType := ctx.StructType(argTypes, false)
		bitcast := builder.CreateBitCast(alloca, llvm.PointerType(encodeType, 0), "")
		builder.CreateStore(args[0], builder.CreateStructGEP(bitcast, 0, ""))
		builder.CreateStore(args[1], builder.CreateStructGEP(bitcast, 1, ""))

	default:
		panic("unexpected argTypes size")
	}

	return builder.CreateLoad(alloca, "")
}

func (ai *directArgInfo) decode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder) llvm.Value {
	var args []llvm.Value
	fn := builder.GetInsertBlock().Parent()
	for i, _ := range ai.argTypes {
		args = append(args, fn.Param(ai.argOffset+i))
	}
	return directDecode(ctx, allocaBuilder, builder, ai.valType, args)
}

type indirectArgInfo struct {
	argOffset int
}

func (ai *indirectArgInfo) encode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, args []llvm.Value, val llvm.Value) {
	alloca := allocaBuilder.CreateAlloca(val.Type(), "")
	builder.CreateStore(val, alloca)
	args[ai.argOffset] = alloca
}

func (ai *indirectArgInfo) decode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder) llvm.Value {
	fn := builder.GetInsertBlock().Parent()
	return builder.CreateLoad(fn.Param(ai.argOffset), "")
}

type directRetInfo struct {
	numResults  int
	retTypes    []llvm.Type
	resultsType llvm.Type
}

func (ri *directRetInfo) prepare(ctx llvm.Context, allocaBuilder llvm.Builder, args []llvm.Value) {
}

func (ri *directRetInfo) decode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, call llvm.Value) []llvm.Value {
	var args []llvm.Value
	switch len(ri.retTypes) {
	case 0:
		return nil
	case 1:
		args = []llvm.Value{call}
	default:
		args = make([]llvm.Value, len(ri.retTypes))
		for i := 0; i != len(ri.retTypes); i++ {
			args[i] = builder.CreateExtractValue(call, i, "")
		}
	}

	d := directDecode(ctx, allocaBuilder, builder, ri.resultsType, args)

	if ri.numResults == 1 {
		return []llvm.Value{d}
	} else {
		results := make([]llvm.Value, ri.numResults)
		for i := 0; i != ri.numResults; i++ {
			results[i] = builder.CreateExtractValue(d, i, "")
		}
		return results
	}
}

func (ri *directRetInfo) encode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, vals []llvm.Value) {
	if len(ri.retTypes) == 0 {
		builder.CreateRetVoid()
		return
	}

	var val llvm.Value
	switch ri.numResults {
	case 1:
		val = vals[0]
	default:
		val = llvm.Undef(ri.resultsType)
		for i, v := range vals {
			val = builder.CreateInsertValue(val, v, i, "")
		}
	}

	args := make([]llvm.Value, len(ri.retTypes))
	directEncode(ctx, allocaBuilder, builder, ri.retTypes, args, val)

	var retval llvm.Value
	switch len(ri.retTypes) {
	case 1:
		retval = args[0]
	default:
		retval = llvm.Undef(ctx.StructType(ri.retTypes, false))
		for i, a := range args {
			retval = builder.CreateInsertValue(retval, a, i, "")
		}
	}
	builder.CreateRet(retval)
}

type indirectRetInfo struct {
	numResults  int
	sretSlot    llvm.Value
	resultsType llvm.Type
}

func (ri *indirectRetInfo) prepare(ctx llvm.Context, allocaBuilder llvm.Builder, args []llvm.Value) {
	ri.sretSlot = allocaBuilder.CreateAlloca(ri.resultsType, "")
	args[0] = ri.sretSlot
}

func (ri *indirectRetInfo) decode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, call llvm.Value) []llvm.Value {
	if ri.numResults == 1 {
		return []llvm.Value{builder.CreateLoad(ri.sretSlot, "")}
	} else {
		vals := make([]llvm.Value, ri.numResults)
		for i, _ := range vals {
			vals[i] = builder.CreateLoad(builder.CreateStructGEP(ri.sretSlot, i, ""), "")
		}
		return vals
	}
}

func (ri *indirectRetInfo) encode(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, vals []llvm.Value) {
	fn := builder.GetInsertBlock().Parent()
	sretSlot := fn.Param(0)

	if ri.numResults == 1 {
		builder.CreateStore(vals[0], sretSlot)
	} else {
		for i, v := range vals {
			builder.CreateStore(v, builder.CreateStructGEP(sretSlot, i, ""))
		}
	}
	builder.CreateRetVoid()
}

type functionTypeInfo struct {
	functionType llvm.Type
	argAttrs     []llvm.Attribute
	retAttr      llvm.Attribute
	argInfos     []argInfo
	retInf       retInfo
	chainIndex   int
}

func (fi *functionTypeInfo) declare(m llvm.Module, name string) llvm.Value {
	fn := llvm.AddFunction(m, name, fi.functionType)
	if fi.retAttr.GetEnumKind() != 0 {
		fn.AddAttributeAtIndex(0, fi.retAttr)
	}
	for i, a := range fi.argAttrs {
		if a.GetEnumKind() != 0 {
			fn.AddAttributeAtIndex(i + 1, a)
		}
	}
	return fn
}

func (fi *functionTypeInfo) call(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, callee llvm.Value, chain llvm.Value, args []llvm.Value) []llvm.Value {
	callArgs := make([]llvm.Value, len(fi.argAttrs))
	if chain.C == nil {
		chain = llvm.Undef(llvm.PointerType(ctx.Int8Type(), 0))
	}
	callArgs[fi.chainIndex] = chain
	for i, a := range args {
		fi.argInfos[i].encode(ctx, allocaBuilder, builder, callArgs, a)
	}
	fi.retInf.prepare(ctx, allocaBuilder, callArgs)
	typedCallee := builder.CreateBitCast(callee, llvm.PointerType(fi.functionType, 0), "")
	call := builder.CreateCall(typedCallee, callArgs, "")
	if fi.retAttr.GetEnumKind() != 0 {
		call.AddCallSiteAttribute(0, fi.retAttr)
	}
	for i, a := range fi.argAttrs {
		if a.GetEnumKind() != 0 {
			call.AddCallSiteAttribute(i + 1, a)
		}
	}
	return fi.retInf.decode(ctx, allocaBuilder, builder, call)
}

func (fi *functionTypeInfo) invoke(ctx llvm.Context, allocaBuilder llvm.Builder, builder llvm.Builder, callee llvm.Value, chain llvm.Value, args []llvm.Value, cont, lpad llvm.BasicBlock) []llvm.Value {
	callArgs := make([]llvm.Value, len(fi.argAttrs))
	if chain.C == nil {
		chain = llvm.Undef(llvm.PointerType(ctx.Int8Type(), 0))
	}
	callArgs[fi.chainIndex] = chain
	for i, a := range args {
		fi.argInfos[i].encode(ctx, allocaBuilder, builder, callArgs, a)
	}
	fi.retInf.prepare(ctx, allocaBuilder, callArgs)
	typedCallee := builder.CreateBitCast(callee, llvm.PointerType(fi.functionType, 0), "")
	call := builder.CreateInvoke(typedCallee, callArgs, cont, lpad, "")
	if fi.retAttr.GetEnumKind() != 0 {
		call.AddCallSiteAttribute(0, fi.retAttr)
	}
	for i, a := range fi.argAttrs {
		if a.GetEnumKind() != 0 {
			call.AddCallSiteAttribute(i + 1, a)
		}
	}
	builder.SetInsertPointAtEnd(cont)
	return fi.retInf.decode(ctx, allocaBuilder, builder, call)
}

func (tm *llvmTypeMap) getFunctionTypeInfo(args []types.Type, results []types.Type) (fi functionTypeInfo) {
	var returnType llvm.Type
	var argTypes []llvm.Type
	var argAttrKind uint
	if len(results) == 0 {
		returnType = llvm.VoidType()
		fi.retInf = &directRetInfo{}
	} else {
		aik := tm.classify(results...)

		var resultsType llvm.Type
		if len(results) == 1 {
			resultsType = tm.ToLLVM(results[0])
		} else {
			elements := make([]llvm.Type, len(results))
			for i := range elements {
				elements[i] = tm.ToLLVM(results[i])
			}
			resultsType = tm.ctx.StructType(elements, false)
		}

		switch aik {
		case AIK_Direct:
			var retFields []backendType
			for _, t := range results {
				retFields = append(retFields, tm.getBackendType(t))
			}
			bt := &structBType{retFields}

			retTypes, retAttrs, _, _ := tm.expandType(nil, nil, bt)
			switch len(retTypes) {
			case 0: // e.g., empty struct
				returnType = llvm.VoidType()
			case 1:
				returnType = retTypes[0]
				fi.retAttr = retAttrs[0]
			case 2:
				returnType = llvm.StructType(retTypes, false)
			default:
				panic("unexpected expandType result")
			}
			fi.retInf = &directRetInfo{numResults: len(results), retTypes: retTypes, resultsType: resultsType}

		case AIK_Indirect:
			returnType = llvm.VoidType()
			argTypes = []llvm.Type{llvm.PointerType(resultsType, 0)}
			argAttrKind = llvm.AttributeKindID("sret")
			fi.argAttrs = []llvm.Attribute{tm.ctx.CreateEnumAttribute(argAttrKind, 0)}
			fi.retInf = &indirectRetInfo{numResults: len(results), resultsType: resultsType}
		}
	}

	// Allocate an argument for the call chain.
	fi.chainIndex = len(argTypes)
	argTypes = append(argTypes, llvm.PointerType(tm.ctx.Int8Type(), 0))
	argAttrKind = llvm.AttributeKindID("nest")
	fi.argAttrs = append(fi.argAttrs, tm.ctx.CreateEnumAttribute(argAttrKind, 0))

	// Keep track of the number of INTEGER/SSE class registers remaining.
	remainingInt := 6
	remainingSSE := 8

	for _, arg := range args {
		aik := tm.classify(arg)

		isDirect := aik == AIK_Direct
		if isDirect {
			bt := tm.getBackendType(arg)
			directArgTypes, directArgAttrs, numInt, numSSE := tm.expandType(argTypes, fi.argAttrs, bt)

			// Check if the argument can fit into the remaining registers, or if
			// it would just occupy one register (which pushes the whole argument
			// onto the stack anyway).
			if numInt <= remainingInt && numSSE <= remainingSSE || numInt+numSSE == 1 {
				remainingInt -= numInt
				remainingSSE -= numSSE
				argInfo := &directArgInfo{argOffset: len(argTypes), valType: bt.ToLLVM(tm.ctx)}
				fi.argInfos = append(fi.argInfos, argInfo)
				argTypes = directArgTypes
				fi.argAttrs = directArgAttrs
				argInfo.argTypes = argTypes[argInfo.argOffset:len(argTypes)]
			} else {
				// No remaining registers; pass on the stack.
				isDirect = false
			}
		}

		if !isDirect {
			fi.argInfos = append(fi.argInfos, &indirectArgInfo{len(argTypes)})
			argTypes = append(argTypes, llvm.PointerType(tm.ToLLVM(arg), 0))
			argAttrKind = llvm.AttributeKindID("byval")
			fi.argAttrs = append(fi.argAttrs, tm.ctx.CreateEnumAttribute(argAttrKind, 0))
		}
	}

	fi.functionType = llvm.FunctionType(returnType, argTypes, false)
	return
}

func (tm *llvmTypeMap) getSignatureInfo(sig *types.Signature) functionTypeInfo {
	var args, results []types.Type
	if sig.Recv() != nil {
		recvtype := sig.Recv().Type()
		if _, ok := recvtype.Underlying().(*types.Pointer); !ok && recvtype != types.Typ[types.UnsafePointer] {
			recvtype = types.NewPointer(recvtype)
		}
		args = []types.Type{recvtype}
	}

	for i := 0; i != sig.Params().Len(); i++ {
		args = append(args, sig.Params().At(i).Type())
	}
	for i := 0; i != sig.Results().Len(); i++ {
		results = append(results, sig.Results().At(i).Type())
	}
	return tm.getFunctionTypeInfo(args, results)
}
