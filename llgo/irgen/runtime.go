//===- runtime.go - IR generation for runtime calls -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements IR generation for calls to the runtime library.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"strconv"

	"llvm.org/llgo/third_party/gotools/go/types"

	"llvm.org/llvm/bindings/go/llvm"
)

type runtimeFnInfo struct {
	fi *functionTypeInfo
	fn llvm.Value
}

func (rfi *runtimeFnInfo) init(tm *llvmTypeMap, m llvm.Module, name string, args []types.Type, results []types.Type) {
	rfi.fi = new(functionTypeInfo)
	*rfi.fi = tm.getFunctionTypeInfo(args, results)
	rfi.fn = rfi.fi.declare(m, name)
}

func (rfi *runtimeFnInfo) call(f *frame, args ...llvm.Value) []llvm.Value {
	if f.unwindBlock.IsNil() {
		return rfi.callOnly(f, args...)
	} else {
		return rfi.invoke(f, f.unwindBlock, args...)
	}
}

func (rfi *runtimeFnInfo) callOnly(f *frame, args ...llvm.Value) []llvm.Value {
	return rfi.fi.call(f.llvmtypes.ctx, f.allocaBuilder, f.builder, rfi.fn, llvm.Value{nil}, args)
}

func (rfi *runtimeFnInfo) invoke(f *frame, lpad llvm.BasicBlock, args ...llvm.Value) []llvm.Value {
	contbb := llvm.AddBasicBlock(f.function, "")
	return rfi.fi.invoke(f.llvmtypes.ctx, f.allocaBuilder, f.builder, rfi.fn, llvm.Value{nil}, args, contbb, lpad)
}

// runtimeInterface is a struct containing references to
// runtime types and intrinsic function declarations.
type runtimeInterface struct {
	// LLVM intrinsics
	memcpy,
	memset,
	returnaddress llvm.Value

	// Exception handling support
	gccgoPersonality   llvm.Value
	gccgoExceptionType llvm.Type

	// Runtime intrinsics
	append,
	assertInterface,
	byteArrayToString,
	canRecover,
	chanCap,
	chanLen,
	chanrecv2,
	checkDefer,
	checkInterfaceType,
	builtinClose,
	convertInterface,
	copy,
	Defer,
	deferredRecover,
	emptyInterfaceCompare,
	Go,
	ifaceE2I2,
	ifaceI2I2,
	intArrayToString,
	interfaceCompare,
	intToString,
	makeSlice,
	mapdelete,
	mapiter2,
	mapiterinit,
	mapiternext,
	mapIndex,
	mapLen,
	New,
	newChannel,
	newMap,
	newSelect,
	panic,
	printBool,
	printComplex,
	printDouble,
	printEmptyInterface,
	printInterface,
	printInt64,
	printNl,
	printPointer,
	printSlice,
	printSpace,
	printString,
	printUint64,
	receive,
	recover,
	registerGcRoots,
	runtimeError,
	selectdefault,
	selectrecv2,
	selectsend,
	selectgo,
	sendBig,
	setDeferRetaddr,
	strcmp,
	stringiter2,
	stringPlus,
	stringSlice,
	stringToByteArray,
	stringToIntArray,
	typeDescriptorsEqual,
	undefer runtimeFnInfo
}

func newRuntimeInterface(module llvm.Module, tm *llvmTypeMap) (*runtimeInterface, error) {
	var ri runtimeInterface

	Bool := types.Typ[types.Bool]
	Complex128 := types.Typ[types.Complex128]
	Float64 := types.Typ[types.Float64]
	Int32 := types.Typ[types.Int32]
	Int64 := types.Typ[types.Int64]
	Int := types.Typ[types.Int]
	Rune := types.Typ[types.Rune]
	String := types.Typ[types.String]
	Uintptr := types.Typ[types.Uintptr]
	UnsafePointer := types.Typ[types.UnsafePointer]

	EmptyInterface := types.NewInterface(nil, nil)
	ByteSlice := types.NewSlice(types.Typ[types.Byte])
	IntSlice := types.NewSlice(types.Typ[types.Int])

	AttrKind := llvm.AttributeKindID("nounwind")
	NoUnwindAttr := module.Context().CreateEnumAttribute(AttrKind, 0)
	AttrKind = llvm.AttributeKindID("noreturn")
	NoReturnAttr := module.Context().CreateEnumAttribute(AttrKind, 0)

	for _, rt := range [...]struct {
		name      string
		rfi       *runtimeFnInfo
		args, res []types.Type
		attrs     []llvm.Attribute
	}{
		{
			name: "__go_append",
			rfi:  &ri.append,
			args: []types.Type{IntSlice, UnsafePointer, Uintptr, Uintptr},
			res:  []types.Type{IntSlice},
		},
		{
			name: "__go_assert_interface",
			rfi:  &ri.assertInterface,
			args: []types.Type{UnsafePointer, UnsafePointer},
			res:  []types.Type{UnsafePointer},
		},
		{
			name:  "__go_byte_array_to_string",
			rfi:   &ri.byteArrayToString,
			args:  []types.Type{UnsafePointer, Int},
			res:   []types.Type{String},
			attrs: []llvm.Attribute{NoUnwindAttr},
		},
		{
			name: "__go_can_recover",
			rfi:  &ri.canRecover,
			args: []types.Type{UnsafePointer},
			res:  []types.Type{Bool},
		},
		{
			name: "__go_chan_cap",
			rfi:  &ri.chanCap,
			args: []types.Type{UnsafePointer},
			res:  []types.Type{Int},
		},
		{
			name: "__go_chan_len",
			rfi:  &ri.chanLen,
			args: []types.Type{UnsafePointer},
			res:  []types.Type{Int},
		},
		{
			name: "runtime.chanrecv2",
			rfi:  &ri.chanrecv2,
			args: []types.Type{UnsafePointer, UnsafePointer, UnsafePointer},
			res:  []types.Type{Bool},
		},
		{
			name: "__go_check_defer",
			rfi:  &ri.checkDefer,
			args: []types.Type{UnsafePointer},
		},
		{
			name: "__go_check_interface_type",
			rfi:  &ri.checkInterfaceType,
			args: []types.Type{UnsafePointer, UnsafePointer, UnsafePointer},
		},
		{
			name: "__go_builtin_close",
			rfi:  &ri.builtinClose,
			args: []types.Type{UnsafePointer},
		},
		{
			name: "__go_convert_interface",
			rfi:  &ri.convertInterface,
			args: []types.Type{UnsafePointer, UnsafePointer},
			res:  []types.Type{UnsafePointer},
		},
		{
			name: "__go_copy",
			rfi:  &ri.copy,
			args: []types.Type{UnsafePointer, UnsafePointer, Uintptr},
		},
		{
			name: "__go_defer",
			rfi:  &ri.Defer,
			args: []types.Type{UnsafePointer, UnsafePointer, UnsafePointer},
		},
		{
			name: "__go_deferred_recover",
			rfi:  &ri.deferredRecover,
			res:  []types.Type{EmptyInterface},
		},
		{
			name: "__go_empty_interface_compare",
			rfi:  &ri.emptyInterfaceCompare,
			args: []types.Type{EmptyInterface, EmptyInterface},
			res:  []types.Type{Int},
		},
		{
			name: "__go_go",
			rfi:  &ri.Go,
			args: []types.Type{UnsafePointer, UnsafePointer},
		},
		{
			name: "runtime.ifaceE2I2",
			rfi:  &ri.ifaceE2I2,
			args: []types.Type{UnsafePointer, EmptyInterface},
			res:  []types.Type{EmptyInterface, Bool},
		},
		{
			name: "runtime.ifaceI2I2",
			rfi:  &ri.ifaceI2I2,
			args: []types.Type{UnsafePointer, EmptyInterface},
			res:  []types.Type{EmptyInterface, Bool},
		},
		{
			name: "__go_int_array_to_string",
			rfi:  &ri.intArrayToString,
			args: []types.Type{UnsafePointer, Int},
			res:  []types.Type{String},
		},
		{
			name: "__go_int_to_string",
			rfi:  &ri.intToString,
			args: []types.Type{Int},
			res:  []types.Type{String},
		},
		{
			name: "__go_interface_compare",
			rfi:  &ri.interfaceCompare,
			args: []types.Type{EmptyInterface, EmptyInterface},
			res:  []types.Type{Int},
		},
		{
			name: "__go_make_slice2",
			rfi:  &ri.makeSlice,
			args: []types.Type{UnsafePointer, Uintptr, Uintptr},
			res:  []types.Type{IntSlice},
		},
		{
			name: "runtime.mapdelete",
			rfi:  &ri.mapdelete,
			args: []types.Type{UnsafePointer, UnsafePointer},
		},
		{
			name: "runtime.mapiter2",
			rfi:  &ri.mapiter2,
			args: []types.Type{UnsafePointer, UnsafePointer, UnsafePointer},
		},
		{
			name: "runtime.mapiterinit",
			rfi:  &ri.mapiterinit,
			args: []types.Type{UnsafePointer, UnsafePointer},
		},
		{
			name: "runtime.mapiternext",
			rfi:  &ri.mapiternext,
			args: []types.Type{UnsafePointer},
		},
		{
			name: "__go_map_index",
			rfi:  &ri.mapIndex,
			args: []types.Type{UnsafePointer, UnsafePointer, Bool},
			res:  []types.Type{UnsafePointer},
		},
		{
			name: "__go_map_len",
			rfi:  &ri.mapLen,
			args: []types.Type{UnsafePointer},
			res:  []types.Type{Int},
		},
		{
			name:  "__go_new",
			rfi:   &ri.New,
			args:  []types.Type{UnsafePointer, Uintptr},
			res:   []types.Type{UnsafePointer},
			attrs: []llvm.Attribute{NoUnwindAttr},
		},
		{
			name: "__go_new_channel",
			rfi:  &ri.newChannel,
			args: []types.Type{UnsafePointer, Uintptr},
			res:  []types.Type{UnsafePointer},
		},
		{
			name: "__go_new_map",
			rfi:  &ri.newMap,
			args: []types.Type{UnsafePointer, Uintptr},
			res:  []types.Type{UnsafePointer},
		},
		{
			name: "runtime.newselect",
			rfi:  &ri.newSelect,
			args: []types.Type{Int32},
			res:  []types.Type{UnsafePointer},
		},
		{
			name:  "__go_panic",
			rfi:   &ri.panic,
			args:  []types.Type{EmptyInterface},
			attrs: []llvm.Attribute{NoReturnAttr},
		},
		{
			name: "__go_print_bool",
			rfi:  &ri.printBool,
			args: []types.Type{Bool},
		},
		{
			name: "__go_print_complex",
			rfi:  &ri.printComplex,
			args: []types.Type{Complex128},
		},
		{
			name: "__go_print_double",
			rfi:  &ri.printDouble,
			args: []types.Type{Float64},
		},
		{
			name: "__go_print_empty_interface",
			rfi:  &ri.printEmptyInterface,
			args: []types.Type{EmptyInterface},
		},
		{
			name: "__go_print_interface",
			rfi:  &ri.printInterface,
			args: []types.Type{EmptyInterface},
		},
		{
			name: "__go_print_int64",
			rfi:  &ri.printInt64,
			args: []types.Type{Int64},
		},
		{
			name: "__go_print_nl",
			rfi:  &ri.printNl,
		},
		{
			name: "__go_print_pointer",
			rfi:  &ri.printPointer,
			args: []types.Type{UnsafePointer},
		},
		{
			name: "__go_print_slice",
			rfi:  &ri.printSlice,
			args: []types.Type{IntSlice},
		},
		{
			name: "__go_print_space",
			rfi:  &ri.printSpace,
		},
		{
			name: "__go_print_string",
			rfi:  &ri.printString,
			args: []types.Type{String},
		},
		{
			name: "__go_print_uint64",
			rfi:  &ri.printUint64,
			args: []types.Type{Int64},
		},
		{
			name: "__go_receive",
			rfi:  &ri.receive,
			args: []types.Type{UnsafePointer, UnsafePointer, UnsafePointer},
		},
		{
			name: "__go_recover",
			rfi:  &ri.recover,
			res:  []types.Type{EmptyInterface},
		},
		{
			name: "__go_register_gc_roots",
			rfi:  &ri.registerGcRoots,
			args: []types.Type{UnsafePointer},
		},
		{
			name:  "__go_runtime_error",
			rfi:   &ri.runtimeError,
			args:  []types.Type{Int32},
			attrs: []llvm.Attribute{NoReturnAttr},
		},
		{
			name: "runtime.selectdefault",
			rfi:  &ri.selectdefault,
			args: []types.Type{UnsafePointer, Int32},
		},
		{
			name: "runtime.selectgo",
			rfi:  &ri.selectgo,
			args: []types.Type{UnsafePointer},
			res:  []types.Type{Int},
		},
		{
			name: "runtime.selectrecv2",
			rfi:  &ri.selectrecv2,
			args: []types.Type{UnsafePointer, UnsafePointer, UnsafePointer, UnsafePointer, Int32},
		},
		{
			name: "runtime.selectsend",
			rfi:  &ri.selectsend,
			args: []types.Type{UnsafePointer, UnsafePointer, UnsafePointer, Int32},
		},
		{
			name: "__go_send_big",
			rfi:  &ri.sendBig,
			args: []types.Type{UnsafePointer, UnsafePointer, UnsafePointer},
		},
		{
			name: "__go_set_defer_retaddr",
			rfi:  &ri.setDeferRetaddr,
			args: []types.Type{UnsafePointer},
			res:  []types.Type{Bool},
		},
		{
			name: "__go_strcmp",
			rfi:  &ri.strcmp,
			args: []types.Type{String, String},
			res:  []types.Type{Int},
		},
		{
			name: "__go_string_plus",
			rfi:  &ri.stringPlus,
			args: []types.Type{String, String},
			res:  []types.Type{String},
		},
		{
			name: "__go_string_slice",
			rfi:  &ri.stringSlice,
			args: []types.Type{String, Int, Int},
			res:  []types.Type{String},
		},
		{
			name:  "__go_string_to_byte_array",
			rfi:   &ri.stringToByteArray,
			args:  []types.Type{String},
			res:   []types.Type{ByteSlice},
			attrs: []llvm.Attribute{NoUnwindAttr},
		},
		{
			name: "__go_string_to_int_array",
			rfi:  &ri.stringToIntArray,
			args: []types.Type{String},
			res:  []types.Type{IntSlice},
		},
		{
			name: "runtime.stringiter2",
			rfi:  &ri.stringiter2,
			args: []types.Type{String, Int},
			res:  []types.Type{Int, Rune},
		},
		{
			name: "__go_type_descriptors_equal",
			rfi:  &ri.typeDescriptorsEqual,
			args: []types.Type{UnsafePointer, UnsafePointer},
			res:  []types.Type{Bool},
		},
		{
			name: "__go_undefer",
			rfi:  &ri.undefer,
			args: []types.Type{UnsafePointer},
		},
	} {
		rt.rfi.init(tm, module, rt.name, rt.args, rt.res)
		for _, attr := range rt.attrs {
			rt.rfi.fn.AddFunctionAttr(attr)
		}
	}

	memsetName := "llvm.memset.p0i8.i" + strconv.Itoa(tm.target.IntPtrType().IntTypeWidth())
	memsetType := llvm.FunctionType(
		llvm.VoidType(),
		[]llvm.Type{
			llvm.PointerType(llvm.Int8Type(), 0),
			llvm.Int8Type(),
			tm.target.IntPtrType(),
			llvm.Int32Type(),
			llvm.Int1Type(),
		},
		false,
	)
	ri.memset = llvm.AddFunction(module, memsetName, memsetType)

	memcpyName := "llvm.memcpy.p0i8.p0i8.i" + strconv.Itoa(tm.target.IntPtrType().IntTypeWidth())
	memcpyType := llvm.FunctionType(
		llvm.VoidType(),
		[]llvm.Type{
			llvm.PointerType(llvm.Int8Type(), 0),
			llvm.PointerType(llvm.Int8Type(), 0),
			tm.target.IntPtrType(),
			llvm.Int32Type(),
			llvm.Int1Type(),
		},
		false,
	)
	ri.memcpy = llvm.AddFunction(module, memcpyName, memcpyType)

	returnaddressType := llvm.FunctionType(
		llvm.PointerType(llvm.Int8Type(), 0),
		[]llvm.Type{llvm.Int32Type()},
		false,
	)
	ri.returnaddress = llvm.AddFunction(module, "llvm.returnaddress", returnaddressType)

	gccgoPersonalityType := llvm.FunctionType(
		llvm.Int32Type(),
		[]llvm.Type{
			llvm.Int32Type(),
			llvm.Int64Type(),
			llvm.PointerType(llvm.Int8Type(), 0),
			llvm.PointerType(llvm.Int8Type(), 0),
		},
		false,
	)
	ri.gccgoPersonality = llvm.AddFunction(module, "__gccgo_personality_v0", gccgoPersonalityType)

	ri.gccgoExceptionType = llvm.StructType(
		[]llvm.Type{
			llvm.PointerType(llvm.Int8Type(), 0),
			llvm.Int32Type(),
		},
		false,
	)

	return &ri, nil
}

func (fr *frame) createZExtOrTrunc(v llvm.Value, t llvm.Type, name string) llvm.Value {
	switch n := v.Type().IntTypeWidth() - t.IntTypeWidth(); {
	case n < 0:
		v = fr.builder.CreateZExt(v, fr.target.IntPtrType(), name)
	case n > 0:
		v = fr.builder.CreateTrunc(v, fr.target.IntPtrType(), name)
	}
	return v
}

func (fr *frame) createTypeMalloc(t types.Type) llvm.Value {
	size := llvm.ConstInt(fr.target.IntPtrType(), uint64(fr.llvmtypes.Sizeof(t)), false)
	malloc := fr.runtime.New.callOnly(fr, fr.types.ToRuntime(t), size)[0]
	return fr.builder.CreateBitCast(malloc, llvm.PointerType(fr.types.ToLLVM(t), 0), "")
}

func (fr *frame) memsetZero(ptr llvm.Value, size llvm.Value) {
	memset := fr.runtime.memset
	ptr = fr.builder.CreateBitCast(ptr, llvm.PointerType(llvm.Int8Type(), 0), "")
	fill := llvm.ConstNull(llvm.Int8Type())
	size = fr.createZExtOrTrunc(size, fr.target.IntPtrType(), "")
	align := llvm.ConstInt(llvm.Int32Type(), 1, false)
	isvolatile := llvm.ConstNull(llvm.Int1Type())
	fr.builder.CreateCall(memset, []llvm.Value{ptr, fill, size, align, isvolatile}, "")
}

func (fr *frame) memcpy(dest llvm.Value, src llvm.Value, size llvm.Value) {
	memcpy := fr.runtime.memcpy
	dest = fr.builder.CreateBitCast(dest, llvm.PointerType(llvm.Int8Type(), 0), "")
	src = fr.builder.CreateBitCast(src, llvm.PointerType(llvm.Int8Type(), 0), "")
	size = fr.createZExtOrTrunc(size, fr.target.IntPtrType(), "")
	align := llvm.ConstInt(llvm.Int32Type(), 1, false)
	isvolatile := llvm.ConstNull(llvm.Int1Type())
	fr.builder.CreateCall(memcpy, []llvm.Value{dest, src, size, align, isvolatile}, "")
}

func (fr *frame) returnAddress(level uint64) llvm.Value {
	returnaddress := fr.runtime.returnaddress
	levelValue := llvm.ConstInt(llvm.Int32Type(), level, false)
	return fr.builder.CreateCall(returnaddress, []llvm.Value{levelValue}, "")
}
