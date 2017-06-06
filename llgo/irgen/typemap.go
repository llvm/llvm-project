//===- typemap.go - type and type descriptor mapping ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the mapping from go/types types to LLVM types and to
// type descriptors.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"bytes"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"llvm.org/llgo/third_party/gotools/go/ssa"
	"llvm.org/llgo/third_party/gotools/go/ssa/ssautil"
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llgo/third_party/gotools/go/types/typeutil"
	"llvm.org/llvm/bindings/go/llvm"
)

type MethodResolver interface {
	ResolveMethod(*types.Selection) *govalue
}

// llvmTypeMap is provides a means of mapping from a types.Map
// to llgo's corresponding LLVM type representation.
type llvmTypeMap struct {
	sizes      *types.StdSizes
	ctx        llvm.Context
	target     llvm.TargetData
	inttype    llvm.Type
	stringType llvm.Type

	types typeutil.Map
}

type typeDescInfo struct {
	global        llvm.Value
	commonTypePtr llvm.Value
	mapDescPtr    llvm.Value
	gc, gcPtr     llvm.Value

	interfaceMethodTables typeutil.Map
}

type TypeMap struct {
	*llvmTypeMap
	mc manglerContext

	module         llvm.Module
	pkgpath        string
	types, algs    typeutil.Map
	runtime        *runtimeInterface
	methodResolver MethodResolver
	types.MethodSetCache

	commonTypeType, uncommonTypeType, ptrTypeType, funcTypeType, arrayTypeType, sliceTypeType, mapTypeType, chanTypeType, interfaceTypeType, structTypeType llvm.Type
	mapDescType                                                                                                                                             llvm.Type

	methodType, imethodType, structFieldType llvm.Type

	typeSliceType, methodSliceType, imethodSliceType, structFieldSliceType llvm.Type

	funcValType             llvm.Type
	hashFnType, equalFnType llvm.Type

	algsEmptyInterface,
	algsInterface,
	algsFloat,
	algsComplex,
	algsString,
	algsIdentity,
	algsError algorithms
}

type algorithms struct {
	hash, hashDescriptor, equal, equalDescriptor llvm.Value
}

func NewLLVMTypeMap(ctx llvm.Context, target llvm.TargetData) *llvmTypeMap {
	// spec says int is either 32-bit or 64-bit.
	// ABI currently requires sizeof(int) == sizeof(uint) == sizeof(uintptr).
	inttype := ctx.IntType(8 * target.PointerSize())

	i8ptr := llvm.PointerType(llvm.Int8Type(), 0)
	elements := []llvm.Type{i8ptr, inttype}
	stringType := llvm.StructType(elements, false)

	return &llvmTypeMap{
		ctx: ctx,
		sizes: &types.StdSizes{
			WordSize: int64(target.PointerSize()),
			MaxAlign: 8,
		},
		target:     target,
		inttype:    inttype,
		stringType: stringType,
	}
}

func NewTypeMap(pkg *ssa.Package, llvmtm *llvmTypeMap, module llvm.Module, r *runtimeInterface, mr MethodResolver) *TypeMap {
	tm := &TypeMap{
		llvmTypeMap:    llvmtm,
		module:         module,
		pkgpath:        pkg.Object.Path(),
		runtime:        r,
		methodResolver: mr,
	}

	tm.mc.init(pkg.Prog, &tm.MethodSetCache)

	uintptrType := tm.inttype
	voidPtrType := llvm.PointerType(tm.ctx.Int8Type(), 0)
	boolType := llvm.Int8Type()
	stringPtrType := llvm.PointerType(tm.stringType, 0)

	tm.funcValType = tm.ctx.StructCreateNamed("funcVal")
	tm.funcValType.StructSetBody([]llvm.Type{
		llvm.PointerType(llvm.FunctionType(llvm.VoidType(), []llvm.Type{}, false), 0),
	}, false)

	params := []llvm.Type{voidPtrType, uintptrType}
	tm.hashFnType = llvm.FunctionType(uintptrType, params, false)
	params = []llvm.Type{voidPtrType, voidPtrType, uintptrType}
	tm.equalFnType = llvm.FunctionType(boolType, params, false)

	typeAlgorithms := [...]struct {
		Name string
		*algorithms
	}{
		{"empty_interface", &tm.algsEmptyInterface},
		{"interface", &tm.algsInterface},
		{"float", &tm.algsFloat},
		{"complex", &tm.algsComplex},
		{"string", &tm.algsString},
		{"identity", &tm.algsIdentity},
		{"error", &tm.algsError},
	}
	for _, typeAlgs := range typeAlgorithms {
		hashFnName := "__go_type_hash_" + typeAlgs.Name
		hashDescriptorName := hashFnName + "_descriptor"
		equalFnName := "__go_type_equal_" + typeAlgs.Name
		equalDescriptorName := equalFnName + "_descriptor"
		typeAlgs.hash = llvm.AddFunction(tm.module, hashFnName, tm.hashFnType)
		typeAlgs.hashDescriptor = llvm.AddGlobal(tm.module, tm.funcValType, hashDescriptorName)
		typeAlgs.equal = llvm.AddFunction(tm.module, equalFnName, tm.equalFnType)
		typeAlgs.equalDescriptor = llvm.AddGlobal(tm.module, tm.funcValType, equalDescriptorName)
	}

	tm.commonTypeType = tm.ctx.StructCreateNamed("commonType")
	commonTypeTypePtr := llvm.PointerType(tm.commonTypeType, 0)

	tm.methodType = tm.ctx.StructCreateNamed("method")
	tm.methodType.StructSetBody([]llvm.Type{
		stringPtrType,     // name
		stringPtrType,     // pkgPath
		commonTypeTypePtr, // mtype (without receiver)
		commonTypeTypePtr, // type (with receiver)
		voidPtrType,       // function
	}, false)

	tm.methodSliceType = tm.makeNamedSliceType("methodSlice", tm.methodType)

	tm.uncommonTypeType = tm.ctx.StructCreateNamed("uncommonType")
	tm.uncommonTypeType.StructSetBody([]llvm.Type{
		stringPtrType,      // name
		stringPtrType,      // pkgPath
		tm.methodSliceType, // methods
	}, false)

	tm.commonTypeType.StructSetBody([]llvm.Type{
		tm.ctx.Int8Type(),                        // Kind
		tm.ctx.Int8Type(),                        // align
		tm.ctx.Int8Type(),                        // fieldAlign
		uintptrType,                              // size
		tm.ctx.Int32Type(),                       // hash
		llvm.PointerType(tm.funcValType, 0),      // hashfn
		llvm.PointerType(tm.funcValType, 0),      // equalfn
		voidPtrType,                              // gc
		stringPtrType,                            // string
		llvm.PointerType(tm.uncommonTypeType, 0), // uncommonType
		commonTypeTypePtr,                        // ptrToThis
	}, false)

	tm.typeSliceType = tm.makeNamedSliceType("typeSlice", commonTypeTypePtr)

	tm.ptrTypeType = tm.ctx.StructCreateNamed("ptrType")
	tm.ptrTypeType.StructSetBody([]llvm.Type{
		tm.commonTypeType,
		commonTypeTypePtr,
	}, false)

	tm.funcTypeType = tm.ctx.StructCreateNamed("funcType")
	tm.funcTypeType.StructSetBody([]llvm.Type{
		tm.commonTypeType,
		tm.ctx.Int8Type(), // dotdotdot
		tm.typeSliceType,  // in
		tm.typeSliceType,  // out
	}, false)

	tm.arrayTypeType = tm.ctx.StructCreateNamed("arrayType")
	tm.arrayTypeType.StructSetBody([]llvm.Type{
		tm.commonTypeType,
		commonTypeTypePtr, // elem
		commonTypeTypePtr, // slice
		tm.inttype,        // len
	}, false)

	tm.sliceTypeType = tm.ctx.StructCreateNamed("sliceType")
	tm.sliceTypeType.StructSetBody([]llvm.Type{
		tm.commonTypeType,
		commonTypeTypePtr, // elem
	}, false)

	tm.mapTypeType = tm.ctx.StructCreateNamed("mapType")
	tm.mapTypeType.StructSetBody([]llvm.Type{
		tm.commonTypeType,
		commonTypeTypePtr, // key
		commonTypeTypePtr, // elem
	}, false)

	tm.chanTypeType = tm.ctx.StructCreateNamed("chanType")
	tm.chanTypeType.StructSetBody([]llvm.Type{
		tm.commonTypeType,
		commonTypeTypePtr, // elem
		tm.inttype,        // dir
	}, false)

	tm.imethodType = tm.ctx.StructCreateNamed("imethod")
	tm.imethodType.StructSetBody([]llvm.Type{
		stringPtrType,     // name
		stringPtrType,     // pkgPath
		commonTypeTypePtr, // typ
	}, false)

	tm.imethodSliceType = tm.makeNamedSliceType("imethodSlice", tm.imethodType)

	tm.interfaceTypeType = tm.ctx.StructCreateNamed("interfaceType")
	tm.interfaceTypeType.StructSetBody([]llvm.Type{
		tm.commonTypeType,
		tm.imethodSliceType,
	}, false)

	tm.structFieldType = tm.ctx.StructCreateNamed("structField")
	tm.structFieldType.StructSetBody([]llvm.Type{
		stringPtrType,     // name
		stringPtrType,     // pkgPath
		commonTypeTypePtr, // typ
		stringPtrType,     // tag
		tm.inttype,        // offset
	}, false)

	tm.structFieldSliceType = tm.makeNamedSliceType("structFieldSlice", tm.structFieldType)

	tm.structTypeType = tm.ctx.StructCreateNamed("structType")
	tm.structTypeType.StructSetBody([]llvm.Type{
		tm.commonTypeType,
		tm.structFieldSliceType, // fields
	}, false)

	tm.mapDescType = tm.ctx.StructCreateNamed("mapDesc")
	tm.mapDescType.StructSetBody([]llvm.Type{
		commonTypeTypePtr, // map_descriptor
		tm.inttype,        // entry_size
		tm.inttype,        // key_offset
		tm.inttype,        // value_offset
	}, false)

	return tm
}

func (tm *llvmTypeMap) ToLLVM(t types.Type) llvm.Type {
	return tm.toLLVM(t, "")
}

func (tm *llvmTypeMap) toLLVM(t types.Type, name string) llvm.Type {
	lt, ok := tm.types.At(t).(llvm.Type)
	if !ok {
		lt = tm.makeLLVMType(t, name)
		if lt.IsNil() {
			panic(fmt.Sprint("Failed to create LLVM type for: ", t))
		}
		tm.types.Set(t, lt)
	}
	return lt
}

func (tm *llvmTypeMap) makeLLVMType(t types.Type, name string) llvm.Type {
	return tm.getBackendType(t).ToLLVM(tm.ctx)
}

func (tm *llvmTypeMap) Offsetsof(fields []*types.Var) []int64 {
	offsets := make([]int64, len(fields))
	var o int64
	for i, f := range fields {
		a := tm.Alignof(f.Type())
		o = align(o, a)
		offsets[i] = o
		o += tm.Sizeof(f.Type())
	}
	return offsets
}

var basicSizes = [...]byte{
	types.Bool:       1,
	types.Int8:       1,
	types.Int16:      2,
	types.Int32:      4,
	types.Int64:      8,
	types.Uint8:      1,
	types.Uint16:     2,
	types.Uint32:     4,
	types.Uint64:     8,
	types.Float32:    4,
	types.Float64:    8,
	types.Complex64:  8,
	types.Complex128: 16,
}

func (tm *llvmTypeMap) Sizeof(T types.Type) int64 {
	switch t := T.Underlying().(type) {
	case *types.Basic:
		k := t.Kind()
		if int(k) < len(basicSizes) {
			if s := basicSizes[k]; s > 0 {
				return int64(s)
			}
		}
		if k == types.String {
			return tm.sizes.WordSize * 2
		}
	case *types.Array:
		a := tm.Alignof(t.Elem())
		z := tm.Sizeof(t.Elem())
		return align(z, a) * t.Len() // may be 0
	case *types.Slice:
		return tm.sizes.WordSize * 3
	case *types.Struct:
		n := t.NumFields()
		if n == 0 {
			return 0
		}
		fields := make([]*types.Var, t.NumFields())
		for i := 0; i != t.NumFields(); i++ {
			fields[i] = t.Field(i)
		}
		offsets := tm.Offsetsof(fields)
		return align(offsets[n-1]+tm.Sizeof(t.Field(n-1).Type()), tm.Alignof(t))
	case *types.Interface:
		return tm.sizes.WordSize * 2
	}
	return tm.sizes.WordSize // catch-all
}

func (tm *llvmTypeMap) Alignof(t types.Type) int64 {
	return tm.sizes.Alignof(t)
}

///////////////////////////////////////////////////////////////////////////////

func (tm *TypeMap) ToRuntime(t types.Type) llvm.Value {
	return llvm.ConstBitCast(tm.getTypeDescriptorPointer(t), llvm.PointerType(llvm.Int8Type(), 0))
}

type localNamedTypeInfo struct {
	functionName string
	scopeNum     int
}

type namedTypeInfo struct {
	pkgname, pkgpath string
	name             string
	localNamedTypeInfo
}

type manglerContext struct {
	ti  map[*types.Named]localNamedTypeInfo
	msc *types.MethodSetCache
}

// Assembles the method set into the order that gccgo uses (unexported methods first).
// TODO(pcc): cache this.
func orderedMethodSet(ms *types.MethodSet) []*types.Selection {
	oms := make([]*types.Selection, ms.Len())
	omsi := 0
	for i := 0; i != ms.Len(); i++ {
		if sel := ms.At(i); !sel.Obj().Exported() {
			oms[omsi] = sel
			omsi++
		}
	}
	for i := 0; i != ms.Len(); i++ {
		if sel := ms.At(i); sel.Obj().Exported() {
			oms[omsi] = sel
			omsi++
		}
	}
	return oms
}

func (ctx *manglerContext) init(prog *ssa.Program, msc *types.MethodSetCache) {
	ctx.msc = msc
	ctx.ti = make(map[*types.Named]localNamedTypeInfo)
	for f, _ := range ssautil.AllFunctions(prog) {
		scopeNum := 0
		var addNamedTypesToMap func(*types.Scope)
		addNamedTypesToMap = func(scope *types.Scope) {
			hasNamedTypes := false
			for _, n := range scope.Names() {
				if tn, ok := scope.Lookup(n).(*types.TypeName); ok {
					hasNamedTypes = true
					ctx.ti[tn.Type().(*types.Named)] = localNamedTypeInfo{f.Name(), scopeNum}
				}
			}
			if hasNamedTypes {
				scopeNum++
			}
			for i := 0; i != scope.NumChildren(); i++ {
				addNamedTypesToMap(scope.Child(i))
			}
		}
		if fobj, ok := f.Object().(*types.Func); ok && fobj.Scope() != nil {
			addNamedTypesToMap(fobj.Scope())
		}
	}
}

func (ctx *manglerContext) getNamedTypeInfo(t types.Type) (nti namedTypeInfo) {
	switch t := t.(type) {
	case *types.Basic:
		switch t.Kind() {
		case types.Byte:
			nti.name = "uint8"
		case types.Rune:
			nti.name = "int32"
		case types.UnsafePointer:
			nti.pkgname = "unsafe"
			nti.pkgpath = "unsafe"
			nti.name = "Pointer"
		default:
			nti.name = t.Name()
		}

	case *types.Named:
		obj := t.Obj()
		if pkg := obj.Pkg(); pkg != nil {
			nti.pkgname = obj.Pkg().Name()
			nti.pkgpath = obj.Pkg().Path()
		}
		nti.name = obj.Name()
		nti.localNamedTypeInfo = ctx.ti[t]

	default:
		panic("not a named type")
	}

	return
}

func (ctx *manglerContext) mangleSignature(s *types.Signature, recv *types.Var, b *bytes.Buffer) {
	b.WriteRune('F')
	if recv != nil {
		b.WriteRune('m')
		ctx.mangleType(recv.Type(), b)
	}

	if p := s.Params(); p.Len() != 0 {
		b.WriteRune('p')
		for i := 0; i != p.Len(); i++ {
			ctx.mangleType(p.At(i).Type(), b)
		}
		if s.Variadic() {
			b.WriteRune('V')
		}
		b.WriteRune('e')
	}

	if r := s.Results(); r.Len() != 0 {
		b.WriteRune('r')
		for i := 0; i != r.Len(); i++ {
			ctx.mangleType(r.At(i).Type(), b)
		}
		b.WriteRune('e')
	}

	b.WriteRune('e')
}

func ManglePackagePath(pkgpath string) string {
	pkgpath = strings.Replace(pkgpath, "/", "_", -1)
	pkgpath = strings.Replace(pkgpath, ".", "_", -1)
	return pkgpath
}

func (ctx *manglerContext) mangleType(t types.Type, b *bytes.Buffer) {
	switch t := t.(type) {
	case *types.Basic, *types.Named:
		var nb bytes.Buffer
		ti := ctx.getNamedTypeInfo(t)
		if ti.pkgpath != "" {
			nb.WriteString(ManglePackagePath(ti.pkgpath))
			nb.WriteRune('.')
		}
		if ti.functionName != "" {
			nb.WriteString(ti.functionName)
			nb.WriteRune('$')
			if ti.scopeNum != 0 {
				nb.WriteString(strconv.Itoa(ti.scopeNum))
				nb.WriteRune('$')
			}
		}
		nb.WriteString(ti.name)

		b.WriteRune('N')
		b.WriteString(strconv.Itoa(nb.Len()))
		b.WriteRune('_')
		b.WriteString(nb.String())

	case *types.Pointer:
		b.WriteRune('p')
		ctx.mangleType(t.Elem(), b)

	case *types.Map:
		b.WriteRune('M')
		ctx.mangleType(t.Key(), b)
		b.WriteString("__")
		ctx.mangleType(t.Elem(), b)

	case *types.Chan:
		b.WriteRune('C')
		ctx.mangleType(t.Elem(), b)
		switch t.Dir() {
		case types.SendOnly:
			b.WriteRune('s')
		case types.RecvOnly:
			b.WriteRune('r')
		case types.SendRecv:
			b.WriteString("sr")
		}
		b.WriteRune('e')

	case *types.Signature:
		ctx.mangleSignature(t, t.Recv(), b)

	case *types.Array:
		b.WriteRune('A')
		ctx.mangleType(t.Elem(), b)
		b.WriteString(strconv.FormatInt(t.Len(), 10))
		b.WriteRune('e')

	case *types.Slice:
		b.WriteRune('A')
		ctx.mangleType(t.Elem(), b)
		b.WriteRune('e')

	case *types.Struct:
		b.WriteRune('S')
		for i := 0; i != t.NumFields(); i++ {
			f := t.Field(i)
			if f.Anonymous() {
				b.WriteString("0_")
			} else {
				b.WriteString(strconv.Itoa(len(f.Name())))
				b.WriteRune('_')
				b.WriteString(f.Name())
			}
			ctx.mangleType(f.Type(), b)
			// TODO: tags are mangled here
		}
		b.WriteRune('e')

	case *types.Interface:
		b.WriteRune('I')
		methodset := ctx.msc.MethodSet(t)
		for _, m := range orderedMethodSet(methodset) {
			method := m.Obj()
			var nb bytes.Buffer
			if !method.Exported() {
				nb.WriteRune('.')
				nb.WriteString(method.Pkg().Path())
				nb.WriteRune('.')
			}
			nb.WriteString(method.Name())

			b.WriteString(strconv.Itoa(nb.Len()))
			b.WriteRune('_')
			b.WriteString(nb.String())

			ctx.mangleSignature(method.Type().(*types.Signature), nil, b)
		}
		b.WriteRune('e')

	default:
		panic(fmt.Sprintf("unhandled type: %#v", t))
	}
}

func (ctx *manglerContext) mangleTypeDescriptorName(t types.Type, b *bytes.Buffer) {
	switch t := t.(type) {
	case *types.Basic, *types.Named:
		b.WriteString("__go_tdn_")
		ti := ctx.getNamedTypeInfo(t)
		if ti.pkgpath != "" {
			b.WriteString(ManglePackagePath(ti.pkgpath))
			b.WriteRune('.')
		}
		if ti.functionName != "" {
			b.WriteString(ti.functionName)
			b.WriteRune('.')
			if ti.scopeNum != 0 {
				b.WriteString(strconv.Itoa(ti.scopeNum))
				b.WriteRune('.')
			}
		}
		b.WriteString(ti.name)

	default:
		b.WriteString("__go_td_")
		ctx.mangleType(t, b)
	}
}

func (ctx *manglerContext) mangleMapDescriptorName(t types.Type, b *bytes.Buffer) {
	b.WriteString("__go_map_")
	ctx.mangleType(t, b)
}

func (ctx *manglerContext) mangleImtName(srctype types.Type, targettype *types.Interface, b *bytes.Buffer) {
	b.WriteString("__go_imt_")
	ctx.mangleType(targettype, b)
	b.WriteString("__")
	ctx.mangleType(srctype, b)
}

func (ctx *manglerContext) mangleHashFunctionName(t types.Type) string {
	var b bytes.Buffer
	b.WriteString("__go_type_hash_")
	ctx.mangleType(t, &b)
	return b.String()
}

func (ctx *manglerContext) mangleEqualFunctionName(t types.Type) string {
	var b bytes.Buffer
	b.WriteString("__go_type_equal_")
	ctx.mangleType(t, &b)
	return b.String()
}

func (ctx *manglerContext) mangleFunctionName(f *ssa.Function) string {
	var b bytes.Buffer

	if f.Parent() != nil {
		// Anonymous functions are not guaranteed to
		// have unique identifiers at the global scope.
		b.WriteString(ctx.mangleFunctionName(f.Parent()))
		b.WriteRune(':')
		b.WriteString(f.String())
		return b.String()
	}

	// Synthetic bound and thunk functions are special cases; they can only be
	// distinguished using private data that is only exposed via String().
	if strings.HasSuffix(f.Name(), "$bound") || strings.HasSuffix(f.Name(), "$thunk") {
		b.WriteString(f.String())
		return b.String()
	}

	var pkg *types.Package
	if f.Pkg != nil {
		pkg = f.Pkg.Object
	} else if !f.Object().Exported() {
		pkg = f.Object().Pkg()
	}

	if pkg != nil {
		b.WriteString(ManglePackagePath(pkg.Path()))
		b.WriteRune('.')
	}

	if f.Signature.Recv() == nil && f.Name() == "init" {
		b.WriteString(".import")
	} else {
		b.WriteString(f.Name())
	}
	if f.Signature.Recv() != nil {
		b.WriteRune('.')
		ctx.mangleType(f.Signature.Recv().Type(), &b)
	}

	return b.String()
}

func (ctx *manglerContext) mangleGlobalName(g *ssa.Global) string {
	var b bytes.Buffer

	b.WriteString(ManglePackagePath(g.Pkg.Object.Path()))
	b.WriteRune('.')
	b.WriteString(g.Name())

	return b.String()
}

const (
	// From gofrontend/types.h
	gccgoTypeClassERROR = iota
	gccgoTypeClassVOID
	gccgoTypeClassBOOLEAN
	gccgoTypeClassINTEGER
	gccgoTypeClassFLOAT
	gccgoTypeClassCOMPLEX
	gccgoTypeClassSTRING
	gccgoTypeClassSINK
	gccgoTypeClassFUNCTION
	gccgoTypeClassPOINTER
	gccgoTypeClassNIL
	gccgoTypeClassCALL_MULTIPLE_RESULT
	gccgoTypeClassSTRUCT
	gccgoTypeClassARRAY
	gccgoTypeClassMAP
	gccgoTypeClassCHANNEL
	gccgoTypeClassINTERFACE
	gccgoTypeClassNAMED
	gccgoTypeClassFORWARD
)

func getStringHash(s string, h uint32) uint32 {
	for _, c := range []byte(s) {
		h ^= uint32(c)
		h += 16777619
	}
	return h
}

func (tm *TypeMap) getTypeHash(t types.Type) uint32 {
	switch t := t.(type) {
	case *types.Basic, *types.Named:
		nti := tm.mc.getNamedTypeInfo(t)
		h := getStringHash(nti.functionName+nti.name+nti.pkgpath, 0)
		h ^= uint32(nti.scopeNum)
		return gccgoTypeClassNAMED + h

	case *types.Signature:
		var h uint32

		p := t.Params()
		for i := 0; i != p.Len(); i++ {
			h += tm.getTypeHash(p.At(i).Type()) << uint32(i+1)
		}

		r := t.Results()
		for i := 0; i != r.Len(); i++ {
			h += tm.getTypeHash(r.At(i).Type()) << uint32(i+2)
		}

		if t.Variadic() {
			h += 1
		}
		h <<= 4
		return gccgoTypeClassFUNCTION + h

	case *types.Pointer:
		return gccgoTypeClassPOINTER + (tm.getTypeHash(t.Elem()) << 4)

	case *types.Struct:
		var h uint32
		for i := 0; i != t.NumFields(); i++ {
			h = (h << 1) + tm.getTypeHash(t.Field(i).Type())
		}
		h <<= 2
		return gccgoTypeClassSTRUCT + h

	case *types.Array:
		return gccgoTypeClassARRAY + tm.getTypeHash(t.Elem()) + 1

	case *types.Slice:
		return gccgoTypeClassARRAY + tm.getTypeHash(t.Elem()) + 1

	case *types.Map:
		return gccgoTypeClassMAP + tm.getTypeHash(t.Key()) + tm.getTypeHash(t.Elem()) + 2

	case *types.Chan:
		var h uint32

		switch t.Dir() {
		case types.SendOnly:
			h = 1
		case types.RecvOnly:
			h = 2
		case types.SendRecv:
			h = 3
		}

		h += tm.getTypeHash(t.Elem()) << 2
		h <<= 3
		return gccgoTypeClassCHANNEL + h

	case *types.Interface:
		var h uint32
		for _, m := range orderedMethodSet(tm.MethodSet(t)) {
			h = getStringHash(m.Obj().Name(), h)
			h <<= 1
		}
		return gccgoTypeClassINTERFACE + h

	default:
		panic(fmt.Sprintf("unhandled type: %#v", t))
	}
}

func (tm *TypeMap) writeType(typ types.Type, b *bytes.Buffer) {
	switch t := typ.(type) {
	case *types.Basic, *types.Named:
		ti := tm.mc.getNamedTypeInfo(t)
		if ti.pkgpath != "" {
			b.WriteByte('\t')
			b.WriteString(ManglePackagePath(ti.pkgpath))
			b.WriteByte('\t')
			b.WriteString(ti.pkgname)
			b.WriteByte('.')
		}
		if ti.functionName != "" {
			b.WriteByte('\t')
			b.WriteString(ti.functionName)
			b.WriteByte('$')
			if ti.scopeNum != 0 {
				b.WriteString(strconv.Itoa(ti.scopeNum))
				b.WriteByte('$')
			}
			b.WriteByte('\t')
		}
		b.WriteString(ti.name)

	case *types.Array:
		fmt.Fprintf(b, "[%d]", t.Len())
		tm.writeType(t.Elem(), b)

	case *types.Slice:
		b.WriteString("[]")
		tm.writeType(t.Elem(), b)

	case *types.Struct:
		if t.NumFields() == 0 {
			b.WriteString("struct {}")
			return
		}
		b.WriteString("struct { ")
		for i := 0; i != t.NumFields(); i++ {
			f := t.Field(i)
			if i > 0 {
				b.WriteString("; ")
			}
			if !f.Anonymous() {
				b.WriteString(f.Name())
				b.WriteByte(' ')
			}
			tm.writeType(f.Type(), b)
			if tag := t.Tag(i); tag != "" {
				fmt.Fprintf(b, " %q", tag)
			}
		}
		b.WriteString(" }")

	case *types.Pointer:
		b.WriteByte('*')
		tm.writeType(t.Elem(), b)

	case *types.Signature:
		b.WriteString("func")
		tm.writeSignature(t, b)

	case *types.Interface:
		if t.NumMethods() == 0 && t.NumEmbeddeds() == 0 {
			b.WriteString("interface {}")
			return
		}
		// We write the source-level methods and embedded types rather
		// than the actual method set since resolved method signatures
		// may have non-printable cycles if parameters have anonymous
		// interface types that (directly or indirectly) embed the
		// current interface. For instance, consider the result type
		// of m:
		//
		//     type T interface{
		//         m() interface{ T }
		//     }
		//
		b.WriteString("interface { ")
		// print explicit interface methods and embedded types
		for i := 0; i != t.NumMethods(); i++ {
			m := t.Method(i)
			if i > 0 {
				b.WriteString("; ")
			}
			if !m.Exported() {
				b.WriteString(m.Pkg().Path())
				b.WriteByte('.')
			}
			b.WriteString(m.Name())
			tm.writeSignature(m.Type().(*types.Signature), b)
		}
		for i := 0; i != t.NumEmbeddeds(); i++ {
			typ := t.Embedded(i)
			if i > 0 || t.NumMethods() > 0 {
				b.WriteString("; ")
			}
			tm.writeType(typ, b)
		}
		b.WriteString(" }")

	case *types.Map:
		b.WriteString("map[")
		tm.writeType(t.Key(), b)
		b.WriteByte(']')
		tm.writeType(t.Elem(), b)

	case *types.Chan:
		var s string
		var parens bool
		switch t.Dir() {
		case types.SendRecv:
			s = "chan "
			// chan (<-chan T) requires parentheses
			if c, _ := t.Elem().(*types.Chan); c != nil && c.Dir() == types.RecvOnly {
				parens = true
			}
		case types.SendOnly:
			s = "chan<- "
		case types.RecvOnly:
			s = "<-chan "
		default:
			panic("unreachable")
		}
		b.WriteString(s)
		if parens {
			b.WriteByte('(')
		}
		tm.writeType(t.Elem(), b)
		if parens {
			b.WriteByte(')')
		}

	default:
		panic(fmt.Sprintf("unhandled type: %#v", t))
	}
}

func (tm *TypeMap) writeTuple(tup *types.Tuple, variadic bool, b *bytes.Buffer) {
	b.WriteByte('(')
	if tup != nil {
		for i := 0; i != tup.Len(); i++ {
			v := tup.At(i)
			if i > 0 {
				b.WriteString(", ")
			}
			typ := v.Type()
			if variadic && i == tup.Len()-1 {
				b.WriteString("...")
				typ = typ.(*types.Slice).Elem()
			}
			tm.writeType(typ, b)
		}
	}
	b.WriteByte(')')
}

func (tm *TypeMap) writeSignature(sig *types.Signature, b *bytes.Buffer) {
	tm.writeTuple(sig.Params(), sig.Variadic(), b)

	n := sig.Results().Len()
	if n == 0 {
		// no result
		return
	}

	b.WriteByte(' ')
	if n == 1 {
		tm.writeType(sig.Results().At(0).Type(), b)
		return
	}

	// multiple results
	tm.writeTuple(sig.Results(), false, b)
}

func (tm *TypeMap) getTypeDescType(t types.Type) llvm.Type {
	switch t.Underlying().(type) {
	case *types.Basic:
		return tm.commonTypeType
	case *types.Pointer:
		return tm.ptrTypeType
	case *types.Signature:
		return tm.funcTypeType
	case *types.Array:
		return tm.arrayTypeType
	case *types.Slice:
		return tm.sliceTypeType
	case *types.Map:
		return tm.mapTypeType
	case *types.Chan:
		return tm.chanTypeType
	case *types.Struct:
		return tm.structTypeType
	case *types.Interface:
		return tm.interfaceTypeType
	default:
		panic(fmt.Sprintf("unhandled type: %#v", t))
	}
}

func (tm *TypeMap) getNamedTypeLinkage(nt *types.Named) (linkage llvm.Linkage, emit bool) {
	if pkg := nt.Obj().Pkg(); pkg != nil {
		linkage = llvm.ExternalLinkage
		emit = pkg.Path() == tm.pkgpath
	} else {
		linkage = llvm.LinkOnceODRLinkage
		emit = true
	}

	return
}

func (tm *TypeMap) getTypeDescLinkage(t types.Type) (linkage llvm.Linkage, emit bool) {
	switch t := t.(type) {
	case *types.Named:
		linkage, emit = tm.getNamedTypeLinkage(t)

	case *types.Pointer:
		elem := t.Elem()
		if nt, ok := elem.(*types.Named); ok {
			// Thanks to the ptrToThis member, pointers to named types appear
			// in exactly the same objects as the named types themselves, so
			// we can give them the same linkage.
			linkage, emit = tm.getNamedTypeLinkage(nt)
			return
		}
		linkage = llvm.LinkOnceODRLinkage
		emit = true

	default:
		linkage = llvm.LinkOnceODRLinkage
		emit = true
	}

	return
}

type typeAndInfo struct {
	typ        types.Type
	typeString string
	tdi        *typeDescInfo
}

type byTypeName []typeAndInfo

func (ts byTypeName) Len() int { return len(ts) }
func (ts byTypeName) Swap(i, j int) {
	ts[i], ts[j] = ts[j], ts[i]
}
func (ts byTypeName) Less(i, j int) bool {
	return ts[i].typeString < ts[j].typeString
}

func (tm *TypeMap) emitTypeDescInitializers() {
	var maxSize, maxAlign int64
	maxAlign = 1

	for changed := true; changed; {
		changed = false

		var ts []typeAndInfo

		tm.types.Iterate(func(key types.Type, value interface{}) {
			tdi := value.(*typeDescInfo)
			if tdi.global.Initializer().C == nil {
				linkage, emit := tm.getTypeDescLinkage(key)
				tdi.global.SetLinkage(linkage)
				tdi.gc.SetLinkage(linkage)
				if emit {
					changed = true
					ts = append(ts, typeAndInfo{key, key.String(), tdi})
				}
			}
		})

		if changed {
			sort.Sort(byTypeName(ts))
			for _, t := range ts {
				tm.emitTypeDescInitializer(t.typ, t.tdi)
				if size := tm.Sizeof(t.typ); size > maxSize {
					maxSize = size
				}
				if align := tm.Alignof(t.typ); align > maxAlign {
					maxAlign = align
				}
			}
		}
	}
}

const (
	// From libgo/runtime/mgc0.h
	gcOpcodeEND = iota
	gcOpcodePTR
	gcOpcodeAPTR
	gcOpcodeARRAY_START
	gcOpcodeARRAY_NEXT
	gcOpcodeCALL
	gcOpcodeCHAN_PTR
	gcOpcodeSTRING
	gcOpcodeEFACE
	gcOpcodeIFACE
	gcOpcodeSLICE
	gcOpcodeREGION

	gcStackCapacity = 8
)

func (tm *TypeMap) makeGcInst(val int64) llvm.Value {
	c := llvm.ConstInt(tm.inttype, uint64(val), false)
	return llvm.ConstIntToPtr(c, llvm.PointerType(tm.ctx.Int8Type(), 0))
}

func (tm *TypeMap) appendGcInsts(insts []llvm.Value, t types.Type, offset, stackSize int64) []llvm.Value {
	switch u := t.Underlying().(type) {
	case *types.Basic:
		switch u.Kind() {
		case types.String:
			insts = append(insts, tm.makeGcInst(gcOpcodeSTRING), tm.makeGcInst(offset))
		case types.UnsafePointer:
			insts = append(insts, tm.makeGcInst(gcOpcodeAPTR), tm.makeGcInst(offset))
		}
	case *types.Pointer:
		insts = append(insts, tm.makeGcInst(gcOpcodePTR), tm.makeGcInst(offset),
			tm.getGcPointer(u.Elem()))
	case *types.Signature, *types.Map:
		insts = append(insts, tm.makeGcInst(gcOpcodeAPTR), tm.makeGcInst(offset))
	case *types.Array:
		if u.Len() == 0 {
			return insts
		} else if stackSize >= gcStackCapacity {
			insts = append(insts, tm.makeGcInst(gcOpcodeREGION), tm.makeGcInst(offset),
				tm.makeGcInst(tm.Sizeof(t)), tm.getGcPointer(t))
		} else {
			insts = append(insts, tm.makeGcInst(gcOpcodeARRAY_START), tm.makeGcInst(offset),
				tm.makeGcInst(u.Len()), tm.makeGcInst(tm.Sizeof(u.Elem())))
			insts = tm.appendGcInsts(insts, u.Elem(), 0, stackSize+1)
			insts = append(insts, tm.makeGcInst(gcOpcodeARRAY_NEXT))
		}
	case *types.Slice:
		if tm.Sizeof(u.Elem()) == 0 {
			insts = append(insts, tm.makeGcInst(gcOpcodeAPTR), tm.makeGcInst(offset))
		} else {
			insts = append(insts, tm.makeGcInst(gcOpcodeSLICE), tm.makeGcInst(offset),
				tm.getGcPointer(u.Elem()))
		}
	case *types.Chan:
		insts = append(insts, tm.makeGcInst(gcOpcodeCHAN_PTR), tm.makeGcInst(offset),
			tm.ToRuntime(t))
	case *types.Struct:
		fields := make([]*types.Var, u.NumFields())
		for i := range fields {
			fields[i] = u.Field(i)
		}
		offsets := tm.Offsetsof(fields)

		for i, field := range fields {
			insts = tm.appendGcInsts(insts, field.Type(), offset+offsets[i], stackSize)
		}
	case *types.Interface:
		if u.NumMethods() == 0 {
			insts = append(insts, tm.makeGcInst(gcOpcodeEFACE), tm.makeGcInst(offset))
		} else {
			insts = append(insts, tm.makeGcInst(gcOpcodeIFACE), tm.makeGcInst(offset))
		}
	default:
		panic(fmt.Sprintf("unhandled type: %#v", t))
	}

	return insts
}

func (tm *TypeMap) emitTypeDescInitializer(t types.Type, tdi *typeDescInfo) {
	// initialize type descriptor
	tdi.global.SetInitializer(tm.makeTypeDescInitializer(t))

	// initialize GC program
	insts := []llvm.Value{tm.makeGcInst(tm.Sizeof(t))}
	insts = tm.appendGcInsts(insts, t, 0, 0)
	insts = append(insts, tm.makeGcInst(gcOpcodeEND))

	i8ptr := llvm.PointerType(llvm.Int8Type(), 0)
	instArray := llvm.ConstArray(i8ptr, insts)

	newGc := llvm.AddGlobal(tm.module, instArray.Type(), "")
	newGc.SetGlobalConstant(true)
	newGc.SetInitializer(instArray)
	gcName := tdi.gc.Name()
	tdi.gc.SetName("")
	newGc.SetName(gcName)
	newGc.SetLinkage(tdi.gc.Linkage())

	tdi.gc.ReplaceAllUsesWith(llvm.ConstBitCast(newGc, tdi.gc.Type()))
	tdi.gc.EraseFromParentAsGlobal()
	tdi.gc = llvm.Value{nil}
	tdi.gcPtr = llvm.ConstBitCast(newGc, i8ptr)
}

func (tm *TypeMap) makeTypeDescInitializer(t types.Type) llvm.Value {
	switch u := t.Underlying().(type) {
	case *types.Basic:
		return tm.makeBasicType(t, u)
	case *types.Pointer:
		return tm.makePointerType(t, u)
	case *types.Signature:
		return tm.makeFuncType(t, u)
	case *types.Array:
		return tm.makeArrayType(t, u)
	case *types.Slice:
		return tm.makeSliceType(t, u)
	case *types.Map:
		return tm.makeMapType(t, u)
	case *types.Chan:
		return tm.makeChanType(t, u)
	case *types.Struct:
		return tm.makeStructType(t, u)
	case *types.Interface:
		return tm.makeInterfaceType(t, u)
	default:
		panic(fmt.Sprintf("unhandled type: %#v", t))
	}
}

func (tm *TypeMap) getStructAlgorithms(st *types.Struct) algorithms {
	if algs, ok := tm.algs.At(st).(algorithms); ok {
		return algs
	}

	hashes := make([]llvm.Value, st.NumFields())
	equals := make([]llvm.Value, st.NumFields())

	for i := range hashes {
		algs := tm.getAlgorithms(st.Field(i).Type())
		if algs.hashDescriptor == tm.algsError.hashDescriptor {
			return algs
		}
		hashes[i], equals[i] = algs.hash, algs.equal
	}

	i8ptr := llvm.PointerType(tm.ctx.Int8Type(), 0)
	llsptrty := llvm.PointerType(tm.ToLLVM(st), 0)

	builder := tm.ctx.NewBuilder()
	defer builder.Dispose()

	hashFunctionName := tm.mc.mangleHashFunctionName(st)
	hash := llvm.AddFunction(tm.module, hashFunctionName, tm.hashFnType)
	hash.SetLinkage(llvm.LinkOnceODRLinkage)
	hashDescriptor := tm.createAlgorithmDescriptor(hashFunctionName+"_descriptor", hash)

	builder.SetInsertPointAtEnd(llvm.AddBasicBlock(hash, "entry"))
	sptr := builder.CreateBitCast(hash.Param(0), llsptrty, "")

	hashval := llvm.ConstNull(tm.inttype)
	i33 := llvm.ConstInt(tm.inttype, 33, false)

	for i, fhash := range hashes {
		fptr := builder.CreateStructGEP(sptr, i, "")
		fptr = builder.CreateBitCast(fptr, i8ptr, "")
		fsize := llvm.ConstInt(tm.inttype, uint64(tm.sizes.Sizeof(st.Field(i).Type())), false)
		hashcall := builder.CreateCall(fhash, []llvm.Value{fptr, fsize}, "")
		hashval = builder.CreateMul(hashval, i33, "")
		hashval = builder.CreateAdd(hashval, hashcall, "")
	}

	builder.CreateRet(hashval)

	equalFunctionName := tm.mc.mangleEqualFunctionName(st)
	equal := llvm.AddFunction(tm.module, equalFunctionName, tm.equalFnType)
	equal.SetLinkage(llvm.LinkOnceODRLinkage)
	equalDescriptor := tm.createAlgorithmDescriptor(equalFunctionName+"_descriptor", equal)

	eqentrybb := llvm.AddBasicBlock(equal, "entry")
	eqretzerobb := llvm.AddBasicBlock(equal, "retzero")
	builder.SetInsertPointAtEnd(eqentrybb)
	s1ptr := builder.CreateBitCast(equal.Param(0), llsptrty, "")
	s2ptr := builder.CreateBitCast(equal.Param(1), llsptrty, "")

	zerobool := llvm.ConstNull(tm.ctx.Int8Type())
	onebool := llvm.ConstInt(tm.ctx.Int8Type(), 1, false)

	for i, fequal := range equals {
		f1ptr := builder.CreateStructGEP(s1ptr, i, "")
		f1ptr = builder.CreateBitCast(f1ptr, i8ptr, "")
		f2ptr := builder.CreateStructGEP(s2ptr, i, "")
		f2ptr = builder.CreateBitCast(f2ptr, i8ptr, "")
		fsize := llvm.ConstInt(tm.inttype, uint64(tm.sizes.Sizeof(st.Field(i).Type())), false)
		equalcall := builder.CreateCall(fequal, []llvm.Value{f1ptr, f2ptr, fsize}, "")
		equaleqzero := builder.CreateICmp(llvm.IntEQ, equalcall, zerobool, "")
		contbb := llvm.AddBasicBlock(equal, "cont")
		builder.CreateCondBr(equaleqzero, eqretzerobb, contbb)
		builder.SetInsertPointAtEnd(contbb)
	}

	builder.CreateRet(onebool)

	builder.SetInsertPointAtEnd(eqretzerobb)
	builder.CreateRet(zerobool)

	algs := algorithms{
		hash:            hash,
		hashDescriptor:  hashDescriptor,
		equal:           equal,
		equalDescriptor: equalDescriptor,
	}
	tm.algs.Set(st, algs)
	return algs
}

func (tm *TypeMap) getArrayAlgorithms(at *types.Array) algorithms {
	if algs, ok := tm.algs.At(at).(algorithms); ok {
		return algs
	}

	elemAlgs := tm.getAlgorithms(at.Elem())
	if elemAlgs.hashDescriptor == tm.algsError.hashDescriptor {
		return elemAlgs
	}

	i8ptr := llvm.PointerType(tm.ctx.Int8Type(), 0)
	llelemty := llvm.PointerType(tm.ToLLVM(at.Elem()), 0)

	i1 := llvm.ConstInt(tm.inttype, 1, false)
	alen := llvm.ConstInt(tm.inttype, uint64(at.Len()), false)
	esize := llvm.ConstInt(tm.inttype, uint64(tm.sizes.Sizeof(at.Elem())), false)

	builder := tm.ctx.NewBuilder()
	defer builder.Dispose()

	hashFunctionName := tm.mc.mangleHashFunctionName(at)
	hash := llvm.AddFunction(tm.module, hashFunctionName, tm.hashFnType)
	hash.SetLinkage(llvm.LinkOnceODRLinkage)
	hashDescriptor := tm.createAlgorithmDescriptor(hashFunctionName+"_descriptor", hash)
	equalFunctionName := tm.mc.mangleHashFunctionName(at)
	equal := llvm.AddFunction(tm.module, equalFunctionName, tm.equalFnType)
	equal.SetLinkage(llvm.LinkOnceODRLinkage)
	equalDescriptor := tm.createAlgorithmDescriptor(equalFunctionName+"_descriptor", equal)
	algs := algorithms{
		hash:            hash,
		hashDescriptor:  hashDescriptor,
		equal:           equal,
		equalDescriptor: equalDescriptor,
	}
	tm.algs.Set(at, algs)

	hashentrybb := llvm.AddBasicBlock(hash, "entry")
	builder.SetInsertPointAtEnd(hashentrybb)
	if at.Len() == 0 {
		builder.CreateRet(llvm.ConstNull(tm.inttype))
	} else {
		i33 := llvm.ConstInt(tm.inttype, 33, false)

		aptr := builder.CreateBitCast(hash.Param(0), llelemty, "")
		loopbb := llvm.AddBasicBlock(hash, "loop")
		builder.CreateBr(loopbb)

		exitbb := llvm.AddBasicBlock(hash, "exit")

		builder.SetInsertPointAtEnd(loopbb)
		indexphi := builder.CreatePHI(tm.inttype, "")
		index := indexphi
		hashvalphi := builder.CreatePHI(tm.inttype, "")
		hashval := hashvalphi

		eptr := builder.CreateGEP(aptr, []llvm.Value{index}, "")
		eptr = builder.CreateBitCast(eptr, i8ptr, "")

		hashcall := builder.CreateCall(elemAlgs.hash, []llvm.Value{eptr, esize}, "")
		hashval = builder.CreateMul(hashval, i33, "")
		hashval = builder.CreateAdd(hashval, hashcall, "")

		index = builder.CreateAdd(index, i1, "")

		indexphi.AddIncoming(
			[]llvm.Value{llvm.ConstNull(tm.inttype), index},
			[]llvm.BasicBlock{hashentrybb, loopbb},
		)
		hashvalphi.AddIncoming(
			[]llvm.Value{llvm.ConstNull(tm.inttype), hashval},
			[]llvm.BasicBlock{hashentrybb, loopbb},
		)

		exit := builder.CreateICmp(llvm.IntEQ, index, alen, "")
		builder.CreateCondBr(exit, exitbb, loopbb)

		builder.SetInsertPointAtEnd(exitbb)
		builder.CreateRet(hashval)
	}

	zerobool := llvm.ConstNull(tm.ctx.Int8Type())
	onebool := llvm.ConstInt(tm.ctx.Int8Type(), 1, false)

	eqentrybb := llvm.AddBasicBlock(equal, "entry")
	builder.SetInsertPointAtEnd(eqentrybb)
	if at.Len() == 0 {
		builder.CreateRet(onebool)
	} else {
		a1ptr := builder.CreateBitCast(equal.Param(0), llelemty, "")
		a2ptr := builder.CreateBitCast(equal.Param(1), llelemty, "")
		loopbb := llvm.AddBasicBlock(equal, "loop")
		builder.CreateBr(loopbb)

		exitbb := llvm.AddBasicBlock(equal, "exit")
		retzerobb := llvm.AddBasicBlock(equal, "retzero")

		builder.SetInsertPointAtEnd(loopbb)
		indexphi := builder.CreatePHI(tm.inttype, "")
		index := indexphi

		e1ptr := builder.CreateGEP(a1ptr, []llvm.Value{index}, "")
		e1ptr = builder.CreateBitCast(e1ptr, i8ptr, "")
		e2ptr := builder.CreateGEP(a2ptr, []llvm.Value{index}, "")
		e2ptr = builder.CreateBitCast(e2ptr, i8ptr, "")

		equalcall := builder.CreateCall(elemAlgs.equal, []llvm.Value{e1ptr, e2ptr, esize}, "")
		equaleqzero := builder.CreateICmp(llvm.IntEQ, equalcall, zerobool, "")

		contbb := llvm.AddBasicBlock(equal, "cont")
		builder.CreateCondBr(equaleqzero, retzerobb, contbb)

		builder.SetInsertPointAtEnd(contbb)

		index = builder.CreateAdd(index, i1, "")

		indexphi.AddIncoming(
			[]llvm.Value{llvm.ConstNull(tm.inttype), index},
			[]llvm.BasicBlock{eqentrybb, contbb},
		)

		exit := builder.CreateICmp(llvm.IntEQ, index, alen, "")
		builder.CreateCondBr(exit, exitbb, loopbb)

		builder.SetInsertPointAtEnd(exitbb)
		builder.CreateRet(onebool)

		builder.SetInsertPointAtEnd(retzerobb)
		builder.CreateRet(zerobool)
	}

	return algs
}

func (tm *TypeMap) createAlgorithmDescriptor(name string, fn llvm.Value) llvm.Value {
	d := llvm.AddGlobal(tm.module, tm.funcValType, name)
	d.SetLinkage(llvm.LinkOnceODRLinkage)
	d.SetGlobalConstant(true)
	fn = llvm.ConstBitCast(fn, tm.funcValType.StructElementTypes()[0])
	init := llvm.ConstNull(tm.funcValType)
	init = llvm.ConstInsertValue(init, fn, []uint32{0})
	d.SetInitializer(init)
	return d
}

func (tm *TypeMap) getAlgorithms(t types.Type) algorithms {
	switch t := t.Underlying().(type) {
	case *types.Interface:
		if t.NumMethods() == 0 {
			return tm.algsEmptyInterface
		}
		return tm.algsInterface
	case *types.Basic:
		switch t.Kind() {
		case types.Float32, types.Float64:
			return tm.algsFloat
		case types.Complex64, types.Complex128:
			return tm.algsComplex
		case types.String:
			return tm.algsString
		}
		return tm.algsIdentity
	case *types.Signature, *types.Map, *types.Slice:
		return tm.algsError
	case *types.Struct:
		return tm.getStructAlgorithms(t)
	case *types.Array:
		return tm.getArrayAlgorithms(t)
	}
	return tm.algsIdentity
}

func (tm *TypeMap) getTypeDescInfo(t types.Type) *typeDescInfo {
	if tdi, ok := tm.types.At(t).(*typeDescInfo); ok {
		return tdi
	}

	var b bytes.Buffer
	tm.mc.mangleTypeDescriptorName(t, &b)

	global := llvm.AddGlobal(tm.module, tm.getTypeDescType(t), b.String())
	global.SetGlobalConstant(true)
	ptr := llvm.ConstBitCast(global, llvm.PointerType(tm.commonTypeType, 0))

	gc := llvm.AddGlobal(tm.module, llvm.PointerType(llvm.Int8Type(), 0), b.String()+"$gc")
	gc.SetGlobalConstant(true)
	gcPtr := llvm.ConstBitCast(gc, llvm.PointerType(tm.ctx.Int8Type(), 0))

	var mapDescPtr llvm.Value
	if m, ok := t.Underlying().(*types.Map); ok {
		var mapb bytes.Buffer
		tm.mc.mangleMapDescriptorName(t, &mapb)

		mapDescPtr = llvm.AddGlobal(tm.module, tm.mapDescType, mapb.String())
		mapDescPtr.SetGlobalConstant(true)
		mapDescPtr.SetLinkage(llvm.LinkOnceODRLinkage)
		mapDescPtr.SetInitializer(tm.makeMapDesc(ptr, m))
	}

	tdi := &typeDescInfo{
		global:        global,
		commonTypePtr: ptr,
		mapDescPtr:    mapDescPtr,
		gc:            gc,
		gcPtr:         gcPtr,
	}
	tm.types.Set(t, tdi)
	return tdi
}

func (tm *TypeMap) getTypeDescriptorPointer(t types.Type) llvm.Value {
	return tm.getTypeDescInfo(t).commonTypePtr
}

func (tm *TypeMap) getMapDescriptorPointer(t types.Type) llvm.Value {
	return tm.getTypeDescInfo(t).mapDescPtr
}

func (tm *TypeMap) getGcPointer(t types.Type) llvm.Value {
	return tm.getTypeDescInfo(t).gcPtr
}

func (tm *TypeMap) getItabPointer(srctype types.Type, targettype *types.Interface) llvm.Value {
	if targettype.NumMethods() == 0 {
		return tm.ToRuntime(srctype)
	} else {
		return tm.getImtPointer(srctype, targettype)
	}
}

func (tm *TypeMap) getImtPointer(srctype types.Type, targettype *types.Interface) llvm.Value {
	tdi := tm.getTypeDescInfo(srctype)

	if ptr, ok := tdi.interfaceMethodTables.At(targettype).(llvm.Value); ok {
		return ptr
	}

	srcms := tm.MethodSet(srctype)
	targetms := tm.MethodSet(targettype)

	i8ptr := llvm.PointerType(llvm.Int8Type(), 0)

	elems := make([]llvm.Value, targetms.Len()+1)
	elems[0] = tm.ToRuntime(srctype)
	for i, targetm := range orderedMethodSet(targetms) {
		srcm := srcms.Lookup(targetm.Obj().Pkg(), targetm.Obj().Name())

		elems[i+1] = tm.methodResolver.ResolveMethod(srcm).value
	}
	imtinit := llvm.ConstArray(i8ptr, elems)

	var b bytes.Buffer
	tm.mc.mangleImtName(srctype, targettype, &b)
	imt := llvm.AddGlobal(tm.module, imtinit.Type(), b.String())
	imt.SetGlobalConstant(true)
	imt.SetInitializer(imtinit)
	imt.SetLinkage(llvm.LinkOnceODRLinkage)

	imtptr := llvm.ConstBitCast(imt, i8ptr)
	tdi.interfaceMethodTables.Set(targettype, imtptr)
	return imtptr
}

const (
	// From gofrontend/types.h
	gccgoRuntimeTypeKindBOOL           = 1
	gccgoRuntimeTypeKindINT            = 2
	gccgoRuntimeTypeKindINT8           = 3
	gccgoRuntimeTypeKindINT16          = 4
	gccgoRuntimeTypeKindINT32          = 5
	gccgoRuntimeTypeKindINT64          = 6
	gccgoRuntimeTypeKindUINT           = 7
	gccgoRuntimeTypeKindUINT8          = 8
	gccgoRuntimeTypeKindUINT16         = 9
	gccgoRuntimeTypeKindUINT32         = 10
	gccgoRuntimeTypeKindUINT64         = 11
	gccgoRuntimeTypeKindUINTPTR        = 12
	gccgoRuntimeTypeKindFLOAT32        = 13
	gccgoRuntimeTypeKindFLOAT64        = 14
	gccgoRuntimeTypeKindCOMPLEX64      = 15
	gccgoRuntimeTypeKindCOMPLEX128     = 16
	gccgoRuntimeTypeKindARRAY          = 17
	gccgoRuntimeTypeKindCHAN           = 18
	gccgoRuntimeTypeKindFUNC           = 19
	gccgoRuntimeTypeKindINTERFACE      = 20
	gccgoRuntimeTypeKindMAP            = 21
	gccgoRuntimeTypeKindPTR            = 22
	gccgoRuntimeTypeKindSLICE          = 23
	gccgoRuntimeTypeKindSTRING         = 24
	gccgoRuntimeTypeKindSTRUCT         = 25
	gccgoRuntimeTypeKindUNSAFE_POINTER = 26
	gccgoRuntimeTypeKindDIRECT_IFACE   = (1 << 5)
	gccgoRuntimeTypeKindNO_POINTERS    = (1 << 7)
)

func hasPointers(t types.Type) bool {
	switch t := t.(type) {
	case *types.Basic:
		return t.Kind() == types.String || t.Kind() == types.UnsafePointer

	case *types.Signature, *types.Pointer, *types.Slice, *types.Map, *types.Chan, *types.Interface:
		return true

	case *types.Struct:
		for i := 0; i != t.NumFields(); i++ {
			if hasPointers(t.Field(i).Type()) {
				return true
			}
		}
		return false

	case *types.Named:
		return hasPointers(t.Underlying())

	case *types.Array:
		return hasPointers(t.Elem())

	default:
		panic("unrecognized type")
	}
}

func runtimeTypeKind(t types.Type) (k uint8) {
	switch t := t.(type) {
	case *types.Basic:
		switch t.Kind() {
		case types.Bool:
			k = gccgoRuntimeTypeKindBOOL
		case types.Int:
			k = gccgoRuntimeTypeKindINT
		case types.Int8:
			k = gccgoRuntimeTypeKindINT8
		case types.Int16:
			k = gccgoRuntimeTypeKindINT16
		case types.Int32:
			k = gccgoRuntimeTypeKindINT32
		case types.Int64:
			k = gccgoRuntimeTypeKindINT64
		case types.Uint:
			k = gccgoRuntimeTypeKindUINT
		case types.Uint8:
			k = gccgoRuntimeTypeKindUINT8
		case types.Uint16:
			k = gccgoRuntimeTypeKindUINT16
		case types.Uint32:
			k = gccgoRuntimeTypeKindUINT32
		case types.Uint64:
			k = gccgoRuntimeTypeKindUINT64
		case types.Uintptr:
			k = gccgoRuntimeTypeKindUINTPTR
		case types.Float32:
			k = gccgoRuntimeTypeKindFLOAT32
		case types.Float64:
			k = gccgoRuntimeTypeKindFLOAT64
		case types.Complex64:
			k = gccgoRuntimeTypeKindCOMPLEX64
		case types.Complex128:
			k = gccgoRuntimeTypeKindCOMPLEX128
		case types.String:
			k = gccgoRuntimeTypeKindSTRING
		case types.UnsafePointer:
			k = gccgoRuntimeTypeKindUNSAFE_POINTER | gccgoRuntimeTypeKindDIRECT_IFACE
		default:
			panic("unrecognized builtin type")
		}
	case *types.Array:
		k = gccgoRuntimeTypeKindARRAY
	case *types.Slice:
		k = gccgoRuntimeTypeKindSLICE
	case *types.Struct:
		k = gccgoRuntimeTypeKindSTRUCT
	case *types.Pointer:
		k = gccgoRuntimeTypeKindPTR | gccgoRuntimeTypeKindDIRECT_IFACE
	case *types.Signature:
		k = gccgoRuntimeTypeKindFUNC
	case *types.Interface:
		k = gccgoRuntimeTypeKindINTERFACE
	case *types.Map:
		k = gccgoRuntimeTypeKindMAP
	case *types.Chan:
		k = gccgoRuntimeTypeKindCHAN
	case *types.Named:
		return runtimeTypeKind(t.Underlying())
	default:
		panic("unrecognized type")
	}

	if !hasPointers(t) {
		k |= gccgoRuntimeTypeKindNO_POINTERS
	}

	return
}

func (tm *TypeMap) makeCommonType(t types.Type) llvm.Value {
	var vals [11]llvm.Value
	vals[0] = llvm.ConstInt(tm.ctx.Int8Type(), uint64(runtimeTypeKind(t)), false)
	vals[1] = llvm.ConstInt(tm.ctx.Int8Type(), uint64(tm.Alignof(t)), false)
	vals[2] = vals[1]
	vals[3] = llvm.ConstInt(tm.inttype, uint64(tm.Sizeof(t)), false)
	vals[4] = llvm.ConstInt(tm.ctx.Int32Type(), uint64(tm.getTypeHash(t)), false)
	algs := tm.getAlgorithms(t)
	vals[5] = algs.hashDescriptor
	vals[6] = algs.equalDescriptor
	vals[7] = tm.getGcPointer(t)
	var b bytes.Buffer
	tm.writeType(t, &b)
	vals[8] = tm.globalStringPtr(b.String())
	vals[9] = tm.makeUncommonTypePtr(t)
	switch t.(type) {
	case *types.Named, *types.Struct:
		vals[10] = tm.getTypeDescriptorPointer(types.NewPointer(t))
	default:
		vals[10] = llvm.ConstPointerNull(llvm.PointerType(tm.commonTypeType, 0))
	}
	return llvm.ConstNamedStruct(tm.commonTypeType, vals[:])
}

func (tm *TypeMap) makeBasicType(t types.Type, u *types.Basic) llvm.Value {
	return tm.makeCommonType(t)
}

func (tm *TypeMap) makeArrayType(t types.Type, a *types.Array) llvm.Value {
	var vals [4]llvm.Value
	vals[0] = tm.makeCommonType(t)
	vals[1] = tm.getTypeDescriptorPointer(a.Elem())
	vals[2] = tm.getTypeDescriptorPointer(types.NewSlice(a.Elem()))
	vals[3] = llvm.ConstInt(tm.inttype, uint64(a.Len()), false)

	return llvm.ConstNamedStruct(tm.arrayTypeType, vals[:])
}

func (tm *TypeMap) makeSliceType(t types.Type, s *types.Slice) llvm.Value {
	var vals [2]llvm.Value
	vals[0] = tm.makeCommonType(t)
	vals[1] = tm.getTypeDescriptorPointer(s.Elem())

	return llvm.ConstNamedStruct(tm.sliceTypeType, vals[:])
}

func (tm *TypeMap) makeStructType(t types.Type, s *types.Struct) llvm.Value {
	var vals [2]llvm.Value
	vals[0] = tm.makeCommonType(t)

	fieldVars := make([]*types.Var, s.NumFields())
	for i := range fieldVars {
		fieldVars[i] = s.Field(i)
	}
	offsets := tm.Offsetsof(fieldVars)
	structFields := make([]llvm.Value, len(fieldVars))
	for i, field := range fieldVars {
		var sfvals [5]llvm.Value
		if !field.Anonymous() {
			sfvals[0] = tm.globalStringPtr(field.Name())
		} else {
			sfvals[0] = llvm.ConstPointerNull(llvm.PointerType(tm.stringType, 0))
		}
		if !field.Exported() && field.Pkg() != nil {
			sfvals[1] = tm.globalStringPtr(field.Pkg().Path())
		} else {
			sfvals[1] = llvm.ConstPointerNull(llvm.PointerType(tm.stringType, 0))
		}
		sfvals[2] = tm.getTypeDescriptorPointer(field.Type())
		if tag := s.Tag(i); tag != "" {
			sfvals[3] = tm.globalStringPtr(tag)
		} else {
			sfvals[3] = llvm.ConstPointerNull(llvm.PointerType(tm.stringType, 0))
		}
		sfvals[4] = llvm.ConstInt(tm.inttype, uint64(offsets[i]), false)

		structFields[i] = llvm.ConstNamedStruct(tm.structFieldType, sfvals[:])
	}
	vals[1] = tm.makeSlice(structFields, tm.structFieldSliceType)

	return llvm.ConstNamedStruct(tm.structTypeType, vals[:])
}

func (tm *TypeMap) makePointerType(t types.Type, p *types.Pointer) llvm.Value {
	var vals [2]llvm.Value
	vals[0] = tm.makeCommonType(t)
	vals[1] = tm.getTypeDescriptorPointer(p.Elem())

	return llvm.ConstNamedStruct(tm.ptrTypeType, vals[:])
}

func (tm *TypeMap) rtypeSlice(t *types.Tuple) llvm.Value {
	rtypes := make([]llvm.Value, t.Len())
	for i := range rtypes {
		rtypes[i] = tm.getTypeDescriptorPointer(t.At(i).Type())
	}
	return tm.makeSlice(rtypes, tm.typeSliceType)
}

func (tm *TypeMap) makeFuncType(t types.Type, f *types.Signature) llvm.Value {
	var vals [4]llvm.Value
	vals[0] = tm.makeCommonType(t)
	// dotdotdot
	variadic := 0
	if f.Variadic() {
		variadic = 1
	}
	vals[1] = llvm.ConstInt(llvm.Int8Type(), uint64(variadic), false)
	// in
	vals[2] = tm.rtypeSlice(f.Params())
	// out
	vals[3] = tm.rtypeSlice(f.Results())

	return llvm.ConstNamedStruct(tm.funcTypeType, vals[:])
}

func (tm *TypeMap) makeInterfaceType(t types.Type, i *types.Interface) llvm.Value {
	var vals [2]llvm.Value
	vals[0] = tm.makeCommonType(t)

	methodset := tm.MethodSet(i)
	imethods := make([]llvm.Value, methodset.Len())
	for index, ms := range orderedMethodSet(methodset) {
		method := ms.Obj()
		var imvals [3]llvm.Value
		imvals[0] = tm.globalStringPtr(method.Name())
		if !method.Exported() && method.Pkg() != nil {
			imvals[1] = tm.globalStringPtr(method.Pkg().Path())
		} else {
			imvals[1] = llvm.ConstPointerNull(llvm.PointerType(tm.stringType, 0))
		}
		mtyp := method.Type().(*types.Signature)
		mftyp := types.NewSignature(nil, nil, mtyp.Params(), mtyp.Results(), mtyp.Variadic())
		imvals[2] = tm.getTypeDescriptorPointer(mftyp)

		imethods[index] = llvm.ConstNamedStruct(tm.imethodType, imvals[:])
	}
	vals[1] = tm.makeSlice(imethods, tm.imethodSliceType)

	return llvm.ConstNamedStruct(tm.interfaceTypeType, vals[:])
}

func (tm *TypeMap) makeMapType(t types.Type, m *types.Map) llvm.Value {
	var vals [3]llvm.Value
	vals[0] = tm.makeCommonType(t)
	vals[1] = tm.getTypeDescriptorPointer(m.Key())
	vals[2] = tm.getTypeDescriptorPointer(m.Elem())

	return llvm.ConstNamedStruct(tm.mapTypeType, vals[:])
}

func (tm *TypeMap) makeMapDesc(ptr llvm.Value, m *types.Map) llvm.Value {
	mapEntryType := structBType{[]backendType{
		tm.getBackendType(types.Typ[types.UnsafePointer]),
		tm.getBackendType(m.Key()),
		tm.getBackendType(m.Elem()),
	}}.ToLLVM(tm.ctx)

	var vals [4]llvm.Value
	// map_descriptor
	vals[0] = ptr
	// entry_size
	vals[1] = llvm.ConstInt(tm.inttype, tm.target.TypeAllocSize(mapEntryType), false)
	// key_offset
	vals[2] = llvm.ConstInt(tm.inttype, tm.target.ElementOffset(mapEntryType, 1), false)
	// value_offset
	vals[3] = llvm.ConstInt(tm.inttype, tm.target.ElementOffset(mapEntryType, 2), false)

	return llvm.ConstNamedStruct(tm.mapDescType, vals[:])
}

func (tm *TypeMap) makeChanType(t types.Type, c *types.Chan) llvm.Value {
	var vals [3]llvm.Value
	vals[0] = tm.makeCommonType(t)
	vals[1] = tm.getTypeDescriptorPointer(c.Elem())

	// From gofrontend/go/types.cc
	// These bits must match the ones in libgo/runtime/go-type.h.
	var dir int
	switch c.Dir() {
	case types.RecvOnly:
		dir = 1
	case types.SendOnly:
		dir = 2
	case types.SendRecv:
		dir = 3
	}
	vals[2] = llvm.ConstInt(tm.inttype, uint64(dir), false)

	return llvm.ConstNamedStruct(tm.chanTypeType, vals[:])
}

func (tm *TypeMap) makeUncommonTypePtr(t types.Type) llvm.Value {
	_, isbasic := t.(*types.Basic)
	_, isnamed := t.(*types.Named)

	var mset types.MethodSet
	// We store interface methods on the interface type.
	if _, ok := t.Underlying().(*types.Interface); !ok {
		mset = *tm.MethodSet(t)
	}

	if !isbasic && !isnamed && mset.Len() == 0 {
		return llvm.ConstPointerNull(llvm.PointerType(tm.uncommonTypeType, 0))
	}

	var vals [3]llvm.Value

	nullStringPtr := llvm.ConstPointerNull(llvm.PointerType(tm.stringType, 0))
	vals[0] = nullStringPtr
	vals[1] = nullStringPtr

	if isbasic || isnamed {
		nti := tm.mc.getNamedTypeInfo(t)
		vals[0] = tm.globalStringPtr(nti.name)
		if nti.pkgpath != "" {
			path := nti.pkgpath
			if nti.functionName != "" {
				path += "." + nti.functionName
				if nti.scopeNum != 0 {
					path += "$" + strconv.Itoa(nti.scopeNum)
				}
			}
			vals[1] = tm.globalStringPtr(path)
		}
	}

	// Store methods. All methods must be stored, not only exported ones;
	// this is to allow satisfying of interfaces with non-exported methods.
	methods := make([]llvm.Value, mset.Len())
	omset := orderedMethodSet(&mset)
	for i := range methods {
		var mvals [5]llvm.Value

		sel := omset[i]
		mname := sel.Obj().Name()
		mfunc := tm.methodResolver.ResolveMethod(sel)
		ftyp := mfunc.Type().(*types.Signature)

		// name
		mvals[0] = tm.globalStringPtr(mname)

		// pkgPath
		mvals[1] = nullStringPtr
		if pkg := sel.Obj().Pkg(); pkg != nil && !sel.Obj().Exported() {
			mvals[1] = tm.globalStringPtr(pkg.Path())
		}

		// mtyp (method type, no receiver)
		mftyp := types.NewSignature(nil, nil, ftyp.Params(), ftyp.Results(), ftyp.Variadic())
		mvals[2] = tm.getTypeDescriptorPointer(mftyp)

		// typ (function type, with receiver)
		recvparam := types.NewParam(0, nil, "", t)
		params := ftyp.Params()
		rfparams := make([]*types.Var, params.Len()+1)
		rfparams[0] = recvparam
		for i := 0; i != ftyp.Params().Len(); i++ {
			rfparams[i+1] = params.At(i)
		}
		rftyp := types.NewSignature(nil, nil, types.NewTuple(rfparams...), ftyp.Results(), ftyp.Variadic())
		mvals[3] = tm.getTypeDescriptorPointer(rftyp)

		// function
		mvals[4] = mfunc.value

		methods[i] = llvm.ConstNamedStruct(tm.methodType, mvals[:])
	}

	vals[2] = tm.makeSlice(methods, tm.methodSliceType)

	uncommonType := llvm.ConstNamedStruct(tm.uncommonTypeType, vals[:])

	uncommonTypePtr := llvm.AddGlobal(tm.module, tm.uncommonTypeType, "")
	uncommonTypePtr.SetGlobalConstant(true)
	uncommonTypePtr.SetInitializer(uncommonType)
	uncommonTypePtr.SetLinkage(llvm.InternalLinkage)
	return uncommonTypePtr
}

// globalStringPtr returns a *string with the specified value.
func (tm *TypeMap) globalStringPtr(value string) llvm.Value {
	strval := llvm.ConstString(value, false)
	strglobal := llvm.AddGlobal(tm.module, strval.Type(), "")
	strglobal.SetGlobalConstant(true)
	strglobal.SetLinkage(llvm.InternalLinkage)
	strglobal.SetInitializer(strval)
	strglobal = llvm.ConstBitCast(strglobal, llvm.PointerType(llvm.Int8Type(), 0))
	strlen := llvm.ConstInt(tm.inttype, uint64(len(value)), false)
	str := llvm.ConstStruct([]llvm.Value{strglobal, strlen}, false)
	g := llvm.AddGlobal(tm.module, str.Type(), "")
	g.SetGlobalConstant(true)
	g.SetLinkage(llvm.InternalLinkage)
	g.SetInitializer(str)
	return g
}

func (tm *TypeMap) makeNamedSliceType(tname string, elemtyp llvm.Type) llvm.Type {
	t := tm.ctx.StructCreateNamed(tname)
	t.StructSetBody([]llvm.Type{
		llvm.PointerType(elemtyp, 0),
		tm.inttype,
		tm.inttype,
	}, false)
	return t
}

func (tm *TypeMap) makeSlice(values []llvm.Value, slicetyp llvm.Type) llvm.Value {
	ptrtyp := slicetyp.StructElementTypes()[0]
	var globalptr llvm.Value
	if len(values) > 0 {
		array := llvm.ConstArray(ptrtyp.ElementType(), values)
		globalptr = llvm.AddGlobal(tm.module, array.Type(), "")
		globalptr.SetGlobalConstant(true)
		globalptr.SetLinkage(llvm.InternalLinkage)
		globalptr.SetInitializer(array)
		globalptr = llvm.ConstBitCast(globalptr, ptrtyp)
	} else {
		globalptr = llvm.ConstNull(ptrtyp)
	}
	len_ := llvm.ConstInt(tm.inttype, uint64(len(values)), false)
	slice := llvm.ConstNull(slicetyp)
	slice = llvm.ConstInsertValue(slice, globalptr, []uint32{0})
	slice = llvm.ConstInsertValue(slice, len_, []uint32{1})
	slice = llvm.ConstInsertValue(slice, len_, []uint32{2})
	return slice
}

func isGlobalObject(obj types.Object) bool {
	pkg := obj.Pkg()
	return pkg == nil || obj.Parent() == pkg.Scope()
}
