//===- debug.go - debug info builder --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This package builds LLVM debug info from go/* data structures.
//
//===----------------------------------------------------------------------===//

package debug

import (
	"debug/dwarf"
	"fmt"
	"go/token"
	"os"
	"strings"

	"llvm.org/llgo/third_party/gotools/go/ssa"
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llgo/third_party/gotools/go/types/typeutil"

	"llvm.org/llvm/bindings/go/llvm"
)

const (
	// non-standard debug metadata tags
	tagAutoVariable dwarf.Tag = 0x100
	tagArgVariable  dwarf.Tag = 0x101
)

type PrefixMap struct {
	Source, Replacement string
}

// DIBuilder builds debug metadata for Go programs.
type DIBuilder struct {
	// builder is the current builder; there is one per CU.
	builder    *llvm.DIBuilder
	module     llvm.Module
	files      map[*token.File]llvm.Metadata
	cu, fn, lb llvm.Metadata
	fnFile     string
	sizes      types.Sizes
	fset       *token.FileSet
	prefixMaps []PrefixMap
	types      typeutil.Map
	voidType   llvm.Metadata
}

// NewDIBuilder creates a new debug information builder.
func NewDIBuilder(sizes types.Sizes, module llvm.Module, fset *token.FileSet, prefixMaps []PrefixMap) *DIBuilder {
	var d DIBuilder
	d.module = module
	d.files = make(map[*token.File]llvm.Metadata)
	d.sizes = sizes
	d.fset = fset
	d.prefixMaps = prefixMaps
	d.builder = llvm.NewDIBuilder(d.module)
	d.cu = d.createCompileUnit()
	return &d
}

// Destroy destroys the DIBuilder.
func (d *DIBuilder) Destroy() {
	d.builder.Destroy()
}

func (d *DIBuilder) scope() llvm.Metadata {
	if d.lb.C != nil {
		return d.lb
	}
	if d.fn.C != nil {
		return d.fn
	}
	return d.cu
}

func (d *DIBuilder) remapFilePath(path string) string {
	for _, pm := range d.prefixMaps {
		if strings.HasPrefix(path, pm.Source) {
			return pm.Replacement + path[len(pm.Source):]
		}
	}
	return path
}

func (d *DIBuilder) getFile(file *token.File) llvm.Metadata {
	if diFile := d.files[file]; diFile.C != nil {
		return diFile
	}
	diFile := d.builder.CreateFile(d.remapFilePath(file.Name()), "")
	d.files[file] = diFile
	return diFile
}

// createCompileUnit creates and returns debug metadata for the compile
// unit as a whole, using the first file in the file set as a representative
// (the choice of file is arbitrary).
func (d *DIBuilder) createCompileUnit() llvm.Metadata {
	var file *token.File
	d.fset.Iterate(func(f *token.File) bool {
		file = f
		return false
	})
	dir, err := os.Getwd()
	if err != nil {
		panic("could not get current directory: " + err.Error())
	}
	return d.builder.CreateCompileUnit(llvm.DICompileUnit{
		Language: llvm.DW_LANG_Go,
		File:     d.remapFilePath(file.Name()),
		Dir:      dir,
		Producer: "llgo",
	})
}

// PushFunction creates debug metadata for the specified function,
// and pushes it onto the scope stack.
func (d *DIBuilder) PushFunction(fnptr llvm.Value, sig *types.Signature, pos token.Pos) {
	var diFile llvm.Metadata
	var line int
	if file := d.fset.File(pos); file != nil {
		d.fnFile = file.Name()
		diFile = d.getFile(file)
		line = file.Line(pos)
	}
	d.fn = d.builder.CreateFunction(d.scope(), llvm.DIFunction{
		Name:         fnptr.Name(), // TODO(axw) unmangled name?
		LinkageName:  fnptr.Name(),
		File:         diFile,
		Line:         line,
		Type:         d.DIType(sig),
		IsDefinition: true,
	})
	fnptr.SetSubprogram(d.fn)
}

// PopFunction pops the previously pushed function off the scope stack.
func (d *DIBuilder) PopFunction() {
	d.lb = llvm.Metadata{}
	d.fn = llvm.Metadata{}
	d.fnFile = ""
}

// Value creates an llvm.dbg.value call for the specified register value.
func (d *DIBuilder) Value(b llvm.Builder, v ssa.Value, llv llvm.Value, paramIndex int) {
	// TODO(axw)
}

// SetLocation sets the current debug location.
func (d *DIBuilder) SetLocation(b llvm.Builder, pos token.Pos) {
	position := d.fset.Position(pos)
	d.lb = llvm.Metadata{}
	if position.Filename != d.fnFile && position.Filename != "" {
		// This can happen rarely, e.g. in init functions.
		diFile := d.builder.CreateFile(d.remapFilePath(position.Filename), "")
		d.lb = d.builder.CreateLexicalBlockFile(d.scope(), diFile, 0)
	}
	b.SetCurrentDebugLocation(uint(position.Line), uint(position.Column), d.scope(), llvm.Metadata{})
}

// Finalize must be called after all compilation units are translated,
// generating the final debug metadata for the module.
func (d *DIBuilder) Finalize() {
	d.module.AddNamedMetadataOperand(
		"llvm.module.flags",
		llvm.GlobalContext().MDNode([]llvm.Metadata{
			llvm.ConstInt(llvm.Int32Type(), 2, false).ConstantAsMetadata(), // Warn on mismatch
			llvm.GlobalContext().MDString("Dwarf Version"),
			llvm.ConstInt(llvm.Int32Type(), 4, false).ConstantAsMetadata(),
		}),
	)
	d.module.AddNamedMetadataOperand(
		"llvm.module.flags",
		llvm.GlobalContext().MDNode([]llvm.Metadata{
			llvm.ConstInt(llvm.Int32Type(), 1, false).ConstantAsMetadata(), // Error on mismatch
			llvm.GlobalContext().MDString("Debug Info Version"),
			llvm.ConstInt(llvm.Int32Type(), 3, false).ConstantAsMetadata(),
		}),
	)
	d.builder.Finalize()
}

// DIType maps a Go type to DIType debug metadata value.
func (d *DIBuilder) DIType(t types.Type) llvm.Metadata {
	return d.typeDebugDescriptor(t, types.TypeString(nil, t))
}

func (d *DIBuilder) typeDebugDescriptor(t types.Type, name string) llvm.Metadata {
	// Signature needs to be handled specially, to preprocess
	// methods, moving the receiver to the parameter list.
	if t, ok := t.(*types.Signature); ok {
		return d.descriptorSignature(t, name)
	}
	if t == nil {
		if d.voidType.C == nil {
			d.voidType = d.builder.CreateBasicType(llvm.DIBasicType{Name: "void"})
		}
		return d.voidType
	}
	if dt, ok := d.types.At(t).(llvm.Metadata); ok {
		return dt
	}
	dt := d.descriptor(t, name)
	d.types.Set(t, dt)
	return dt
}

func (d *DIBuilder) descriptor(t types.Type, name string) llvm.Metadata {
	switch t := t.(type) {
	case *types.Basic:
		return d.descriptorBasic(t, name)
	case *types.Pointer:
		return d.descriptorPointer(t)
	case *types.Struct:
		return d.descriptorStruct(t, name)
	case *types.Named:
		return d.descriptorNamed(t)
	case *types.Array:
		return d.descriptorArray(t, name)
	case *types.Slice:
		return d.descriptorSlice(t, name)
	case *types.Map:
		return d.descriptorMap(t, name)
	case *types.Chan:
		return d.descriptorChan(t, name)
	case *types.Interface:
		return d.descriptorInterface(t, name)
	default:
		panic(fmt.Sprintf("unhandled type: %T", t))
	}
}

func (d *DIBuilder) descriptorBasic(t *types.Basic, name string) llvm.Metadata {
	switch t.Kind() {
	case types.String:
		return d.typeDebugDescriptor(types.NewStruct([]*types.Var{
			types.NewVar(0, nil, "ptr", types.NewPointer(types.Typ[types.Uint8])),
			types.NewVar(0, nil, "len", types.Typ[types.Int]),
		}, nil), name)
	case types.UnsafePointer:
		return d.builder.CreateBasicType(llvm.DIBasicType{
			Name:        name,
			SizeInBits:  uint64(d.sizes.Sizeof(t) * 8),
			Encoding:    llvm.DW_ATE_unsigned,
		})
	default:
		bt := llvm.DIBasicType{
			Name:        t.String(),
			SizeInBits:  uint64(d.sizes.Sizeof(t) * 8),
		}
		switch bi := t.Info(); {
		case bi&types.IsBoolean != 0:
			bt.Encoding = llvm.DW_ATE_boolean
		case bi&types.IsUnsigned != 0:
			bt.Encoding = llvm.DW_ATE_unsigned
		case bi&types.IsInteger != 0:
			bt.Encoding = llvm.DW_ATE_signed
		case bi&types.IsFloat != 0:
			bt.Encoding = llvm.DW_ATE_float
		case bi&types.IsComplex != 0:
			bt.Encoding = llvm.DW_ATE_imaginary_float
		case bi&types.IsUnsigned != 0:
			bt.Encoding = llvm.DW_ATE_unsigned
		default:
			panic(fmt.Sprintf("unhandled: %#v", t))
		}
		return d.builder.CreateBasicType(bt)
	}
}

func (d *DIBuilder) descriptorPointer(t *types.Pointer) llvm.Metadata {
	return d.builder.CreatePointerType(llvm.DIPointerType{
		Pointee:     d.DIType(t.Elem()),
		SizeInBits:  uint64(d.sizes.Sizeof(t) * 8),
		AlignInBits: uint32(d.sizes.Alignof(t) * 8),
	})
}

func (d *DIBuilder) descriptorStruct(t *types.Struct, name string) llvm.Metadata {
	fields := make([]*types.Var, t.NumFields())
	for i := range fields {
		fields[i] = t.Field(i)
	}
	offsets := d.sizes.Offsetsof(fields)
	members := make([]llvm.Metadata, len(fields))
	for i, f := range fields {
		// TODO(axw) file/line where member is defined.
		t := f.Type()
		members[i] = d.builder.CreateMemberType(d.cu, llvm.DIMemberType{
			Name:         f.Name(),
			Type:         d.DIType(t),
			SizeInBits:   uint64(d.sizes.Sizeof(t) * 8),
			AlignInBits:  uint32(d.sizes.Alignof(t) * 8),
			OffsetInBits: uint64(offsets[i] * 8),
		})
	}
	// TODO(axw) file/line where struct is defined.
	return d.builder.CreateStructType(d.cu, llvm.DIStructType{
		Name:        name,
		SizeInBits:  uint64(d.sizes.Sizeof(t) * 8),
		AlignInBits: uint32(d.sizes.Alignof(t) * 8),
		Elements:    members,
	})
}

func (d *DIBuilder) descriptorNamed(t *types.Named) llvm.Metadata {
	var diFile llvm.Metadata
	var line int
	if file := d.fset.File(t.Obj().Pos()); file != nil {
		line = file.Line(t.Obj().Pos())
		diFile = d.getFile(file)
	}

	// Create a placeholder for the named type, to terminate cycles.
	name := t.Obj().Name()
	placeholder := d.builder.CreateReplaceableCompositeType(d.scope(), llvm.DIReplaceableCompositeType{
		Tag:  dwarf.TagStructType,
		Name: name,
		File: diFile,
		Line: line,
	})
	d.types.Set(t, placeholder)

	typedef := d.builder.CreateTypedef(llvm.DITypedef{
		Type: d.DIType(t.Underlying()),
		Name: name,
		File: diFile,
		Line: line,
	})
	placeholder.ReplaceAllUsesWith(typedef)
	return typedef
}

func (d *DIBuilder) descriptorArray(t *types.Array, name string) llvm.Metadata {
	return d.builder.CreateArrayType(llvm.DIArrayType{
		SizeInBits:  uint64(d.sizes.Sizeof(t) * 8),
		AlignInBits: uint32(d.sizes.Alignof(t) * 8),
		ElementType: d.DIType(t.Elem()),
		Subscripts:  []llvm.DISubrange{{Count: t.Len()}},
	})
}

func (d *DIBuilder) descriptorSlice(t *types.Slice, name string) llvm.Metadata {
	sliceStruct := types.NewStruct([]*types.Var{
		types.NewVar(0, nil, "ptr", types.NewPointer(t.Elem())),
		types.NewVar(0, nil, "len", types.Typ[types.Int]),
		types.NewVar(0, nil, "cap", types.Typ[types.Int]),
	}, nil)
	return d.typeDebugDescriptor(sliceStruct, name)
}

func (d *DIBuilder) descriptorMap(t *types.Map, name string) llvm.Metadata {
	// FIXME: This should be DW_TAG_pointer_type to __go_map.
	return d.descriptorBasic(types.Typ[types.Uintptr], name)
}

func (d *DIBuilder) descriptorChan(t *types.Chan, name string) llvm.Metadata {
	// FIXME: This should be DW_TAG_pointer_type to __go_channel.
	return d.descriptorBasic(types.Typ[types.Uintptr], name)
}

func (d *DIBuilder) descriptorInterface(t *types.Interface, name string) llvm.Metadata {
	ifaceStruct := types.NewStruct([]*types.Var{
		types.NewVar(0, nil, "type", types.NewPointer(types.Typ[types.Uint8])),
		types.NewVar(0, nil, "data", types.NewPointer(types.Typ[types.Uint8])),
	}, nil)
	return d.typeDebugDescriptor(ifaceStruct, name)
}

func (d *DIBuilder) descriptorSignature(t *types.Signature, name string) llvm.Metadata {
	// If there's a receiver change the receiver to an
	// additional (first) parameter, and take the value of
	// the resulting signature instead.
	if recv := t.Recv(); recv != nil {
		params := t.Params()
		paramvars := make([]*types.Var, int(params.Len()+1))
		paramvars[0] = recv
		for i := 0; i < int(params.Len()); i++ {
			paramvars[i+1] = params.At(i)
		}
		params = types.NewTuple(paramvars...)
		t := types.NewSignature(nil, nil, params, t.Results(), t.Variadic())
		return d.typeDebugDescriptor(t, name)
	}
	if dt, ok := d.types.At(t).(llvm.Metadata); ok {
		return dt
	}

	var returnType llvm.Metadata
	results := t.Results()
	switch n := results.Len(); n {
	case 0:
		returnType = d.DIType(nil) // void
	case 1:
		returnType = d.DIType(results.At(0).Type())
	default:
		fields := make([]*types.Var, results.Len())
		for i := range fields {
			f := results.At(i)
			// Structs may not have multiple fields
			// with the same name, excepting "_".
			if f.Name() == "" {
				f = types.NewVar(f.Pos(), f.Pkg(), "_", f.Type())
			}
			fields[i] = f
		}
		returnType = d.typeDebugDescriptor(types.NewStruct(fields, nil), "")
	}

	var paramTypes []llvm.Metadata
	params := t.Params()
	if params != nil && params.Len() > 0 {
		paramTypes = make([]llvm.Metadata, params.Len()+1)
		paramTypes[0] = returnType
		for i := range paramTypes[1:] {
			paramTypes[i+1] = d.DIType(params.At(i).Type())
		}
	} else {
		paramTypes = []llvm.Metadata{returnType}
	}

	// TODO(axw) get position of type definition for File field
	return d.builder.CreateSubroutineType(llvm.DISubroutineType{
		Parameters: paramTypes,
	})
}
