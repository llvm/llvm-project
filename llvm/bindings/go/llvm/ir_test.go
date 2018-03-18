//===- ir_test.go - Tests for ir ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests bindings for the ir component.
//
//===----------------------------------------------------------------------===//

package llvm

import (
	"strings"
	"testing"
)

func testAttribute(t *testing.T, name string) {
	mod := NewModule("")
	defer mod.Dispose()

	ftyp := FunctionType(VoidType(), nil, false)
	fn := AddFunction(mod, "foo", ftyp)

	kind := AttributeKindID(name)
	attr := mod.Context().CreateEnumAttribute(kind, 0)

	fn.AddFunctionAttr(attr)
	newattr := fn.GetEnumFunctionAttribute(kind)
	if attr != newattr {
		t.Errorf("got attribute mask %d, want %d", newattr, attr)
	}

	text := mod.String()
	if !strings.Contains(text, " "+name+" ") {
		t.Errorf("expected attribute '%s', got:\n%s", name, text)
	}

	fn.RemoveEnumFunctionAttribute(kind)
	newattr = fn.GetEnumFunctionAttribute(kind)
	if !newattr.IsNil() {
		t.Errorf("got attribute mask %d, want 0", newattr)
	}
}

func TestAttributes(t *testing.T) {
	// Tests that our attribute constants haven't drifted from LLVM's.
	attrTests := []string{
		"sanitize_address",
		"alwaysinline",
		"builtin",
		"byval",
		"convergent",
		"inalloca",
		"inlinehint",
		"inreg",
		"jumptable",
		"minsize",
		"naked",
		"nest",
		"noalias",
		"nobuiltin",
		"nocapture",
		"noduplicate",
		"noimplicitfloat",
		"noinline",
		"nonlazybind",
		"nonnull",
		"noredzone",
		"noreturn",
		"nounwind",
		"optnone",
		"optsize",
		"readnone",
		"readonly",
		"returned",
		"returns_twice",
		"signext",
		"safestack",
		"ssp",
		"sspreq",
		"sspstrong",
		"sret",
		"sanitize_thread",
		"sanitize_memory",
		"uwtable",
		"zeroext",
		"cold",
		"nocf_check",
	}

	for _, name := range attrTests {
		testAttribute(t, name)
	}
}

func TestDebugLoc(t *testing.T) {
	mod := NewModule("")
	defer mod.Dispose()

	ctx := mod.Context()

	b := ctx.NewBuilder()
	defer b.Dispose()

	d := NewDIBuilder(mod)
	defer func() {
		d.Destroy()
	}()
	file := d.CreateFile("dummy_file", "dummy_dir")
	voidInfo := d.CreateBasicType(DIBasicType{Name: "void"})
	typeInfo := d.CreateSubroutineType(DISubroutineType{file, []Metadata{voidInfo}})
	scope := d.CreateFunction(file, DIFunction{
		Name:         "foo",
		LinkageName:  "foo",
		Line:         10,
		ScopeLine:    10,
		Type:         typeInfo,
		File:         file,
		IsDefinition: true,
	})

	b.SetCurrentDebugLocation(10, 20, scope, Metadata{})
	loc := b.GetCurrentDebugLocation()
	if loc.Line != 10 {
		t.Errorf("Got line %d, though wanted 10", loc.Line)
	}
	if loc.Col != 20 {
		t.Errorf("Got column %d, though wanted 20", loc.Col)
	}
	if loc.Scope.C != scope.C {
		t.Errorf("Got metadata %v as scope, though wanted %v", loc.Scope.C, scope.C)
	}
}

func TestSubtypes(t *testing.T) {
	cont := NewContext()
	defer cont.Dispose()

	int_pointer := PointerType(cont.Int32Type(), 0)
	int_inner := int_pointer.Subtypes()
	if len(int_inner) != 1 {
		t.Errorf("Got size %d, though wanted 1", len(int_inner))
	}
	if int_inner[0] != cont.Int32Type() {
		t.Errorf("Expected int32 type")
	}

	st_pointer := cont.StructType([]Type{cont.Int32Type(), cont.Int8Type()}, false)
	st_inner := st_pointer.Subtypes()
	if len(st_inner) != 2 {
		t.Errorf("Got size %d, though wanted 2", len(int_inner))
	}
	if st_inner[0] != cont.Int32Type() {
		t.Errorf("Expected first struct field to be int32")
	}
	if st_inner[1] != cont.Int8Type() {
		t.Errorf("Expected second struct field to be int8")
	}
}
