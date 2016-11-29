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
	}

	for _, name := range attrTests {
		testAttribute(t, name)
	}
}
