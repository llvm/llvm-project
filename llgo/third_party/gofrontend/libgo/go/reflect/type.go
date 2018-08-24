// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package reflect implements run-time reflection, allowing a program to
// manipulate objects with arbitrary types.  The typical use is to take a value
// with static type interface{} and extract its dynamic type information by
// calling TypeOf, which returns a Type.
//
// A call to ValueOf returns a Value representing the run-time data.
// Zero takes a Type and returns a Value representing a zero value
// for that type.
//
// See "The Laws of Reflection" for an introduction to reflection in Go:
// https://golang.org/doc/articles/laws_of_reflection.html
package reflect

import (
	"runtime"
	"strconv"
	"sync"
	"unsafe"
)

// Type is the representation of a Go type.
//
// Not all methods apply to all kinds of types.  Restrictions,
// if any, are noted in the documentation for each method.
// Use the Kind method to find out the kind of type before
// calling kind-specific methods.  Calling a method
// inappropriate to the kind of type causes a run-time panic.
type Type interface {
	// Methods applicable to all types.

	// Align returns the alignment in bytes of a value of
	// this type when allocated in memory.
	Align() int

	// FieldAlign returns the alignment in bytes of a value of
	// this type when used as a field in a struct.
	FieldAlign() int

	// Method returns the i'th method in the type's method set.
	// It panics if i is not in the range [0, NumMethod()).
	//
	// For a non-interface type T or *T, the returned Method's Type and Func
	// fields describe a function whose first argument is the receiver.
	//
	// For an interface type, the returned Method's Type field gives the
	// method signature, without a receiver, and the Func field is nil.
	Method(int) Method

	// MethodByName returns the method with that name in the type's
	// method set and a boolean indicating if the method was found.
	//
	// For a non-interface type T or *T, the returned Method's Type and Func
	// fields describe a function whose first argument is the receiver.
	//
	// For an interface type, the returned Method's Type field gives the
	// method signature, without a receiver, and the Func field is nil.
	MethodByName(string) (Method, bool)

	// NumMethod returns the number of methods in the type's method set.
	NumMethod() int

	// Name returns the type's name within its package.
	// It returns an empty string for unnamed types.
	Name() string

	// PkgPath returns a named type's package path, that is, the import path
	// that uniquely identifies the package, such as "encoding/base64".
	// If the type was predeclared (string, error) or unnamed (*T, struct{}, []int),
	// the package path will be the empty string.
	PkgPath() string

	// Size returns the number of bytes needed to store
	// a value of the given type; it is analogous to unsafe.Sizeof.
	Size() uintptr

	// String returns a string representation of the type.
	// The string representation may use shortened package names
	// (e.g., base64 instead of "encoding/base64") and is not
	// guaranteed to be unique among types.  To test for equality,
	// compare the Types directly.
	String() string

	// Used internally by gccgo--the string retaining quoting.
	rawString() string

	// Kind returns the specific kind of this type.
	Kind() Kind

	// Implements reports whether the type implements the interface type u.
	Implements(u Type) bool

	// AssignableTo reports whether a value of the type is assignable to type u.
	AssignableTo(u Type) bool

	// ConvertibleTo reports whether a value of the type is convertible to type u.
	ConvertibleTo(u Type) bool

	// Comparable reports whether values of this type are comparable.
	Comparable() bool

	// Methods applicable only to some types, depending on Kind.
	// The methods allowed for each kind are:
	//
	//	Int*, Uint*, Float*, Complex*: Bits
	//	Array: Elem, Len
	//	Chan: ChanDir, Elem
	//	Func: In, NumIn, Out, NumOut, IsVariadic.
	//	Map: Key, Elem
	//	Ptr: Elem
	//	Slice: Elem
	//	Struct: Field, FieldByIndex, FieldByName, FieldByNameFunc, NumField

	// Bits returns the size of the type in bits.
	// It panics if the type's Kind is not one of the
	// sized or unsized Int, Uint, Float, or Complex kinds.
	Bits() int

	// ChanDir returns a channel type's direction.
	// It panics if the type's Kind is not Chan.
	ChanDir() ChanDir

	// IsVariadic reports whether a function type's final input parameter
	// is a "..." parameter.  If so, t.In(t.NumIn() - 1) returns the parameter's
	// implicit actual type []T.
	//
	// For concreteness, if t represents func(x int, y ... float64), then
	//
	//	t.NumIn() == 2
	//	t.In(0) is the reflect.Type for "int"
	//	t.In(1) is the reflect.Type for "[]float64"
	//	t.IsVariadic() == true
	//
	// IsVariadic panics if the type's Kind is not Func.
	IsVariadic() bool

	// Elem returns a type's element type.
	// It panics if the type's Kind is not Array, Chan, Map, Ptr, or Slice.
	Elem() Type

	// Field returns a struct type's i'th field.
	// It panics if the type's Kind is not Struct.
	// It panics if i is not in the range [0, NumField()).
	Field(i int) StructField

	// FieldByIndex returns the nested field corresponding
	// to the index sequence.  It is equivalent to calling Field
	// successively for each index i.
	// It panics if the type's Kind is not Struct.
	FieldByIndex(index []int) StructField

	// FieldByName returns the struct field with the given name
	// and a boolean indicating if the field was found.
	FieldByName(name string) (StructField, bool)

	// FieldByNameFunc returns the first struct field with a name
	// that satisfies the match function and a boolean indicating if
	// the field was found.
	FieldByNameFunc(match func(string) bool) (StructField, bool)

	// In returns the type of a function type's i'th input parameter.
	// It panics if the type's Kind is not Func.
	// It panics if i is not in the range [0, NumIn()).
	In(i int) Type

	// Key returns a map type's key type.
	// It panics if the type's Kind is not Map.
	Key() Type

	// Len returns an array type's length.
	// It panics if the type's Kind is not Array.
	Len() int

	// NumField returns a struct type's field count.
	// It panics if the type's Kind is not Struct.
	NumField() int

	// NumIn returns a function type's input parameter count.
	// It panics if the type's Kind is not Func.
	NumIn() int

	// NumOut returns a function type's output parameter count.
	// It panics if the type's Kind is not Func.
	NumOut() int

	// Out returns the type of a function type's i'th output parameter.
	// It panics if the type's Kind is not Func.
	// It panics if i is not in the range [0, NumOut()).
	Out(i int) Type

	common() *rtype
	uncommon() *uncommonType
}

// BUG(rsc): FieldByName and related functions consider struct field names to be equal
// if the names are equal, even if they are unexported names originating
// in different packages. The practical effect of this is that the result of
// t.FieldByName("x") is not well defined if the struct type t contains
// multiple fields named x (embedded from different packages).
// FieldByName may return one of the fields named x or may report that there are none.
// See golang.org/issue/4876 for more details.

/*
 * These data structures are known to the compiler (../../cmd/internal/gc/reflect.go).
 * A few are known to ../runtime/type.go to convey to debuggers.
 * They are also known to ../runtime/type.go.
 */

// A Kind represents the specific kind of type that a Type represents.
// The zero Kind is not a valid kind.
type Kind uint

const (
	Invalid Kind = iota
	Bool
	Int
	Int8
	Int16
	Int32
	Int64
	Uint
	Uint8
	Uint16
	Uint32
	Uint64
	Uintptr
	Float32
	Float64
	Complex64
	Complex128
	Array
	Chan
	Func
	Interface
	Map
	Ptr
	Slice
	String
	Struct
	UnsafePointer
)

// rtype is the common implementation of most values.
// It is embedded in other, public struct types, but always
// with a unique tag like `reflect:"array"` or `reflect:"ptr"`
// so that code cannot convert from, say, *arrayType to *ptrType.
type rtype struct {
	kind       uint8 // enumeration for C
	align      int8  // alignment of variable with this type
	fieldAlign uint8 // alignment of struct field with this type
	_          uint8 // unused/padding
	size       uintptr
	hash       uint32 // hash of type; avoids computation in hash tables

	hashfn  func(unsafe.Pointer, uintptr) uintptr              // hash function
	equalfn func(unsafe.Pointer, unsafe.Pointer, uintptr) bool // equality function

	gc            unsafe.Pointer // garbage collection data
	string        *string        // string form; unnecessary  but undeniably useful
	*uncommonType                // (relatively) uncommon fields
	ptrToThis     *rtype         // type for pointer to this type, if used in binary or has methods
}

// Method on non-interface type
type method struct {
	name    *string        // name of method
	pkgPath *string        // nil for exported Names; otherwise import path
	mtyp    *rtype         // method type (without receiver)
	typ     *rtype         // .(*FuncType) underneath (with receiver)
	tfn     unsafe.Pointer // fn used for normal method call
}

// uncommonType is present only for types with names or methods
// (if T is a named type, the uncommonTypes for T and *T have methods).
// Using a pointer to this struct reduces the overall size required
// to describe an unnamed type with no methods.
type uncommonType struct {
	name    *string  // name of type
	pkgPath *string  // import path; nil for built-in types like int, string
	methods []method // methods associated with type
}

// ChanDir represents a channel type's direction.
type ChanDir int

const (
	RecvDir ChanDir             = 1 << iota // <-chan
	SendDir                                 // chan<-
	BothDir = RecvDir | SendDir             // chan
)

// arrayType represents a fixed array type.
type arrayType struct {
	rtype `reflect:"array"`
	elem  *rtype // array element type
	slice *rtype // slice type
	len   uintptr
}

// chanType represents a channel type.
type chanType struct {
	rtype `reflect:"chan"`
	elem  *rtype  // channel element type
	dir   uintptr // channel direction (ChanDir)
}

// funcType represents a function type.
type funcType struct {
	rtype     `reflect:"func"`
	dotdotdot bool     // last input parameter is ...
	in        []*rtype // input parameter types
	out       []*rtype // output parameter types
}

// imethod represents a method on an interface type
type imethod struct {
	name    *string // name of method
	pkgPath *string // nil for exported Names; otherwise import path
	typ     *rtype  // .(*FuncType) underneath
}

// interfaceType represents an interface type.
type interfaceType struct {
	rtype   `reflect:"interface"`
	methods []imethod // sorted by hash
}

// mapType represents a map type.
type mapType struct {
	rtype `reflect:"map"`
	key   *rtype // map key type
	elem  *rtype // map element (value) type
}

// ptrType represents a pointer type.
type ptrType struct {
	rtype `reflect:"ptr"`
	elem  *rtype // pointer element (pointed at) type
}

// sliceType represents a slice type.
type sliceType struct {
	rtype `reflect:"slice"`
	elem  *rtype // slice element type
}

// Struct field
type structField struct {
	name    *string // nil for embedded fields
	pkgPath *string // nil for exported Names; otherwise import path
	typ     *rtype  // type of field
	tag     *string // nil if no tag
	offset  uintptr // byte offset of field within struct
}

// structType represents a struct type.
type structType struct {
	rtype  `reflect:"struct"`
	fields []structField // sorted by offset
}

// NOTE: These are copied from ../runtime/mgc0.h.
// They must be kept in sync.
const (
	_GC_END = iota
	_GC_PTR
	_GC_APTR
	_GC_ARRAY_START
	_GC_ARRAY_NEXT
	_GC_CALL
	_GC_CHAN_PTR
	_GC_STRING
	_GC_EFACE
	_GC_IFACE
	_GC_SLICE
	_GC_REGION
	_GC_NUM_INSTR
)

/*
 * The compiler knows the exact layout of all the data structures above.
 * The compiler does not know about the data structures and methods below.
 */

// Method represents a single method.
type Method struct {
	// Name is the method name.
	// PkgPath is the package path that qualifies a lower case (unexported)
	// method name.  It is empty for upper case (exported) method names.
	// The combination of PkgPath and Name uniquely identifies a method
	// in a method set.
	// See https://golang.org/ref/spec#Uniqueness_of_identifiers
	Name    string
	PkgPath string

	Type  Type  // method type
	Func  Value // func with receiver as first argument
	Index int   // index for Type.Method
}

const (
	kindDirectIface = 1 << 5
	kindGCProg      = 1 << 6 // Type.gc points to GC program
	kindNoPointers  = 1 << 7
	kindMask        = (1 << 5) - 1
)

func (k Kind) String() string {
	if int(k) < len(kindNames) {
		return kindNames[k]
	}
	return "kind" + strconv.Itoa(int(k))
}

var kindNames = []string{
	Invalid:       "invalid",
	Bool:          "bool",
	Int:           "int",
	Int8:          "int8",
	Int16:         "int16",
	Int32:         "int32",
	Int64:         "int64",
	Uint:          "uint",
	Uint8:         "uint8",
	Uint16:        "uint16",
	Uint32:        "uint32",
	Uint64:        "uint64",
	Uintptr:       "uintptr",
	Float32:       "float32",
	Float64:       "float64",
	Complex64:     "complex64",
	Complex128:    "complex128",
	Array:         "array",
	Chan:          "chan",
	Func:          "func",
	Interface:     "interface",
	Map:           "map",
	Ptr:           "ptr",
	Slice:         "slice",
	String:        "string",
	Struct:        "struct",
	UnsafePointer: "unsafe.Pointer",
}

func (t *uncommonType) uncommon() *uncommonType {
	return t
}

func (t *uncommonType) PkgPath() string {
	if t == nil || t.pkgPath == nil {
		return ""
	}
	return *t.pkgPath
}

func (t *uncommonType) Name() string {
	if t == nil || t.name == nil {
		return ""
	}
	return *t.name
}

func (t *rtype) rawString() string { return *t.string }

func (t *rtype) String() string {
	// For gccgo, strip out quoted strings.
	s := *t.string
	var q bool
	r := make([]byte, len(s))
	j := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\t' {
			q = !q
		} else if !q {
			r[j] = s[i]
			j++
		}
	}
	return string(r[:j])
}

func (t *rtype) Size() uintptr { return t.size }

func (t *rtype) Bits() int {
	if t == nil {
		panic("reflect: Bits of nil Type")
	}
	k := t.Kind()
	if k < Int || k > Complex128 {
		panic("reflect: Bits of non-arithmetic Type " + t.String())
	}
	return int(t.size) * 8
}

func (t *rtype) Align() int { return int(t.align) }

func (t *rtype) FieldAlign() int { return int(t.fieldAlign) }

func (t *rtype) Kind() Kind { return Kind(t.kind & kindMask) }

func (t *rtype) pointers() bool { return t.kind&kindNoPointers == 0 }

func (t *rtype) common() *rtype { return t }

func (t *uncommonType) Method(i int) (m Method) {
	if t == nil || i < 0 || i >= len(t.methods) {
		panic("reflect: Method index out of range")
	}
	p := &t.methods[i]
	if p.name != nil {
		m.Name = *p.name
	}
	fl := flag(Func)
	if p.pkgPath != nil {
		m.PkgPath = *p.pkgPath
		fl |= flagStickyRO
	}
	mt := p.typ
	m.Type = toType(mt)
	x := new(unsafe.Pointer)
	*x = unsafe.Pointer(&p.tfn)
	m.Func = Value{mt, unsafe.Pointer(x), fl | flagIndir | flagMethodFn}
	m.Index = i
	return
}

func (t *uncommonType) NumMethod() int {
	if t == nil {
		return 0
	}
	return len(t.methods)
}

func (t *uncommonType) MethodByName(name string) (m Method, ok bool) {
	if t == nil {
		return
	}
	var p *method
	for i := range t.methods {
		p = &t.methods[i]
		if p.name != nil && *p.name == name {
			return t.Method(i), true
		}
	}
	return
}

// TODO(rsc): gc supplies these, but they are not
// as efficient as they could be: they have commonType
// as the receiver instead of *rtype.
func (t *rtype) NumMethod() int {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t))
		return tt.NumMethod()
	}
	return t.uncommonType.NumMethod()
}

func (t *rtype) Method(i int) (m Method) {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t))
		return tt.Method(i)
	}
	return t.uncommonType.Method(i)
}

func (t *rtype) MethodByName(name string) (m Method, ok bool) {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t))
		return tt.MethodByName(name)
	}
	return t.uncommonType.MethodByName(name)
}

func (t *rtype) PkgPath() string {
	return t.uncommonType.PkgPath()
}

func (t *rtype) Name() string {
	return t.uncommonType.Name()
}

func (t *rtype) ChanDir() ChanDir {
	if t.Kind() != Chan {
		panic("reflect: ChanDir of non-chan type")
	}
	tt := (*chanType)(unsafe.Pointer(t))
	return ChanDir(tt.dir)
}

func (t *rtype) IsVariadic() bool {
	if t.Kind() != Func {
		panic("reflect: IsVariadic of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return tt.dotdotdot
}

func (t *rtype) Elem() Type {
	switch t.Kind() {
	case Array:
		tt := (*arrayType)(unsafe.Pointer(t))
		return toType(tt.elem)
	case Chan:
		tt := (*chanType)(unsafe.Pointer(t))
		return toType(tt.elem)
	case Map:
		tt := (*mapType)(unsafe.Pointer(t))
		return toType(tt.elem)
	case Ptr:
		tt := (*ptrType)(unsafe.Pointer(t))
		return toType(tt.elem)
	case Slice:
		tt := (*sliceType)(unsafe.Pointer(t))
		return toType(tt.elem)
	}
	panic("reflect: Elem of invalid type")
}

func (t *rtype) Field(i int) StructField {
	if t.Kind() != Struct {
		panic("reflect: Field of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.Field(i)
}

func (t *rtype) FieldByIndex(index []int) StructField {
	if t.Kind() != Struct {
		panic("reflect: FieldByIndex of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.FieldByIndex(index)
}

func (t *rtype) FieldByName(name string) (StructField, bool) {
	if t.Kind() != Struct {
		panic("reflect: FieldByName of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.FieldByName(name)
}

func (t *rtype) FieldByNameFunc(match func(string) bool) (StructField, bool) {
	if t.Kind() != Struct {
		panic("reflect: FieldByNameFunc of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.FieldByNameFunc(match)
}

func (t *rtype) In(i int) Type {
	if t.Kind() != Func {
		panic("reflect: In of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return toType(tt.in[i])
}

func (t *rtype) Key() Type {
	if t.Kind() != Map {
		panic("reflect: Key of non-map type")
	}
	tt := (*mapType)(unsafe.Pointer(t))
	return toType(tt.key)
}

func (t *rtype) Len() int {
	if t.Kind() != Array {
		panic("reflect: Len of non-array type")
	}
	tt := (*arrayType)(unsafe.Pointer(t))
	return int(tt.len)
}

func (t *rtype) NumField() int {
	if t.Kind() != Struct {
		panic("reflect: NumField of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return len(tt.fields)
}

func (t *rtype) NumIn() int {
	if t.Kind() != Func {
		panic("reflect: NumIn of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return len(tt.in)
}

func (t *rtype) NumOut() int {
	if t.Kind() != Func {
		panic("reflect: NumOut of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return len(tt.out)
}

func (t *rtype) Out(i int) Type {
	if t.Kind() != Func {
		panic("reflect: Out of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return toType(tt.out[i])
}

func (d ChanDir) String() string {
	switch d {
	case SendDir:
		return "chan<-"
	case RecvDir:
		return "<-chan"
	case BothDir:
		return "chan"
	}
	return "ChanDir" + strconv.Itoa(int(d))
}

// Method returns the i'th method in the type's method set.
func (t *interfaceType) Method(i int) (m Method) {
	if i < 0 || i >= len(t.methods) {
		return
	}
	p := &t.methods[i]
	m.Name = *p.name
	if p.pkgPath != nil {
		m.PkgPath = *p.pkgPath
	}
	m.Type = toType(p.typ)
	m.Index = i
	return
}

// NumMethod returns the number of interface methods in the type's method set.
func (t *interfaceType) NumMethod() int { return len(t.methods) }

// MethodByName method with the given name in the type's method set.
func (t *interfaceType) MethodByName(name string) (m Method, ok bool) {
	if t == nil {
		return
	}
	var p *imethod
	for i := range t.methods {
		p = &t.methods[i]
		if *p.name == name {
			return t.Method(i), true
		}
	}
	return
}

// A StructField describes a single field in a struct.
type StructField struct {
	// Name is the field name.
	// PkgPath is the package path that qualifies a lower case (unexported)
	// field name.  It is empty for upper case (exported) field names.
	// See https://golang.org/ref/spec#Uniqueness_of_identifiers
	Name    string
	PkgPath string

	Type      Type      // field type
	Tag       StructTag // field tag string
	Offset    uintptr   // offset within struct, in bytes
	Index     []int     // index sequence for Type.FieldByIndex
	Anonymous bool      // is an embedded field
}

// A StructTag is the tag string in a struct field.
//
// By convention, tag strings are a concatenation of
// optionally space-separated key:"value" pairs.
// Each key is a non-empty string consisting of non-control
// characters other than space (U+0020 ' '), quote (U+0022 '"'),
// and colon (U+003A ':').  Each value is quoted using U+0022 '"'
// characters and Go string literal syntax.
type StructTag string

// Get returns the value associated with key in the tag string.
// If there is no such key in the tag, Get returns the empty string.
// If the tag does not have the conventional format, the value
// returned by Get is unspecified.
func (tag StructTag) Get(key string) string {
	// When modifying this code, also update the validateStructTag code
	// in golang.org/x/tools/cmd/vet/structtag.go.

	for tag != "" {
		// Skip leading space.
		i := 0
		for i < len(tag) && tag[i] == ' ' {
			i++
		}
		tag = tag[i:]
		if tag == "" {
			break
		}

		// Scan to colon. A space, a quote or a control character is a syntax error.
		// Strictly speaking, control chars include the range [0x7f, 0x9f], not just
		// [0x00, 0x1f], but in practice, we ignore the multi-byte control characters
		// as it is simpler to inspect the tag's bytes than the tag's runes.
		i = 0
		for i < len(tag) && tag[i] > ' ' && tag[i] != ':' && tag[i] != '"' && tag[i] != 0x7f {
			i++
		}
		if i == 0 || i+1 >= len(tag) || tag[i] != ':' || tag[i+1] != '"' {
			break
		}
		name := string(tag[:i])
		tag = tag[i+1:]

		// Scan quoted string to find value.
		i = 1
		for i < len(tag) && tag[i] != '"' {
			if tag[i] == '\\' {
				i++
			}
			i++
		}
		if i >= len(tag) {
			break
		}
		qvalue := string(tag[:i+1])
		tag = tag[i+1:]

		if key == name {
			value, err := strconv.Unquote(qvalue)
			if err != nil {
				break
			}
			return value
		}
	}
	return ""
}

// Field returns the i'th struct field.
func (t *structType) Field(i int) (f StructField) {
	if i < 0 || i >= len(t.fields) {
		return
	}
	p := &t.fields[i]
	f.Type = toType(p.typ)
	if p.name != nil {
		f.Name = *p.name
	} else {
		t := f.Type
		if t.Kind() == Ptr {
			t = t.Elem()
		}
		f.Name = t.Name()
		f.Anonymous = true
	}
	if p.pkgPath != nil {
		f.PkgPath = *p.pkgPath
	}
	if p.tag != nil {
		f.Tag = StructTag(*p.tag)
	}
	f.Offset = p.offset

	// NOTE(rsc): This is the only allocation in the interface
	// presented by a reflect.Type.  It would be nice to avoid,
	// at least in the common cases, but we need to make sure
	// that misbehaving clients of reflect cannot affect other
	// uses of reflect.  One possibility is CL 5371098, but we
	// postponed that ugliness until there is a demonstrated
	// need for the performance.  This is issue 2320.
	f.Index = []int{i}
	return
}

// TODO(gri): Should there be an error/bool indicator if the index
//            is wrong for FieldByIndex?

// FieldByIndex returns the nested field corresponding to index.
func (t *structType) FieldByIndex(index []int) (f StructField) {
	f.Type = toType(&t.rtype)
	for i, x := range index {
		if i > 0 {
			ft := f.Type
			if ft.Kind() == Ptr && ft.Elem().Kind() == Struct {
				ft = ft.Elem()
			}
			f.Type = ft
		}
		f = f.Type.Field(x)
	}
	return
}

// A fieldScan represents an item on the fieldByNameFunc scan work list.
type fieldScan struct {
	typ   *structType
	index []int
}

// FieldByNameFunc returns the struct field with a name that satisfies the
// match function and a boolean to indicate if the field was found.
func (t *structType) FieldByNameFunc(match func(string) bool) (result StructField, ok bool) {
	// This uses the same condition that the Go language does: there must be a unique instance
	// of the match at a given depth level. If there are multiple instances of a match at the
	// same depth, they annihilate each other and inhibit any possible match at a lower level.
	// The algorithm is breadth first search, one depth level at a time.

	// The current and next slices are work queues:
	// current lists the fields to visit on this depth level,
	// and next lists the fields on the next lower level.
	current := []fieldScan{}
	next := []fieldScan{{typ: t}}

	// nextCount records the number of times an embedded type has been
	// encountered and considered for queueing in the 'next' slice.
	// We only queue the first one, but we increment the count on each.
	// If a struct type T can be reached more than once at a given depth level,
	// then it annihilates itself and need not be considered at all when we
	// process that next depth level.
	var nextCount map[*structType]int

	// visited records the structs that have been considered already.
	// Embedded pointer fields can create cycles in the graph of
	// reachable embedded types; visited avoids following those cycles.
	// It also avoids duplicated effort: if we didn't find the field in an
	// embedded type T at level 2, we won't find it in one at level 4 either.
	visited := map[*structType]bool{}

	for len(next) > 0 {
		current, next = next, current[:0]
		count := nextCount
		nextCount = nil

		// Process all the fields at this depth, now listed in 'current'.
		// The loop queues embedded fields found in 'next', for processing during the next
		// iteration. The multiplicity of the 'current' field counts is recorded
		// in 'count'; the multiplicity of the 'next' field counts is recorded in 'nextCount'.
		for _, scan := range current {
			t := scan.typ
			if visited[t] {
				// We've looked through this type before, at a higher level.
				// That higher level would shadow the lower level we're now at,
				// so this one can't be useful to us. Ignore it.
				continue
			}
			visited[t] = true
			for i := range t.fields {
				f := &t.fields[i]
				// Find name and type for field f.
				var fname string
				var ntyp *rtype
				if f.name != nil {
					fname = *f.name
				} else {
					// Anonymous field of type T or *T.
					// Name taken from type.
					ntyp = f.typ
					if ntyp.Kind() == Ptr {
						ntyp = ntyp.Elem().common()
					}
					fname = ntyp.Name()
				}

				// Does it match?
				if match(fname) {
					// Potential match
					if count[t] > 1 || ok {
						// Name appeared multiple times at this level: annihilate.
						return StructField{}, false
					}
					result = t.Field(i)
					result.Index = nil
					result.Index = append(result.Index, scan.index...)
					result.Index = append(result.Index, i)
					ok = true
					continue
				}

				// Queue embedded struct fields for processing with next level,
				// but only if we haven't seen a match yet at this level and only
				// if the embedded types haven't already been queued.
				if ok || ntyp == nil || ntyp.Kind() != Struct {
					continue
				}
				ntyp = toType(ntyp).common()
				styp := (*structType)(unsafe.Pointer(ntyp))
				if nextCount[styp] > 0 {
					nextCount[styp] = 2 // exact multiple doesn't matter
					continue
				}
				if nextCount == nil {
					nextCount = map[*structType]int{}
				}
				nextCount[styp] = 1
				if count[t] > 1 {
					nextCount[styp] = 2 // exact multiple doesn't matter
				}
				var index []int
				index = append(index, scan.index...)
				index = append(index, i)
				next = append(next, fieldScan{styp, index})
			}
		}
		if ok {
			break
		}
	}
	return
}

// FieldByName returns the struct field with the given name
// and a boolean to indicate if the field was found.
func (t *structType) FieldByName(name string) (f StructField, present bool) {
	// Quick check for top-level name, or struct without anonymous fields.
	hasAnon := false
	if name != "" {
		for i := range t.fields {
			tf := &t.fields[i]
			if tf.name == nil {
				hasAnon = true
				continue
			}
			if *tf.name == name {
				return t.Field(i), true
			}
		}
	}
	if !hasAnon {
		return
	}
	return t.FieldByNameFunc(func(s string) bool { return s == name })
}

// TypeOf returns the reflection Type that represents the dynamic type of i.
// If i is a nil interface value, TypeOf returns nil.
func TypeOf(i interface{}) Type {
	eface := *(*emptyInterface)(unsafe.Pointer(&i))
	return toType(eface.typ)
}

// ptrMap is the cache for PtrTo.
var ptrMap struct {
	sync.RWMutex
	m map[*rtype]*ptrType
}

// garbage collection bytecode program for pointer to memory without pointers.
// See ../../cmd/gc/reflect.c:/^dgcsym1 and :/^dgcsym.
type ptrDataGC struct {
	width uintptr // sizeof(ptr)
	op    uintptr // _GC_APTR
	off   uintptr // 0
	end   uintptr // _GC_END
}

var ptrDataGCProg = ptrDataGC{
	width: unsafe.Sizeof((*byte)(nil)),
	op:    _GC_APTR,
	off:   0,
	end:   _GC_END,
}

// garbage collection bytecode program for pointer to memory with pointers.
// See ../../cmd/gc/reflect.c:/^dgcsym1 and :/^dgcsym.
type ptrGC struct {
	width  uintptr        // sizeof(ptr)
	op     uintptr        // _GC_PTR
	off    uintptr        // 0
	elemgc unsafe.Pointer // element gc type
	end    uintptr        // _GC_END
}

// PtrTo returns the pointer type with element t.
// For example, if t represents type Foo, PtrTo(t) represents *Foo.
func PtrTo(t Type) Type {
	return t.(*rtype).ptrTo()
}

func (t *rtype) ptrTo() *rtype {
	if p := t.ptrToThis; p != nil {
		return p
	}

	// Otherwise, synthesize one.
	// This only happens for pointers with no methods.
	// We keep the mapping in a map on the side, because
	// this operation is rare and a separate map lets us keep
	// the type structures in read-only memory.
	ptrMap.RLock()
	if m := ptrMap.m; m != nil {
		if p := m[t]; p != nil {
			ptrMap.RUnlock()
			return &p.rtype
		}
	}
	ptrMap.RUnlock()
	ptrMap.Lock()
	if ptrMap.m == nil {
		ptrMap.m = make(map[*rtype]*ptrType)
	}
	p := ptrMap.m[t]
	if p != nil {
		// some other goroutine won the race and created it
		ptrMap.Unlock()
		return &p.rtype
	}

	s := "*" + *t.string

	canonicalTypeLock.RLock()
	r, ok := canonicalType[s]
	canonicalTypeLock.RUnlock()
	if ok {
		ptrMap.m[t] = (*ptrType)(unsafe.Pointer(r.(*rtype)))
		ptrMap.Unlock()
		return r.(*rtype)
	}

	// Create a new ptrType starting with the description
	// of an *unsafe.Pointer.
	p = new(ptrType)
	var iptr interface{} = (*unsafe.Pointer)(nil)
	prototype := *(**ptrType)(unsafe.Pointer(&iptr))
	*p = *prototype

	p.string = &s

	// For the type structures linked into the binary, the
	// compiler provides a good hash of the string.
	// Create a good hash for the new string by using
	// the FNV-1 hash's mixing function to combine the
	// old hash and the new "*".
	// p.hash = fnv1(t.hash, '*')
	// This is the gccgo version.
	p.hash = (t.hash << 4) + 9

	p.uncommonType = nil
	p.ptrToThis = nil
	p.elem = t

	if t.kind&kindNoPointers != 0 {
		p.gc = unsafe.Pointer(&ptrDataGCProg)
	} else {
		p.gc = unsafe.Pointer(&ptrGC{
			width:  p.size,
			op:     _GC_PTR,
			off:    0,
			elemgc: t.gc,
			end:    _GC_END,
		})
	}

	q := canonicalize(&p.rtype)
	p = (*ptrType)(unsafe.Pointer(q.(*rtype)))

	ptrMap.m[t] = p
	ptrMap.Unlock()
	return &p.rtype
}

// fnv1 incorporates the list of bytes into the hash x using the FNV-1 hash function.
func fnv1(x uint32, list ...byte) uint32 {
	for _, b := range list {
		x = x*16777619 ^ uint32(b)
	}
	return x
}

func (t *rtype) Implements(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.Implements")
	}
	if u.Kind() != Interface {
		panic("reflect: non-interface type passed to Type.Implements")
	}
	return implements(u.(*rtype), t)
}

func (t *rtype) AssignableTo(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.AssignableTo")
	}
	uu := u.(*rtype)
	return directlyAssignable(uu, t) || implements(uu, t)
}

func (t *rtype) ConvertibleTo(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.ConvertibleTo")
	}
	uu := u.(*rtype)
	return convertOp(uu, t) != nil
}

func (t *rtype) Comparable() bool {
	switch t.Kind() {
	case Bool, Int, Int8, Int16, Int32, Int64,
		Uint, Uint8, Uint16, Uint32, Uint64, Uintptr,
		Float32, Float64, Complex64, Complex128,
		Chan, Interface, Ptr, String, UnsafePointer:
		return true

	case Func, Map, Slice:
		return false

	case Array:
		return (*arrayType)(unsafe.Pointer(t)).elem.Comparable()

	case Struct:
		tt := (*structType)(unsafe.Pointer(t))
		for i := range tt.fields {
			if !tt.fields[i].typ.Comparable() {
				return false
			}
		}
		return true

	default:
		panic("reflect: impossible")
	}
}

// implements reports whether the type V implements the interface type T.
func implements(T, V *rtype) bool {
	if T.Kind() != Interface {
		return false
	}
	t := (*interfaceType)(unsafe.Pointer(T))
	if len(t.methods) == 0 {
		return true
	}

	// The same algorithm applies in both cases, but the
	// method tables for an interface type and a concrete type
	// are different, so the code is duplicated.
	// In both cases the algorithm is a linear scan over the two
	// lists - T's methods and V's methods - simultaneously.
	// Since method tables are stored in a unique sorted order
	// (alphabetical, with no duplicate method names), the scan
	// through V's methods must hit a match for each of T's
	// methods along the way, or else V does not implement T.
	// This lets us run the scan in overall linear time instead of
	// the quadratic time  a naive search would require.
	// See also ../runtime/iface.go.
	if V.Kind() == Interface {
		v := (*interfaceType)(unsafe.Pointer(V))
		i := 0
		for j := 0; j < len(v.methods); j++ {
			tm := &t.methods[i]
			vm := &v.methods[j]
			if *vm.name == *tm.name && (vm.pkgPath == tm.pkgPath || (vm.pkgPath != nil && tm.pkgPath != nil && *vm.pkgPath == *tm.pkgPath)) && toType(vm.typ).common() == toType(tm.typ).common() {
				if i++; i >= len(t.methods) {
					return true
				}
			}
		}
		return false
	}

	v := V.uncommon()
	if v == nil {
		return false
	}
	i := 0
	for j := 0; j < len(v.methods); j++ {
		tm := &t.methods[i]
		vm := &v.methods[j]
		if *vm.name == *tm.name && (vm.pkgPath == tm.pkgPath || (vm.pkgPath != nil && tm.pkgPath != nil && *vm.pkgPath == *tm.pkgPath)) && toType(vm.mtyp).common() == toType(tm.typ).common() {
			if i++; i >= len(t.methods) {
				return true
			}
		}
	}
	return false
}

// directlyAssignable reports whether a value x of type V can be directly
// assigned (using memmove) to a value of type T.
// https://golang.org/doc/go_spec.html#Assignability
// Ignoring the interface rules (implemented elsewhere)
// and the ideal constant rules (no ideal constants at run time).
func directlyAssignable(T, V *rtype) bool {
	// x's type V is identical to T?
	if T == V {
		return true
	}

	// Otherwise at least one of T and V must be unnamed
	// and they must have the same kind.
	if T.Name() != "" && V.Name() != "" || T.Kind() != V.Kind() {
		return false
	}

	// x's type T and V must  have identical underlying types.
	return haveIdenticalUnderlyingType(T, V)
}

func haveIdenticalUnderlyingType(T, V *rtype) bool {
	if T == V {
		return true
	}

	kind := T.Kind()
	if kind != V.Kind() {
		return false
	}

	// Non-composite types of equal kind have same underlying type
	// (the predefined instance of the type).
	if Bool <= kind && kind <= Complex128 || kind == String || kind == UnsafePointer {
		return true
	}

	// Composite types.
	switch kind {
	case Array:
		return T.Elem() == V.Elem() && T.Len() == V.Len()

	case Chan:
		// Special case:
		// x is a bidirectional channel value, T is a channel type,
		// and x's type V and T have identical element types.
		if V.ChanDir() == BothDir && T.Elem() == V.Elem() {
			return true
		}

		// Otherwise continue test for identical underlying type.
		return V.ChanDir() == T.ChanDir() && T.Elem() == V.Elem()

	case Func:
		t := (*funcType)(unsafe.Pointer(T))
		v := (*funcType)(unsafe.Pointer(V))
		if t.dotdotdot != v.dotdotdot || len(t.in) != len(v.in) || len(t.out) != len(v.out) {
			return false
		}
		for i, typ := range t.in {
			if typ != v.in[i] {
				return false
			}
		}
		for i, typ := range t.out {
			if typ != v.out[i] {
				return false
			}
		}
		return true

	case Interface:
		t := (*interfaceType)(unsafe.Pointer(T))
		v := (*interfaceType)(unsafe.Pointer(V))
		if len(t.methods) == 0 && len(v.methods) == 0 {
			return true
		}
		// Might have the same methods but still
		// need a run time conversion.
		return false

	case Map:
		return T.Key() == V.Key() && T.Elem() == V.Elem()

	case Ptr, Slice:
		return T.Elem() == V.Elem()

	case Struct:
		t := (*structType)(unsafe.Pointer(T))
		v := (*structType)(unsafe.Pointer(V))
		if len(t.fields) != len(v.fields) {
			return false
		}
		for i := range t.fields {
			tf := &t.fields[i]
			vf := &v.fields[i]
			if tf.name != vf.name && (tf.name == nil || vf.name == nil || *tf.name != *vf.name) {
				return false
			}
			if tf.pkgPath != vf.pkgPath && (tf.pkgPath == nil || vf.pkgPath == nil || *tf.pkgPath != *vf.pkgPath) {
				return false
			}
			if tf.typ != vf.typ {
				return false
			}
			if tf.tag != vf.tag && (tf.tag == nil || vf.tag == nil || *tf.tag != *vf.tag) {
				return false
			}
			if tf.offset != vf.offset {
				return false
			}
		}
		return true
	}

	return false
}

// The lookupCache caches ChanOf, MapOf, and SliceOf lookups.
var lookupCache struct {
	sync.RWMutex
	m map[cacheKey]*rtype
}

// A cacheKey is the key for use in the lookupCache.
// Four values describe any of the types we are looking for:
// type kind, one or two subtypes, and an extra integer.
type cacheKey struct {
	kind  Kind
	t1    *rtype
	t2    *rtype
	extra uintptr
}

// cacheGet looks for a type under the key k in the lookupCache.
// If it finds one, it returns that type.
// If not, it returns nil with the cache locked.
// The caller is expected to use cachePut to unlock the cache.
func cacheGet(k cacheKey) Type {
	lookupCache.RLock()
	t := lookupCache.m[k]
	lookupCache.RUnlock()
	if t != nil {
		return t
	}

	lookupCache.Lock()
	t = lookupCache.m[k]
	if t != nil {
		lookupCache.Unlock()
		return t
	}

	if lookupCache.m == nil {
		lookupCache.m = make(map[cacheKey]*rtype)
	}

	return nil
}

// cachePut stores the given type in the cache, unlocks the cache,
// and returns the type. It is expected that the cache is locked
// because cacheGet returned nil.
func cachePut(k cacheKey, t *rtype) Type {
	t = toType(t).common()
	lookupCache.m[k] = t
	lookupCache.Unlock()
	return t
}

// garbage collection bytecode program for chan.
// See ../../cmd/gc/reflect.c:/^dgcsym1 and :/^dgcsym.
type chanGC struct {
	width uintptr // sizeof(map)
	op    uintptr // _GC_CHAN_PTR
	off   uintptr // 0
	typ   *rtype  // map type
	end   uintptr // _GC_END
}

// The funcLookupCache caches FuncOf lookups.
// FuncOf does not share the common lookupCache since cacheKey is not
// sufficient to represent functions unambiguously.
var funcLookupCache struct {
	sync.RWMutex
	m map[uint32][]*rtype // keyed by hash calculated in FuncOf
}

// ChanOf returns the channel type with the given direction and element type.
// For example, if t represents int, ChanOf(RecvDir, t) represents <-chan int.
//
// The gc runtime imposes a limit of 64 kB on channel element types.
// If t's size is equal to or exceeds this limit, ChanOf panics.
func ChanOf(dir ChanDir, t Type) Type {
	typ := t.(*rtype)

	// Look in cache.
	ckey := cacheKey{Chan, typ, nil, uintptr(dir)}
	if ch := cacheGet(ckey); ch != nil {
		return ch
	}

	// This restriction is imposed by the gc compiler and the runtime.
	if typ.size >= 1<<16 {
		lookupCache.Unlock()
		panic("reflect.ChanOf: element size too large")
	}

	// Look in known types.
	// TODO: Precedence when constructing string.
	var s string
	switch dir {
	default:
		lookupCache.Unlock()
		panic("reflect.ChanOf: invalid dir")
	case SendDir:
		s = "chan<- " + *typ.string
	case RecvDir:
		s = "<-chan " + *typ.string
	case BothDir:
		s = "chan " + *typ.string
	}

	// Make a channel type.
	var ichan interface{} = (chan unsafe.Pointer)(nil)
	prototype := *(**chanType)(unsafe.Pointer(&ichan))
	ch := new(chanType)
	*ch = *prototype
	ch.dir = uintptr(dir)
	ch.string = &s

	// gccgo uses a different hash.
	// ch.hash = fnv1(typ.hash, 'c', byte(dir))
	ch.hash = 0
	if dir&SendDir != 0 {
		ch.hash += 1
	}
	if dir&RecvDir != 0 {
		ch.hash += 2
	}
	ch.hash += typ.hash << 2
	ch.hash <<= 3
	ch.hash += 15

	ch.elem = typ
	ch.uncommonType = nil
	ch.ptrToThis = nil

	ch.gc = unsafe.Pointer(&chanGC{
		width: ch.size,
		op:    _GC_CHAN_PTR,
		off:   0,
		typ:   &ch.rtype,
		end:   _GC_END,
	})

	// INCORRECT. Uncomment to check that TestChanOfGC fails when ch.gc is wrong.
	// ch.gc = unsafe.Pointer(&badGC{width: ch.size, end: _GC_END})

	return cachePut(ckey, &ch.rtype)
}

func ismapkey(*rtype) bool // implemented in runtime

// MapOf returns the map type with the given key and element types.
// For example, if k represents int and e represents string,
// MapOf(k, e) represents map[int]string.
//
// If the key type is not a valid map key type (that is, if it does
// not implement Go's == operator), MapOf panics.
func MapOf(key, elem Type) Type {
	ktyp := key.(*rtype)
	etyp := elem.(*rtype)

	if !ismapkey(ktyp) {
		panic("reflect.MapOf: invalid key type " + ktyp.String())
	}

	// Look in cache.
	ckey := cacheKey{Map, ktyp, etyp, 0}
	if mt := cacheGet(ckey); mt != nil {
		return mt
	}

	// Look in known types.
	s := "map[" + *ktyp.string + "]" + *etyp.string

	// Make a map type.
	var imap interface{} = (map[unsafe.Pointer]unsafe.Pointer)(nil)
	mt := new(mapType)
	*mt = **(**mapType)(unsafe.Pointer(&imap))
	mt.string = &s

	// gccgo uses a different hash
	// mt.hash = fnv1(etyp.hash, 'm', byte(ktyp.hash>>24), byte(ktyp.hash>>16), byte(ktyp.hash>>8), byte(ktyp.hash))
	mt.hash = ktyp.hash + etyp.hash + 2 + 14

	mt.key = ktyp
	mt.elem = etyp
	mt.uncommonType = nil
	mt.ptrToThis = nil
	// mt.gc = unsafe.Pointer(&ptrGC{
	// 	width:  unsafe.Sizeof(uintptr(0)),
	// 	op:     _GC_PTR,
	// 	off:    0,
	// 	elemgc: nil,
	// 	end:    _GC_END,
	// })

	// TODO(cmang): Generate GC data for Map elements.
	mt.gc = unsafe.Pointer(&ptrDataGCProg)

	// INCORRECT. Uncomment to check that TestMapOfGC and TestMapOfGCValues
	// fail when mt.gc is wrong.
	//mt.gc = unsafe.Pointer(&badGC{width: mt.size, end: _GC_END})

	return cachePut(ckey, &mt.rtype)
}

// FuncOf returns the function type with the given argument and result types.
// For example if k represents int and e represents string,
// FuncOf([]Type{k}, []Type{e}, false) represents func(int) string.
//
// The variadic argument controls whether the function is variadic. FuncOf
// panics if the in[len(in)-1] does not represent a slice and variadic is
// true.
func FuncOf(in, out []Type, variadic bool) Type {
	if variadic && (len(in) == 0 || in[len(in)-1].Kind() != Slice) {
		panic("reflect.FuncOf: last arg of variadic func must be slice")
	}

	// Make a func type.
	var ifunc interface{} = (func())(nil)
	prototype := *(**funcType)(unsafe.Pointer(&ifunc))
	ft := new(funcType)
	*ft = *prototype

	// Build a hash and minimally populate ft.
	var hash uint32 = 8
	var fin, fout []*rtype
	shift := uint(1)
	for _, in := range in {
		t := in.(*rtype)
		fin = append(fin, t)
		hash += t.hash << shift
		shift++
	}
	shift = 2
	for _, out := range out {
		t := out.(*rtype)
		fout = append(fout, t)
		hash += t.hash << shift
		shift++
	}
	if variadic {
		hash++
	}
	hash <<= 4
	ft.hash = hash
	ft.in = fin
	ft.out = fout
	ft.dotdotdot = variadic

	// Look in cache.
	funcLookupCache.RLock()
	for _, t := range funcLookupCache.m[hash] {
		if haveIdenticalUnderlyingType(&ft.rtype, t) {
			funcLookupCache.RUnlock()
			return t
		}
	}
	funcLookupCache.RUnlock()

	// Not in cache, lock and retry.
	funcLookupCache.Lock()
	defer funcLookupCache.Unlock()
	if funcLookupCache.m == nil {
		funcLookupCache.m = make(map[uint32][]*rtype)
	}
	for _, t := range funcLookupCache.m[hash] {
		if haveIdenticalUnderlyingType(&ft.rtype, t) {
			return t
		}
	}

	str := funcStr(ft)

	// Populate the remaining fields of ft and store in cache.
	ft.string = &str
	ft.uncommonType = nil
	ft.ptrToThis = nil

	// TODO(cmang): Generate GC data for funcs.
	ft.gc = unsafe.Pointer(&ptrDataGCProg)

	funcLookupCache.m[hash] = append(funcLookupCache.m[hash], &ft.rtype)

	return toType(&ft.rtype)
}

// funcStr builds a string representation of a funcType.
func funcStr(ft *funcType) string {
	repr := make([]byte, 0, 64)
	repr = append(repr, "func("...)
	for i, t := range ft.in {
		if i > 0 {
			repr = append(repr, ", "...)
		}
		if ft.dotdotdot && i == len(ft.in)-1 {
			repr = append(repr, "..."...)
			repr = append(repr, *(*sliceType)(unsafe.Pointer(t)).elem.string...)
		} else {
			repr = append(repr, *t.string...)
		}
	}
	repr = append(repr, ')')
	if l := len(ft.out); l == 1 {
		repr = append(repr, ' ')
	} else if l > 1 {
		repr = append(repr, " ("...)
	}
	for i, t := range ft.out {
		if i > 0 {
			repr = append(repr, ", "...)
		}
		repr = append(repr, *t.string...)
	}
	if len(ft.out) > 1 {
		repr = append(repr, ')')
	}
	return string(repr)
}

// isReflexive reports whether the == operation on the type is reflexive.
// That is, x == x for all values x of type t.
func isReflexive(t *rtype) bool {
	switch t.Kind() {
	case Bool, Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Uintptr, Chan, Ptr, String, UnsafePointer:
		return true
	case Float32, Float64, Complex64, Complex128, Interface:
		return false
	case Array:
		tt := (*arrayType)(unsafe.Pointer(t))
		return isReflexive(tt.elem)
	case Struct:
		tt := (*structType)(unsafe.Pointer(t))
		for _, f := range tt.fields {
			if !isReflexive(f.typ) {
				return false
			}
		}
		return true
	default:
		// Func, Map, Slice, Invalid
		panic("isReflexive called on non-key type " + t.String())
	}
}

// Make sure these routines stay in sync with ../../runtime/hashmap.go!
// These types exist only for GC, so we only fill out GC relevant info.
// Currently, that's just size and the GC program.  We also fill in string
// for possible debugging use.
const (
	bucketSize uintptr = 8
	maxKeySize uintptr = 128
	maxValSize uintptr = 128
)

func bucketOf(ktyp, etyp *rtype) *rtype {
	// See comment on hmap.overflow in ../runtime/hashmap.go.
	var kind uint8
	if ktyp.kind&kindNoPointers != 0 && etyp.kind&kindNoPointers != 0 &&
		ktyp.size <= maxKeySize && etyp.size <= maxValSize {
		kind = kindNoPointers
	}

	if ktyp.size > maxKeySize {
		ktyp = PtrTo(ktyp).(*rtype)
	}
	if etyp.size > maxValSize {
		etyp = PtrTo(etyp).(*rtype)
	}

	// Prepare GC data if any.
	// A bucket is at most bucketSize*(1+maxKeySize+maxValSize)+2*ptrSize bytes,
	// or 2072 bytes, or 259 pointer-size words, or 33 bytes of pointer bitmap.
	// Normally the enforced limit on pointer maps is 16 bytes,
	// but larger ones are acceptable, 33 bytes isn't too too big,
	// and it's easier to generate a pointer bitmap than a GC program.
	// Note that since the key and value are known to be <= 128 bytes,
	// they're guaranteed to have bitmaps instead of GC programs.
	// var gcdata *byte
	var ptrdata uintptr
	var overflowPad uintptr

	// On NaCl, pad if needed to make overflow end at the proper struct alignment.
	// On other systems, align > ptrSize is not possible.
	if runtime.GOARCH == "amd64p32" && (ktyp.align > ptrSize || etyp.align > ptrSize) {
		overflowPad = ptrSize
	}
	size := bucketSize*(1+ktyp.size+etyp.size) + overflowPad + ptrSize
	if size&uintptr(ktyp.align-1) != 0 || size&uintptr(etyp.align-1) != 0 {
		panic("reflect: bad size computation in MapOf")
	}

	if kind != kindNoPointers {
		nptr := (bucketSize*(1+ktyp.size+etyp.size) + ptrSize) / ptrSize
		mask := make([]byte, (nptr+7)/8)
		base := bucketSize / ptrSize

		if ktyp.kind&kindNoPointers == 0 {
			if ktyp.kind&kindGCProg != 0 {
				panic("reflect: unexpected GC program in MapOf")
			}
			kmask := (*[16]byte)(unsafe.Pointer( /*ktyp.gcdata*/ nil))
			for i := uintptr(0); i < ktyp.size/ptrSize; i++ {
				if (kmask[i/8]>>(i%8))&1 != 0 {
					for j := uintptr(0); j < bucketSize; j++ {
						word := base + j*ktyp.size/ptrSize + i
						mask[word/8] |= 1 << (word % 8)
					}
				}
			}
		}
		base += bucketSize * ktyp.size / ptrSize

		if etyp.kind&kindNoPointers == 0 {
			if etyp.kind&kindGCProg != 0 {
				panic("reflect: unexpected GC program in MapOf")
			}
			emask := (*[16]byte)(unsafe.Pointer( /*etyp.gcdata*/ nil))
			for i := uintptr(0); i < etyp.size/ptrSize; i++ {
				if (emask[i/8]>>(i%8))&1 != 0 {
					for j := uintptr(0); j < bucketSize; j++ {
						word := base + j*etyp.size/ptrSize + i
						mask[word/8] |= 1 << (word % 8)
					}
				}
			}
		}
		base += bucketSize * etyp.size / ptrSize
		base += overflowPad / ptrSize

		word := base
		mask[word/8] |= 1 << (word % 8)
		// gcdata = &mask[0]
		ptrdata = (word + 1) * ptrSize

		// overflow word must be last
		if ptrdata != size {
			panic("reflect: bad layout computation in MapOf")
		}
	}

	b := new(rtype)
	// b.size = gc.size
	// b.gc[0], _ = gc.finalize()
	b.kind |= kindGCProg
	s := "bucket(" + *ktyp.string + "," + *etyp.string + ")"
	b.string = &s
	return b
}

// Take the GC program for "t" and append it to the GC program "gc".
func appendGCProgram(gc []uintptr, t *rtype) []uintptr {
	p := t.gc
	p = unsafe.Pointer(uintptr(p) + unsafe.Sizeof(uintptr(0))) // skip size
loop:
	for {
		var argcnt int
		switch *(*uintptr)(p) {
		case _GC_END:
			// Note: _GC_END not included in append
			break loop
		case _GC_ARRAY_NEXT:
			argcnt = 0
		case _GC_APTR, _GC_STRING, _GC_EFACE, _GC_IFACE:
			argcnt = 1
		case _GC_PTR, _GC_CALL, _GC_CHAN_PTR, _GC_SLICE:
			argcnt = 2
		case _GC_ARRAY_START, _GC_REGION:
			argcnt = 3
		default:
			panic("unknown GC program op for " + *t.string + ": " + strconv.FormatUint(*(*uint64)(p), 10))
		}
		for i := 0; i < argcnt+1; i++ {
			gc = append(gc, *(*uintptr)(p))
			p = unsafe.Pointer(uintptr(p) + unsafe.Sizeof(uintptr(0)))
		}
	}
	return gc
}
func hMapOf(bucket *rtype) *rtype {
	ptrsize := unsafe.Sizeof(uintptr(0))

	// make gc program & compute hmap size
	gc := make([]uintptr, 1)           // first entry is size, filled in at the end
	offset := unsafe.Sizeof(uint(0))   // count
	offset += unsafe.Sizeof(uint32(0)) // flags
	offset += unsafe.Sizeof(uint32(0)) // hash0
	offset += unsafe.Sizeof(uint8(0))  // B
	offset += unsafe.Sizeof(uint8(0))  // keysize
	offset += unsafe.Sizeof(uint8(0))  // valuesize
	offset = (offset + 1) / 2 * 2
	offset += unsafe.Sizeof(uint16(0)) // bucketsize
	offset = (offset + ptrsize - 1) / ptrsize * ptrsize
	// gc = append(gc, _GC_PTR, offset, uintptr(bucket.gc)) // buckets
	offset += ptrsize
	// gc = append(gc, _GC_PTR, offset, uintptr(bucket.gc)) // oldbuckets
	offset += ptrsize
	offset += ptrsize // nevacuate
	gc = append(gc, _GC_END)
	gc[0] = offset

	h := new(rtype)
	h.size = offset
	// h.gc = unsafe.Pointer(&gc[0])
	s := "hmap(" + *bucket.string + ")"
	h.string = &s
	return h
}

// garbage collection bytecode program for slice of non-zero-length values.
// See ../../cmd/gc/reflect.c:/^dgcsym1 and :/^dgcsym.
type sliceGC struct {
	width  uintptr        // sizeof(slice)
	op     uintptr        // _GC_SLICE
	off    uintptr        // 0
	elemgc unsafe.Pointer // element gc program
	end    uintptr        // _GC_END
}

// garbage collection bytecode program for slice of zero-length values.
// See ../../cmd/gc/reflect.c:/^dgcsym1 and :/^dgcsym.
type sliceEmptyGC struct {
	width uintptr // sizeof(slice)
	op    uintptr // _GC_APTR
	off   uintptr // 0
	end   uintptr // _GC_END
}

var sliceEmptyGCProg = sliceEmptyGC{
	width: unsafe.Sizeof([]byte(nil)),
	op:    _GC_APTR,
	off:   0,
	end:   _GC_END,
}

// SliceOf returns the slice type with element type t.
// For example, if t represents int, SliceOf(t) represents []int.
func SliceOf(t Type) Type {
	typ := t.(*rtype)

	// Look in cache.
	ckey := cacheKey{Slice, typ, nil, 0}
	if slice := cacheGet(ckey); slice != nil {
		return slice
	}

	// Look in known types.
	s := "[]" + *typ.string

	// Make a slice type.
	var islice interface{} = ([]unsafe.Pointer)(nil)
	prototype := *(**sliceType)(unsafe.Pointer(&islice))
	slice := new(sliceType)
	*slice = *prototype
	slice.string = &s

	// gccgo uses a different hash.
	// slice.hash = fnv1(typ.hash, '[')
	slice.hash = typ.hash + 1 + 13

	slice.elem = typ
	slice.uncommonType = nil
	slice.ptrToThis = nil

	if typ.size == 0 {
		slice.gc = unsafe.Pointer(&sliceEmptyGCProg)
	} else {
		slice.gc = unsafe.Pointer(&sliceGC{
			width:  slice.size,
			op:     _GC_SLICE,
			off:    0,
			elemgc: typ.gc,
			end:    _GC_END,
		})
	}

	// INCORRECT. Uncomment to check that TestSliceOfOfGC fails when slice.gc is wrong.
	// slice.gc = unsafe.Pointer(&badGC{width: slice.size, end: _GC_END})

	return cachePut(ckey, &slice.rtype)
}

// See cmd/compile/internal/gc/reflect.go for derivation of constant.
const maxPtrmaskBytes = 2048

// ArrayOf returns the array type with the given count and element type.
// For example, if t represents int, ArrayOf(5, t) represents [5]int.
//
// If the resulting type would be larger than the available address space,
// ArrayOf panics.
func ArrayOf(count int, elem Type) Type {
	typ := elem.(*rtype)
	// call SliceOf here as it calls cacheGet/cachePut.
	// ArrayOf also calls cacheGet/cachePut and thus may modify the state of
	// the lookupCache mutex.
	slice := SliceOf(elem)

	// Look in cache.
	ckey := cacheKey{Array, typ, nil, uintptr(count)}
	if array := cacheGet(ckey); array != nil {
		return array
	}

	// Look in known types.
	s := "[" + strconv.Itoa(count) + "]" + *typ.string

	// Make an array type.
	var iarray interface{} = [1]unsafe.Pointer{}
	prototype := *(**arrayType)(unsafe.Pointer(&iarray))
	array := new(arrayType)
	*array = *prototype
	array.string = &s

	// gccgo uses a different hash.
	// array.hash = fnv1(typ.hash, '[')
	// for n := uint32(count); n > 0; n >>= 8 {
	// 	array.hash = fnv1(array.hash, byte(n))
	// }
	// array.hash = fnv1(array.hash, ']')
	array.hash = typ.hash + 1 + 13

	array.elem = typ
	max := ^uintptr(0) / typ.size
	if uintptr(count) > max {
		panic("reflect.ArrayOf: array size would exceed virtual address space")
	}
	array.size = typ.size * uintptr(count)
	// if count > 0 && typ.ptrdata != 0 {
	// 	array.ptrdata = typ.size*uintptr(count-1) + typ.ptrdata
	// }
	array.align = typ.align
	array.fieldAlign = typ.fieldAlign
	array.uncommonType = nil
	array.ptrToThis = nil
	array.len = uintptr(count)
	array.slice = slice.(*rtype)

	array.kind &^= kindNoPointers
	switch {
	case typ.kind&kindNoPointers != 0 || array.size == 0:
		// No pointers.
		array.kind |= kindNoPointers
		gc := [...]uintptr{array.size, _GC_END}
		array.gc = unsafe.Pointer(&gc[0])

	case count == 1:
		// In memory, 1-element array looks just like the element.
		array.kind |= typ.kind & kindGCProg
		array.gc = typ.gc

	default:
		gc := []uintptr{array.size, _GC_ARRAY_START, 0, uintptr(count), typ.size}
		gc = appendGCProgram(gc, typ)
		gc = append(gc, _GC_ARRAY_NEXT, _GC_END)
		array.gc = unsafe.Pointer(&gc[0])
	}

	array.kind &^= kindDirectIface

	array.hashfn = func(p unsafe.Pointer, size uintptr) uintptr {
		ret := uintptr(0)
		for i := 0; i < count; i++ {
			ret *= 33
			ret += typ.hashfn(p, typ.size)
			p = unsafe.Pointer(uintptr(p) + typ.size)
		}
		return ret
	}

	array.equalfn = func(p1, p2 unsafe.Pointer, size uintptr) bool {
		for i := 0; i < count; i++ {
			if !typ.equalfn(p1, p2, typ.size) {
				return false
			}
			p1 = unsafe.Pointer(uintptr(p1) + typ.size)
			p2 = unsafe.Pointer(uintptr(p2) + typ.size)
		}
		return true
	}

	return cachePut(ckey, &array.rtype)
}

func appendVarint(x []byte, v uintptr) []byte {
	for ; v >= 0x80; v >>= 7 {
		x = append(x, byte(v|0x80))
	}
	x = append(x, byte(v))
	return x
}

// toType converts from a *rtype to a Type that can be returned
// to the client of package reflect. In gc, the only concern is that
// a nil *rtype must be replaced by a nil Type, but in gccgo this
// function takes care of ensuring that multiple *rtype for the same
// type are coalesced into a single Type.
var canonicalType = make(map[string]Type)

var canonicalTypeLock sync.RWMutex

func canonicalize(t Type) Type {
	if t == nil {
		return nil
	}
	s := t.rawString()
	canonicalTypeLock.RLock()
	if r, ok := canonicalType[s]; ok {
		canonicalTypeLock.RUnlock()
		return r
	}
	canonicalTypeLock.RUnlock()
	canonicalTypeLock.Lock()
	if r, ok := canonicalType[s]; ok {
		canonicalTypeLock.Unlock()
		return r
	}
	canonicalType[s] = t
	canonicalTypeLock.Unlock()
	return t
}

func toType(p *rtype) Type {
	if p == nil {
		return nil
	}
	return canonicalize(p)
}

// ifaceIndir reports whether t is stored indirectly in an interface value.
func ifaceIndir(t *rtype) bool {
	return t.kind&kindDirectIface == 0
}

// Layout matches runtime.BitVector (well enough).
type bitVector struct {
	n    uint32 // number of bits
	data []byte
}

// append a bit to the bitmap.
func (bv *bitVector) append(bit uint8) {
	if bv.n%8 == 0 {
		bv.data = append(bv.data, 0)
	}
	bv.data[bv.n/8] |= bit << (bv.n % 8)
	bv.n++
}

func addTypeBits(bv *bitVector, offset uintptr, t *rtype) {
	if t.kind&kindNoPointers != 0 {
		return
	}

	switch Kind(t.kind & kindMask) {
	case Chan, Func, Map, Ptr, Slice, String, UnsafePointer:
		// 1 pointer at start of representation
		for bv.n < uint32(offset/uintptr(ptrSize)) {
			bv.append(0)
		}
		bv.append(1)

	case Interface:
		// 2 pointers
		for bv.n < uint32(offset/uintptr(ptrSize)) {
			bv.append(0)
		}
		bv.append(1)
		bv.append(1)

	case Array:
		// repeat inner type
		tt := (*arrayType)(unsafe.Pointer(t))
		for i := 0; i < int(tt.len); i++ {
			addTypeBits(bv, offset+uintptr(i)*tt.elem.size, tt.elem)
		}

	case Struct:
		// apply fields
		tt := (*structType)(unsafe.Pointer(t))
		for i := range tt.fields {
			f := &tt.fields[i]
			addTypeBits(bv, offset+f.offset, f.typ)
		}
	}
}
