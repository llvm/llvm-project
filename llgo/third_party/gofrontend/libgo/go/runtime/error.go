// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// The Error interface identifies a run time error.
type Error interface {
	error

	// RuntimeError is a no-op function but
	// serves to distinguish types that are run time
	// errors from ordinary errors: a type is a
	// run time error if it has a RuntimeError method.
	RuntimeError()
}

// A TypeAssertionError explains a failed type assertion.
type TypeAssertionError struct {
	interfaceString string
	concreteString  string
	assertedString  string
	missingMethod   string // one method needed by Interface, missing from Concrete
}

func (*TypeAssertionError) RuntimeError() {}

func (e *TypeAssertionError) Error() string {
	inter := e.interfaceString
	if inter == "" {
		inter = "interface"
	}
	if e.concreteString == "" {
		return "interface conversion: " + inter + " is nil, not " + e.assertedString
	}
	if e.missingMethod == "" {
		return "interface conversion: " + inter + " is " + e.concreteString +
			", not " + e.assertedString
	}
	return "interface conversion: " + e.concreteString + " is not " + e.assertedString +
		": missing method " + e.missingMethod
}

// For calling from C.
func NewTypeAssertionError(ps1, ps2, ps3 *string, pmeth *string, ret *interface{}) {
	var s1, s2, s3, meth string

	if ps1 != nil {
		s1 = *ps1
	}
	if ps2 != nil {
		s2 = *ps2
	}
	if ps3 != nil {
		s3 = *ps3
	}
	if pmeth != nil {
		meth = *pmeth
	}

	// For gccgo, strip out quoted strings.
	s1 = unquote(s1)
	s2 = unquote(s2)
	s3 = unquote(s3)

	*ret = &TypeAssertionError{s1, s2, s3, meth}
}

// Remove quoted strings from gccgo reflection strings.
func unquote(s string) string {
	ls := len(s)
	var i int
	for i = 0; i < ls; i++ {
		if s[i] == '\t' {
			break
		}
	}
	if i == ls {
		return s
	}
	var q bool
	r := make([]byte, len(s))
	j := 0
	for i = 0; i < ls; i++ {
		if s[i] == '\t' {
			q = !q
		} else if !q {
			r[j] = s[i]
			j++
		}
	}
	return string(r[:j])
}

// An errorString represents a runtime error described by a single string.
type errorString string

func (e errorString) RuntimeError() {}

func (e errorString) Error() string {
	return "runtime error: " + string(e)
}

// For calling from C.
func NewErrorString(s string, ret *interface{}) {
	*ret = errorString(s)
}

// An errorCString represents a runtime error described by a single C string.
// Not "type errorCString uintptr" because of http://golang.org/issue/7084.
type errorCString struct{ cstr uintptr }

func (e errorCString) RuntimeError() {}

func cstringToGo(uintptr) string

func (e errorCString) Error() string {
	return "runtime error: " + cstringToGo(e.cstr)
}

// For calling from C.
func NewErrorCString(s uintptr, ret *interface{}) {
	*ret = errorCString{s}
}

type stringer interface {
	String() string
}

func typestring(interface{}) string

// For calling from C.
// Prints an argument passed to panic.
// There's room for arbitrary complexity here, but we keep it
// simple and handle just a few important cases: int, string, and Stringer.
func Printany(i interface{}) {
	switch v := i.(type) {
	case nil:
		print("nil")
	case stringer:
		print(v.String())
	case error:
		print(v.Error())
	case int:
		print(v)
	case string:
		print(v)
	default:
		print("(", typestring(i), ") ", i)
	}
}

// called from generated code
func panicwrap(pkg, typ, meth string) {
	panic("value method " + pkg + "." + typ + "." + meth + " called using nil *" + typ + " pointer")
}
