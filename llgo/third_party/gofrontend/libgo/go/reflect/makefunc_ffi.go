// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"unsafe"
)

// The makeFuncFFI function, written in C, fills in an FFI closure.
// It arranges for ffiCall to be invoked directly from FFI.
func makeFuncFFI(ftyp *funcType, impl unsafe.Pointer)

// FFICallbackGo implements the Go side of the libffi callback.
// It is exported so that C code can call it.
//
// The call chain arriving here looks like
//   some_go_caller
//   ->some_ffi_internals
//     ->ffi_callback (in C)
//       ->FFICallbackGo
//
// The ffi_callback handles __go_makefunc_can_recover, and
// then passes off the data as received from ffi here.

func FFICallbackGo(results unsafe.Pointer, params unsafe.Pointer, impl *makeFuncImpl) {
	ftyp := impl.typ
	in := make([]Value, 0, len(ftyp.in))
	ap := params
	for _, rt := range ftyp.in {
		p := unsafe_New(rt)
		memmove(p, *(*unsafe.Pointer)(ap), rt.size)
		v := Value{rt, p, flag(rt.Kind()) | flagIndir}
		in = append(in, v)
		ap = (unsafe.Pointer)(uintptr(ap) + ptrSize)
	}

	out := impl.call(in)

	off := uintptr(0)
	for i, typ := range ftyp.out {
		v := out[i]
		if v.typ != typ {
			panic("reflect: function created by MakeFunc using " + funcName(impl.fn) +
				" returned wrong type: have " +
				out[i].typ.String() + " for " + typ.String())
		}
		if v.flag&flagRO != 0 {
			panic("reflect: function created by MakeFunc using " + funcName(impl.fn) +
				" returned value obtained from unexported field")
		}

		off = align(off, uintptr(typ.fieldAlign))
		addr := unsafe.Pointer(uintptr(results) + off)
		if v.flag&flagIndir == 0 && (v.kind() == Ptr || v.kind() == UnsafePointer) {
			*(*unsafe.Pointer)(addr) = v.ptr
		} else {
			memmove(addr, v.ptr, typ.size)
		}
		off += typ.size
	}
}
