// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

// MakeRO returns a copy of v with the read-only flag set.
func MakeRO(v Value) Value {
	v.flag |= flagStickyRO
	return v
}

// IsRO reports whether v's read-only flag is set.
func IsRO(v Value) bool {
	return v.flag&flagStickyRO != 0
}

var CallGC = &callGC

const PtrSize = ptrSize

func FuncLayout(t Type, rcvr Type) (frametype Type, argSize, retOffset uintptr, stack []byte, gc []byte, ptrs bool) {
	return
}

func TypeLinks() []string {
	return nil
}

var GCBits = gcbits

// Will be provided by runtime eventually.
func gcbits(interface{}) []byte {
	return nil
}

func MapBucketOf(x, y Type) Type {
	return nil
}

func CachedBucketOf(m Type) Type {
	return nil
}
