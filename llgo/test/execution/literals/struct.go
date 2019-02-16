// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 1 2
// CHECK-NEXT: 1 2
// CHECK-NEXT: 0 1 2
// CHECK-NEXT: 1 2
// CHECK-NEXT: 3 4

package main

type E struct {
	e *E
}

type S struct {
	*E
	a, b int
}

type File struct {
}

type Reader struct {
}

type Response struct {
}

type reader struct {
	*Reader
	fd   *File
	resp *Response
}

type Range32 struct {
	Lo     uint32
	Hi     uint32
	Stride uint32
}

func main() {
	s := &S{nil, 1, 2}
	println(s.a, s.b)
	s = &S{a: 1, b: 2}
	println(s.a, s.b)

	_ = &reader{}

	r := Range32{
		Lo:     0,
		Stride: 2,
		Hi:     1,
	}
	println(r.Lo, r.Hi, r.Stride)

	// slice of structs
	ss := []S{{nil, 1, 2}, {nil, 3, 4}}
	for _, s := range ss {
		println(s.a, s.b)
	}
}
