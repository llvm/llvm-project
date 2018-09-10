// RUN: llgo -o %t %s
// RUN: %t 2>&1 | count 0

package main

import "io"

type rdr struct{}

func (r rdr) Read(b []byte) (int, error) {
	return 0, nil
}

func F(i interface{}) {
	_ = i.(io.Reader)
}

func main() {
	var r rdr
	F(r)
	F(&r)
}
