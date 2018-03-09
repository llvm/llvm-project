// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func (ts *Timestruc) Unix() (sec int64, nsec int64) {
	return int64(ts.Sec), int64(ts.Nsec)
}

func (ts *Timestruc) Nano() int64 {
	return int64(ts.Sec)*1e9 + int64(ts.Nsec)
}
