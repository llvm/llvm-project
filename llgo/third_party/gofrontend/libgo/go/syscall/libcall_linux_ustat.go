// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GNU/Linux library ustat call.
// This is not supported on some kernels, such as arm64.

package syscall

//sys	Ustat(dev int, ubuf *Ustat_t) (err error)
//ustat(dev _dev_t, ubuf *Ustat_t) _C_int
