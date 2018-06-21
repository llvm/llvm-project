// sleep_select.go -- Sleep using select.

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func Sleep(nsec int64) (err error) {
	tv := NsecToTimeval(nsec);
	_, err = Select(0, nil, nil, nil, &tv);
	return err;
}
