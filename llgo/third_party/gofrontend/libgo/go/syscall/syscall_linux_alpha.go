// syscall_linux_alpha.go -- GNU/Linux ALPHA specific support

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

type PtraceRegs struct {
	R0      uint64
	R1      uint64
	R2      uint64
	R3      uint64
	R4      uint64
	R5      uint64
	R6      uint64
	R7      uint64
	R8      uint64
	R19     uint64
	R20     uint64
	R21     uint64
	R22     uint64
	R23     uint64
	R24     uint64
	R25     uint64
	R26     uint64
	R27     uint64
	R28     uint64
	Hae     uint64
	Trap_a0 uint64
	Trap_a1 uint64
	Trap_a2 uint64
	Ps      uint64
	Pc      uint64
	Gp      uint64
	R16     uint64
	R17     uint64
	R18     uint64
}

func (r *PtraceRegs) PC() uint64 {
	return r.Pc
}

func (r *PtraceRegs) SetPC(pc uint64) {
	r.Pc = pc
}

func PtraceGetRegs(pid int, regsout *PtraceRegs) (err error) {
	return ptrace(PTRACE_GETREGS, pid, 0, uintptr(unsafe.Pointer(regsout)))
}

func PtraceSetRegs(pid int, regs *PtraceRegs) (err error) {
	return ptrace(PTRACE_SETREGS, pid, 0, uintptr(unsafe.Pointer(regs)))
}
