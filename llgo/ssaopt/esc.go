// Copyright 2014 The llgo Authors.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package ssaopt

import (
	"go/token"

	"llvm.org/llgo/third_party/gotools/go/ssa"
)

func escapes(val ssa.Value, bb *ssa.BasicBlock, pending []ssa.Value) bool {
	for _, p := range pending {
		if val == p {
			return false
		}
	}

	for _, ref := range *val.Referrers() {
		switch ref := ref.(type) {
		case *ssa.Phi:
			// We must consider the variable to have escaped if it is
			// possible for the program to see more than one "version"
			// of the variable at once, as this requires the program
			// to use heap allocation for the multiple versions.
			//
			// I (pcc) think that this is only possible (without stores)
			// in the case where a phi node that (directly or indirectly)
			// refers to the allocation dominates the allocation.
			if ref.Block().Dominates(bb) {
				return true
			}
			if escapes(ref, bb, append(pending, val)) {
				return true
			}

		case *ssa.BinOp, *ssa.ChangeType, *ssa.Convert, *ssa.ChangeInterface, *ssa.MakeInterface, *ssa.Slice, *ssa.FieldAddr, *ssa.IndexAddr, *ssa.TypeAssert, *ssa.Extract:
			if escapes(ref.(ssa.Value), bb, append(pending, val)) {
				return true
			}

		case *ssa.Range, *ssa.DebugRef:
			continue

		case *ssa.UnOp:
			if ref.Op == token.MUL || ref.Op == token.ARROW {
				continue
			}
			if escapes(ref, bb, append(pending, val)) {
				return true
			}

		case *ssa.Store:
			if val == ref.Val {
				return true
			}

		case *ssa.Call:
			if builtin, ok := ref.Call.Value.(*ssa.Builtin); ok {
				switch builtin.Name() {
				case "cap", "len", "copy", "ssa:wrapnilchk":
					continue
				case "append":
					if ref.Call.Args[0] == val && escapes(ref, bb, append(pending, val)) {
						return true
					}
				default:
					return true
				}
			} else {
				return true
			}

		default:
			return true
		}
	}

	return false
}

func LowerAllocsToStack(f *ssa.Function) {
	pending := make([]ssa.Value, 0, 10)

	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			if alloc, ok := instr.(*ssa.Alloc); ok && alloc.Heap && !escapes(alloc, alloc.Block(), pending) {
				alloc.Heap = false
				f.Locals = append(f.Locals, alloc)
			}
		}
	}
}
