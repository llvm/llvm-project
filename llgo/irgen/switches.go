//===- switches.go - misc utils -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements transformations and IR generation for switches.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"go/token"

	"llvm.org/llgo/third_party/gotools/go/exact"
	"llvm.org/llgo/third_party/gotools/go/ssa"
	"llvm.org/llgo/third_party/gotools/go/ssa/ssautil"
	"llvm.org/llvm/bindings/go/llvm"
)

// switchInstr is an instruction representing a switch on constant
// integer values.
type switchInstr struct {
	ssa.Instruction
	ssautil.Switch
}

func (sw *switchInstr) String() string {
	return sw.Switch.String()
}

func (sw *switchInstr) Parent() *ssa.Function {
	return sw.Default.Instrs[0].Parent()
}

func (sw *switchInstr) Block() *ssa.BasicBlock {
	return sw.Start
}

func (sw *switchInstr) Operands(rands []*ssa.Value) []*ssa.Value {
	return nil
}

func (sw *switchInstr) Pos() token.Pos {
	return token.NoPos
}

// emitSwitch emits an LLVM switch instruction.
func (fr *frame) emitSwitch(instr *switchInstr) {
	cases, _ := dedupConstCases(fr, instr.ConstCases)
	ncases := len(cases)
	elseblock := fr.block(instr.Default)
	llswitch := fr.builder.CreateSwitch(fr.llvmvalue(instr.X), elseblock, ncases)
	for _, c := range cases {
		llswitch.AddCase(fr.llvmvalue(c.Value), fr.block(c.Body))
	}
}

// transformSwitches replaces the final If statement in start blocks
// with a high-level switch instruction, and erases chained condition
// blocks.
func (fr *frame) transformSwitches(f *ssa.Function) {
	for _, sw := range ssautil.Switches(f) {
		if sw.ConstCases == nil {
			// TODO(axw) investigate switch
			// on hashes in type switches.
			continue
		}
		if !isInteger(sw.X.Type()) && !isBoolean(sw.X.Type()) {
			// LLVM switches can only operate on integers.
			continue
		}
		instr := &switchInstr{Switch: sw}
		sw.Start.Instrs[len(sw.Start.Instrs)-1] = instr
		for _, c := range sw.ConstCases[1:] {
			fr.blocks[c.Block.Index].EraseFromParent()
			fr.blocks[c.Block.Index] = llvm.BasicBlock{}
		}

		// Fix predecessors in successor blocks for fixupPhis.
		cases, duplicates := dedupConstCases(fr, instr.ConstCases)
		for _, c := range cases {
			for _, succ := range c.Block.Succs {
				for i, pred := range succ.Preds {
					if pred == c.Block {
						succ.Preds[i] = sw.Start
						break
					}
				}
			}
		}

		// Remove redundant edges corresponding to duplicate cases
		// that will not feature in the LLVM switch instruction.
		for _, c := range duplicates {
			for _, succ := range c.Block.Succs {
				for i, pred := range succ.Preds {
					if pred == c.Block {
						head := succ.Preds[:i]
						tail := succ.Preds[i+1:]
						succ.Preds = append(head, tail...)
						removePhiEdge(succ, i)
						break
					}
				}
			}
		}
	}
}

// dedupConstCases separates duplicate const cases.
//
// TODO(axw) fix this in go/ssa/ssautil.
func dedupConstCases(fr *frame, in []ssautil.ConstCase) (unique, duplicates []ssautil.ConstCase) {
	unique = make([]ssautil.ConstCase, 0, len(in))
dedup:
	for i, c1 := range in {
		for _, c2 := range in[i+1:] {
			if exact.Compare(c1.Value.Value, token.EQL, c2.Value.Value) {
				duplicates = append(duplicates, c1)
				continue dedup
			}
		}
		unique = append(unique, c1)
	}
	return unique, duplicates
}

// removePhiEdge removes the i'th edge from each PHI
// instruction in the specified basic block.
func removePhiEdge(bb *ssa.BasicBlock, i int) {
	for _, instr := range bb.Instrs {
		instr, ok := instr.(*ssa.Phi)
		if !ok {
			return
		}
		head := instr.Edges[:i]
		tail := instr.Edges[i+1:]
		instr.Edges = append(head, tail...)
	}
}
