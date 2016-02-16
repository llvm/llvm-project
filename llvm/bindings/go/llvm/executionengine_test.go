//===- executionengine_test.go - Tests for executionengine ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests bindings for the executionengine component.
//
//===----------------------------------------------------------------------===//

package llvm

import (
	"testing"
)

func TestFactorial(t *testing.T) {
	LinkInMCJIT()
	InitializeNativeTarget()
	InitializeNativeAsmPrinter()

	mod := NewModule("fac_module")

	fac_args := []Type{Int32Type()}
	fac_type := FunctionType(Int32Type(), fac_args, false)
	fac := AddFunction(mod, "fac", fac_type)
	fac.SetFunctionCallConv(CCallConv)
	n := fac.Param(0)

	entry := AddBasicBlock(fac, "entry")
	iftrue := AddBasicBlock(fac, "iftrue")
	iffalse := AddBasicBlock(fac, "iffalse")
	end := AddBasicBlock(fac, "end")

	builder := NewBuilder()
	defer builder.Dispose()

	builder.SetInsertPointAtEnd(entry)
	If := builder.CreateICmp(IntEQ, n, ConstInt(Int32Type(), 0, false), "cmptmp")
	builder.CreateCondBr(If, iftrue, iffalse)

	builder.SetInsertPointAtEnd(iftrue)
	res_iftrue := ConstInt(Int32Type(), 1, false)
	builder.CreateBr(end)

	builder.SetInsertPointAtEnd(iffalse)
	n_minus := builder.CreateSub(n, ConstInt(Int32Type(), 1, false), "subtmp")
	call_fac_args := []Value{n_minus}
	call_fac := builder.CreateCall(fac, call_fac_args, "calltmp")
	res_iffalse := builder.CreateMul(n, call_fac, "multmp")
	builder.CreateBr(end)

	builder.SetInsertPointAtEnd(end)
	res := builder.CreatePHI(Int32Type(), "result")
	phi_vals := []Value{res_iftrue, res_iffalse}
	phi_blocks := []BasicBlock{iftrue, iffalse}
	res.AddIncoming(phi_vals, phi_blocks)
	builder.CreateRet(res)

	err := VerifyModule(mod, ReturnStatusAction)
	if err != nil {
		t.Errorf("Error verifying module: %s", err)
		return
	}

	options := NewMCJITCompilerOptions()
	options.SetMCJITOptimizationLevel(2)
	options.SetMCJITEnableFastISel(true)
	options.SetMCJITNoFramePointerElim(true)
	options.SetMCJITCodeModel(CodeModelJITDefault)
	engine, err := NewMCJITCompiler(mod, options)
	if err != nil {
		t.Errorf("Error creating JIT: %s", err)
		return
	}
	defer engine.Dispose()

	pass := NewPassManager()
	defer pass.Dispose()

	pass.AddConstantPropagationPass()
	pass.AddInstructionCombiningPass()
	pass.AddPromoteMemoryToRegisterPass()
	pass.AddGVNPass()
	pass.AddCFGSimplificationPass()
	pass.Run(mod)

	exec_args := []GenericValue{NewGenericValueFromInt(Int32Type(), 10, false)}
	exec_res := engine.RunFunction(fac, exec_args)
	var fac10 uint64 = 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
	if exec_res.Int(false) != fac10 {
		t.Errorf("Expected %d, got %d", fac10, exec_res.Int(false))
	}
}
