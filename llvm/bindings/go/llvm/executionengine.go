//===- executionengine.go - Bindings for executionengine ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the executionengine component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Core.h"
#include "llvm-c/ExecutionEngine.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"
import "errors"

func LinkInMCJIT()       { C.LLVMLinkInMCJIT() }
func LinkInInterpreter() { C.LLVMLinkInInterpreter() }

type GenericValue struct {
	C C.LLVMGenericValueRef
}
type ExecutionEngine struct {
	C C.LLVMExecutionEngineRef
}

type MCJITCompilerOptions struct {
	C C.struct_LLVMMCJITCompilerOptions
}

func (options *MCJITCompilerOptions) SetMCJITOptimizationLevel(level uint) {
	options.C.OptLevel = C.uint(level)
}

func (options *MCJITCompilerOptions) SetMCJITNoFramePointerElim(nfp bool) {
	options.C.NoFramePointerElim = boolToLLVMBool(nfp)
}

func (options *MCJITCompilerOptions) SetMCJITEnableFastISel(fastisel bool) {
	options.C.EnableFastISel = boolToLLVMBool(fastisel)
}

func (options *MCJITCompilerOptions) SetMCJITCodeModel(CodeModel CodeModel) {
	options.C.CodeModel = C.LLVMCodeModel(CodeModel)
}

// helpers
func llvmGenericValueRefPtr(t *GenericValue) *C.LLVMGenericValueRef {
	return (*C.LLVMGenericValueRef)(unsafe.Pointer(t))
}

//-------------------------------------------------------------------------
// llvm.GenericValue
//-------------------------------------------------------------------------

func NewGenericValueFromInt(t Type, n uint64, signed bool) (g GenericValue) {
	g.C = C.LLVMCreateGenericValueOfInt(t.C, C.ulonglong(n), boolToLLVMBool(signed))
	return
}
func NewGenericValueFromPointer(p unsafe.Pointer) (g GenericValue) {
	g.C = C.LLVMCreateGenericValueOfPointer(p)
	return
}
func NewGenericValueFromFloat(t Type, n float64) (g GenericValue) {
	g.C = C.LLVMCreateGenericValueOfFloat(t.C, C.double(n))
	return
}
func (g GenericValue) IntWidth() int { return int(C.LLVMGenericValueIntWidth(g.C)) }
func (g GenericValue) Int(signed bool) uint64 {
	return uint64(C.LLVMGenericValueToInt(g.C, boolToLLVMBool(signed)))
}
func (g GenericValue) Float(t Type) float64 {
	return float64(C.LLVMGenericValueToFloat(t.C, g.C))
}
func (g GenericValue) Pointer() unsafe.Pointer {
	return C.LLVMGenericValueToPointer(g.C)
}
func (g GenericValue) Dispose() { C.LLVMDisposeGenericValue(g.C) }

//-------------------------------------------------------------------------
// llvm.ExecutionEngine
//-------------------------------------------------------------------------

func NewExecutionEngine(m Module) (ee ExecutionEngine, err error) {
	var cmsg *C.char
	fail := C.LLVMCreateExecutionEngineForModule(&ee.C, m.C, &cmsg)
	if fail != 0 {
		ee.C = nil
		err = errors.New(C.GoString(cmsg))
		C.LLVMDisposeMessage(cmsg)
	}
	return
}

func NewInterpreter(m Module) (ee ExecutionEngine, err error) {
	var cmsg *C.char
	fail := C.LLVMCreateInterpreterForModule(&ee.C, m.C, &cmsg)
	if fail != 0 {
		ee.C = nil
		err = errors.New(C.GoString(cmsg))
		C.LLVMDisposeMessage(cmsg)
	}
	return
}

func NewMCJITCompilerOptions() MCJITCompilerOptions {
	var options C.struct_LLVMMCJITCompilerOptions
	C.LLVMInitializeMCJITCompilerOptions(&options, C.size_t(unsafe.Sizeof(C.struct_LLVMMCJITCompilerOptions{})))
	return MCJITCompilerOptions{options}
}

func NewMCJITCompiler(m Module, options MCJITCompilerOptions) (ee ExecutionEngine, err error) {
	var cmsg *C.char
	fail := C.LLVMCreateMCJITCompilerForModule(&ee.C, m.C, &options.C, C.size_t(unsafe.Sizeof(C.struct_LLVMMCJITCompilerOptions{})), &cmsg)
	if fail != 0 {
		ee.C = nil
		err = errors.New(C.GoString(cmsg))
		C.LLVMDisposeMessage(cmsg)
	}
	return
}

func (ee ExecutionEngine) Dispose()               { C.LLVMDisposeExecutionEngine(ee.C) }
func (ee ExecutionEngine) RunStaticConstructors() { C.LLVMRunStaticConstructors(ee.C) }
func (ee ExecutionEngine) RunStaticDestructors()  { C.LLVMRunStaticDestructors(ee.C) }

func (ee ExecutionEngine) RunFunction(f Value, args []GenericValue) (g GenericValue) {
	nargs := len(args)
	var argptr *GenericValue
	if nargs > 0 {
		argptr = &args[0]
	}
	g.C = C.LLVMRunFunction(ee.C, f.C,
		C.unsigned(nargs), llvmGenericValueRefPtr(argptr))
	return
}

func (ee ExecutionEngine) FreeMachineCodeForFunction(f Value) {
	C.LLVMFreeMachineCodeForFunction(ee.C, f.C)
}
func (ee ExecutionEngine) AddModule(m Module) { C.LLVMAddModule(ee.C, m.C) }

func (ee ExecutionEngine) RemoveModule(m Module) {
	var modtmp C.LLVMModuleRef
	C.LLVMRemoveModule(ee.C, m.C, &modtmp, nil)
}

func (ee ExecutionEngine) FindFunction(name string) (f Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.LLVMFindFunction(ee.C, cname, &f.C)
	return
}

func (ee ExecutionEngine) RecompileAndRelinkFunction(f Value) unsafe.Pointer {
	return C.LLVMRecompileAndRelinkFunction(ee.C, f.C)
}

func (ee ExecutionEngine) TargetData() (td TargetData) {
	td.C = C.LLVMGetExecutionEngineTargetData(ee.C)
	return
}

func (ee ExecutionEngine) AddGlobalMapping(global Value, addr unsafe.Pointer) {
	C.LLVMAddGlobalMapping(ee.C, global.C, addr)
}

func (ee ExecutionEngine) PointerToGlobal(global Value) unsafe.Pointer {
	return C.LLVMGetPointerToGlobal(ee.C, global.C)
}
