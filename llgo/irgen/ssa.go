//===- ssa.go - IR generation from go/ssa ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the top-level LLVM IR generation from go/ssa form.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"sort"

	"llvm.org/llgo/ssaopt"
	"llvm.org/llgo/third_party/gotools/go/ssa"
	"llvm.org/llgo/third_party/gotools/go/ssa/ssautil"
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llvm/bindings/go/llvm"
)

// A globalInit is used to temporarily store a global's initializer until
// we are ready to build it.
type globalInit struct {
	val   llvm.Value
	elems []globalInit
}

func (gi *globalInit) update(typ llvm.Type, indices []uint32, val llvm.Value) {
	if len(indices) == 0 {
		gi.val = val
		return
	}

	if gi.val.C != nil {
		gi.val = llvm.ConstInsertValue(gi.val, val, indices)
	}

	tk := typ.TypeKind()

	if len(gi.elems) == 0 {
		switch tk {
		case llvm.StructTypeKind:
			gi.elems = make([]globalInit, typ.StructElementTypesCount())
		case llvm.ArrayTypeKind:
			gi.elems = make([]globalInit, typ.ArrayLength())
		default:
			panic("unexpected type")
		}
	}

	var eltyp llvm.Type
	switch tk {
	case llvm.StructTypeKind:
		eltyp = typ.StructElementTypes()[indices[0]]
	case llvm.ArrayTypeKind:
		eltyp = typ.ElementType()
	default:
		panic("unexpected type")
	}

	gi.elems[indices[0]].update(eltyp, indices[1:], val)
}

func (gi *globalInit) build(typ llvm.Type) llvm.Value {
	if gi.val.C != nil {
		return gi.val
	}
	if len(gi.elems) == 0 {
		return llvm.ConstNull(typ)
	}

	switch typ.TypeKind() {
	case llvm.StructTypeKind:
		eltypes := typ.StructElementTypes()
		elems := make([]llvm.Value, len(eltypes))
		for i, eltyp := range eltypes {
			elems[i] = gi.elems[i].build(eltyp)
		}
		return llvm.ConstStruct(elems, false)
	case llvm.ArrayTypeKind:
		eltyp := typ.ElementType()
		elems := make([]llvm.Value, len(gi.elems))
		for i := range gi.elems {
			elems[i] = gi.elems[i].build(eltyp)
		}
		return llvm.ConstArray(eltyp, elems)
	default:
		panic("unexpected type")
	}
}

type unit struct {
	*compiler
	pkg         *ssa.Package
	globals     map[ssa.Value]llvm.Value
	globalInits map[llvm.Value]*globalInit

	// funcDescriptors maps *ssa.Functions to function descriptors,
	// the first-class representation of functions.
	funcDescriptors map[*ssa.Function]llvm.Value

	// undefinedFuncs contains functions that have been resolved
	// (declared) but not defined.
	undefinedFuncs map[*ssa.Function]bool

	gcRoots []llvm.Value
}

func newUnit(c *compiler, pkg *ssa.Package) *unit {
	u := &unit{
		compiler:        c,
		pkg:             pkg,
		globals:         make(map[ssa.Value]llvm.Value),
		globalInits:     make(map[llvm.Value]*globalInit),
		funcDescriptors: make(map[*ssa.Function]llvm.Value),
		undefinedFuncs:  make(map[*ssa.Function]bool),
	}
	return u
}

type byMemberName []ssa.Member

func (ms byMemberName) Len() int { return len(ms) }
func (ms byMemberName) Swap(i, j int) {
	ms[i], ms[j] = ms[j], ms[i]
}
func (ms byMemberName) Less(i, j int) bool {
	return ms[i].Name() < ms[j].Name()
}

type byFunctionString []*ssa.Function

func (fns byFunctionString) Len() int { return len(fns) }
func (fns byFunctionString) Swap(i, j int) {
	fns[i], fns[j] = fns[j], fns[i]
}
func (fns byFunctionString) Less(i, j int) bool {
	return fns[i].String() < fns[j].String()
}

// Emit functions in order of their fully qualified names. This is so that a
// bootstrap build can be verified by comparing the stage2 and stage3 binaries.
func (u *unit) defineFunctionsInOrder(functions map[*ssa.Function]bool) {
	fns := []*ssa.Function{}
	for f, _ := range functions {
		fns = append(fns, f)
	}
	sort.Sort(byFunctionString(fns))
	for _, f := range fns {
		u.defineFunction(f)
	}
}

// translatePackage translates an *ssa.Package into an LLVM module, and returns
// the translation unit information.
func (u *unit) translatePackage(pkg *ssa.Package) {
	ms := make([]ssa.Member, len(pkg.Members))
	i := 0
	for _, m := range pkg.Members {
		ms[i] = m
		i++
	}

	sort.Sort(byMemberName(ms))

	// Initialize global storage and type descriptors for this package.
	// We must create globals regardless of whether they're referenced,
	// hence the duplication in frame.value.
	for _, m := range ms {
		switch v := m.(type) {
		case *ssa.Global:
			elemtyp := deref(v.Type())
			llelemtyp := u.llvmtypes.ToLLVM(elemtyp)
			vname := u.types.mc.mangleGlobalName(v)
			global := llvm.AddGlobal(u.module.Module, llelemtyp, vname)
			if !v.Object().Exported() {
				global.SetLinkage(llvm.InternalLinkage)
			}
			u.addGlobal(global, elemtyp)
			global = llvm.ConstBitCast(global, u.llvmtypes.ToLLVM(v.Type()))
			u.globals[v] = global
		case *ssa.Type:
			u.types.getTypeDescriptorPointer(v.Type())
		}
	}

	// Define functions.
	u.defineFunctionsInOrder(ssautil.AllFunctions(pkg.Prog))

	// Emit initializers for type descriptors, which may trigger
	// the resolution of additional functions.
	u.types.emitTypeDescInitializers()

	// Define remaining functions that were resolved during
	// runtime type mapping, but not defined.
	u.defineFunctionsInOrder(u.undefinedFuncs)

	// Set initializers for globals.
	for global, init := range u.globalInits {
		initval := init.build(global.Type().ElementType())
		global.SetInitializer(initval)
	}
}

func (u *unit) addGlobal(global llvm.Value, ty types.Type) {
	u.globalInits[global] = new(globalInit)

	if hasPointers(ty) {
		global = llvm.ConstBitCast(global, llvm.PointerType(llvm.Int8Type(), 0))
		size := llvm.ConstInt(u.types.inttype, uint64(u.types.Sizeof(ty)), false)
		root := llvm.ConstStruct([]llvm.Value{global, size}, false)
		u.gcRoots = append(u.gcRoots, root)
	}
}

// ResolveMethod implements MethodResolver.ResolveMethod.
func (u *unit) ResolveMethod(s *types.Selection) *govalue {
	m := u.pkg.Prog.Method(s)
	llfn := u.resolveFunctionGlobal(m)
	llfn = llvm.ConstBitCast(llfn, llvm.PointerType(llvm.Int8Type(), 0))
	return newValue(llfn, m.Signature)
}

// resolveFunctionDescriptorGlobal returns a reference to the LLVM global
// storing the function's descriptor.
func (u *unit) resolveFunctionDescriptorGlobal(f *ssa.Function) llvm.Value {
	llfd, ok := u.funcDescriptors[f]
	if !ok {
		name := u.types.mc.mangleFunctionName(f) + "$descriptor"
		llfd = llvm.AddGlobal(u.module.Module, llvm.PointerType(llvm.Int8Type(), 0), name)
		llfd.SetGlobalConstant(true)
		u.funcDescriptors[f] = llfd
	}
	return llfd
}

// resolveFunctionDescriptor returns a function's
// first-class value representation.
func (u *unit) resolveFunctionDescriptor(f *ssa.Function) *govalue {
	llfd := u.resolveFunctionDescriptorGlobal(f)
	llfd = llvm.ConstBitCast(llfd, llvm.PointerType(llvm.Int8Type(), 0))
	return newValue(llfd, f.Signature)
}

// resolveFunctionGlobal returns an llvm.Value for a function global.
func (u *unit) resolveFunctionGlobal(f *ssa.Function) llvm.Value {
	if v, ok := u.globals[f]; ok {
		return v
	}
	name := u.types.mc.mangleFunctionName(f)
	// It's possible that the function already exists in the module;
	// for example, if it's a runtime intrinsic that the compiler
	// has already referenced.
	llvmFunction := u.module.Module.NamedFunction(name)
	if llvmFunction.IsNil() {
		fti := u.llvmtypes.getSignatureInfo(f.Signature)
		llvmFunction = fti.declare(u.module.Module, name)
		u.undefinedFuncs[f] = true
	}
	u.globals[f] = llvmFunction
	return llvmFunction
}

func (u *unit) getFunctionLinkage(f *ssa.Function) llvm.Linkage {
	switch {
	case f.Pkg == nil:
		// Synthetic functions outside packages may appear in multiple packages.
		return llvm.LinkOnceODRLinkage

	case f.Parent() != nil:
		// Anonymous.
		return llvm.InternalLinkage

	case f.Signature.Recv() == nil && !ast.IsExported(f.Name()) &&
		!(f.Name() == "main" && f.Pkg.Object.Path() == "main") &&
		f.Name() != "init":
		// Unexported methods may be referenced as part of an interface method
		// table in another package. TODO(pcc): detect when this cannot happen.
		return llvm.InternalLinkage

	default:
		return llvm.ExternalLinkage
	}
}

func (u *unit) defineFunction(f *ssa.Function) {
	// Only define functions from this package, or synthetic
	// wrappers (which do not have a package).
	if f.Pkg != nil && f.Pkg != u.pkg {
		return
	}

	llfn := u.resolveFunctionGlobal(f)
	linkage := u.getFunctionLinkage(f)

	isMethod := f.Signature.Recv() != nil

	// Methods cannot be referred to via a descriptor.
	if !isMethod {
		llfd := u.resolveFunctionDescriptorGlobal(f)
		llfd.SetInitializer(llvm.ConstBitCast(llfn, llvm.PointerType(llvm.Int8Type(), 0)))
		llfd.SetLinkage(linkage)
	}

	// We only need to emit a descriptor for functions without bodies.
	if len(f.Blocks) == 0 {
		return
	}

	ssaopt.LowerAllocsToStack(f)

	if u.DumpSSA {
		f.WriteTo(os.Stderr)
	}

	fr := newFrame(u, llfn)
	defer fr.dispose()
	fr.addCommonFunctionAttrs(fr.function)
	fr.function.SetLinkage(linkage)

	fr.logf("Define function: %s @ %s", f.String(), fr.pkg.Prog.Fset.Position(f.Pos()))
	fti := u.llvmtypes.getSignatureInfo(f.Signature)
	delete(u.undefinedFuncs, f)
	fr.retInf = fti.retInf

	// Push the compile unit and function onto the debug context.
	if u.GenerateDebug {
		u.debug.PushFunction(fr.function, f.Signature, f.Pos())
		defer u.debug.PopFunction()
		u.debug.SetLocation(fr.builder, f.Pos())
	}

	// If a function calls recover, we create a separate function to
	// hold the real function, and this function calls __go_can_recover
	// and bridges to it.
	if callsRecover(f) {
		fr = fr.bridgeRecoverFunc(fr.function, fti)
	}

	fr.blocks = make([]llvm.BasicBlock, len(f.Blocks))
	fr.lastBlocks = make([]llvm.BasicBlock, len(f.Blocks))
	for i, block := range f.Blocks {
		fr.blocks[i] = llvm.AddBasicBlock(fr.function, fmt.Sprintf(".%d.%s", i, block.Comment))
	}
	fr.builder.SetInsertPointAtEnd(fr.blocks[0])
	fr.transformSwitches(f)

	prologueBlock := llvm.InsertBasicBlock(fr.blocks[0], "prologue")
	fr.builder.SetInsertPointAtEnd(prologueBlock)

	for i, param := range f.Params {
		llparam := fti.argInfos[i].decode(llvm.GlobalContext(), fr.builder, fr.builder)
		if isMethod && i == 0 {
			if _, ok := param.Type().Underlying().(*types.Pointer); !ok {
				llparam = fr.builder.CreateBitCast(llparam, llvm.PointerType(fr.types.ToLLVM(param.Type()), 0), "")
				llparam = fr.builder.CreateLoad(llparam, "")
			}
		}
		fr.env[param] = newValue(llparam, param.Type())
	}

	// Load closure, extract free vars.
	if len(f.FreeVars) > 0 {
		for _, fv := range f.FreeVars {
			fr.env[fv] = newValue(llvm.ConstNull(u.llvmtypes.ToLLVM(fv.Type())), fv.Type())
		}
		elemTypes := make([]llvm.Type, len(f.FreeVars)+1)
		elemTypes[0] = llvm.PointerType(llvm.Int8Type(), 0) // function pointer
		for i, fv := range f.FreeVars {
			elemTypes[i+1] = u.llvmtypes.ToLLVM(fv.Type())
		}
		structType := llvm.StructType(elemTypes, false)
		closure := fr.function.Param(fti.chainIndex)
		closure = fr.builder.CreateBitCast(closure, llvm.PointerType(structType, 0), "")
		for i, fv := range f.FreeVars {
			ptr := fr.builder.CreateStructGEP(closure, i+1, "")
			ptr = fr.builder.CreateLoad(ptr, "")
			fr.env[fv] = newValue(ptr, fv.Type())
		}
	}

	// Allocate stack space for locals in the prologue block.
	for _, local := range f.Locals {
		typ := fr.llvmtypes.ToLLVM(deref(local.Type()))
		alloca := fr.builder.CreateAlloca(typ, local.Comment)
		fr.memsetZero(alloca, llvm.SizeOf(typ))
		bcalloca := fr.builder.CreateBitCast(alloca, llvm.PointerType(llvm.Int8Type(), 0), "")
		value := newValue(bcalloca, local.Type())
		fr.env[local] = value
	}

	// If the function contains any defers, we must first create
	// an unwind block. We can short-circuit the check for defers with
	// f.Recover != nil.
	if f.Recover != nil || hasDefer(f) {
		fr.unwindBlock = llvm.AddBasicBlock(fr.function, "unwind")
		fr.frameptr = fr.builder.CreateAlloca(llvm.Int8Type(), "")
	}

	// Keep track of the block into which we need to insert the call
	// to __go_register_gc_roots. This needs to be inserted after the
	// init guard check under the llgo ABI.
	var registerGcBlock llvm.BasicBlock

	// If this is the "init" function, emit the init guard check and
	// enable init-specific optimizations.
	if !isMethod && f.Name() == "init" {
		registerGcBlock = fr.emitInitPrologue()
		fr.isInit = true
	}

	fr.builder.CreateBr(fr.blocks[0])
	fr.allocaBuilder.SetInsertPointBefore(prologueBlock.FirstInstruction())

	for _, block := range f.DomPreorder() {
		llblock := fr.blocks[block.Index]
		if llblock.IsNil() {
			continue
		}
		fr.translateBlock(block, llblock)
	}

	fr.fixupPhis()

	if !fr.unwindBlock.IsNil() {
		fr.setupUnwindBlock(f.Recover)
	}

	// The init function needs to register the GC roots first. We do this
	// after generating code for it because allocations may have caused
	// additional GC roots to be created.
	if fr.isInit {
		fr.builder.SetInsertPointBefore(registerGcBlock.FirstInstruction())
		fr.registerGcRoots()
	}
}

type pendingPhi struct {
	ssa  *ssa.Phi
	llvm llvm.Value
}

type frame struct {
	*unit
	function               llvm.Value
	builder, allocaBuilder llvm.Builder
	retInf                 retInfo
	blocks                 []llvm.BasicBlock
	lastBlocks             []llvm.BasicBlock
	runtimeErrorBlocks     [gccgoRuntimeErrorCount]llvm.BasicBlock
	unwindBlock            llvm.BasicBlock
	frameptr               llvm.Value
	env                    map[ssa.Value]*govalue
	ptr                    map[ssa.Value]llvm.Value
	tuples                 map[ssa.Value][]*govalue
	phis                   []pendingPhi
	canRecover             llvm.Value
	isInit                 bool
}

func newFrame(u *unit, fn llvm.Value) *frame {
	return &frame{
		unit:          u,
		function:      fn,
		builder:       llvm.GlobalContext().NewBuilder(),
		allocaBuilder: llvm.GlobalContext().NewBuilder(),
		env:           make(map[ssa.Value]*govalue),
		ptr:           make(map[ssa.Value]llvm.Value),
		tuples:        make(map[ssa.Value][]*govalue),
	}
}

func (fr *frame) dispose() {
	fr.builder.Dispose()
	fr.allocaBuilder.Dispose()
}

// emitInitPrologue emits the init-specific function prologue (guard check and
// initialization of dependent packages under the llgo native ABI), and returns
// the basic block into which the GC registration call should be emitted.
func (fr *frame) emitInitPrologue() llvm.BasicBlock {
	if fr.GccgoABI {
		return fr.builder.GetInsertBlock()
	}

	initGuard := llvm.AddGlobal(fr.module.Module, llvm.Int1Type(), "init$guard")
	initGuard.SetLinkage(llvm.InternalLinkage)
	initGuard.SetInitializer(llvm.ConstNull(llvm.Int1Type()))

	returnBlock := llvm.AddBasicBlock(fr.function, "")
	initBlock := llvm.AddBasicBlock(fr.function, "")

	initGuardVal := fr.builder.CreateLoad(initGuard, "")
	fr.builder.CreateCondBr(initGuardVal, returnBlock, initBlock)

	fr.builder.SetInsertPointAtEnd(returnBlock)
	fr.builder.CreateRetVoid()

	fr.builder.SetInsertPointAtEnd(initBlock)
	fr.builder.CreateStore(llvm.ConstInt(llvm.Int1Type(), 1, false), initGuard)
	int8ptr := llvm.PointerType(fr.types.ctx.Int8Type(), 0)
	ftyp := llvm.FunctionType(llvm.VoidType(), []llvm.Type{int8ptr}, false)
	for _, pkg := range fr.pkg.Object.Imports() {
		initname := ManglePackagePath(pkg.Path()) + "..import"
		initfn := fr.module.Module.NamedFunction(initname)
		if initfn.IsNil() {
			initfn = llvm.AddFunction(fr.module.Module, initname, ftyp)
		}
		args := []llvm.Value{llvm.Undef(int8ptr)}
		fr.builder.CreateCall(initfn, args, "")
	}

	return initBlock
}

// bridgeRecoverFunc creates a function that may call recover(), and creates
// a call to it from the current frame. The created function will be called
// with a boolean parameter that indicates whether it may call recover().
//
// The created function will have the same name as the current frame's function
// with "$recover" appended, having the same return types and parameters with
// an additional boolean parameter appended.
//
// A new frame will be returned for the newly created function.
func (fr *frame) bridgeRecoverFunc(llfn llvm.Value, fti functionTypeInfo) *frame {
	// The bridging function must not be inlined, or the return address
	// may not correspond to the source function.
	attrKind := llvm.AttributeKindID("noinline")
	noInlineAttr := fr.module.Context().CreateEnumAttribute(attrKind, 0)
	llfn.AddFunctionAttr(noInlineAttr)

	// Call __go_can_recover, passing in the function's return address.
	entry := llvm.AddBasicBlock(llfn, "entry")
	fr.builder.SetInsertPointAtEnd(entry)
	canRecover := fr.runtime.canRecover.call(fr, fr.returnAddress(0))[0]
	returnType := fti.functionType.ReturnType()
	argTypes := fti.functionType.ParamTypes()
	argTypes = append(argTypes, canRecover.Type())

	// Create and call the $recover function.
	ftiRecover := fti
	ftiRecover.functionType = llvm.FunctionType(returnType, argTypes, false)
	llfnRecover := ftiRecover.declare(fr.module.Module, llfn.Name()+"$recover")
	fr.addCommonFunctionAttrs(llfnRecover)
	llfnRecover.SetLinkage(llvm.InternalLinkage)
	args := make([]llvm.Value, len(argTypes)-1, len(argTypes))
	for i := range args {
		args[i] = llfn.Param(i)
	}
	args = append(args, canRecover)
	result := fr.builder.CreateCall(llfnRecover, args, "")
	if returnType.TypeKind() == llvm.VoidTypeKind {
		fr.builder.CreateRetVoid()
	} else {
		fr.builder.CreateRet(result)
	}

	// The $recover function must condition calls to __go_recover on
	// the result of __go_can_recover passed in as an argument.
	fr = newFrame(fr.unit, llfnRecover)
	fr.retInf = ftiRecover.retInf
	fr.canRecover = fr.function.Param(len(argTypes) - 1)
	return fr
}

func (fr *frame) registerGcRoots() {
	if len(fr.gcRoots) != 0 {
		rootty := fr.gcRoots[0].Type()
		roots := append(fr.gcRoots, llvm.ConstNull(rootty))
		rootsarr := llvm.ConstArray(rootty, roots)
		rootsstruct := llvm.ConstStruct([]llvm.Value{llvm.ConstNull(llvm.PointerType(llvm.Int8Type(), 0)), rootsarr}, false)

		rootsglobal := llvm.AddGlobal(fr.module.Module, rootsstruct.Type(), "")
		rootsglobal.SetInitializer(rootsstruct)
		rootsglobal.SetLinkage(llvm.InternalLinkage)
		fr.runtime.registerGcRoots.callOnly(fr, llvm.ConstBitCast(rootsglobal, llvm.PointerType(llvm.Int8Type(), 0)))
	}
}

func (fr *frame) fixupPhis() {
	for _, phi := range fr.phis {
		values := make([]llvm.Value, len(phi.ssa.Edges))
		blocks := make([]llvm.BasicBlock, len(phi.ssa.Edges))
		block := phi.ssa.Block()
		for i, edge := range phi.ssa.Edges {
			values[i] = fr.llvmvalue(edge)
			blocks[i] = fr.lastBlock(block.Preds[i])
		}
		phi.llvm.AddIncoming(values, blocks)
	}
}

func (fr *frame) createLandingPad(cleanup bool) llvm.Value {
	fr.function.SetPersonality(fr.runtime.gccgoPersonality)
	lp := fr.builder.CreateLandingPad(fr.runtime.gccgoExceptionType, 0, "")
	if cleanup {
		lp.SetCleanup(true)
	} else {
		lp.AddClause(llvm.ConstNull(llvm.PointerType(llvm.Int8Type(), 0)))
	}
	return lp
}

// Runs defers. If a defer panics, check for recovers in later defers.
func (fr *frame) runDefers() {
	loopbb := llvm.AddBasicBlock(fr.function, "")
	fr.builder.CreateBr(loopbb)

	retrylpad := llvm.AddBasicBlock(fr.function, "")
	fr.builder.SetInsertPointAtEnd(retrylpad)
	fr.createLandingPad(false)
	fr.runtime.checkDefer.callOnly(fr, fr.frameptr)
	fr.builder.CreateBr(loopbb)

	fr.builder.SetInsertPointAtEnd(loopbb)
	fr.runtime.undefer.invoke(fr, retrylpad, fr.frameptr)
}

func (fr *frame) setupUnwindBlock(rec *ssa.BasicBlock) {
	var recoverbb llvm.BasicBlock
	if rec != nil {
		recoverbb = fr.blocks[rec.Index]
	} else {
		recoverbb = llvm.AddBasicBlock(fr.function, "recover")
		fr.builder.SetInsertPointAtEnd(recoverbb)
		fr.builder.CreateUnreachable()
	}

	checkunwindbb := llvm.AddBasicBlock(fr.function, "")
	fr.builder.SetInsertPointAtEnd(checkunwindbb)
	exc := fr.createLandingPad(true)
	fr.runDefers()

	frame := fr.builder.CreateLoad(fr.frameptr, "")
	shouldresume := fr.builder.CreateIsNull(frame, "")

	resumebb := llvm.AddBasicBlock(fr.function, "")
	fr.builder.CreateCondBr(shouldresume, resumebb, recoverbb)

	fr.builder.SetInsertPointAtEnd(resumebb)
	fr.builder.CreateResume(exc)

	fr.builder.SetInsertPointAtEnd(fr.unwindBlock)
	fr.createLandingPad(false)
	fr.runtime.checkDefer.invoke(fr, checkunwindbb, fr.frameptr)
	fr.runDefers()
	fr.builder.CreateBr(recoverbb)
}

func (fr *frame) translateBlock(b *ssa.BasicBlock, llb llvm.BasicBlock) {
	fr.builder.SetInsertPointAtEnd(llb)
	for _, instr := range b.Instrs {
		fr.instruction(instr)
	}
	fr.lastBlocks[b.Index] = fr.builder.GetInsertBlock()
}

func (fr *frame) block(b *ssa.BasicBlock) llvm.BasicBlock {
	return fr.blocks[b.Index]
}

func (fr *frame) lastBlock(b *ssa.BasicBlock) llvm.BasicBlock {
	return fr.lastBlocks[b.Index]
}

func (fr *frame) value(v ssa.Value) (result *govalue) {
	switch v := v.(type) {
	case nil:
		return nil
	case *ssa.Function:
		return fr.resolveFunctionDescriptor(v)
	case *ssa.Const:
		return fr.newValueFromConst(v.Value, v.Type())
	case *ssa.Global:
		if g, ok := fr.globals[v]; ok {
			return newValue(g, v.Type())
		}
		// Create an external global. Globals for this package are defined
		// on entry to translatePackage, and have initialisers.
		llelemtyp := fr.llvmtypes.ToLLVM(deref(v.Type()))
		vname := fr.types.mc.mangleGlobalName(v)
		llglobal := llvm.AddGlobal(fr.module.Module, llelemtyp, vname)
		llglobal = llvm.ConstBitCast(llglobal, fr.llvmtypes.ToLLVM(v.Type()))
		fr.globals[v] = llglobal
		return newValue(llglobal, v.Type())
	}
	if value, ok := fr.env[v]; ok {
		return value
	}

	panic(fmt.Errorf("Instruction %q not visited yet", v.Name()))
}

func (fr *frame) llvmvalue(v ssa.Value) llvm.Value {
	if gv := fr.value(v); gv != nil {
		return gv.value
	} else {
		return llvm.Value{nil}
	}
}

func (fr *frame) isNonNull(v ssa.Value) bool {
	switch v.(type) {
	case
		// Globals have a fixed (non-nil) address.
		*ssa.Global,
		// The language does not specify what happens if an allocation fails.
		*ssa.Alloc,
		// These have already been nil checked.
		*ssa.FieldAddr, *ssa.IndexAddr:
		return true
	default:
		return false
	}
}

func (fr *frame) nilCheck(v ssa.Value, llptr llvm.Value) {
	if !fr.isNonNull(v) {
		ptrnull := fr.builder.CreateIsNull(llptr, "")
		fr.condBrRuntimeError(ptrnull, gccgoRuntimeErrorNIL_DEREFERENCE)
	}
}

func (fr *frame) canAvoidElementLoad(ptr ssa.Value) bool {
	for _, ref := range *ptr.Referrers() {
		switch ref := ref.(type) {
		case *ssa.Field:
		case *ssa.Index:
			if ref.X != ptr {
				return false
			}
			// ok
		default:
			return false
		}
	}

	return true
}

// If this value is sufficiently large, look through referrers to see if we can
// avoid a load.
func (fr *frame) canAvoidLoad(instr *ssa.UnOp, op llvm.Value) bool {
	if fr.types.Sizeof(instr.Type()) < 2*fr.types.Sizeof(types.Typ[types.Int]) {
		// Don't bother with small values.
		return false
	}

	// Keep track of whether our pointer may escape. We conservatively assume
	// that MakeInterfaces will escape.
	esc := false

	// We only know how to avoid loads if they are used to create an interface
	// or read an element of the structure. If we see any other referrer, abort.
	for _, ref := range *instr.Referrers() {
		switch ref := ref.(type) {
		case *ssa.MakeInterface:
			esc = true
		case *ssa.Field:
		case *ssa.Index:
			if ref.X != instr {
				// This should never happen, as indices are always of type int
				// and we don't bother with values smaller than 2*sizeof(int).
				panic("impossible")
			}
			// ok
		default:
			return false
		}
	}

	var opcopy llvm.Value
	if esc {
		opcopy = fr.createTypeMalloc(instr.Type())
	} else {
		opcopy = fr.allocaBuilder.CreateAlloca(fr.types.ToLLVM(instr.Type()), "")
	}
	fr.memcpy(opcopy, op, llvm.ConstInt(fr.types.inttype, uint64(fr.types.Sizeof(instr.Type())), false))

	fr.ptr[instr] = opcopy
	return true
}

// Return true iff we think it might be beneficial to turn this alloc instruction
// into a statically allocated global.
// Precondition: we are compiling the init function.
func (fr *frame) shouldStaticallyAllocate(alloc *ssa.Alloc) bool {
	// First, see if the allocated type is an array or struct, and if so determine
	// the number of elements in the type. If the type is anything else, we
	// statically allocate unconditionally.
	var numElems int64
	switch ty := deref(alloc.Type()).Underlying().(type) {
	case *types.Array:
		numElems = ty.Len()
	case *types.Struct:
		numElems = int64(ty.NumFields())
	default:
		return true
	}

	// We treat the number of referrers to the alloc instruction as a rough
	// proxy for the number of elements initialized. If the data structure
	// is densely initialized (> 1/4 elements initialized), enable the
	// optimization.
	return int64(len(*alloc.Referrers()))*4 > numElems
}

// If val is a constant and addr refers to a global variable which is defined in
// this module or an element thereof, simulate the effect of storing val at addr
// in the global variable's initializer and return true, otherwise return false.
// Precondition: we are compiling the init function.
func (fr *frame) maybeStoreInInitializer(val, addr llvm.Value) bool {
	if val.IsAConstant().IsNil() {
		return false
	}

	if !addr.IsAConstantExpr().IsNil() && addr.OperandsCount() >= 2 &&
		// TODO(pcc): Explicitly check that this is a constant GEP.
		// I don't think there are any other kinds of constantexpr which
		// satisfy the conditions we test for here, so this is probably safe.
		!addr.Operand(0).IsAGlobalVariable().IsNil() &&
		addr.Operand(1).IsNull() {
		gv := addr.Operand(0)
		globalInit, ok := fr.globalInits[gv]
		if !ok {
			return false
		}
		indices := make([]uint32, addr.OperandsCount()-2)
		for i := range indices {
			op := addr.Operand(i + 2)
			if op.IsAConstantInt().IsNil() {
				return false
			}
			indices[i] = uint32(op.ZExtValue())
		}
		globalInit.update(gv.Type().ElementType(), indices, val)
		return true
	} else if !addr.IsAGlobalVariable().IsNil() {
		if globalInit, ok := fr.globalInits[addr]; ok {
			globalInit.update(addr.Type().ElementType(), nil, val)
			return true
		}
		return false
	} else {
		return false
	}
}

func (fr *frame) instruction(instr ssa.Instruction) {
	fr.logf("[%T] %v @ %s\n", instr, instr, fr.pkg.Prog.Fset.Position(instr.Pos()))
	if fr.GenerateDebug {
		fr.debug.SetLocation(fr.builder, instr.Pos())
	}

	switch instr := instr.(type) {
	case *ssa.Alloc:
		typ := deref(instr.Type())
		llvmtyp := fr.llvmtypes.ToLLVM(typ)
		var value llvm.Value
		if !instr.Heap {
			value = fr.env[instr].value
			fr.memsetZero(value, llvm.SizeOf(llvmtyp))
		} else if fr.isInit && fr.shouldStaticallyAllocate(instr) {
			// If this is the init function and we think it may be beneficial,
			// allocate memory statically in the object file rather than on the
			// heap. This allows us to optimize constant stores into such
			// variables as static initializations.
			global := llvm.AddGlobal(fr.module.Module, llvmtyp, "")
			global.SetLinkage(llvm.InternalLinkage)
			fr.addGlobal(global, typ)
			ptr := llvm.ConstBitCast(global, llvm.PointerType(llvm.Int8Type(), 0))
			fr.env[instr] = newValue(ptr, instr.Type())
		} else {
			value = fr.createTypeMalloc(typ)
			value.SetName(instr.Comment)
			value = fr.builder.CreateBitCast(value, llvm.PointerType(llvm.Int8Type(), 0), "")
			fr.env[instr] = newValue(value, instr.Type())
		}

	case *ssa.BinOp:
		lhs, rhs := fr.value(instr.X), fr.value(instr.Y)
		fr.env[instr] = fr.binaryOp(lhs, instr.Op, rhs)

	case *ssa.Call:
		tuple := fr.callInstruction(instr)
		if len(tuple) == 1 {
			fr.env[instr] = tuple[0]
		} else {
			fr.tuples[instr] = tuple
		}

	case *ssa.ChangeInterface:
		x := fr.value(instr.X)
		// The source type must be a non-empty interface,
		// as ChangeInterface cannot fail (E2I may fail).
		if instr.Type().Underlying().(*types.Interface).NumMethods() > 0 {
			x = fr.changeInterface(x, instr.Type(), false)
		} else {
			x = fr.convertI2E(x)
		}
		fr.env[instr] = x

	case *ssa.ChangeType:
		value := fr.llvmvalue(instr.X)
		if _, ok := instr.Type().Underlying().(*types.Pointer); ok {
			value = fr.builder.CreateBitCast(value, fr.llvmtypes.ToLLVM(instr.Type()), "")
		}
		fr.env[instr] = newValue(value, instr.Type())

	case *ssa.Convert:
		v := fr.value(instr.X)
		fr.env[instr] = fr.convert(v, instr.Type())

	case *ssa.Defer:
		fn, arg := fr.createThunk(instr)
		fr.runtime.Defer.call(fr, fr.frameptr, fn, arg)

	case *ssa.Extract:
		var elem llvm.Value
		if t, ok := fr.tuples[instr.Tuple]; ok {
			elem = t[instr.Index].value
		} else {
			tuple := fr.llvmvalue(instr.Tuple)
			elem = fr.builder.CreateExtractValue(tuple, instr.Index, instr.Name())
		}
		elemtyp := instr.Type()
		fr.env[instr] = newValue(elem, elemtyp)

	case *ssa.Field:
		fieldtyp := instr.Type()
		if p, ok := fr.ptr[instr.X]; ok {
			field := fr.builder.CreateStructGEP(p, instr.Field, instr.Name())
			if fr.canAvoidElementLoad(instr) {
				fr.ptr[instr] = field
			} else {
				fr.env[instr] = newValue(fr.builder.CreateLoad(field, ""), fieldtyp)
			}
		} else {
			value := fr.llvmvalue(instr.X)
			field := fr.builder.CreateExtractValue(value, instr.Field, instr.Name())
			fr.env[instr] = newValue(field, fieldtyp)
		}

	case *ssa.FieldAddr:
		ptr := fr.llvmvalue(instr.X)
		fr.nilCheck(instr.X, ptr)
		xtyp := instr.X.Type().Underlying().(*types.Pointer).Elem()
		ptrtyp := llvm.PointerType(fr.llvmtypes.ToLLVM(xtyp), 0)
		ptr = fr.builder.CreateBitCast(ptr, ptrtyp, "")
		fieldptr := fr.builder.CreateStructGEP(ptr, instr.Field, instr.Name())
		fieldptr = fr.builder.CreateBitCast(fieldptr, llvm.PointerType(llvm.Int8Type(), 0), "")
		fieldptrtyp := instr.Type()
		fr.env[instr] = newValue(fieldptr, fieldptrtyp)

	case *ssa.Go:
		fn, arg := fr.createThunk(instr)
		fr.runtime.Go.call(fr, fn, arg)

	case *ssa.If:
		cond := fr.llvmvalue(instr.Cond)
		block := instr.Block()
		trueBlock := fr.block(block.Succs[0])
		falseBlock := fr.block(block.Succs[1])
		cond = fr.builder.CreateTrunc(cond, llvm.Int1Type(), "")
		fr.builder.CreateCondBr(cond, trueBlock, falseBlock)

	case *ssa.Index:
		var arrayptr llvm.Value

		if ptr, ok := fr.ptr[instr.X]; ok {
			arrayptr = ptr
		} else {
			array := fr.llvmvalue(instr.X)
			arrayptr = fr.allocaBuilder.CreateAlloca(array.Type(), "")

			fr.builder.CreateStore(array, arrayptr)
		}
		index := fr.llvmvalue(instr.Index)

		arraytyp := instr.X.Type().Underlying().(*types.Array)
		arraylen := llvm.ConstInt(fr.llvmtypes.inttype, uint64(arraytyp.Len()), false)

		// The index may not have been promoted to int (for example, if it
		// came from a composite literal).
		index = fr.createZExtOrTrunc(index, fr.types.inttype, "")

		// Bounds checking: 0 <= index < len
		zero := llvm.ConstNull(fr.types.inttype)
		i0 := fr.builder.CreateICmp(llvm.IntSLT, index, zero, "")
		li := fr.builder.CreateICmp(llvm.IntSLE, arraylen, index, "")

		cond := fr.builder.CreateOr(i0, li, "")

		fr.condBrRuntimeError(cond, gccgoRuntimeErrorARRAY_INDEX_OUT_OF_BOUNDS)

		addr := fr.builder.CreateGEP(arrayptr, []llvm.Value{zero, index}, "")
		if fr.canAvoidElementLoad(instr) {
			fr.ptr[instr] = addr
		} else {
			fr.env[instr] = newValue(fr.builder.CreateLoad(addr, ""), instr.Type())
		}

	case *ssa.IndexAddr:
		x := fr.llvmvalue(instr.X)
		index := fr.llvmvalue(instr.Index)
		var arrayptr, arraylen llvm.Value
		var elemtyp types.Type
		var errcode uint64
		switch typ := instr.X.Type().Underlying().(type) {
		case *types.Slice:
			elemtyp = typ.Elem()
			arrayptr = fr.builder.CreateExtractValue(x, 0, "")
			arraylen = fr.builder.CreateExtractValue(x, 1, "")
			errcode = gccgoRuntimeErrorSLICE_INDEX_OUT_OF_BOUNDS
		case *types.Pointer: // *array
			arraytyp := typ.Elem().Underlying().(*types.Array)
			elemtyp = arraytyp.Elem()
			fr.nilCheck(instr.X, x)
			arrayptr = x
			arraylen = llvm.ConstInt(fr.llvmtypes.inttype, uint64(arraytyp.Len()), false)
			errcode = gccgoRuntimeErrorARRAY_INDEX_OUT_OF_BOUNDS
		}

		// The index may not have been promoted to int (for example, if it
		// came from a composite literal).
		index = fr.createZExtOrTrunc(index, fr.types.inttype, "")

		// Bounds checking: 0 <= index < len
		zero := llvm.ConstNull(fr.types.inttype)
		i0 := fr.builder.CreateICmp(llvm.IntSLT, index, zero, "")
		li := fr.builder.CreateICmp(llvm.IntSLE, arraylen, index, "")

		cond := fr.builder.CreateOr(i0, li, "")

		fr.condBrRuntimeError(cond, errcode)

		ptrtyp := llvm.PointerType(fr.llvmtypes.ToLLVM(elemtyp), 0)
		arrayptr = fr.builder.CreateBitCast(arrayptr, ptrtyp, "")
		addr := fr.builder.CreateGEP(arrayptr, []llvm.Value{index}, "")
		addr = fr.builder.CreateBitCast(addr, llvm.PointerType(llvm.Int8Type(), 0), "")
		fr.env[instr] = newValue(addr, types.NewPointer(elemtyp))

	case *ssa.Jump:
		succ := instr.Block().Succs[0]
		fr.builder.CreateBr(fr.block(succ))

	case *ssa.Lookup:
		x := fr.value(instr.X)
		index := fr.value(instr.Index)
		if isString(x.Type().Underlying()) {
			fr.env[instr] = fr.stringIndex(x, index)
		} else {
			v, ok := fr.mapLookup(x, index)
			if instr.CommaOk {
				fr.tuples[instr] = []*govalue{v, ok}
			} else {
				fr.env[instr] = v
			}
		}

	case *ssa.MakeChan:
		fr.env[instr] = fr.makeChan(instr.Type(), fr.value(instr.Size))

	case *ssa.MakeClosure:
		llfn := fr.resolveFunctionGlobal(instr.Fn.(*ssa.Function))
		llfn = llvm.ConstBitCast(llfn, llvm.PointerType(llvm.Int8Type(), 0))
		fn := newValue(llfn, instr.Fn.(*ssa.Function).Signature)
		bindings := make([]*govalue, len(instr.Bindings))
		for i, binding := range instr.Bindings {
			bindings[i] = fr.value(binding)
		}
		fr.env[instr] = fr.makeClosure(fn, bindings)

	case *ssa.MakeInterface:
		// fr.ptr[instr.X] will be set if a pointer load was elided by canAvoidLoad
		if ptr, ok := fr.ptr[instr.X]; ok {
			fr.env[instr] = fr.makeInterfaceFromPointer(ptr, instr.X.Type(), instr.Type())
		} else {
			receiver := fr.llvmvalue(instr.X)
			fr.env[instr] = fr.makeInterface(receiver, instr.X.Type(), instr.Type())
		}

	case *ssa.MakeMap:
		fr.env[instr] = fr.makeMap(instr.Type(), fr.value(instr.Reserve))

	case *ssa.MakeSlice:
		length := fr.value(instr.Len)
		capacity := fr.value(instr.Cap)
		fr.env[instr] = fr.makeSlice(instr.Type(), length, capacity)

	case *ssa.MapUpdate:
		m := fr.value(instr.Map)
		k := fr.value(instr.Key)
		v := fr.value(instr.Value)
		fr.mapUpdate(m, k, v)

	case *ssa.Next:
		iter := fr.tuples[instr.Iter]
		if instr.IsString {
			fr.tuples[instr] = fr.stringIterNext(iter)
		} else {
			fr.tuples[instr] = fr.mapIterNext(iter)
		}

	case *ssa.Panic:
		arg := fr.value(instr.X)
		fr.callPanic(arg, true)

	case *ssa.Phi:
		typ := instr.Type()
		phi := fr.builder.CreatePHI(fr.llvmtypes.ToLLVM(typ), instr.Comment)
		fr.env[instr] = newValue(phi, typ)
		fr.phis = append(fr.phis, pendingPhi{instr, phi})

	case *ssa.Range:
		x := fr.value(instr.X)
		switch x.Type().Underlying().(type) {
		case *types.Map:
			fr.tuples[instr] = fr.mapIterInit(x)
		case *types.Basic: // string
			fr.tuples[instr] = fr.stringIterInit(x)
		default:
			panic(fmt.Sprintf("unhandled range for type %T", x.Type()))
		}

	case *ssa.Return:
		vals := make([]llvm.Value, len(instr.Results))
		for i, res := range instr.Results {
			vals[i] = fr.llvmvalue(res)
		}
		fr.retInf.encode(llvm.GlobalContext(), fr.allocaBuilder, fr.builder, vals)

	case *ssa.RunDefers:
		fr.runDefers()

	case *ssa.Select:
		index, recvOk, recvElems := fr.chanSelect(instr)
		tuple := append([]*govalue{index, recvOk}, recvElems...)
		fr.tuples[instr] = tuple

	case *ssa.Send:
		fr.chanSend(fr.value(instr.Chan), fr.value(instr.X))

	case *ssa.Slice:
		x := fr.llvmvalue(instr.X)
		low := fr.llvmvalue(instr.Low)
		high := fr.llvmvalue(instr.High)
		max := fr.llvmvalue(instr.Max)
		slice := fr.slice(x, instr.X.Type(), low, high, max)
		fr.env[instr] = newValue(slice, instr.Type())

	case *ssa.Store:
		addr := fr.llvmvalue(instr.Addr)
		value := fr.llvmvalue(instr.Val)
		addr = fr.builder.CreateBitCast(addr, llvm.PointerType(value.Type(), 0), "")
		// If this is the init function, see if we can simulate the effect
		// of the store in a global's initializer, in which case we can avoid
		// generating code for it.
		if !fr.isInit || !fr.maybeStoreInInitializer(value, addr) {
			fr.nilCheck(instr.Addr, addr)
			fr.builder.CreateStore(value, addr)
		}

	case *switchInstr:
		fr.emitSwitch(instr)

	case *ssa.TypeAssert:
		x := fr.value(instr.X)
		if instr.CommaOk {
			v, ok := fr.interfaceTypeCheck(x, instr.AssertedType)
			fr.tuples[instr] = []*govalue{v, ok}
		} else {
			fr.env[instr] = fr.interfaceTypeAssert(x, instr.AssertedType)
		}

	case *ssa.UnOp:
		operand := fr.value(instr.X)
		switch instr.Op {
		case token.ARROW:
			x, ok := fr.chanRecv(operand, instr.CommaOk)
			if instr.CommaOk {
				fr.tuples[instr] = []*govalue{x, ok}
			} else {
				fr.env[instr] = x
			}
		case token.MUL:
			fr.nilCheck(instr.X, operand.value)
			if !fr.canAvoidLoad(instr, operand.value) {
				// The bitcast is necessary to handle recursive pointer loads.
				llptr := fr.builder.CreateBitCast(operand.value, llvm.PointerType(fr.llvmtypes.ToLLVM(instr.Type()), 0), "")
				fr.env[instr] = newValue(fr.builder.CreateLoad(llptr, ""), instr.Type())
			}
		default:
			fr.env[instr] = fr.unaryOp(operand, instr.Op)
		}

	default:
		panic(fmt.Sprintf("unhandled: %v", instr))
	}
}

func (fr *frame) callBuiltin(typ types.Type, builtin *ssa.Builtin, args []ssa.Value) []*govalue {
	switch builtin.Name() {
	case "print", "println":
		llargs := make([]*govalue, len(args))
		for i, arg := range args {
			llargs[i] = fr.value(arg)
		}
		fr.printValues(builtin.Name() == "println", llargs...)
		return nil

	case "panic":
		fr.callPanic(fr.value(args[0]), false)
		return nil

	case "recover":
		return []*govalue{fr.callRecover(false)}

	case "append":
		return []*govalue{fr.callAppend(fr.value(args[0]), fr.value(args[1]))}

	case "close":
		fr.chanClose(fr.value(args[0]))
		return nil

	case "cap":
		return []*govalue{fr.callCap(fr.value(args[0]))}

	case "len":
		return []*govalue{fr.callLen(fr.value(args[0]))}

	case "copy":
		return []*govalue{fr.callCopy(fr.value(args[0]), fr.value(args[1]))}

	case "delete":
		fr.mapDelete(fr.value(args[0]), fr.value(args[1]))
		return nil

	case "real":
		return []*govalue{fr.extractRealValue(fr.value(args[0]))}

	case "imag":
		return []*govalue{fr.extractImagValue(fr.value(args[0]))}

	case "complex":
		r := fr.llvmvalue(args[0])
		i := fr.llvmvalue(args[1])
		cmplx := llvm.Undef(fr.llvmtypes.ToLLVM(typ))
		cmplx = fr.builder.CreateInsertValue(cmplx, r, 0, "")
		cmplx = fr.builder.CreateInsertValue(cmplx, i, 1, "")
		return []*govalue{newValue(cmplx, typ)}

	case "ssa:wrapnilchk":
		ptr := fr.value(args[0])
		fr.nilCheck(args[0], ptr.value)
		return []*govalue{ptr}

	default:
		panic("unimplemented: " + builtin.Name())
	}
}

// callInstruction translates function call instructions.
func (fr *frame) callInstruction(instr ssa.CallInstruction) []*govalue {
	call := instr.Common()
	if builtin, ok := call.Value.(*ssa.Builtin); ok {
		var typ types.Type
		if v := instr.Value(); v != nil {
			typ = v.Type()
		}
		return fr.callBuiltin(typ, builtin, call.Args)
	}

	args := make([]*govalue, len(call.Args))
	for i, arg := range call.Args {
		args[i] = fr.value(arg)
	}

	var fn *govalue
	var chain llvm.Value
	if call.IsInvoke() {
		var recv *govalue
		fn, recv = fr.interfaceMethod(fr.llvmvalue(call.Value), call.Value.Type(), call.Method)
		args = append([]*govalue{recv}, args...)
	} else {
		if ssafn, ok := call.Value.(*ssa.Function); ok {
			llfn := fr.resolveFunctionGlobal(ssafn)
			llfn = llvm.ConstBitCast(llfn, llvm.PointerType(llvm.Int8Type(), 0))
			fn = newValue(llfn, ssafn.Type())
		} else {
			// First-class function values are stored as *{*fnptr}, so
			// we must extract the function pointer. We must also
			// set the chain, in case the function is a closure.
			fn = fr.value(call.Value)
			chain = fn.value
			fnptr := fr.builder.CreateBitCast(fn.value, llvm.PointerType(fn.value.Type(), 0), "")
			fnptr = fr.builder.CreateLoad(fnptr, "")
			fn = newValue(fnptr, fn.Type())
		}
		if recv := call.Signature().Recv(); recv != nil {
			if _, ok := recv.Type().Underlying().(*types.Pointer); !ok {
				recvalloca := fr.allocaBuilder.CreateAlloca(args[0].value.Type(), "")
				fr.builder.CreateStore(args[0].value, recvalloca)
				args[0] = newValue(recvalloca, types.NewPointer(args[0].Type()))
			}
		}
	}
	return fr.createCall(fn, chain, args)
}

func hasDefer(f *ssa.Function) bool {
	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			if _, ok := instr.(*ssa.Defer); ok {
				return true
			}
		}
	}
	return false
}

func callsRecover(f *ssa.Function) bool {
	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			if instr, ok := instr.(ssa.CallInstruction); ok {
				b, ok := instr.Common().Value.(*ssa.Builtin)
				if ok && b.Name() == "recover" {
					return true
				}
			}
		}
	}
	return false
}
