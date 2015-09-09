//===- ir.go - Bindings for ir --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the ir component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Core.h"
#include "IRBindings.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"
import "errors"

type (
	// We use these weird structs here because *Ref types are pointers and
	// Go's spec says that a pointer cannot be used as a receiver base type.
	Context struct {
		C C.LLVMContextRef
	}
	Module struct {
		C C.LLVMModuleRef
	}
	Type struct {
		C C.LLVMTypeRef
	}
	Value struct {
		C C.LLVMValueRef
	}
	BasicBlock struct {
		C C.LLVMBasicBlockRef
	}
	Builder struct {
		C C.LLVMBuilderRef
	}
	ModuleProvider struct {
		C C.LLVMModuleProviderRef
	}
	MemoryBuffer struct {
		C C.LLVMMemoryBufferRef
	}
	PassManager struct {
		C C.LLVMPassManagerRef
	}
	Use struct {
		C C.LLVMUseRef
	}
	Metadata struct {
		C C.LLVMMetadataRef
	}
	Attribute        uint64
	Opcode           C.LLVMOpcode
	TypeKind         C.LLVMTypeKind
	Linkage          C.LLVMLinkage
	Visibility       C.LLVMVisibility
	CallConv         C.LLVMCallConv
	IntPredicate     C.LLVMIntPredicate
	FloatPredicate   C.LLVMRealPredicate
	LandingPadClause C.LLVMLandingPadClauseTy
)

func (c Context) IsNil() bool        { return c.C == nil }
func (c Module) IsNil() bool         { return c.C == nil }
func (c Type) IsNil() bool           { return c.C == nil }
func (c Value) IsNil() bool          { return c.C == nil }
func (c BasicBlock) IsNil() bool     { return c.C == nil }
func (c Builder) IsNil() bool        { return c.C == nil }
func (c ModuleProvider) IsNil() bool { return c.C == nil }
func (c MemoryBuffer) IsNil() bool   { return c.C == nil }
func (c PassManager) IsNil() bool    { return c.C == nil }
func (c Use) IsNil() bool            { return c.C == nil }

// helpers
func llvmTypeRefPtr(t *Type) *C.LLVMTypeRef    { return (*C.LLVMTypeRef)(unsafe.Pointer(t)) }
func llvmValueRefPtr(t *Value) *C.LLVMValueRef { return (*C.LLVMValueRef)(unsafe.Pointer(t)) }
func llvmMetadataRefPtr(t *Metadata) *C.LLVMMetadataRef {
	return (*C.LLVMMetadataRef)(unsafe.Pointer(t))
}
func llvmBasicBlockRefPtr(t *BasicBlock) *C.LLVMBasicBlockRef {
	return (*C.LLVMBasicBlockRef)(unsafe.Pointer(t))
}
func boolToLLVMBool(b bool) C.LLVMBool {
	if b {
		return C.LLVMBool(1)
	}
	return C.LLVMBool(0)
}

func llvmValueRefs(values []Value) (*C.LLVMValueRef, C.unsigned) {
	var pt *C.LLVMValueRef
	ptlen := C.unsigned(len(values))
	if ptlen > 0 {
		pt = llvmValueRefPtr(&values[0])
	}
	return pt, ptlen
}

func llvmMetadataRefs(mds []Metadata) (*C.LLVMMetadataRef, C.unsigned) {
	var pt *C.LLVMMetadataRef
	ptlen := C.unsigned(len(mds))
	if ptlen > 0 {
		pt = llvmMetadataRefPtr(&mds[0])
	}
	return pt, ptlen
}

//-------------------------------------------------------------------------
// llvm.Attribute
//-------------------------------------------------------------------------

const (
	NoneAttribute               Attribute = 0
	ZExtAttribute               Attribute = C.LLVMZExtAttribute
	SExtAttribute               Attribute = C.LLVMSExtAttribute
	NoReturnAttribute           Attribute = C.LLVMNoReturnAttribute
	InRegAttribute              Attribute = C.LLVMInRegAttribute
	StructRetAttribute          Attribute = C.LLVMStructRetAttribute
	NoUnwindAttribute           Attribute = C.LLVMNoUnwindAttribute
	NoAliasAttribute            Attribute = C.LLVMNoAliasAttribute
	ByValAttribute              Attribute = C.LLVMByValAttribute
	NestAttribute               Attribute = C.LLVMNestAttribute
	ReadNoneAttribute           Attribute = C.LLVMReadNoneAttribute
	ReadOnlyAttribute           Attribute = C.LLVMReadOnlyAttribute
	NoInlineAttribute           Attribute = C.LLVMNoInlineAttribute
	AlwaysInlineAttribute       Attribute = C.LLVMAlwaysInlineAttribute
	OptimizeForSizeAttribute    Attribute = C.LLVMOptimizeForSizeAttribute
	StackProtectAttribute       Attribute = C.LLVMStackProtectAttribute
	StackProtectReqAttribute    Attribute = C.LLVMStackProtectReqAttribute
	Alignment                   Attribute = C.LLVMAlignment
	NoCaptureAttribute          Attribute = C.LLVMNoCaptureAttribute
	NoRedZoneAttribute          Attribute = C.LLVMNoRedZoneAttribute
	NoImplicitFloatAttribute    Attribute = C.LLVMNoImplicitFloatAttribute
	NakedAttribute              Attribute = C.LLVMNakedAttribute
	InlineHintAttribute         Attribute = C.LLVMInlineHintAttribute
	StackAlignment              Attribute = C.LLVMStackAlignment
	ReturnsTwiceAttribute       Attribute = C.LLVMReturnsTwice
	UWTableAttribute            Attribute = C.LLVMUWTable
	NonLazyBindAttribute        Attribute = 1 << 31
	SanitizeAddressAttribute    Attribute = 1 << 32
	MinSizeAttribute            Attribute = 1 << 33
	NoDuplicateAttribute        Attribute = 1 << 34
	StackProtectStrongAttribute Attribute = 1 << 35
	SanitizeThreadAttribute     Attribute = 1 << 36
	SanitizeMemoryAttribute     Attribute = 1 << 37
	NoBuiltinAttribute          Attribute = 1 << 38
	ReturnedAttribute           Attribute = 1 << 39
	ColdAttribute               Attribute = 1 << 40
	BuiltinAttribute            Attribute = 1 << 41
	OptimizeNoneAttribute       Attribute = 1 << 42
	InAllocaAttribute           Attribute = 1 << 43
	NonNullAttribute            Attribute = 1 << 44
	JumpTableAttribute          Attribute = 1 << 45
	ConvergentAttribute         Attribute = 1 << 46
	SafeStackAttribute          Attribute = 1 << 47
)

//-------------------------------------------------------------------------
// llvm.Opcode
//-------------------------------------------------------------------------

const (
	Ret         Opcode = C.LLVMRet
	Br          Opcode = C.LLVMBr
	Switch      Opcode = C.LLVMSwitch
	IndirectBr  Opcode = C.LLVMIndirectBr
	Invoke      Opcode = C.LLVMInvoke
	Unreachable Opcode = C.LLVMUnreachable

	// Standard Binary Operators
	Add  Opcode = C.LLVMAdd
	FAdd Opcode = C.LLVMFAdd
	Sub  Opcode = C.LLVMSub
	FSub Opcode = C.LLVMFSub
	Mul  Opcode = C.LLVMMul
	FMul Opcode = C.LLVMFMul
	UDiv Opcode = C.LLVMUDiv
	SDiv Opcode = C.LLVMSDiv
	FDiv Opcode = C.LLVMFDiv
	URem Opcode = C.LLVMURem
	SRem Opcode = C.LLVMSRem
	FRem Opcode = C.LLVMFRem

	// Logical Operators
	Shl  Opcode = C.LLVMShl
	LShr Opcode = C.LLVMLShr
	AShr Opcode = C.LLVMAShr
	And  Opcode = C.LLVMAnd
	Or   Opcode = C.LLVMOr
	Xor  Opcode = C.LLVMXor

	// Memory Operators
	Alloca        Opcode = C.LLVMAlloca
	Load          Opcode = C.LLVMLoad
	Store         Opcode = C.LLVMStore
	GetElementPtr Opcode = C.LLVMGetElementPtr

	// Cast Operators
	Trunc    Opcode = C.LLVMTrunc
	ZExt     Opcode = C.LLVMZExt
	SExt     Opcode = C.LLVMSExt
	FPToUI   Opcode = C.LLVMFPToUI
	FPToSI   Opcode = C.LLVMFPToSI
	UIToFP   Opcode = C.LLVMUIToFP
	SIToFP   Opcode = C.LLVMSIToFP
	FPTrunc  Opcode = C.LLVMFPTrunc
	FPExt    Opcode = C.LLVMFPExt
	PtrToInt Opcode = C.LLVMPtrToInt
	IntToPtr Opcode = C.LLVMIntToPtr
	BitCast  Opcode = C.LLVMBitCast

	// Other Operators
	ICmp   Opcode = C.LLVMICmp
	FCmp   Opcode = C.LLVMFCmp
	PHI    Opcode = C.LLVMPHI
	Call   Opcode = C.LLVMCall
	Select Opcode = C.LLVMSelect
	// UserOp1
	// UserOp2
	VAArg          Opcode = C.LLVMVAArg
	ExtractElement Opcode = C.LLVMExtractElement
	InsertElement  Opcode = C.LLVMInsertElement
	ShuffleVector  Opcode = C.LLVMShuffleVector
	ExtractValue   Opcode = C.LLVMExtractValue
	InsertValue    Opcode = C.LLVMInsertValue
)

//-------------------------------------------------------------------------
// llvm.TypeKind
//-------------------------------------------------------------------------

const (
	VoidTypeKind      TypeKind = C.LLVMVoidTypeKind
	FloatTypeKind     TypeKind = C.LLVMFloatTypeKind
	DoubleTypeKind    TypeKind = C.LLVMDoubleTypeKind
	X86_FP80TypeKind  TypeKind = C.LLVMX86_FP80TypeKind
	FP128TypeKind     TypeKind = C.LLVMFP128TypeKind
	PPC_FP128TypeKind TypeKind = C.LLVMPPC_FP128TypeKind
	LabelTypeKind     TypeKind = C.LLVMLabelTypeKind
	IntegerTypeKind   TypeKind = C.LLVMIntegerTypeKind
	FunctionTypeKind  TypeKind = C.LLVMFunctionTypeKind
	StructTypeKind    TypeKind = C.LLVMStructTypeKind
	ArrayTypeKind     TypeKind = C.LLVMArrayTypeKind
	PointerTypeKind   TypeKind = C.LLVMPointerTypeKind
	VectorTypeKind    TypeKind = C.LLVMVectorTypeKind
	MetadataTypeKind  TypeKind = C.LLVMMetadataTypeKind
)

//-------------------------------------------------------------------------
// llvm.Linkage
//-------------------------------------------------------------------------

const (
	ExternalLinkage            Linkage = C.LLVMExternalLinkage
	AvailableExternallyLinkage Linkage = C.LLVMAvailableExternallyLinkage
	LinkOnceAnyLinkage         Linkage = C.LLVMLinkOnceAnyLinkage
	LinkOnceODRLinkage         Linkage = C.LLVMLinkOnceODRLinkage
	WeakAnyLinkage             Linkage = C.LLVMWeakAnyLinkage
	WeakODRLinkage             Linkage = C.LLVMWeakODRLinkage
	AppendingLinkage           Linkage = C.LLVMAppendingLinkage
	InternalLinkage            Linkage = C.LLVMInternalLinkage
	PrivateLinkage             Linkage = C.LLVMPrivateLinkage
	ExternalWeakLinkage        Linkage = C.LLVMExternalWeakLinkage
	CommonLinkage              Linkage = C.LLVMCommonLinkage
)

//-------------------------------------------------------------------------
// llvm.Visibility
//-------------------------------------------------------------------------

const (
	DefaultVisibility   Visibility = C.LLVMDefaultVisibility
	HiddenVisibility    Visibility = C.LLVMHiddenVisibility
	ProtectedVisibility Visibility = C.LLVMProtectedVisibility
)

//-------------------------------------------------------------------------
// llvm.CallConv
//-------------------------------------------------------------------------

const (
	CCallConv           CallConv = C.LLVMCCallConv
	FastCallConv        CallConv = C.LLVMFastCallConv
	ColdCallConv        CallConv = C.LLVMColdCallConv
	X86StdcallCallConv  CallConv = C.LLVMX86StdcallCallConv
	X86FastcallCallConv CallConv = C.LLVMX86FastcallCallConv
)

//-------------------------------------------------------------------------
// llvm.IntPredicate
//-------------------------------------------------------------------------

const (
	IntEQ  IntPredicate = C.LLVMIntEQ
	IntNE  IntPredicate = C.LLVMIntNE
	IntUGT IntPredicate = C.LLVMIntUGT
	IntUGE IntPredicate = C.LLVMIntUGE
	IntULT IntPredicate = C.LLVMIntULT
	IntULE IntPredicate = C.LLVMIntULE
	IntSGT IntPredicate = C.LLVMIntSGT
	IntSGE IntPredicate = C.LLVMIntSGE
	IntSLT IntPredicate = C.LLVMIntSLT
	IntSLE IntPredicate = C.LLVMIntSLE
)

//-------------------------------------------------------------------------
// llvm.FloatPredicate
//-------------------------------------------------------------------------

const (
	FloatPredicateFalse FloatPredicate = C.LLVMRealPredicateFalse
	FloatOEQ            FloatPredicate = C.LLVMRealOEQ
	FloatOGT            FloatPredicate = C.LLVMRealOGT
	FloatOGE            FloatPredicate = C.LLVMRealOGE
	FloatOLT            FloatPredicate = C.LLVMRealOLT
	FloatOLE            FloatPredicate = C.LLVMRealOLE
	FloatONE            FloatPredicate = C.LLVMRealONE
	FloatORD            FloatPredicate = C.LLVMRealORD
	FloatUNO            FloatPredicate = C.LLVMRealUNO
	FloatUEQ            FloatPredicate = C.LLVMRealUEQ
	FloatUGT            FloatPredicate = C.LLVMRealUGT
	FloatUGE            FloatPredicate = C.LLVMRealUGE
	FloatULT            FloatPredicate = C.LLVMRealULT
	FloatULE            FloatPredicate = C.LLVMRealULE
	FloatUNE            FloatPredicate = C.LLVMRealUNE
	FloatPredicateTrue  FloatPredicate = C.LLVMRealPredicateTrue
)

//-------------------------------------------------------------------------
// llvm.LandingPadClause
//-------------------------------------------------------------------------

const (
	LandingPadCatch  LandingPadClause = C.LLVMLandingPadCatch
	LandingPadFilter LandingPadClause = C.LLVMLandingPadFilter
)

//-------------------------------------------------------------------------
// llvm.Context
//-------------------------------------------------------------------------

func NewContext() Context    { return Context{C.LLVMContextCreate()} }
func GlobalContext() Context { return Context{C.LLVMGetGlobalContext()} }
func (c Context) Dispose()   { C.LLVMContextDispose(c.C) }

func (c Context) MDKindID(name string) (id int) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	id = int(C.LLVMGetMDKindIDInContext(c.C, cname, C.unsigned(len(name))))
	return
}

func MDKindID(name string) (id int) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	id = int(C.LLVMGetMDKindID(cname, C.unsigned(len(name))))
	return
}

//-------------------------------------------------------------------------
// llvm.Module
//-------------------------------------------------------------------------

// Create and destroy modules.
// See llvm::Module::Module.
func NewModule(name string) (m Module) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	m.C = C.LLVMModuleCreateWithName(cname)
	return
}

func (c Context) NewModule(name string) (m Module) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	m.C = C.LLVMModuleCreateWithNameInContext(cname, c.C)
	return
}

// See llvm::Module::~Module
func (m Module) Dispose() { C.LLVMDisposeModule(m.C) }

// Data layout. See Module::getDataLayout.
func (m Module) DataLayout() string {
	clayout := C.LLVMGetDataLayout(m.C)
	return C.GoString(clayout)
}

func (m Module) SetDataLayout(layout string) {
	clayout := C.CString(layout)
	defer C.free(unsafe.Pointer(clayout))
	C.LLVMSetDataLayout(m.C, clayout)
}

// Target triple. See Module::getTargetTriple.
func (m Module) Target() string {
	ctarget := C.LLVMGetTarget(m.C)
	return C.GoString(ctarget)
}
func (m Module) SetTarget(target string) {
	ctarget := C.CString(target)
	defer C.free(unsafe.Pointer(ctarget))
	C.LLVMSetTarget(m.C, ctarget)
}

func (m Module) GetTypeByName(name string) (t Type) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	t.C = C.LLVMGetTypeByName(m.C, cname)
	return
}

// See Module::dump.
func (m Module) Dump() {
	C.LLVMDumpModule(m.C)
}

func (m Module) String() string {
	cir := C.LLVMPrintModuleToString(m.C)
	defer C.free(unsafe.Pointer(cir))
	ir := C.GoString(cir)
	return ir
}

// See Module::setModuleInlineAsm.
func (m Module) SetInlineAsm(asm string) {
	casm := C.CString(asm)
	defer C.free(unsafe.Pointer(casm))
	C.LLVMSetModuleInlineAsm(m.C, casm)
}

func (m Module) AddNamedMetadataOperand(name string, operand Metadata) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.LLVMAddNamedMetadataOperand2(m.C, cname, operand.C)
}

func (m Module) Context() (c Context) {
	c.C = C.LLVMGetModuleContext(m.C)
	return
}

//-------------------------------------------------------------------------
// llvm.Type
//-------------------------------------------------------------------------

// LLVM types conform to the following hierarchy:
//
//   types:
//     integer type
//     real type
//     function type
//     sequence types:
//       array type
//       pointer type
//       vector type
//     void type
//     label type
//     opaque type

// See llvm::LLVMTypeKind::getTypeID.
func (t Type) TypeKind() TypeKind { return TypeKind(C.LLVMGetTypeKind(t.C)) }

// See llvm::LLVMType::getContext.
func (t Type) Context() (c Context) {
	c.C = C.LLVMGetTypeContext(t.C)
	return
}

// Operations on integer types
func (c Context) Int1Type() (t Type)  { t.C = C.LLVMInt1TypeInContext(c.C); return }
func (c Context) Int8Type() (t Type)  { t.C = C.LLVMInt8TypeInContext(c.C); return }
func (c Context) Int16Type() (t Type) { t.C = C.LLVMInt16TypeInContext(c.C); return }
func (c Context) Int32Type() (t Type) { t.C = C.LLVMInt32TypeInContext(c.C); return }
func (c Context) Int64Type() (t Type) { t.C = C.LLVMInt64TypeInContext(c.C); return }
func (c Context) IntType(numbits int) (t Type) {
	t.C = C.LLVMIntTypeInContext(c.C, C.unsigned(numbits))
	return
}

func Int1Type() (t Type)  { t.C = C.LLVMInt1Type(); return }
func Int8Type() (t Type)  { t.C = C.LLVMInt8Type(); return }
func Int16Type() (t Type) { t.C = C.LLVMInt16Type(); return }
func Int32Type() (t Type) { t.C = C.LLVMInt32Type(); return }
func Int64Type() (t Type) { t.C = C.LLVMInt64Type(); return }

func IntType(numbits int) (t Type) {
	t.C = C.LLVMIntType(C.unsigned(numbits))
	return
}

func (t Type) IntTypeWidth() int {
	return int(C.LLVMGetIntTypeWidth(t.C))
}

// Operations on real types
func (c Context) FloatType() (t Type)    { t.C = C.LLVMFloatTypeInContext(c.C); return }
func (c Context) DoubleType() (t Type)   { t.C = C.LLVMDoubleTypeInContext(c.C); return }
func (c Context) X86FP80Type() (t Type)  { t.C = C.LLVMX86FP80TypeInContext(c.C); return }
func (c Context) FP128Type() (t Type)    { t.C = C.LLVMFP128TypeInContext(c.C); return }
func (c Context) PPCFP128Type() (t Type) { t.C = C.LLVMPPCFP128TypeInContext(c.C); return }

func FloatType() (t Type)    { t.C = C.LLVMFloatType(); return }
func DoubleType() (t Type)   { t.C = C.LLVMDoubleType(); return }
func X86FP80Type() (t Type)  { t.C = C.LLVMX86FP80Type(); return }
func FP128Type() (t Type)    { t.C = C.LLVMFP128Type(); return }
func PPCFP128Type() (t Type) { t.C = C.LLVMPPCFP128Type(); return }

// Operations on function types
func FunctionType(returnType Type, paramTypes []Type, isVarArg bool) (t Type) {
	var pt *C.LLVMTypeRef
	var ptlen C.unsigned
	if len(paramTypes) > 0 {
		pt = llvmTypeRefPtr(&paramTypes[0])
		ptlen = C.unsigned(len(paramTypes))
	}
	t.C = C.LLVMFunctionType(returnType.C,
		pt,
		ptlen,
		boolToLLVMBool(isVarArg))
	return
}

func (t Type) IsFunctionVarArg() bool { return C.LLVMIsFunctionVarArg(t.C) != 0 }
func (t Type) ReturnType() (rt Type)  { rt.C = C.LLVMGetReturnType(t.C); return }
func (t Type) ParamTypesCount() int   { return int(C.LLVMCountParamTypes(t.C)) }
func (t Type) ParamTypes() []Type {
	count := t.ParamTypesCount()
	if count > 0 {
		out := make([]Type, count)
		C.LLVMGetParamTypes(t.C, llvmTypeRefPtr(&out[0]))
		return out
	}
	return nil
}

// Operations on struct types
func (c Context) StructType(elementTypes []Type, packed bool) (t Type) {
	var pt *C.LLVMTypeRef
	var ptlen C.unsigned
	if len(elementTypes) > 0 {
		pt = llvmTypeRefPtr(&elementTypes[0])
		ptlen = C.unsigned(len(elementTypes))
	}
	t.C = C.LLVMStructTypeInContext(c.C,
		pt,
		ptlen,
		boolToLLVMBool(packed))
	return
}

func StructType(elementTypes []Type, packed bool) (t Type) {
	var pt *C.LLVMTypeRef
	var ptlen C.unsigned
	if len(elementTypes) > 0 {
		pt = llvmTypeRefPtr(&elementTypes[0])
		ptlen = C.unsigned(len(elementTypes))
	}
	t.C = C.LLVMStructType(pt, ptlen, boolToLLVMBool(packed))
	return
}

func (c Context) StructCreateNamed(name string) (t Type) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	t.C = C.LLVMStructCreateNamed(c.C, cname)
	return
}

func (t Type) StructName() string {
	return C.GoString(C.LLVMGetStructName(t.C))
}

func (t Type) StructSetBody(elementTypes []Type, packed bool) {
	var pt *C.LLVMTypeRef
	var ptlen C.unsigned
	if len(elementTypes) > 0 {
		pt = llvmTypeRefPtr(&elementTypes[0])
		ptlen = C.unsigned(len(elementTypes))
	}
	C.LLVMStructSetBody(t.C, pt, ptlen, boolToLLVMBool(packed))
}

func (t Type) IsStructPacked() bool         { return C.LLVMIsPackedStruct(t.C) != 0 }
func (t Type) StructElementTypesCount() int { return int(C.LLVMCountStructElementTypes(t.C)) }
func (t Type) StructElementTypes() []Type {
	out := make([]Type, t.StructElementTypesCount())
	if len(out) > 0 {
		C.LLVMGetStructElementTypes(t.C, llvmTypeRefPtr(&out[0]))
	}
	return out
}

// Operations on array, pointer, and vector types (sequence types)
func ArrayType(elementType Type, elementCount int) (t Type) {
	t.C = C.LLVMArrayType(elementType.C, C.unsigned(elementCount))
	return
}
func PointerType(elementType Type, addressSpace int) (t Type) {
	t.C = C.LLVMPointerType(elementType.C, C.unsigned(addressSpace))
	return
}
func VectorType(elementType Type, elementCount int) (t Type) {
	t.C = C.LLVMVectorType(elementType.C, C.unsigned(elementCount))
	return
}

func (t Type) ElementType() (rt Type)   { rt.C = C.LLVMGetElementType(t.C); return }
func (t Type) ArrayLength() int         { return int(C.LLVMGetArrayLength(t.C)) }
func (t Type) PointerAddressSpace() int { return int(C.LLVMGetPointerAddressSpace(t.C)) }
func (t Type) VectorSize() int          { return int(C.LLVMGetVectorSize(t.C)) }

// Operations on other types
func (c Context) VoidType() (t Type)  { t.C = C.LLVMVoidTypeInContext(c.C); return }
func (c Context) LabelType() (t Type) { t.C = C.LLVMLabelTypeInContext(c.C); return }

func VoidType() (t Type)  { t.C = C.LLVMVoidType(); return }
func LabelType() (t Type) { t.C = C.LLVMLabelType(); return }

//-------------------------------------------------------------------------
// llvm.Value
//-------------------------------------------------------------------------

// Operations on all values
func (v Value) Type() (t Type) { t.C = C.LLVMTypeOf(v.C); return }
func (v Value) Name() string   { return C.GoString(C.LLVMGetValueName(v.C)) }
func (v Value) SetName(name string) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.LLVMSetValueName(v.C, cname)
}
func (v Value) Dump()                       { C.LLVMDumpValue(v.C) }
func (v Value) ReplaceAllUsesWith(nv Value) { C.LLVMReplaceAllUsesWith(v.C, nv.C) }
func (v Value) HasMetadata() bool           { return C.LLVMHasMetadata(v.C) != 0 }
func (v Value) Metadata(kind int) (rv Value) {
	rv.C = C.LLVMGetMetadata(v.C, C.unsigned(kind))
	return
}
func (v Value) SetMetadata(kind int, node Metadata) {
	C.LLVMSetMetadata2(v.C, C.unsigned(kind), node.C)
}

// Conversion functions.
// Return the input value if it is an instance of the specified class, otherwise NULL.
// See llvm::dyn_cast_or_null<>.
func (v Value) IsAArgument() (rv Value)   { rv.C = C.LLVMIsAArgument(v.C); return }
func (v Value) IsABasicBlock() (rv Value) { rv.C = C.LLVMIsABasicBlock(v.C); return }
func (v Value) IsAInlineAsm() (rv Value)  { rv.C = C.LLVMIsAInlineAsm(v.C); return }
func (v Value) IsAUser() (rv Value)       { rv.C = C.LLVMIsAUser(v.C); return }
func (v Value) IsAConstant() (rv Value)   { rv.C = C.LLVMIsAConstant(v.C); return }
func (v Value) IsAConstantAggregateZero() (rv Value) {
	rv.C = C.LLVMIsAConstantAggregateZero(v.C)
	return
}
func (v Value) IsAConstantArray() (rv Value)       { rv.C = C.LLVMIsAConstantArray(v.C); return }
func (v Value) IsAConstantExpr() (rv Value)        { rv.C = C.LLVMIsAConstantExpr(v.C); return }
func (v Value) IsAConstantFP() (rv Value)          { rv.C = C.LLVMIsAConstantFP(v.C); return }
func (v Value) IsAConstantInt() (rv Value)         { rv.C = C.LLVMIsAConstantInt(v.C); return }
func (v Value) IsAConstantPointerNull() (rv Value) { rv.C = C.LLVMIsAConstantPointerNull(v.C); return }
func (v Value) IsAConstantStruct() (rv Value)      { rv.C = C.LLVMIsAConstantStruct(v.C); return }
func (v Value) IsAConstantVector() (rv Value)      { rv.C = C.LLVMIsAConstantVector(v.C); return }
func (v Value) IsAGlobalValue() (rv Value)         { rv.C = C.LLVMIsAGlobalValue(v.C); return }
func (v Value) IsAFunction() (rv Value)            { rv.C = C.LLVMIsAFunction(v.C); return }
func (v Value) IsAGlobalAlias() (rv Value)         { rv.C = C.LLVMIsAGlobalAlias(v.C); return }
func (v Value) IsAGlobalVariable() (rv Value)      { rv.C = C.LLVMIsAGlobalVariable(v.C); return }
func (v Value) IsAUndefValue() (rv Value)          { rv.C = C.LLVMIsAUndefValue(v.C); return }
func (v Value) IsAInstruction() (rv Value)         { rv.C = C.LLVMIsAInstruction(v.C); return }
func (v Value) IsABinaryOperator() (rv Value)      { rv.C = C.LLVMIsABinaryOperator(v.C); return }
func (v Value) IsACallInst() (rv Value)            { rv.C = C.LLVMIsACallInst(v.C); return }
func (v Value) IsAIntrinsicInst() (rv Value)       { rv.C = C.LLVMIsAIntrinsicInst(v.C); return }
func (v Value) IsADbgInfoIntrinsic() (rv Value)    { rv.C = C.LLVMIsADbgInfoIntrinsic(v.C); return }
func (v Value) IsADbgDeclareInst() (rv Value)      { rv.C = C.LLVMIsADbgDeclareInst(v.C); return }
func (v Value) IsAMemIntrinsic() (rv Value)        { rv.C = C.LLVMIsAMemIntrinsic(v.C); return }
func (v Value) IsAMemCpyInst() (rv Value)          { rv.C = C.LLVMIsAMemCpyInst(v.C); return }
func (v Value) IsAMemMoveInst() (rv Value)         { rv.C = C.LLVMIsAMemMoveInst(v.C); return }
func (v Value) IsAMemSetInst() (rv Value)          { rv.C = C.LLVMIsAMemSetInst(v.C); return }
func (v Value) IsACmpInst() (rv Value)             { rv.C = C.LLVMIsACmpInst(v.C); return }
func (v Value) IsAFCmpInst() (rv Value)            { rv.C = C.LLVMIsAFCmpInst(v.C); return }
func (v Value) IsAICmpInst() (rv Value)            { rv.C = C.LLVMIsAICmpInst(v.C); return }
func (v Value) IsAExtractElementInst() (rv Value)  { rv.C = C.LLVMIsAExtractElementInst(v.C); return }
func (v Value) IsAGetElementPtrInst() (rv Value)   { rv.C = C.LLVMIsAGetElementPtrInst(v.C); return }
func (v Value) IsAInsertElementInst() (rv Value)   { rv.C = C.LLVMIsAInsertElementInst(v.C); return }
func (v Value) IsAInsertValueInst() (rv Value)     { rv.C = C.LLVMIsAInsertValueInst(v.C); return }
func (v Value) IsAPHINode() (rv Value)             { rv.C = C.LLVMIsAPHINode(v.C); return }
func (v Value) IsASelectInst() (rv Value)          { rv.C = C.LLVMIsASelectInst(v.C); return }
func (v Value) IsAShuffleVectorInst() (rv Value)   { rv.C = C.LLVMIsAShuffleVectorInst(v.C); return }
func (v Value) IsAStoreInst() (rv Value)           { rv.C = C.LLVMIsAStoreInst(v.C); return }
func (v Value) IsATerminatorInst() (rv Value)      { rv.C = C.LLVMIsATerminatorInst(v.C); return }
func (v Value) IsABranchInst() (rv Value)          { rv.C = C.LLVMIsABranchInst(v.C); return }
func (v Value) IsAInvokeInst() (rv Value)          { rv.C = C.LLVMIsAInvokeInst(v.C); return }
func (v Value) IsAReturnInst() (rv Value)          { rv.C = C.LLVMIsAReturnInst(v.C); return }
func (v Value) IsASwitchInst() (rv Value)          { rv.C = C.LLVMIsASwitchInst(v.C); return }
func (v Value) IsAUnreachableInst() (rv Value)     { rv.C = C.LLVMIsAUnreachableInst(v.C); return }
func (v Value) IsAUnaryInstruction() (rv Value)    { rv.C = C.LLVMIsAUnaryInstruction(v.C); return }
func (v Value) IsAAllocaInst() (rv Value)          { rv.C = C.LLVMIsAAllocaInst(v.C); return }
func (v Value) IsACastInst() (rv Value)            { rv.C = C.LLVMIsACastInst(v.C); return }
func (v Value) IsABitCastInst() (rv Value)         { rv.C = C.LLVMIsABitCastInst(v.C); return }
func (v Value) IsAFPExtInst() (rv Value)           { rv.C = C.LLVMIsAFPExtInst(v.C); return }
func (v Value) IsAFPToSIInst() (rv Value)          { rv.C = C.LLVMIsAFPToSIInst(v.C); return }
func (v Value) IsAFPToUIInst() (rv Value)          { rv.C = C.LLVMIsAFPToUIInst(v.C); return }
func (v Value) IsAFPTruncInst() (rv Value)         { rv.C = C.LLVMIsAFPTruncInst(v.C); return }
func (v Value) IsAIntToPtrInst() (rv Value)        { rv.C = C.LLVMIsAIntToPtrInst(v.C); return }
func (v Value) IsAPtrToIntInst() (rv Value)        { rv.C = C.LLVMIsAPtrToIntInst(v.C); return }
func (v Value) IsASExtInst() (rv Value)            { rv.C = C.LLVMIsASExtInst(v.C); return }
func (v Value) IsASIToFPInst() (rv Value)          { rv.C = C.LLVMIsASIToFPInst(v.C); return }
func (v Value) IsATruncInst() (rv Value)           { rv.C = C.LLVMIsATruncInst(v.C); return }
func (v Value) IsAUIToFPInst() (rv Value)          { rv.C = C.LLVMIsAUIToFPInst(v.C); return }
func (v Value) IsAZExtInst() (rv Value)            { rv.C = C.LLVMIsAZExtInst(v.C); return }
func (v Value) IsAExtractValueInst() (rv Value)    { rv.C = C.LLVMIsAExtractValueInst(v.C); return }
func (v Value) IsALoadInst() (rv Value)            { rv.C = C.LLVMIsALoadInst(v.C); return }
func (v Value) IsAVAArgInst() (rv Value)           { rv.C = C.LLVMIsAVAArgInst(v.C); return }

// Operations on Uses
func (v Value) FirstUse() (u Use)  { u.C = C.LLVMGetFirstUse(v.C); return }
func (u Use) NextUse() (ru Use)    { ru.C = C.LLVMGetNextUse(u.C); return }
func (u Use) User() (v Value)      { v.C = C.LLVMGetUser(u.C); return }
func (u Use) UsedValue() (v Value) { v.C = C.LLVMGetUsedValue(u.C); return }

// Operations on Users
func (v Value) Operand(i int) (rv Value)   { rv.C = C.LLVMGetOperand(v.C, C.unsigned(i)); return }
func (v Value) SetOperand(i int, op Value) { C.LLVMSetOperand(v.C, C.unsigned(i), op.C) }
func (v Value) OperandsCount() int         { return int(C.LLVMGetNumOperands(v.C)) }

// Operations on constants of any type
func ConstNull(t Type) (v Value)        { v.C = C.LLVMConstNull(t.C); return }
func ConstAllOnes(t Type) (v Value)     { v.C = C.LLVMConstAllOnes(t.C); return }
func Undef(t Type) (v Value)            { v.C = C.LLVMGetUndef(t.C); return }
func (v Value) IsConstant() bool        { return C.LLVMIsConstant(v.C) != 0 }
func (v Value) IsNull() bool            { return C.LLVMIsNull(v.C) != 0 }
func (v Value) IsUndef() bool           { return C.LLVMIsUndef(v.C) != 0 }
func ConstPointerNull(t Type) (v Value) { v.C = C.LLVMConstPointerNull(t.C); return }

// Operations on metadata
func (c Context) MDString(str string) (md Metadata) {
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))
	md.C = C.LLVMMDString2(c.C, cstr, C.unsigned(len(str)))
	return
}
func (c Context) MDNode(mds []Metadata) (md Metadata) {
	ptr, nvals := llvmMetadataRefs(mds)
	md.C = C.LLVMMDNode2(c.C, ptr, nvals)
	return
}
func (c Context) TemporaryMDNode(mds []Metadata) (md Metadata) {
	ptr, nvals := llvmMetadataRefs(mds)
	md.C = C.LLVMTemporaryMDNode(c.C, ptr, nvals)
	return
}
func (v Value) ConstantAsMetadata() (md Metadata) {
	md.C = C.LLVMConstantAsMetadata(v.C)
	return
}

// Operations on scalar constants
func ConstInt(t Type, n uint64, signExtend bool) (v Value) {
	v.C = C.LLVMConstInt(t.C,
		C.ulonglong(n),
		boolToLLVMBool(signExtend))
	return
}
func ConstIntFromString(t Type, str string, radix int) (v Value) {
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))
	v.C = C.LLVMConstIntOfString(t.C, cstr, C.uint8_t(radix))
	return
}
func ConstFloat(t Type, n float64) (v Value) {
	v.C = C.LLVMConstReal(t.C, C.double(n))
	return
}
func ConstFloatFromString(t Type, str string) (v Value) {
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))
	v.C = C.LLVMConstRealOfString(t.C, cstr)
	return
}

func (v Value) ZExtValue() uint64 { return uint64(C.LLVMConstIntGetZExtValue(v.C)) }
func (v Value) SExtValue() int64  { return int64(C.LLVMConstIntGetSExtValue(v.C)) }

// Operations on composite constants
func (c Context) ConstString(str string, addnull bool) (v Value) {
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))
	v.C = C.LLVMConstStringInContext(c.C, cstr,
		C.unsigned(len(str)), boolToLLVMBool(!addnull))
	return
}
func (c Context) ConstStruct(constVals []Value, packed bool) (v Value) {
	ptr, nvals := llvmValueRefs(constVals)
	v.C = C.LLVMConstStructInContext(c.C, ptr, nvals,
		boolToLLVMBool(packed))
	return
}
func ConstNamedStruct(t Type, constVals []Value) (v Value) {
	ptr, nvals := llvmValueRefs(constVals)
	v.C = C.LLVMConstNamedStruct(t.C, ptr, nvals)
	return
}
func ConstString(str string, addnull bool) (v Value) {
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))
	v.C = C.LLVMConstString(cstr,
		C.unsigned(len(str)), boolToLLVMBool(!addnull))
	return
}
func ConstArray(t Type, constVals []Value) (v Value) {
	ptr, nvals := llvmValueRefs(constVals)
	v.C = C.LLVMConstArray(t.C, ptr, nvals)
	return
}
func ConstStruct(constVals []Value, packed bool) (v Value) {
	ptr, nvals := llvmValueRefs(constVals)
	v.C = C.LLVMConstStruct(ptr, nvals, boolToLLVMBool(packed))
	return
}
func ConstVector(scalarConstVals []Value, packed bool) (v Value) {
	ptr, nvals := llvmValueRefs(scalarConstVals)
	v.C = C.LLVMConstVector(ptr, nvals)
	return
}

// Constant expressions
func (v Value) Opcode() Opcode                { return Opcode(C.LLVMGetConstOpcode(v.C)) }
func (v Value) InstructionOpcode() Opcode     { return Opcode(C.LLVMGetInstructionOpcode(v.C)) }
func AlignOf(t Type) (v Value)                { v.C = C.LLVMAlignOf(t.C); return }
func SizeOf(t Type) (v Value)                 { v.C = C.LLVMSizeOf(t.C); return }
func ConstNeg(v Value) (rv Value)             { rv.C = C.LLVMConstNeg(v.C); return }
func ConstNSWNeg(v Value) (rv Value)          { rv.C = C.LLVMConstNSWNeg(v.C); return }
func ConstNUWNeg(v Value) (rv Value)          { rv.C = C.LLVMConstNUWNeg(v.C); return }
func ConstFNeg(v Value) (rv Value)            { rv.C = C.LLVMConstFNeg(v.C); return }
func ConstNot(v Value) (rv Value)             { rv.C = C.LLVMConstNot(v.C); return }
func ConstAdd(lhs, rhs Value) (v Value)       { v.C = C.LLVMConstAdd(lhs.C, rhs.C); return }
func ConstNSWAdd(lhs, rhs Value) (v Value)    { v.C = C.LLVMConstNSWAdd(lhs.C, rhs.C); return }
func ConstNUWAdd(lhs, rhs Value) (v Value)    { v.C = C.LLVMConstNUWAdd(lhs.C, rhs.C); return }
func ConstFAdd(lhs, rhs Value) (v Value)      { v.C = C.LLVMConstFAdd(lhs.C, rhs.C); return }
func ConstSub(lhs, rhs Value) (v Value)       { v.C = C.LLVMConstSub(lhs.C, rhs.C); return }
func ConstNSWSub(lhs, rhs Value) (v Value)    { v.C = C.LLVMConstNSWSub(lhs.C, rhs.C); return }
func ConstNUWSub(lhs, rhs Value) (v Value)    { v.C = C.LLVMConstNUWSub(lhs.C, rhs.C); return }
func ConstFSub(lhs, rhs Value) (v Value)      { v.C = C.LLVMConstFSub(lhs.C, rhs.C); return }
func ConstMul(lhs, rhs Value) (v Value)       { v.C = C.LLVMConstMul(lhs.C, rhs.C); return }
func ConstNSWMul(lhs, rhs Value) (v Value)    { v.C = C.LLVMConstNSWMul(lhs.C, rhs.C); return }
func ConstNUWMul(lhs, rhs Value) (v Value)    { v.C = C.LLVMConstNUWMul(lhs.C, rhs.C); return }
func ConstFMul(lhs, rhs Value) (v Value)      { v.C = C.LLVMConstFMul(lhs.C, rhs.C); return }
func ConstUDiv(lhs, rhs Value) (v Value)      { v.C = C.LLVMConstUDiv(lhs.C, rhs.C); return }
func ConstSDiv(lhs, rhs Value) (v Value)      { v.C = C.LLVMConstSDiv(lhs.C, rhs.C); return }
func ConstExactSDiv(lhs, rhs Value) (v Value) { v.C = C.LLVMConstExactSDiv(lhs.C, rhs.C); return }
func ConstFDiv(lhs, rhs Value) (v Value)      { v.C = C.LLVMConstFDiv(lhs.C, rhs.C); return }
func ConstURem(lhs, rhs Value) (v Value)      { v.C = C.LLVMConstURem(lhs.C, rhs.C); return }
func ConstSRem(lhs, rhs Value) (v Value)      { v.C = C.LLVMConstSRem(lhs.C, rhs.C); return }
func ConstFRem(lhs, rhs Value) (v Value)      { v.C = C.LLVMConstFRem(lhs.C, rhs.C); return }
func ConstAnd(lhs, rhs Value) (v Value)       { v.C = C.LLVMConstAnd(lhs.C, rhs.C); return }
func ConstOr(lhs, rhs Value) (v Value)        { v.C = C.LLVMConstOr(lhs.C, rhs.C); return }
func ConstXor(lhs, rhs Value) (v Value)       { v.C = C.LLVMConstXor(lhs.C, rhs.C); return }

func ConstICmp(pred IntPredicate, lhs, rhs Value) (v Value) {
	v.C = C.LLVMConstICmp(C.LLVMIntPredicate(pred), lhs.C, rhs.C)
	return
}
func ConstFCmp(pred FloatPredicate, lhs, rhs Value) (v Value) {
	v.C = C.LLVMConstFCmp(C.LLVMRealPredicate(pred), lhs.C, rhs.C)
	return
}

func ConstShl(lhs, rhs Value) (v Value)  { v.C = C.LLVMConstShl(lhs.C, rhs.C); return }
func ConstLShr(lhs, rhs Value) (v Value) { v.C = C.LLVMConstLShr(lhs.C, rhs.C); return }
func ConstAShr(lhs, rhs Value) (v Value) { v.C = C.LLVMConstAShr(lhs.C, rhs.C); return }

func ConstGEP(v Value, indices []Value) (rv Value) {
	ptr, nvals := llvmValueRefs(indices)
	rv.C = C.LLVMConstGEP(v.C, ptr, nvals)
	return
}
func ConstInBoundsGEP(v Value, indices []Value) (rv Value) {
	ptr, nvals := llvmValueRefs(indices)
	rv.C = C.LLVMConstInBoundsGEP(v.C, ptr, nvals)
	return
}
func ConstTrunc(v Value, t Type) (rv Value)         { rv.C = C.LLVMConstTrunc(v.C, t.C); return }
func ConstSExt(v Value, t Type) (rv Value)          { rv.C = C.LLVMConstSExt(v.C, t.C); return }
func ConstZExt(v Value, t Type) (rv Value)          { rv.C = C.LLVMConstZExt(v.C, t.C); return }
func ConstFPTrunc(v Value, t Type) (rv Value)       { rv.C = C.LLVMConstFPTrunc(v.C, t.C); return }
func ConstFPExt(v Value, t Type) (rv Value)         { rv.C = C.LLVMConstFPExt(v.C, t.C); return }
func ConstUIToFP(v Value, t Type) (rv Value)        { rv.C = C.LLVMConstUIToFP(v.C, t.C); return }
func ConstSIToFP(v Value, t Type) (rv Value)        { rv.C = C.LLVMConstSIToFP(v.C, t.C); return }
func ConstFPToUI(v Value, t Type) (rv Value)        { rv.C = C.LLVMConstFPToUI(v.C, t.C); return }
func ConstFPToSI(v Value, t Type) (rv Value)        { rv.C = C.LLVMConstFPToSI(v.C, t.C); return }
func ConstPtrToInt(v Value, t Type) (rv Value)      { rv.C = C.LLVMConstPtrToInt(v.C, t.C); return }
func ConstIntToPtr(v Value, t Type) (rv Value)      { rv.C = C.LLVMConstIntToPtr(v.C, t.C); return }
func ConstBitCast(v Value, t Type) (rv Value)       { rv.C = C.LLVMConstBitCast(v.C, t.C); return }
func ConstZExtOrBitCast(v Value, t Type) (rv Value) { rv.C = C.LLVMConstZExtOrBitCast(v.C, t.C); return }
func ConstSExtOrBitCast(v Value, t Type) (rv Value) { rv.C = C.LLVMConstSExtOrBitCast(v.C, t.C); return }
func ConstTruncOrBitCast(v Value, t Type) (rv Value) {
	rv.C = C.LLVMConstTruncOrBitCast(v.C, t.C)
	return
}
func ConstPointerCast(v Value, t Type) (rv Value) { rv.C = C.LLVMConstPointerCast(v.C, t.C); return }
func ConstIntCast(v Value, t Type, signed bool) (rv Value) {
	rv.C = C.LLVMConstIntCast(v.C, t.C, boolToLLVMBool(signed))
	return
}
func ConstFPCast(v Value, t Type) (rv Value) { rv.C = C.LLVMConstFPCast(v.C, t.C); return }
func ConstSelect(cond, iftrue, iffalse Value) (rv Value) {
	rv.C = C.LLVMConstSelect(cond.C, iftrue.C, iffalse.C)
	return
}
func ConstExtractElement(vec, i Value) (rv Value) {
	rv.C = C.LLVMConstExtractElement(vec.C, i.C)
	return
}
func ConstInsertElement(vec, elem, i Value) (rv Value) {
	rv.C = C.LLVMConstInsertElement(vec.C, elem.C, i.C)
	return
}
func ConstShuffleVector(veca, vecb, mask Value) (rv Value) {
	rv.C = C.LLVMConstShuffleVector(veca.C, vecb.C, mask.C)
	return
}

//TODO
//LLVMValueRef LLVMConstExtractValue(LLVMValueRef AggConstant, unsigned *IdxList,
//                                   unsigned NumIdx);

func ConstExtractValue(agg Value, indices []uint32) (rv Value) {
	n := len(indices)
	if n == 0 {
		panic("one or more indices are required")
	}
	ptr := (*C.unsigned)(&indices[0])
	rv.C = C.LLVMConstExtractValue(agg.C, ptr, C.unsigned(n))
	return
}

func ConstInsertValue(agg, val Value, indices []uint32) (rv Value) {
	n := len(indices)
	if n == 0 {
		panic("one or more indices are required")
	}
	ptr := (*C.unsigned)(&indices[0])
	rv.C = C.LLVMConstInsertValue(agg.C, val.C, ptr, C.unsigned(n))
	return
}

func BlockAddress(f Value, bb BasicBlock) (v Value) {
	v.C = C.LLVMBlockAddress(f.C, bb.C)
	return
}

// Operations on global variables, functions, and aliases (globals)
func (v Value) GlobalParent() (m Module) { m.C = C.LLVMGetGlobalParent(v.C); return }
func (v Value) IsDeclaration() bool      { return C.LLVMIsDeclaration(v.C) != 0 }
func (v Value) Linkage() Linkage         { return Linkage(C.LLVMGetLinkage(v.C)) }
func (v Value) SetLinkage(l Linkage)     { C.LLVMSetLinkage(v.C, C.LLVMLinkage(l)) }
func (v Value) Section() string          { return C.GoString(C.LLVMGetSection(v.C)) }
func (v Value) SetSection(str string) {
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))
	C.LLVMSetSection(v.C, cstr)
}
func (v Value) Visibility() Visibility      { return Visibility(C.LLVMGetVisibility(v.C)) }
func (v Value) SetVisibility(vi Visibility) { C.LLVMSetVisibility(v.C, C.LLVMVisibility(vi)) }
func (v Value) Alignment() int              { return int(C.LLVMGetAlignment(v.C)) }
func (v Value) SetAlignment(a int)          { C.LLVMSetAlignment(v.C, C.unsigned(a)) }
func (v Value) SetUnnamedAddr(ua bool)      { C.LLVMSetUnnamedAddr(v.C, boolToLLVMBool(ua)) }

// Operations on global variables
func AddGlobal(m Module, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMAddGlobal(m.C, t.C, cname)
	return
}
func AddGlobalInAddressSpace(m Module, t Type, name string, addressSpace int) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMAddGlobalInAddressSpace(m.C, t.C, cname, C.unsigned(addressSpace))
	return
}
func (m Module) NamedGlobal(name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMGetNamedGlobal(m.C, cname)
	return
}

func (m Module) FirstGlobal() (v Value)   { v.C = C.LLVMGetFirstGlobal(m.C); return }
func (m Module) LastGlobal() (v Value)    { v.C = C.LLVMGetLastGlobal(m.C); return }
func NextGlobal(v Value) (rv Value)       { rv.C = C.LLVMGetNextGlobal(v.C); return }
func PrevGlobal(v Value) (rv Value)       { rv.C = C.LLVMGetPreviousGlobal(v.C); return }
func (v Value) EraseFromParentAsGlobal()  { C.LLVMDeleteGlobal(v.C) }
func (v Value) Initializer() (rv Value)   { rv.C = C.LLVMGetInitializer(v.C); return }
func (v Value) SetInitializer(cv Value)   { C.LLVMSetInitializer(v.C, cv.C) }
func (v Value) IsThreadLocal() bool       { return C.LLVMIsThreadLocal(v.C) != 0 }
func (v Value) SetThreadLocal(tl bool)    { C.LLVMSetThreadLocal(v.C, boolToLLVMBool(tl)) }
func (v Value) IsGlobalConstant() bool    { return C.LLVMIsGlobalConstant(v.C) != 0 }
func (v Value) SetGlobalConstant(gc bool) { C.LLVMSetGlobalConstant(v.C, boolToLLVMBool(gc)) }

// Operations on aliases
func AddAlias(m Module, t Type, aliasee Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMAddAlias(m.C, t.C, aliasee.C, cname)
	return
}

// Operations on functions
func AddFunction(m Module, name string, ft Type) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMAddFunction(m.C, cname, ft.C)
	return
}

func (m Module) NamedFunction(name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMGetNamedFunction(m.C, cname)
	return
}

func (m Module) FirstFunction() (v Value)  { v.C = C.LLVMGetFirstFunction(m.C); return }
func (m Module) LastFunction() (v Value)   { v.C = C.LLVMGetLastFunction(m.C); return }
func NextFunction(v Value) (rv Value)      { rv.C = C.LLVMGetNextFunction(v.C); return }
func PrevFunction(v Value) (rv Value)      { rv.C = C.LLVMGetPreviousFunction(v.C); return }
func (v Value) EraseFromParentAsFunction() { C.LLVMDeleteFunction(v.C) }
func (v Value) IntrinsicID() int           { return int(C.LLVMGetIntrinsicID(v.C)) }
func (v Value) FunctionCallConv() CallConv {
	return CallConv(C.LLVMCallConv(C.LLVMGetFunctionCallConv(v.C)))
}
func (v Value) SetFunctionCallConv(cc CallConv) { C.LLVMSetFunctionCallConv(v.C, C.unsigned(cc)) }
func (v Value) GC() string                      { return C.GoString(C.LLVMGetGC(v.C)) }
func (v Value) SetGC(name string) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.LLVMSetGC(v.C, cname)
}
func (v Value) AddFunctionAttr(a Attribute)    { C.LLVMAddFunctionAttr2(v.C, C.uint64_t(a)) }
func (v Value) FunctionAttr() Attribute        { return Attribute(C.LLVMGetFunctionAttr2(v.C)) }
func (v Value) RemoveFunctionAttr(a Attribute) { C.LLVMRemoveFunctionAttr2(v.C, C.uint64_t(a)) }
func (v Value) AddTargetDependentFunctionAttr(attr, value string) {
	cattr := C.CString(attr)
	defer C.free(unsafe.Pointer(cattr))
	cvalue := C.CString(value)
	defer C.free(unsafe.Pointer(cvalue))
	C.LLVMAddTargetDependentFunctionAttr(v.C, cattr, cvalue)
}

// Operations on parameters
func (v Value) ParamsCount() int { return int(C.LLVMCountParams(v.C)) }
func (v Value) Params() []Value {
	out := make([]Value, v.ParamsCount())
	if len(out) > 0 {
		C.LLVMGetParams(v.C, llvmValueRefPtr(&out[0]))
	}
	return out
}
func (v Value) Param(i int) (rv Value)  { rv.C = C.LLVMGetParam(v.C, C.unsigned(i)); return }
func (v Value) ParamParent() (rv Value) { rv.C = C.LLVMGetParamParent(v.C); return }
func (v Value) FirstParam() (rv Value)  { rv.C = C.LLVMGetFirstParam(v.C); return }
func (v Value) LastParam() (rv Value)   { rv.C = C.LLVMGetLastParam(v.C); return }
func NextParam(v Value) (rv Value)      { rv.C = C.LLVMGetNextParam(v.C); return }
func PrevParam(v Value) (rv Value)      { rv.C = C.LLVMGetPreviousParam(v.C); return }
func (v Value) AddAttribute(a Attribute) {
	if a >= 1<<32 {
		panic("attribute value currently unsupported")
	}
	C.LLVMAddAttribute(v.C, C.LLVMAttribute(a))
}
func (v Value) RemoveAttribute(a Attribute) {
	if a >= 1<<32 {
		panic("attribute value currently unsupported")
	}
	C.LLVMRemoveAttribute(v.C, C.LLVMAttribute(a))
}
func (v Value) Attribute() Attribute        { return Attribute(C.LLVMGetAttribute(v.C)) }
func (v Value) SetParamAlignment(align int) { C.LLVMSetParamAlignment(v.C, C.unsigned(align)) }

// Operations on basic blocks
func (bb BasicBlock) AsValue() (v Value)      { v.C = C.LLVMBasicBlockAsValue(bb.C); return }
func (v Value) IsBasicBlock() bool            { return C.LLVMValueIsBasicBlock(v.C) != 0 }
func (v Value) AsBasicBlock() (bb BasicBlock) { bb.C = C.LLVMValueAsBasicBlock(v.C); return }
func (bb BasicBlock) Parent() (v Value)       { v.C = C.LLVMGetBasicBlockParent(bb.C); return }
func (v Value) BasicBlocksCount() int         { return int(C.LLVMCountBasicBlocks(v.C)) }
func (v Value) BasicBlocks() []BasicBlock {
	out := make([]BasicBlock, v.BasicBlocksCount())
	C.LLVMGetBasicBlocks(v.C, llvmBasicBlockRefPtr(&out[0]))
	return out
}
func (v Value) FirstBasicBlock() (bb BasicBlock)    { bb.C = C.LLVMGetFirstBasicBlock(v.C); return }
func (v Value) LastBasicBlock() (bb BasicBlock)     { bb.C = C.LLVMGetLastBasicBlock(v.C); return }
func NextBasicBlock(bb BasicBlock) (rbb BasicBlock) { rbb.C = C.LLVMGetNextBasicBlock(bb.C); return }
func PrevBasicBlock(bb BasicBlock) (rbb BasicBlock) { rbb.C = C.LLVMGetPreviousBasicBlock(bb.C); return }
func (v Value) EntryBasicBlock() (bb BasicBlock)    { bb.C = C.LLVMGetEntryBasicBlock(v.C); return }
func (c Context) AddBasicBlock(f Value, name string) (bb BasicBlock) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	bb.C = C.LLVMAppendBasicBlockInContext(c.C, f.C, cname)
	return
}
func (c Context) InsertBasicBlock(ref BasicBlock, name string) (bb BasicBlock) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	bb.C = C.LLVMInsertBasicBlockInContext(c.C, ref.C, cname)
	return
}
func AddBasicBlock(f Value, name string) (bb BasicBlock) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	bb.C = C.LLVMAppendBasicBlock(f.C, cname)
	return
}
func InsertBasicBlock(ref BasicBlock, name string) (bb BasicBlock) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	bb.C = C.LLVMInsertBasicBlock(ref.C, cname)
	return
}
func (bb BasicBlock) EraseFromParent()          { C.LLVMDeleteBasicBlock(bb.C) }
func (bb BasicBlock) MoveBefore(pos BasicBlock) { C.LLVMMoveBasicBlockBefore(bb.C, pos.C) }
func (bb BasicBlock) MoveAfter(pos BasicBlock)  { C.LLVMMoveBasicBlockAfter(bb.C, pos.C) }

// Operations on instructions
func (v Value) InstructionParent() (bb BasicBlock) { bb.C = C.LLVMGetInstructionParent(v.C); return }
func (bb BasicBlock) FirstInstruction() (v Value)  { v.C = C.LLVMGetFirstInstruction(bb.C); return }
func (bb BasicBlock) LastInstruction() (v Value)   { v.C = C.LLVMGetLastInstruction(bb.C); return }
func NextInstruction(v Value) (rv Value)           { rv.C = C.LLVMGetNextInstruction(v.C); return }
func PrevInstruction(v Value) (rv Value)           { rv.C = C.LLVMGetPreviousInstruction(v.C); return }

// Operations on call sites
func (v Value) SetInstructionCallConv(cc CallConv) {
	C.LLVMSetInstructionCallConv(v.C, C.unsigned(cc))
}
func (v Value) InstructionCallConv() CallConv {
	return CallConv(C.LLVMCallConv(C.LLVMGetInstructionCallConv(v.C)))
}
func (v Value) AddInstrAttribute(i int, a Attribute) {
	if a >= 1<<32 {
		panic("attribute value currently unsupported")
	}
	C.LLVMAddInstrAttribute(v.C, C.unsigned(i), C.LLVMAttribute(a))
}
func (v Value) RemoveInstrAttribute(i int, a Attribute) {
	if a >= 1<<32 {
		panic("attribute value currently unsupported")
	}
	C.LLVMRemoveInstrAttribute(v.C, C.unsigned(i), C.LLVMAttribute(a))
}
func (v Value) SetInstrParamAlignment(i int, align int) {
	C.LLVMSetInstrParamAlignment(v.C, C.unsigned(i), C.unsigned(align))
}

// Operations on call instructions (only)
func (v Value) IsTailCall() bool    { return C.LLVMIsTailCall(v.C) != 0 }
func (v Value) SetTailCall(is bool) { C.LLVMSetTailCall(v.C, boolToLLVMBool(is)) }

// Operations on phi nodes
func (v Value) AddIncoming(vals []Value, blocks []BasicBlock) {
	ptr, nvals := llvmValueRefs(vals)
	C.LLVMAddIncoming(v.C, ptr, llvmBasicBlockRefPtr(&blocks[0]), nvals)
}
func (v Value) IncomingCount() int { return int(C.LLVMCountIncoming(v.C)) }
func (v Value) IncomingValue(i int) (rv Value) {
	rv.C = C.LLVMGetIncomingValue(v.C, C.unsigned(i))
	return
}
func (v Value) IncomingBlock(i int) (bb BasicBlock) {
	bb.C = C.LLVMGetIncomingBlock(v.C, C.unsigned(i))
	return
}

//-------------------------------------------------------------------------
// llvm.Builder
//-------------------------------------------------------------------------

// An instruction builder represents a point within a basic block, and is the
// exclusive means of building instructions using the C interface.

func (c Context) NewBuilder() (b Builder) { b.C = C.LLVMCreateBuilderInContext(c.C); return }
func NewBuilder() (b Builder)             { b.C = C.LLVMCreateBuilder(); return }
func (b Builder) SetInsertPoint(block BasicBlock, instr Value) {
	C.LLVMPositionBuilder(b.C, block.C, instr.C)
}
func (b Builder) SetInsertPointBefore(instr Value)     { C.LLVMPositionBuilderBefore(b.C, instr.C) }
func (b Builder) SetInsertPointAtEnd(block BasicBlock) { C.LLVMPositionBuilderAtEnd(b.C, block.C) }
func (b Builder) GetInsertBlock() (bb BasicBlock)      { bb.C = C.LLVMGetInsertBlock(b.C); return }
func (b Builder) ClearInsertionPoint()                 { C.LLVMClearInsertionPosition(b.C) }
func (b Builder) Insert(instr Value)                   { C.LLVMInsertIntoBuilder(b.C, instr.C) }
func (b Builder) InsertWithName(instr Value, name string) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.LLVMInsertIntoBuilderWithName(b.C, instr.C, cname)
}
func (b Builder) Dispose() { C.LLVMDisposeBuilder(b.C) }

// Metadata
func (b Builder) SetCurrentDebugLocation(line, col uint, scope, inlinedAt Metadata) {
	C.LLVMSetCurrentDebugLocation2(b.C, C.unsigned(line), C.unsigned(col), scope.C, inlinedAt.C)
}
func (b Builder) SetInstDebugLocation(v Value)    { C.LLVMSetInstDebugLocation(b.C, v.C) }
func (b Builder) InsertDeclare(module Module, storage Value, md Value) Value {
	f := module.NamedFunction("llvm.dbg.declare")
	if f.IsNil() {
		ftyp := FunctionType(VoidType(), []Type{storage.Type(), md.Type()}, false)
		f = AddFunction(module, "llvm.dbg.declare", ftyp)
	}
	return b.CreateCall(f, []Value{storage, md}, "")
}

// Terminators
func (b Builder) CreateRetVoid() (rv Value)    { rv.C = C.LLVMBuildRetVoid(b.C); return }
func (b Builder) CreateRet(v Value) (rv Value) { rv.C = C.LLVMBuildRet(b.C, v.C); return }
func (b Builder) CreateAggregateRet(vs []Value) (rv Value) {
	ptr, nvals := llvmValueRefs(vs)
	rv.C = C.LLVMBuildAggregateRet(b.C, ptr, nvals)
	return
}
func (b Builder) CreateBr(bb BasicBlock) (rv Value) { rv.C = C.LLVMBuildBr(b.C, bb.C); return }
func (b Builder) CreateCondBr(ifv Value, thenb, elseb BasicBlock) (rv Value) {
	rv.C = C.LLVMBuildCondBr(b.C, ifv.C, thenb.C, elseb.C)
	return
}
func (b Builder) CreateSwitch(v Value, elseb BasicBlock, numCases int) (rv Value) {
	rv.C = C.LLVMBuildSwitch(b.C, v.C, elseb.C, C.unsigned(numCases))
	return
}
func (b Builder) CreateIndirectBr(addr Value, numDests int) (rv Value) {
	rv.C = C.LLVMBuildIndirectBr(b.C, addr.C, C.unsigned(numDests))
	return
}
func (b Builder) CreateInvoke(fn Value, args []Value, then, catch BasicBlock, name string) (rv Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	ptr, nvals := llvmValueRefs(args)
	rv.C = C.LLVMBuildInvoke(b.C, fn.C, ptr, nvals, then.C, catch.C, cname)
	return
}
func (b Builder) CreateUnreachable() (rv Value) { rv.C = C.LLVMBuildUnreachable(b.C); return }

// Add a case to the switch instruction
func (v Value) AddCase(on Value, dest BasicBlock) { C.LLVMAddCase(v.C, on.C, dest.C) }

// Add a destination to the indirectbr instruction
func (v Value) AddDest(dest BasicBlock) { C.LLVMAddDestination(v.C, dest.C) }

// Arithmetic
func (b Builder) CreateAdd(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildAdd(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateNSWAdd(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildNSWAdd(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateNUWAdd(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildNUWAdd(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateFAdd(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildFAdd(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateSub(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildSub(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateNSWSub(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildNSWSub(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateNUWSub(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildNUWSub(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateFSub(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	v.C = C.LLVMBuildFSub(b.C, lhs.C, rhs.C, cname)
	C.free(unsafe.Pointer(cname))
	return
}
func (b Builder) CreateMul(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildMul(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateNSWMul(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildNSWMul(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateNUWMul(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildNUWMul(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateFMul(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildFMul(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateUDiv(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildUDiv(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateSDiv(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildSDiv(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateExactSDiv(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildExactSDiv(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateFDiv(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildFDiv(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateURem(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildURem(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateSRem(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildSRem(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateFRem(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildFRem(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateShl(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildShl(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateLShr(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildLShr(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateAShr(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildAShr(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateAnd(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildAnd(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateOr(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildOr(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateXor(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildXor(b.C, lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateBinOp(op Opcode, lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildBinOp(b.C, C.LLVMOpcode(op), lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateNeg(v Value, name string) (rv Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	rv.C = C.LLVMBuildNeg(b.C, v.C, cname)
	return
}
func (b Builder) CreateNSWNeg(v Value, name string) (rv Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	rv.C = C.LLVMBuildNSWNeg(b.C, v.C, cname)
	return
}
func (b Builder) CreateNUWNeg(v Value, name string) (rv Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	rv.C = C.LLVMBuildNUWNeg(b.C, v.C, cname)
	return
}
func (b Builder) CreateFNeg(v Value, name string) (rv Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	rv.C = C.LLVMBuildFNeg(b.C, v.C, cname)
	return
}
func (b Builder) CreateNot(v Value, name string) (rv Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	rv.C = C.LLVMBuildNot(b.C, v.C, cname)
	return
}

// Memory

func (b Builder) CreateMalloc(t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildMalloc(b.C, t.C, cname)
	return
}
func (b Builder) CreateArrayMalloc(t Type, val Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildArrayMalloc(b.C, t.C, val.C, cname)
	return
}
func (b Builder) CreateAlloca(t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildAlloca(b.C, t.C, cname)
	return
}
func (b Builder) CreateArrayAlloca(t Type, val Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildArrayAlloca(b.C, t.C, val.C, cname)
	return
}
func (b Builder) CreateFree(p Value) (v Value) {
	v.C = C.LLVMBuildFree(b.C, p.C)
	return
}
func (b Builder) CreateLoad(p Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildLoad(b.C, p.C, cname)
	return
}
func (b Builder) CreateStore(val Value, p Value) (v Value) {
	v.C = C.LLVMBuildStore(b.C, val.C, p.C)
	return
}
func (b Builder) CreateGEP(p Value, indices []Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	ptr, nvals := llvmValueRefs(indices)
	v.C = C.LLVMBuildGEP(b.C, p.C, ptr, nvals, cname)
	return
}
func (b Builder) CreateInBoundsGEP(p Value, indices []Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	ptr, nvals := llvmValueRefs(indices)
	v.C = C.LLVMBuildInBoundsGEP(b.C, p.C, ptr, nvals, cname)
	return
}
func (b Builder) CreateStructGEP(p Value, i int, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildStructGEP(b.C, p.C, C.unsigned(i), cname)
	return
}
func (b Builder) CreateGlobalString(str, name string) (v Value) {
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildGlobalString(b.C, cstr, cname)
	return
}
func (b Builder) CreateGlobalStringPtr(str, name string) (v Value) {
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildGlobalStringPtr(b.C, cstr, cname)
	return
}

// Casts
func (b Builder) CreateTrunc(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildTrunc(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateZExt(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildZExt(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateSExt(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildSExt(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateFPToUI(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildFPToUI(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateFPToSI(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildFPToSI(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateUIToFP(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildUIToFP(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateSIToFP(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildSIToFP(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateFPTrunc(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildFPTrunc(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateFPExt(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildFPExt(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreatePtrToInt(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildPtrToInt(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateIntToPtr(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildIntToPtr(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateBitCast(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildBitCast(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateZExtOrBitCast(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildZExtOrBitCast(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateSExtOrBitCast(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildSExtOrBitCast(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateTruncOrBitCast(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildTruncOrBitCast(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateCast(val Value, op Opcode, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildCast(b.C, C.LLVMOpcode(op), val.C, t.C, cname)
	return
} //
func (b Builder) CreatePointerCast(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildPointerCast(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateIntCast(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildIntCast(b.C, val.C, t.C, cname)
	return
}
func (b Builder) CreateFPCast(val Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildFPCast(b.C, val.C, t.C, cname)
	return
}

// Comparisons
func (b Builder) CreateICmp(pred IntPredicate, lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildICmp(b.C, C.LLVMIntPredicate(pred), lhs.C, rhs.C, cname)
	return
}
func (b Builder) CreateFCmp(pred FloatPredicate, lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildFCmp(b.C, C.LLVMRealPredicate(pred), lhs.C, rhs.C, cname)
	return
}

// Miscellaneous instructions
func (b Builder) CreatePHI(t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildPhi(b.C, t.C, cname)
	return
}
func (b Builder) CreateCall(fn Value, args []Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	ptr, nvals := llvmValueRefs(args)
	v.C = C.LLVMBuildCall(b.C, fn.C, ptr, nvals, cname)
	return
}

func (b Builder) CreateSelect(ifv, thenv, elsev Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildSelect(b.C, ifv.C, thenv.C, elsev.C, cname)
	return
}

func (b Builder) CreateVAArg(list Value, t Type, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildVAArg(b.C, list.C, t.C, cname)
	return
}
func (b Builder) CreateExtractElement(vec, i Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildExtractElement(b.C, vec.C, i.C, cname)
	return
}
func (b Builder) CreateInsertElement(vec, elt, i Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildInsertElement(b.C, vec.C, elt.C, i.C, cname)
	return
}
func (b Builder) CreateShuffleVector(v1, v2, mask Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildShuffleVector(b.C, v1.C, v2.C, mask.C, cname)
	return
}
func (b Builder) CreateExtractValue(agg Value, i int, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildExtractValue(b.C, agg.C, C.unsigned(i), cname)
	return
}
func (b Builder) CreateInsertValue(agg, elt Value, i int, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildInsertValue(b.C, agg.C, elt.C, C.unsigned(i), cname)
	return
}

func (b Builder) CreateIsNull(val Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildIsNull(b.C, val.C, cname)
	return
}
func (b Builder) CreateIsNotNull(val Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildIsNotNull(b.C, val.C, cname)
	return
}
func (b Builder) CreatePtrDiff(lhs, rhs Value, name string) (v Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	v.C = C.LLVMBuildPtrDiff(b.C, lhs.C, rhs.C, cname)
	return
}

func (b Builder) CreateLandingPad(t Type, personality Value, nclauses int, name string) (l Value) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	l.C = C.LLVMBuildLandingPad(b.C, t.C, nil, C.unsigned(nclauses), cname)
	return l
}

func (l Value) AddClause(v Value) {
	C.LLVMAddClause(l.C, v.C)
}

func (l Value) SetCleanup(cleanup bool) {
	C.LLVMSetCleanup(l.C, boolToLLVMBool(cleanup))
}

func (b Builder) CreateResume(ex Value) (v Value) {
	v.C = C.LLVMBuildResume(b.C, ex.C)
	return
}

//-------------------------------------------------------------------------
// llvm.ModuleProvider
//-------------------------------------------------------------------------

// Changes the type of M so it can be passed to FunctionPassManagers and the
// JIT. They take ModuleProviders for historical reasons.
func NewModuleProviderForModule(m Module) (mp ModuleProvider) {
	mp.C = C.LLVMCreateModuleProviderForExistingModule(m.C)
	return
}

// Destroys the module M.
func (mp ModuleProvider) Dispose() { C.LLVMDisposeModuleProvider(mp.C) }

//-------------------------------------------------------------------------
// llvm.MemoryBuffer
//-------------------------------------------------------------------------

func NewMemoryBufferFromFile(path string) (b MemoryBuffer, err error) {
	var cmsg *C.char
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	fail := C.LLVMCreateMemoryBufferWithContentsOfFile(cpath, &b.C, &cmsg)
	if fail != 0 {
		b.C = nil
		err = errors.New(C.GoString(cmsg))
		C.LLVMDisposeMessage(cmsg)
	}
	return
}

func NewMemoryBufferFromStdin() (b MemoryBuffer, err error) {
	var cmsg *C.char
	fail := C.LLVMCreateMemoryBufferWithSTDIN(&b.C, &cmsg)
	if fail != 0 {
		b.C = nil
		err = errors.New(C.GoString(cmsg))
		C.LLVMDisposeMessage(cmsg)
	}
	return
}

func (b MemoryBuffer) Bytes() []byte {
	cstart := C.LLVMGetBufferStart(b.C)
	csize := C.LLVMGetBufferSize(b.C)
	return C.GoBytes(unsafe.Pointer(cstart), C.int(csize))
}

func (b MemoryBuffer) Dispose() { C.LLVMDisposeMemoryBuffer(b.C) }

//-------------------------------------------------------------------------
// llvm.PassManager
//-------------------------------------------------------------------------

// Constructs a new whole-module pass pipeline. This type of pipeline is
// suitable for link-time optimization and whole-module transformations.
// See llvm::PassManager::PassManager.
func NewPassManager() (pm PassManager) { pm.C = C.LLVMCreatePassManager(); return }

// Constructs a new function-by-function pass pipeline over the module
// provider. It does not take ownership of the module provider. This type of
// pipeline is suitable for code generation and JIT compilation tasks.
// See llvm::FunctionPassManager::FunctionPassManager.
func NewFunctionPassManagerForModule(m Module) (pm PassManager) {
	pm.C = C.LLVMCreateFunctionPassManagerForModule(m.C)
	return
}

// Initializes, executes on the provided module, and finalizes all of the
// passes scheduled in the pass manager. Returns 1 if any of the passes
// modified the module, 0 otherwise. See llvm::PassManager::run(Module&).
func (pm PassManager) Run(m Module) bool { return C.LLVMRunPassManager(pm.C, m.C) != 0 }

// Initializes all of the function passes scheduled in the function pass
// manager. Returns 1 if any of the passes modified the module, 0 otherwise.
// See llvm::FunctionPassManager::doInitialization.
func (pm PassManager) InitializeFunc() bool { return C.LLVMInitializeFunctionPassManager(pm.C) != 0 }

// Executes all of the function passes scheduled in the function pass manager
// on the provided function. Returns 1 if any of the passes modified the
// function, false otherwise.
// See llvm::FunctionPassManager::run(Function&).
func (pm PassManager) RunFunc(f Value) bool { return C.LLVMRunFunctionPassManager(pm.C, f.C) != 0 }

// Finalizes all of the function passes scheduled in in the function pass
// manager. Returns 1 if any of the passes modified the module, 0 otherwise.
// See llvm::FunctionPassManager::doFinalization.
func (pm PassManager) FinalizeFunc() bool { return C.LLVMFinalizeFunctionPassManager(pm.C) != 0 }

// Frees the memory of a pass pipeline. For function pipelines, does not free
// the module provider.
// See llvm::PassManagerBase::~PassManagerBase.
func (pm PassManager) Dispose() { C.LLVMDisposePassManager(pm.C) }

//-------------------------------------------------------------------------
// llvm.Metadata
//-------------------------------------------------------------------------

func (md Metadata) ReplaceAllUsesWith(new Metadata) {
	C.LLVMMetadataReplaceAllUsesWith(md.C, new.C)
}
