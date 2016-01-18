//===- target.go - Bindings for target ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines bindings for the target component.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"
import "errors"

type (
	TargetData struct {
		C C.LLVMTargetDataRef
	}
	Target struct {
		C C.LLVMTargetRef
	}
	TargetMachine struct {
		C C.LLVMTargetMachineRef
	}
	ByteOrdering    C.enum_LLVMByteOrdering
	RelocMode       C.LLVMRelocMode
	CodeGenOptLevel C.LLVMCodeGenOptLevel
	CodeGenFileType C.LLVMCodeGenFileType
	CodeModel       C.LLVMCodeModel
)

const (
	BigEndian    ByteOrdering = C.LLVMBigEndian
	LittleEndian ByteOrdering = C.LLVMLittleEndian
)

const (
	RelocDefault      RelocMode = C.LLVMRelocDefault
	RelocStatic       RelocMode = C.LLVMRelocStatic
	RelocPIC          RelocMode = C.LLVMRelocPIC
	RelocDynamicNoPic RelocMode = C.LLVMRelocDynamicNoPic
)

const (
	CodeGenLevelNone       CodeGenOptLevel = C.LLVMCodeGenLevelNone
	CodeGenLevelLess       CodeGenOptLevel = C.LLVMCodeGenLevelLess
	CodeGenLevelDefault    CodeGenOptLevel = C.LLVMCodeGenLevelDefault
	CodeGenLevelAggressive CodeGenOptLevel = C.LLVMCodeGenLevelAggressive
)

const (
	CodeModelDefault    CodeModel = C.LLVMCodeModelDefault
	CodeModelJITDefault CodeModel = C.LLVMCodeModelJITDefault
	CodeModelSmall      CodeModel = C.LLVMCodeModelSmall
	CodeModelKernel     CodeModel = C.LLVMCodeModelKernel
	CodeModelMedium     CodeModel = C.LLVMCodeModelMedium
	CodeModelLarge      CodeModel = C.LLVMCodeModelLarge
)

const (
	AssemblyFile CodeGenFileType = C.LLVMAssemblyFile
	ObjectFile   CodeGenFileType = C.LLVMObjectFile
)

// InitializeAllTargetInfos - The main program should call this function if it
// wants access to all available targets that LLVM is configured to support.
func InitializeAllTargetInfos() { C.LLVMInitializeAllTargetInfos() }

// InitializeAllTargets - The main program should call this function if it wants
// to link in all available targets that LLVM is configured to support.
func InitializeAllTargets() { C.LLVMInitializeAllTargets() }

func InitializeAllTargetMCs() { C.LLVMInitializeAllTargetMCs() }

func InitializeAllAsmParsers() { C.LLVMInitializeAllAsmParsers() }

func InitializeAllAsmPrinters() { C.LLVMInitializeAllAsmPrinters() }

var initializeNativeTargetError = errors.New("Failed to initialize native target")

// InitializeNativeTarget - The main program should call this function to
// initialize the native target corresponding to the host. This is useful
// for JIT applications to ensure that the target gets linked in correctly.
func InitializeNativeTarget() error {
	fail := C.LLVMInitializeNativeTarget()
	if fail != 0 {
		return initializeNativeTargetError
	}
	return nil
}

func InitializeNativeAsmPrinter() error {
	fail := C.LLVMInitializeNativeAsmPrinter()
	if fail != 0 {
		return initializeNativeTargetError
	}
	return nil
}

//-------------------------------------------------------------------------
// llvm.TargetData
//-------------------------------------------------------------------------

// Creates target data from a target layout string.
// See the constructor llvm::TargetData::TargetData.
func NewTargetData(rep string) (td TargetData) {
	crep := C.CString(rep)
	defer C.free(unsafe.Pointer(crep))
	td.C = C.LLVMCreateTargetData(crep)
	return
}

// Adds target data information to a pass manager. This does not take ownership
// of the target data.
// See the method llvm::PassManagerBase::add.
func (pm PassManager) Add(td TargetData) {
	C.LLVMAddTargetData(td.C, pm.C)
}

// Converts target data to a target layout string. The string must be disposed
// with LLVMDisposeMessage.
// See the constructor llvm::TargetData::TargetData.
func (td TargetData) String() (s string) {
	cmsg := C.LLVMCopyStringRepOfTargetData(td.C)
	s = C.GoString(cmsg)
	C.LLVMDisposeMessage(cmsg)
	return
}

// Returns the byte order of a target, either BigEndian or LittleEndian.
// See the method llvm::TargetData::isLittleEndian.
func (td TargetData) ByteOrder() ByteOrdering { return ByteOrdering(C.LLVMByteOrder(td.C)) }

// Returns the pointer size in bytes for a target.
// See the method llvm::TargetData::getPointerSize.
func (td TargetData) PointerSize() int { return int(C.LLVMPointerSize(td.C)) }

// Returns the integer type that is the same size as a pointer on a target.
// See the method llvm::TargetData::getIntPtrType.
func (td TargetData) IntPtrType() (t Type) { t.C = C.LLVMIntPtrType(td.C); return }

// Computes the size of a type in bytes for a target.
// See the method llvm::TargetData::getTypeSizeInBits.
func (td TargetData) TypeSizeInBits(t Type) uint64 {
	return uint64(C.LLVMSizeOfTypeInBits(td.C, t.C))
}

// Computes the storage size of a type in bytes for a target.
// See the method llvm::TargetData::getTypeStoreSize.
func (td TargetData) TypeStoreSize(t Type) uint64 {
	return uint64(C.LLVMStoreSizeOfType(td.C, t.C))
}

// Computes the ABI size of a type in bytes for a target.
// See the method llvm::TargetData::getTypeAllocSize.
func (td TargetData) TypeAllocSize(t Type) uint64 {
	return uint64(C.LLVMABISizeOfType(td.C, t.C))
}

// Computes the ABI alignment of a type in bytes for a target.
// See the method llvm::TargetData::getABITypeAlignment.
func (td TargetData) ABITypeAlignment(t Type) int {
	return int(C.LLVMABIAlignmentOfType(td.C, t.C))
}

// Computes the call frame alignment of a type in bytes for a target.
// See the method llvm::TargetData::getCallFrameTypeAlignment.
func (td TargetData) CallFrameTypeAlignment(t Type) int {
	return int(C.LLVMCallFrameAlignmentOfType(td.C, t.C))
}

// Computes the preferred alignment of a type in bytes for a target.
// See the method llvm::TargetData::getPrefTypeAlignment.
func (td TargetData) PrefTypeAlignment(t Type) int {
	return int(C.LLVMPreferredAlignmentOfType(td.C, t.C))
}

// Computes the preferred alignment of a global variable in bytes for a target.
// See the method llvm::TargetData::getPreferredAlignment.
func (td TargetData) PreferredAlignment(g Value) int {
	return int(C.LLVMPreferredAlignmentOfGlobal(td.C, g.C))
}

// Computes the structure element that contains the byte offset for a target.
// See the method llvm::StructLayout::getElementContainingOffset.
func (td TargetData) ElementContainingOffset(t Type, offset uint64) int {
	return int(C.LLVMElementAtOffset(td.C, t.C, C.ulonglong(offset)))
}

// Computes the byte offset of the indexed struct element for a target.
// See the method llvm::StructLayout::getElementOffset.
func (td TargetData) ElementOffset(t Type, element int) uint64 {
	return uint64(C.LLVMOffsetOfElement(td.C, t.C, C.unsigned(element)))
}

// Deallocates a TargetData.
// See the destructor llvm::TargetData::~TargetData.
func (td TargetData) Dispose() { C.LLVMDisposeTargetData(td.C) }

//-------------------------------------------------------------------------
// llvm.Target
//-------------------------------------------------------------------------

func FirstTarget() Target {
	return Target{C.LLVMGetFirstTarget()}
}

func (t Target) NextTarget() Target {
	return Target{C.LLVMGetNextTarget(t.C)}
}

func GetTargetFromTriple(triple string) (t Target, err error) {
	var errstr *C.char
	ctriple := C.CString(triple)
	defer C.free(unsafe.Pointer(ctriple))
	fail := C.LLVMGetTargetFromTriple(ctriple, &t.C, &errstr)
	if fail != 0 {
		err = errors.New(C.GoString(errstr))
		C.free(unsafe.Pointer(errstr))
	}
	return
}

func (t Target) Name() string {
	return C.GoString(C.LLVMGetTargetName(t.C))
}

func (t Target) Description() string {
	return C.GoString(C.LLVMGetTargetDescription(t.C))
}

//-------------------------------------------------------------------------
// llvm.TargetMachine
//-------------------------------------------------------------------------

// CreateTargetMachine creates a new TargetMachine.
func (t Target) CreateTargetMachine(Triple string, CPU string, Features string,
	Level CodeGenOptLevel, Reloc RelocMode,
	CodeModel CodeModel) (tm TargetMachine) {
	cTriple := C.CString(Triple)
	defer C.free(unsafe.Pointer(cTriple))
	cCPU := C.CString(CPU)
	defer C.free(unsafe.Pointer(cCPU))
	cFeatures := C.CString(Features)
	defer C.free(unsafe.Pointer(cFeatures))
	tm.C = C.LLVMCreateTargetMachine(t.C, cTriple, cCPU, cFeatures,
		C.LLVMCodeGenOptLevel(Level),
		C.LLVMRelocMode(Reloc),
		C.LLVMCodeModel(CodeModel))
	return
}

// Triple returns the triple describing the machine (arch-vendor-os).
func (tm TargetMachine) Triple() string {
	cstr := C.LLVMGetTargetMachineTriple(tm.C)
	return C.GoString(cstr)
}

// TargetData returns the TargetData for the machine.
func (tm TargetMachine) TargetData() TargetData {
	return TargetData{C.LLVMGetTargetMachineData(tm.C)}
}

func (tm TargetMachine) EmitToMemoryBuffer(m Module, ft CodeGenFileType) (MemoryBuffer, error) {
	var errstr *C.char
	var mb MemoryBuffer
	fail := C.LLVMTargetMachineEmitToMemoryBuffer(tm.C, m.C, C.LLVMCodeGenFileType(ft), &errstr, &mb.C)
	if fail != 0 {
		err := errors.New(C.GoString(errstr))
		C.free(unsafe.Pointer(errstr))
		return MemoryBuffer{}, err
	}
	return mb, nil
}

func (tm TargetMachine) AddAnalysisPasses(pm PassManager) {
	C.LLVMAddAnalysisPasses(tm.C, pm.C)
}

// Dispose releases resources related to the TargetMachine.
func (tm TargetMachine) Dispose() {
	C.LLVMDisposeTargetMachine(tm.C)
}

func DefaultTargetTriple() (triple string) {
	cTriple := C.LLVMGetDefaultTargetTriple()
	defer C.free(unsafe.Pointer(cTriple))
	triple = C.GoString(cTriple)
	return
}
