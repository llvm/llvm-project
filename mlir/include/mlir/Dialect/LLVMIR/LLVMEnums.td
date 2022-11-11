//===-- LLVMEnums.td - LLVM IR dialect enum file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_ENUMS_TD
#define LLVMIR_ENUMS_TD

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// AsmDialect
//===----------------------------------------------------------------------===//

def AsmATT : LLVM_EnumAttrCase<
  /*string cppSym=*/"AD_ATT", /*string irSym=*/"att",
  /*string llvmSym=*/"AD_ATT", /*int val=*/0>;
def AsmIntel : LLVM_EnumAttrCase<
  /*string cppSym=*/"AD_Intel", /*string irSym=*/"intel",
  /*string llvmSym=*/"AD_Intel", /*int val=*/1>;
def AsmATTOrIntel : LLVM_EnumAttr<
  /*string name=*/"AsmDialect",
  /*string llvmName=*/"::llvm::InlineAsm::AsmDialect",
  /*string description=*/"ATT (0) or Intel (1) asm dialect",
  /*list<LLVM_EnumAttrCase> cases=*/[AsmATT, AsmIntel]> {
  let cppNamespace = "::mlir::LLVM";
}

//===----------------------------------------------------------------------===//
// Atomic Operations
//===----------------------------------------------------------------------===//

def AtomicBinOpXchg : I64EnumAttrCase<"xchg", 0>;
def AtomicBinOpAdd  : I64EnumAttrCase<"add", 1>;
def AtomicBinOpSub  : I64EnumAttrCase<"sub", 2>;
def AtomicBinOpAnd  : I64EnumAttrCase<"_and", 3>;
def AtomicBinOpNand : I64EnumAttrCase<"nand", 4>;
def AtomicBinOpOr   : I64EnumAttrCase<"_or", 5>;
def AtomicBinOpXor  : I64EnumAttrCase<"_xor", 6>;
def AtomicBinOpMax  : I64EnumAttrCase<"max", 7>;
def AtomicBinOpMin  : I64EnumAttrCase<"min", 8>;
def AtomicBinOpUMax : I64EnumAttrCase<"umax", 9>;
def AtomicBinOpUMin : I64EnumAttrCase<"umin", 10>;
def AtomicBinOpFAdd : I64EnumAttrCase<"fadd", 11>;
def AtomicBinOpFSub : I64EnumAttrCase<"fsub", 12>;
def AtomicBinOp : I64EnumAttr<
    "AtomicBinOp",
    "llvm.atomicrmw binary operations",
    [AtomicBinOpXchg, AtomicBinOpAdd, AtomicBinOpSub, AtomicBinOpAnd,
     AtomicBinOpNand, AtomicBinOpOr, AtomicBinOpXor, AtomicBinOpMax,
     AtomicBinOpMin, AtomicBinOpUMax, AtomicBinOpUMin, AtomicBinOpFAdd,
     AtomicBinOpFSub]> {
  let cppNamespace = "::mlir::LLVM";
}

def AtomicOrderingNotAtomic              : I64EnumAttrCase<"not_atomic", 0>;
def AtomicOrderingUnordered              : I64EnumAttrCase<"unordered", 1>;
def AtomicOrderingMonotonic              : I64EnumAttrCase<"monotonic", 2>;
def AtomicOrderingAcquire                : I64EnumAttrCase<"acquire", 4>;
def AtomicOrderingRelease                : I64EnumAttrCase<"release", 5>;
def AtomicOrderingAcquireRelease         : I64EnumAttrCase<"acq_rel", 6>;
def AtomicOrderingSequentiallyConsistent : I64EnumAttrCase<"seq_cst", 7>;
def AtomicOrdering : I64EnumAttr<
    "AtomicOrdering",
    "Atomic ordering for LLVM's memory model",
    [AtomicOrderingNotAtomic, AtomicOrderingUnordered, AtomicOrderingMonotonic,
     AtomicOrderingAcquire, AtomicOrderingRelease, AtomicOrderingAcquireRelease,
     AtomicOrderingSequentiallyConsistent]> {
  let cppNamespace = "::mlir::LLVM";
}

//===----------------------------------------------------------------------===//
// CallingConvention
//===----------------------------------------------------------------------===//

// These values must match llvm::CallingConv ones.
// See https://llvm.org/doxygen/namespacellvm_1_1CallingConv.html for full list
// of supported calling conventions.
def CConvC : LLVM_EnumAttrCase<"C", "ccc", "C", 0>;
def CConvFast : LLVM_EnumAttrCase<"Fast", "fastcc", "Fast", 8>;
def CConvCold : LLVM_EnumAttrCase<"Cold", "coldcc", "Cold", 9>;
def CConvGHC : LLVM_EnumAttrCase<"GHC", "cc_10", "GHC", 10>;
def CConvHiPE : LLVM_EnumAttrCase<"HiPE", "cc_11", "HiPE", 11>;
def CConvWebKitJS : LLVM_EnumAttrCase<"WebKit_JS", "webkit_jscc",
                                      "WebKit_JS", 12>;
def CConvAnyReg : LLVM_EnumAttrCase<"AnyReg", "anyregcc", "AnyReg", 13>;
def CConvPreserveMost : LLVM_EnumAttrCase<"PreserveMost", "preserve_mostcc",
                                          "PreserveMost", 14>;
def CConvPreserveAll : LLVM_EnumAttrCase<"PreserveAll", "preserve_allcc",
                                         "PreserveAll", 15>;
def CConvSwift : LLVM_EnumAttrCase<"Swift", "swiftcc", "Swift", 16>;
def CConvCXXFastTLS : LLVM_EnumAttrCase<"CXX_FAST_TLS", "cxx_fast_tlscc",
                                        "CXX_FAST_TLS", 17>;
def CConvTail : LLVM_EnumAttrCase<"Tail", "tailcc", "Tail", 18>;
def CConvCFGuard_Check : LLVM_EnumAttrCase<"CFGuard_Check",
                                            "cfguard_checkcc",
                                            "CFGuard_Check", 19>;
def CConvSwiftTail : LLVM_EnumAttrCase<"SwiftTail", "swifttailcc",
                                       "SwiftTail", 20>;
def CConvX86_StdCall : LLVM_EnumAttrCase<"X86_StdCall", "x86_stdcallcc",
                                         "X86_StdCall", 64>;
def CConvX86_FastCall : LLVM_EnumAttrCase<"X86_FastCall", "x86_fastcallcc",
                                          "X86_FastCall", 65>;
def CConvARM_APCS : LLVM_EnumAttrCase<"ARM_APCS", "arm_apcscc", "ARM_APCS", 66>;
def CConvARM_AAPCS : LLVM_EnumAttrCase<"ARM_AAPCS", "arm_aapcscc", "ARM_AAPCS",
                                       67>;
def CConvARM_AAPCS_VFP : LLVM_EnumAttrCase<"ARM_AAPCS_VFP", "arm_aapcs_vfpcc",
                                           "ARM_AAPCS_VFP", 68>;
def CConvMSP430_INTR : LLVM_EnumAttrCase<"MSP430_INTR", "msp430_intrcc",
                                          "MSP430_INTR", 69>;
def CConvX86_ThisCall : LLVM_EnumAttrCase<"X86_ThisCall", "x86_thiscallcc",
                                          "X86_ThisCall", 70>;
def CConvPTX_Kernel : LLVM_EnumAttrCase<"PTX_Kernel", "ptx_kernelcc",
                                        "PTX_Kernel", 71>;
def CConvPTX_Device : LLVM_EnumAttrCase<"PTX_Device", "ptx_devicecc",
                                        "PTX_Device", 72>;
def CConvSPIR_FUNC : LLVM_EnumAttrCase<"SPIR_FUNC", "spir_funccc",
                                       "SPIR_FUNC", 75>;
def CConvSPIR_KERNEL : LLVM_EnumAttrCase<"SPIR_KERNEL", "spir_kernelcc",
                                         "SPIR_KERNEL", 76>;
def CConvIntel_OCL_BI : LLVM_EnumAttrCase<"Intel_OCL_BI", "intel_ocl_bicc",
                                          "Intel_OCL_BI", 77>;
def CConvX86_64_SysV : LLVM_EnumAttrCase<"X86_64_SysV", "x86_64_sysvcc",
                                         "X86_64_SysV", 78>;
def CConvWin64 : LLVM_EnumAttrCase<"Win64", "win64cc", "Win64", 79>;
def CConvX86_VectorCall : LLVM_EnumAttrCase<"X86_VectorCall",
                                            "x86_vectorcallcc",
                                            "X86_VectorCall", 80>;
def CConvHHVM : LLVM_EnumAttrCase<"HHVM", "hhvmcc", "HHVM", 81>;
def CConvHHVM_C : LLVM_EnumAttrCase<"HHVM_C", "hhvm_ccc", "HHVM_C", 82>;
def CConvX86_INTR : LLVM_EnumAttrCase<"X86_INTR", "x86_intrcc", "X86_INTR", 83>;
def CConvAVR_INTR : LLVM_EnumAttrCase<"AVR_INTR", "avr_intrcc", "AVR_INTR", 84>;
def CConvAVR_SIGNAL : LLVM_EnumAttrCase<"AVR_SIGNAL", "avr_signalcc",
                                        "AVR_SIGNAL", 85>;
def CConvAVR_BUILTIN : LLVM_EnumAttrCase<"AVR_BUILTIN", "avr_builtincc",
                                         "AVR_BUILTIN", 86>;
def CConvAMDGPU_VS : LLVM_EnumAttrCase<"AMDGPU_VS", "amdgpu_vscc", "AMDGPU_VS",
                                       87>;
def CConvAMDGPU_GS : LLVM_EnumAttrCase<"AMDGPU_GS", "amdgpu_gscc", "AMDGPU_GS",
                                       88>;
def CConvAMDGPU_PS : LLVM_EnumAttrCase<"AMDGPU_PS", "amdgpu_pscc", "AMDGPU_PS",
                                       89>;
def CConvAMDGPU_CS : LLVM_EnumAttrCase<"AMDGPU_CS", "amdgpu_cscc", "AMDGPU_CS",
                                       90>;
def CConvAMDGPU_KERNEL : LLVM_EnumAttrCase<"AMDGPU_KERNEL", "amdgpu_kernelcc",
                                           "AMDGPU_KERNEL", 91>;
def CConvX86_RegCall : LLVM_EnumAttrCase<"X86_RegCall", "x86_regcallcc",
                                         "X86_RegCall", 92>;
def CConvAMDGPU_HS : LLVM_EnumAttrCase<"AMDGPU_HS", "amdgpu_hscc", "AMDGPU_HS",
                                       93>;
def CConvMSP430_BUILTIN : LLVM_EnumAttrCase<"MSP430_BUILTIN",
                                             "msp430_builtincc",
                                             "MSP430_BUILTIN", 94>;
def CConvAMDGPU_LS : LLVM_EnumAttrCase<"AMDGPU_LS", "amdgpu_lscc", "AMDGPU_LS",
                                       95>;
def CConvAMDGPU_ES : LLVM_EnumAttrCase<"AMDGPU_ES", "amdgpu_escc", "AMDGPU_ES",
                                       96>;
def CConvAArch64_VectorCall : LLVM_EnumAttrCase<"AArch64_VectorCall",
                                                "aarch64_vectorcallcc",
                                                "AArch64_VectorCall", 97>;
def CConvAArch64_SVE_VectorCall : LLVM_EnumAttrCase<"AArch64_SVE_VectorCall",
                                                    "aarch64_sve_vectorcallcc",
                                                    "AArch64_SVE_VectorCall",
                                                    98>;
def CConvWASM_EmscriptenInvoke : LLVM_EnumAttrCase<"WASM_EmscriptenInvoke",
                                                   "wasm_emscripten_invokecc",
                                                   "WASM_EmscriptenInvoke", 99>;
def CConvAMDGPU_Gfx : LLVM_EnumAttrCase<"AMDGPU_Gfx", "amdgpu_gfxcc",
                                        "AMDGPU_Gfx", 100>;
def CConvM68k_INTR : LLVM_EnumAttrCase<"M68k_INTR", "m68k_intrcc", "M68k_INTR",
                                       101>;

def CConvEnum : LLVM_CEnumAttr<
    "CConv",
    "::llvm::CallingConv",
    "Calling Conventions",
    [CConvC, CConvFast, CConvCold, CConvGHC, CConvHiPE, CConvWebKitJS,
     CConvAnyReg, CConvPreserveMost, CConvPreserveAll, CConvSwift,
     CConvCXXFastTLS, CConvTail, CConvCFGuard_Check, CConvSwiftTail,
     CConvX86_StdCall, CConvX86_FastCall, CConvARM_APCS,
     CConvARM_AAPCS, CConvARM_AAPCS_VFP, CConvMSP430_INTR, CConvX86_ThisCall,
     CConvPTX_Kernel, CConvPTX_Device, CConvSPIR_FUNC, CConvSPIR_KERNEL,
     CConvIntel_OCL_BI, CConvX86_64_SysV, CConvWin64, CConvX86_VectorCall,
     CConvHHVM, CConvHHVM_C, CConvX86_INTR, CConvAVR_INTR, CConvAVR_BUILTIN,
     CConvAMDGPU_VS, CConvAMDGPU_GS, CConvAMDGPU_CS, CConvAMDGPU_KERNEL,
     CConvX86_RegCall, CConvAMDGPU_HS, CConvMSP430_BUILTIN, CConvAMDGPU_LS,
     CConvAMDGPU_ES, CConvAArch64_VectorCall, CConvAArch64_SVE_VectorCall,
     CConvWASM_EmscriptenInvoke, CConvAMDGPU_Gfx, CConvM68k_INTR
    ]> {
  let cppNamespace = "::mlir::LLVM::cconv";
}

def CConv : DialectAttr<
    LLVM_Dialect,
    CPred<"$_self.isa<::mlir::LLVM::CConvAttr>()">,
    "LLVM Calling Convention specification"> {
  let storageType = "::mlir::LLVM::CConvAttr";
  let returnType = "::mlir::LLVM::cconv::CConv";
  let convertFromStorage = "$_self.getCallingConv()";
  let constBuilderCall =
          "::mlir::LLVM::CConvAttr::get($_builder.getContext(), $0)";
}

//===----------------------------------------------------------------------===//
// DIEmissionKind
//===----------------------------------------------------------------------===//

def LLVM_DIEmissionNone                : I64EnumAttrCase<"None", 0>;
def LLVM_DIEmissionFull                : I64EnumAttrCase<"Full", 1>;
def LLVM_DIEmissionLineTablesOnly      : I64EnumAttrCase<"LineTablesOnly", 2>;
def LLVM_DIEmissionDebugDirectivesOnly : I64EnumAttrCase<"DebugDirectivesOnly", 3>;

def LLVM_DIEmissionKind : I64EnumAttr<
    "DIEmissionKind",
    "LLVM debug emission kind", [
      LLVM_DIEmissionNone,
      LLVM_DIEmissionFull,
      LLVM_DIEmissionLineTablesOnly,
      LLVM_DIEmissionDebugDirectivesOnly,
    ]> {
  let cppNamespace = "::mlir::LLVM";
}

//===----------------------------------------------------------------------===//
// DIFlags
//===----------------------------------------------------------------------===//

def LLVM_DIFlagZero : I32BitEnumAttrCaseNone<"Zero">;

// The first two bits are used for a visibility enum.
//   * Private = 1
//   * Protected = 2
//   * Public = 3
def LLVM_DIFlagVisibilityBit0 : I32BitEnumAttrCaseBit<"Bit0", 0>;
def LLVM_DIFlagVisibilityBit1 : I32BitEnumAttrCaseBit<"Bit1", 1>;
def LLVM_DIFlagPrivate : I32BitEnumAttrCaseGroup<"Private",
  [LLVM_DIFlagVisibilityBit0]
>;
def LLVM_DIFlagProtected : I32BitEnumAttrCaseGroup<"Protected",
  [LLVM_DIFlagVisibilityBit1]
>;
def LLVM_DIFlagPublic : I32BitEnumAttrCaseGroup<"Public",
  [LLVM_DIFlagVisibilityBit0, LLVM_DIFlagVisibilityBit1]
>;

// The remaining bits are used for the various flags.
def LLVM_DIFlagFwdDecl             : I32BitEnumAttrCaseBit<"FwdDecl", 2>;
def LLVM_DIFlagAppleBlock          : I32BitEnumAttrCaseBit<"AppleBlock", 3>;
def LLVM_DIFlagReservedBit4        : I32BitEnumAttrCaseBit<"ReservedBit4", 4>;
def LLVM_DIFlagVirtual             : I32BitEnumAttrCaseBit<"Virtual", 5>;
def LLVM_DIFlagArtificial          : I32BitEnumAttrCaseBit<"Artificial", 6>;
def LLVM_DIFlagExplicit            : I32BitEnumAttrCaseBit<"Explicit", 7>;
def LLVM_DIFlagPrototyped          : I32BitEnumAttrCaseBit<"Prototyped", 8>;
def LLVM_DIFlagObjcClassComplete   : I32BitEnumAttrCaseBit<"ObjcClassComplete", 9>;
def LLVM_DIFlagObjectPointer       : I32BitEnumAttrCaseBit<"ObjectPointer", 10>;
def LLVM_DIFlagVector              : I32BitEnumAttrCaseBit<"Vector", 11>;
def LLVM_DIFlagStaticMember        : I32BitEnumAttrCaseBit<"StaticMember", 12>;
def LLVM_DIFlagLValueReference     : I32BitEnumAttrCaseBit<"LValueReference", 13>;
def LLVM_DIFlagRValueReference     : I32BitEnumAttrCaseBit<"RValueReference", 14>;
def LLVM_DIFlagExportSymbols       : I32BitEnumAttrCaseBit<"ExportSymbols", 15>;
def LLVM_DIFlagSingleInheritance   : I32BitEnumAttrCaseBit<"SingleInheritance", 16>;
def LLVM_DIFlagMultipleInheritance : I32BitEnumAttrCaseBit<"MultipleInheritance", 16>;
def LLVM_DIFlagVirtualInheritance  : I32BitEnumAttrCaseBit<"VirtualInheritance", 16>;
def LLVM_DIFlagIntroducedVirtual   : I32BitEnumAttrCaseBit<"IntroducedVirtual", 18>;
def LLVM_DIFlagBitField            : I32BitEnumAttrCaseBit<"BitField", 19>;
def LLVM_DIFlagNoReturn            : I32BitEnumAttrCaseBit<"NoReturn", 20>;
def LLVM_DIFlagTypePassByValue     : I32BitEnumAttrCaseBit<"TypePassByValue", 22>;
def LLVM_DIFlagTypePassByReference : I32BitEnumAttrCaseBit<"TypePassByReference", 23>;
def LLVM_DIFlagEnumClass           : I32BitEnumAttrCaseBit<"EnumClass", 24>;
def LLVM_DIFlagThunk               : I32BitEnumAttrCaseBit<"Thunk", 25>;
def LLVM_DIFlagNonTrivial          : I32BitEnumAttrCaseBit<"NonTrivial", 26>;
def LLVM_DIFlagBigEndian           : I32BitEnumAttrCaseBit<"BigEndian", 27>;
def LLVM_DIFlagLittleEndian        : I32BitEnumAttrCaseBit<"LittleEndian", 28>;
def LLVM_DIFlagAllCallsDescribed   : I32BitEnumAttrCaseBit<"AllCallsDescribed", 29>;

def DIFlags : I32BitEnumAttr<
    "DIFlags",
    "LLVM DI flags", [
      LLVM_DIFlagZero,
      LLVM_DIFlagVisibilityBit0,
      LLVM_DIFlagVisibilityBit1,
      LLVM_DIFlagPrivate,
      LLVM_DIFlagProtected,
      LLVM_DIFlagPublic,
      LLVM_DIFlagFwdDecl,
      LLVM_DIFlagAppleBlock,
      LLVM_DIFlagReservedBit4,
      LLVM_DIFlagVirtual,
      LLVM_DIFlagArtificial,
      LLVM_DIFlagExplicit,
      LLVM_DIFlagPrototyped,
      LLVM_DIFlagObjcClassComplete,
      LLVM_DIFlagObjectPointer,
      LLVM_DIFlagVector,
      LLVM_DIFlagStaticMember,
      LLVM_DIFlagLValueReference,
      LLVM_DIFlagRValueReference,
      LLVM_DIFlagExportSymbols,
      LLVM_DIFlagSingleInheritance,
      LLVM_DIFlagMultipleInheritance,
      LLVM_DIFlagVirtualInheritance,
      LLVM_DIFlagIntroducedVirtual,
      LLVM_DIFlagBitField,
      LLVM_DIFlagNoReturn,
      LLVM_DIFlagTypePassByValue,
      LLVM_DIFlagTypePassByReference,
      LLVM_DIFlagEnumClass,
      LLVM_DIFlagThunk,
      LLVM_DIFlagNonTrivial,
      LLVM_DIFlagBigEndian,
      LLVM_DIFlagLittleEndian,
      LLVM_DIFlagAllCallsDescribed
    ]> {
  let cppNamespace = "::mlir::LLVM";
  let printBitEnumPrimaryGroups = 1;
}

//===----------------------------------------------------------------------===//
// DISubprogramFlags
//===----------------------------------------------------------------------===//

def LLVM_DISPVirtual        : I32BitEnumAttrCaseBit<"Virtual", 0>;
def LLVM_DISPPureVirtual    : I32BitEnumAttrCaseBit<"PureVirtual", 1>;
def LLVM_DISPLocalToUnit    : I32BitEnumAttrCaseBit<"LocalToUnit", 2>;
def LLVM_DISPDefinition     : I32BitEnumAttrCaseBit<"Definition", 3>;
def LLVM_DISPOptimized      : I32BitEnumAttrCaseBit<"Optimized", 4>;
def LLVM_DISPPure           : I32BitEnumAttrCaseBit<"Pure", 5>;
def LLVM_DISPElemental      : I32BitEnumAttrCaseBit<"Elemental", 6>;
def LLVM_DISPRecursive      : I32BitEnumAttrCaseBit<"Recursive", 7>;
def LLVM_DISPMainSubprogram : I32BitEnumAttrCaseBit<"MainSubprogram", 8>;
def LLVM_DISPDeleted        : I32BitEnumAttrCaseBit<"Deleted", 9>;
def LLVM_DISPObjCDirect     : I32BitEnumAttrCaseBit<"ObjCDirect", 11>;

def DISubprogramFlags : I32BitEnumAttr<
    "DISubprogramFlags",
    "LLVM DISubprogram flags", [
      LLVM_DISPVirtual,
      LLVM_DISPPureVirtual,
      LLVM_DISPLocalToUnit,
      LLVM_DISPDefinition,
      LLVM_DISPOptimized,
      LLVM_DISPPure, 
      LLVM_DISPElemental,
      LLVM_DISPRecursive,
      LLVM_DISPMainSubprogram,
      LLVM_DISPDeleted,
      LLVM_DISPObjCDirect
    ]> {
  let cppNamespace = "::mlir::LLVM";
  let printBitEnumPrimaryGroups = 1;
}

//===----------------------------------------------------------------------===//
// FastmathFlags
//===----------------------------------------------------------------------===//

def FMFnone     : I32BitEnumAttrCaseNone<"none">;
def FMFnnan     : I32BitEnumAttrCaseBit<"nnan", 0>;
def FMFninf     : I32BitEnumAttrCaseBit<"ninf", 1>;
def FMFnsz      : I32BitEnumAttrCaseBit<"nsz", 2>;
def FMFarcp     : I32BitEnumAttrCaseBit<"arcp", 3>;
def FMFcontract : I32BitEnumAttrCaseBit<"contract", 4>;
def FMFafn      : I32BitEnumAttrCaseBit<"afn", 5>;
def FMFreassoc  : I32BitEnumAttrCaseBit<"reassoc", 6>;
def FMFfast     : I32BitEnumAttrCaseGroup<"fast",
  [ FMFnnan, FMFninf, FMFnsz, FMFarcp, FMFcontract, FMFafn, FMFreassoc]>;

def FastmathFlags : I32BitEnumAttr<
    "FastmathFlags",
    "LLVM fastmath flags",
    [FMFnone, FMFnnan, FMFninf, FMFnsz, FMFarcp, FMFcontract, FMFafn,
     FMFreassoc, FMFfast]> {
  let separator = ", ";
  let cppNamespace = "::mlir::LLVM";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

//===----------------------------------------------------------------------===//
// FCmp Predicates
//===----------------------------------------------------------------------===//

// Predicate for float comparisons
def FCmpPredicateFALSE  : I64EnumAttrCase<"_false", 0>;
def FCmpPredicateOEQ    : I64EnumAttrCase<"oeq", 1>;
def FCmpPredicateOGT    : I64EnumAttrCase<"ogt", 2>;
def FCmpPredicateOGE    : I64EnumAttrCase<"oge", 3>;
def FCmpPredicateOLT    : I64EnumAttrCase<"olt", 4>;
def FCmpPredicateOLE    : I64EnumAttrCase<"ole", 5>;
def FCmpPredicateONE    : I64EnumAttrCase<"one", 6>;
def FCmpPredicateORD    : I64EnumAttrCase<"ord", 7>;
def FCmpPredicateUEQ    : I64EnumAttrCase<"ueq", 8>;
def FCmpPredicateUGT    : I64EnumAttrCase<"ugt", 9>;
def FCmpPredicateUGE    : I64EnumAttrCase<"uge", 10>;
def FCmpPredicateULT    : I64EnumAttrCase<"ult", 11>;
def FCmpPredicateULE    : I64EnumAttrCase<"ule", 12>;
def FCmpPredicateUNE    : I64EnumAttrCase<"une", 13>;
def FCmpPredicateUNO    : I64EnumAttrCase<"uno", 14>;
def FCmpPredicateTRUE   : I64EnumAttrCase<"_true", 15>;

def FCmpPredicate : I64EnumAttr<
    "FCmpPredicate",
    "llvm.fcmp comparison predicate",
    [FCmpPredicateFALSE, FCmpPredicateOEQ, FCmpPredicateOGT, FCmpPredicateOGE,
     FCmpPredicateOLT, FCmpPredicateOLE, FCmpPredicateONE, FCmpPredicateORD,
     FCmpPredicateUEQ, FCmpPredicateUGT, FCmpPredicateUGE, FCmpPredicateULT,
     FCmpPredicateULE, FCmpPredicateUNE, FCmpPredicateUNO, FCmpPredicateTRUE
    ]> {
  let cppNamespace = "::mlir::LLVM";
}

//===----------------------------------------------------------------------===//
// ICmp Predicates
//===----------------------------------------------------------------------===//

// Predicate for integer comparisons.
def ICmpPredicateEQ  : I64EnumAttrCase<"eq", 0>;
def ICmpPredicateNE  : I64EnumAttrCase<"ne", 1>;
def ICmpPredicateSLT : I64EnumAttrCase<"slt", 2>;
def ICmpPredicateSLE : I64EnumAttrCase<"sle", 3>;
def ICmpPredicateSGT : I64EnumAttrCase<"sgt", 4>;
def ICmpPredicateSGE : I64EnumAttrCase<"sge", 5>;
def ICmpPredicateULT : I64EnumAttrCase<"ult", 6>;
def ICmpPredicateULE : I64EnumAttrCase<"ule", 7>;
def ICmpPredicateUGT : I64EnumAttrCase<"ugt", 8>;
def ICmpPredicateUGE : I64EnumAttrCase<"uge", 9>;
def ICmpPredicate : I64EnumAttr<
    "ICmpPredicate",
    "llvm.icmp comparison predicate",
    [ICmpPredicateEQ, ICmpPredicateNE, ICmpPredicateSLT, ICmpPredicateSLE,
     ICmpPredicateSGT, ICmpPredicateSGE, ICmpPredicateULT, ICmpPredicateULE,
     ICmpPredicateUGT, ICmpPredicateUGE]> {
  let cppNamespace = "::mlir::LLVM";
}

//===----------------------------------------------------------------------===//
// Linkage
//===----------------------------------------------------------------------===//

// Linkage attribute is used on functions and globals. The order follows that of
// https://llvm.org/docs/LangRef.html#linkage-types. The names are equivalent to
// visible names in the IR rather than to enum values names in llvm::GlobalValue
// since the latter is easier to change.
def LinkagePrivate
    : LLVM_EnumAttrCase<"Private", "private", "PrivateLinkage", 0>;
def LinkageInternal
    : LLVM_EnumAttrCase<"Internal", "internal", "InternalLinkage", 1>;
def LinkageAvailableExternally
    : LLVM_EnumAttrCase<"AvailableExternally", "available_externally",
                        "AvailableExternallyLinkage", 2>;
def LinkageLinkonce
    : LLVM_EnumAttrCase<"Linkonce", "linkonce", "LinkOnceAnyLinkage", 3>;
def LinkageWeak
    : LLVM_EnumAttrCase<"Weak", "weak", "WeakAnyLinkage", 4>;
def LinkageCommon
    : LLVM_EnumAttrCase<"Common", "common", "CommonLinkage", 5>;
def LinkageAppending
    : LLVM_EnumAttrCase<"Appending", "appending", "AppendingLinkage", 6>;
def LinkageExternWeak
   : LLVM_EnumAttrCase<"ExternWeak", "extern_weak", "ExternalWeakLinkage", 7>;
def LinkageLinkonceODR
    : LLVM_EnumAttrCase<"LinkonceODR", "linkonce_odr", "LinkOnceODRLinkage", 8>;
def LinkageWeakODR
    : LLVM_EnumAttrCase<"WeakODR", "weak_odr", "WeakODRLinkage", 9>;
def LinkageExternal
    : LLVM_EnumAttrCase<"External", "external", "ExternalLinkage", 10>;

def LinkageEnum : LLVM_EnumAttr<
    "Linkage",
    "::llvm::GlobalValue::LinkageTypes",
    "LLVM linkage types",
    [LinkagePrivate, LinkageInternal, LinkageAvailableExternally,
     LinkageLinkonce, LinkageWeak, LinkageCommon, LinkageAppending,
     LinkageExternWeak, LinkageLinkonceODR, LinkageWeakODR, LinkageExternal]> {
  let cppNamespace = "::mlir::LLVM::linkage";
}

def Linkage : DialectAttr<
    LLVM_Dialect,
    CPred<"$_self.isa<::mlir::LLVM::LinkageAttr>()">,
    "LLVM Linkage specification"> {
  let storageType = "::mlir::LLVM::LinkageAttr";
  let returnType = "::mlir::LLVM::Linkage";
  let convertFromStorage = "$_self.getLinkage()";
  let constBuilderCall =
          "::mlir::LLVM::LinkageAttr::get($_builder.getContext(), $0)";
}

//===----------------------------------------------------------------------===//
// LoopOptions
//===----------------------------------------------------------------------===//

def LOptDisableUnroll : I32EnumAttrCase<"disable_unroll", 1>;
def LOptDisableLICM : I32EnumAttrCase<"disable_licm", 2>;
def LOptInterleaveCount : I32EnumAttrCase<"interleave_count", 3>;
def LOptDisablePipeline : I32EnumAttrCase<"disable_pipeline", 4>;
def LOptPipelineInitiationInterval : I32EnumAttrCase<"pipeline_initiation_interval", 5>;

def LoopOptionCase : I32EnumAttr<
    "LoopOptionCase",
    "LLVM loop option",
    [LOptDisableUnroll, LOptDisableLICM, LOptInterleaveCount,
     LOptDisablePipeline, LOptPipelineInitiationInterval
    ]> {
  let cppNamespace = "::mlir::LLVM";
}

//===----------------------------------------------------------------------===//
// UnnamedAddr
//===----------------------------------------------------------------------===//

def UnnamedAddrNone : LLVM_EnumAttrCase<"None", "", "None", 0>;
def UnnamedAddrLocal : LLVM_EnumAttrCase<"Local", "local_unnamed_addr", "Local", 1>;
def UnnamedAddrGlobal : LLVM_EnumAttrCase<"Global", "unnamed_addr", "Global", 2>;

def UnnamedAddr : LLVM_EnumAttr<
    "UnnamedAddr",
    "::llvm::GlobalValue::UnnamedAddr",
    "LLVM GlobalValue UnnamedAddr",
    [UnnamedAddrNone, UnnamedAddrLocal, UnnamedAddrGlobal]> {
  let cppNamespace = "::mlir::LLVM";
}

#endif // LLVMIR_ENUMS_TD
