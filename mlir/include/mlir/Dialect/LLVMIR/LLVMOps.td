//===-- LLVMOps.td - LLVM IR dialect op definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the LLVM IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_OPS
#define LLVMIR_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
    [FMFnnan, FMFninf, FMFnsz, FMFarcp, FMFcontract, FMFafn, FMFreassoc, FMFfast
    ]> {
  let separator = ", ";
  let cppNamespace = "::mlir::LLVM";
  let printBitEnumPrimaryGroups = 1;
}

def LLVM_FMFAttr : DialectAttr<
    LLVM_Dialect,
    CPred<"$_self.isa<::mlir::LLVM::FMFAttr>()">,
    "LLVM fastmath flags"> {
  let storageType = "::mlir::LLVM::FMFAttr";
  let returnType = "::mlir::LLVM::FastmathFlags";
  let convertFromStorage = "$_self.getFlags()";
  let constBuilderCall =
          "::mlir::LLVM::FMFAttr::get($_builder.getContext(), $0)";
}

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

class LLVM_Builder<string builder> {
  string llvmBuilder = builder;
}

// Base class for LLVM terminator operations.  All terminator operations have
// zero results and an optional list of successors.
class LLVM_TerminatorOp<string mnemonic, list<Trait> traits = []> :
    LLVM_Op<mnemonic, !listconcat(traits, [Terminator])>;

// Class for arithmetic binary operations.
class LLVM_ArithmeticOpBase<Type type, string mnemonic,
                            string builderFunc, list<Trait> traits = []> :
    LLVM_Op<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($lhs, $rhs);"> {
  dag commonArgs = (ins LLVM_ScalarOrVectorOf<type>:$lhs,
                    LLVM_ScalarOrVectorOf<type>:$rhs);
  let results = (outs LLVM_ScalarOrVectorOf<type>:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$lhs `,` $rhs custom<LLVMOpAttrs>(attr-dict) `:` type($res)";
}
class LLVM_IntArithmeticOp<string mnemonic, string builderFunc,
                           list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<AnyInteger, mnemonic, builderFunc, traits> {
  let arguments = commonArgs;
}
class LLVM_FloatArithmeticOp<string mnemonic, string builderFunc,
                             list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<LLVM_AnyFloat, mnemonic, builderFunc,
    !listconcat([DeclareOpInterfaceMethods<FastmathFlagsInterface>], traits)> {
  dag fmfArg = (ins DefaultValuedAttr<LLVM_FMFAttr, "{}">:$fastmathFlags);
  let arguments = !con(commonArgs, fmfArg);
}

// Class for arithmetic unary operations.
class LLVM_UnaryFloatArithmeticOp<Type type, string mnemonic,
                             string builderFunc, list<Trait> traits = []> :
    LLVM_Op<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType, DeclareOpInterfaceMethods<FastmathFlagsInterface>], traits)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($operand);"> {
  let arguments = (ins type:$operand, DefaultValuedAttr<LLVM_FMFAttr, "{}">:$fastmathFlags);
  let results = (outs type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$operand custom<LLVMOpAttrs>(attr-dict) `:` type($res)";
}

// Integer binary operations.
def LLVM_AddOp : LLVM_IntArithmeticOp<"add", "CreateAdd", [Commutative]>;
def LLVM_SubOp : LLVM_IntArithmeticOp<"sub", "CreateSub">;
def LLVM_MulOp : LLVM_IntArithmeticOp<"mul", "CreateMul", [Commutative]>;
def LLVM_UDivOp : LLVM_IntArithmeticOp<"udiv", "CreateUDiv">;
def LLVM_SDivOp : LLVM_IntArithmeticOp<"sdiv", "CreateSDiv">;
def LLVM_URemOp : LLVM_IntArithmeticOp<"urem", "CreateURem">;
def LLVM_SRemOp : LLVM_IntArithmeticOp<"srem", "CreateSRem">;
def LLVM_AndOp : LLVM_IntArithmeticOp<"and", "CreateAnd">;
def LLVM_OrOp : LLVM_IntArithmeticOp<"or", "CreateOr">;
def LLVM_XOrOp : LLVM_IntArithmeticOp<"xor", "CreateXor">;
def LLVM_ShlOp : LLVM_IntArithmeticOp<"shl", "CreateShl">;
def LLVM_LShrOp : LLVM_IntArithmeticOp<"lshr", "CreateLShr">;
def LLVM_AShrOp : LLVM_IntArithmeticOp<"ashr", "CreateAShr">;

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

// Other integer operations.
def LLVM_ICmpOp : LLVM_Op<"icmp", [NoSideEffect]> {
  let arguments = (ins ICmpPredicate:$predicate,
                   AnyTypeOf<[LLVM_ScalarOrVectorOf<AnyInteger>, LLVM_ScalarOrVectorOf<LLVM_AnyPointer>]>:$lhs,
                   AnyTypeOf<[LLVM_ScalarOrVectorOf<AnyInteger>, LLVM_ScalarOrVectorOf<LLVM_AnyPointer>]>:$rhs);
  let results = (outs LLVM_ScalarOrVectorOf<I1>:$res);
  let llvmBuilder = [{
    $res = builder.CreateICmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let builders = [
    OpBuilder<(ins "ICmpPredicate":$predicate, "Value":$lhs, "Value":$rhs)>
  ];
  let hasCustomAssemblyFormat = 1;
}

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

// Other floating-point operations.
def LLVM_FCmpOp : LLVM_Op<"fcmp", [
    NoSideEffect, DeclareOpInterfaceMethods<FastmathFlagsInterface>]> {
  let arguments = (ins FCmpPredicate:$predicate,
                   LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$lhs,
                   LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$rhs,
                   DefaultValuedAttr<LLVM_FMFAttr, "{}">:$fastmathFlags);
  let results = (outs LLVM_ScalarOrVectorOf<I1>:$res);
  let llvmBuilder = [{
    $res = builder.CreateFCmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let hasCustomAssemblyFormat = 1;
}

// Floating point binary operations.
def LLVM_FAddOp : LLVM_FloatArithmeticOp<"fadd", "CreateFAdd">;
def LLVM_FSubOp : LLVM_FloatArithmeticOp<"fsub", "CreateFSub">;
def LLVM_FMulOp : LLVM_FloatArithmeticOp<"fmul", "CreateFMul">;
def LLVM_FDivOp : LLVM_FloatArithmeticOp<"fdiv", "CreateFDiv">;
def LLVM_FRemOp : LLVM_FloatArithmeticOp<"frem", "CreateFRem">;
def LLVM_FNegOp : LLVM_UnaryFloatArithmeticOp<
  LLVM_ScalarOrVectorOf<LLVM_AnyFloat>, "fneg", "CreateFNeg">;

// Common code definition that is used to verify and set the alignment attribute
// of LLVM ops that accept such an attribute.
class MemoryOpWithAlignmentBase {
  code setAlignmentCode = [{
    if ($alignment.has_value()) {
      auto align = $alignment.value();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  }];
}

// Code definition that is used for nontemporal metadata creation.
class MemoryOpWithAlignmentAndAttributes : MemoryOpWithAlignmentBase {
  code setNonTemporalMetadataCode = [{
    if ($nontemporal) {
      llvm::Module *module = builder.GetInsertBlock()->getModule();
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(module->getMDKindID("nontemporal"), metadata);
    }
  }];

  code setAccessGroupsMetadataCode = [{
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  }];

  code setAliasScopeMetadataCode = [{
    moduleTranslation.setAliasScopeMetadata(op, inst);
  }];
}

// Memory-related operations.
def LLVM_AllocaOp : LLVM_Op<"alloca">, MemoryOpWithAlignmentBase {
  let arguments = (ins AnyInteger:$arraySize,
                   OptionalAttr<I64Attr>:$alignment,
                   OptionalAttr<TypeAttr>:$elem_type);
  let results = (outs Res<LLVM_AnyPointer, "",
                          [MemAlloc<AutomaticAllocationScopeResource>]>:$res);
  string llvmBuilder = [{
    auto addrSpace = $_resultType->getPointerAddressSpace();
    llvm::Type *elementType = moduleTranslation.convertType(
        $elem_type ? *$elem_type
                   : op.getType().cast<LLVMPointerType>().getElementType());
    auto *inst = builder.CreateAlloca(elementType, addrSpace, $arraySize);
    }] # setAlignmentCode # [{
    $res = inst;
  }];
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$arraySize,
               "unsigned":$alignment),
    [{
      assert(!resultType.cast<LLVMPointerType>().isOpaque() &&
             "pass the allocated type explicitly if opaque pointers are used");
      if (alignment == 0)
        return build($_builder, $_state, resultType, arraySize, IntegerAttr(),
                     TypeAttr());
      build($_builder, $_state, resultType, arraySize,
        $_builder.getI64IntegerAttr(alignment), TypeAttr());
    }]>,
    OpBuilder<(ins "Type":$resultType, "Type":$elementType, "Value":$arraySize,
               CArg<"unsigned", "0">:$alignment),
    [{
      TypeAttr elemTypeAttr =
          resultType.cast<LLVMPointerType>().isOpaque() ?
          TypeAttr::get(elementType) : TypeAttr();
      build($_builder, $_state, resultType, arraySize,
            alignment == 0 ? IntegerAttr()
                           : $_builder.getI64IntegerAttr(alignment),
            elemTypeAttr);

    }]>
    ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_GEPOp : LLVM_Op<"getelementptr", [NoSideEffect]> {
  let arguments = (ins LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$base,
                   Variadic<LLVM_ScalarOrVectorOf<AnyInteger>>:$dynamicIndices,
                   DenseI32ArrayAttr:$rawConstantIndices,
                   OptionalAttr<TypeAttr>:$elem_type);
  let results = (outs LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$res);
  let skipDefaultBuilders = 1;

  let description = [{
    This operation mirrors LLVM IRs 'getelementptr' operation that is used to
    perform pointer arithmetic.

    Like in LLVM IR, it is possible to use both constants as well as SSA values
    as indices. In the case of indexing within a structure, it is required to
    either use constant indices directly, or supply a constant SSA value.

    Examples:

    ```mlir
    // GEP with an SSA value offset
    %0 = llvm.getelementptr %1[%2] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>

    // GEP with a constant offset
    %0 = llvm.getelementptr %1[3] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>

    // GEP with constant offsets into a structure
    %0 = llvm.getelementptr %1[0, 1]
       : (!llvm.ptr<struct(i32, f32)>) -> !llvm.ptr<f32>
    ```
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Type":$basePtrType, "Value":$basePtr,
               "ValueRange":$indices,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Type":$resultType, "Value":$basePtr, "ValueRange":$indices,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Type":$resultType, "Value":$basePtr, "ArrayRef<GEPArg>":$indices,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Type":$resultType, "Type":$basePtrType, "Value":$basePtr,
               "ArrayRef<GEPArg>":$indices,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];
  let llvmBuilder = [{
    SmallVector<llvm::Value *> indices;
    indices.reserve($rawConstantIndices.size());
    GEPIndicesAdaptor<decltype($dynamicIndices)>
        gepIndices(op.getRawConstantIndicesAttr(), $dynamicIndices);
    for (PointerUnion<IntegerAttr, llvm::Value*> valueOrAttr : gepIndices) {
      if (llvm::Value* value = valueOrAttr.dyn_cast<llvm::Value*>())
        indices.push_back(value);
      else
        indices.push_back(
            builder.getInt32(valueOrAttr.get<IntegerAttr>().getInt()));
    }
    Type baseElementType = op.getSourceElementType();
    llvm::Type *elementType = moduleTranslation.convertType(baseElementType);
    $res = builder.CreateGEP(elementType, $base, indices);
  }];
  let assemblyFormat = [{
    $base `[` custom<GEPIndices>($dynamicIndices, $rawConstantIndices) `]` attr-dict
    `:` functional-type(operands, results) (`,` $elem_type^)?
  }];

  let extraClassDeclaration = [{
    constexpr static int32_t kDynamicIndex = std::numeric_limits<int32_t>::min();

    /// Returns the type pointed to by the pointer argument of this GEP.
    Type getSourceElementType();

    GEPIndicesAdaptor<ValueRange> getIndices();
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def LLVM_LoadOp : LLVM_Op<"load">, MemoryOpWithAlignmentAndAttributes {
  let arguments = (ins Arg<LLVM_PointerTo<LLVM_LoadableType>, "", [MemRead]>:$addr,
                   OptionalAttr<SymbolRefArrayAttr>:$access_groups,
                   OptionalAttr<SymbolRefArrayAttr>:$alias_scopes,
                   OptionalAttr<SymbolRefArrayAttr>:$noalias_scopes,
                   OptionalAttr<I64Attr>:$alignment, UnitAttr:$volatile_,
                   UnitAttr:$nontemporal);
  let results = (outs LLVM_LoadableType:$res);
  string llvmBuilder = [{
    auto *inst = builder.CreateLoad($_resultType, $addr, $volatile_);
  }] # setAlignmentCode
     # setNonTemporalMetadataCode
     # setAccessGroupsMetadataCode
     # setAliasScopeMetadataCode
     # [{
    $res = inst;
  }];
  let builders = [
    OpBuilder<(ins "Value":$addr, CArg<"unsigned", "0">:$alignment,
      CArg<"bool", "false">:$isVolatile, CArg<"bool", "false">:$isNonTemporal),
    [{
      auto type = addr.getType().cast<LLVMPointerType>().getElementType();
      assert(type && "must provide explicit element type to the constructor "
                     "when the pointer type is opaque");
      build($_builder, $_state, type, addr, alignment, isVolatile, isNonTemporal);
    }]>,
    OpBuilder<(ins "Type":$t, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal)>,];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_StoreOp : LLVM_Op<"store">, MemoryOpWithAlignmentAndAttributes {
  let arguments = (ins LLVM_LoadableType:$value,
                   Arg<LLVM_PointerTo<LLVM_LoadableType>,"",[MemWrite]>:$addr,
                   OptionalAttr<SymbolRefArrayAttr>:$access_groups,
                   OptionalAttr<SymbolRefArrayAttr>:$alias_scopes,
                   OptionalAttr<SymbolRefArrayAttr>:$noalias_scopes,
                   OptionalAttr<I64Attr>:$alignment, UnitAttr:$volatile_,
                   UnitAttr:$nontemporal);
  string llvmBuilder = [{
    auto *inst = builder.CreateStore($value, $addr, $volatile_);
  }] # setAlignmentCode
     # setNonTemporalMetadataCode
     # setAccessGroupsMetadataCode
     # setAliasScopeMetadataCode;
  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal)>
    ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// Casts.
class LLVM_CastOp<string mnemonic, string builderFunc, Type type,
                  Type resultType, list<Trait> traits = []> :
    LLVM_Op<mnemonic, !listconcat([NoSideEffect], traits)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($arg, $_resultType);"> {
  let arguments = (ins type:$arg);
  let results = (outs resultType:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}
def LLVM_BitcastOp : LLVM_CastOp<"bitcast", "CreateBitCast",
                                 LLVM_AnyNonAggregate, LLVM_AnyNonAggregate> {
  let hasFolder = 1;
}
def LLVM_AddrSpaceCastOp : LLVM_CastOp<"addrspacecast", "CreateAddrSpaceCast",
                                       LLVM_ScalarOrVectorOf<LLVM_AnyPointer>,
                                       LLVM_ScalarOrVectorOf<LLVM_AnyPointer>> {
  let hasFolder = 1;
}
def LLVM_IntToPtrOp : LLVM_CastOp<"inttoptr", "CreateIntToPtr",
                                  LLVM_ScalarOrVectorOf<AnyInteger>,
                                  LLVM_ScalarOrVectorOf<LLVM_AnyPointer>>;
def LLVM_PtrToIntOp : LLVM_CastOp<"ptrtoint", "CreatePtrToInt",
                                  LLVM_ScalarOrVectorOf<LLVM_AnyPointer>,
                                  LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_SExtOp : LLVM_CastOp<"sext", "CreateSExt",
                              LLVM_ScalarOrVectorOf<AnyInteger>,
                              LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_ZExtOp : LLVM_CastOp<"zext", "CreateZExt",
                              LLVM_ScalarOrVectorOf<AnyInteger>,
                              LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_TruncOp : LLVM_CastOp<"trunc", "CreateTrunc",
                               LLVM_ScalarOrVectorOf<AnyInteger>,
                               LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_SIToFPOp : LLVM_CastOp<"sitofp", "CreateSIToFP",
                                LLVM_ScalarOrVectorOf<AnyInteger>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_UIToFPOp : LLVM_CastOp<"uitofp", "CreateUIToFP",
                                LLVM_ScalarOrVectorOf<AnyInteger>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_FPToSIOp : LLVM_CastOp<"fptosi", "CreateFPToSI",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_FPToUIOp : LLVM_CastOp<"fptoui", "CreateFPToUI",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_FPExtOp : LLVM_CastOp<"fpext", "CreateFPExt",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_FPTruncOp : LLVM_CastOp<"fptrunc", "CreateFPTrunc",
                                 LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                 LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;

// Call-related operations.
def LLVM_InvokeOp : LLVM_Op<"invoke", [
                      AttrSizedOperandSegments,
                      DeclareOpInterfaceMethods<BranchOpInterface>,
                      DeclareOpInterfaceMethods<CallOpInterface>, Terminator]> {
  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                   Variadic<LLVM_Type>:$callee_operands,
                   Variadic<LLVM_Type>:$normalDestOperands,
                   Variadic<LLVM_Type>:$unwindDestOperands);
  let results = (outs Variadic<LLVM_Type>);
  let successors = (successor AnySuccessor:$normalDest,
                              AnySuccessor:$unwindDest);

  let builders = [
    OpBuilder<(ins "TypeRange":$tys, "FlatSymbolRefAttr":$callee,
      "ValueRange":$ops, "Block*":$normal, "ValueRange":$normalOps,
      "Block*":$unwind, "ValueRange":$unwindOps),
    [{
      $_state.addAttribute("callee", callee);
      build($_builder, $_state, tys, ops, normal, normalOps, unwind, unwindOps);
    }]>,
    OpBuilder<(ins "TypeRange":$tys, "ValueRange":$ops, "Block*":$normal,
      "ValueRange":$normalOps, "Block*":$unwind, "ValueRange":$unwindOps),
    [{
      build($_builder, $_state, tys, /*callee=*/FlatSymbolRefAttr(), ops, normalOps,
            unwindOps, normal, unwind);
    }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_LandingpadOp : LLVM_Op<"landingpad"> {
  let arguments = (ins UnitAttr:$cleanup, Variadic<LLVM_Type>);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_CallOp : LLVM_Op<"call",
                          [DeclareOpInterfaceMethods<FastmathFlagsInterface>,
                           DeclareOpInterfaceMethods<CallOpInterface>,
                           DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call to an LLVM function.";
  let description = [{


    In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect
    implements this behavior by providing a variadic `call` operation for 0- and
    1-result functions. Even though MLIR supports multi-result functions, LLVM
    IR dialect disallows them.

    The `call` instruction supports both direct and indirect calls. Direct calls
    start with a function name (`@`-prefixed) and indirect calls start with an
    SSA value (`%`-prefixed). The direct callee, if present, is stored as a
    function attribute `callee`. The trailing type of the instruction is always
    the MLIR function type, which may be different from the indirect callee that
    has the wrapped LLVM IR function type.

    Examples:

    ```mlir
    // Direct call without arguments and with one result.
    %0 = llvm.call @foo() : () -> (f32)

    // Direct call with arguments and without a result.
    llvm.call @bar(%0) : (f32) -> ()

    // Indirect call with an argument and without a result.
    llvm.call %1(%0) : (f32) -> ()
    ```
  }];
  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                   Variadic<LLVM_Type>,
                   DefaultValuedAttr<LLVM_FMFAttr, "{}">:$fastmathFlags);
  let results = (outs Variadic<LLVM_Type>);
  let builders = [
    OpBuilder<(ins "LLVMFuncOp":$func, "ValueRange":$operands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes), [{
      Type resultType = func.getFunctionType().getReturnType();
      if (!resultType.isa<LLVM::LLVMVoidType>())
        $_state.addTypes(resultType);
      $_state.addAttribute("callee", SymbolRefAttr::get(func));
      $_state.addAttributes(attributes);
      $_state.addOperands(operands);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "StringAttr":$callee,
                   CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, results, SymbolRefAttr::get(callee), operands);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "StringRef":$callee,
                   CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, results,
            StringAttr::get($_builder.getContext(), callee), operands);
    }]>];
  let hasCustomAssemblyFormat = 1;
}

def LLVM_ExtractElementOp : LLVM_Op<"extractelement", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyVector:$vector, AnyInteger:$position);
  let results = (outs LLVM_Type:$res);
  string llvmBuilder = [{
    $res = builder.CreateExtractElement($vector, $position);
  }];
  let builders = [
    OpBuilder<(ins "Value":$vector, "Value":$position,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
def LLVM_ExtractValueOp : LLVM_Op<"extractvalue", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyAggregate:$container, ArrayAttr:$position);
  let results = (outs LLVM_Type:$res);
  string llvmBuilder = [{
    $res = builder.CreateExtractValue($container, extractPosition($position));
  }];
  let builders = [LLVM_OneResultOpBuilder];
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}
def LLVM_InsertElementOp : LLVM_Op<"insertelement", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyVector:$vector, LLVM_PrimitiveType:$value,
                   AnyInteger:$position);
  let results = (outs LLVM_AnyVector:$res);
  string llvmBuilder = [{
    $res = builder.CreateInsertElement($vector, $value, $position);
  }];
  let builders = [LLVM_OneResultOpBuilder];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
def LLVM_InsertValueOp : LLVM_Op<"insertvalue", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyAggregate:$container, LLVM_PrimitiveType:$value,
                   ArrayAttr:$position);
  let results = (outs LLVM_AnyAggregate:$res);
  string llvmBuilder = [{
    $res = builder.CreateInsertValue($container, $value,
                                     extractPosition($position));
  }];
  let builders = [
    OpBuilder<(ins "Value":$container, "Value":$value, "ArrayAttr":$position),
    [{
      build($_builder, $_state, container.getType(), container, value, position);
    }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
def LLVM_ShuffleVectorOp : LLVM_Op<"shufflevector", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyVector:$v1, LLVM_AnyVector:$v2, ArrayAttr:$mask);
  let results = (outs LLVM_AnyVector:$res);
  string llvmBuilder = [{
      SmallVector<unsigned, 4> position = extractPosition($mask);
      SmallVector<int, 4> mask(position.begin(), position.end());
      $res = builder.CreateShuffleVector($v1, $v2, mask);
  }];
  let builders = [
    OpBuilder<(ins "Value":$v1, "Value":$v2, "ArrayAttr":$mask,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// Misc operations.
def LLVM_SelectOp
    : LLVM_Op<"select",
          [NoSideEffect, AllTypesMatch<["trueValue", "falseValue", "res"]>]>,
      LLVM_Builder<
          "$res = builder.CreateSelect($condition, $trueValue, $falseValue);"> {
  let arguments = (ins LLVM_ScalarOrVectorOf<I1>:$condition,
                   LLVM_Type:$trueValue, LLVM_Type:$falseValue);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "operands attr-dict `:` type($condition) `,` type($res)";
}
def LLVM_FreezeOp : LLVM_Op<"freeze", [SameOperandsAndResultType]> {
  let arguments = (ins LLVM_Type:$val);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$val attr-dict `:` type($val)";
  string llvmBuilder = "builder.CreateFreeze($val);";
}

// Terminators.
def LLVM_BrOp : LLVM_TerminatorOp<"br",
    [DeclareOpInterfaceMethods<BranchOpInterface>, NoSideEffect]> {
  let arguments = (ins Variadic<LLVM_Type>:$destOperands);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
  let builders = [LLVM_TerminatorPassthroughOpBuilder];
}
def LLVM_CondBrOp : LLVM_TerminatorOp<"cond_br",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
     NoSideEffect]> {
  let arguments = (ins I1:$condition,
                   Variadic<LLVM_Type>:$trueDestOperands,
                   Variadic<LLVM_Type>:$falseDestOperands,
                   OptionalAttr<ElementsAttr>:$branch_weights);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = [{
    $condition ( `weights` `(` $branch_weights^ `)` )? `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
      "ValueRange":$trueOperands, "Block *":$falseDest,
      "ValueRange":$falseOperands,
      CArg<"Optional<std::pair<uint32_t, uint32_t>>", "{}">:$weights),
    [{
        ElementsAttr weightsAttr;
        if (weights) {
          weightsAttr =
              $_builder.getI32VectorAttr({static_cast<int32_t>(weights->first),
                                       static_cast<int32_t>(weights->second)});
        }
        build($_builder, $_state, condition, trueOperands, falseOperands, weightsAttr,
              trueDest, falseDest);
  }]>,
  OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
    "Block *":$falseDest, CArg<"ValueRange", "{}">:$falseOperands),
  [{
      build($_builder, $_state, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
  }]>, LLVM_TerminatorPassthroughOpBuilder];
}
def LLVM_ReturnOp : LLVM_TerminatorOp<"return", [NoSideEffect]> {
  let arguments = (ins Variadic<LLVM_Type>:$args);
  string llvmBuilder = [{
    if ($_numOperands != 0)
      builder.CreateRet($args[0]);
    else
      builder.CreateRetVoid();
  }];

  let assemblyFormat = "attr-dict ($args^ `:` type($args))?";
  let hasVerifier = 1;
}
def LLVM_ResumeOp : LLVM_TerminatorOp<"resume", []> {
  let arguments = (ins LLVM_Type:$value);
  string llvmBuilder = [{ builder.CreateResume($value); }];
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasVerifier = 1;
}
def LLVM_UnreachableOp : LLVM_TerminatorOp<"unreachable", []> {
  string llvmBuilder = [{ builder.CreateUnreachable(); }];
  let assemblyFormat = "attr-dict";
}

def LLVM_SwitchOp : LLVM_TerminatorOp<"switch",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
     NoSideEffect]> {
  let arguments = (ins
    AnyInteger:$value,
    Variadic<AnyType>:$defaultOperands,
    VariadicOfVariadic<AnyType, "case_operand_segments">:$caseOperands,
    OptionalAttr<ElementsAttr>:$case_values,
    ElementsAttr:$case_operand_segments,
    OptionalAttr<ElementsAttr>:$branch_weights
  );
  let successors = (successor
    AnySuccessor:$defaultDestination,
    VariadicSuccessor<AnySuccessor>:$caseDestinations
  );

  let assemblyFormat = [{
    $value `:` type($value) `,`
    $defaultDestination (`(` $defaultOperands^ `:` type($defaultOperands) `)`)?
    `[` `\n` custom<SwitchOpCases>(ref(type($value)), $case_values, $caseDestinations,
                                   $caseOperands, type($caseOperands)) `]`
    attr-dict
  }];
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$value,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"ArrayRef<int32_t>", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands,
      CArg<"ArrayRef<int32_t>", "{}">:$branchWeights)>,
    LLVM_TerminatorPassthroughOpBuilder
  ];

  let extraClassDeclaration = [{
    /// Return the operands for the case destination block at the given index.
    OperandRange getCaseOperands(unsigned index) {
      return getCaseOperands()[index];
    }

    /// Return a mutable range of operands for the case destination block at the
    /// given index.
    MutableOperandRange getCaseOperandsMutable(unsigned index) {
      return getCaseOperandsMutable()[index];
    }
  }];
}

////////////////////////////////////////////////////////////////////////////////
// Auxiliary operations (do not appear in LLVM IR but necessary for the dialect
// to work correctly).
////////////////////////////////////////////////////////////////////////////////

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

def LLVM_AddressOfOp : LLVM_Op<"mlir.addressof",
    [NoSideEffect, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs LLVM_AnyPointer:$res);

  let summary = "Creates a pointer pointing to a global or a function";

  let description = [{
    Creates an SSA value containing a pointer to a global variable or constant
    defined by `llvm.mlir.global`. The global value can be defined after its
    first referenced. If the global value is a constant, storing into it is not
    allowed.

    Examples:

    ```mlir
    func @foo() {
      // Get the address of a global variable.
      %0 = llvm.mlir.addressof @const : !llvm.ptr<i32>

      // Use it as a regular pointer.
      %1 = llvm.load %0 : !llvm.ptr<i32>

      // Get the address of a function.
      %2 = llvm.mlir.addressof @foo : !llvm.ptr<func<void ()>>

      // The function address can be used for indirect calls.
      llvm.call %2() : () -> ()
    }

    // Define the global.
    llvm.mlir.global @const(42 : i32) : i32
    ```
  }];

  let builders = [
    OpBuilder<(ins "GlobalOp":$global,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state,
            LLVM::LLVMPointerType::get(global.getType(), global.getAddrSpace()),
            global.getSymName());
      $_state.addAttributes(attrs);
    }]>,
    OpBuilder<(ins "LLVMFuncOp":$func,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state,
            LLVM::LLVMPointerType::get(func.getFunctionType()), func.getName());
      $_state.addAttributes(attrs);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the llvm.mlir.global operation that defined the value referenced
    /// here.
    GlobalOp getGlobal(SymbolTableCollection &symbolTable);

    /// Return the llvm.func operation that is referenced here.
    LLVMFuncOp getFunction(SymbolTableCollection &symbolTable);
  }];

  let assemblyFormat = "$global_name attr-dict `:` type($res)";
}

def LLVM_MetadataOp : LLVM_Op<"metadata", [
   NoRegionArguments, SymbolTable, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name
  );
  let summary = "LLVM dialect metadata.";
  let description = [{
    llvm.metadata op defines one or more metadata nodes.

    Example:
      llvm.metadata @metadata {
        llvm.access_group @group1
        llvm.access_group @group2
        llvm.return
      }
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def LLVM_AliasScopeDomainMetadataOp : LLVM_Op<"alias_scope_domain", [
  HasParent<"MetadataOp">, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$description
  );
  let summary = "LLVM dialect alias.scope domain metadata.";
  let description = [{
    Defines a domain that may be associated with an alias scope.

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
  }];
  let assemblyFormat = "$sym_name attr-dict";
}

def LLVM_AliasScopeMetadataOp : LLVM_Op<"alias_scope", [
  HasParent<"MetadataOp">, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$domain,
    OptionalAttr<StrAttr>:$description
  );
  let summary = "LLVM dialect alias.scope metadata.";
  let description = [{
    Defines an alias scope that can be attached to a memory-accessing operation.
    Such scopes can be used in combination with `noalias` metadata to indicate
    that sets of memory-affecting operations in one scope do not alias with
    memory-affecting operations in another scope.

    Example:
      module {
        llvm.func @foo(%ptr1 : !llvm.ptr<i32>) {
            %c0 = llvm.mlir.constant(0 : i32) : i32
            %c4 = llvm.mlir.constant(4 : i32) : i32
            %1 = llvm.ptrtoint %ptr1 : !llvm.ptr<i32> to i32
            %2 = llvm.add %1, %c1 : i32
            %ptr2 = llvm.inttoptr %2 : i32 to !llvm.ptr<i32>
            llvm.store %c0, %ptr1 { alias_scopes = [@metadata::@scope1], llvm.noalias = [@metadata::@scope2] } : !llvm.ptr<i32>
            llvm.store %c4, %ptr2 { alias_scopes = [@metadata::@scope2], llvm.noalias = [@metadata::@scope1] } : !llvm.ptr<i32>
            llvm.return
        }

        llvm.metadata @metadata {
          llvm.alias_scope_domain @unused_domain
          llvm.alias_scope_domain @domain { description = "Optional domain description"}
          llvm.alias_scope @scope1 { domain = @domain }
          llvm.alias_scope @scope2 { domain = @domain, description = "Optional scope description" }
          llvm.return
        }
      }

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
  }];
  let assemblyFormat = "$sym_name attr-dict";
}

def LLVM_AccessGroupMetadataOp : LLVM_Op<"access_group", [
  HasParent<"MetadataOp">, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name
  );
  let summary = "LLVM dialect access group metadata.";
  let description = [{
    Defines an access group metadata that can be attached to any instruction
    that potentially accesses memory. The access group may be attached to a
    memory accessing instruction via the `llvm.access.group` metadata and
    a branch instruction in the loop latch block via the
    `llvm.loop.parallel_accesses` metadata.

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#llvm-access-group-metadata
  }];
  let assemblyFormat = "$sym_name attr-dict";
}

def LLVM_GlobalOp : LLVM_Op<"mlir.global",
    [IsolatedFromAbove, SingleBlockImplicitTerminator<"ReturnOp">, Symbol]> {
  let arguments = (ins
    TypeAttr:$global_type,
    UnitAttr:$constant,
    StrAttr:$sym_name,
    Linkage:$linkage,
    UnitAttr:$dso_local,
    UnitAttr:$thread_local_,
    OptionalAttr<AnyAttr>:$value,
    OptionalAttr<I64Attr>:$alignment,
    DefaultValuedAttr<Confined<I32Attr, [IntNonNegative]>, "0">:$addr_space,
    OptionalAttr<UnnamedAddr>:$unnamed_addr,
    OptionalAttr<StrAttr>:$section
  );
  let summary = "LLVM dialect global.";
  let description = [{
    Since MLIR allows for arbitrary operations to be present at the top level,
    global variables are defined using the `llvm.mlir.global` operation. Both
    global constants and variables can be defined, and the value may also be
    initialized in both cases.

    There are two forms of initialization syntax. Simple constants that can be
    represented as MLIR attributes can be given in-line:

    ```mlir
    llvm.mlir.global @variable(32.0 : f32) : f32
    ```

    This initialization and type syntax is similar to `llvm.mlir.constant` and
    may use two types: one for MLIR attribute and another for the LLVM value.
    These types must be compatible.

    More complex constants that cannot be represented as MLIR attributes can be
    given in an initializer region:

    ```mlir
    // This global is initialized with the equivalent of:
    //   i32* getelementptr (i32* @g2, i32 2)
    llvm.mlir.global constant @int_gep() : !llvm.ptr<i32> {
      %0 = llvm.mlir.addressof @g2 : !llvm.ptr<i32>
      %1 = llvm.mlir.constant(2 : i32) : i32
      %2 = llvm.getelementptr %0[%1]
         : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
      // The initializer region must end with `llvm.return`.
      llvm.return %2 : !llvm.ptr<i32>
    }
    ```

    Only one of the initializer attribute or initializer region may be provided.

    `llvm.mlir.global` must appear at top-level of the enclosing module. It uses
    an @-identifier for its value, which will be uniqued by the module with
    respect to other @-identifiers in it.

    Examples:

    ```mlir
    // Global values use @-identifiers.
    llvm.mlir.global constant @cst(42 : i32) : i32

    // Non-constant values must also be initialized.
    llvm.mlir.global @variable(32.0 : f32) : f32

    // Strings are expected to be of wrapped LLVM i8 array type and do not
    // automatically include the trailing zero.
    llvm.mlir.global @string("abc") : !llvm.array<3 x i8>

    // For strings globals, the trailing type may be omitted.
    llvm.mlir.global constant @no_trailing_type("foo bar")

    // A complex initializer is constructed with an initializer region.
    llvm.mlir.global constant @int_gep() : !llvm.ptr<i32> {
      %0 = llvm.mlir.addressof @g2 : !llvm.ptr<i32>
      %1 = llvm.mlir.constant(2 : i32) : i32
      %2 = llvm.getelementptr %0[%1]
         : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
      llvm.return %2 : !llvm.ptr<i32>
    }
    ```

    Similarly to functions, globals have a linkage attribute. In the custom
    syntax, this attribute is placed between `llvm.mlir.global` and the optional
    `constant` keyword. If the attribute is omitted, `external` linkage is
    assumed by default.

    Examples:

    ```mlir
    // A constant with internal linkage will not participate in linking.
    llvm.mlir.global internal constant @cst(42 : i32) : i32

    // By default, "external" linkage is assumed and the global participates in
    // symbol resolution at link-time.
    llvm.mlir.global @glob(0 : f32) : f32

    // Alignment is optional
    llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) : !llvm.array<8 x f32>
    ```

    Like global variables in LLVM IR, globals can have an (optional)
    alignment attribute using keyword `alignment`. The integer value of the
    alignment must be a positive integer that is a power of 2.

    Examples:

    ```mlir
    // Alignment is optional
    llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) { alignment = 32 : i64 } : !llvm.array<8 x f32>
    ```

  }];
  let regions = (region AnyRegion:$initializer);

  let builders = [
    OpBuilder<(ins "Type":$type, "bool":$isConstant, "Linkage":$linkage,
      "StringRef":$name, "Attribute":$value,
      CArg<"uint64_t", "0">:$alignment,
      CArg<"unsigned", "0">:$addrSpace,
      CArg<"bool", "false">:$dsoLocal,
      CArg<"bool", "false">:$thread_local_,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// Return the LLVM type of the global.
    Type getType() {
      return getGlobalType();
    }
    /// Return the initializer attribute if it exists, or a null attribute.
    Attribute getValueOrNull() {
      return getValue().value_or(Attribute());
    }
    /// Return the initializer region. This may be empty, but if it is not it
    /// terminates in an `llvm.return` op with the initializer value.
    Region &getInitializerRegion() {
      return getOperation()->getRegion(0);
    }
    /// Return the initializer block. If the initializer region is empty this
    /// is nullptr. If it is not nullptr, it terminates with an `llvm.return`
    /// op with the initializer value.
    Block *getInitializerBlock() {
      return getInitializerRegion().empty() ?
        nullptr : &getInitializerRegion().front();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def LLVM_GlobalCtorsOp : LLVM_Op<"mlir.global_ctors", [
                           DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefArrayAttr
                   : $ctors, I32ArrayAttr
                   : $priorities);
  let summary = "LLVM dialect global_ctors.";
  let description = [{
    Specifies a list of constructor functions and priorities. The functions
    referenced by this array will be called in ascending order of priority (i.e.
    lowest first) when the module is loaded. The order of functions with the
    same priority is not defined. This operation is translated to LLVM's
    global_ctors global variable. The initializer functions are run at load
    time. The `data` field present in LLVM's global_ctors variable is not
    modeled here.

    Examples:

    ```mlir
    llvm.mlir.global_ctors {@ctor}

    llvm.func @ctor() {
      ...
      llvm.return
    }
    ```

  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def LLVM_GlobalDtorsOp : LLVM_Op<"mlir.global_dtors", [
                           DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins
    FlatSymbolRefArrayAttr:$dtors,
    I32ArrayAttr:$priorities
  );
  let summary = "LLVM dialect global_dtors.";
  let description = [{
    Specifies a list of destructor functions and priorities. The functions
    referenced by this array will be called in descending order of priority (i.e.
    highest first) when the module is unloaded. The order of functions with the
    same priority is not defined. This operation is translated to LLVM's
    global_dtors global variable. The `data` field present in LLVM's
    global_dtors variable is not modeled here.

    Examples:

    ```mlir
    llvm.func @dtor() {
      llvm.return
    }
    llvm.mlir.global_dtors {@dtor}
    ```

  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def LLVM_LLVMFuncOp : LLVM_Op<"func", [
    AutomaticAllocationScope, IsolatedFromAbove, FunctionOpInterface,
    CallableOpInterface, Symbol
  ]> {
  let summary = "LLVM dialect function.";

  let description = [{
    MLIR functions are defined by an operation that is not built into the IR
    itself. The LLVM dialect provides an `llvm.func` operation to define
    functions compatible with LLVM IR. These functions have LLVM dialect
    function type but use MLIR syntax to express it. They are required to have
    exactly one result type. LLVM function operation is intended to capture
    additional properties of LLVM functions, such as linkage and calling
    convention, that may be modeled differently by the built-in MLIR function.

    ```mlir
    // The type of @bar is !llvm<"i64 (i64)">
    llvm.func @bar(%arg0: i64) -> i64 {
      llvm.return %arg0 : i64
    }

    // Type type of @foo is !llvm<"void (i64)">
    // !llvm.void type is omitted
    llvm.func @foo(%arg0: i64) {
      llvm.return
    }

    // A function with `internal` linkage.
    llvm.func internal @internal_func() {
      llvm.return
    }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<LLVM_FunctionType>:$function_type,
    DefaultValuedAttr<Linkage, "Linkage::External">:$linkage,
    UnitAttr:$dso_local,
    DefaultValuedAttr<CConv, "CConv::C">:$CConv,
    OptionalAttr<FlatSymbolRefAttr>:$personality,
    OptionalAttr<StrAttr>:$garbageCollector,
    OptionalAttr<ArrayAttr>:$passthrough
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "Type":$type,
      CArg<"Linkage", "Linkage::External">:$linkage,
      CArg<"bool", "false">:$dsoLocal,
      CArg<"CConv", "CConv::C">:$cconv,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock();

    bool isVarArg() { return getFunctionType().isVarArg(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getParams(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getReturnTypes(); }

    /// Returns the callable region, which is the function body.
    Region *getCallableRegion() { return &getBody(); }

    /// Returns the callable result type, which is the function return type.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getReturnType(); }

  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def LLVM_NullOp
    : LLVM_Op<"mlir.null", [NoSideEffect]>,
      LLVM_Builder<"$res = llvm::ConstantPointerNull::get("
                   "    cast<llvm::PointerType>($_resultType));"> {
  let summary = "Defines a value containing a null pointer to LLVM type.";
  let description = [{
    Unlike LLVM IR, MLIR does not have first-class null pointers. They must be
    explicitly created as SSA values using `llvm.mlir.null`. This operation has
    no operands or attributes, and returns a null value of a wrapped LLVM IR
    pointer type.

    Examples:

    ```mlir
    // Null pointer to i8.
    %0 = llvm.mlir.null : !llvm.ptr<i8>

    // Null pointer to a function with signature void().
    %1 = llvm.mlir.null : !llvm.ptr<func<void ()>>
    ```
  }];

  let results = (outs LLVM_AnyPointer:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_UndefOp : LLVM_Op<"mlir.undef", [NoSideEffect]>,
                   LLVM_Builder<"$res = llvm::UndefValue::get($_resultType);"> {
  let summary = "Creates an undefined value of LLVM dialect type.";
  let description = [{
    Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
    must be created as SSA values using `llvm.mlir.undef`. This operation has no
    operands or attributes. It creates an undefined value of the specified LLVM
    IR dialect type wrapping an LLVM IR structure type.

    Example:

    ```mlir
    // Create a structure with a 32-bit integer followed by a float.
    %0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>
    ```
  }];
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_ConstantOp
    : LLVM_Op<"mlir.constant", [NoSideEffect, ConstantLike]>,
      LLVM_Builder<[{$res = getLLVMConstant($_resultType, $value, $_location,
                                            moduleTranslation);}]>
{
  let summary = "Defines a constant of LLVM type.";
  let description = [{
    Unlike LLVM IR, MLIR does not have first-class constant values. Therefore,
    all constants must be created as SSA values before being used in other
    operations. `llvm.mlir.constant` creates such values for scalars and
    vectors. It has a mandatory `value` attribute, which may be an integer,
    floating point attribute; dense or sparse attribute containing integers or
    floats. The type of the attribute is one of the corresponding MLIR builtin
    types. It may be omitted for `i64` and `f64` types that are implied. The
    operation produces a new SSA value of the specified LLVM IR dialect type.
    The type of that value _must_ correspond to the attribute type converted to
    LLVM IR.

    Examples:

    ```mlir
    // Integer constant, internal i32 is mandatory
    %0 = llvm.mlir.constant(42 : i32) : i32

    // It's okay to omit i64.
    %1 = llvm.mlir.constant(42) : i64

    // Floating point constant.
    %2 = llvm.mlir.constant(42.0 : f32) : f32

    // Splat dense vector constant.
    %3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

// Atomic operations.
//

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

def LLVM_AtomicRMWType : AnyTypeOf<[LLVM_AnyFloat, AnyInteger]>;

// FIXME: Need to add alignment attribute to MLIR atomicrmw operation.
def LLVM_AtomicRMWOp : LLVM_Op<"atomicrmw"> {
  let arguments = (ins AtomicBinOp:$bin_op,
                   LLVM_PointerTo<LLVM_AtomicRMWType>:$ptr,
                   LLVM_AtomicRMWType:$val, AtomicOrdering:$ordering);
  let results = (outs LLVM_AtomicRMWType:$res);
  let llvmBuilder = [{
    $res = builder.CreateAtomicRMW(getLLVMAtomicBinOp($bin_op), $ptr, $val,
                                   llvm::MaybeAlign(),
                                   getLLVMAtomicOrdering($ordering));
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_AtomicCmpXchgType : AnyTypeOf<[AnyInteger, LLVM_AnyPointer]>;
def LLVM_AtomicCmpXchgResultType : Type<And<[
  LLVM_AnyStruct.predicate,
  CPred<"$_self.cast<::mlir::LLVM::LLVMStructType>().getBody().size() == 2">,
  SubstLeaves<"$_self",
              "$_self.cast<::mlir::LLVM::LLVMStructType>().getBody()[0]",
              LLVM_AtomicCmpXchgType.predicate>,
  SubstLeaves<"$_self",
              "$_self.cast<::mlir::LLVM::LLVMStructType>().getBody()[1]",
              I1.predicate>]>,
 "an LLVM struct type with any integer or pointer followed by a single-bit "
 "integer">;

// FIXME: Need to add alignment attribute to MLIR cmpxchg operation.
def LLVM_AtomicCmpXchgOp : LLVM_Op<"cmpxchg"> {
  let arguments = (ins LLVM_PointerTo<LLVM_AtomicCmpXchgType>:$ptr,
                   LLVM_AtomicCmpXchgType:$cmp, LLVM_AtomicCmpXchgType:$val,
                   AtomicOrdering:$success_ordering,
                   AtomicOrdering:$failure_ordering);
  let results = (outs LLVM_AtomicCmpXchgResultType:$res);
  let llvmBuilder = [{
    $res = builder.CreateAtomicCmpXchg($ptr, $cmp, $val, llvm::MaybeAlign(),
                   getLLVMAtomicOrdering($success_ordering),
                   getLLVMAtomicOrdering($failure_ordering));
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_FenceOp : LLVM_Op<"fence"> {
  let arguments = (ins AtomicOrdering:$ordering, StrAttr:$syncscope);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  let llvmBuilder = [{
    llvm::LLVMContext &llvmContext = builder.getContext();
    builder.CreateFence(getLLVMAtomicOrdering($ordering),
      llvmContext.getOrInsertSyncScopeID($syncscope));
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

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

def LLVM_InlineAsmOp : LLVM_Op<"inline_asm", []> {
  let description = [{
    The InlineAsmOp mirrors the underlying LLVM semantics with a notable
    exception: the embedded `asm_string` is not allowed to define or reference
    any symbol or any global variable: only the operands of the op may be read,
    written, or referenced.
    Attempting to define or reference any symbol or any global behavior is
    considered undefined behavior at this time.
  }];
  let arguments = (
    ins Variadic<LLVM_Type>:$operands,
        StrAttr:$asm_string,
        StrAttr:$constraints,
        UnitAttr:$has_side_effects,
        UnitAttr:$is_align_stack,
        OptionalAttr<
          DefaultValuedAttr<AsmATTOrIntel, "AsmDialect::AD_ATT">>:$asm_dialect,
        OptionalAttr<ArrayAttr>:$operand_attrs);

  let results = (outs Optional<LLVM_Type>:$res);

  let assemblyFormat = [{
    (`has_side_effects` $has_side_effects^)?
    (`is_align_stack` $is_align_stack^)?
    (`asm_dialect` `=` $asm_dialect^)?
    (`operand_attrs` `=` $operand_attrs^)?
    attr-dict
    $asm_string `,` $constraints
    operands `:` functional-type(operands, results)
   }];

  let extraClassDeclaration = [{
    static StringRef getElementTypeAttrName() {
      return "elementtype";
    }
  }];
}

#endif // LLVMIR_OPS
