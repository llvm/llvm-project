//===-- SuperHInstrInfo.td - Main SuperH Instruction Definition -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the SuperH instruction set, defining the instructions
/// and properties of the instructions which are needed for code generation,
/// machine code emission, and analysis.
///
/// The subsystems are ordered after the instruction classification that
/// The SuperH ISA manuals use.
///
/// See: https://www.shared-ptr.com/sh_insns.html
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

include "SuperHInstrFormats.td"

//===----------------------------------------------------------------------===//
// Instruction Patterns
//===----------------------------------------------------------------------===//

defvar SH = DefaultMode;
def sh_ptr_rc : RegClassByHwMode<[SH], [GPR]>;

// Both cases can use the same decoder method, so avoid the dispatch
// by hwmode by setting an explicit DecoderMethod
def ptr_op : RegisterOperand<sh_ptr_rc> {
  let DecoderMethod = "DecodeIntRegsRegisterClass";
}


//===----------------------------------------------------------------------===//
// SuperH Type Profiles
//===----------------------------------------------------------------------===//

def SHSDT_CallSeqStart 	: SDCallSeqStart<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SHSDT_CallSeqEnd	: SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SHSDT_Call    		: SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;
def SHSDT_Ret           : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

def SHCallSeqStart  : SDNode<"ISD::CALLSEQ_START", SHSDT_CallSeqStart,
							 [SDNPHasChain, SDNPOutGlue]>;

def SHCallSeqEnd    : SDNode<"ISD::CALLSEQ_END", SHSDT_CallSeqEnd,
                             [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def SHCall          : SDNode<"SHISD::CALL", SHSDT_Call,
                             [SDNPHasChain, SDNPOutGlue,
                             SDNPOptInGlue, SDNPVariadic]>;

def SHRet 		    : SDNode<"SHISD::RET", SHSDT_Ret,
                             [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

//===----------------------------------------------------------------------===//
// Operand Classes
//===----------------------------------------------------------------------===//

// Base Operand Class
class SHOpClass<string name, 
                list<AsmOperandClass> superClasses = []> : AsmOperandClass {
  let Name = name;
  let ParserMethod = "parseOperand";
  let SuperClasses = superClasses;
}

// Immediates
class SHImmClass<int width> : SHOpClass<"Imm" # width> {
    let PredicateMethod = "isImm";
    let RenderMethod = "addImmOperands";
}

// Displacements
class SHDispClass<int width> : SHOpClass<"Disp" # width> {
    let PredicateMethod = "isDisp";
    let RenderMethod = "addDispOperands";
}

// Memory Ops
class SHMemClass : SHOpClass<"Mem"> {
    let PredicateMethod = "isMem";
    let RenderMethod = "addMemOperands";
}

//===----------------------------------------------------------------------===//
// Operands
//===----------------------------------------------------------------------===//

// Immediate
class SHImmOp<int width, ValueType vt>
      : Operand<vt> {
    let ParserMatchClass = SHImmClass<width>;
    let OperandType = "OPERAND_IMMEDIATE";
    let MCOperandPredicate = [{
        int64_t Imm;
        if (MCOp.evaluateAsConstantImm(Imm))
            return isInt<#width>(Imm);
        return MCOp.isBareSymbolRef();
    }];
}

def imm8    : SHImmOp<8, i8>;

// Displacement
class SHDispOp<int width, ValueType vt>
      : Operand<vt> {
    let ParserMatchClass = SHDispClass<width>;
    let OperandType = "OPERAND_IMMEDIATE";
    let MCOperandPredicate = [{
        int64_t Imm;
        if (MCOp.evaluateAsConstantImm(Imm))
            return isInt<#width>(Imm);
        return MCOp.isBareSymbolRef();
    }];
}
def disp4   : SHDispOp<4, i4>;
def disp8   : SHDispOp<8, i8>;
def disp12  : SHDispOp<12, i16>;



// Memory
class SHMemRegOp<RegisterClass regClass> 
      : RegisterOperand<regClass> {
    let ParserMatchClass = SHMemClass<>;
    let OperandType = "OPERAND_MEMORY";
}
def GPRMem : SHMemRegOp<GPR>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {

    // No-arg instruction
    class Op<bits<16> op, string opcodestr>
        : SHInstOP<op, (outs), (ins), opcodestr>;

    // Rn
    class Op_Rn<bits<16> op, string opcodestr>
        : SHInstOP_N4<op, (outs GPR:$Rn), (ins),
                         !strconcat(opcodestr, " $Rn")>;

    // R0 -> (OP) -> Rn
    class Op_R0_Rn<bits<16> op, string opcodestr>
        : SHInstOP_N4<op, (outs GPR:$Rn), (ins),
                         !strconcat(opcodestr, " r0,$Rn")>;

    // Rm -> (OP) -> Rn
    class Op_Rm_Rn<bits<16> op, string opcodestr>
        : SHInstOP_N4_M4<op, (outs GPR:$Rn), (ins GPR:$Rm),
                         !strconcat(opcodestr, " $Rm,$Rn")>;

    // imm -> (OP) -> Rn
    class Op_Imm_Rn<bits<16> op, string opcodestr>
        : SHInstOP_N4_I8<op, (outs GPR:$Rn), (ins imm8:$imm),
                         !strconcat(opcodestr, " $imm,$Rn")>;

    // imm -> (OP) -> Rn
    class Op_Imm_R0<bits<16> op, string opcodestr>
        : SHInstOP_I8<op, (outs), (ins imm8:$imm),
                      !strconcat(opcodestr, " $imm,R0")>;
}


//===----------------------------------------------------------------------===//
// Basic Instructions
//===----------------------------------------------------------------------===//

// NOP instruction, does nothing.
def NOP : Op<0b0000000000001001, "nop">;

//===----------------------------------------------------------------------===//
// Subsystems
//===----------------------------------------------------------------------===//

include "SuperHInstrData.td"
include "SuperHInstrArithmetic.td"
include "SuperHInstrBranch.td"

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

let Defs = [R0], Uses = [R0] in {
def ADJCALLSTACKDOWN : SHPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                               "!ADJCALLSTACKDOWN $amt1, $amt2",
                               [(SHCallSeqStart timm:$amt1, timm:$amt2)]>;
def ADJCALLSTACKUP : SHPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                            "!ADJCALLSTACKUP $amt1",
                            [(SHCallSeqEnd timm:$amt1, timm:$amt2)]>;
}