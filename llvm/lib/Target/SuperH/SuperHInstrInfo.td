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
// Operands
//===----------------------------------------------------------------------===//

class ImmAsmOperand<int width> : AsmOperandClass {
    let Name = "Imm" # width;
    let RenderMethod = "addImmOperands";
}

def imm8 : Operand<i8>, ImmLeaf<i32, [{return isInt<8>(Imm);}]> {
    let ParserMatchClass = ImmAsmOperand<8>;
    let EncoderMethod = "getImm";
    let DecoderMethod = "decodeImmOperand<8>";
    let MCOperandPredicate = [{
        int64_t Imm;
        if (MCOp.evaluateAsConstantImm(Imm))
            return isInt<8>(Imm);
        return MCOp.isBareSymbolRef();
    }];
}

//===----------------------------------------------------------------------===//
// Basic Instructions
//===----------------------------------------------------------------------===//

// NOP instruction, does nothing.
def NOP : SHInstOP<0b0000000000000000, (outs), (ins), "nop">;

//===----------------------------------------------------------------------===//
// Subsystems
//===----------------------------------------------------------------------===//

include "SuperHInstrData.td"

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