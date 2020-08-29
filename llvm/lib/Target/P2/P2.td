//===-- p2.td - Describe the p2 Target Machine -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the p2 target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

include "P2RegisterInfo.td"
include "P2InstrInfo.td"
include "P2CallingConv.td"

def P2InstrInfo : InstrInfo;

def P2AsmParser : AsmParser {
    let ShouldEmitMatchRegisterName = 0;
    let HasMnemonicFirst = 0;
}

def P2AsmParserVariant : AsmParserVariant {
    int Variant = 0;

    // Recognize hard coded registers.
    string RegisterPrefix = "$";
}

class Proc<string Name> : Processor<Name, NoItineraries, []>;

def : Proc<"Propeller 2">;

def P2 : Target {
    let InstructionSet = P2InstrInfo;
    let AssemblyParsers = [P2AsmParser];
    let AssemblyParserVariants = [P2AsmParserVariant];
}
