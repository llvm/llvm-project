//===-- CSKY.td - Describe the CSKY Target Machine ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Registers, calling conventions, instruction descriptions.
//===----------------------------------------------------------------------===//

include "CSKYRegisterInfo.td"
include "CSKYInstrInfo.td"

//===----------------------------------------------------------------------===//
// CSKY processors supported.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"generic", NoSchedModel, []>;

//===----------------------------------------------------------------------===//
// Define the CSKY target.
//===----------------------------------------------------------------------===//

def CSKYInstrInfo : InstrInfo;


def CSKYAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
  let AllowDuplicateRegisterNames = 1;
}

def CSKYAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def CSKY : Target {
  let InstructionSet = CSKYInstrInfo;
  let AssemblyParsers = [CSKYAsmParser];
  let AssemblyWriters = [CSKYAsmWriter];
  let AllowRegisterRenaming = 1;
}
