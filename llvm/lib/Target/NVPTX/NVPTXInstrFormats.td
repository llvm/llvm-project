//===- NVPTXInstrFormats.td - NVPTX Instruction Formats-------*- tblgen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe NVPTX instructions format
//
//===----------------------------------------------------------------------===//

// Generic NVPTX Format

class NVPTXInst<dag outs, dag ins, string asmstr, list<dag> pattern = []>
  : Instruction {
  field bits<14> Inst;

  let Namespace = "NVPTX";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;

  // TSFlagFields
  bit IsLoad = false;
  bit IsStore = false;

  bit IsTex = false;
  bit IsSust = false;
  bit IsSurfTexQuery = false;
  bit IsTexModeUnified = false;

  // The following field is encoded as log2 of the vector size minus one,
  // with 0 meaning the operation is not a surface instruction.  For example,
  // if IsSuld == 2, then the instruction is a suld instruction with vector size
  // 2**(2-1) = 2.
  bits<2> IsSuld = 0;

  let TSFlags{4}      = IsLoad;
  let TSFlags{5}      = IsStore;
  let TSFlags{6}      = IsTex;
  let TSFlags{8...7}  = IsSuld;
  let TSFlags{9}      = IsSust;
  let TSFlags{10}     = IsSurfTexQuery;
  let TSFlags{11}     = IsTexModeUnified;
}
