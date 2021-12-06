//===- VarLenCodeEmitterGen.h - CEG for variable-length insts ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declare the CodeEmitterGen component for variable-length
// instructions. See the .cpp file for more details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_VARLENCODEEMITTERGEN_H
#define LLVM_UTILS_TABLEGEN_VARLENCODEEMITTERGEN_H

namespace llvm {

class RecordKeeper;
class raw_ostream;

void emitVarLenCodeEmitter(RecordKeeper &R, raw_ostream &OS);

} // end namespace llvm
#endif
