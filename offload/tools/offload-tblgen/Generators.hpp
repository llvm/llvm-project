//===- offload-tblgen/Generators.hpp - Offload generator declarations -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/TableGen/Record.h"

void EmitOffloadAPI(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitOffloadFuncNames(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitOffloadImplFuncDecls(llvm::RecordKeeper &Records,
                              llvm::raw_ostream &OS);
void EmitOffloadEntryPoints(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitOffloadPrintHeader(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitOffloadExports(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
