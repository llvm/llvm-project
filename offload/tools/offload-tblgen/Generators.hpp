//===- offload-tblgen/Generators.hpp - Offload generator declarations -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/TableGen/Record.h"

void EmitOffloadAPI(const llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitOffloadFuncNames(const llvm::RecordKeeper &Records,
                          llvm::raw_ostream &OS);
void EmitOffloadImplFuncDecls(const llvm::RecordKeeper &Records,
                              llvm::raw_ostream &OS);
void EmitOffloadEntryPoints(const llvm::RecordKeeper &Records,
                            llvm::raw_ostream &OS);
void EmitOffloadPrintHeader(const llvm::RecordKeeper &Records,
                            llvm::raw_ostream &OS);
void EmitOffloadExports(const llvm::RecordKeeper &Records,
                        llvm::raw_ostream &OS);
