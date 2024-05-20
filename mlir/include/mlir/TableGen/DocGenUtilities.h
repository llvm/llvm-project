//===- DocGenUtilities.h - MLIR doc gen utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for generating documents from tablegen
// structures.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_DOCGENUTILITIES_H_
#define MLIR_TOOLS_MLIRTBLGEN_DOCGENUTILITIES_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tblgen {

// Emit the summary. To avoid confusion, the summary is styled differently from
// the description.
void emitSummary(llvm::StringRef summary, llvm::raw_ostream &os);

// Emit the description by aligning the text to the left per line (e.g.
// removing the minimum indentation across the block).
//
// This expects that the description in the tablegen file is already formatted
// in a way the user wanted but has some additional indenting due to being
// nested.
void emitDescription(llvm::StringRef description, llvm::raw_ostream &os);

// Emit the description as a C++ comment while realigning it.
void emitDescriptionComment(llvm::StringRef description, llvm::raw_ostream &os,
                            llvm::StringRef prefix = "");

void emitAttrOrTypeDefDoc(const llvm::RecordKeeper &recordKeeper,
                          llvm::raw_ostream &os,
                          llvm::StringRef recordTypeName);

void emitOpDoc(const llvm::RecordKeeper &recordKeeper, llvm::raw_ostream &os,
               const std::string &emitOpDoc, bool allowHugoSpecificFeatures,
               const std::string &opIncFilter, const std::string &opExcFilter);

bool emitDialectDoc(const llvm::RecordKeeper &recordKeeper,
                    llvm::raw_ostream &os, const std::string &selectedDialect,
                    const std::string &opIncFilter,
                    const std::string &opExcFilter,
                    const std::string &stripPrefix,
                    bool allowHugoSpecificFeatures);
void emitDocs(const llvm::RecordKeeper &recordKeeper, llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_DOCGENUTILITIES_H_
