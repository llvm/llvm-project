//===--- QuerySession.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H
#define MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H

#include "mlir/IR/Operation.h"
#include "mlir/Query/Matcher/Registry.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir::query {

class Registry;
// Represents the state for a particular mlir-query session.
class QuerySession {
public:
  QuerySession(Operation *rootOp, llvm::SourceMgr &sourceMgr, unsigned bufferId,
               const matcher::Registry &matcherRegistry)
      : rootOp(rootOp), sourceMgr(sourceMgr), bufferId(bufferId),
        matcherRegistry(matcherRegistry) {}

  Operation *getRootOp() { return rootOp; }
  llvm::SourceMgr &getSourceManager() const { return sourceMgr; }
  unsigned getBufferId() { return bufferId; }
  const matcher::Registry &getRegistryData() const { return matcherRegistry; }

  llvm::StringMap<matcher::VariantValue> namedValues;
  bool terminate = false;

private:
  Operation *rootOp;
  llvm::SourceMgr &sourceMgr;
  unsigned bufferId;
  const matcher::Registry &matcherRegistry;
};

} // namespace mlir::query

#endif // MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H
