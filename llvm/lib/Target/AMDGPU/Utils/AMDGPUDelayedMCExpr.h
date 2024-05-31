//===- AMDGPUDelayedMCExpr.h - Delayed MCExpr resolve -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUDELAYEDMCEXPR_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUDELAYEDMCEXPR_H

#include "llvm/BinaryFormat/MsgPackDocument.h"
#include <deque>

namespace llvm {
class MCExpr;

class DelayedMCExpr {
  struct DelayedExpr {
    msgpack::DocNode &DN;
    msgpack::Type Type;
    const MCExpr *Expr;
    DelayedExpr(msgpack::DocNode &DN, msgpack::Type Type, const MCExpr *Expr)
        : DN(DN), Type(Type), Expr(Expr) {}
  };

  std::deque<DelayedExpr> DelayedExprs;

public:
  bool resolveDelayedExpressions();
  void assignDocNode(msgpack::DocNode &DN, msgpack::Type Type,
                     const MCExpr *Expr);
  void clear();
  bool empty();
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUDELAYEDMCEXPR_H
