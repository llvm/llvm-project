//===-- llvm/CodeGen/SelectionDAGAddressAnalysis.h  ------- DAG Address Analysis
//---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_CODEGEN_SELECTIONDAGADDRESSANALYSIS_H
#define LLVM_CODEGEN_SELECTIONDAGADDRESSANALYSIS_H

#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"

namespace llvm {
/// Helper struct to parse and store a memory address as base + index + offset.
/// We ignore sign extensions when it is safe to do so.
/// The following two expressions are not equivalent. To differentiate we need
/// to store whether there was a sign extension involved in the index
/// computation.
///  (load (i64 add (i64 copyfromreg %c)
///                 (i64 signextend (add (i8 load %index)
///                                      (i8 1))))
/// vs
///
/// (load (i64 add (i64 copyfromreg %c)
///                (i64 signextend (i32 add (i32 signextend (i8 load %index))
///                                         (i32 1)))))
class BaseIndexOffset {
private:
  SDValue Base;
  SDValue Index;
  int64_t Offset;
  bool IsIndexSignExt;

public:
  BaseIndexOffset() : Offset(0), IsIndexSignExt(false) {}

  BaseIndexOffset(SDValue Base, SDValue Index, int64_t Offset,
                  bool IsIndexSignExt)
      : Base(Base), Index(Index), Offset(Offset),
        IsIndexSignExt(IsIndexSignExt) {}

  SDValue getBase() { return Base; }
  SDValue getIndex() { return Index; }

  bool equalBaseIndex(BaseIndexOffset &Other, const SelectionDAG &DAG) {
    int64_t Off;
    return equalBaseIndex(Other, DAG, Off);
  }

  bool equalBaseIndex(BaseIndexOffset &Other, const SelectionDAG &DAG,
                      int64_t &Off);

  /// Parses tree in Ptr for base, index, offset addresses.
  static BaseIndexOffset match(SDValue Ptr, const SelectionDAG &DAG);
};
} // namespace llvm

#endif
