//===--- llvm/CodeGen/WasmEHFuncInfo.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Data structures for Wasm exception handling schemes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_WASMEHFUNCINFO_H
#define LLVM_CODEGEN_WASMEHFUNCINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"

namespace llvm {

class BasicBlock;
class Function;
class MachineBasicBlock;

namespace WebAssembly {
enum EventTag { CPP_EXCEPTION = 0, C_LONGJMP = 1 };
}

using BBOrMBB = PointerUnion<const BasicBlock *, MachineBasicBlock *>;

struct WasmEHFuncInfo {
  // When there is an entry <A, B>, if an exception is not caught by A, it
  // should next unwind to the EH pad B.
  DenseMap<BBOrMBB, BBOrMBB> SrcToUnwindDest;

  // Helper functions
  const BasicBlock *getUnwindDest(const BasicBlock *BB) const {
    return SrcToUnwindDest.lookup(BB).get<const BasicBlock *>();
  }
  void setUnwindDest(const BasicBlock *BB, const BasicBlock *Dest) {
    SrcToUnwindDest[BB] = Dest;
  }
  bool hasUnwindDest(const BasicBlock *BB) const {
    return SrcToUnwindDest.count(BB);
  }

  MachineBasicBlock *getUnwindDest(MachineBasicBlock *MBB) const {
    return SrcToUnwindDest.lookup(MBB).get<MachineBasicBlock *>();
  }
  void setUnwindDest(MachineBasicBlock *MBB, MachineBasicBlock *Dest) {
    SrcToUnwindDest[MBB] = Dest;
  }
  bool hasUnwindDest(MachineBasicBlock *MBB) const {
    return SrcToUnwindDest.count(MBB);
  }
};

// Analyze the IR in the given function to build WasmEHFuncInfo.
void calculateWasmEHInfo(const Function *F, WasmEHFuncInfo &EHInfo);

} // namespace llvm

#endif // LLVM_CODEGEN_WASMEHFUNCINFO_H
