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
  DenseMap<BBOrMBB, BBOrMBB> EHPadUnwindMap;

  // Helper functions
  const BasicBlock *getEHPadUnwindDest(const BasicBlock *BB) const {
    return EHPadUnwindMap.lookup(BB).get<const BasicBlock *>();
  }
  void setEHPadUnwindDest(const BasicBlock *BB, const BasicBlock *Dest) {
    EHPadUnwindMap[BB] = Dest;
  }
  bool hasEHPadUnwindDest(const BasicBlock *BB) const {
    return EHPadUnwindMap.count(BB);
  }

  MachineBasicBlock *getEHPadUnwindDest(MachineBasicBlock *MBB) const {
    return EHPadUnwindMap.lookup(MBB).get<MachineBasicBlock *>();
  }
  void setEHPadUnwindDest(MachineBasicBlock *MBB, MachineBasicBlock *Dest) {
    EHPadUnwindMap[MBB] = Dest;
  }
  bool hasEHPadUnwindDest(MachineBasicBlock *MBB) const {
    return EHPadUnwindMap.count(MBB);
  }
};

// Analyze the IR in the given function to build WasmEHFuncInfo.
void calculateWasmEHInfo(const Function *F, WasmEHFuncInfo &EHInfo);

} // namespace llvm

#endif // LLVM_CODEGEN_WASMEHFUNCINFO_H
