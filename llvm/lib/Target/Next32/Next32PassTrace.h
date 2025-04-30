//===-- Next32PassTrace.h - Next32 Pass tracing util ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions used in Next32 modules
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NEXT32_NEXT32PASSTRACE_H
#define LLVM_LIB_TARGET_NEXT32_NEXT32PASSTRACE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/Debug.h"

namespace llvm {

#define _PASS_TRACE(X) DEBUG_WITH_TYPE(DebugType.data(), X)
class Next32PassTrace {
  StringRef DebugType;
  MachineFunction &Func;

public:
  inline Next32PassTrace(StringRef DebugType, MachineFunction &Func)
      : DebugType(DebugType), Func(Func) {
#ifdef NDEBUG
    static_cast<void>(this->DebugType);
#endif
    _PASS_TRACE(dbgs() << "**** Start " << DebugType << " ****\n");
    _PASS_TRACE(Func.dump());
  }

  inline ~Next32PassTrace() {
    _PASS_TRACE(Func.dump());
    _PASS_TRACE(dbgs() << "**** End " << DebugType << " ****\n");
  }

  template <typename T> struct tracing_mbb_iterator : public T {
    StringRef DebugType;
    tracing_mbb_iterator(StringRef DebugType, T &&it)
        : T(std::move(it)), DebugType(DebugType) {}

    decltype(auto) operator*() const {
      auto &&MBB = T::operator*();
      _PASS_TRACE(dbgs() << "Processing MBB #" << MBB.getNumber() << "\n");
      return MBB;
    }

    decltype(auto) operator->() const {
      auto MBB = T::operator->();
      _PASS_TRACE(dbgs() << "Processing MBB #" << MBB->getNumber() << "\n");
      return MBB;
    }
  };
  using iterator = tracing_mbb_iterator<MachineFunction::iterator>;
  using const_iterator = tracing_mbb_iterator<MachineFunction::const_iterator>;

  iterator begin() { return iterator(DebugType, Func.begin()); }
  const_iterator begin() const {
    return const_iterator(DebugType, Func.begin());
  }
  iterator end() { return iterator(DebugType, Func.end()); }
  const_iterator end() const { return const_iterator(DebugType, Func.end()); }
};

} // namespace llvm
#endif // LLVM_LIB_TARGET_NEXT32_NEXT32PASSTRACE_H
