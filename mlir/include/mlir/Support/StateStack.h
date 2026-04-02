//===- StateStack.h - Utility for storing a stack of state ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for storing a stack of generic context.
// The context can be arbitrary data, possibly including file-scoped types. Data
// must be derived from StateStackFrameBase and implement MLIR TypeID.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_STACKFRAME_H
#define MLIR_SUPPORT_STACKFRAME_H

#include "mlir/Support/TypeID.h"
#include "mlir/Support/WalkResult.h"
#include <memory>

namespace mlir {

/// Common CRTP base class for StateStack frames.
class StateStackFrame {
public:
  virtual ~StateStackFrame() = default;
  TypeID getTypeID() const { return typeID; }

protected:
  explicit StateStackFrame(TypeID typeID) : typeID(typeID) {}

private:
  const TypeID typeID;
  virtual void anchor();
};

/// Concrete CRTP base class for StateStack frames. This is used for keeping a
/// stack of common state useful for recursive IR conversions. For example, when
/// translating operations with regions, users of StateStack can store state on
/// StateStack before entering the region and inspect it when converting
/// operations nested within that region. Users are expected to derive this
/// class and put any relevant information into fields of the derived class. The
/// usual isa/dyn_cast functionality is available for instances of derived
/// classes.
template <typename Derived>
class StateStackFrameBase : public StateStackFrame {
public:
  explicit StateStackFrameBase() : StateStackFrame(TypeID::get<Derived>()) {}
};

class StateStack {
public:
  /// Creates a stack frame of type `T` on StateStack. `T` must
  /// be derived from `StackFrameBase<T>` and constructible from the provided
  /// arguments. Doing this before entering the region of the op being
  /// translated makes the frame available when translating ops within that
  /// region.
  template <typename T, typename... Args>
  void stackPush(Args &&...args) {
    static_assert(std::is_base_of<StateStackFrame, T>::value,
                  "can only push instances of StackFrame on StateStack");
    stack.push_back(std::make_unique<T>(std::forward<Args>(args)...));
  }

  /// Pops the last element from the StateStack.
  void stackPop() { stack.pop_back(); }

  /// Calls `callback` for every StateStack frame of type `T`
  /// starting from the top of the stack.
  template <typename T>
  WalkResult stackWalk(llvm::function_ref<WalkResult(T &)> callback) {
    static_assert(std::is_base_of<StateStackFrame, T>::value,
                  "expected T derived from StackFrame");
    if (!callback)
      return WalkResult::skip();
    for (std::unique_ptr<StateStackFrame> &frame : llvm::reverse(stack)) {
      if (T *ptr = dyn_cast_or_null<T>(frame.get())) {
        WalkResult result = callback(*ptr);
        if (result.wasInterrupted())
          return result;
      }
    }
    return WalkResult::advance();
  }

  /// Get the top instance of frame type `T` or nullptr if none are found
  template <typename T>
  T *getStackTop() {
    T *top = nullptr;
    stackWalk<T>([&](T &frame) -> mlir::WalkResult {
      top = &frame;
      return mlir::WalkResult::interrupt();
    });
    return top;
  }

private:
  SmallVector<std::unique_ptr<StateStackFrame>> stack;
};

/// RAII object calling stackPush/stackPop on construction/destruction.
/// HostClass could be a StateStack or some other class which forwards calls to
/// one.
template <typename T, typename HostClass = StateStack>
struct SaveStateStack {
  template <typename... Args>
  explicit SaveStateStack(HostClass &host, Args &&...args) : host(host) {
    host.template stackPush<T>(std::forward<Args>(args)...);
  }
  ~SaveStateStack() { host.stackPop(); }

private:
  HostClass &host;
};

} // namespace mlir

namespace llvm {
template <typename T>
struct isa_impl<T, ::mlir::StateStackFrame> {
  static inline bool doit(const ::mlir::StateStackFrame &frame) {
    return frame.getTypeID() == ::mlir::TypeID::get<T>();
  }
};
} // namespace llvm

#endif // MLIR_SUPPORT_STACKFRAME_H
