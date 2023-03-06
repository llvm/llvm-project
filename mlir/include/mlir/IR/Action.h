//===- Action.h -  Action Support ---------------------*- C++ -*-=============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the action framework. This framework
// allows for external entities to control certain actions taken by the compiler
// by registering handler functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ACTION_H
#define MLIR_IR_ACTION_H

#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <type_traits>

namespace mlir {
namespace tracing {

/// An action is a specific action that is to be taken by the compiler,
/// that can be toggled and controlled by an external user. There are no
/// constraints on the granularity of an action, it could be as simple as
/// "perform this fold" and as complex as "run this pass pipeline". Via template
/// parameters `ParameterTs`, a user may provide the set of argument types that
/// are provided when handling a query on this action.
///
/// This class represents the base class of the ActionImpl class (see below).
/// This holds the template-invariant elements of the Action class.
class Action {
public:
  virtual ~Action() = default;

  /// Return the unique action id of this action, use for casting
  /// functionality.
  TypeID getActionID() const { return actionID; }

  /// Return a string "tag" which intends to uniquely identify this type of
  /// action. For example "pass-application" or "pattern-rewrite".
  virtual StringRef getTag() const = 0;

  virtual void print(raw_ostream &os) const {
    os << "Action \"" << getTag() << "\"";
  }

protected:
  Action(TypeID actionID) : actionID(actionID) {}

  /// The type of the derived action class, used for `isa`/`dyn_cast`.
  TypeID actionID;
};

//===----------------------------------------------------------------------===//
// ActionManager
//===----------------------------------------------------------------------===//

/// This class represents manages actions, and orchestrates the
/// communication between action queries and action handlers. An action handler
/// is either an action specific handler, i.e. a derived class of
/// `MyActionType::Handler`, or a generic handler, i.e. a derived class of
/// `ActionManager::GenericHandler`. For more details on action specific
/// handlers, see the definition of `Action::Handler` below. For more
/// details on generic handlers, see `ActionManager::GenericHandler` below.
class ActionManager {
public:
  //===--------------------------------------------------------------------===//
  // Handlers
  //===--------------------------------------------------------------------===//

  /// This class represents the base class of an action handler.
  class HandlerBase {
  public:
    virtual ~HandlerBase() = default;

    /// Return the unique handler id of this handler, use for casting
    /// functionality.
    TypeID getHandlerID() const { return handlerID; }

  protected:
    HandlerBase(TypeID handlerID) : handlerID(handlerID) {}

    /// The type of the derived handler class. This allows for detecting if a
    /// handler can handle a given action type.
    TypeID handlerID;
  };

  /// This class represents a generic action handler. A generic handler allows
  /// for handling any action type. Handlers of this type are useful for
  /// implementing general functionality that doesn't necessarily need to
  /// interpret the exact action parameters, or can rely on an external
  /// interpreter (such as the user). Given that these handlers are generic,
  /// they take a set of opaque parameters that try to map the context of the
  /// action type in a generic way.
  class GenericHandler : public HandlerBase {
  public:
    GenericHandler() : HandlerBase(TypeID::get<GenericHandler>()) {}

    /// This hook allows for controlling the execution of an action. It should
    /// return failure if the handler could not process the action, or whether
    /// the `transform` was executed or not.
    virtual FailureOr<bool> execute(function_ref<void()> transform,
                                    const Action &action) {
      return failure();
    }

    /// Provide classof to allow casting between handler types.
    static bool classof(const ActionManager::HandlerBase *handler) {
      return handler->getHandlerID() == TypeID::get<GenericHandler>();
    }
  };

  /// Register the given action handler with the manager.
  void registerActionHandler(std::unique_ptr<HandlerBase> handler) {
    actionHandlers.emplace_back(std::move(handler));
  }
  template <typename T>
  void registerActionHandler() {
    registerActionHandler(std::make_unique<T>());
  }

  //===--------------------------------------------------------------------===//
  // Action Queries
  //===--------------------------------------------------------------------===//

  /// Dispatch an action represented by the `transform` callback. If no handler
  /// is found, the `transform` callback is invoked directly.
  /// Return true if the action was executed, false otherwise.
  template <typename ActionType, typename... Args>
  bool execute(function_ref<void()> transform, Args &&...args) {
    if (actionHandlers.empty()) {
      transform();
      return true;
    }

    // Invoke the `execute` method on the provided handler.
    auto executeFn = [&](auto *handler, auto &&...handlerParams) {
      return handler->execute(
          transform,
          ActionType(std::forward<decltype(handlerParams)>(handlerParams)...));
    };
    FailureOr<bool> result = dispatchToHandler<ActionType, bool>(
        executeFn, std::forward<Args>(args)...);
    // no handler found, execute the transform directly.
    if (failed(result)) {
      transform();
      return true;
    }

    // Return the result of the handler.
    return *result;
  }

private:
  //===--------------------------------------------------------------------===//
  // Query to Handler Dispatch
  //===--------------------------------------------------------------------===//

  /// Dispath a given callback on any handlers that are able to process queries
  /// on the given action type. This method returns failure if no handlers could
  /// process the action, or success(with a result) if a handler processed the
  /// action.
  template <typename ActionType, typename ResultT, typename HandlerCallbackT,
            typename... Args>
  FailureOr<ResultT> dispatchToHandler(HandlerCallbackT &&handlerCallback,
                                       Args &&...args) {
    static_assert(ActionType::template canHandleWith<Args...>(),
                  "cannot execute action with the given set of parameters");

    // Process any generic or action specific handlers.
    // The first handler that gives us a result is the one that we will return.
    for (std::unique_ptr<HandlerBase> &it : reverse(actionHandlers)) {
      FailureOr<ResultT> result = failure();
      if (auto *handler = dyn_cast<typename ActionType::Handler>(&*it)) {
        result = handlerCallback(handler, std::forward<Args>(args)...);
      } else if (auto *genericHandler = dyn_cast<GenericHandler>(&*it)) {
        result = handlerCallback(genericHandler, std::forward<Args>(args)...);
      }

      // If the handler succeeded, return the result. Otherwise, try a new
      // handler.
      if (succeeded(result))
        return result;
    }
    return failure();
  }

  /// The set of action handlers that have been registered with the manager.
  SmallVector<std::unique_ptr<HandlerBase>> actionHandlers;
};

/// CRTP Implementation of an action. This class provides a base class for
/// implementing specific actions.
///  Derived classes are expected to provide the following:
///   * static constexpr StringLiteral tag = "...";
///     - This method returns a unique string identifier, similar to a command
///       line flag or DEBUG_TYPE.
template <typename Derived, typename... ParameterTs>
class ActionImpl : public Action {
public:
  ActionImpl() : Action(TypeID::get<Derived>()) {}

  /// Provide classof to allow casting between action types.
  static bool classof(const Action *action) {
    return action->getActionID() == TypeID::get<Derived>();
  }

  /// Forward tag access to the derived class.
  StringRef getTag() const final { return Derived::tag; }

  class Handler : public ActionManager::HandlerBase {
  public:
    Handler() : HandlerBase(TypeID::get<Derived>()) {}

    /// This hook allows for controlling the execution of an action.
    /// `parameters` correspond to the set of values provided by the
    /// action as context. It should return failure if the handler could not
    /// process the action, passing it to the next registered handler.
    virtual FailureOr<bool> execute(function_ref<void()> transform,
                                    const Derived &action) {
      return failure();
    }

    /// Provide classof to allow casting between handler types.
    static bool classof(const ActionManager::HandlerBase *handler) {
      return handler->getHandlerID() == TypeID::get<Derived>();
    }
  };

private:
  /// Returns true if the action can be handled within the given set of
  /// parameter types.
  template <typename... CallerParameterTs>
  static constexpr bool canHandleWith() {
    return std::is_invocable_v<function_ref<void(ParameterTs...)>,
                               CallerParameterTs...>;
  }

  /// Allow access to `canHandleWith`.
  friend class ActionManager;
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_IR_ACTION_H
