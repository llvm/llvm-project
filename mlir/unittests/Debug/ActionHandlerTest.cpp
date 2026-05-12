//===- ActionHandlerTest.cpp - Debug Action Handler Tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Action.h"
#include "mlir/Support/TypeID.h"
#include "gmock/gmock.h"

#include <gtest/gtest.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Action.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Debug/ExecutionContext.h>
#include <llvm/ADT/StringRef.h>

#include <memory>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::tracing;

namespace {

struct DummyAction final : ActionImpl<DummyAction> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DummyAction)
    static constexpr StringLiteral tag = "dummy-action";
};

} // namespace

namespace {

// State class — lives on the heap, shared across all copies of the handler
struct HandlerState {
    bool enabled{true};
};

/// Owner of a shared_ptr to the state
/// Every copy of the functor points at the same HandlerState object.
struct StatefulHandler {
    std::shared_ptr<HandlerState> state;

    void operator()(mlir::function_ref<void()> actionFn,
                    const mlir::tracing::Action& /*action*/) const {
        if (!state->enabled) {
            // Skip execution entirely when disabled.
            return;
        }
        actionFn();
    }
};

TEST(ActionHandlerSharedState, SingleCopyEnabledState) {
    mlir::MLIRContext ctx;

    ctx.registerActionHandler(StatefulHandler{std::make_shared<HandlerState>()});

    // Retrieve a copy of the handler.
    auto handlerCopy = ctx.getActionHandler();
    ASSERT_TRUE(static_cast<bool>(handlerCopy));

    int executionCount = 0;
    auto workFn = [&]() { ++executionCount; };

    DummyAction action;

    handlerCopy(workFn, action);

    // Recover the shared_ptr from the handler copy via target<StatefulHandler>().
    // target<T>() returns a non-null pointer only when the stored callable type
    // matches T exactly — which is guaranteed here since we registered StatefulHandler.
    auto* recovered = handlerCopy.target<StatefulHandler>();
    ASSERT_NE(recovered, nullptr);

    EXPECT_EQ(executionCount, 1);
    EXPECT_TRUE(recovered->state->enabled == true);
}

TEST(ActionHandlerSharedState, MultipleCopiesDisabledState) {
    mlir::MLIRContext ctx;

    ctx.registerActionHandler(StatefulHandler{std::make_shared<HandlerState>()});

    // Recover the shared_ptr and disable the state
    auto handlerCopy = ctx.getActionHandler();
    auto* recovered = handlerCopy.target<StatefulHandler>();
    ASSERT_NE(recovered, nullptr);
    recovered->state->enabled = false;

    // A second independent copy also sees enabled==false
    auto handlerCopy2 = ctx.getActionHandler();

    int executionCount = 0;
    auto workFn = [&]() { ++executionCount; };

    DummyAction action;

    handlerCopy2(workFn, action);

    // workFn was skipped because the handler saw enabled==false
    EXPECT_EQ(executionCount, 0);
    EXPECT_TRUE(recovered->state->enabled == false);
}

} // namespace
