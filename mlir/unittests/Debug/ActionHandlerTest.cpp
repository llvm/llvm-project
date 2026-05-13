//===- ActionHandlerTest.cpp - Debug Action Handler Tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Action.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

#include <memory>

using namespace mlir;
using namespace mlir::tracing;

namespace {

struct DummyAction final : ActionImpl<DummyAction> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DummyAction)
    static constexpr StringLiteral tag = "dummy-action";

    DummyAction(llvm::ArrayRef<IRUnit> irUnits) {}
};

} // namespace

namespace {

// State class
struct HandlerState {
    bool enabled{true};
};

// Owner of a shared_ptr to the state
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

TEST(ActionHandlerSharedState, EnabledState) {
    mlir::MLIRContext ctx;

    ctx.registerActionHandler(StatefulHandler{std::make_shared<HandlerState>()});

    auto handlerRef = ctx.getActionHandler();
    ASSERT_TRUE(static_cast<bool>(handlerRef));

    int executionCount = 0;
    auto workFn = [&]() { ++executionCount; };

    ctx.executeAction<DummyAction>(workFn, {});

    // Recover the shared_ptr from the handler via target<StatefulHandler>().
    // target<T>() returns a non-null pointer only when the stored callable type
    // matches T exactly — which is guaranteed here since we registered StatefulHandler.
    auto* recovered = handlerRef.target<StatefulHandler>();
    ASSERT_NE(recovered, nullptr);

    EXPECT_EQ(executionCount, 1);
    EXPECT_TRUE(recovered->state->enabled == true);
}

TEST(ActionHandlerSharedState, DisabledState) {
    mlir::MLIRContext ctx;

    ctx.registerActionHandler(StatefulHandler{std::make_shared<HandlerState>()});

    // Recover the shared_ptr and disable the state
    auto handlerRef = ctx.getActionHandler();
    auto* recovered = handlerRef.target<StatefulHandler>();
    ASSERT_NE(recovered, nullptr);
    recovered->state->enabled = false;

    int executionCount = 0;
    auto workFn = [&]() { ++executionCount; };

    ctx.executeAction<DummyAction>(workFn, {});

    // workFn was skipped because the handler saw enabled==false
    EXPECT_EQ(executionCount, 0);
    EXPECT_TRUE(recovered->state->enabled == false);
}

} // namespace
