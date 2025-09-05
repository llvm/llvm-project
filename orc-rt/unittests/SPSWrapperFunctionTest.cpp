//===-- SPSWrapperFunctionTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SPSWrapperFunction and associated utilities.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/SPSWrapperFunction.h"
#include "orc-rt/WrapperFunction.h"
#include "orc-rt/move_only_function.h"

#include "gtest/gtest.h"

using namespace orc_rt;

/// Make calls and call result handlers directly on the current thread.
class DirectCaller {
private:
  class DirectResultSender {
  public:
    virtual ~DirectResultSender() {}
    virtual void send(orc_rt_SessionRef Session,
                      WrapperFunctionBuffer ResultBytes) = 0;
    static void send(orc_rt_SessionRef Session, void *CallCtx,
                     orc_rt_WrapperFunctionBuffer ResultBytes) {
      std::unique_ptr<DirectResultSender>(
          reinterpret_cast<DirectResultSender *>(CallCtx))
          ->send(Session, ResultBytes);
    }
  };

  template <typename ImplFn>
  class DirectResultSenderImpl : public DirectResultSender {
  public:
    DirectResultSenderImpl(ImplFn &&Fn) : Fn(std::forward<ImplFn>(Fn)) {}
    void send(orc_rt_SessionRef Session,
              WrapperFunctionBuffer ResultBytes) override {
      Fn(Session, std::move(ResultBytes));
    }

  private:
    std::decay_t<ImplFn> Fn;
  };

  template <typename ImplFn>
  static std::unique_ptr<DirectResultSender>
  makeDirectResultSender(ImplFn &&Fn) {
    return std::make_unique<DirectResultSenderImpl<ImplFn>>(
        std::forward<ImplFn>(Fn));
  }

public:
  DirectCaller(orc_rt_SessionRef Session, orc_rt_WrapperFunction Fn)
      : Session(Session), Fn(Fn) {}

  template <typename HandleResultFn>
  void operator()(HandleResultFn &&HandleResult,
                  WrapperFunctionBuffer ArgBytes) {
    auto DR =
        makeDirectResultSender(std::forward<HandleResultFn>(HandleResult));
    Fn(Session, reinterpret_cast<void *>(DR.release()),
       DirectResultSender::send, ArgBytes.release());
  }

private:
  orc_rt_SessionRef Session;
  orc_rt_WrapperFunction Fn;
};

static void void_noop_sps_wrapper(orc_rt_SessionRef Session, void *CallCtx,
                                  orc_rt_WrapperFunctionReturn Return,
                                  orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<void()>::handle(
      Session, CallCtx, Return, ArgBytes,
      [](move_only_function<void()> Return) { Return(); });
}

TEST(SPSWrapperFunctionUtilsTest, TestVoidNoop) {
  bool Ran = false;
  SPSWrapperFunction<void()>::call(DirectCaller(nullptr, void_noop_sps_wrapper),
                                   [&](Error Err) {
                                     cantFail(std::move(Err));
                                     Ran = true;
                                   });
  EXPECT_TRUE(Ran);
}

static void add_sps_wrapper(orc_rt_SessionRef Session, void *CallCtx,
                            orc_rt_WrapperFunctionReturn Return,
                            orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::handle(
      Session, CallCtx, Return, ArgBytes,
      [](move_only_function<void(int32_t)> Return, int32_t X, int32_t Y) {
        Return(X + Y);
      });
}

TEST(SPSWrapperFunctionUtilsTest, TestAdd) {
  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      DirectCaller(nullptr, add_sps_wrapper),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);
  EXPECT_EQ(Result, 42);
}
