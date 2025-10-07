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

#include "CommonTestUtils.h"

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

TEST(SPSWrapperFunctionUtilsTest, VoidNoop) {
  bool Ran = false;
  SPSWrapperFunction<void()>::call(DirectCaller(nullptr, void_noop_sps_wrapper),
                                   [&](Error Err) {
                                     cantFail(std::move(Err));
                                     Ran = true;
                                   });
  EXPECT_TRUE(Ran);
}

static void add_via_lambda_sps_wrapper(orc_rt_SessionRef Session, void *CallCtx,
                                       orc_rt_WrapperFunctionReturn Return,
                                       orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::handle(
      Session, CallCtx, Return, ArgBytes,
      [](move_only_function<void(int32_t)> Return, int32_t X, int32_t Y) {
        Return(X + Y);
      });
}

TEST(SPSWrapperFunctionUtilsTest, BinaryOpViaLambda) {
  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      DirectCaller(nullptr, add_via_lambda_sps_wrapper),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);
  EXPECT_EQ(Result, 42);
}

static void add_via_function(move_only_function<void(int32_t)> Return,
                             int32_t X, int32_t Y) {
  Return(X + Y);
}

static void
add_via_function_sps_wrapper(orc_rt_SessionRef Session, void *CallCtx,
                             orc_rt_WrapperFunctionReturn Return,
                             orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::handle(
      Session, CallCtx, Return, ArgBytes, add_via_function);
}

TEST(SPSWrapperFunctionUtilsTest, BinaryOpViaFunction) {
  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      DirectCaller(nullptr, add_via_function_sps_wrapper),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);
  EXPECT_EQ(Result, 42);
}

static void
add_via_function_pointer_sps_wrapper(orc_rt_SessionRef Session, void *CallCtx,
                                     orc_rt_WrapperFunctionReturn Return,
                                     orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::handle(
      Session, CallCtx, Return, ArgBytes, &add_via_function);
}

TEST(SPSWrapperFunctionUtilsTest, BinaryOpViaFunctionPointer) {
  int32_t Result = 0;
  SPSWrapperFunction<int32_t(int32_t, int32_t)>::call(
      DirectCaller(nullptr, add_via_function_pointer_sps_wrapper),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); }, 41, 1);
  EXPECT_EQ(Result, 42);
}

static void improbable_feat_sps_wrapper(orc_rt_SessionRef Session,
                                        void *CallCtx,
                                        orc_rt_WrapperFunctionReturn Return,
                                        orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<SPSError(bool)>::handle(
      Session, CallCtx, Return, ArgBytes,
      [](move_only_function<void(Error)> Return, bool LuckyHat) {
        if (LuckyHat)
          Return(Error::success());
        else
          Return(make_error<StringError>("crushed by boulder"));
      });
}

TEST(SPSWrapperFunctionUtilsTest, TransparentConversionErrorSuccessCase) {
  bool DidRun = false;
  SPSWrapperFunction<SPSError(bool)>::call(
      DirectCaller(nullptr, improbable_feat_sps_wrapper),
      [&](Expected<Error> E) {
        DidRun = true;
        cantFail(cantFail(std::move(E)));
      },
      true);

  EXPECT_TRUE(DidRun);
}

TEST(SPSWrapperFunctionUtilsTest, TransparentConversionErrorFailureCase) {
  std::string ErrMsg;
  SPSWrapperFunction<SPSError(bool)>::call(
      DirectCaller(nullptr, improbable_feat_sps_wrapper),
      [&](Expected<Error> E) { ErrMsg = toString(cantFail(std::move(E))); },
      false);

  EXPECT_EQ(ErrMsg, "crushed by boulder");
}

static void halve_number_sps_wrapper(orc_rt_SessionRef Session, void *CallCtx,
                                     orc_rt_WrapperFunctionReturn Return,
                                     orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<SPSExpected<int32_t>(int32_t)>::handle(
      Session, CallCtx, Return, ArgBytes,
      [](move_only_function<void(Expected<int32_t>)> Return, int N) {
        if (N % 2 == 0)
          Return(N >> 1);
        else
          Return(make_error<StringError>("N is not a multiple of 2"));
      });
}

TEST(SPSWrapperFunctionUtilsTest, TransparentConversionExpectedSuccessCase) {
  int32_t Result = 0;
  SPSWrapperFunction<SPSExpected<int32_t>(int32_t)>::call(
      DirectCaller(nullptr, halve_number_sps_wrapper),
      [&](Expected<Expected<int32_t>> R) {
        Result = cantFail(cantFail(std::move(R)));
      },
      2);

  EXPECT_EQ(Result, 1);
}

TEST(SPSWrapperFunctionUtilsTest, TransparentConversionExpectedFailureCase) {
  std::string ErrMsg;
  SPSWrapperFunction<SPSExpected<int32_t>(int32_t)>::call(
      DirectCaller(nullptr, halve_number_sps_wrapper),
      [&](Expected<Expected<int32_t>> R) {
        ErrMsg = toString(cantFail(std::move(R)).takeError());
      },
      3);

  EXPECT_EQ(ErrMsg, "N is not a multiple of 2");
}

static void
round_trip_int_pointer_sps_wrapper(orc_rt_SessionRef Session, void *CallCtx,
                                   orc_rt_WrapperFunctionReturn Return,
                                   orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<SPSExecutorAddr(SPSExecutorAddr)>::handle(
      Session, CallCtx, Return, ArgBytes,
      [](move_only_function<void(int32_t *)> Return, int32_t *P) {
        Return(P);
      });
}

TEST(SPSWrapperFunctionUtilsTest, TransparentSerializationPointers) {
  int X = 42;
  int *P = nullptr;
  SPSWrapperFunction<SPSExecutorAddr(SPSExecutorAddr)>::call(
      DirectCaller(nullptr, round_trip_int_pointer_sps_wrapper),
      [&](Expected<int32_t *> R) { P = cantFail(std::move(R)); }, &X);

  EXPECT_EQ(P, &X);
}

template <size_t N> struct SPSOpCounter {};

namespace orc_rt {
template <size_t N>
class SPSSerializationTraits<SPSOpCounter<N>, OpCounter<N>> {
public:
  static size_t size(const OpCounter<N> &O) { return 0; }
  static bool serialize(SPSOutputBuffer &OB, const OpCounter<N> &O) {
    return true;
  }
  static bool deserialize(SPSInputBuffer &OB, OpCounter<N> &O) { return true; }
};
} // namespace orc_rt

static void
handle_with_reference_types_sps_wrapper(orc_rt_SessionRef Session,
                                        void *CallCtx,
                                        orc_rt_WrapperFunctionReturn Return,
                                        orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<void(
      SPSOpCounter<0>, SPSOpCounter<1>, SPSOpCounter<2>,
      SPSOpCounter<3>)>::handle(Session, CallCtx, Return, ArgBytes,
                                [](move_only_function<void()> Return,
                                   OpCounter<0>, OpCounter<1> &,
                                   const OpCounter<2> &,
                                   OpCounter<3> &&) { Return(); });
}

TEST(SPSWrapperFunctionUtilsTest, HandlerWithReferences) {
  // Test that we can handle by-value, by-ref, by-const-ref, and by-rvalue-ref
  // arguments, and that we generate the expected number of moves.
  OpCounter<0>::reset();
  OpCounter<1>::reset();
  OpCounter<2>::reset();
  OpCounter<3>::reset();

  bool DidRun = false;
  SPSWrapperFunction<void(SPSOpCounter<0>, SPSOpCounter<1>, SPSOpCounter<2>,
                          SPSOpCounter<3>)>::
      call(
          DirectCaller(nullptr, handle_with_reference_types_sps_wrapper),
          [&](Error R) {
            cantFail(std::move(R));
            DidRun = true;
          },
          OpCounter<0>(), OpCounter<1>(), OpCounter<2>(), OpCounter<3>());

  EXPECT_TRUE(DidRun);

  // We expect two default constructions for each parameter: one for the
  // argument to call, and one for the object to deserialize into.
  EXPECT_EQ(OpCounter<0>::defaultConstructions(), 2U);
  EXPECT_EQ(OpCounter<1>::defaultConstructions(), 2U);
  EXPECT_EQ(OpCounter<2>::defaultConstructions(), 2U);
  EXPECT_EQ(OpCounter<3>::defaultConstructions(), 2U);

  // Pass-by-value: we expect two moves (one for SPS transparent conversion,
  // one to copy the value to the parameter), and no copies.
  EXPECT_EQ(OpCounter<0>::moves(), 2U);
  EXPECT_EQ(OpCounter<0>::copies(), 0U);

  // Pass-by-lvalue-reference: we expect one move (for SPS transparent
  // conversion), no copies.
  EXPECT_EQ(OpCounter<1>::moves(), 1U);
  EXPECT_EQ(OpCounter<1>::copies(), 0U);

  // Pass-by-const-lvalue-reference: we expect one move (for SPS transparent
  // conversion), no copies.
  EXPECT_EQ(OpCounter<2>::moves(), 1U);
  EXPECT_EQ(OpCounter<2>::copies(), 0U);

  // Pass-by-rvalue-reference: we expect one move (for SPS transparent
  // conversion), no copies.
  EXPECT_EQ(OpCounter<3>::moves(), 1U);
  EXPECT_EQ(OpCounter<3>::copies(), 0U);
}

namespace {
class Adder {
public:
  int32_t addSync(int32_t X, int32_t Y) { return X + Y; }
  void addAsync(move_only_function<void(int32_t)> Return, int32_t X,
                int32_t Y) {
    Return(addSync(X, Y));
  }
};
} // anonymous namespace

static void adder_add_async_sps_wrapper(orc_rt_SessionRef Session,
                                        void *CallCtx,
                                        orc_rt_WrapperFunctionReturn Return,
                                        orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<int32_t(SPSExecutorAddr, int32_t, int32_t)>::handle(
      Session, CallCtx, Return, ArgBytes,
      WrapperFunction::handleWithAsyncMethod(&Adder::addAsync));
}

TEST(SPSWrapperFunctionUtilsTest, HandleWtihAsyncMethod) {
  auto A = std::make_unique<Adder>();
  int32_t Result = 0;
  SPSWrapperFunction<int32_t(SPSExecutorAddr, int32_t, int32_t)>::call(
      DirectCaller(nullptr, adder_add_async_sps_wrapper),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); },
      ExecutorAddr::fromPtr(A.get()), 41, 1);

  EXPECT_EQ(Result, 42);
}

static void adder_add_sync_sps_wrapper(orc_rt_SessionRef Session, void *CallCtx,
                                       orc_rt_WrapperFunctionReturn Return,
                                       orc_rt_WrapperFunctionBuffer ArgBytes) {
  SPSWrapperFunction<int32_t(SPSExecutorAddr, int32_t, int32_t)>::handle(
      Session, CallCtx, Return, ArgBytes,
      WrapperFunction::handleWithSyncMethod(&Adder::addSync));
}

TEST(SPSWrapperFunctionUtilsTest, HandleWithSyncMethod) {
  auto A = std::make_unique<Adder>();
  int32_t Result = 0;
  SPSWrapperFunction<int32_t(SPSExecutorAddr, int32_t, int32_t)>::call(
      DirectCaller(nullptr, adder_add_sync_sps_wrapper),
      [&](Expected<int32_t> R) { Result = cantFail(std::move(R)); },
      ExecutorAddr::fromPtr(A.get()), 41, 1);

  EXPECT_EQ(Result, 42);
}
