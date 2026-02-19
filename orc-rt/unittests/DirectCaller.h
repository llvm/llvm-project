//===- DirectCaller.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_DIRECTCALLER_H
#define ORC_RT_UNITTEST_DIRECTCALLER_H

#include "orc-rt/WrapperFunction.h"

#include <memory>
#include <utility>

/// Make calls and call result handlers directly on the current thread.
class DirectCaller {
private:
  class DirectResultSender {
  public:
    virtual ~DirectResultSender() {}
    virtual void send(orc_rt_SessionRef S,
                      orc_rt::WrapperFunctionBuffer ResultBytes) = 0;
    static void send(orc_rt_SessionRef S, uint64_t CallId,
                     orc_rt_WrapperFunctionBuffer ResultBytes) {
      std::unique_ptr<DirectResultSender>(
          reinterpret_cast<DirectResultSender *>(
              static_cast<uintptr_t>(CallId)))
          ->send(S, orc_rt::WrapperFunctionBuffer(ResultBytes));
    }
  };

  template <typename ImplFn>
  class DirectResultSenderImpl : public DirectResultSender {
  public:
    DirectResultSenderImpl(ImplFn &&Fn) : Fn(std::forward<ImplFn>(Fn)) {}
    void send(orc_rt_SessionRef S,
              orc_rt::WrapperFunctionBuffer ResultBytes) override {
      Fn(std::move(ResultBytes));
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
  DirectCaller(orc_rt_SessionRef S, orc_rt_WrapperFunction Fn) : S(S), Fn(Fn) {}

  template <typename HandleResultFn>
  void operator()(HandleResultFn &&HandleResult,
                  orc_rt::WrapperFunctionBuffer ArgBytes) {
    auto DR =
        makeDirectResultSender(std::forward<HandleResultFn>(HandleResult));
    Fn(S, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DR.release())),
       DirectResultSender::send, ArgBytes.release());
  }

private:
  orc_rt_SessionRef S;
  orc_rt_WrapperFunction Fn;
};

#endif // ORC_RT_UNITTEST_DIRECTCALLER_H
