//===-- SystemRuntimeMetaCoroutine.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYSTEMRUNTIME_METACOROUTINE_SYSTEMRUNTIMEMETACOROUTINE_H
#define LLDB_SOURCE_PLUGINS_SYSTEMRUNTIME_METACOROUTINE_SYSTEMRUNTIMEMETACOROUTINE_H

#include "llvm/ADT/StringRef.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/QueueItem.h"
#include "lldb/Target/SystemRuntime.h"
#include "lldb/Utility/ConstString.h"

class SystemRuntimeMetaCoroutine : public lldb_private::SystemRuntime {
public:
  SystemRuntimeMetaCoroutine(lldb_private::Process *process);

  ~SystemRuntimeMetaCoroutine() override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "systemruntime-metacoroutine";
  }

  static lldb_private::SystemRuntime *
  CreateInstance(lldb_private::Process *process);

  const std::vector<lldb_private::ConstString> &
  GetExtendedBacktraceTypes() override;

  lldb::ThreadSP
  GetExtendedBacktraceThread(lldb::ThreadSP thread,
                             lldb_private::ConstString type) override;

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

private:
  SystemRuntimeMetaCoroutine(const SystemRuntimeMetaCoroutine &) = delete;
  const SystemRuntimeMetaCoroutine &
  operator=(const SystemRuntimeMetaCoroutine &) = delete;
};

#endif // LLDB_SOURCE_PLUGINS_SYSTEMRUNTIME_METACOROUTINE_SYSTEMRUNTIMEMETACOROUTINE_H
