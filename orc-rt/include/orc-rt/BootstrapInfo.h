//===- BootstrapInfo.h - Bootstrap syms & values for controller -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BootstrapInfo API.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BOOTSTRAPINFO_H
#define ORC_RT_BOOTSTRAPINFO_H

#include "orc-rt/Error.h"
#include "orc-rt/SimpleSymbolTable.h"
#include "orc-rt/move_only_function.h"
#include "orc-rt/sps-ci/AllSPSCI.h"
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace orc_rt {

class ExecutorProcessInfo;
class Session;

/// Holds initial values that will be used to bootstrap the controller's
/// ExecutorProcessControl object.
class BootstrapInfo {
public:
  using ValueMap = std::unordered_map<std::string, std::string>;
  using InitialSymbolsBuilder = move_only_function<Error(SimpleSymbolTable &)>;
  using InitialValuesBuilder = move_only_function<Error(ValueMap &)>;

  /// Construct a BootstrapInfo object from the given Session, Symbols, and
  /// Values.
  BootstrapInfo(Session &S, SimpleSymbolTable Symbols = {},
                ValueMap Values = {});

  /// Construct with a default initial symbols and values.
  static Expected<BootstrapInfo>
  CreateDefault(Session &S,
                InitialSymbolsBuilder AddInitialSymbols = sps_ci::addAll,
                InitialValuesBuilder AddInitialValues = {});

  const Session &session() const noexcept { return S; }

  const ExecutorProcessInfo &processInfo() const noexcept;

  SimpleSymbolTable &symbols() noexcept { return Symbols; }
  const SimpleSymbolTable &symbols() const noexcept { return Symbols; }

  ValueMap &values() noexcept { return Values; }
  const ValueMap &values() const noexcept { return Values; }

private:
  Session &S;
  SimpleSymbolTable Symbols;
  ValueMap Values;
};

} // namespace orc_rt

#endif // ORC_RT_BOOTSTRAPINFO_H
