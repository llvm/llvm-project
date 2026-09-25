//===- ConnectionSpec.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Connection spec parsing.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/ConnectionSpec.h"

#include "orc-rt-internal/support/StringExtras.h"

namespace orc_rt {

Expected<ConnectionSpec> ConnectionSpec::parse(std::string_view Spec) noexcept {
  auto Fail = [&](std::string_view Msg) -> Error {
    StringOutputStream OS;
    OS << "in connection spec '" << Spec << "': " << Msg;
    return make_error<StringError>(std::move(OS).str());
  };

  auto EqPos = Spec.find('=');
  if (EqPos == std::string_view::npos)
    return Fail("expected '<transport>[:<action>]=<descriptor>', but found "
                "no '='");

  auto Descriptor = Spec.substr(EqPos + 1);
  auto LHS = Spec.substr(0, EqPos);
  std::string_view Transport, Action;

  if (auto ColonPos = LHS.find(':'); ColonPos != std::string_view::npos) {
    Transport = LHS.substr(0, ColonPos);
    Action = LHS.substr(ColonPos + 1);
    if (Action.empty())
      return Fail("empty action name");
  } else
    Transport = LHS;

  if (Transport.empty())
    return Fail("empty transport name");

  return ConnectionSpec(std::string(Transport), std::string(Action),
                        std::string(Descriptor));
}

} // namespace orc_rt
