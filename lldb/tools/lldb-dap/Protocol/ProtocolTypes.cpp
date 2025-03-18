//===-- ProtocolTypes.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include <optional>

using namespace llvm;

namespace lldb_dap::protocol {

bool fromJSON(const json::Value &Params, Source::PresentationHint &PH,
              json::Path P) {
  auto rawHint = Params.getAsString();
  if (!rawHint) {
    P.report("expected a string");
    return false;
  }
  std::optional<Source::PresentationHint> hint =
      StringSwitch<std::optional<Source::PresentationHint>>(*rawHint)
          .Case("normal", Source::PresentationHint::normal)
          .Case("emphasize", Source::PresentationHint::emphasize)
          .Case("deemphasize", Source::PresentationHint::deemphasize)
          .Default(std::nullopt);
  if (!hint) {
    P.report("unexpected value");
    return false;
  }
  PH = *hint;
  return true;
}

json::Value toJSON(const Source::PresentationHint &P) {
  switch (P) {
  case Source::PresentationHint::normal:
    return "normal";
  case Source::PresentationHint::emphasize:
    return "emphasize";
  case Source::PresentationHint::deemphasize:
    return "deemphasize";
  }
}

json::Value toJSON(const Source &S) {
  json::Object result;

  if (S.name)
    result.insert({"name", *S.name});
  if (S.path)
    result.insert({"path", *S.path});
  if (S.sourceReference)
    result.insert({"sourceReference", *S.sourceReference});
  if (S.presentationHint)
    result.insert({"presentationHint", *S.presentationHint});

  return std::move(result);
}

bool fromJSON(const json::Value &Params, Source &S, json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("name", S.name) && O.mapOptional("path", S.path) &&
         O.mapOptional("presentationHint", S.presentationHint) &&
         O.mapOptional("sourceReference", S.sourceReference);
}

} // namespace lldb_dap::protocol
