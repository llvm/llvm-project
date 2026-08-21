//===------------------- StaticHandler.cpp - LLVM Advisor ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Serves the embedded web UI as static HTML.
// The UI is a single-page application bundled from Assets/*.
// Regenerate via: bash Assets/bundle.sh > Assets/bundled.html
//   then: python3 -c "..." (see Assets/ for details)
//
//===----------------------------------------------------------------------===//

#include "Client/HTTP/Handlers/StaticHandler.h"

// Include the generated HTML content.
#include "Client/HTTP/Assets/index_html.inc"

using namespace llvm;
using namespace llvm::advisor;

StringRef StaticHandler::index() const {
  return StringRef(IndexHTML, sizeof(IndexHTML) - 1);
}
