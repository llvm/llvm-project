//===-- RegisterType.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/RegisterType.h"

using namespace lldb_private;

void RegisterType::ToXML(
    Stream &strm, std::unordered_set<const RegisterType *> &previously_emitted,
    const RegisterType *user) const {
  for (auto dep : m_dependencies)
    if (previously_emitted.find(dep) == previously_emitted.end()) {
      dep->ToXML(strm, previously_emitted, this);
      previously_emitted.insert(dep);
    }

  ToXMLElement(strm, user);
}