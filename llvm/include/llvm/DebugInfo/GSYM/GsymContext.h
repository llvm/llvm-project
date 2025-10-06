//===-- GsymContext.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#ifndef LLVM_DEBUGINFO_GSYM_GSYMCONTEXT_H
#define LLVM_DEBUGINFO_GSYM_GSYMCONTEXT_H

#include "llvm/DebugInfo/DIContext.h"
#include <cstdint>
#include <memory>
#include <string>

namespace llvm {

namespace gsym {

class GsymReader;

/// GSYM DI Context
/// This data structure is the top level entity that deals with GSYM
/// symbolication.
/// This data structure exists only when there is a need for a transparent
/// interface to different symbolication formats (e.g. GSYM, PDB and DWARF).
/// More control and power over the debug information access can be had by using
/// the GSYM interfaces directly.
class GsymContext : public DIContext {
public:
  GsymContext(std::unique_ptr<GsymReader> Reader);
  ~GsymContext();

  GsymContext(GsymContext &) = delete;
  GsymContext &operator=(GsymContext &) = delete;

  static bool classof(const DIContext *DICtx) {
    return DICtx->getKind() == CK_GSYM;
  }

  void dump(raw_ostream &OS, DIDumpOptions DIDumpOpts) override;

  std::optional<DILineInfo> getLineInfoForAddress(
      object::SectionedAddress Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) override;
  std::optional<DILineInfo>
  getLineInfoForDataAddress(object::SectionedAddress Address) override;
  DILineInfoTable getLineInfoForAddressRange(
      object::SectionedAddress Address, uint64_t Size,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) override;
  DIInliningInfo getInliningInfoForAddress(
      object::SectionedAddress Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) override;

  std::vector<DILocal>
  getLocalsForAddress(object::SectionedAddress Address) override;

private:
  const std::unique_ptr<GsymReader> Reader;
};

} // end namespace gsym

} // end namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMCONTEXT_H
