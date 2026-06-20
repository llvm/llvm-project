//===--- RuntimeRunner.h - LLVM Advisor ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CapabilityRunner wrappers for runtime ingestors.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

class PGOInstrRunner final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "runtime.pgo.instr"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

class PGOSampleRunner final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "runtime.pgo.sample"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

class MemProfRunner final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "runtime.pgo.memprof"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

class CoverageRunner final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "runtime.coverage"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

class XRayRunner final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "runtime.xray"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

class SancovRunner final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "runtime.sancov"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};

class SanitizerRunner final : public CapabilityRunner {
public:
  explicit SanitizerRunner(StringRef CapabilityID, StringRef Pattern)
      : CapabilityID(CapabilityID.str()), Pattern(Pattern.str()) {}
  StringRef getCapabilityID() const override { return CapabilityID; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;

private:
  std::string CapabilityID;
  std::string Pattern;
};

} // namespace llvm::advisor
