//===--- AnalyzerBase.h - LLVM Advisor -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of AnalyzerBase in Analysis
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Core/AdvisorTypes.h"

namespace llvm::advisor {

struct CapabilityContext {
  UnitRecord Unit;
  std::string ToolchainVersion;
  std::string SourcePath;
  std::string ObjectPath;
  std::string IRPath;
  // Pre-generated or discovered remarks file path. Set by CaptureCore when
  // the artifact is synthesized in-process; checked first by findRemarksPath.
  std::string RemarksPath;
  std::string WorkingDirectory;
};

class CapabilityResult {
public:
  virtual ~CapabilityResult() = default;
  virtual json::Value toJSON() const = 0;
  virtual std::string getContentAddress() const { return {}; }
};

class JSONCapabilityResult final : public CapabilityResult {
public:
  explicit JSONCapabilityResult(json::Value Value) : Value(std::move(Value)) {}

  json::Value toJSON() const override { return Value; }

private:
  json::Value Value;
};

class CapabilityRunner {
public:
  virtual ~CapabilityRunner() = default;

  virtual StringRef getCapabilityID() const = 0;
  virtual Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) = 0;
  virtual bool supportsStreaming() const { return false; }
  virtual std::optional<StringRef> getExternalFallback() const {
    return std::nullopt;
  }
  virtual std::optional<SmallVector<std::string, 4>> getPrerequisites() const {
    return std::nullopt;
  }
};

class SimpleAnalyzer : public CapabilityRunner {
public:
  SimpleAnalyzer(StringRef CapabilityID, StringRef Summary)
      : CapabilityID(CapabilityID.str()), Summary(Summary.str()) {}

  StringRef getCapabilityID() const override { return CapabilityID; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;

private:
  std::string CapabilityID;
  std::string Summary;
};

// Returns the first existing remarks file for the given compilation unit,
// checking source-adjacent paths (.opt.yaml/.opt.json) and the working
// directory.
std::string findRemarksPath(const CapabilityContext &Context);

} // namespace llvm::advisor
