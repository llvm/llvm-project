//===- llvm/unittest/Telemetry/TelemetryTest.cpp - Telemetry unittests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Telemetry/Telemetry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <chrono>
#include <ctime>
#include <vector>

// Testing parameters.
// These are set by each test to force certain outcomes.
// Since the tests may be run in parellel, these should probably
// be thread_local.
static thread_local bool HasExitError = false;
static thread_local std::string ExitMsg = "";
static thread_local bool HasVendorConfig = false;
static thread_local bool SanitizeData = false;
static thread_local std::string Buffer = "";
static thread_local std::vector<llvm::json::Object> EmittedJsons;
static thread_local std::string ExpectedUuid = "";

namespace llvm {
namespace telemetry {
namespace vendor_code {

// Generate unique (but deterministic "uuid" for testing purposes).
static std::string nextUuid() {
  static std::atomic<int> seed = 1111;
  return std::to_string(seed.fetch_add(1, std::memory_order_acquire));
}

struct VendorEntryKind {
  // TODO: should avoid dup with other vendors' Types?
  static const KindType VendorCommon = 0b010101000;
  static const KindType Startup = 0b010101001;
  static const KindType Exit = 0b010101010;
};

// Demonstrates that the TelemetryInfo (data courier) struct can be extended
// by downstream code to store additional data as needed.
// It can also define additional data serialization method.
struct VendorCommonTelemetryInfo : public TelemetryInfo {
  static bool classof(const TelemetryInfo *T) {
    // Subclasses of this is also acceptable.
    return (T->getEntryKind() & VendorEntryKind::VendorCommon) ==
           VendorEntryKind::VendorCommon;
  }

  KindType getEntryKind() const override {
    return VendorEntryKind::VendorCommon;
  }

  virtual void serializeToStream(llvm::raw_ostream &OS) const = 0;
};

struct StartupEvent : public VendorCommonTelemetryInfo {
  std::string MagicStartupMsg;

  StartupEvent() = default;
  StartupEvent(const StartupEvent &E) {
    SessionUuid = E.SessionUuid;
    Stats = E.Stats;
    ExitDesc = E.ExitDesc;
    Counter = E.Counter;

    MagicStartupMsg = E.MagicStartupMsg;
  }

  static bool classof(const TelemetryInfo *T) {
    return T->getEntryKind() == VendorEntryKind::Startup;
  }

  KindType getEntryKind() const override { return VendorEntryKind::Startup; }

  void serializeToStream(llvm::raw_ostream &OS) const override {
    OS << "UUID:" << SessionUuid << "\n";
    OS << "MagicStartupMsg:" << MagicStartupMsg << "\n";
  }

  json::Object serializeToJson() const override {
    return json::Object{
        {"Startup",
         {{"UUID", SessionUuid}, {"MagicStartupMsg", MagicStartupMsg}}},
    };
  }
};

struct ExitEvent : public VendorCommonTelemetryInfo {
  std::string MagicExitMsg;

  ExitEvent() = default;
  // Provide a copy ctor because we may need to make a copy
  // before sanitizing the Entry.
  ExitEvent(const ExitEvent &E) {
    SessionUuid = E.SessionUuid;
    Stats = E.Stats;
    ExitDesc = E.ExitDesc;
    Counter = E.Counter;

    MagicExitMsg = E.MagicExitMsg;
  }

  static bool classof(const TelemetryInfo *T) {
    return T->getEntryKind() == VendorEntryKind::Exit;
  }

  unsigned getEntryKind() const override { return VendorEntryKind::Exit; }

  void serializeToStream(llvm::raw_ostream &OS) const override {
    OS << "UUID:" << SessionUuid << "\n";
    if (ExitDesc.has_value())
      OS << "ExitCode:" << ExitDesc->ExitCode << "\n";
    OS << "MagicExitMsg:" << MagicExitMsg << "\n";
  }

  json::Object serializeToJson() const override {
    json::Array I = json::Array{
        {"UUID", SessionUuid},
        {"MagicExitMsg", MagicExitMsg},
    };
    if (ExitDesc.has_value())
      I.push_back(json::Value({"ExitCode", ExitDesc->ExitCode}));
    return json::Object{
        {"Exit", std::move(I)},
    };
  }
};

struct CustomTelemetryEvent : public VendorCommonTelemetryInfo {
  std::vector<std::string> Msgs;

  CustomTelemetryEvent() = default;
  CustomTelemetryEvent(const CustomTelemetryEvent &E) {
    SessionUuid = E.SessionUuid;
    Stats = E.Stats;
    ExitDesc = E.ExitDesc;
    Counter = E.Counter;

    Msgs = E.Msgs;
  }

  void serializeToStream(llvm::raw_ostream &OS) const override {
    OS << "UUID:" << SessionUuid << "\n";
    int I = 0;
    for (const std::string &M : Msgs) {
      OS << "MSG_" << I << ":" << M << "\n";
      ++I;
    }
  }

  json::Object serializeToJson() const override {
    json::Object Inner;
    Inner.try_emplace("UUID", SessionUuid);
    int I = 0;
    for (const std::string &M : Msgs) {
      Inner.try_emplace(("MSG_" + llvm::Twine(I)).str(), M);
      ++I;
    }

    return json::Object{{"Midpoint", std::move(Inner)}};
  }
};

// The following classes demonstrate how downstream code can
// define one or more custom TelemetryDestination(s) to handle
// Telemetry data differently, specifically:
//    + which data to send (fullset or sanitized)
//    + where to send the data
//    + in what form

const std::string STRING_DEST("STRING");
const std::string JSON_DEST("JSON");

// This Destination sends data to a std::string given at ctor.
class StringDestination : public TelemetryDestination {
public:
  // ShouldSanitize: if true, sanitize the data before emitting, otherwise, emit
  // the full set.
  StringDestination(bool ShouldSanitize, std::string &Buf)
      : ShouldSanitize(ShouldSanitize), OS(Buf) {}

  Error emitEntry(const TelemetryInfo *Entry) override {
    if (isa<VendorCommonTelemetryInfo>(Entry)) {
      if (auto *E = dyn_cast<VendorCommonTelemetryInfo>(Entry)) {
        if (ShouldSanitize) {
          if (isa<StartupEvent>(E) || isa<ExitEvent>(E)) {
            // There is nothing to sanitize for this type of data, so keep
            // as-is.
            E->serializeToStream(OS);
          } else if (isa<CustomTelemetryEvent>(E)) {
            auto Sanitized = sanitizeFields(dyn_cast<CustomTelemetryEvent>(E));
            Sanitized.serializeToStream(OS);
          } else {
            llvm_unreachable("unexpected type");
          }
        } else {
          E->serializeToStream(OS);
        }
      }
    } else {
      // Unfamiliar entries, just send the entry's UUID
      OS << "UUID:" << Entry->SessionUuid << "\n";
    }
    return Error::success();
  }

  std::string name() const override { return STRING_DEST; }

private:
  // Returns a copy of the given entry, but with some fields sanitized.
  CustomTelemetryEvent sanitizeFields(const CustomTelemetryEvent *Entry) {
    CustomTelemetryEvent Sanitized(*Entry);
    // Pretend that messages stored at ODD positions are "sensitive",
    // hence need to be sanitized away.
    int S = Sanitized.Msgs.size() - 1;
    for (int I = S % 2 == 0 ? S - 1 : S; I >= 0; I -= 2)
      Sanitized.Msgs[I] = "";
    return Sanitized;
  }

  bool ShouldSanitize;
  llvm::raw_string_ostream OS;
};

// This Destination sends data to some "blackbox" in form of JSON.
class JsonStreamDestination : public TelemetryDestination {
public:
  JsonStreamDestination(bool ShouldSanitize) : ShouldSanitize(ShouldSanitize) {}

  Error emitEntry(const TelemetryInfo *Entry) override {
    if (auto *E = dyn_cast<VendorCommonTelemetryInfo>(Entry)) {
      if (ShouldSanitize) {
        if (isa<StartupEvent>(E) || isa<ExitEvent>(E)) {
          // There is nothing to sanitize for this type of data, so keep as-is.
          return SendToBlackbox(E->serializeToJson());
        } else if (isa<CustomTelemetryEvent>(E)) {
          auto Sanitized = sanitizeFields(dyn_cast<CustomTelemetryEvent>(E));
          return SendToBlackbox(Sanitized.serializeToJson());
        } else {
          llvm_unreachable("unexpected type");
        }
      } else {
        return SendToBlackbox(E->serializeToJson());
      }
    } else {
      // Unfamiliar entries, just send the entry's UUID
      return SendToBlackbox(json::Object{{"UUID", Entry->SessionUuid}});
    }
    return make_error<StringError>("unhandled codepath in emitEntry",
                                   inconvertibleErrorCode());
  }

  std::string name() const override { return JSON_DEST; }

private:
  // Returns a copy of the given entry, but with some fields sanitized.
  CustomTelemetryEvent sanitizeFields(const CustomTelemetryEvent *Entry) {
    CustomTelemetryEvent Sanitized(*Entry);
    // Pretend that messages stored at EVEN positions are "sensitive",
    // hence need to be sanitized away.
    int S = Sanitized.Msgs.size() - 1;
    for (int I = S % 2 == 0 ? S : S - 1; I >= 0; I -= 2)
      Sanitized.Msgs[I] = "";

    return Sanitized;
  }

  llvm::Error SendToBlackbox(json::Object O) {
    // Here is where the vendor-defined Destination class can
    // send the data to some internal storage.
    // For testing purposes, we just queue up the entries to
    // the vector for validation.
    EmittedJsons.push_back(std::move(O));
    return Error::success();
  }
  bool ShouldSanitize;
};

// Custom vendor-defined Telemeter that has additional data-collection point.
class TestTelemeter : public Telemeter {
public:
  TestTelemeter(std::string SessionUuid) : Uuid(SessionUuid), Counter(0) {}

  static std::unique_ptr<TestTelemeter>
  createInstance(TelemetryConfig *config) {
    llvm::errs() << "============================== createInstance is called"
                 << "\n";
    if (!config->EnableTelemetry)
      return nullptr;
    ExpectedUuid = nextUuid();
    std::unique_ptr<TestTelemeter> Telemeter =
        std::make_unique<TestTelemeter>(ExpectedUuid);
    // Set up Destination based on the given config.
    for (const std::string &Dest : config->AdditionalDestinations) {
      // The destination(s) are ALSO defined by vendor, so it should understand
      // what the name of each destination signifies.
      if (Dest == JSON_DEST) {
        Telemeter->addDestination(
            new vendor_code::JsonStreamDestination(SanitizeData));
      } else if (Dest == STRING_DEST) {
        Telemeter->addDestination(
            new vendor_code::StringDestination(SanitizeData, Buffer));
      } else {
        llvm_unreachable(
            llvm::Twine("unknown destination: ", Dest).str().c_str());
      }
    }
    return Telemeter;
  }

  void logStartup(llvm::StringRef ToolPath, TelemetryInfo *Entry) override {
    ToolName = ToolPath.str();

    // The vendor can add additional stuff to the entry before logging.
    if (auto *S = dyn_cast<StartupEvent>(Entry)) {
      S->MagicStartupMsg = llvm::Twine("One_", ToolPath).str();
    }
    emitToDestinations(Entry);
  }

  void logExit(llvm::StringRef ToolPath, TelemetryInfo *Entry) override {
    // Ensure we're shutting down the same tool we started with.
    if (ToolPath != ToolName) {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << "Expected tool with name" << ToolName << ", but got " << ToolPath;
      llvm_unreachable(Str.c_str());
    }

    // The vendor can add additional stuff to the entry before logging.
    if (auto *E = dyn_cast<ExitEvent>(Entry)) {
      E->MagicExitMsg = llvm::Twine("Three_", ToolPath).str();
    }

    emitToDestinations(Entry);
  }

  void addDestination(TelemetryDestination *Dest) override {
    Destinations.push_back(Dest);
  }

  void logMidpoint(TelemetryInfo *Entry) {
    // The custom Telemeter can record and send additional data.
    if (auto *C = dyn_cast<CustomTelemetryEvent>(Entry)) {
      C->Msgs.push_back("Two");
      C->Msgs.push_back("Deux");
      C->Msgs.push_back("Zwei");
    }

    emitToDestinations(Entry);
  }

  const std::string &getUuid() const { return Uuid; }

  ~TestTelemeter() {
    for (auto *Dest : Destinations)
      delete Dest;
  }

  template <typename T> T makeDefaultTelemetryInfo() {
    T Ret;
    Ret.SessionUuid = Uuid;
    Ret.Counter = Counter++;
    return Ret;
  }

private:
  void emitToDestinations(TelemetryInfo *Entry) {
    for (TelemetryDestination *Dest : Destinations) {
      llvm::Error err = Dest->emitEntry(Entry);
      if (err) {
        // Log it and move on.
      }
    }
  }

  const std::string Uuid;
  size_t Counter;
  std::string ToolName;
  std::vector<TelemetryDestination *> Destinations;
};

// Pretend to be a "weakly" defined vendor-specific function.
void ApplyVendorSpecificConfigs(TelemetryConfig *config) {
  config->EnableTelemetry = true;
}

} // namespace vendor_code
} // namespace telemetry
} // namespace llvm

namespace {

void ApplyCommonConfig(llvm::telemetry::TelemetryConfig *config) {
  // Any shareable configs for the upstream tool can go here.
  // .....
}

std::shared_ptr<llvm::telemetry::TelemetryConfig> GetTelemetryConfig() {
  // Telemetry is disabled by default.
  // The vendor can enable in their config.
  auto Config = std::make_shared<llvm::telemetry::TelemetryConfig>();
  Config->EnableTelemetry = false;

  ApplyCommonConfig(Config.get());

  // Apply vendor specific config, if present.
  // In principle, this would be a build-time param, configured by the vendor.
  // Eg:
  //
  // #ifdef HAS_VENDOR_TELEMETRY_CONFIG
  //     llvm::telemetry::vendor_code::ApplyVendorSpecificConfigs(config.get());
  // #endif
  //
  // But for unit testing, we use the testing params defined at the top.
  if (HasVendorConfig) {
    llvm::telemetry::vendor_code::ApplyVendorSpecificConfigs(Config.get());
  }
  return Config;
}

using namespace llvm;
using namespace llvm::telemetry;

// For deterministic tests, pre-defined certain important time-points
// rather than using now().
//
// Preset StartTime to EPOCH.
auto StartTime = std::chrono::time_point<std::chrono::steady_clock>{};
// Pretend the time it takes for the tool's initialization is EPOCH + 5
// milliseconds
auto InitCompleteTime = StartTime + std::chrono::milliseconds(5);
auto MidPointTime = StartTime + std::chrono::milliseconds(10);
auto MidPointCompleteTime = MidPointTime + std::chrono::milliseconds(5);
// Preset ExitTime to EPOCH + 20 milliseconds
auto ExitTime = StartTime + std::chrono::milliseconds(20);
// Pretend the time it takes to complete tearing down the tool is 10
// milliseconds.
auto ExitCompleteTime = ExitTime + std::chrono::milliseconds(10);

void AtToolStart(std::string ToolName, vendor_code::TestTelemeter *T) {
  vendor_code::StartupEvent Entry =
      T->makeDefaultTelemetryInfo<vendor_code::StartupEvent>();
  Entry.Stats = {StartTime, InitCompleteTime};
  T->logStartup(ToolName, &Entry);
}

void AtToolExit(std::string ToolName, vendor_code::TestTelemeter *T) {
  vendor_code::ExitEvent Entry =
      T->makeDefaultTelemetryInfo<vendor_code::ExitEvent>();
  Entry.Stats = {ExitTime, ExitCompleteTime};

  if (HasExitError) {
    Entry.ExitDesc = {1, ExitMsg};
  }
  T->logExit(ToolName, &Entry);
}

void AtToolMidPoint(vendor_code::TestTelemeter *T) {
  vendor_code::CustomTelemetryEvent Entry =
      T->makeDefaultTelemetryInfo<vendor_code::CustomTelemetryEvent>();
  Entry.Stats = {MidPointTime, MidPointCompleteTime};
  T->logMidpoint(&Entry);
}

// Helper function to print the given object content to string.
static std::string ValueToString(const json::Value *V) {
  std::string Ret;
  llvm::raw_string_ostream P(Ret);
  P << *V;
  return Ret;
}

// Without vendor's implementation, telemetry is not enabled by default.
TEST(TelemetryTest, TelemetryDefault) {
  HasVendorConfig = false;
  std::shared_ptr<llvm::telemetry::TelemetryConfig> Config =
      GetTelemetryConfig();
  auto Tool = vendor_code::TestTelemeter::createInstance(Config.get());

  EXPECT_EQ(nullptr, Tool.get());
}

TEST(TelemetryTest, TelemetryEnabled) {
  const std::string ToolName = "TelemetryTest";

  // Preset some test params.
  HasVendorConfig = true;
  SanitizeData = false;
  Buffer.clear();
  EmittedJsons.clear();

  std::shared_ptr<llvm::telemetry::TelemetryConfig> Config =
      GetTelemetryConfig();

  // Add some destinations
  Config->AdditionalDestinations.push_back(vendor_code::STRING_DEST);
  Config->AdditionalDestinations.push_back(vendor_code::JSON_DEST);

  auto Tool = vendor_code::TestTelemeter::createInstance(Config.get());

  AtToolStart(ToolName, Tool.get());
  AtToolMidPoint(Tool.get());
  AtToolExit(ToolName, Tool.get());

  // Check that the Tool uses the expected UUID.
  EXPECT_STREQ(Tool->getUuid().c_str(), ExpectedUuid.c_str());

  // Check that the StringDestination emitted properly
  {
    std::string ExpectedBuffer =
        ("UUID:" + llvm::Twine(ExpectedUuid) + "\n" + "MagicStartupMsg:One_" +
         llvm::Twine(ToolName) + "\n" + "UUID:" + llvm::Twine(ExpectedUuid) +
         "\n" + "MSG_0:Two\n" + "MSG_1:Deux\n" + "MSG_2:Zwei\n" +
         "UUID:" + llvm::Twine(ExpectedUuid) + "\n" + "MagicExitMsg:Three_" +
         llvm::Twine(ToolName) + "\n")
            .str();

    EXPECT_STREQ(ExpectedBuffer.c_str(), Buffer.c_str());
  }

  // Check that the JsonDestination emitted properly
  {

    // There should be 3 events emitted by the Telemeter (start, midpoint, exit)
    EXPECT_EQ(3, EmittedJsons.size());

    const json::Value *StartupEntry = EmittedJsons[0].get("Startup");
    ASSERT_NE(StartupEntry, nullptr);
    EXPECT_STREQ(("[[\"UUID\",\"" + llvm::Twine(ExpectedUuid) +
                  "\"],[\"MagicStartupMsg\",\"One_" + llvm::Twine(ToolName) +
                  "\"]]")
                     .str()
                     .c_str(),
                 ValueToString(StartupEntry).c_str());

    const json::Value *MidpointEntry = EmittedJsons[1].get("Midpoint");
    ASSERT_NE(MidpointEntry, nullptr);
    // TODO: This is a bit flaky in that the json string printer sort the
    // entries (for now), so the "UUID" field is put at the end of the array
    // even though it was emitted first.
    EXPECT_STREQ(("{\"MSG_0\":\"Two\",\"MSG_1\":\"Deux\",\"MSG_2\":\"Zwei\","
                  "\"UUID\":\"" +
                  llvm::Twine(ExpectedUuid) + "\"}")
                     .str()
                     .c_str(),
                 ValueToString(MidpointEntry).c_str());

    const json::Value *ExitEntry = EmittedJsons[2].get("Exit");
    ASSERT_NE(ExitEntry, nullptr);
    EXPECT_STREQ(("[[\"UUID\",\"" + llvm::Twine(ExpectedUuid) +
                  "\"],[\"MagicExitMsg\",\"Three_" + llvm::Twine(ToolName) +
                  "\"]]")
                     .str()
                     .c_str(),
                 ValueToString(ExitEntry).c_str());
  }
}

// Similar to previous tests, but toggling the data-sanitization option ON.
// The recorded data should have some fields removed.
TEST(TelemetryTest, TelemetryEnabledSanitizeData) {
  const std::string ToolName = "TelemetryTest_SanitizedData";

  // Preset some test params.
  HasVendorConfig = true;
  SanitizeData = true;
  Buffer.clear();
  EmittedJsons.clear();

  std::shared_ptr<llvm::telemetry::TelemetryConfig> Config =
      GetTelemetryConfig();

  // Add some destinations
  Config->AdditionalDestinations.push_back(vendor_code::STRING_DEST);
  Config->AdditionalDestinations.push_back(vendor_code::JSON_DEST);

  auto Tool = vendor_code::TestTelemeter::createInstance(Config.get());

  AtToolStart(ToolName, Tool.get());
  AtToolMidPoint(Tool.get());
  AtToolExit(ToolName, Tool.get());

  // Check that the StringDestination emitted properly
  {
    // The StringDestination should have removed the odd-positioned msgs.

    std::string ExpectedBuffer =
        ("UUID:" + llvm::Twine(ExpectedUuid) + "\n" + "MagicStartupMsg:One_" +
         llvm::Twine(ToolName) + "\n" + "UUID:" + llvm::Twine(ExpectedUuid) +
         "\n" + "MSG_0:Two\n" + "MSG_1:\n" + // <<< was sanitized away.
         "MSG_2:Zwei\n" + "UUID:" + llvm::Twine(ExpectedUuid) + "\n" +
         "MagicExitMsg:Three_" + llvm::Twine(ToolName) + "\n")
            .str();
    EXPECT_STREQ(ExpectedBuffer.c_str(), Buffer.c_str());
  }

  // Check that the JsonDestination emitted properly
  {

    // There should be 3 events emitted by the Telemeter (start, midpoint, exit)
    EXPECT_EQ(3, EmittedJsons.size());

    const json::Value *StartupEntry = EmittedJsons[0].get("Startup");
    ASSERT_NE(StartupEntry, nullptr);
    EXPECT_STREQ(("[[\"UUID\",\"" + llvm::Twine(ExpectedUuid) +
                  "\"],[\"MagicStartupMsg\",\"One_" + llvm::Twine(ToolName) +
                  "\"]]")
                     .str()
                     .c_str(),
                 ValueToString(StartupEntry).c_str());

    const json::Value *MidpointEntry = EmittedJsons[1].get("Midpoint");
    ASSERT_NE(MidpointEntry, nullptr);
    // The JsonDestination should have removed the even-positioned msgs.
    EXPECT_STREQ(
        ("{\"MSG_0\":\"\",\"MSG_1\":\"Deux\",\"MSG_2\":\"\",\"UUID\":\"" +
         llvm::Twine(ExpectedUuid) + "\"}")
            .str()
            .c_str(),
        ValueToString(MidpointEntry).c_str());

    const json::Value *ExitEntry = EmittedJsons[2].get("Exit");
    ASSERT_NE(ExitEntry, nullptr);
    EXPECT_STREQ(("[[\"UUID\",\"" + llvm::Twine(ExpectedUuid) +
                  "\"],[\"MagicExitMsg\",\"Three_" + llvm::Twine(ToolName) +
                  "\"]]")
                     .str()
                     .c_str(),
                 ValueToString(ExitEntry).c_str());
  }
}

} // namespace
