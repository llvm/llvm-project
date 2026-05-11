//===------------------- HTTPServer.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HTTP server for REST API and embedded web UI.
// Accepts HTTP connections and routes to handlers.
//
//===----------------------------------------------------------------------===//

#include "Client/HTTP/HTTPServer.h"
#include "Client/HTTP/Handlers/StaticHandler.h"
#include "Utils/JSON.h"

#include <cerrno>
#include <cstring>
#include <cmath>
#include <string>
#include <thread>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace llvm;
using namespace llvm::advisor;

static std::string renderJSON(const json::Value &Value) {
  std::string Body;
  raw_string_ostream OS(Body);
  json::OStream JOS(OS);
  JOS.value(Value);
  OS.flush();
  return Body;
}

#ifndef _WIN32
static std::string makeRawHTTPResponse(unsigned Code, const char *ContentType,
                                       StringRef Body, bool KeepAlive = false) {
  std::string Out;
  raw_string_ostream OS(Out);
  const char *Reason =
      Code == 200
          ? "OK"
          : (Code == 201 ? "Created" : (Code == 404 ? "Not Found" : "Error"));
  OS << "HTTP/1.1 " << Code << ' ' << Reason << "\r\n";
  OS << "Content-Type: " << ContentType << "\r\n";
  OS << "Content-Length: " << Body.size() << "\r\n";
  OS << "Access-Control-Allow-Origin: *\r\n";
  OS << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
  OS << "Access-Control-Allow-Headers: Content-Type\r\n";
  OS << "Connection: " << (KeepAlive ? "keep-alive" : "close") << "\r\n\r\n";
  OS << Body;
  OS.flush();
  return Out;
}

static bool sendAll(int FD, StringRef Data) {
  size_t Off = 0;
  while (Off < Data.size()) {
    ssize_t N = ::send(FD, Data.data() + Off, Data.size() - Off, 0);
    if (N <= 0)
      return false;
    Off += static_cast<size_t>(N);
  }
  return true;
}

struct ParsedRequest {
  std::string Method;
  std::string Path;
  std::string Query;
  std::string Body;
  std::string AuthHeader;
  bool KeepAlive = false;
  bool Valid = false;
};

static bool readFullRequest(int FD, ParsedRequest &Out) {
  std::string Raw;
  Raw.reserve(4096);
  char Chunk[4096];
  size_t HeaderEnd = std::string::npos;
  while (HeaderEnd == std::string::npos) {
    ssize_t N = ::recv(FD, Chunk, sizeof(Chunk), 0);
    if (N <= 0)
      return false;
    Raw.append(Chunk, static_cast<size_t>(N));
    HeaderEnd = Raw.find("\r\n\r\n");
    if (Raw.size() > 1024 * 1024)
      return false;
  }
  StringRef Req(Raw);
  size_t EOL = Req.find("\r\n");
  if (EOL == StringRef::npos)
    return false;
  StringRef RequestLine = Req.substr(0, EOL);
  SmallVector<StringRef, 3> Parts;
  RequestLine.split(Parts, ' ');
  if (Parts.size() < 2)
    return false;

  Out.Method = Parts[0].str();
  StringRef FullPath = Parts[1];
  size_t QPos = FullPath.find('?');
  Out.Path =
      (QPos == StringRef::npos ? FullPath : FullPath.substr(0, QPos)).str();
  Out.Query =
      (QPos == StringRef::npos ? StringRef() : FullPath.substr(QPos + 1)).str();
  Out.Valid = true;

  StringRef Headers = Req.substr(EOL + 2, HeaderEnd - EOL - 2);
  size_t ContentLength = 0;
  SmallVector<StringRef, 32> HeaderLines;
  Headers.split(HeaderLines, "\r\n");
  for (StringRef Line : HeaderLines) {
    if (Line.starts_with_insensitive("content-length:")) {
      StringRef Val = Line.drop_front(15).trim();
      Val.getAsInteger(10, ContentLength);
    } else if (Line.starts_with_insensitive("authorization:")) {
      StringRef Val = Line.drop_front(14).trim();
      if (Val.starts_with_insensitive("bearer "))
        Out.AuthHeader = Val.drop_front(7).trim().str();
      else
        Out.AuthHeader = Val.str();
    } else if (Line.starts_with_insensitive("connection:")) {
      StringRef Val = Line.drop_front(11).trim();
      Out.KeepAlive = Val.equals_insensitive("keep-alive");
    }
  }

  if (ContentLength == 0)
    return true;
  size_t BodyOffset = HeaderEnd + 4;
  size_t AlreadyRead = Raw.size() > BodyOffset ? Raw.size() - BodyOffset : 0;
  Out.Body = Raw.substr(BodyOffset);
  while (AlreadyRead < ContentLength) {
    size_t Remaining = ContentLength - AlreadyRead;
    size_t ToRead = std::min(Remaining, sizeof(Chunk));
    ssize_t N = ::recv(FD, Chunk, ToRead, 0);
    if (N <= 0)
      break;
    Out.Body.append(Chunk, static_cast<size_t>(N));
    AlreadyRead += static_cast<size_t>(N);
  }
  return true;
}

#endif

static std::string urlDecode(StringRef S) {
  std::string Out;
  Out.reserve(S.size());
  for (size_t I = 0, N = S.size(); I < N; ++I) {
    if (S[I] == '%' && I + 2 < N) {
      unsigned Hi = 0, Lo = 0;
      if (!StringRef(S.data() + I + 1, 1).getAsInteger(16, Hi) &&
          !StringRef(S.data() + I + 2, 1).getAsInteger(16, Lo)) {
        Out += static_cast<char>((Hi << 4) | Lo);
        I += 2;
        continue;
      }
    }
    Out += S[I];
  }
  return Out;
}

// --- Shared API Handlers ---

struct HTTPResult {
  unsigned Code = 200;
  const char *ContentType = "application/json";
  std::string Body;
};

static HTTPResult makeJSONError(unsigned Code, Error Err) {
  return {
      Code, "application/json",
      renderJSON(errorEnvelope("request_failed", toString(std::move(Err))))};
}

static HTTPResult makeJSONErrorStr(unsigned Code, StringRef Msg) {
  return {Code, "application/json",
          renderJSON(errorEnvelope("request_failed", Msg))};
}

static HTTPResult makeJSONSuccess(unsigned Code, json::Value Result) {
  return {Code, "application/json",
          renderJSON(successEnvelope(std::move(Result)))};
}

static SmallVector<std::string, 16> parseCapabilityList(StringRef Text) {
  SmallVector<std::string, 16> Caps;
  SmallVector<StringRef, 16> Parts;
  Text.split(Parts, ',', -1, false);
  for (StringRef Part : Parts) {
    Part = Part.trim();
    if (!Part.empty())
      Caps.push_back(Part.str());
  }
  return Caps;
}

static StringRef inspectModeToCapability(StringRef Mode, StringRef Explicit) {
  if (!Explicit.empty())
    return Explicit;
  static const std::pair<StringRef, StringRef> Mapping[] = {
      {"ir", "llvm.ir.view"},
      {"ir-diff", "llvm.ir.diff"},
      {"cfg", "llvm.cfg"},
      {"dom", "llvm.dom_tree"},
      {"callgraph", "llvm.call_graph"},
      {"dag", "llvm.selection_dag"},
      {"mir", "llvm.machine_ir"},
      {"asm", "llvm.asm.view"},
      {"remarks", "llvm.remarks.detail"},
      {"debug", "llvm.debug.detail"},
      {"passes", "llvm.ir.passes.list"},
      {"loop", "llvm.loop_info"},
      {"mca", "llvm.mca.report"},
      {"exegesis", "llvm.exegesis"},
      {"offload", "offload.binary.inspect"},
  };
  for (const auto &KV : Mapping)
    if (KV.first == Mode)
      return KV.second;
  return StringRef();
}

struct FamilyRule {
  StringRef Prefix;
  StringRef Family;
  bool Exact;
};
static StringRef capabilityFamily(StringRef ID) {
  static const FamilyRule Rules[] = {
      {"build.", "Build", false},
      {"clang.", "Clang", false},
      {"llvm.inlining.tree", "IR", true},
      {"llvm.ir.", "IR", false},
      {"llvm.remarks.", "IR", false},
      {"llvm.pass.", "IR", false},
      {"llvm.obj.", "Binary", false},
      {"llvm.debug.", "Binary", false},
      {"llvm.cgdata", "Binary", true},
      {"lld.mapfile", "Binary", false},
      {"llvm.cfg", "Inspection", false},
      {"llvm.dom_tree", "Inspection", false},
      {"llvm.call_graph", "Inspection", false},
      {"llvm.loop_info", "Inspection", false},
      {"llvm.selection_dag", "Inspection", false},
      {"llvm.machine_ir", "Inspection", false},
      {"llvm.asm.", "Inspection", false},
      {"llvm.mca.", "Inspection", false},
      {"llvm.exegesis", "Inspection", true},
      {"llvm.lto.", "LTO", false},
      {"offload.", "Offload", false},
      {"runtime.", "Runtime", false},
  };
  for (const auto &Rule : Rules)
    if (Rule.Exact ? ID == Rule.Prefix : ID.starts_with(Rule.Prefix))
      return Rule.Family;
  return "Other";
}

static bool shouldQuerySummaryCapability(StringRef ID) {
  return ID != "llvm.ir.diff" && ID != "llvm.remarks.size_diff";
}

static bool shouldIgnoreSummaryMetric(StringRef Key) {
  return Key == "available" || Key == "unit_id" || Key == "snapshot_id" ||
         Key == "capability" || Key == "reason" || Key == "summary" ||
         Key == "source_path" || Key == "directory" || Key == "module" ||
         Key == "format" || Key == "arch" || Key == "tool" || Key == "input" ||
         Key == "version";
}

static void accumulateSummaryMetrics(const json::Object &Object,
                                     StringMap<int64_t> &Totals) {
  for (const auto &KV : Object) {
    StringRef Key = KV.first;
    if (shouldIgnoreSummaryMetric(Key))
      continue;
    if (std::optional<int64_t> Integer = KV.second.getAsInteger())
      Totals[Key] += *Integer;
  }
  if (const json::Object *ByType = Object.getObject("by_type")) {
    for (const auto &KV : *ByType) {
      if (std::optional<int64_t> Integer = KV.second.getAsInteger()) {
        SmallString<64> TypeKey;
        TypeKey.append("type_");
        TypeKey.append(KV.first);
        Totals[TypeKey] += *Integer;
      }
    }
  }
}

static HTTPResult handleGetHealth(CoreClient &Client) {
  return makeJSONSuccess(200, toJSON(Client.health()));
}

static HTTPResult handleGetStatus(CoreClient &Client) {
  return makeJSONSuccess(200, Client.inspectStorage());
}

static HTTPResult handleGetMetrics(CoreClient &Client) {
  HealthStatus Health = Client.health();
  std::string Body;
  raw_string_ostream OS(Body);
  OS << "advisor_snapshots " << Health.Snapshots << '\n';
  OS << "advisor_units " << Health.Units << '\n';
  OS << "advisor_store_ok " << (Health.OK ? 1 : 0) << '\n';
  OS.flush();
  return {200, "text/plain; version=0.0.4", Body};
}

static HTTPResult handleGetCapabilities(CoreClient &Client, StringRef ID = "") {
  if (ID.empty()) {
    json::Array Array;
    for (const CapabilitySpec &Spec : Client.listCapabilities())
      Array.push_back(toJSON(Spec));
    return makeJSONSuccess(200, std::move(Array));
  }
  for (const CapabilitySpec &Spec : Client.listCapabilities()) {
    if (Spec.ID == ID)
      return makeJSONSuccess(200, toJSON(Spec));
  }
  return makeJSONErrorStr(404, "unknown capability");
}

static HTTPResult handleGetSnapshots(CoreClient &Client, StringRef ID = "") {
  if (ID.empty()) {
    json::Array Array;
    for (const SnapshotRecord &Snapshot : Client.listSnapshots())
      Array.push_back(toJSON(Snapshot));
    return makeJSONSuccess(200, std::move(Array));
  }
  Expected<SnapshotRecord> Snapshot =
      Client.storage().metadata().getSnapshot(ID);
  if (!Snapshot)
    return makeJSONError(404, Snapshot.takeError());
  return makeJSONSuccess(200, toJSON(*Snapshot));
}

static HTTPResult handleGetUnits(CoreClient &Client, StringRef SnapID,
                                 StringRef UnitID = "") {
  if (UnitID.empty()) {
    json::Array Array;
    for (const UnitRecord &Unit : Client.listUnits(SnapID))
      Array.push_back(toJSON(Unit));
    return makeJSONSuccess(200, std::move(Array));
  }
  Expected<UnitRecord> Unit = Client.storage().metadata().getUnit(UnitID);
  if (!Unit || Unit->SnapshotID != SnapID) {
    if (!Unit)
      consumeError(Unit.takeError());
    return makeJSONErrorStr(404, "unknown unit");
  }
  return makeJSONSuccess(200, toJSON(*Unit));
}

static HTTPResult handleGetInsights(CoreClient &Client, StringRef SnapID,
                                    StringRef InsightName = "",
                                    StringRef UnitID = "",
                                    StringRef BaselineSnapID = "") {
  if (InsightName.empty()) {
    Expected<json::Array> L = Client.listInsights(SnapID);
    if (!L)
      return makeJSONError(400, L.takeError());
    return makeJSONSuccess(200, std::move(*L));
  }
  std::string Baseline(BaselineSnapID);
  if (UnitID.empty()) {
    Expected<json::Object> R =
        Client.runInsight(InsightName, SnapID, StringRef(), Baseline);
    if (!R)
      return makeJSONError(400, R.takeError());
    return makeJSONSuccess(200, std::move(*R));
  }
  Expected<json::Object> R =
      Client.runInsight(InsightName, SnapID, UnitID, Baseline);
  if (!R)
    return makeJSONError(400, R.takeError());
  return makeJSONSuccess(200, std::move(*R));
}

static HTTPResult handleGetEntities(CoreClient &Client, StringRef SnapID,
                                    StringRef Kind) {
  std::string EntityKind =
      Kind == "link-units" ? "link_unit" : Kind.drop_back().str();
  json::Array Array;
  for (const EntityRecord &Entity :
       Client.storage().metadata().listEntities(EntityKind, SnapID))
    Array.push_back(toJSON(Entity));
  return makeJSONSuccess(200, std::move(Array));
}

// Resolve "latest" to the most-recently-created snapshot ID.
static std::string resolveSnapshotHTTP(CoreClient &Client, StringRef ID) {
  if (ID != "latest")
    return std::string(ID);
  SmallVector<SnapshotRecord, 16> Snapshots = Client.listSnapshots();
  if (Snapshots.empty())
    return std::string(ID);
  const SnapshotRecord *Best = &Snapshots[0];
  for (const SnapshotRecord &S : Snapshots)
    if (S.CreatedUnix > Best->CreatedUnix)
      Best = &S;
  return Best->ID;
}

static HTTPResult handleGetJobs(CoreClient &Client, StringRef ID = "") {
  if (ID.empty()) {
    json::Array Array;
    for (const JobRecord &Job : Client.listJobs())
      Array.push_back(toJSON(Job));
    return makeJSONSuccess(200, std::move(Array));
  }
  Expected<JobRecord> Job = Client.getJob(ID);
  if (!Job)
    return makeJSONError(404, Job.takeError());
  return makeJSONSuccess(200, toJSON(*Job));
}

static HTTPResult handleGetRepresentation(CoreClient &Client, StringRef ID) {
  Expected<EntityRecord> Entity = Client.storage().metadata().getEntity(ID);
  if (!Entity || Entity->Kind != "representation") {
    if (!Entity)
      consumeError(Entity.takeError());
    return makeJSONErrorStr(404, "unknown representation");
  }
  return makeJSONSuccess(200, toJSON(*Entity));
}

static HTTPResult handleGetBlob(CoreClient &Client, StringRef ID) {
  Expected<std::string> Blob = Client.storage().blobs().get(ID);
  if (!Blob)
    return makeJSONError(404, Blob.takeError());
  return {200, "application/octet-stream", *Blob};
}

static HTTPResult handleGetCorrelate(CoreClient &Client, StringRef ID) {
  Expected<json::Value> R = Client.correlateRuntime(ID);
  if (!R)
    return makeJSONError(400, R.takeError());
  return makeJSONSuccess(200, std::move(*R));
}

static bool isSummarySafeCapability(StringRef ID) {
  return ID == "llvm.ir.summary" || ID == "llvm.ir.function_stats" ||
         ID == "clang.diag.summary" || ID == "llvm.obj.summary" ||
         ID == "llvm.remarks.summary" || ID == "llvm.debug.summary" ||
         ID == "llvm.obj.sections" || ID == "llvm.obj.symbols" ||
         ID == "build.compile_commands" || ID == "llvm.lto.summary" ||
         ID == "llvm.lto.function_stats" || ID == "clang.ast.summary";
}

static HTTPResult handleGetSummary(CoreClient &Client, StringRef SnapID) {
  json::Object Summary;
  Summary["snapshot_id"] = SnapID;
  auto Units = Client.listUnits(SnapID);
  Summary["unit_count"] = static_cast<int64_t>(Units.size());
  SmallVector<std::string, 32> Caps;
  for (const CapabilitySpec &Spec : Client.listCapabilities())
    if (shouldQuerySummaryCapability(Spec.ID) && isSummarySafeCapability(Spec.ID))
      Caps.push_back(Spec.ID);
  Expected<json::Array> Query = Client.querySnapshot(SnapID, Caps);
  int64_t Instructions = 0;
  int64_t Functions = 0;
  int64_t Warnings = 0;
  int64_t Errors = 0;
  int64_t Remarks = 0;
  int64_t Available = 0;
  int64_t Missing = 0;
  StringMap<int64_t> MetricTotals;
  StringMap<std::pair<int64_t, int64_t>> FamilyCoverage;
  if (Query) {
    for (const json::Value &UnitValue : *Query) {
      const json::Object *UnitObj = UnitValue.getAsObject();
      const json::Array *Results =
          UnitObj ? UnitObj->getArray("results") : nullptr;
      if (!Results)
        continue;
      for (const json::Value &ResultValue : *Results) {
        const json::Object *ResultObj = ResultValue.getAsObject();
        const json::Object *ValueObj =
            ResultObj ? ResultObj->getObject("value") : nullptr;
        if (!ResultObj || !ValueObj)
          continue;
        std::optional<StringRef> Capability =
            ResultObj->getString("capability");
        bool IsAvailable = ValueObj->getBoolean("available").value_or(true);
        if (IsAvailable)
          ++Available;
        else
          ++Missing;
        if (!Capability)
          continue;
        auto &Coverage = FamilyCoverage[capabilityFamily(*Capability)];
        if (IsAvailable)
          ++Coverage.first;
        else
          ++Coverage.second;
        if (IsAvailable)
          accumulateSummaryMetrics(*ValueObj, MetricTotals);
        if (*Capability == "llvm.ir.summary") {
          Instructions += ValueObj->getInteger("instructions")
                              .value_or(ValueObj->getInteger("instruction_count").value_or(0));
          Functions += ValueObj->getInteger("functions")
                           .value_or(ValueObj->getInteger("function_count").value_or(0));
        } else if (*Capability == "clang.diag.summary") {
          Warnings += ValueObj->getInteger("warnings").value_or(0);
          Errors += ValueObj->getInteger("errors").value_or(0);
        } else if (*Capability == "llvm.remarks.summary") {
          Remarks += ValueObj->getInteger("count")
                         .value_or(ValueObj->getInteger("remark_count").value_or(0));
        }
      }
    }
  } else {
    consumeError(Query.takeError());
  }
  int64_t Health = 100;
  if (Errors > 0)
    Health = std::min<int64_t>(Health, 60);
  Health -= std::min<int64_t>(Warnings, 25);
  Health -= std::min<int64_t>(Missing, 20);
  if (Health < 0)
    Health = 0;
  Summary["instructions"] = Instructions;
  Summary["functions"] = Functions;
  Summary["warnings"] = Warnings;
  Summary["errors"] = Errors;
  Summary["remarks"] = Remarks;
  Summary["available_results"] = Available;
  Summary["missing_results"] = Missing;
  Summary["health_score"] = Health;
  json::Object Metrics;
  for (const auto &KV : MetricTotals)
    Metrics[KV.first()] = KV.second;
  Summary["metrics"] = std::move(Metrics);
  json::Array Families;
  for (const auto &KV : FamilyCoverage) {
    Families.push_back(json::Object{
        {"family", KV.first()},
        {"available", KV.second.first},
        {"missing", KV.second.second},
    });
  }
  Summary["families"] = std::move(Families);
  return makeJSONSuccess(200, std::move(Summary));
}

static HTTPResult handleGetQueryUnit(CoreClient &Client, StringRef UnitID,
                                     StringRef Capabilities) {
  SmallVector<std::string, 16> Caps = parseCapabilityList(Capabilities);
  Expected<json::Array> R = Client.queryUnit(UnitID, Caps);
  if (!R)
    return makeJSONError(400, R.takeError());
  return makeJSONSuccess(200, std::move(*R));
}

static HTTPResult handleGetQuerySnapshot(CoreClient &Client,
                                         StringRef SnapshotID,
                                         StringRef Capabilities) {
  SmallVector<std::string, 16> Caps = parseCapabilityList(Capabilities);
  Expected<json::Array> R = Client.querySnapshot(SnapshotID, Caps);
  if (!R)
    return makeJSONError(400, R.takeError());
  return makeJSONSuccess(200, std::move(*R));
}

static HTTPResult handleGetCompare(CoreClient &Client, StringRef Before,
                                   StringRef After) {
  if (Before.empty() || After.empty())
    return makeJSONErrorStr(400, "before and after snapshot ids are required");
  return makeJSONSuccess(200, Client.compare(Before, After));
}

static HTTPResult handleGetCompareCapability(CoreClient &Client,
                                             StringRef Before, StringRef After,
                                             StringRef CapID) {
  if (Before.empty() || After.empty() || CapID.empty())
    return makeJSONErrorStr(400,
                            "before, after, and capability id are required");
  return makeJSONSuccess(200, Client.compareCapability(Before, After, CapID));
}

static HTTPResult handleInspect(CoreClient &Client, StringRef Mode,
                                StringRef Body) {
  Expected<json::Value> Parsed = json::parse(Body);
  if (!Parsed)
    return makeJSONError(400, Parsed.takeError());
  const json::Object *Object = Parsed->getAsObject();
  if (!Object)
    return makeJSONErrorStr(400, "inspect request body must be a JSON object");

  std::string SnapshotID = resolveSnapshotHTTP(
      Client, Object->getString("snapshot_id").value_or(""));
  StringRef UnitSelector = Object->getString("unit").value_or("");
  StringRef ExplicitCapability = Object->getString("capability").value_or("");
  std::string BaselineSnapshotID = resolveSnapshotHTTP(
      Client, Object->getString("baseline_snapshot_id").value_or(""));

  Expected<std::string> UnitID = Client.resolveUnitID(SnapshotID, UnitSelector);
  if (!UnitID)
    return makeJSONError(400, UnitID.takeError());

  if (Mode == "signals") {
    if (SnapshotID.empty() || UnitID->empty())
      return makeJSONErrorStr(400,
                              "inspect signals requires snapshot_id and unit");
    InspectionFilter Filter;
    Filter.Function = Object->getString("function").value_or("").str();
    Filter.Pass = Object->getString("pass").value_or("").str();
    Filter.Severity = Object->getString("severity").value_or("").str();
    Filter.File = Object->getString("file").value_or("").str();
    Filter.Line = Object->getInteger("line").value_or(-1);
    Filter.Index = Object->getInteger("index").value_or(-1);
    Expected<json::Object> Result =
        Client.inspectSignals(SnapshotID, *UnitID, Filter);
    if (!Result)
      return makeJSONError(400, Result.takeError());
    return makeJSONSuccess(200, std::move(*Result));
  }
  StringRef Capability = inspectModeToCapability(Mode, ExplicitCapability);
  if (SnapshotID.empty() || UnitID->empty() || Capability.empty())
    return makeJSONErrorStr(
        400, "inspect requires snapshot_id, unit, and capability");

  InspectionFilter Filter;
  Filter.Function = Object->getString("function").value_or("").str();
  Filter.Pass = Object->getString("pass").value_or("").str();
  Filter.Severity = Object->getString("severity").value_or("").str();
  Filter.File = Object->getString("file").value_or("").str();
  Filter.Line = Object->getInteger("line").value_or(-1);
  Filter.Index = Object->getInteger("index").value_or(-1);

  Expected<json::Object> Result =
      BaselineSnapshotID.empty()
          ? Client.inspect(SnapshotID, *UnitID, Capability, Filter)
          : Client.inspectCompare(BaselineSnapshotID, SnapshotID, *UnitID,
                                  Capability, Filter);
  if (!Result)
    return makeJSONError(400, Result.takeError());
  return makeJSONSuccess(200, std::move(*Result));
}

bool HTTPServer::checkAuth(const std::string &AuthHeader) const {
  if (AuthToken.empty())
    return true;
  return AuthHeader == AuthToken;
}

void HTTPServer::shutdown() {
  ShutdownFlag.store(true);
  if (PipeFD[1] >= 0) {
    char Byte = 1;
    [[maybe_unused]] ssize_t W = ::write(PipeFD[1], &Byte, 1);
    (void)W;
  }
}

// --- Server Run ---

Error llvm::advisor::HTTPServer::run() {
  if (Port == 0)
    return createStringError(inconvertibleErrorCode(), "invalid port");

  // Load optional auth token from environment
  if (const char *EnvTok = std::getenv("LLVM_ADVISOR_TOKEN"))
    AuthToken = EnvTok;

  std::string Index = StaticHandler().index().str();
#ifdef _WIN32
  return createStringError(inconvertibleErrorCode(),
                           "embedded HTTP server is unsupported on Windows");
#else
  int ListenFD = ::socket(AF_INET, SOCK_STREAM, 0);
  if (ListenFD < 0)
    return createStringError(inconvertibleErrorCode(), "socket failed: %s",
                             std::strerror(errno));
  int Reuse = 1;
  (void)::setsockopt(ListenFD, SOL_SOCKET, SO_REUSEADDR, &Reuse, sizeof(Reuse));
  sockaddr_in Addr{};
  Addr.sin_family = AF_INET;
  Addr.sin_port = htons(static_cast<uint16_t>(Port));
  Addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::bind(ListenFD, reinterpret_cast<sockaddr *>(&Addr), sizeof(Addr)) !=
      0) {
    ::close(ListenFD);
    return createStringError(inconvertibleErrorCode(), "bind failed: %s",
                             std::strerror(errno));
  }
  if (::listen(ListenFD, 16) != 0) {
    ::close(ListenFD);
    return createStringError(inconvertibleErrorCode(), "listen failed: %s",
                             std::strerror(errno));
  }

  // Self-pipe for graceful shutdown
  if (::pipe(PipeFD) != 0) {
    ::close(ListenFD);
    return createStringError(inconvertibleErrorCode(), "pipe failed: %s",
                             std::strerror(errno));
  }

  while (!ShutdownFlag.load()) {
    struct pollfd PollFds[2];
    PollFds[0].fd = ListenFD;
    PollFds[0].events = POLLIN;
    PollFds[1].fd = PipeFD[0];
    PollFds[1].events = POLLIN;

    int Ready = ::poll(PollFds, 2, -1);
    if (Ready < 0) {
      if (errno == EINTR)
        continue;
      break;
    }
    if (PollFds[1].revents & POLLIN)
      break;
    if (!(PollFds[0].revents & POLLIN))
      continue;

    int FD = ::accept(ListenFD, nullptr, nullptr);
    if (FD < 0)
      continue;

    // Set socket timeouts (30s recv, 30s send)
    struct timeval TV;
    TV.tv_sec = 30;
    TV.tv_usec = 0;
    (void)::setsockopt(FD, SOL_SOCKET, SO_RCVTIMEO, &TV, sizeof(TV));
    (void)::setsockopt(FD, SOL_SOCKET, SO_SNDTIMEO, &TV, sizeof(TV));

    Pool.async([this, FD, Index]() {
      ParsedRequest Req;
      if (!readFullRequest(FD, Req) || !Req.Valid) {
        ::close(FD);
        return;
      }

      StringRef Method(Req.Method);
      StringRef Path(Req.Path);
      StringRef QueryStr(Req.Query);
      HTTPResult Res = {404, "text/plain", "not found\n"};

      // Parse query parameters into a map.
      StringMap<std::string> QueryParams;
      {
        SmallVector<StringRef, 8> QSegs;
        QueryStr.split(QSegs, '&', -1, false);
        for (StringRef QSeg : QSegs) {
          auto [Key, Val] = QSeg.split('=');
          QueryParams[Key] = urlDecode(Val);
        }
      }

      SmallVector<StringRef, 16> Segs;
      Path.split(Segs, '/', -1, false);
      bool IsAPI = Segs.size() >= 3 && Segs[0] == "api" && Segs[1] == "v1";

      // Auth check for API routes
      if (IsAPI && !checkAuth(Req.AuthHeader)) {
        Res = makeJSONErrorStr(401, "unauthorized");
        std::string Out =
            makeRawHTTPResponse(Res.Code, Res.ContentType, Res.Body);
        (void)sendAll(FD, Out);
        ::close(FD);
        return;
      }

      if (Method == "GET") {
        if (Path == "/")
          Res = {200, "text/html", Index};
        else if (Path == "/api/v1/health")
          Res = handleGetHealth(Client);
        else if (Path == "/api/v1/status" || Path == "/api/v1/storage")
          Res = handleGetStatus(Client);
        else if (Path == "/api/v1/metrics")
          Res = handleGetMetrics(Client);
        else if (Path == "/api/v1/capabilities")
          Res = handleGetCapabilities(Client);
        else if (IsAPI && Segs.size() == 4 && Segs[2] == "capabilities")
          Res = handleGetCapabilities(Client, Segs[3]);
        else if (Path == "/api/v1/snapshots")
          Res = handleGetSnapshots(Client);
        else if (IsAPI && Segs[2] == "snapshots") {
          std::string ResolvedSnap = resolveSnapshotHTTP(Client, Segs[3]);
          if (Segs.size() == 4)
            Res = handleGetSnapshots(Client, ResolvedSnap);
          else if (Segs.size() == 5 && Segs[4] == "summary")
            Res = handleGetSummary(Client, ResolvedSnap);
          else if (Segs.size() == 5 && Segs[4] == "units")
            Res = handleGetUnits(Client, ResolvedSnap);
          else if (Segs.size() == 6 && Segs[4] == "units")
            Res = handleGetUnits(Client, ResolvedSnap, urlDecode(Segs[5]));
          else if (Segs.size() == 5 && Segs[4] == "insights")
            Res = handleGetInsights(Client, ResolvedSnap);
          else if (Segs.size() == 6 && Segs[4] == "insights") {
            auto BIt = QueryParams.find("baseline");
            std::string BL = BIt != QueryParams.end()
                                 ? resolveSnapshotHTTP(Client, BIt->second)
                                 : "";
            Res = handleGetInsights(Client, ResolvedSnap, Segs[5], /*UnitID=*/"", BL);
          } else if (Segs.size() == 8 && Segs[4] == "units" &&
                   Segs[6] == "insights")
            Res = handleGetInsights(Client, ResolvedSnap, Segs[7], Segs[5]);
          else if (Segs.size() == 5 &&
                   (Segs[4] == "representations" || Segs[4] == "findings" ||
                    Segs[4] == "mappings" || Segs[4] == "link-units"))
            Res = handleGetEntities(Client, ResolvedSnap, Segs[4]);
        } else if (Path == "/api/v1/jobs")
          Res = handleGetJobs(Client);
        else if (IsAPI && Segs.size() == 4 && Segs[2] == "jobs")
          Res = handleGetJobs(Client, Segs[3]);
        else if (IsAPI && Segs.size() == 4 && Segs[2] == "representations")
          Res = handleGetRepresentation(Client, Segs[3]);
        else if (IsAPI && Segs.size() == 5 && Segs[2] == "representations" &&
                 Segs[4] == "metadata")
          Res = handleGetRepresentation(Client, Segs[3]);
        else if (IsAPI && Segs.size() == 4 && Segs[2] == "blobs")
          Res = handleGetBlob(Client, Segs[3]);
        else if (IsAPI && Segs.size() == 5 && Segs[2] == "runtime" &&
                 Segs[4] == "correlate")
          Res = handleGetCorrelate(Client, resolveSnapshotHTTP(Client, Segs[3]));
        else if (IsAPI && Segs.size() == 6 && Segs[2] == "query" &&
                 Segs[3] == "unit")
          Res = handleGetQueryUnit(Client, urlDecode(Segs[4]), Segs[5]);
        else if (IsAPI && Segs.size() == 6 && Segs[2] == "query" &&
                 Segs[3] == "snapshot")
          Res = handleGetQuerySnapshot(Client, resolveSnapshotHTTP(Client, urlDecode(Segs[4])), Segs[5]);
        else if (IsAPI && Segs.size() == 5 && Segs[2] == "compare")
          Res =
              handleGetCompare(Client, resolveSnapshotHTTP(Client, urlDecode(Segs[3])), resolveSnapshotHTTP(Client, urlDecode(Segs[4])));
        else if (IsAPI && Segs.size() == 7 && Segs[2] == "compare" &&
                 Segs[5] == "capability")
          Res = handleGetCompareCapability(Client, resolveSnapshotHTTP(Client, urlDecode(Segs[3])),
                                           resolveSnapshotHTTP(Client, urlDecode(Segs[4])),
                                           urlDecode(Segs[6]));
        else if (!IsAPI)
          Res = {200, "text/html", Index};
      } else if (Method == "POST") {
        if (IsAPI && Segs.size() == 4 && Segs[2] == "inspect")
          Res = handleInspect(Client, Segs[3], Req.Body);
        else
          Res = makeJSONErrorStr(405, "unsupported POST route");
      } else if (Method == "OPTIONS") {
        Res = {200, "text/plain", ""};
      } else {
        Res = makeJSONErrorStr(405, "method not allowed");
      }

      std::string Out = makeRawHTTPResponse(Res.Code, Res.ContentType, Res.Body,
                                            Req.KeepAlive);
      (void)sendAll(FD, Out);
      ::close(FD);
    });
  }

  ::close(ListenFD);
  if (PipeFD[0] >= 0)
    ::close(PipeFD[0]);
  if (PipeFD[1] >= 0)
    ::close(PipeFD[1]);
  Pool.wait();
  return Error::success();
#endif
}
