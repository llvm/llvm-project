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

#include "Analysis/IR/RemarksRelationalSchema.h"
#include "Client/HTTP/HTTPServer.h"
#include "Client/HTTP/Handlers/StaticHandler.h"
#include "Utils/JSON.h"
#include "Utils/Normalization.h"

#include "llvm/Support/MemoryBuffer.h"

#include <cerrno>
#include <cstring>
#include <cmath>
#include <mutex>
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

static std::mutex HeavyQueryMutex;

static std::string renderJSON(const json::Value &Value) {
  std::string Body;
  raw_string_ostream OS(Body);
  json::OStream JOS(OS);
  JOS.value(Value);
  OS.flush();
  return Body;
}

#ifndef _WIN32
static std::string makeRawHTTPHeader(unsigned Code, const char *ContentType,
                                     size_t BodyLen, bool KeepAlive = false) {
  std::string Out;
  raw_string_ostream OS(Out);
  const char *Reason =
      Code == 200
          ? "OK"
          : (Code == 201 ? "Created" : (Code == 404 ? "Not Found" : "Error"));
  OS << "HTTP/1.1 " << Code << ' ' << Reason << "\r\n";
  OS << "Content-Type: " << ContentType << "\r\n";
  OS << "Content-Length: " << BodyLen << "\r\n";
  OS << "Access-Control-Allow-Origin: *\r\n";
  OS << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
  OS << "Access-Control-Allow-Headers: Content-Type\r\n";
  OS << "Connection: " << (KeepAlive ? "keep-alive" : "close") << "\r\n\r\n";
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
      {"remarks", "llvm.remarks.detail"},
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
      {"llvm.remarks.", "IR", false},
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

static bool isSummarySafeCapability(StringRef ID) {
  return ID == "llvm.ir.summary" || ID == "llvm.ir.function_stats" ||
         ID == "clang.diag.summary" || ID == "llvm.obj.summary" ||
         ID == "llvm.remarks.summary" || ID == "llvm.debug.summary" ||
         ID == "llvm.obj.sections" || ID == "llvm.obj.symbols" ||
         ID == "build.compile_commands" || ID == "llvm.lto.summary" ||
         ID == "llvm.lto.function_stats" || ID == "clang.ast.summary";
}

static HTTPResult handleGetSummary(CoreClient &Client, StringRef SnapID) {
  std::lock_guard<std::mutex> Lock(HeavyQueryMutex);
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


namespace {

class RelationalMerger {
public:

  void absorb(const json::Object &Envelope) {
    if (!Envelope.getBoolean("available").value_or(false))
      return;
    const json::Object *Strs = Envelope.getObject("strings");
    const json::Object *Cols = Envelope.getObject("columns");
    if (!Strs || !Cols)
      return;

    int64_t UnitIdx = static_cast<int64_t>(
        Unit.getOrAdd(Envelope.getString("unit_id").value_or("")));

    std::vector<int64_t> PassRemap = remapStrings(Strs, "pass", Pass);
    std::vector<int64_t> NameRemap = remapStrings(Strs, "name", Name);
    std::vector<int64_t> FuncRemap = remapStrings(Strs, "function", Function);
    std::vector<int64_t> FileRemap = remapStrings(Strs, "file", File);

    const json::Array *PassCol = Cols->getArray("pass");
    const json::Array *NameCol = Cols->getArray("name");
    const json::Array *TypeCol = Cols->getArray("type");
    const json::Array *FuncCol = Cols->getArray("function");
    const json::Array *FileCol = Cols->getArray("file");
    const json::Array *LineCol = Cols->getArray("line");
    const json::Array *ColumnCol = Cols->getArray("column");
    const json::Array *HotnessCol = Cols->getArray("hotness");
    if (!PassCol || !NameCol || !TypeCol || !FuncCol || !FileCol || !LineCol ||
        !ColumnCol || !HotnessCol)
      return;
    size_t N = PassCol->size();
    if (NameCol->size() != N || TypeCol->size() != N ||
        FuncCol->size() != N || FileCol->size() != N || LineCol->size() != N ||
        ColumnCol->size() != N || HotnessCol->size() != N)
      return;

    for (size_t I = 0; I < N; ++I) {
      UnitColG.push_back(UnitIdx);
      PassColG.push_back(translate(readInt(*PassCol, I), PassRemap));
      NameColG.push_back(translate(readInt(*NameCol, I), NameRemap));
      TypeColG.push_back(readInt(*TypeCol, I));
      FuncColG.push_back(translate(readInt(*FuncCol, I), FuncRemap));
      FileColG.push_back(translate(readInt(*FileCol, I), FileRemap));
      LineColG.push_back(readInt(*LineCol, I));
      ColumnColG.push_back(readInt(*ColumnCol, I));
      HotnessColG.push_back(readInt(*HotnessCol, I));
    }
  }

  void write(json::OStream &JOS, StringRef SnapshotID) const {
    JOS.objectBegin();
    JOS.attribute("snapshot_id", SnapshotID);
    JOS.attribute("schema_version", 1);
    JOS.attribute("count", static_cast<int64_t>(UnitColG.size()));

    JOS.attributeBegin("strings");
    JOS.objectBegin();
    JOS.attributeBegin("unit");     Unit.writeJSON(JOS);     JOS.attributeEnd();
    JOS.attributeBegin("pass");     Pass.writeJSON(JOS);     JOS.attributeEnd();
    JOS.attributeBegin("name");     Name.writeJSON(JOS);     JOS.attributeEnd();
    JOS.attributeBegin("function"); Function.writeJSON(JOS); JOS.attributeEnd();
    JOS.attributeBegin("file");     File.writeJSON(JOS);     JOS.attributeEnd();
    JOS.objectEnd();
    JOS.attributeEnd();

    JOS.attributeBegin("columns");
    JOS.objectBegin();
    writeInt64Column(JOS, "unit",     UnitColG);
    writeInt64Column(JOS, "pass",     PassColG);
    writeInt64Column(JOS, "name",     NameColG);
    writeInt64Column(JOS, "type",     TypeColG);
    writeInt64Column(JOS, "function", FuncColG);
    writeInt64Column(JOS, "file",     FileColG);
    writeInt64Column(JOS, "line",     LineColG);
    writeInt64Column(JOS, "column",   ColumnColG);
    writeInt64Column(JOS, "hotness",  HotnessColG);
    JOS.objectEnd();
    JOS.attributeEnd();

    JOS.objectEnd();
  }

  size_t size() const { return UnitColG.size(); }

  RelationalStringTable Unit, Pass, Name, Function, File;
  std::vector<int64_t> UnitColG, PassColG, NameColG, TypeColG, FuncColG,
      FileColG, LineColG, ColumnColG, HotnessColG;

private:

  static std::vector<int64_t> remapStrings(const json::Object *Strs,
                                           StringRef Field, RelationalStringTable &Dst) {
    std::vector<int64_t> Map;
    const json::Array *Arr = Strs->getArray(Field);
    if (!Arr)
      return Map;
    Map.reserve(Arr->size());
    for (const json::Value &V : *Arr) {
      std::optional<StringRef> S = V.getAsString();
      Map.push_back(S ? static_cast<int64_t>(Dst.getOrAdd(*S)) : -1);
    }
    return Map;
  }

  static int64_t readInt(const json::Array &A, size_t I) {
    return A[I].getAsInteger().value_or(-1);
  }


  static int64_t translate(int64_t Local, ArrayRef<int64_t> Map) {
    if (Local < 0 || Local >= static_cast<int64_t>(Map.size()))
      return -1;
    return Map[Local];
  }
};

} // namespace

struct RelationalFilter {
  StringRef Pass;
  StringRef Name;
  int64_t Type = -1;
  StringRef Function;
  StringRef File;
  int64_t MinHotness = -1;
};

static HTTPResult handleGetRemarksRelational(CoreClient &Client,
                                             StringRef SnapID,
                                             const RelationalFilter &Filter,
                                             int64_t Offset, int64_t Limit) {
  std::lock_guard<std::mutex> Lock(HeavyQueryMutex);
  SmallVector<UnitRecord, 64> Units =
      Client.storage().metadata().listUnits(SnapID);
  if (Units.empty())
    return makeJSONErrorStr(404, "snapshot has no captured units");

  if (Offset < 0) Offset = 0;
  if (Limit <= 0) Limit = 10000;
  if (Limit > 100000) Limit = 100000;

  const SmallVector<std::string, 1> Caps{"llvm.remarks.relational"};
  bool HasFilter = !Filter.Pass.empty() || !Filter.Name.empty() ||
                   Filter.Type >= 0 || !Filter.Function.empty() ||
                   !Filter.File.empty() || Filter.MinHotness >= 0;

  auto matchesFilter = [&](const json::Array *PassStrs,
                           const json::Array *NameStrs,
                           const json::Array *FuncStrs,
                           const json::Array *FileStrs,
                           int64_t PassIdx, int64_t NameIdx, int64_t Type,
                           int64_t FuncIdx, int64_t FileIdx,
                           int64_t Hotness) -> bool {
    if (Filter.Type >= 0 && Type != Filter.Type) return false;
    if (Filter.MinHotness >= 0 && Hotness < Filter.MinHotness) return false;
    if (!Filter.Pass.empty()) {
      if (PassIdx < 0 || PassIdx >= (int64_t)PassStrs->size()) return false;
      StringRef S = (*PassStrs)[PassIdx].getAsString().value_or("");
      if (!S.contains_insensitive(Filter.Pass)) return false;
    }
    if (!Filter.Name.empty()) {
      if (NameIdx < 0 || NameIdx >= (int64_t)NameStrs->size()) return false;
      StringRef S = (*NameStrs)[NameIdx].getAsString().value_or("");
      if (!S.contains_insensitive(Filter.Name)) return false;
    }
    if (!Filter.Function.empty()) {
      if (FuncIdx < 0 || FuncIdx >= (int64_t)FuncStrs->size()) return false;
      StringRef S = (*FuncStrs)[FuncIdx].getAsString().value_or("");
      if (!S.contains_insensitive(Filter.Function)) return false;
    }
    if (!Filter.File.empty()) {
      if (FileIdx < 0 || FileIdx >= (int64_t)FileStrs->size()) return false;
      StringRef S = (*FileStrs)[FileIdx].getAsString().value_or("");
      if (!S.contains_insensitive(Filter.File)) return false;
    }
    return true;
  };

  auto getUnitResults = [&](const UnitRecord &U)
      -> SmallVector<const json::Object *, 1> {
    SmallVector<const json::Object *, 1> Out;
    Expected<json::Array> Results = Client.queryUnit(U.ID, Caps);
    if (!Results) { consumeError(Results.takeError()); return Out; }
    for (const json::Value &RV : *Results) {
      const json::Object *RO = RV.getAsObject();
      if (!RO) continue;
      if (RO->getString("capability").value_or("") != "llvm.remarks.relational")
        continue;
      if (const json::Object *VO = RO->getObject("value"))
        Out.push_back(VO);
    }
    return Out;
  };

  // Pass 1: count total matching rows (streaming, one unit at a time)
  int64_t TotalMatching = 0;
  for (const UnitRecord &U : Units) {
    Expected<json::Array> Results = Client.queryUnit(U.ID, Caps);
    if (!Results) { consumeError(Results.takeError()); continue; }
    for (const json::Value &RV : *Results) {
      const json::Object *RO = RV.getAsObject();
      if (!RO || RO->getString("capability").value_or("") != "llvm.remarks.relational")
        continue;
      const json::Object *VO = RO->getObject("value");
      if (!VO) continue;
      const json::Object *Cols = VO->getObject("columns");
      const json::Object *Strs = VO->getObject("strings");
      if (!Cols || !Strs) continue;
      const json::Array *PassCol = Cols->getArray("pass");
      if (!PassCol) continue;
      size_t N = PassCol->size();
      if (!HasFilter) { TotalMatching += N; continue; }
      const json::Array *NameCol = Cols->getArray("name");
      const json::Array *TypeCol = Cols->getArray("type");
      const json::Array *FuncCol = Cols->getArray("function");
      const json::Array *FileCol = Cols->getArray("file");
      const json::Array *HotnessCol = Cols->getArray("hotness");
      const json::Array *PassStrs = Strs->getArray("pass");
      const json::Array *NameStrs = Strs->getArray("name");
      const json::Array *FuncStrs = Strs->getArray("function");
      const json::Array *FileStrs = Strs->getArray("file");
      if (!NameCol || !TypeCol || !FuncCol || !FileCol || !HotnessCol ||
          !PassStrs || !NameStrs || !FuncStrs || !FileStrs) continue;
      for (size_t I = 0; I < N; ++I) {
        int64_t PI = (*PassCol)[I].getAsInteger().value_or(-1);
        int64_t NI = (*NameCol)[I].getAsInteger().value_or(-1);
        int64_t T  = (*TypeCol)[I].getAsInteger().value_or(-1);
        int64_t FI = (*FuncCol)[I].getAsInteger().value_or(-1);
        int64_t FiI = (*FileCol)[I].getAsInteger().value_or(-1);
        int64_t H  = (*HotnessCol)[I].getAsInteger().value_or(-1);
        if (matchesFilter(PassStrs, NameStrs, FuncStrs, FileStrs, PI, NI, T, FI, FiI, H))
          ++TotalMatching;
      }
    }
  }

  int64_t Count = std::min(Limit, std::max((int64_t)0, TotalMatching - Offset));

  // Pass 2: collect only the page rows into a small merger
  RelationalMerger Page;
  int64_t Seen = 0;
  int64_t Collected = 0;
  for (const UnitRecord &U : Units) {
    if (Collected >= Count) break;
    Expected<json::Array> Results = Client.queryUnit(U.ID, Caps);
    if (!Results) { consumeError(Results.takeError()); continue; }
    for (const json::Value &RV : *Results) {
      if (Collected >= Count) break;
      const json::Object *RO = RV.getAsObject();
      if (!RO || RO->getString("capability").value_or("") != "llvm.remarks.relational")
        continue;
      const json::Object *VO = RO->getObject("value");
      if (!VO) continue;
      const json::Object *Cols = VO->getObject("columns");
      const json::Object *Strs = VO->getObject("strings");
      if (!Cols || !Strs) continue;
      const json::Array *PassCol = Cols->getArray("pass");
      const json::Array *NameCol = Cols->getArray("name");
      const json::Array *TypeCol = Cols->getArray("type");
      const json::Array *FuncCol = Cols->getArray("function");
      const json::Array *FileCol = Cols->getArray("file");
      const json::Array *LineCol = Cols->getArray("line");
      const json::Array *ColumnCol = Cols->getArray("column");
      const json::Array *HotnessCol = Cols->getArray("hotness");
      const json::Array *PassStrs = Strs->getArray("pass");
      const json::Array *NameStrs = Strs->getArray("name");
      const json::Array *FuncStrs = Strs->getArray("function");
      const json::Array *FileStrs = Strs->getArray("file");
      if (!PassCol || !NameCol || !TypeCol || !FuncCol || !FileCol ||
          !LineCol || !ColumnCol || !HotnessCol || !PassStrs || !NameStrs ||
          !FuncStrs || !FileStrs)
        continue;
      StringRef UnitID = VO->getString("unit_id").value_or(U.ID);
      size_t N = PassCol->size();
      for (size_t I = 0; I < N; ++I) {
        if (Collected >= Count) break;
        int64_t PI = (*PassCol)[I].getAsInteger().value_or(-1);
        int64_t NI = (*NameCol)[I].getAsInteger().value_or(-1);
        int64_t T  = (*TypeCol)[I].getAsInteger().value_or(-1);
        int64_t FI = (*FuncCol)[I].getAsInteger().value_or(-1);
        int64_t FiI = (*FileCol)[I].getAsInteger().value_or(-1);
        int64_t H  = (*HotnessCol)[I].getAsInteger().value_or(-1);
        if (HasFilter &&
            !matchesFilter(PassStrs, NameStrs, FuncStrs, FileStrs, PI, NI, T, FI, FiI, H))
          continue;
        if (Seen < Offset) { ++Seen; continue; }
        ++Seen;
        Page.UnitColG.push_back(static_cast<int64_t>(Page.Unit.getOrAdd(UnitID)));
        Page.PassColG.push_back(static_cast<int64_t>(
            Page.Pass.getOrAdd(PI >= 0 ? (*PassStrs)[PI].getAsString().value_or("") : "")));
        Page.NameColG.push_back(static_cast<int64_t>(
            Page.Name.getOrAdd(NI >= 0 ? (*NameStrs)[NI].getAsString().value_or("") : "")));
        Page.TypeColG.push_back(T);
        Page.FuncColG.push_back(static_cast<int64_t>(
            Page.Function.getOrAdd(FI >= 0 ? (*FuncStrs)[FI].getAsString().value_or("") : "")));
        Page.FileColG.push_back(static_cast<int64_t>(
            Page.File.getOrAdd(FiI >= 0 ? (*FileStrs)[FiI].getAsString().value_or("") : "")));
        Page.LineColG.push_back((*LineCol)[I].getAsInteger().value_or(-1));
        Page.ColumnColG.push_back((*ColumnCol)[I].getAsInteger().value_or(-1));
        Page.HotnessColG.push_back(H);
        ++Collected;
      }
    }
  }

  std::string Body;
  raw_string_ostream OS(Body);
  writeSuccessEnvelope(OS, [&](json::OStream &JOS) {
    JOS.objectBegin();
    JOS.attribute("snapshot_id", SnapID);
    JOS.attribute("schema_version", 1);
    JOS.attribute("total", TotalMatching);
    JOS.attribute("offset", Offset);
    JOS.attribute("limit", Limit);
    JOS.attribute("count", Collected);

    JOS.attributeBegin("strings");
    JOS.objectBegin();
    JOS.attributeBegin("unit");     Page.Unit.writeJSON(JOS);     JOS.attributeEnd();
    JOS.attributeBegin("pass");     Page.Pass.writeJSON(JOS);     JOS.attributeEnd();
    JOS.attributeBegin("name");     Page.Name.writeJSON(JOS);     JOS.attributeEnd();
    JOS.attributeBegin("function"); Page.Function.writeJSON(JOS); JOS.attributeEnd();
    JOS.attributeBegin("file");     Page.File.writeJSON(JOS);     JOS.attributeEnd();
    JOS.objectEnd();
    JOS.attributeEnd();

    JOS.attributeBegin("columns");
    JOS.objectBegin();
    writeInt64Column(JOS, "unit",     Page.UnitColG);
    writeInt64Column(JOS, "pass",     Page.PassColG);
    writeInt64Column(JOS, "name",     Page.NameColG);
    writeInt64Column(JOS, "type",     Page.TypeColG);
    writeInt64Column(JOS, "function", Page.FuncColG);
    writeInt64Column(JOS, "file",     Page.FileColG);
    writeInt64Column(JOS, "line",     Page.LineColG);
    writeInt64Column(JOS, "column",   Page.ColumnColG);
    writeInt64Column(JOS, "hotness",  Page.HotnessColG);
    JOS.objectEnd();
    JOS.attributeEnd();

    JOS.objectEnd();
  });
  OS.flush();
  return HTTPResult{200, "application/json", std::move(Body)};
}

static HTTPResult handleGetSourceFiles(CoreClient &Client,
                                       StringRef SnapID) {
  const SmallVector<std::string, 1> Caps{"llvm.remarks.relational"};
  Expected<json::Array> Query = Client.querySnapshot(SnapID, Caps);
  if (!Query)
    return makeJSONError(400, Query.takeError());

  StringMap<int64_t> FileCounts;
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
      if (!ValueObj)
        continue;
      std::optional<StringRef> Capability =
          ResultObj->getString("capability");
      if (!Capability || *Capability != "llvm.remarks.relational")
        continue;
      const json::Object *Cols = ValueObj->getObject("columns");
      const json::Object *Strs = ValueObj->getObject("strings");
      if (!Cols || !Strs)
        continue;
      const json::Array *FileCol = Cols->getArray("file");
      const json::Array *FileStrs = Strs->getArray("file");
      if (!FileCol || !FileStrs)
        continue;
      for (const json::Value &V : *FileCol) {
        std::optional<int64_t> Idx = V.getAsInteger();
        if (!Idx || *Idx < 0 ||
            *Idx >= static_cast<int64_t>(FileStrs->size()))
          continue;
        StringRef FilePath =
            (*FileStrs)[static_cast<size_t>(*Idx)]
                .getAsString()
                .value_or("");
        if (!FilePath.empty())
          FileCounts[FilePath] += 1;
      }
    }
  }

  json::Array Files;
  for (const auto &KV : FileCounts) {
    Files.push_back(
        json::Object{{"path", KV.first()}, {"remarks_count", KV.second}});
  }
  std::sort(Files.begin(), Files.end(),
            [](const json::Value &A, const json::Value &B) {
              const json::Object *OA = A.getAsObject();
              const json::Object *OB = B.getAsObject();
              int64_t CA =
                  OA ? OA->getInteger("remarks_count").value_or(0) : 0;
              int64_t CB =
                  OB ? OB->getInteger("remarks_count").value_or(0) : 0;
              return CA > CB;
            });
  return makeJSONSuccess(200, std::move(Files));
}

static HTTPResult handleGetSource(CoreClient &Client, StringRef SnapID,
                                  StringRef FilePath) {
  Expected<SnapshotRecord> Snap =
      Client.storage().metadata().getSnapshot(SnapID);
  if (!Snap)
    return makeJSONError(404, Snap.takeError());

  SmallVector<StringRef, 2> Roots;
  if (!Snap->SourceRoot.empty())
    Roots.push_back(Snap->SourceRoot);
  if (!Snap->BuildRoot.empty())
    Roots.push_back(Snap->BuildRoot);

  if (Roots.empty())
    return makeJSONErrorStr(400, "snapshot has no source or build root");

  std::string ResolvedPath;
  bool Found = false;

  if (sys::path::is_absolute(FilePath)) {
    Expected<std::string> R = canonicalizePath(FilePath, Roots);
    if (R) { ResolvedPath = std::move(*R); Found = true; }
    else {
      consumeError(R.takeError());
      SmallString<256> Real;
      if (!sys::fs::real_path(FilePath, Real)) {
        ResolvedPath = std::string(Real);
        Found = true;
      }
    }
  }

  if (!Found) {
    for (StringRef Root : Roots) {
      for (StringRef Base : {Root, StringRef(sys::path::parent_path(Root))}) {
        SmallString<256> Joined(Base);
        sys::path::append(Joined, FilePath);
        if (sys::fs::exists(Joined)) {
          SmallString<256> Real;
          if (!sys::fs::real_path(Joined, Real)) {
            ResolvedPath = std::string(Real);
            Found = true;
            break;
          }
        }
      }
      if (Found) break;
    }
  }

  if (!Found)
    return makeJSONErrorStr(404, "source file not found");

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
      MemoryBuffer::getFile(ResolvedPath);
  if (!Buf)
    return makeJSONErrorStr(404, "source file not found");

  StringRef Content = (*Buf)->getBuffer();
  json::Object Result;
  Result["path"] = ResolvedPath;
  Result["content"] = Content.str();
  Result["lines"] = static_cast<int64_t>(
                        std::count(Content.begin(), Content.end(), '\n')) +
                    1;
  return makeJSONSuccess(200, std::move(Result));
}

static HTTPResult handleGetSourceRemarks(CoreClient &Client, StringRef SnapID,
                                         StringRef FilePath, StringRef FilterPass,
                                         StringRef FilterName, int64_t FilterType) {
  if (FilePath.empty())
    return makeJSONErrorStr(400, "path parameter is required");

  std::lock_guard<std::mutex> Lock(HeavyQueryMutex);
  const SmallVector<std::string, 1> Caps{"llvm.remarks.relational"};
  SmallVector<UnitRecord, 64> Units =
      Client.storage().metadata().listUnits(SnapID);

  std::string Body;
  raw_string_ostream OS(Body);
  int64_t Count = 0;

  writeSuccessEnvelope(OS, [&](json::OStream &JOS) {
    JOS.object([&] {
      JOS.attribute("path", FilePath);
      JOS.attributeBegin("remarks");
      JOS.arrayBegin();

      for (const UnitRecord &Unit : Units) {
        Expected<json::Array> Results = Client.queryUnit(Unit.ID, Caps);
        if (!Results) { consumeError(Results.takeError()); continue; }
        for (const json::Value &RV : *Results) {
          const json::Object *RO = RV.getAsObject();
          if (!RO) continue;
          if (RO->getString("capability").value_or("") != "llvm.remarks.relational")
            continue;
          const json::Object *VO = RO->getObject("value");
          if (!VO) continue;
          const json::Object *Cols = VO->getObject("columns");
          const json::Object *Strs = VO->getObject("strings");
          if (!Cols || !Strs) continue;
          const json::Array *FileCol = Cols->getArray("file");
          const json::Array *FileStrs = Strs->getArray("file");
          const json::Array *LineCol = Cols->getArray("line");
          const json::Array *ColumnCol = Cols->getArray("column");
          const json::Array *PassCol = Cols->getArray("pass");
          const json::Array *NameCol = Cols->getArray("name");
          const json::Array *TypeCol = Cols->getArray("type");
          const json::Array *HotnessCol = Cols->getArray("hotness");
          const json::Array *FuncCol = Cols->getArray("function");
          const json::Array *PassStrs = Strs->getArray("pass");
          const json::Array *NameStrs = Strs->getArray("name");
          const json::Array *FuncStrs = Strs->getArray("function");
          if (!FileCol || !FileStrs || !LineCol || !ColumnCol || !PassCol ||
              !NameCol || !TypeCol || !HotnessCol || !FuncCol || !PassStrs ||
              !NameStrs || !FuncStrs)
            continue;

          int64_t TargetIdx = -1;
          for (size_t I = 0; I < FileStrs->size(); ++I) {
            std::optional<StringRef> S = (*FileStrs)[I].getAsString();
            if (S && *S == FilePath) { TargetIdx = I; break; }
          }
          if (TargetIdx < 0) continue;

          for (size_t I = 0; I < FileCol->size(); ++I) {
            if ((*FileCol)[I].getAsInteger().value_or(-1) != TargetIdx) continue;
            int64_t PI = (*PassCol)[I].getAsInteger().value_or(-1);
            int64_t NI = (*NameCol)[I].getAsInteger().value_or(-1);
            int64_t T = (*TypeCol)[I].getAsInteger().value_or(-1);
            if (FilterType >= 0 && T != FilterType) continue;
            StringRef PassStr = PI >= 0 && PI < (int64_t)PassStrs->size()
                ? (*PassStrs)[PI].getAsString().value_or("") : "";
            StringRef NameStr = NI >= 0 && NI < (int64_t)NameStrs->size()
                ? (*NameStrs)[NI].getAsString().value_or("") : "";
            if (!FilterPass.empty() && !PassStr.contains_insensitive(FilterPass)) continue;
            if (!FilterName.empty() && !NameStr.contains_insensitive(FilterName)) continue;

            int64_t FI = (*FuncCol)[I].getAsInteger().value_or(-1);
            JOS.object([&] {
              JOS.attribute("line", (*LineCol)[I].getAsInteger().value_or(-1));
              JOS.attribute("column", (*ColumnCol)[I].getAsInteger().value_or(-1));
              JOS.attribute("type", T);
              JOS.attribute("pass", PassStr);
              JOS.attribute("name", NameStr);
              int64_t H = (*HotnessCol)[I].getAsInteger().value_or(-1);
              if (H >= 0) JOS.attribute("hotness", H);
              if (FI >= 0 && FI < (int64_t)FuncStrs->size())
                JOS.attribute("function", (*FuncStrs)[FI].getAsString().value_or(""));
            });
            ++Count;
          }
        }
      }

      JOS.arrayEnd();
      JOS.attributeEnd();
      JOS.attribute("count", Count);
    });
  });
  OS.flush();
  return HTTPResult{200, "application/json", std::move(Body)};
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
  std::lock_guard<std::mutex> Lock(HeavyQueryMutex);
  SmallVector<std::string, 16> Caps = parseCapabilityList(Capabilities);
  llvm::erase_if(Caps, [](const std::string &C) {
    return C == "llvm.remarks.detail";
  });

  SmallVector<UnitRecord, 64> Units =
      Client.storage().metadata().listUnits(SnapshotID);
  if (Units.empty())
    return makeJSONErrorStr(404, "snapshot has no units");

  static constexpr size_t MaxUnits = 200;
  size_t Limit = std::min(Units.size(), MaxUnits);
  json::Array Out;
  for (size_t I = 0; I < Limit; ++I) {
    Expected<json::Array> R = Client.queryUnit(Units[I].ID, Caps);
    if (!R) { consumeError(R.takeError()); continue; }
    Out.push_back(json::Object{{"unit_id", Units[I].ID},
                               {"source_path", Units[I].SourcePath},
                               {"results", std::move(*R)}});
  }
  return makeJSONSuccess(200, std::move(Out));
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

struct FuncProfile {
  int64_t Total = 0, Missed = 0, Passed = 0, Analysis = 0, HotnessSum = 0;
};

static void buildFuncProfiles(CoreClient &Client, StringRef SnapID,
                              StringMap<FuncProfile> &Out) {
  const SmallVector<std::string, 1> Caps{"llvm.remarks.relational"};
  SmallVector<UnitRecord, 64> Units =
      Client.storage().metadata().listUnits(SnapID);
  for (const UnitRecord &U : Units) {
    Expected<json::Array> Results = Client.queryUnit(U.ID, Caps);
    if (!Results) { consumeError(Results.takeError()); continue; }
    for (const json::Value &RV : *Results) {
      const json::Object *RO = RV.getAsObject();
      if (!RO || RO->getString("capability").value_or("") != "llvm.remarks.relational")
        continue;
      const json::Object *VO = RO->getObject("value");
      if (!VO) continue;
      const json::Object *Cols = VO->getObject("columns");
      const json::Object *Strs = VO->getObject("strings");
      if (!Cols || !Strs) continue;
      const json::Array *TypeCol = Cols->getArray("type");
      const json::Array *FuncCol = Cols->getArray("function");
      const json::Array *HotnessCol = Cols->getArray("hotness");
      const json::Array *FuncStrs = Strs->getArray("function");
      if (!TypeCol || !FuncCol || !HotnessCol || !FuncStrs) continue;
      size_t N = TypeCol->size();
      for (size_t I = 0; I < N; ++I) {
        int64_t FI = (*FuncCol)[I].getAsInteger().value_or(-1);
        if (FI < 0 || FI >= (int64_t)FuncStrs->size()) continue;
        StringRef Func = (*FuncStrs)[FI].getAsString().value_or("");
        if (Func.empty()) continue;
        int64_t T = (*TypeCol)[I].getAsInteger().value_or(0);
        int64_t H = (*HotnessCol)[I].getAsInteger().value_or(0);
        FuncProfile &P = Out[Func];
        P.Total++;
        if (T == 1) P.Passed++;
        else if (T == 2) P.Missed++;
        else if (T == 3) P.Analysis++;
        if (H > 0) P.HotnessSum += H;
      }
    }
  }
}

static HTTPResult handleGetCompareRemarks(CoreClient &Client,
                                          StringRef Before, StringRef After,
                                          int64_t Offset, int64_t Limit) {
  std::lock_guard<std::mutex> Lock(HeavyQueryMutex);

  StringMap<FuncProfile> BeforeProf, AfterProf;
  buildFuncProfiles(Client, Before, BeforeProf);
  buildFuncProfiles(Client, After, AfterProf);

  struct FuncDiff {
    StringRef Name;
    FuncProfile Before, After;
    int64_t DeltaMissed;
    int64_t DeltaTotal;
  };

  SmallVector<FuncDiff, 256> Diffs;
  StringSet<> Seen;

  for (auto &KV : AfterProf) {
    Seen.insert(KV.first());
    FuncProfile B = BeforeProf.lookup(KV.first());
    FuncDiff D;
    D.Name = KV.first();
    D.Before = B;
    D.After = KV.second;
    D.DeltaMissed = KV.second.Missed - B.Missed;
    D.DeltaTotal = KV.second.Total - B.Total;
    if (D.DeltaMissed != 0 || D.DeltaTotal != 0 || B.Total == 0)
      Diffs.push_back(D);
  }
  for (auto &KV : BeforeProf) {
    if (Seen.contains(KV.first())) continue;
    FuncDiff D;
    D.Name = KV.first();
    D.Before = KV.second;
    D.DeltaMissed = -KV.second.Missed;
    D.DeltaTotal = -KV.second.Total;
    Diffs.push_back(D);
  }

  llvm::sort(Diffs, [](const FuncDiff &A, const FuncDiff &B) {
    return std::abs(A.DeltaMissed) > std::abs(B.DeltaMissed);
  });

  int64_t TotalChanged = Diffs.size();
  int64_t Added = 0, Removed = 0, NewMissed = 0, ResolvedMissed = 0;
  for (auto &D : Diffs) {
    if (D.Before.Total == 0) Added++;
    else if (D.After.Total == 0) Removed++;
    if (D.DeltaMissed > 0) NewMissed += D.DeltaMissed;
    else ResolvedMissed += -D.DeltaMissed;
  }

  if (Offset < 0) Offset = 0;
  if (Limit <= 0) Limit = 100;

  std::string Body;
  raw_string_ostream OS(Body);
  writeSuccessEnvelope(OS, [&](json::OStream &JOS) {
    JOS.object([&] {
      JOS.attribute("total", TotalChanged);
      JOS.attribute("offset", Offset);
      JOS.attribute("limit", Limit);
      JOS.attributeBegin("summary");
      JOS.object([&] {
        JOS.attribute("functions_changed", TotalChanged);
        JOS.attribute("functions_added", Added);
        JOS.attribute("functions_removed", Removed);
        JOS.attribute("new_missed", NewMissed);
        JOS.attribute("resolved_missed", ResolvedMissed);
      });
      JOS.attributeEnd();
      JOS.attributeBegin("functions");
      JOS.arrayBegin();
      int64_t End = std::min(Offset + Limit, TotalChanged);
      for (int64_t I = Offset; I < End; ++I) {
        auto &D = Diffs[I];
        JOS.object([&] {
          JOS.attribute("name", D.Name);
          JOS.attributeBegin("before");
          JOS.object([&] {
            JOS.attribute("total", D.Before.Total);
            JOS.attribute("missed", D.Before.Missed);
            JOS.attribute("passed", D.Before.Passed);
            JOS.attribute("analysis", D.Before.Analysis);
            JOS.attribute("hotness_sum", D.Before.HotnessSum);
          });
          JOS.attributeEnd();
          JOS.attributeBegin("after");
          JOS.object([&] {
            JOS.attribute("total", D.After.Total);
            JOS.attribute("missed", D.After.Missed);
            JOS.attribute("passed", D.After.Passed);
            JOS.attribute("analysis", D.After.Analysis);
            JOS.attribute("hotness_sum", D.After.HotnessSum);
          });
          JOS.attributeEnd();
          JOS.attribute("delta_missed", D.DeltaMissed);
          JOS.attribute("delta_total", D.DeltaTotal);
        });
      }
      JOS.arrayEnd();
      JOS.attributeEnd();
    });
  });
  OS.flush();
  return HTTPResult{200, "application/json", std::move(Body)};
}

static HTTPResult handleGetCompareFunctionDetail(CoreClient &Client,
                                                 StringRef Before,
                                                 StringRef After,
                                                 StringRef FuncName) {
  std::lock_guard<std::mutex> Lock(HeavyQueryMutex);

  struct Remark { std::string Pass, Name; int64_t Type, Line, Hotness; };

  auto collectRemarks = [&](StringRef SnapID) {
    std::vector<Remark> Out;
    const SmallVector<std::string, 1> Caps{"llvm.remarks.relational"};
    SmallVector<UnitRecord, 64> Units =
        Client.storage().metadata().listUnits(SnapID);
    for (const UnitRecord &U : Units) {
      Expected<json::Array> Results = Client.queryUnit(U.ID, Caps);
      if (!Results) { consumeError(Results.takeError()); continue; }
      for (const json::Value &RV : *Results) {
        const json::Object *RO = RV.getAsObject();
        if (!RO || RO->getString("capability").value_or("") != "llvm.remarks.relational")
          continue;
        const json::Object *VO = RO->getObject("value");
        if (!VO) continue;
        const json::Object *Cols = VO->getObject("columns");
        const json::Object *Strs = VO->getObject("strings");
        if (!Cols || !Strs) continue;
        const json::Array *FuncCol = Cols->getArray("function");
        const json::Array *FuncStrs = Strs->getArray("function");
        const json::Array *PassCol = Cols->getArray("pass");
        const json::Array *NameCol = Cols->getArray("name");
        const json::Array *TypeCol = Cols->getArray("type");
        const json::Array *LineCol = Cols->getArray("line");
        const json::Array *HotnessCol = Cols->getArray("hotness");
        const json::Array *PassStrs = Strs->getArray("pass");
        const json::Array *NameStrs = Strs->getArray("name");
        if (!FuncCol || !FuncStrs || !PassCol || !NameCol || !TypeCol ||
            !LineCol || !HotnessCol || !PassStrs || !NameStrs) continue;
        size_t N = FuncCol->size();
        for (size_t I = 0; I < N; ++I) {
          int64_t FI = (*FuncCol)[I].getAsInteger().value_or(-1);
          if (FI < 0 || FI >= (int64_t)FuncStrs->size()) continue;
          if ((*FuncStrs)[FI].getAsString().value_or("") != FuncName) continue;
          int64_t PI = (*PassCol)[I].getAsInteger().value_or(-1);
          int64_t NI = (*NameCol)[I].getAsInteger().value_or(-1);
          Remark R;
          R.Pass = PI >= 0 && PI < (int64_t)PassStrs->size()
              ? (*PassStrs)[PI].getAsString().value_or("").str() : "";
          R.Name = NI >= 0 && NI < (int64_t)NameStrs->size()
              ? (*NameStrs)[NI].getAsString().value_or("").str() : "";
          R.Type = (*TypeCol)[I].getAsInteger().value_or(-1);
          R.Line = (*LineCol)[I].getAsInteger().value_or(-1);
          R.Hotness = (*HotnessCol)[I].getAsInteger().value_or(-1);
          Out.push_back(std::move(R));
        }
      }
    }
    return Out;
  };

  std::vector<Remark> BeforeRems = collectRemarks(Before);
  std::vector<Remark> AfterRems = collectRemarks(After);

  // Match by (pass, name, type) — group and count
  struct Key { std::string Pass, Name; int64_t Type; };
  auto makeKey = [](const Remark &R) { return R.Pass + "\0" + R.Name + "\0" + std::to_string(R.Type); };

  StringMap<int64_t> BeforeCounts, AfterCounts;
  for (auto &R : BeforeRems) BeforeCounts[makeKey(R)]++;
  for (auto &R : AfterRems) AfterCounts[makeKey(R)]++;

  json::Array Added, Removed;
  StringSet<> AllKeys;
  for (auto &KV : AfterCounts) AllKeys.insert(KV.first());
  for (auto &KV : BeforeCounts) AllKeys.insert(KV.first());

  for (auto &K : AllKeys) {
    int64_t B = BeforeCounts.lookup(K.getKey());
    int64_t A = AfterCounts.lookup(K.getKey());
    if (A > B) {
      // Parse key back
      StringRef S = K.getKey();
      auto [PassName, Rest] = S.split('\0');
      auto [Name, TypeStr] = Rest.split('\0');
      int64_t Type = 0; TypeStr.getAsInteger(10, Type);
      for (int64_t I = 0; I < A - B; ++I)
        Added.push_back(json::Object{{"pass", PassName}, {"name", Name}, {"type", Type}, {"count", A - B}});
      // Only push once
      break;
    }
  }
  // Rebuild properly
  Added.clear();
  Removed.clear();
  for (auto &K : AllKeys) {
    int64_t B = BeforeCounts.lookup(K.getKey());
    int64_t A = AfterCounts.lookup(K.getKey());
    if (A == B) continue;
    StringRef S = K.getKey();
    auto [Pass, Rest] = S.split('\0');
    auto [Name, TypeStr] = Rest.split('\0');
    int64_t Type = 0; TypeStr.getAsInteger(10, Type);
    json::Object Entry;
    Entry["pass"] = Pass; Entry["name"] = Name; Entry["type"] = Type;
    Entry["before_count"] = B; Entry["after_count"] = A;
    Entry["delta"] = A - B;
    if (A > B) Added.push_back(std::move(Entry));
    else Removed.push_back(std::move(Entry));
  }

  json::Object Result;
  Result["function"] = FuncName.str();
  Result["before_total"] = static_cast<int64_t>(BeforeRems.size());
  Result["after_total"] = static_cast<int64_t>(AfterRems.size());
  Result["added"] = std::move(Added);
  Result["removed"] = std::move(Removed);
  return makeJSONSuccess(200, std::move(Result));
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
        std::string Header =
            makeRawHTTPHeader(Res.Code, Res.ContentType, Res.Body.size());
        (void)sendAll(FD, Header);
        (void)sendAll(FD, Res.Body);
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
          else if (Segs.size() == 5 &&
                   (Segs[4] == "representations" || Segs[4] == "findings" ||
                    Segs[4] == "mappings" || Segs[4] == "link-units"))
            Res = handleGetEntities(Client, ResolvedSnap, Segs[4]);
          else if (Segs.size() == 6 && Segs[4] == "remarks" &&
                   Segs[5] == "relational") {
            RelationalFilter Filter;
            auto getParam = [&](StringRef K) -> StringRef {
              auto It = QueryParams.find(K);
              return It != QueryParams.end() ? StringRef(It->second) : "";
            };
            Filter.Pass = getParam("pass");
            Filter.Name = getParam("name");
            Filter.Function = getParam("function");
            Filter.File = getParam("file");
            auto TypeIt = QueryParams.find("type");
            if (TypeIt != QueryParams.end())
              StringRef(TypeIt->second).getAsInteger(10, Filter.Type);
            auto HotIt = QueryParams.find("min_hotness");
            if (HotIt != QueryParams.end())
              StringRef(HotIt->second).getAsInteger(10, Filter.MinHotness);
            int64_t Offset = 0, Limit = 10000;
            auto OffIt = QueryParams.find("offset");
            if (OffIt != QueryParams.end())
              StringRef(OffIt->second).getAsInteger(10, Offset);
            auto LimIt = QueryParams.find("limit");
            if (LimIt != QueryParams.end())
              StringRef(LimIt->second).getAsInteger(10, Limit);
            Res = handleGetRemarksRelational(Client, ResolvedSnap, Filter,
                                            Offset, Limit);
          }
          else if (Segs.size() == 5 && Segs[4] == "files")
            Res = handleGetSourceFiles(Client, ResolvedSnap);
        } else if (Path == "/api/v1/source/remarks") {
          auto PathIt = QueryParams.find("path");
          auto SnapIt = QueryParams.find("snapshot_id");
          if (PathIt == QueryParams.end() || SnapIt == QueryParams.end())
            Res = makeJSONErrorStr(400, "path and snapshot_id required");
          else {
            StringRef FPass, FName;
            int64_t FType = -1;
            auto PIt = QueryParams.find("pass");
            if (PIt != QueryParams.end()) FPass = PIt->second;
            auto NIt = QueryParams.find("name");
            if (NIt != QueryParams.end()) FName = NIt->second;
            auto TIt = QueryParams.find("type");
            if (TIt != QueryParams.end())
              StringRef(TIt->second).getAsInteger(10, FType);
            Res = handleGetSourceRemarks(
                Client, resolveSnapshotHTTP(Client, SnapIt->second),
                PathIt->second, FPass, FName, FType);
          }
        } else if (Path == "/api/v1/source") {
          auto PathIt = QueryParams.find("path");
          auto SnapIt = QueryParams.find("snapshot_id");
          if (PathIt == QueryParams.end() || SnapIt == QueryParams.end())
            Res = makeJSONErrorStr(400, "path and snapshot_id required");
          else
            Res = handleGetSource(
                Client, resolveSnapshotHTTP(Client, SnapIt->second),
                PathIt->second);
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
        else if (IsAPI && Segs.size() == 6 && Segs[2] == "compare" &&
                 Segs[5] == "remarks") {
          int64_t Off = 0, Lim = 100;
          auto OIt = QueryParams.find("offset");
          if (OIt != QueryParams.end()) StringRef(OIt->second).getAsInteger(10, Off);
          auto LIt = QueryParams.find("limit");
          if (LIt != QueryParams.end()) StringRef(LIt->second).getAsInteger(10, Lim);
          Res = handleGetCompareRemarks(Client,
              resolveSnapshotHTTP(Client, urlDecode(Segs[3])),
              resolveSnapshotHTTP(Client, urlDecode(Segs[4])), Off, Lim);
        }
        else if (IsAPI && Segs.size() == 7 && Segs[2] == "compare" &&
                 Segs[5] == "remarks")
          Res = handleGetCompareFunctionDetail(Client,
              resolveSnapshotHTTP(Client, urlDecode(Segs[3])),
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

      std::string Header = makeRawHTTPHeader(Res.Code, Res.ContentType,
                                             Res.Body.size(), Req.KeepAlive);
      (void)sendAll(FD, Header);
      (void)sendAll(FD, Res.Body);
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
