#include "InsightsHandler.h"

namespace llvm::advisor {

Expected<json::Object> InsightsHandler::insight(StringRef Name,
                                                const InsightInput &Input) {
  return Client.runInsight(Name, Input.SnapshotId.value_or(""), Input.UnitId,
                           Input.BaselineSnapshotId.value_or(""));
}

json::Array InsightsHandler::listAvailable(const InsightInput &Input) {
  Expected<json::Array> Result =
      Client.listInsights(Input.SnapshotId.value_or(""), Input.UnitId);
  if (!Result) {
    consumeError(Result.takeError());
    return json::Array{};
  }
  return std::move(*Result);
}

} // namespace llvm::advisor
