
#include "Analysis/IR/RemarksRelationalAnalyzer.h"
#include "Analysis/RemarksAnalysisUtils.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {

json::Object buildEmptyEnvelope(StringRef RemarksPath) {
  json::Object Strings{
      {"pass", json::Array{}},
      {"name", json::Array{}},
      {"function", json::Array{}},
      {"file", json::Array{}},
  };
  json::Object Columns{
      {"pass", json::Array{}},     {"name", json::Array{}},
      {"type", json::Array{}},     {"function", json::Array{}},
      {"file", json::Array{}},     {"line", json::Array{}},
      {"column", json::Array{}},   {"hotness", json::Array{}},
  };
  return json::Object{
      {"schema_version", 1},
      {"remarks_path", RemarksPath.str()},
      {"count", static_cast<int64_t>(0)},
      {"strings", std::move(Strings)},
      {"columns", std::move(Columns)},
  };
}

} // namespace

Expected<std::unique_ptr<CapabilityResult>>
RemarksRelationalAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withRemarksFile(
      Context, CapID, UnitID,
      [&](StringRef Path) -> Expected<std::unique_ptr<CapabilityResult>> {
        return makeJSONResult(CapID, UnitID, buildEmptyEnvelope(Path));
      });
}
