
#include "Analysis/IR/RemarksRelationalAnalyzer.h"
#include "Analysis/IR/RemarksRelationalSchema.h"
#include "Analysis/RemarksAnalysisUtils.h"
#include "llvm/Demangle/Demangle.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {

class RelationalBuilder {
public:
  void visit(const remarks::Remark &R) {
    PassCol.push_back(static_cast<int64_t>(Pass.getOrAdd(R.PassName)));
    NameCol.push_back(static_cast<int64_t>(Name.getOrAdd(R.RemarkName)));
    TypeCol.push_back(static_cast<int64_t>(R.RemarkType));

    if (R.FunctionName.empty()) {
      FunctionCol.push_back(-1);
    } else {
      std::string Demangled = demangle(R.FunctionName);
      FunctionCol.push_back(
          static_cast<int64_t>(Function.getOrAdd(Demangled)));
    }

    if (R.Loc) {
      FileCol.push_back(
          static_cast<int64_t>(File.getOrAdd(R.Loc->SourceFilePath)));
      LineCol.push_back(static_cast<int64_t>(R.Loc->SourceLine));
      ColumnCol.push_back(static_cast<int64_t>(R.Loc->SourceColumn));
    } else {
      FileCol.push_back(-1);
      LineCol.push_back(-1);
      ColumnCol.push_back(-1);
    }

    HotnessCol.push_back(R.Hotness ? static_cast<int64_t>(*R.Hotness) : -1);
  }

  json::Object render(StringRef RemarksPath) {
    return json::Object{
        {"schema_version", 1},
        {"remarks_path", RemarksPath.str()},
        {"count", static_cast<int64_t>(PassCol.size())},
        {"strings", json::Object{
                        {"pass", Pass.toJSON()},
                        {"name", Name.toJSON()},
                        {"function", Function.toJSON()},
                        {"file", File.toJSON()},
                    }},
        {"columns", json::Object{
                        {"pass", toArray(PassCol)},
                        {"name", toArray(NameCol)},
                        {"type", toArray(TypeCol)},
                        {"function", toArray(FunctionCol)},
                        {"file", toArray(FileCol)},
                        {"line", toArray(LineCol)},
                        {"column", toArray(ColumnCol)},
                        {"hotness", toArray(HotnessCol)},
                    }},
    };
  }

private:
  static json::Array toArray(ArrayRef<int64_t> Vs) {
    json::Array Out;
    Out.reserve(Vs.size());
    for (int64_t V : Vs)
      Out.push_back(V);
    return Out;
  }

  RelationalStringTable Pass, Name, Function, File;
  std::vector<int64_t> PassCol, NameCol, TypeCol, FunctionCol, FileCol, LineCol,
      ColumnCol, HotnessCol;
};

} // namespace

Expected<std::unique_ptr<CapabilityResult>>
RemarksRelationalAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withRemarksFile(
      Context, CapID, UnitID,
      [&](StringRef Path) -> Expected<std::unique_ptr<CapabilityResult>> {
        RelationalBuilder Builder;
        if (Error E = foreachRemark(
                Path, [&](const remarks::Remark &R) -> Error {
                  Builder.visit(R);
                  return Error::success();
                }))
          return std::move(E);
        return makeJSONResult(CapID, UnitID, Builder.render(Path));
      });
}
