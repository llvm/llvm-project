#include "AdvisorConfig.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace advisor {

AdvisorConfig::AdvisorConfig() {
  // Use relative path as default, will be resolved by CompilationManager
  OutputDir_ = ".llvm-advisor";
}

Expected<bool> AdvisorConfig::loadFromFile(const std::string &path) {
  auto BufferOrError = MemoryBuffer::getFile(path);
  if (!BufferOrError) {
    return createStringError(BufferOrError.getError(),
                             "Cannot read config file");
  }

  auto Buffer = std::move(*BufferOrError);
  Expected<json::Value> JsonOrError = json::parse(Buffer->getBuffer());
  if (!JsonOrError) {
    return JsonOrError.takeError();
  }

  auto &Json = *JsonOrError;
  auto *Obj = Json.getAsObject();
  if (!Obj) {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Config file must contain JSON object");
  }

  if (auto outputDirOpt = Obj->getString("outputDir"); outputDirOpt) {
    OutputDir_ = outputDirOpt->str();
  }

  if (auto verboseOpt = Obj->getBoolean("verbose"); verboseOpt) {
    Verbose_ = *verboseOpt;
  }

  if (auto keepTempsOpt = Obj->getBoolean("keepTemps"); keepTempsOpt) {
    KeepTemps_ = *keepTempsOpt;
  }

  if (auto runProfileOpt = Obj->getBoolean("runProfiler"); runProfileOpt) {
    RunProfiler_ = *runProfileOpt;
  }

  if (auto timeoutOpt = Obj->getInteger("timeout"); timeoutOpt) {
    TimeoutSeconds_ = static_cast<int>(*timeoutOpt);
  }

  return true;
}

std::string AdvisorConfig::getToolPath(const std::string &tool) const {
  // For now, just return the tool name and rely on PATH
  return tool;
}

} // namespace advisor
} // namespace llvm
