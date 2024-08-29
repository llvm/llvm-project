#include "llvm/Telemetry/Telemetry.h"

namespace llvm {
namespace telemetry {

llvm::json::Object TelemetryInfo::serializeToJson() const {
  return json::Object{
      {"UUID", SessionUuid},
  };
};

} // namespace telemetry
} // namespace llvm
