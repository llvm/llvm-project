#include "llvm/Telemetry/Telemetry.h"

namespace llvm {
namespace telemetry {

llvm::json::Object TelemetryInfo::serializeToJson() const {
  return json::Object{
      {"SessionId", SessionId},
  };
};

} // namespace telemetry
} // namespace llvm
