#include "llvm/Telemetry/Telemetry.h"

namespace llvm {
namespace telemetry {

void TelemetryInfo::serialize(Serializer &serializer) const {
  serializer.writeString("SessionId", SessionId);
}

} // namespace telemetry
} // namespace llvm
