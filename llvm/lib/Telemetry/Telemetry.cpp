#include "llvm/Telemetry/Telemetry.h"

namespace llvm {
namespace telemetry {

std::string TelemetryEventStats::ToString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "start_timestamp: " << m_start.time_since_epoch().count()
     << ", end_timestamp: "
     << (m_end.has_value() ? std::to_string(m_end->time_since_epoch().count())
                           : "<NONE>");
  return result;
}

std::string ExitDescription::ToString() const {
  return "exit_code: " + std::to_string(exit_code) +
         ", description: " + description + "\n";
}

std::string BaseTelemetryEntry::ToString() const {
  return "[BaseTelemetryEntry]\n" + ("  session_uuid:" + session_uuid + "\n") +
         ("  stats: " + stats.ToString() + "\n") +
         ("  exit_description: " +
          (exit_description.has_value() ? exit_description->ToString()
                                        : "<NONE>") +
          "\n") +
         ("  counter: " + std::to_string(counter) + "\n");
}

} // namespace telemetry
} // namespace llvm