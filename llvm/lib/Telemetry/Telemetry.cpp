#include "llvm/Telemetry/Telemetry.h"

namespace llvm {
namespace telemetry {

void TelemetryInfo::serialize(Serializer &serializer) const {
  serializer.write("SessionId", SessionId);
}

Error Manager::dispatch(TelemetryInfo *Entry) {
  if (Error Err = preDispatch(Entry))
    return std::move(Err);

  Error AllErrs = Error::success();
  for (auto &Dest : Destinations) {
    if (Error Err = Dest->receiveEntry(Entry))
      AllErrs = joinErrors(std::move(AllErrs), std::move(Err));
  }
  return AllErrs;
}

void Manager::addDestination(std::unique_ptr<Destination> Dest) {
  Destinations.push_back(std::move(Dest));
}

} // namespace telemetry
} // namespace llvm
