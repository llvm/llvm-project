#include "llvm/Telemetry/Telemetry.h"

namespace llvm {
namespace telemetry {

void TelemetryInfo::serialize(Serializer &serializer) const {
  serializer.write("SessionId", SessionId);
}

Error Manager::dispatch(TelemetryInfo *Entry) {
  if (Error Err = preDispatch(Entry))
    return Err;

  Error AllErrs = Error::success();
  for (auto &Dest : Destinations) {
    AllErrs = joinErrors(std::move(AllErrs), Dest->receiveEntry(Entry));
  }
  return AllErrs;
}

void Manager::addDestination(std::unique_ptr<Destination> Dest) {
  Destinations.push_back(std::move(Dest));
}

} // namespace telemetry
} // namespace llvm
