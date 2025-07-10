#include "Protocol/DAPTypes.h"
// #include "llvm/Support/JSON.h"

using namespace llvm;

namespace lldb_dap::protocol {

bool fromJSON(const llvm::json::Value &Params, PersistenceData &PD,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("module", PD.module) &&
         O.mapOptional("fileAddress", PD.fileAddress);
}

llvm::json::Value toJSON(const PersistenceData &PD) {
  json::Object result{
      {"module", PD.module},
      {"fileAddress", PD.fileAddress},
  };

  return result;
}

bool fromJSON(const llvm::json::Value &Params, SourceLLDBData &SLD,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("persistenceData", SLD.persistenceData);
}

llvm::json::Value toJSON(const SourceLLDBData &SLD) {
  json::Object result;
  if (SLD.persistenceData)
    result.insert({"persistenceData", SLD.persistenceData});
  return result;
}

} // namespace lldb_dap::protocol