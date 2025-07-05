#include "Protocol/DAPTypes.h"
// #include "llvm/Support/JSON.h"

using namespace llvm;

namespace lldb_dap::protocol {

bool fromJSON(const llvm::json::Value &Params, PersistenceData &PD,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("module", PD.module) &&
         O.mapOptional("file_addr", PD.file_addr);
}

llvm::json::Value toJSON(const PersistenceData &PD) {
  json::Object result{
      {"module", PD.module},
      {"file_addr", PD.file_addr},
  };

  return result;
}

bool fromJSON(const llvm::json::Value &Params, SourceLLDBData &SLD,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("persistence_data", SLD.persistence_data);
}

llvm::json::Value toJSON(const SourceLLDBData &SLD) {
  json::Object result;
  if (SLD.persistence_data)
    result.insert({"persistence_data", SLD.persistence_data});
  return result;
}

} // namespace lldb_dap::protocol