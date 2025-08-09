#include "Protocol/DAPTypes.h"

using namespace llvm;

namespace lldb_dap::protocol {

bool fromJSON(const llvm::json::Value &Params, PersistenceData &PD,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("module_path", PD.module_path) &&
         O.mapOptional("symbol_name", PD.symbol_name);
}

llvm::json::Value toJSON(const PersistenceData &PD) {
  json::Object result{
      {"module_path", PD.module_path},
      {"symbol_name", PD.symbol_name},
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

bool fromJSON(const llvm::json::Value &Params, DAPSymbol &DS,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.map("userId", DS.userId) &&
         O.map("isDebug", DS.isDebug) &&
         O.map("isSynthesized", DS.isSynthesized) &&
         O.map("isExternal", DS.isExternal) &&
         O.map("type", DS.type) &&
         O.map("fileAddress", DS.fileAddress) &&
         O.mapOptional("loadAddress", DS.loadAddress) &&
         O.map("size", DS.size) &&
         O.map("name", DS.name);
}

llvm::json::Value toJSON(const DAPSymbol &DS) {
  json::Object result{
      {"userId", DS.userId},
      {"isDebug", DS.isDebug},
      {"isSynthesized", DS.isSynthesized},
      {"isExternal", DS.isExternal},
      {"type", DS.type},
      {"fileAddress", DS.fileAddress},
      {"loadAddress", DS.loadAddress},
      {"size", DS.size},
      {"name", DS.name},
  };

  return result;
}

} // namespace lldb_dap::protocol