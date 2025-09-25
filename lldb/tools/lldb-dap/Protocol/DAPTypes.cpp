#include "Protocol/DAPTypes.h"
#include "lldb/API/SBSymbol.h"
#include "lldb/lldb-enumerations.h"

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

bool fromJSON(const llvm::json::Value &Params, Symbol &DS, llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  std::string type_str;
  if (!(O && O.map("id", DS.id) && O.map("isDebug", DS.isDebug) &&
        O.map("isSynthetic", DS.isSynthetic) &&
        O.map("isExternal", DS.isExternal) && O.map("type", type_str) &&
        O.map("fileAddress", DS.fileAddress) &&
        O.mapOptional("loadAddress", DS.loadAddress) &&
        O.map("size", DS.size) && O.map("name", DS.name)))
    return false;

  DS.type = lldb::SBSymbol::GetTypeFromString(type_str.c_str());
  return true;
}

llvm::json::Value toJSON(const Symbol &DS) {
  json::Object result{
      {"id", DS.id},
      {"isDebug", DS.isDebug},
      {"isSynthetic", DS.isSynthetic},
      {"isExternal", DS.isExternal},
      {"type", lldb::SBSymbol::GetTypeAsString(DS.type)},
      {"fileAddress", DS.fileAddress},
      {"loadAddress", DS.loadAddress},
      {"size", DS.size},
      {"name", DS.name},
  };

  return result;
}

} // namespace lldb_dap::protocol
