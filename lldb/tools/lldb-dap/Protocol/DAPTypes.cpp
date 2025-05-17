#include "Protocol/DAPTypes.h"
// #include "llvm/Support/JSON.h"

using namespace llvm;

namespace lldb_dap::protocol {

bool fromJSON(const llvm::json::Value &Params, AssemblyBreakpointData &ABD,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("module", ABD.module) &&
         O.mapOptional("symbol_mangled_name", ABD.symbol_mangled_name) &&
         O.mapOptional("offset", ABD.offset);
}

llvm::json::Value toJSON(const AssemblyBreakpointData &ABD) {
  json::Object result{
      {"module", ABD.module},
      {"symbol_mangled_name", ABD.symbol_mangled_name},
      {"offset", ABD.offset},
  };

  return result;
}

bool fromJSON(const llvm::json::Value &Params, SourceLLDBData &SLD,
              llvm::json::Path P) {
  json::ObjectMapper O(Params, P);
  return O && O.mapOptional("assembly_breakpoint", SLD.assembly_breakpoint);
}

llvm::json::Value toJSON(const SourceLLDBData &SLD) {
  json::Object result;
  if (SLD.assembly_breakpoint)
    result.insert({"assembly_breakpoint", SLD.assembly_breakpoint});
  return result;
}

} // namespace lldb_dap::protocol