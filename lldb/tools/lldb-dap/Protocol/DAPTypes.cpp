#include "Protocol/DAPTypes.h"
#include "lldb/lldb-enumerations.h"

using namespace llvm;

namespace lldb_dap::protocol {

static std::string SymbolTypeToString(lldb::SymbolType symbol_type) {
  switch (symbol_type) {
  case lldb::eSymbolTypeInvalid:
    return "Invalid";
  case lldb::eSymbolTypeAbsolute:
    return "Absolute";
  case lldb::eSymbolTypeCode:
    return "Code";
  case lldb::eSymbolTypeResolver:
    return "Resolver";
  case lldb::eSymbolTypeData:
    return "Data";
  case lldb::eSymbolTypeTrampoline:
    return "Trampoline";
  case lldb::eSymbolTypeRuntime:
    return "Runtime";
  case lldb::eSymbolTypeException:
    return "Exception";
  case lldb::eSymbolTypeSourceFile:
    return "SourceFile";
  case lldb::eSymbolTypeHeaderFile:
    return "HeaderFile";
  case lldb::eSymbolTypeObjectFile:
    return "ObjectFile";
  case lldb::eSymbolTypeCommonBlock:
    return "CommonBlock";
  case lldb::eSymbolTypeBlock:
    return "Block";
  case lldb::eSymbolTypeLocal:
    return "Local";
  case lldb::eSymbolTypeParam:
    return "Param";
  case lldb::eSymbolTypeVariable:
    return "Variable";
  case lldb::eSymbolTypeVariableType:
    return "VariableType";
  case lldb::eSymbolTypeLineEntry:
    return "LineEntry";
  case lldb::eSymbolTypeLineHeader:
    return "LineHeader";
  case lldb::eSymbolTypeScopeBegin:
    return "ScopeBegin";
  case lldb::eSymbolTypeScopeEnd:
    return "ScopeEnd";
  case lldb::eSymbolTypeAdditional:
    return "Additional";
  case lldb::eSymbolTypeCompiler:
    return "Compiler";
  case lldb::eSymbolTypeInstrumentation:
    return "Instrumentation";
  case lldb::eSymbolTypeUndefined:
    return "Undefined";
  case lldb::eSymbolTypeObjCClass:
    return "ObjCClass";
  case lldb::eSymbolTypeObjCMetaClass:
    return "ObjCMetaClass";
  case lldb::eSymbolTypeObjCIVar:
    return "ObjCIVar";
  case lldb::eSymbolTypeReExported:
    return "ReExported";
  }

  llvm_unreachable("unhandled symbol type.");
}

static lldb::SymbolType StringToSymbolType(const std::string &symbol_type) {
  return llvm::StringSwitch<lldb::SymbolType>(symbol_type)
      .Case("Invalid", lldb::eSymbolTypeInvalid)
      .Case("Absolute", lldb::eSymbolTypeAbsolute)
      .Case("Code", lldb::eSymbolTypeCode)
      .Case("Resolver", lldb::eSymbolTypeResolver)
      .Case("Data", lldb::eSymbolTypeData)
      .Case("Trampoline", lldb::eSymbolTypeTrampoline)
      .Case("Runtime", lldb::eSymbolTypeRuntime)
      .Case("Exception", lldb::eSymbolTypeException)
      .Case("SourceFile", lldb::eSymbolTypeSourceFile)
      .Case("HeaderFile", lldb::eSymbolTypeHeaderFile)
      .Case("ObjectFile", lldb::eSymbolTypeObjectFile)
      .Case("CommonBlock", lldb::eSymbolTypeCommonBlock)
      .Case("Block", lldb::eSymbolTypeBlock)
      .Case("Local", lldb::eSymbolTypeLocal)
      .Case("Param", lldb::eSymbolTypeParam)
      .Case("Variable", lldb::eSymbolTypeVariable)
      .Case("VariableType", lldb::eSymbolTypeVariableType)
      .Case("LineEntry", lldb::eSymbolTypeLineEntry)
      .Case("LineHeader", lldb::eSymbolTypeLineHeader)
      .Case("ScopeBegin", lldb::eSymbolTypeScopeBegin)
      .Case("ScopeEnd", lldb::eSymbolTypeScopeEnd)
      .Case("Additional", lldb::eSymbolTypeAdditional)
      .Case("Compiler", lldb::eSymbolTypeCompiler)
      .Case("Instrumentation", lldb::eSymbolTypeInstrumentation)
      .Case("Undefined", lldb::eSymbolTypeUndefined)
      .Case("ObjCClass", lldb::eSymbolTypeObjCClass)
      .Case("ObjCMetaClass", lldb::eSymbolTypeObjCMetaClass)
      .Case("ObjCIVar", lldb::eSymbolTypeObjCIVar)
      .Case("ReExported", lldb::eSymbolTypeReExported)

      .Default(lldb::eSymbolTypeInvalid);
}

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
  if (!(O && O.map("userId", DS.userId) && O.map("isDebug", DS.isDebug) &&
        O.map("isSynthetic", DS.isSynthetic) &&
        O.map("isExternal", DS.isExternal) && O.map("type", type_str) &&
        O.map("fileAddress", DS.fileAddress) &&
        O.mapOptional("loadAddress", DS.loadAddress) &&
        O.map("size", DS.size) && O.map("name", DS.name)))
    return false;

  DS.type = StringToSymbolType(type_str);
  return true;
}

llvm::json::Value toJSON(const Symbol &DS) {
  json::Object result{
      {"userId", DS.userId},
      {"isDebug", DS.isDebug},
      {"isSynthetic", DS.isSynthetic},
      {"isExternal", DS.isExternal},
      {"type", SymbolTypeToString(DS.type)},
      {"fileAddress", DS.fileAddress},
      {"loadAddress", DS.loadAddress},
      {"size", DS.size},
      {"name", DS.name},
  };

  return result;
}

} // namespace lldb_dap::protocol