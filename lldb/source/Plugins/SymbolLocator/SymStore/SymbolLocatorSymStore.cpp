//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolLocatorSymStore.h"

#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/UUID.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/HTTP/HTTPClient.h"
#include "llvm/Support/Path.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SymbolLocatorSymStore)

namespace {

#define LLDB_PROPERTIES_symbollocatorsymstore
#include "SymbolLocatorSymStoreProperties.inc"

enum {
#define LLDB_PROPERTIES_symbollocatorsymstore
#include "SymbolLocatorSymStorePropertiesEnum.inc"
};

class PluginProperties : public Properties {
public:
  static llvm::StringRef GetSettingName() {
    return SymbolLocatorSymStore::GetPluginNameStatic();
  }

  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(GetSettingName());
    m_collection_sp->Initialize(g_symbollocatorsymstore_properties_def);
  }

  Args GetURLs() const {
    Args urls;
    m_collection_sp->GetPropertyAtIndexAsArgs(ePropertySymStoreURLs, urls);
    return urls;
  }
};

} // namespace

static PluginProperties &GetGlobalPluginProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

SymbolLocatorSymStore::SymbolLocatorSymStore() : SymbolLocator() {}

void SymbolLocatorSymStore::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
      nullptr, LocateExecutableSymbolFile, nullptr, nullptr,
      SymbolLocatorSymStore::DebuggerInitialize);
  llvm::HTTPClient::initialize();
}

void SymbolLocatorSymStore::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForSymbolLocatorPlugin(
          debugger, PluginProperties::GetSettingName())) {
    constexpr bool is_global_setting = true;
    PluginManager::CreateSettingForSymbolLocatorPlugin(
        debugger, GetGlobalPluginProperties().GetValueProperties(),
        "Properties for the SymStore Symbol Locator plug-in.",
        is_global_setting);
  }
}

void SymbolLocatorSymStore::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
  llvm::HTTPClient::cleanup();
}

llvm::StringRef SymbolLocatorSymStore::GetPluginDescriptionStatic() {
  return "Symbol locator for PDB in SymStore";
}

SymbolLocator *SymbolLocatorSymStore::CreateInstance() {
  return new SymbolLocatorSymStore();
}

namespace {

// RSDS entries store identity as a 20-byte UUID composed of 16-byte GUID and
// 4-byte age:
//   12345678-1234-5678-9ABC-DEF012345678-00000001
//
// SymStore key is a string with no separators and age as decimal:
//   12345678123456789ABCDEF0123456781
//
std::string formatSymStoreKey(const UUID &uuid) {
  llvm::ArrayRef<uint8_t> bytes = uuid.GetBytes();
  uint32_t age = llvm::support::endian::read32be(bytes.data() + 16);
  constexpr bool LowerCase = false;
  return llvm::toHex(bytes.slice(0, 16), LowerCase) + std::to_string(age);
}

// This is a simple version of Debuginfod's StreamedHTTPResponseHandler. We
// should consider reusing that once we introduce caching.
class FileDownloadHandler : public llvm::HTTPResponseHandler {
private:
  std::error_code m_ec;
  llvm::raw_fd_ostream m_stream;

public:
  FileDownloadHandler(llvm::StringRef file) : m_stream(file.str(), m_ec) {}
  virtual ~FileDownloadHandler() = default;

  llvm::Error handleBodyChunk(llvm::StringRef data) override {
    // Propagate error from ctor.
    if (m_ec)
      return llvm::createStringError(m_ec, "Failed to open file for writing");
    m_stream.write(data.data(), data.size());
    if (std::error_code ec = m_stream.error())
      return llvm::createStringError(ec, "Error writing to file");

    return llvm::Error::success();
  }
};

llvm::Error downloadFileHTTP(llvm::StringRef url, FileDownloadHandler dest) {
  if (!llvm::HTTPClient::isAvailable())
    return llvm::createStringError(
        std::make_error_code(std::errc::not_supported),
        "HTTP client is not available");
  llvm::HTTPRequest Request(url);
  Request.FollowRedirects = true;

  llvm::HTTPClient Client;

  // TODO: Since PDBs can be huge, we should distinguish between resolve,
  // connect, send and receive.
  Client.setTimeout(std::chrono::seconds(60));

  if (llvm::Error Err = Client.perform(Request, dest))
    return Err;

  unsigned ResponseCode = Client.responseCode();
  if (ResponseCode != 200) {
    return llvm::createStringError(std::make_error_code(std::errc::io_error),
                                   "HTTP request failed with status code " +
                                       std::to_string(ResponseCode));
  }

  return llvm::Error::success();
}

bool has_unsafe_characters(llvm::StringRef s) {
  for (unsigned char c : s) {
    // RFC 3986 unreserved characters are safe for file names and URLs.
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
        (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_' ||
        c == '~') {
      continue;
    }

    return true;
  }

  // Avoid path semantics issues.
  return s == "." || s == "..";
}

// TODO: This is a dump initial implementation: It always downloads the file, it
// doesn't validate the result, it doesn't employ proper buffering for large
// files.
std::optional<FileSpec>
requestFileFromSymStoreServerHTTP(llvm::StringRef base_url, llvm::StringRef key,
                                  llvm::StringRef pdb_name) {
  using namespace llvm::sys;
  Log *log = GetLog(LLDBLog::Symbols);

  // Make sure URL will be valid, portable, and compatible with symbol servers.
  if (has_unsafe_characters(pdb_name)) {
    Debugger::ReportWarning(llvm::formatv(
        "rejecting HTTP lookup for PDB file due to unsafe characters in "
        "name: {0}",
        pdb_name));
    return {};
  }

  // Download into a temporary file. Cache coming soon.
  llvm::SmallString<128> tmp_file;
  std::string tmp_file_name =
      llvm::formatv("lldb_symstore_{0}_{1}", key, pdb_name);
  constexpr bool erase_on_reboot = true;
  path::system_temp_directory(erase_on_reboot, tmp_file);
  path::append(tmp_file, tmp_file_name);

  // Server has SymStore directory structure with forward slashes as separators.
  std::string source_url =
      llvm::formatv("{0}/{1}/{2}/{1}", base_url, pdb_name, key);
  if (llvm::Error err = downloadFileHTTP(source_url, tmp_file.str())) {
    LLDB_LOG_ERROR(log, std::move(err),
                   "Failed to download from SymStore '{1}': {0}", source_url);
    return {};
  }

  return FileSpec(tmp_file.str());
}

std::optional<FileSpec> findFileInLocalSymStore(llvm::StringRef root_dir,
                                                llvm::StringRef key,
                                                llvm::StringRef pdb_name) {
  llvm::SmallString<256> path;
  llvm::sys::path::append(path, root_dir, pdb_name, key, pdb_name);
  FileSpec spec(path);
  if (!FileSystem::Instance().Exists(spec))
    return {};

  return spec;
}

std::optional<FileSpec> locateSymStoreEntry(llvm::StringRef base_url,
                                            llvm::StringRef key,
                                            llvm::StringRef pdb_name) {
  if (base_url.starts_with("http://") || base_url.starts_with("https://"))
    return requestFileFromSymStoreServerHTTP(base_url, key, pdb_name);

  if (base_url.starts_with("file://"))
    base_url = base_url.drop_front(7);

  return findFileInLocalSymStore(base_url, key, pdb_name);
}

} // namespace

std::optional<FileSpec> SymbolLocatorSymStore::LocateExecutableSymbolFile(
    const ModuleSpec &module_spec, const FileSpecList &default_search_paths) {
  const UUID &uuid = module_spec.GetUUID();
  if (!uuid.IsValid() ||
      !ModuleList::GetGlobalModuleListProperties().GetEnableExternalLookup())
    return {};

  Log *log = GetLog(LLDBLog::Symbols);
  std::string pdb_name =
      module_spec.GetSymbolFileSpec().GetFilename().GetStringRef().str();
  if (pdb_name.empty()) {
    LLDB_LOG_VERBOSE(log,
                     "Failed to resolve symbol PDB module: PDB name empty");
    return {};
  }

  LLDB_LOG_VERBOSE(log, "LocateExecutableSymbolFile {0} with UUID {1}",
                   pdb_name, uuid.GetAsString());
  if (uuid.GetBytes().size() != 20) {
    LLDB_LOG_VERBOSE(log, "Failed to resolve symbol PDB module: UUID invalid");
    return {};
  }

  std::string key = formatSymStoreKey(uuid);
  Args sym_store_urls = GetGlobalPluginProperties().GetURLs();
  for (const Args::ArgEntry &url : sym_store_urls) {
    if (auto spec = locateSymStoreEntry(url.ref(), key, pdb_name)) {
      LLDB_LOG_VERBOSE(log, "Found {0} in SymStore {1}", pdb_name, url.ref());
      return *spec;
    }
  }

  return {};
}
