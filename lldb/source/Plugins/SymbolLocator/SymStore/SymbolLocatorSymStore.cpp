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
#include "llvm/HTTP/HTTPClient.h"
#include "llvm/HTTP/StreamedHTTPResponseHandler.h"
#include "llvm/Support/Caching.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

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

  std::string GetCachePath() const {
    OptionValueString *s =
        m_collection_sp->GetPropertyAtIndexAsOptionValueString(
            ePropertyCachePath);
    if (s && !s->GetCurrentValueAsRef().empty())
      return s->GetCurrentValue();
    return SymbolLocatorSymStore::GetSystemDefaultCachePath();
  }

  std::optional<std::string> GetTLSCertFingerprint() const {
    OptionValueString *s =
        m_collection_sp->GetPropertyAtIndexAsOptionValueString(
            ePropertyTLSCertFingerprint);
    if (!s)
      return {};
    llvm::StringRef val = s->GetCurrentValueAsRef();
    if (val.empty())
      return {};
    if (val.size() != 64 || !llvm::all_of(val, llvm::isHexDigit)) {
      Debugger::ReportWarning(llvm::formatv(
          "plugin.symbol-locator.symstore.tls-cert-fingerprint: expected a "
          "64-character hex string (SHA-256), but got '{0}', ignoring",
          val));
      return {};
    }
    return val.lower();
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

  std::string default_cache = GetSystemDefaultCachePath();
  if (std::error_code ec = llvm::sys::fs::create_directories(default_cache)) {
    Debugger::ReportWarning(llvm::formatv(
        "default SymStore cache directory '{0}' is not accessible: {1}",
        default_cache, ec.message()));
  }
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

SymbolLocatorSymStore::LookupEntry MakeLookupEntry(llvm::StringRef source) {
  SymbolLocatorSymStore::LookupEntry entry;
  entry.source = source.str();
  entry.cache = std::nullopt;
  return entry;
}

SymbolLocatorSymStore::LookupEntry MakeLookupEntry(llvm::StringRef source,
                                                   llvm::StringRef cache) {
  SymbolLocatorSymStore::LookupEntry entry;
  entry.source = source.str();
  entry.cache = cache.str();
  return entry;
}

std::vector<SymbolLocatorSymStore::LookupEntry> GetGlobalLookupOrder() {
  std::vector<SymbolLocatorSymStore::LookupEntry> result;

  const char *sym_path = std::getenv("_NT_SYMBOL_PATH");
  for (auto entry : SymbolLocatorSymStore::ParseEnvSymbolPaths(sym_path))
    result.push_back(std::move(entry));

  const char *alt_path = std::getenv("_NT_ALT_SYMBOL_PATH");
  for (auto entry : SymbolLocatorSymStore::ParseEnvSymbolPaths(alt_path))
    result.push_back(std::move(entry));

  for (const auto &url : GetGlobalPluginProperties().GetURLs())
    result.push_back(MakeLookupEntry(url.ref()));

  return result;
}

std::optional<SymbolLocatorSymStore::LookupEntry>
ParseSrvEntry(llvm::StringRef entry) {
  llvm::SmallVector<llvm::StringRef, 4> parts;
  entry.trim().split(parts, '*');

  // Format is: srv*[LocalCache*]SymbolStore
  switch (parts.size()) {
  case 2:
    return MakeLookupEntry(parts[1]);
  case 3: {
    // Fall back to the configured default cache for empty values.
    if (parts[1].empty())
      return MakeLookupEntry(parts[2],
                             GetGlobalPluginProperties().GetCachePath());
    return MakeLookupEntry(parts[2], parts[1]);
  }
  default:
    return {}; // Ignore entries with invalid number of parts.
  }
}

std::optional<std::string> ParseCacheEntry(llvm::StringRef entry) {
  llvm::SmallVector<llvm::StringRef, 2> parts;
  entry.trim().split(parts, '*');

  // Ignore entries with invalid number of parts.
  if (parts.size() > 2)
    return {};

  // Empty cache* deliberatly specifies the default cache path.
  llvm::StringRef value;
  if (parts.size() == 2)
    value = parts.back();

  // Fall back to LLDB's default cache for empty values.
  if (value.empty())
    return GetGlobalPluginProperties().GetCachePath();

  return value.str();
}

// RSDS entries store identity as a 20-byte UUID composed of 16-byte GUID and
// 4-byte age:
//   12345678-1234-5678-9ABC-DEF012345678-00000001
//
// SymStore key is a string with no separators and age as decimal:
//   12345678123456789ABCDEF0123456781
//
std::string FormatSymStoreKey(const UUID &uuid) {
  llvm::ArrayRef<uint8_t> bytes = uuid.GetBytes();
  uint32_t age = llvm::support::endian::read32be(bytes.data() + 16);
  constexpr bool lower_case = false;
  return llvm::toHex(bytes.slice(0, 16), lower_case) + std::to_string(age);
}

bool HasUnsafeCharacters(llvm::StringRef s) {
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

std::optional<FileSpec>
RequestFileFromSymStoreServerHTTP(llvm::StringRef base_url, llvm::StringRef key,
                                  llvm::StringRef pdb_name) {
  using namespace llvm::sys;

  // Make sure URL will be valid, portable, and compatible with symbol servers.
  if (HasUnsafeCharacters(pdb_name)) {
    Debugger::ReportWarning(llvm::formatv(
        "rejecting HTTP lookup for PDB file due to unsafe characters in "
        "name: {0}",
        pdb_name));
    return {};
  }

  // Download into a temporary file.
  llvm::SmallString<128> tmp_file;
  constexpr bool erase_on_reboot = true;
  path::system_temp_directory(erase_on_reboot, tmp_file);
  path::append(tmp_file, llvm::formatv("lldb_symstore_{0}_{1}", key, pdb_name));

  // Server has SymStore directory structure with forward slashes as separators.
  std::string source_url =
      llvm::formatv("{0}/{1}/{2}/{1}", base_url, pdb_name, key);

  if (!llvm::HTTPClient::isAvailable()) {
    Debugger::ReportWarning(
        "HTTP client is not available for SymStore download");
    return {};
  }

  llvm::HTTPClient client;
  // TODO: Since PDBs can be huge, we should distinguish between resolve,
  // connect, send and receive.
  client.setTimeout(std::chrono::seconds(60));

  llvm::StreamedHTTPResponseHandler Handler(
      [dest = tmp_file.str().str()]()
          -> llvm::Expected<std::unique_ptr<llvm::CachedFileStream>> {
        std::error_code ec;
        auto os = std::make_unique<llvm::raw_fd_ostream>(dest, ec);
        if (ec)
          return llvm::createStringError(ec, "Failed to open file for writing");
        return std::make_unique<llvm::CachedFileStream>(std::move(os), dest);
      },
      client);

  llvm::HTTPRequest request(source_url);
  request.PinnedCertFingerprint =
      GetGlobalPluginProperties().GetTLSCertFingerprint();
  if (llvm::Error Err = client.perform(request, Handler)) {
    Debugger::ReportWarning(
        llvm::formatv("failed to download from SymStore '{0}': {1}", source_url,
                      llvm::toString(std::move(Err))));
    return {};
  }
  if (llvm::Error Err = Handler.commit()) {
    Debugger::ReportWarning(
        llvm::formatv("failed to download from SymStore '{0}': {1}", source_url,
                      llvm::toString(std::move(Err))));
    return {};
  }

  unsigned responseCode = client.responseCode();
  switch (responseCode) {
  case 404:
    return {}; // file not found
  case 200:
    return FileSpec(tmp_file.str()); // success
  default:
    Debugger::ReportWarning(llvm::formatv(
        "failed to download from SymStore '{0}': response code {1}", source_url,
        responseCode));
    return {};
  }
}

std::optional<FileSpec> FindFileInLocalSymStore(llvm::StringRef root_dir,
                                                llvm::StringRef key,
                                                llvm::StringRef pdb_name) {
  llvm::SmallString<256> path;
  llvm::sys::path::append(path, root_dir, pdb_name, key, pdb_name);
  FileSpec spec(path);
  if (!FileSystem::Instance().Exists(spec))
    return {};

  return spec;
}

std::optional<FileSpec> MoveToLocalSymStore(llvm::StringRef cache,
                                            llvm::StringRef key,
                                            llvm::StringRef pdb_name,
                                            FileSpec tmp_file) {
  // Caches have SymStore directory structure: cache/pdb_name/key/pdb_name
  llvm::SmallString<256> dest_dir;
  llvm::sys::path::append(dest_dir, cache, pdb_name, key);
  if (std::error_code ec = llvm::sys::fs::create_directories(dest_dir)) {
    Debugger::ReportWarning(
        llvm::formatv("failed to create SymStore cache directory '{0}': {1}",
                      dest_dir, ec.message()));
    return {};
  }

  llvm::SmallString<256> dest;
  llvm::sys::path::append(dest, dest_dir, pdb_name);
  std::error_code ec = llvm::sys::fs::rename(tmp_file.GetPath(), dest);

  // Fall back to copy+delete if we move to a different volume.
  if (ec == std::errc::cross_device_link) {
    ec = llvm::sys::fs::copy_file(tmp_file.GetPath(), dest);
    if (!ec)
      llvm::sys::fs::remove(tmp_file.GetPath());
  }
  if (ec) {
    Debugger::ReportWarning(
        llvm::formatv("failed to move '{0}' to SymStore cache '{1}': {2}",
                      tmp_file.GetPath(), dest, ec.message()));
    return {};
  }

  return FileSpec(dest.str());
}

std::string SelectSymStoreCache(std::optional<std::string> sympath_cache) {
  llvm::SmallVector<std::string, 2> candidates;

  // Prefer user cache from symbol path.
  if (sympath_cache) {
    assert(!sympath_cache->empty() && "Empty entries resolve to default cache");
    candidates.push_back(*sympath_cache);
  }

  // Fallback to configured cache from settings.
  candidates.push_back(GetGlobalPluginProperties().GetCachePath());

  Log *log = GetLog(LLDBLog::Symbols);
  for (const auto &path : candidates) {
    if (llvm::sys::fs::is_directory(path))
      return path;
    if (std::error_code ec = llvm::sys::fs::create_directories(path)) {
      LLDB_LOG(log, "Ignoring invalid SymStore cache directory '{0}': {1}",
               path, ec.message());
      continue;
    }
    return path;
  }

  // Last resort is the system default location.
  return SymbolLocatorSymStore::GetSystemDefaultCachePath();
}

std::optional<FileSpec>
LocateSymStoreEntry(const SymbolLocatorSymStore::LookupEntry &entry,
                    llvm::StringRef key, llvm::StringRef pdb_name) {
  Log *log = GetLog(LLDBLog::Symbols);
  llvm::StringRef url = entry.source;
  if (url.starts_with("http://") || url.starts_with("https://")) {
    // Check cache first.
    std::string cache_path = SelectSymStoreCache(entry.cache);
    if (auto spec = FindFileInLocalSymStore(cache_path, key, pdb_name)) {
      LLDB_LOG(log, "Found {0} in SymStore cache {1}", pdb_name, cache_path);
      return *spec;
    }

    // Download and move to cache.
    if (auto tmp_file = RequestFileFromSymStoreServerHTTP(url, key, pdb_name)) {
      LLDB_LOG(log, "Downloaded {0} from SymStore {1}", pdb_name, url);
      auto spec = MoveToLocalSymStore(cache_path, key, pdb_name, *tmp_file);
      if (!spec) {
        // Try the fallback and eventually rather cancel than loading the tmp
        // file, since it might disappear or get overwritten.
        cache_path = SymbolLocatorSymStore::GetSystemDefaultCachePath();
        spec = MoveToLocalSymStore(cache_path, key, pdb_name, *tmp_file);
        if (!spec)
          return {};
      }
      LLDB_LOG(log, "Added {0} to SymStore cache {1}", pdb_name, cache_path);
      return *spec;
    }

    return {};
  }

  llvm::StringRef file = entry.source;
  if (file.starts_with("file://"))
    file = file.drop_front(7);
  if (auto spec = FindFileInLocalSymStore(file, key, pdb_name)) {
    LLDB_LOG(log, "Found {0} in local SymStore {1}", pdb_name, file);
    return *spec;
  }

  return {};
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
    LLDB_LOG(log, "Failed to resolve symbol PDB module: PDB name empty");
    return {};
  }

  LLDB_LOG(log, "LocateExecutableSymbolFile {0} with UUID {1}", pdb_name,
           uuid.GetAsString());
  if (uuid.GetBytes().size() != 20) {
    LLDB_LOG(log, "Failed to resolve symbol PDB module: UUID invalid");
    return {};
  }

  std::string key = FormatSymStoreKey(uuid);
  for (const LookupEntry &entry : GetGlobalLookupOrder()) {
    if (auto spec = LocateSymStoreEntry(entry, key, pdb_name))
      return *spec;
  }

  return {};
}

std::vector<SymbolLocatorSymStore::LookupEntry>
SymbolLocatorSymStore::ParseEnvSymbolPaths(llvm::StringRef val) {
  if (val.empty())
    return {};

  std::vector<LookupEntry> result;
  std::optional<std::string> implicit_cache;
  llvm::SmallVector<llvm::StringRef, 2> entries;
  val.split(entries, ';');

  for (llvm::StringRef raw : entries) {
    llvm::StringRef entry = raw.trim();
    if (entry.empty())
      continue;

    // Explicit cache directives apply to all subsequent srv* entries that don't
    // set their own explicit cache.
    if (entry.starts_with_insensitive("cache*")) {
      if (auto cache = ParseCacheEntry(entry))
        implicit_cache = *cache;
      continue;
    }

    // SymStore directives with explicit interpreters are unsupported
    // explicitly.
    if (entry.starts_with_insensitive("symsrv*")) {
      Debugger::ReportWarning(
          llvm::formatv("ignoring unsupported entry in env: {0}", entry));
      continue;
    }

    // SymStore server directives may include an explicit cache.
    // Format is: srv*[LocalCache*]SymbolStore
    if (entry.starts_with_insensitive("srv*")) {
      if (auto lookup_entry = ParseSrvEntry(entry)) {
        if (!lookup_entry->cache && implicit_cache)
          lookup_entry->cache = implicit_cache;
        result.push_back(*lookup_entry);
      }
      continue;
    }

    // Plain local paths aren't cached.
    result.push_back(MakeLookupEntry(entry));
  }

  return result;
}

std::string SymbolLocatorSymStore::GetSystemDefaultCachePath() {
  // Fall back to the platform cache directory.
  llvm::SmallString<128> cache_dir;
  if (llvm::sys::path::cache_directory(cache_dir)) {
    llvm::sys::path::append(cache_dir, "lldb", "symstore");
    return cache_dir.str().str();
  }
  // Last resort: use a subdirectory of the system temp directory.
  constexpr bool erase_on_reboot = false;
  llvm::sys::path::system_temp_directory(erase_on_reboot, cache_dir);
  llvm::sys::path::append(cache_dir, "lldb", "symstore");
  return cache_dir.str().str();
}
