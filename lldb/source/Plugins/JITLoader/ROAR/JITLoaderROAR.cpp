
//===-- JITLoaderROAR.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITLoaderROAR.h"

#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/BreakpointResolver.h"
#include "lldb/Breakpoint/BreakpointResolverFileLine.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <string>

using namespace lldb;

LLDB_PLUGIN_DEFINE(JITLoaderROAR)

static_assert(alignof(lldb_private::BreakpointLocation) > 1,
              "Breakpoint class is not even 2-byte aligned");
static_assert(alignof(lldb_private::BreakpointLocation) > 1,
              "Breakpoint class is not even 2-byte aligned");

// Debug Interface Structures
static constexpr char ROARDebugShimsSectionName[] = ".roar.lldb_shims";

namespace lldb_private {
enum EnableJITLoaderROAR {
  eEnableJITLoaderROARDefault,
  eEnableJITLoaderROAROn,
  eEnableJITLoaderROAROff,
};

static constexpr lldb_private::OptionEnumValueElement
    g_enable_jit_loader_roar_enumerators[] = {
        {
            eEnableJITLoaderROARDefault,
            "default",
            "Enable ROAR compilation interface",
        },
        {
            eEnableJITLoaderROAROn,
            "on",
            "Enable ROAR compilation interface",
        },
        {
            eEnableJITLoaderROAROff,
            "off",
            "Disable ROAR compilation interface",
        },
};

#define LLDB_PROPERTIES_jitloaderroar
#include "JITLoaderROARProperties.inc"

enum {
#define LLDB_PROPERTIES_jitloaderroar
#include "JITLoaderROARPropertiesEnum.inc"
};
} // namespace lldb_private

namespace {
class PluginProperties : public lldb_private::Properties {
public:
  static lldb_private::ConstString GetSettingName() {
    return lldb_private::ConstString(
        lldb_private::JITLoaderROAR::GetPluginNameStatic());
  }

  PluginProperties() {
    m_collection_sp =
        std::make_shared<lldb_private::OptionValueProperties>(GetSettingName());
    m_collection_sp->Initialize(lldb_private::g_jitloaderroar_properties);
  }

  lldb_private::EnableJITLoaderROAR GetEnable() const {
    return GetPropertyAtIndexAs<lldb_private::EnableJITLoaderROAR>(
        lldb_private::ePropertyEnable,
        static_cast<lldb_private::EnableJITLoaderROAR>(
            lldb_private::g_jitloaderroar_properties
                [lldb_private::ePropertyEnable]
                    .default_uint_value));
  }

  lldb_private::FileSpec GetROARSharedLibraryPath() {
    return GetPropertyAtIndexAs<lldb_private::FileSpec>(
        lldb_private::ePropertyROARSharedLibraryPath, {});
  }

  lldb_private::FileSpec GetROARLLDBShimsSharedLibraryPath() {
    return GetPropertyAtIndexAs<lldb_private::FileSpec>(
        lldb_private::ePropertyROARLLDBShimsSharedLibraryPath, {});
  }

  bool GetEagerSymbolGeneration() const {
    return GetPropertyAtIndexAs<bool>(
        lldb_private::ePropertyEagerSymbolGeneration,
        lldb_private::g_jitloaderroar_properties
            [lldb_private::ePropertyEagerSymbolGeneration]
                .default_uint_value);
  }

  bool GetDisableTrampolineStop() const {
    return GetPropertyAtIndexAs<bool>(
        lldb_private::ePropertyDisableTrampolineStop,
        lldb_private::g_jitloaderroar_properties
            [lldb_private::ePropertyDisableTrampolineStop]
                .default_uint_value);
  }

  bool GetTestShimsErrorCreation() const {
    return GetPropertyAtIndexAs<bool>(
        lldb_private::ePropertyTestShimsErrorCreation,
        lldb_private::g_jitloaderroar_properties
            [lldb_private::ePropertyTestShimsErrorCreation]
                .default_uint_value);
  }
};

static PluginProperties &GetGlobalPluginProperties() {
  static PluginProperties g_settings;
  return g_settings;
}
} // namespace

roar_lldb::ROARError::~ROARError() = default;
roar_lldb::ROARError::ROARError() = default;
namespace {
class JITLoaderROARError : public roar_lldb::ROARError {
public:
  JITLoaderROARError() = default;

  ~JITLoaderROARError() = default;

  /// Destroys instance of impplementation of ROARError.
  virtual void Destroy() override { delete this; }
  /// Sets error string.
  virtual void SetErrorString(const char *msg) override { buffer.append(msg); }
  /// Returns the error code.
  virtual bool Success() const override { return buffer.empty(); }
  /// Returns the error string.
  const char *GetCString() const { return buffer.c_str(); }

private:
  std::string buffer;
};

/// Used in shims to log into LLDB.
void logToLLDB(const char *msg) {
  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);
  LLDB_LOGF(log, "%s", msg);
}

} // namespace

namespace lldb_roar_private {
class PluginSync {
public:
  PluginSync(lldb_private::Process &m_process) : m_process(m_process) {
    m_process.WillCallExternalPlugin();
  }
  ~PluginSync() { m_process.DoneCallingExternalPlugin(); }

private:
  lldb_private::Process &m_process;
};
} // namespace lldb_roar_private

lldb_roar_private::JITLoaderROARSB::JITLoaderROARSB()
    : m_jit_break_id(LLDB_INVALID_BREAK_ID),
      m_jit_dynamic_symbol_arena_addr_break_id(LLDB_INVALID_BREAK_ID),
      m_reading_jit_entries(false), m_process(nullptr) {}

lldb_roar_private::JITLoaderROARSB::~JITLoaderROARSB() { Reset(); }

void lldb_roar_private::JITLoaderROARSB::Reset() {
  if (LLDB_BREAK_ID_IS_VALID(m_jit_break_id))
    m_process->GetTarget().RemoveBreakpointByID(m_jit_break_id);

  if (LLDB_BREAK_ID_IS_VALID(m_jit_dynamic_symbol_arena_addr_break_id))
    m_process->GetTarget().RemoveBreakpointByID(
        m_jit_dynamic_symbol_arena_addr_break_id);
  m_jit_break_id = LLDB_INVALID_BREAK_ID;
  m_jit_dynamic_symbol_arena_addr_break_id = LLDB_INVALID_BREAK_ID;
  m_reading_jit_entries = false;

  if (m_roar_di) {
    PluginSync sync(*m_process);
    m_roar_di->Reset();
  }
}

void lldb_roar_private::JITLoaderROARSB::Init(
    lldb_private::ModuleList &module_list) {
  if (DidSetJITBreakpoint())
    return;

  SetJITBreakpoint(module_list);
  if (!DidSetJITBreakpoint())
    return;
  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);
  lldb_private::BreakpointList &breakpoints =
      m_process->GetTarget().GetBreakpointList();

  for (size_t i = 0, num_bps = breakpoints.GetSize(); i < num_bps; ++i)
    HandleBreakpointEventImpl(log, /*add_bp_locs=*/true, /*loc_event*/ false,
                              /*locations=*/nullptr,
                              *breakpoints.GetBreakpointAtIndex(i));
}

void lldb_private::JITLoaderROAR::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForJITLoaderPlugin(
          debugger, PluginProperties::GetSettingName())) {
    const bool is_global_setting = true;
    PluginManager::CreateSettingForJITLoaderPlugin(
        debugger, GetGlobalPluginProperties().GetValueProperties(),
        ConstString("Properties for the JIT LoaderROAR plug-in."),
        is_global_setting);
  }
}

void lldb_roar_private::JITLoaderROARSB::DidAttach() {
  Reset();

  lldb_private::Target &target = m_process->GetTarget();
  lldb_private::ModuleList &module_list = target.GetImages();
  Init(module_list);
}

void lldb_roar_private::JITLoaderROARSB::DidLaunch() {
  Reset();

  lldb_private::Target &target = m_process->GetTarget();
  lldb_private::ModuleList &module_list = target.GetImages();
  Init(module_list);
}

void lldb_roar_private::JITLoaderROARSB::ModulesDidLoad(
    lldb_private::ModuleList &module_list) {
  if (m_process->IsAlive())
    Init(module_list);
}

bool lldb_roar_private::JITLoaderROARSB::ResolveLoadAddress(
    addr_t load_addr, lldb_private::Address &addr) {
  if (!m_roar_di)
    return false;
  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);
  if (m_load_address) {
    LLDB_LOGF(log,
              "JITLoaderROAR::%s recrusive call is detected for: 0x%" PRIx64,
              __FUNCTION__, load_addr);
    return false;
  }
  auto RemoveAddress = llvm::make_scope_exit([&] { m_load_address = 0; });
  m_load_address = load_addr;

  ReadJITEntries();

  JITLoaderROARError err;
  uint8_t jit_fn_loaded = 0;
  {
    PluginSync sync(*m_process);
    jit_fn_loaded = m_roar_di->GetJITFunctionDefiningAddress(load_addr, err);
  }

  if (!err.Success()) {
    LLDB_LOGF(log,
              "JITLoaderROAR::%s error looking up load address 0x%" PRIx64
              ": %s",
              __FUNCTION__, load_addr, err.GetCString());
    return false;
  } else if (!jit_fn_loaded) {
    LLDB_LOGF(log, "JITLoaderROAR::%s address 0x%" PRIx64 " not found",
              __FUNCTION__, load_addr);
    return false;
  }
  // This is needed when SBAddress is created and then
  // GetTarget().ResolveSymbolContextForAddress is called.
  // The latter uses section information to find out to which Module address
  // belongs. Without this the first time API is invoked section is nullptr, and
  // ResolveSymbolContextForAddress fails.
  m_process->GetTarget().ResolveLoadAddress(load_addr, addr);
  return true;
}

namespace {
// Traits type used for iterating over all breakpoint locations in a Breakpoint,
// or a BreakpointLocationCollection. This is necessary because some breakpoint
// events provide a list of affected locations to
// JITLoader::HandleBreakpointEvent, and some don't. Thus, sometimes the
// location list is the Breakpoint's own location list, and sometimes, it is the
// provided location list -- e.g., when locations are removed from breakpoints.
template <typename ContainerTy> struct ForEachLocationTraits;

// ForEachLocationIn is a template that takes a container type and an action
// callable that's invoked for each BreakpointLocation in the container. The
// ForEachLocationTraits traits type is used to access the container members.
template <typename ContainerTy, typename ActionTy>
void ForEachLocationIn(ContainerTy &container, ActionTy Action) {
  using Traits = ForEachLocationTraits<ContainerTy>;
  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);
  const size_t NumLocations = Traits::GetNumElements(container);
  if (NumLocations == 0) {
    LLDB_LOGF(log, "JITLoaderROAR::%s breakpoint location: no locations",
              __FUNCTION__);
    return;
  }

  for (size_t i = 0; i < NumLocations; ++i)
    Action(Traits::GetElement(container, i));
}

// ForEachLocationTraits for Breakpoints.
template <> struct ForEachLocationTraits<lldb_private::Breakpoint> {
  static size_t GetNumElements(const lldb_private::Breakpoint &bp) {
    return bp.GetNumLocations();
  }

  static BreakpointLocationSP GetElement(lldb_private::Breakpoint &bp,
                                         size_t idx) {
    return bp.GetLocationAtIndex(idx);
  }
};

// ForEachLocationTraits for BreakpointLocationCollection.
template <>
struct ForEachLocationTraits<const lldb_private::BreakpointLocationCollection> {
  static size_t
  GetNumElements(const lldb_private::BreakpointLocationCollection &locations) {
    return locations.GetSize();
  }

  static BreakpointLocationSP
  GetElement(const lldb_private::BreakpointLocationCollection &locations,
             size_t idx) {
    return locations.GetByIndex(idx);
  }
};
} // namespace

void lldb_roar_private::JITLoaderROARSB::NotifyJITToLoadDebugInformation(
    lldb_private::Symbol &symbol) {
  if (!m_roar_di)
    return;
  JITLoaderROARError err;
  lldb_private::Target &target = m_process->GetTarget();
  lldb::addr_t addr = symbol.GetLoadAddress(&target);
  bool IsFunctionTrampoline = false;
  {
    PluginSync sync(*m_process);
    IsFunctionTrampoline = m_roar_di->IsFunctionTrampoline(addr, err);
  }
  if (!IsFunctionTrampoline)
    return;
  ReadJITEntries();
  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);
  // Notify ROAR that when a function for this trampoline address gets jitted it
  // needs to notify debugger.
  {
    PluginSync sync(*m_process);
    m_roar_di->NotifyJITToLoadDebugInformation(addr, err);
  }
  if (!err.Success()) {
    LLDB_LOGF(log, "JITLoaderROAR::NotifyJITToLoadDebugInformation: %s",
              err.GetCString());
  }
}

void lldb_roar_private::JITLoaderROARSB::HandleBreakpointEvent(
    BreakpointEventType sub_type, lldb_private::Breakpoint &breakpoint,
    const lldb_private::BreakpointLocationCollection *locations) {
  if (!m_roar_di)
    return;

  ReadJITEntries();

  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);
  LLDB_LOGF(log, "JITLoaderROAR::%s breakpoint %s %s", __FUNCTION__,
            lldb_private::Breakpoint::BreakpointEventTypeAsCString(sub_type),
            breakpoint.GetBreakpointKind());
  bool loc_event = true;
  bool add_bp_locs = true;
  switch (sub_type) {
  case eBreakpointEventTypeDisabled:
  case eBreakpointEventTypeRemoved:
    loc_event = false;
    LLVM_FALLTHROUGH;
  case eBreakpointEventTypeLocationsRemoved:
    add_bp_locs = false;
    break;
  case eBreakpointEventTypeAdded:
  case eBreakpointEventTypeEnabled:
    loc_event = false;
    break;
  case eBreakpointEventTypeLocationsAdded:
  case eBreakpointEventTypeLocationsResolved:
    break;
  default:
    LLDB_LOGF(log, "JITLoaderROAR::%s breakpoint: unhandled kind",
              __FUNCTION__);
    return;
  }

  HandleBreakpointEventImpl(log, add_bp_locs, loc_event, locations, breakpoint);
}

void lldb_roar_private::JITLoaderROARSB::HandleBreakpointEventImpl(
    lldb_private::Log *log, bool add_bp_locs, bool loc_event,
    const lldb_private::BreakpointLocationCollection *locations,
    lldb_private::Breakpoint &breakpoint) {
  BreakpointResolverSP bp_resolver = breakpoint.GetResolver();
  if (!bp_resolver)
    return;

  switch (bp_resolver->getResolverID()) {
  case lldb_private::BreakpointResolver::NameResolver:
    HandleNameBreakpointEvent(log, add_bp_locs, locations, breakpoint);
    break;
  case lldb_private::BreakpointResolver::FileLineResolver:
    if (!loc_event) {
      // FileLine breakpoints only need to be handled if the whole breakpoint is
      // being modified (versus a location modification).
      HandleFileLineBreakpointEvent(
          log, add_bp_locs, breakpoint,
          *llvm::cast<lldb_private::BreakpointResolverFileLine>(
              bp_resolver.get()));
    }
    break;
  case lldb_private::BreakpointResolver::AddressResolver:
    HandleAddressBreakpointEvent(log, add_bp_locs, locations, breakpoint);
    break;
  default:
    break;
  }
}

bool lldb_roar_private::JITLoaderROARSB::JITDebugTrampolineBreakpointHit(
    void *baton, lldb_private::StoppointCallbackContext *context,
    lldb::user_id_t break_id, lldb::user_id_t break_loc_id) {
  // TODO: If this becomes an issue we can also disable a breakpoint for a
  // location.
  return false;
}

void lldb_roar_private::JITLoaderROARSB::HandleNameBreakpointEvent(
    lldb_private::Log *log, bool add_bp_locs,
    const lldb_private::BreakpointLocationCollection *locations,
    lldb_private::Breakpoint &breakpoint) {
  if (!m_roar_di)
    return;
  // When adding a breakpoint by name, trampoline locations are handled
  // specially.
  //
  // Essentially, when the user sets a breakpoint in a "trampoline", the
  // expected behavior is for a breakpoint to be set in the jitted code
  // (unless the user is debugging ROAR itself). The thing about jit code is,
  // it may not be loaded yet. Complicating things even more is the fact that
  // ROAR compiles may load different versions for each function.
  auto handle_location = [&](const BreakpointLocationSP &bp_loc) {
    lldb_private::Address addr = bp_loc->GetAddress();
    if (!addr.IsValid()) {
      LLDB_LOGF(log, "JITLoaderROAR::%s breakpoint location: invalid address",
                __FUNCTION__);
      return;
    }
    // Figure out if bp_loc's load address lies within the trampoline section.
    lldb_private::Target *target = &m_process->GetTarget();
    addr_t load_address = addr.GetLoadAddress(target);
    JITLoaderROARError err;
    bool IsFunctionTrampoline = false;
    {
      PluginSync sync(*m_process);
      IsFunctionTrampoline =
          GetGlobalPluginProperties().GetDisableTrampolineStop() &&
          m_roar_di->IsFunctionTrampoline(load_address, err);
    }
    if (IsFunctionTrampoline)
      bp_loc->SetCallback(JITDebugTrampolineBreakpointHit, this, true);
    {
      PluginSync sync(*m_process);
      m_roar_di->HandleBreakpointByAddress(
          load_address, BreakpointOrLocationPtr{bp_loc.get()}.getOpaqueValue(),
          add_bp_locs, err);
    }
    if (!err.Success()) {
      LLDB_LOGF(log, "JITLoaderROAR::GetSTEAddressFromTrampoline: %s",
                err.GetCString());
      return;
    }
  };

  // Use the optional locations collection if it was provided; otherwise, use
  // the breakpoint's location list.
  if (locations == nullptr)
    ForEachLocationIn(breakpoint, handle_location);
  else
    ForEachLocationIn(*locations, handle_location);
}

void lldb_roar_private::JITLoaderROARSB::HandleAddressBreakpointEvent(
    lldb_private::Log *log, bool add_bp_locs,
    const lldb_private::BreakpointLocationCollection *locations,
    lldb_private::Breakpoint &breakpoint) {
  HandleNameBreakpointEvent(log, add_bp_locs, locations, breakpoint);
  return;
}

void lldb_roar_private::JITLoaderROARSB::HandleFileLineBreakpointEvent(
    lldb_private::Log *log, bool add_bp, lldb_private::Breakpoint &breakpoint,
    const lldb_private::BreakpointResolverFileLine &resolver) {
  const lldb_private::SourceLocationSpec &src_loc_spec =
      resolver.GetLocationSpec();

  LLDB_LOGF(log, "JITLoaderROAR::%s %s breakpoint", __FUNCTION__,
            (add_bp ? "adding" : "removing"));

  lldb_private::FileSpec fs = src_loc_spec.GetFileSpec();
  std::optional<uint32_t> Line = src_loc_spec.GetLine();
  if (!Line) {
    LLDB_LOGF(log, "JITLoaderROAR::%s source=%s line=???", __FUNCTION__,
              fs.GetFilename().GetCString());
    return;
  }

  JITLoaderROARError err;
  {
    PluginSync sync(*m_process);
    m_roar_di->HandleBreakpointBySourceLocation(
        fs.GetPath().c_str(), *Line,
        BreakpointOrLocationPtr{&breakpoint}.getOpaqueValue(), add_bp, err);
  }

  if (!err.Success()) {
    LLDB_LOGF(log,
              "JITLoaderROAR::%s failed to find STEs for %s:%" PRIu32 ": %s",
              __FUNCTION__, fs.GetPath().c_str(), *Line, err.GetCString());
    return;
  }
}

// Setup the JIT Breakpoint
namespace {
bool SaveRoarShimsToDisk(lldb_private::Log *log,
                         const SectionSP &roar_shims_section_sp,
                         lldb_private::FileSpec &roar_shims_so_name) {
  lldb_private::DataExtractor data_extractor;
  if (!roar_shims_section_sp->GetSectionData(data_extractor))
    return false;
  llvm::ArrayRef<uint8_t> section_data = data_extractor.GetData();

  int RoarShimsSoNameFD;
  llvm::SmallString<0> RoarShimsSoUniqueNameBuffer;
  static constexpr char RoarShimsSoName[] = "libroar-lldb-shims";
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          RoarShimsSoName, "so", RoarShimsSoNameFD,
          RoarShimsSoUniqueNameBuffer)) {
    LLDB_LOGF(log, "[JITLoaderROAR]::%s failed to create temporary file %s: %s",
              __FUNCTION__, RoarShimsSoName, ec.message().c_str());
    return false;
  }
  RoarShimsSoUniqueNameBuffer.push_back('\0');

  llvm::raw_fd_ostream RoarShimsSoOut(RoarShimsSoNameFD, /*shouldClose*/ true,
                                      /*unbuffered*/ true);
  RoarShimsSoOut.write(reinterpret_cast<const char *>(section_data.data()),
                       section_data.size());
  roar_shims_so_name =
      lldb_private::FileSpec(RoarShimsSoUniqueNameBuffer.data());
  LLDB_LOGF(log, "[JITLoaderROAR]::%s saved roar-lldb shims so to %s",
            __FUNCTION__, roar_shims_so_name.GetPath().data());
  return true;
}

bool GetRoarShimsPath(lldb_private::Log *log,
                      const SectionSP &roar_shims_section_sp,
                      lldb_private::FileSpec &roar_shims_so_name) {
  auto LogFileNotFound = [log, FnName = __FUNCTION__](const char *Path,
                                                      llvm::Twine Source) {
    LLDB_LOGF(log, "[JITLoaderROAR]::%s file %s (specified %s) does not exit",
              FnName, Path, Source.str().c_str());
  };
  static constexpr char RoarLldbShimsSoEnvName[] = "ROARLLDBSHIMSSO";
  lldb_private::FileSystem &FS = lldb_private::FileSystem::Instance();
  if (const char *ShimsPathFromEnv = getenv(RoarLldbShimsSoEnvName)) {
    if (FS.Exists(ShimsPathFromEnv)) {
      roar_shims_so_name = lldb_private::FileSpec(ShimsPathFromEnv);
      return true;
    }
    LogFileNotFound(ShimsPathFromEnv, llvm::Twine("in environment variable") +
                                          RoarLldbShimsSoEnvName);
  }

  if (lldb_private::FileSpec ShimsPathFromOpts =
          GetGlobalPluginProperties().GetROARLLDBShimsSharedLibraryPath()) {
    if (FS.Exists(ShimsPathFromOpts)) {
      roar_shims_so_name = ShimsPathFromOpts;
      return true;
    }
    LogFileNotFound(ShimsPathFromOpts.GetPath().c_str(),
                    "via plugin.jit-loader.roar.shims-so-path");
  }

  return SaveRoarShimsToDisk(log, roar_shims_section_sp, roar_shims_so_name);
}
} // namespace

static const char *GetPathToRuntime(lldb_private::Log *log,
                                    std::string &Buffer) {
  auto LogFileNotFound = [log, FnName = __FUNCTION__](const char *Path,
                                                      llvm::Twine Source) {
    LLDB_LOGF(log, "[JITLoaderROAR]::%s file %s (specified %s) does not exit",
              FnName, Path, Source.str().c_str());
  };

  static constexpr char RoarSoEnvName[] = "ROARSO";
  lldb_private::FileSystem &FS = lldb_private::FileSystem::Instance();
  if (const char *SoPathFromEnv = getenv(RoarSoEnvName)) {
    if (FS.Exists(SoPathFromEnv)) {
      return SoPathFromEnv;
    }
    LogFileNotFound(SoPathFromEnv,
                    llvm::Twine("in environment variable") + RoarSoEnvName);
  }

  if (lldb_private::FileSpec SoPathFromOpts =
          GetGlobalPluginProperties().GetROARSharedLibraryPath()) {
    if (FS.Exists(SoPathFromOpts)) {
      Buffer = SoPathFromOpts.GetPath();
      return Buffer.c_str();
    }
    LogFileNotFound(SoPathFromOpts.GetPath().c_str(),
                    "via plugin.jit-loader.roar.so-path");
  }
  LLDB_LOGF(log, "[JITLoaderROAR]::%s no libroar-runtime.so override.",
            __FUNCTION__);
  return "";
}

void lldb_roar_private::JITLoaderROARSB::SetJITBreakpoint(
    lldb_private::ModuleList &module_list) {
  if (DidSetJITBreakpoint())
    return;

  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);
  LLDB_LOGF(log, "JITLoaderROAR::%s looking for JIT register hook",
            __FUNCTION__);

  lldb_private::Target *target = &m_process->GetTarget();
  if (!m_roar_di) {
    m_process->WillCallExternalPlugin();
    for (const ModuleSP &M : module_list.Modules()) {
      SectionSP roar_shims_section_sp = M->GetSectionList()->FindSectionByName(
          lldb_private::ConstString{ROARDebugShimsSectionName});
      if (!roar_shims_section_sp)
        continue;
      lldb_private::FileSpec roar_shims_so_name;
      if (!GetRoarShimsPath(log, roar_shims_section_sp, roar_shims_so_name)) {
        LLDB_LOGF(
            log,
            "JITLoaderROAR::%s failed to save libroar-lldb-shims.so to disk",
            __FUNCTION__);
        continue;
      }
      std::string errMsg;
      llvm::sys::DynamicLibrary roar_shims_so =
          llvm::sys::DynamicLibrary::getPermanentLibrary(
              roar_shims_so_name.GetPath().c_str(), &errMsg);
      if (!roar_shims_so.isValid()) {
        LLDB_LOGF(log, "JITLoaderROAR::%s failed to load '%s': %s",
                  __FUNCTION__, roar_shims_so_name.GetPath().c_str(),
                  errMsg.c_str());
        continue;
      }
      LLDB_LOGF(log, "JITLoaderROAR::%s loaded roar debug shims from '%s'",
                __FUNCTION__, roar_shims_so_name.GetPath().c_str());
      decltype(&CreateROARDebugInterface) create_roar_debug_iterface;
      *reinterpret_cast<void **>(&create_roar_debug_iterface) =
          roar_shims_so.getAddressOfSymbol("CreateROARDebugInterface");
      if (!create_roar_debug_iterface) {
        LLDB_LOGF(log,
                  "JITLoaderROAR::%s could not find CreateROARDebugInterface "
                  "entrypoint",
                  __FUNCTION__);
        continue;
      }
      lldb_private::Debugger &D = target->GetDebugger();
      lldb::user_id_t debugger_id = D.GetID();
      uint32_t target_idx =
          D.GetTargetList().GetIndexOfTarget(target->shared_from_this());
      JITLoaderROARError err;
      m_process->WillCallExternalPlugin();
      if (GetGlobalPluginProperties().GetTestShimsErrorCreation())
        debugger_id = UINT64_MAX;
      m_roar_di.reset((*create_roar_debug_iterface)(
          roar_lldb::ROARDebugInterface::Version, debugger_id, target_idx, err,
          logToLLDB));
      m_process->DoneCallingExternalPlugin();
      if (!err.Success()) {
        LLDB_LOGF(log,
                  "JITLoaderROAR::%s error creating roar debug interface: %s",
                  __FUNCTION__, err.GetCString());
        continue;
      }
      break;
    }
    m_process->DoneCallingExternalPlugin();
  }

  if (!m_roar_di)
    return;
  addr_t jit_register_code_addr = 0;
  addr_t jit_dynamic_symbol_addr_trigger = 0;
  {
    PluginSync sync(*m_process);
    jit_register_code_addr = m_roar_di->GetJitRegisterCodeAddr();
    // Get address of a function that gets triggerd after shared memory for
    // sybmol arena gets initialized.
    jit_dynamic_symbol_addr_trigger =
        m_roar_di->GetDynamicSymbolArenaAddrTrigger();
  }

  LLDB_LOGF(log, "JITLoaderROAR::%s setting JIT breakpoint at 0x%" PRIx64,
            __FUNCTION__, jit_register_code_addr);

  lldb_private::Breakpoint *bp =
      m_process->GetTarget()
          .CreateBreakpoint(jit_register_code_addr, true, false)
          .get();
  bp->SetCallback(JITDebugBreakpointHit, this, true);
  bp->SetBreakpointKind("jit-debug-register");
  m_jit_break_id = bp->GetID();

  LLDB_LOGF(log,
            "JITLoaderROAR::%s setting JIT symbol arena load breakpoint at "
            "0x%" PRIx64,
            __FUNCTION__, jit_dynamic_symbol_addr_trigger);

  lldb_private::Breakpoint *bp_s =
      m_process->GetTarget()
          .CreateBreakpoint(jit_dynamic_symbol_addr_trigger, true, false)
          .get();
  bp_s->SetCallback(JITDebugDynamicSymbolArenaAddrBreakpointHit, this, true);
  bp_s->SetBreakpointKind("jit-dynamic-symbol-arena");
  m_jit_dynamic_symbol_arena_addr_break_id = bp_s->GetID();

  std::string PathToRuntimeBuffer;
  JITLoaderROARError err;
  const char *roarso_path = nullptr;
  {
    PluginSync sync(*m_process);
    roarso_path = m_roar_di->SetPathToRuntime(
        GetPathToRuntime(log, PathToRuntimeBuffer), err);
  }

  if (!err.Success()) {
    LLDB_LOGF(log, "JITLoaderROAR::%s failed to set roar runtime: %s",
              __FUNCTION__, err.GetCString());
    return;
  }
  LLDB_LOGF(log, "JITLoaderROAR::%s using roar runtime '%s'", __FUNCTION__,
            roarso_path);

  ReadJITEntries();
}

bool lldb_roar_private::JITLoaderROARSB::
    JITDebugDynamicSymbolArenaAddrBreakpointHit(
        void *baton, lldb_private::StoppointCallbackContext *context,
        lldb::user_id_t break_id, lldb::user_id_t break_loc_id) {
  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);
  LLDB_LOGF(log, "JITLoaderROAR::%s hit JIT  symbol arena load breakpoint",
            __FUNCTION__);
  static_cast<JITLoaderROARSB *>(baton)->NotifyJITWithInitSymbols();
  return false;
}

bool lldb_roar_private::JITLoaderROARSB::JITDebugBreakpointHit(
    void *baton, lldb_private::StoppointCallbackContext *context,
    user_id_t break_id, user_id_t break_loc_id) {
  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);
  LLDB_LOGF(log, "JITLoaderROAR::%s hit JIT breakpoint", __FUNCTION__);
  static_cast<JITLoaderROARSB *>(baton)->ReadJITEntries();
  return false;
}

void lldb_roar_private::JITLoaderROARSB::NotifyJITWithInitSymbols() {
  if (!m_roar_di)
    return;

  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);
  JITLoaderROARError err;
  {
    PluginSync sync(*m_process);
    m_roar_di->NotifyJITWithInitSymbols(err);
  }
  if (!err.Success()) {
    LLDB_LOGF(log, "JITLoaderROAR::%s failed to set delayed addresses: %s",
              __FUNCTION__, err.GetCString());
    return;
  }
}

void lldb_roar_private::JITLoaderROARSB::ReadJITEntries() {
  if (!m_roar_di)
    return;

  llvm::SaveAndRestore<bool> reading_jit_entries{m_reading_jit_entries, true};
  if (reading_jit_entries.get())
    return;

  lldb_private::Log *log = GetLog(lldb_private::LLDBLog::JITLoader);

  JITLoaderROARError err;
  {
    PluginSync sync(*m_process);
    m_roar_di->ReadJITFunctionList(
        GetGlobalPluginProperties().GetEagerSymbolGeneration(), err);
  }
  if (!err.Success()) {
    LLDB_LOGF(
        log,
        "JITLoaderROAR::%s failed to get jit function list head address: %s",
        __FUNCTION__, err.GetCString());
    return;
  }
}

// PluginInterface protocol
JITLoaderSP lldb_private::JITLoaderROAR::CreateInstance(Process *process,
                                                        bool force) {
  JITLoaderSP jit_loader_sp;
  bool enable;
  switch (GetGlobalPluginProperties().GetEnable()) {
  case EnableJITLoaderROAR::eEnableJITLoaderROARDefault:
  case EnableJITLoaderROAR::eEnableJITLoaderROAROn:
    // ROAR is only supported on 64-bit targets.
    enable = process->GetTarget().GetArchitecture().GetAddressByteSize() == 8;
    break;
  case EnableJITLoaderROAR::eEnableJITLoaderROAROff:
    enable = false;
    break;
  }
  if (enable)
    jit_loader_sp = std::make_shared<JITLoaderROAR>(process);
  return jit_loader_sp;
}

llvm::StringRef lldb_private::JITLoaderROAR::GetPluginDescriptionStatic() {
  return "JIT loader plug-in that watches for ROAR JIT events.";
}

void lldb_private::JITLoaderROAR::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                DebuggerInitialize);
}

void lldb_private::JITLoaderROAR::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

bool lldb_roar_private::JITLoaderROARSB::DidSetJITBreakpoint() const {
  return LLDB_BREAK_ID_IS_VALID(m_jit_break_id);
}
