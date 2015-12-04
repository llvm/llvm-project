//===-- PlatformAndroid.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/StringConvert.h"
#include "Utility/UriParser.h"

// Project includes
#include "AdbClient.h"
#include "PlatformAndroid.h"
#include "PlatformAndroidRemoteGDBServer.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;

static uint32_t g_initialize_count = 0;
static const unsigned int g_android_default_cache_size = 2048; // Fits inside 4k adb packet.

void
PlatformAndroid::Initialize ()
{
    PlatformLinux::Initialize ();

    if (g_initialize_count++ == 0)
    {
#if defined(__ANDROID__)
        PlatformSP default_platform_sp (new PlatformAndroid(true));
        default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
        Platform::SetHostPlatform (default_platform_sp);
#endif
        PluginManager::RegisterPlugin (PlatformAndroid::GetPluginNameStatic(false),
                                       PlatformAndroid::GetPluginDescriptionStatic(false),
                                       PlatformAndroid::CreateInstance);
    }
}

void
PlatformAndroid::Terminate ()
{
    if (g_initialize_count > 0)
    {
        if (--g_initialize_count == 0)
        {
            PluginManager::UnregisterPlugin (PlatformAndroid::CreateInstance);
        }
    }

    PlatformLinux::Terminate ();
}

PlatformSP
PlatformAndroid::CreateInstance (bool force, const ArchSpec *arch)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_PLATFORM));
    if (log)
    {
        const char *arch_name;
        if (arch && arch->GetArchitectureName ())
            arch_name = arch->GetArchitectureName ();
        else
            arch_name = "<null>";

        const char *triple_cstr = arch ? arch->GetTriple ().getTriple ().c_str() : "<null>";

        log->Printf ("PlatformAndroid::%s(force=%s, arch={%s,%s})", __FUNCTION__, force ? "true" : "false", arch_name, triple_cstr);
    }

    bool create = force;
    if (create == false && arch && arch->IsValid())
    {
        const llvm::Triple &triple = arch->GetTriple();
        switch (triple.getVendor())
        {
            case llvm::Triple::PC:
                create = true;
                break;

#if defined(__ANDROID__)
            // Only accept "unknown" for the vendor if the host is android and
            // it "unknown" wasn't specified (it was just returned because it
            // was NOT specified_
            case llvm::Triple::VendorType::UnknownVendor:
                create = !arch->TripleVendorWasSpecified();
                break;
#endif
            default:
                break;
        }
        
        if (create)
        {
            switch (triple.getOS())
            {
                case llvm::Triple::Android:
                    break;

#if defined(__ANDROID__)
                // Only accept "unknown" for the OS if the host is android and
                // it "unknown" wasn't specified (it was just returned because it
                // was NOT specified)
                case llvm::Triple::OSType::UnknownOS:
                    create = !arch->TripleOSWasSpecified();
                    break;
#endif
                default:
                    create = false;
                    break;
            }
        }
    }

    if (create)
    {
        if (log)
            log->Printf ("PlatformAndroid::%s() creating remote-android platform", __FUNCTION__);
        return PlatformSP(new PlatformAndroid(false));
    }

    if (log)
        log->Printf ("PlatformAndroid::%s() aborting creation of remote-android platform", __FUNCTION__);

    return PlatformSP();
}

PlatformAndroid::PlatformAndroid (bool is_host) :
    PlatformLinux(is_host),
    m_sdk_version(0)
{
}

PlatformAndroid::~PlatformAndroid()
{
}

ConstString
PlatformAndroid::GetPluginNameStatic (bool is_host)
{
    if (is_host)
    {
        static ConstString g_host_name(Platform::GetHostPlatformName ());
        return g_host_name;
    }
    else
    {
        static ConstString g_remote_name("remote-android");
        return g_remote_name;
    }
}

const char *
PlatformAndroid::GetPluginDescriptionStatic (bool is_host)
{
    if (is_host)
        return "Local Android user platform plug-in.";
    else
        return "Remote Android user platform plug-in.";
}

ConstString
PlatformAndroid::GetPluginName()
{
    return GetPluginNameStatic(IsHost());
}

Error
PlatformAndroid::ConnectRemote(Args& args)
{
    m_device_id.clear();

    if (IsHost())
    {
        return Error ("can't connect to the host platform '%s', always connected", GetPluginName().GetCString());
    }

    if (!m_remote_platform_sp)
        m_remote_platform_sp = PlatformSP(new PlatformAndroidRemoteGDBServer());

    int port;
    std::string scheme, host, path;
    const char *url = args.GetArgumentAtIndex(0);
    if (!url)
        return Error("URL is null.");
    if (!UriParser::Parse(url, scheme, host, port, path))
        return Error("Invalid URL: %s", url);
    if (host != "localhost")
        m_device_id = host;

    auto error = PlatformLinux::ConnectRemote(args);
    if (error.Success())
    {
        AdbClient adb;
        error = AdbClient::CreateByDeviceID(m_device_id, adb);
        if (error.Fail())
            return error;

        m_device_id = adb.GetDeviceID();
    }
    return error;
}

Error
PlatformAndroid::GetFile (const FileSpec& source,
                          const FileSpec& destination)
{
    if (IsHost() || !m_remote_platform_sp)
        return PlatformLinux::GetFile(source, destination);

    FileSpec source_spec (source.GetPath (false), false, FileSpec::ePathSyntaxPosix);
    if (source_spec.IsRelative())
        source_spec = GetRemoteWorkingDirectory ().CopyByAppendingPathComponent (source_spec.GetCString (false));

    AdbClient adb (m_device_id);
    return adb.PullFile (source_spec, destination);
}

Error
PlatformAndroid::PutFile (const FileSpec& source,
                          const FileSpec& destination,
                          uint32_t uid,
                          uint32_t gid)
{
    if (IsHost() || !m_remote_platform_sp)
        return PlatformLinux::PutFile (source, destination, uid, gid);

    FileSpec destination_spec (destination.GetPath (false), false, FileSpec::ePathSyntaxPosix);
    if (destination_spec.IsRelative())
        destination_spec = GetRemoteWorkingDirectory ().CopyByAppendingPathComponent (destination_spec.GetCString (false));

    AdbClient adb (m_device_id);
    // TODO: Set correct uid and gid on remote file.
    return adb.PushFile(source, destination_spec);
}

const char *
PlatformAndroid::GetCacheHostname ()
{
    return m_device_id.c_str ();
}

Error
PlatformAndroid::DownloadModuleSlice (const FileSpec &src_file_spec,
                                      const uint64_t src_offset,
                                      const uint64_t src_size,
                                      const FileSpec &dst_file_spec)
{
    if (src_offset != 0)
        return Error ("Invalid offset - %" PRIu64, src_offset);

    return GetFile (src_file_spec, dst_file_spec);
}

Error
PlatformAndroid::DisconnectRemote()
{
    Error error = PlatformLinux::DisconnectRemote();
    if (error.Success())
    {
        m_device_id.clear();
        m_sdk_version = 0;
    }
    return error;
}

uint32_t
PlatformAndroid::GetDefaultMemoryCacheLineSize()
{
    return g_android_default_cache_size;
}

uint32_t
PlatformAndroid::GetSdkVersion()
{
    if (!IsConnected())
        return 0;

    if (m_sdk_version != 0)
        return m_sdk_version;

    std::string version_string;
    AdbClient adb(m_device_id);
    Error error = adb.Shell("getprop ro.build.version.sdk", 5000 /* ms */, &version_string);
    version_string = llvm::StringRef(version_string).trim().str();

    if (error.Fail() || version_string.empty())
    {
        Log* log = GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM);
        if (log)
            log->Printf("Get SDK version failed. (error: %s, output: %s)",
                        error.AsCString(), version_string.c_str());
        return 0;
    }

    m_sdk_version = StringConvert::ToUInt32(version_string.c_str());
    return m_sdk_version;
}

Error
PlatformAndroid::DownloadSymbolFile (const lldb::ModuleSP& module_sp,
                                     const FileSpec& dst_file_spec)
{
    // For oat file we can try to fetch additional debug info from the device
    if (module_sp->GetFileSpec().GetFileNameExtension() != ConstString("oat"))
        return Error("Symbol file downloading only supported for oat files");

    // If we have no information about the platform file we can't execute oatdump
    if (!module_sp->GetPlatformFileSpec())
        return Error("No platform file specified");

    // Symbolizer isn't available before SDK version 23
    if (GetSdkVersion() < 23)
        return Error("Symbol file generation only supported on SDK 23+");

    // If we already have symtab then we don't have to try and generate one
    if (module_sp->GetSectionList()->FindSectionByName(ConstString(".symtab")) != nullptr)
        return Error("Symtab already available in the module");

    AdbClient adb(m_device_id);

    std::string tmpdir;
    Error error = adb.Shell("mktemp --directory --tmpdir /data/local/tmp", 5000 /* ms */, &tmpdir);
    if (error.Fail() || tmpdir.empty())
        return Error("Failed to generate temporary directory on the device (%s)", error.AsCString());
    tmpdir = llvm::StringRef(tmpdir).trim().str();

    // Create file remover for the temporary directory created on the device
    std::unique_ptr<std::string, std::function<void(std::string*)>> tmpdir_remover(
        &tmpdir,
        [this, &adb](std::string* s) {
            StreamString command;
            command.Printf("rm -rf %s", s->c_str());
            Error error = adb.Shell(command.GetData(), 5000 /* ms */, nullptr);

            Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_PLATFORM));
            if (error.Fail())
                log->Printf("Failed to remove temp directory: %s", error.AsCString());
        }
    );

    FileSpec symfile_platform_filespec(tmpdir.c_str(), false);
    symfile_platform_filespec.AppendPathComponent("symbolized.oat");

    // Execute oatdump on the remote device to generate a file with symtab
    StreamString command;
    command.Printf("oatdump --symbolize=%s --output=%s",
                   module_sp->GetPlatformFileSpec().GetCString(false),
                   symfile_platform_filespec.GetCString(false));
    error = adb.Shell(command.GetData(), 60000 /* ms */, nullptr);
    if (error.Fail())
        return Error("Oatdump failed: %s", error.AsCString());

    // Download the symbolfile from the remote device
    return GetFile(symfile_platform_filespec, dst_file_spec);
}

bool
PlatformAndroid::GetRemoteOSVersion ()
{
    m_major_os_version = GetSdkVersion();
    m_minor_os_version = 0;
    m_update_os_version = 0;
    return m_major_os_version != 0;
}

const char*
PlatformAndroid::GetLibdlFunctionDeclarations() const
{
    return R"(
              extern "C" void* dlopen(const char*, int) asm("__dl_dlopen");
              extern "C" void* dlsym(void*, const char*) asm("__dl_dlsym");
              extern "C" int   dlclose(void*) asm("__dl_dlclose");
              extern "C" char* dlerror(void) asm("__dl_dlerror");
             )";
}
