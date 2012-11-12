//===-- lldb.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "llvm/ADT/StringRef.h"

#include "Plugins/ABI/MacOSX-i386/ABIMacOSX_i386.h"
#include "Plugins/ABI/MacOSX-arm/ABIMacOSX_arm.h"
#include "Plugins/ABI/SysV-x86_64/ABISysV_x86_64.h"
#include "Plugins/Disassembler/llvm/DisassemblerLLVM.h"
#include "Plugins/Disassembler/llvm/DisassemblerLLVMC.h"
#include "Plugins/Instruction/ARM/EmulateInstructionARM.h"
#include "Plugins/SymbolVendor/MacOSX/SymbolVendorMacOSX.h"
#include "Plugins/ObjectContainer/BSD-Archive/ObjectContainerBSDArchive.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDebugMap.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "Plugins/UnwindAssembly/x86/UnwindAssembly-x86.h"
#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"
#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/DynamicLoader/POSIX-DYLD/DynamicLoaderPOSIXDYLD.h"
#include "Plugins/Platform/FreeBSD/PlatformFreeBSD.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#ifndef LLDB_DISABLE_PYTHON
#include "Plugins/OperatingSystem/Python/OperatingSystemPython.h"
#endif
#if defined (__APPLE__)
#include "Plugins/DynamicLoader/MacOSX-DYLD/DynamicLoaderMacOSXDYLD.h"
#include "Plugins/DynamicLoader/Darwin-Kernel/DynamicLoaderDarwinKernel.h"
#include "Plugins/OperatingSystem/Darwin-Kernel/OperatingSystemDarwinKernel.h"
#include "Plugins/LanguageRuntime/CPlusPlus/ItaniumABI/ItaniumABILanguageRuntime.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV1.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV2.h"
#include "Plugins/ObjectContainer/Universal-Mach-O/ObjectContainerUniversalMachO.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/Process/MacOSX-Kernel/ProcessKDP.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteiOS.h"
#include "Plugins/Platform/MacOSX/PlatformiOSSimulator.h"
#endif

#include "Plugins/Process/mach-core/ProcessMachCore.h"

#if defined (__linux__)
#include "Plugins/Process/Linux/ProcessLinux.h"
#endif

#if defined (__FreeBSD__)
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "Plugins/Process/POSIX/ProcessPOSIX.h"
#include "Plugins/Process/FreeBSD/ProcessFreeBSD.h"
#endif

#include "Plugins/Platform/gdb-server/PlatformRemoteGDBServer.h"
#include "Plugins/DynamicLoader/Static/DynamicLoaderStatic.h"

using namespace lldb;
using namespace lldb_private;

void
lldb_private::Initialize ()
{
    // Make sure we inialize only once
    static Mutex g_inited_mutex(Mutex::eMutexTypeRecursive);
    static bool g_inited = false;

    Mutex::Locker locker(g_inited_mutex);
    if (!g_inited)
    {
        g_inited = true;
        Log::Initialize();
        Timer::Initialize ();
        Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
        
        ABIMacOSX_i386::Initialize();
        ABIMacOSX_arm::Initialize();
        ABISysV_x86_64::Initialize();
        DisassemblerLLVMC::Initialize();
        DisassemblerLLVM::Initialize();
        ObjectContainerBSDArchive::Initialize();
        ObjectFileELF::Initialize();
        SymbolFileDWARF::Initialize();
        SymbolFileSymtab::Initialize();
        UnwindAssemblyInstEmulation::Initialize();
        UnwindAssembly_x86::Initialize();
        EmulateInstructionARM::Initialize ();
        ObjectFilePECOFF::Initialize ();
        DynamicLoaderPOSIXDYLD::Initialize ();
        PlatformFreeBSD::Initialize();
        PlatformLinux::Initialize();
#ifndef LLDB_DISABLE_PYTHON
        OperatingSystemPython::Initialize();
#endif

#if defined (__APPLE__)
        //----------------------------------------------------------------------
        // Apple/Darwin hosted plugins
        //----------------------------------------------------------------------
        DynamicLoaderMacOSXDYLD::Initialize();
        DynamicLoaderDarwinKernel::Initialize();
        OperatingSystemDarwinKernel::Initialize();
        SymbolFileDWARFDebugMap::Initialize();
        ItaniumABILanguageRuntime::Initialize();
        AppleObjCRuntimeV2::Initialize();
        AppleObjCRuntimeV1::Initialize();
        ObjectContainerUniversalMachO::Initialize();
        ObjectFileMachO::Initialize();
        ProcessGDBRemote::Initialize();
        ProcessKDP::Initialize();
        ProcessMachCore::Initialize();
        SymbolVendorMacOSX::Initialize();
        PlatformRemoteiOS::Initialize();
        PlatformMacOSX::Initialize();
        PlatformiOSSimulator::Initialize();
#endif
#if defined (__linux__)
        //----------------------------------------------------------------------
        // Linux hosted plugins
        //----------------------------------------------------------------------
        ProcessLinux::Initialize();
#endif
#if defined (__FreeBSD__)
        ProcessFreeBSD::Initialize();
        ProcessGDBRemote::Initialize();
#endif
        //----------------------------------------------------------------------
        // Platform agnostic plugins
        //----------------------------------------------------------------------
        PlatformRemoteGDBServer::Initialize ();
        DynamicLoaderStatic::Initialize();

        // Scan for any system or user LLDB plug-ins
        PluginManager::Initialize();

        // The process settings need to know about installed plug-ins, so the Settings must be initialized
        // AFTER PluginManager::Initialize is called.
        
        Debugger::SettingsInitialize();
    }
}

void
lldb_private::WillTerminate()
{
    Host::WillTerminate();
}

void
lldb_private::Terminate ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
    
    // Terminate and unload and loaded system or user LLDB plug-ins
    PluginManager::Terminate();

    ABIMacOSX_i386::Terminate();
    ABIMacOSX_arm::Terminate();
    ABISysV_x86_64::Terminate();
    DisassemblerLLVMC::Terminate();
    DisassemblerLLVM::Terminate();
    ObjectContainerBSDArchive::Terminate();
    ObjectFileELF::Terminate();
    SymbolFileDWARF::Terminate();
    SymbolFileSymtab::Terminate();
    UnwindAssembly_x86::Terminate();
    UnwindAssemblyInstEmulation::Terminate();
    EmulateInstructionARM::Terminate ();
    ObjectFilePECOFF::Terminate ();
    DynamicLoaderPOSIXDYLD::Terminate ();
    PlatformFreeBSD::Terminate();
    PlatformLinux::Terminate();
#ifndef LLDB_DISABLE_PYTHON
    OperatingSystemPython::Terminate();
#endif

#if defined (__APPLE__)
    DynamicLoaderMacOSXDYLD::Terminate();
    DynamicLoaderDarwinKernel::Terminate();
    OperatingSystemDarwinKernel::Terminate();
    SymbolFileDWARFDebugMap::Terminate();
    ItaniumABILanguageRuntime::Terminate();
    AppleObjCRuntimeV2::Terminate();
    AppleObjCRuntimeV1::Terminate();
    ObjectContainerUniversalMachO::Terminate();
    ObjectFileMachO::Terminate();
    ProcessMachCore::Terminate();
    ProcessGDBRemote::Terminate();
    ProcessKDP::Terminate();
    SymbolVendorMacOSX::Terminate();
    PlatformMacOSX::Terminate();
    PlatformRemoteiOS::Terminate();
    PlatformiOSSimulator::Terminate();
#endif

    Debugger::SettingsTerminate ();

#if defined (__linux__)
    ProcessLinux::Terminate();
#endif

#if defined (__FreeBSD__)
    ProcessFreeBSD::Terminate();
    ProcessGDBRemote::Terminate();
#endif
    
    DynamicLoaderStatic::Terminate();

    Log::Terminate();
}

extern "C" const double liblldb_coreVersionNumber;
const char *
lldb_private::GetVersion ()
{
    static char g_version_string[32];
    if (g_version_string[0] == '\0')
        ::snprintf (g_version_string, sizeof(g_version_string), "LLDB-%g", liblldb_coreVersionNumber);

    return g_version_string;
}

const char *
lldb_private::GetVoteAsCString (Vote vote)
{
    switch (vote)
    {
    case eVoteNo:           return "no";
    case eVoteNoOpinion:    return "no opinion";
    case eVoteYes:          return "yes";
    default:
        break;
    }
    return "invalid";
}


const char *
lldb_private::GetSectionTypeAsCString (SectionType sect_type)
{
    switch (sect_type)
    {
    case eSectionTypeInvalid: return "invalid";
    case eSectionTypeCode: return "code";
    case eSectionTypeContainer: return "container";
    case eSectionTypeData: return "data";
    case eSectionTypeDataCString: return "data-cstr";
    case eSectionTypeDataCStringPointers: return "data-cstr-ptr";
    case eSectionTypeDataSymbolAddress: return "data-symbol-addr";
    case eSectionTypeData4: return "data-4-byte";
    case eSectionTypeData8: return "data-8-byte";
    case eSectionTypeData16: return "data-16-byte";
    case eSectionTypeDataPointers: return "data-ptrs";
    case eSectionTypeDebug: return "debug";
    case eSectionTypeZeroFill: return "zero-fill";
    case eSectionTypeDataObjCMessageRefs: return "objc-message-refs";
    case eSectionTypeDataObjCCFStrings: return "objc-cfstrings";
    case eSectionTypeDWARFDebugAbbrev: return "dwarf-abbrev";
    case eSectionTypeDWARFDebugAranges: return "dwarf-aranges";
    case eSectionTypeDWARFDebugFrame: return "dwarf-frame";
    case eSectionTypeDWARFDebugInfo: return "dwarf-info";
    case eSectionTypeDWARFDebugLine: return "dwarf-line";
    case eSectionTypeDWARFDebugLoc: return "dwarf-loc";
    case eSectionTypeDWARFDebugMacInfo: return "dwarf-macinfo";
    case eSectionTypeDWARFDebugPubNames: return "dwarf-pubnames";
    case eSectionTypeDWARFDebugPubTypes: return "dwarf-pubtypes";
    case eSectionTypeDWARFDebugRanges: return "dwarf-ranges";
    case eSectionTypeDWARFDebugStr: return "dwarf-str";
    case eSectionTypeDWARFAppleNames: return "apple-names";
    case eSectionTypeDWARFAppleTypes: return "apple-types";
    case eSectionTypeDWARFAppleNamespaces: return "apple-namespaces";
    case eSectionTypeDWARFAppleObjC: return "apple-objc";
    case eSectionTypeEHFrame: return "eh-frame";
    case eSectionTypeOther: return "regular";
    }
    return "unknown";

}

bool
lldb_private::NameMatches (const char *name, 
                           NameMatchType match_type, 
                           const char *match)
{
    if (match_type == eNameMatchIgnore)
        return true;

    if (name == match)
        return true;

    if (name && match)
    {
        llvm::StringRef name_sref(name);
        llvm::StringRef match_sref(match);
        switch (match_type)
        {
        case eNameMatchIgnore:
            return true;
        case eNameMatchEquals:      return name_sref == match_sref;
        case eNameMatchContains:    return name_sref.find (match_sref) != llvm::StringRef::npos;
        case eNameMatchStartsWith:  return name_sref.startswith (match_sref);
        case eNameMatchEndsWith:    return name_sref.endswith (match_sref);
        case eNameMatchRegularExpression:
            {
                RegularExpression regex (match);
                return regex.Execute (name);
            }
            break;
        default:
            assert (!"unhandled NameMatchType in lldb_private::NameMatches()");
            break;
        }
    }
    return false;
}
