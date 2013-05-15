//===-- DynamicLoaderStatic.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DynamicLoaderStatic_h_
#define liblldb_DynamicLoaderStatic_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>
#include <string>

// Other libraries and framework includes
#include "llvm/Support/MachO.h"

#include "lldb/Target/DynamicLoader.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/UUID.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"

class DynamicLoaderStatic : public lldb_private::DynamicLoader
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::DynamicLoader *
    CreateInstance (lldb_private::Process *process, bool force);

    DynamicLoaderStatic (lldb_private::Process *process);

    virtual
    ~DynamicLoaderStatic ();
    //------------------------------------------------------------------
    /// Called after attaching a process.
    ///
    /// Allow DynamicLoader plug-ins to execute some code after
    /// attaching to a process.
    //------------------------------------------------------------------
    virtual void
    DidAttach ();

    virtual void
    DidLaunch ();

    virtual lldb::ThreadPlanSP
    GetStepThroughTrampolinePlan (lldb_private::Thread &thread,
                                  bool stop_others);

    virtual lldb_private::Error
    CanLoadImage ();

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

private:
    void
    LoadAllImagesAtFileAddresses ();

    DISALLOW_COPY_AND_ASSIGN (DynamicLoaderStatic);
};

#endif  // liblldb_DynamicLoaderStatic_h_
