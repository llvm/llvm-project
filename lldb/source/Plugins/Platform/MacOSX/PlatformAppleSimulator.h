//===-- PlatformAppleSimulator.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformAppleSimulator_h_
#define liblldb_PlatformAppleSimulator_h_

// C Includes
// C++ Includes
#include <mutex>

// Other libraries and framework includes
// Project includes
#include "lldb/Host/FileSpec.h"
#include "PlatformDarwin.h"
#include "PlatformiOSSimulatorCoreSimulatorSupport.h"

#include "llvm/ADT/Optional.h"

class PlatformAppleSimulator : public PlatformDarwin
{
public:
    //------------------------------------------------------------
    // Class Functions
    //------------------------------------------------------------
    static void
    Initialize ();
    
    static void
    Terminate ();
    
    //------------------------------------------------------------
    // Class Methods
    //------------------------------------------------------------
    PlatformAppleSimulator ();
    
    virtual
    ~PlatformAppleSimulator();
    
    lldb_private::Error
    LaunchProcess (lldb_private::ProcessLaunchInfo &launch_info) override;

    void
    GetStatus (lldb_private::Stream &strm) override;

    lldb_private::Error
    ConnectRemote (lldb_private::Args& args) override;

    lldb_private::Error
    DisconnectRemote () override;

    lldb::ProcessSP
    DebugProcess (lldb_private::ProcessLaunchInfo &launch_info,
                  lldb_private::Debugger &debugger,
                  lldb_private::Target *target,
                  lldb_private::Error &error) override;

protected:
    std::mutex m_core_sim_path_mutex;
    llvm::Optional<lldb_private::FileSpec> m_core_simulator_framework_path;
    llvm::Optional<CoreSimulatorSupport::Device> m_device;
    
    lldb_private::FileSpec
    GetCoreSimulatorPath();
    
    void
    LoadCoreSimulator ();
    
#if defined(__APPLE__)
    CoreSimulatorSupport::Device
    GetSimulatorDevice ();
#endif
    
private:
    DISALLOW_COPY_AND_ASSIGN (PlatformAppleSimulator);
    
};

#endif  // liblldb_PlatformAppleSimulator_h_
