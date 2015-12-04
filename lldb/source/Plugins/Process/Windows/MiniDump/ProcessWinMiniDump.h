//===-- ProcessWinMiniDump.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessWinMiniDump_h_
#define liblldb_ProcessWinMiniDump_h_

#include <list>
#include <vector>

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Target/Process.h"

#include "Plugins/Process/Windows/Common/ProcessWindows.h"

struct ThreadData;

class ProcessWinMiniDump : public lldb_private::ProcessWindows
{
  public:
    static lldb::ProcessSP
    CreateInstance (lldb::TargetSP target_sp,
                    lldb_private::Listener &listener,
                    const lldb_private::FileSpec *crash_file_path);

    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    ProcessWinMiniDump(lldb::TargetSP target_sp,
                       lldb_private::Listener &listener,
                       const lldb_private::FileSpec &core_file);

    virtual
    ~ProcessWinMiniDump();

    bool
    CanDebug(lldb::TargetSP target_sp, bool plugin_specified_by_name) override;

    lldb_private::Error
    DoLoadCore() override;

    lldb_private::DynamicLoader *
    GetDynamicLoader() override;

    lldb_private::ConstString
    GetPluginName() override;

    uint32_t
    GetPluginVersion() override;

    lldb_private::Error
    DoDestroy() override;

    void
    RefreshStateAfterStop() override;

    bool
    IsAlive() override;

    bool
    WarnBeforeDetach () const override;

    size_t
    ReadMemory(lldb::addr_t addr, void *buf, size_t size, lldb_private::Error &error) override;

    size_t
    DoReadMemory(lldb::addr_t addr, void *buf, size_t size, lldb_private::Error &error) override;

    lldb_private::ArchSpec
    GetArchitecture();

    lldb_private::Error
    GetMemoryRegionInfo(lldb::addr_t load_addr, lldb_private::MemoryRegionInfo &range_info) override;

  protected:
    void
    Clear();

    bool
    UpdateThreadList(lldb_private::ThreadList &old_thread_list,
                     lldb_private::ThreadList &new_thread_list) override;

  private:
    // Describes a range of memory captured in the mini dump.
    struct Range {
      lldb::addr_t start;  // virtual address of the beginning of the range
      size_t size;         // size of the range in bytes
      const uint8_t *ptr;  // absolute pointer to the first byte of the range
    };

    // If the mini dump has a memory range that contains the desired address, it
    // returns true with the details of the range in *range_out.  Otherwise, it
    // returns false.
    bool
    FindMemoryRange(lldb::addr_t addr, Range *range_out) const;

    lldb_private::Error
    MapMiniDumpIntoMemory(const char *file);

    lldb_private::ArchSpec
    DetermineArchitecture();

    void
    ReadExceptionRecord();

    void
    ReadMiscInfo();

    void
    ReadModuleList();

    // A thin wrapper around WinAPI's MiniDumpReadDumpStream to avoid redundant
    // checks.  If there's a failure (e.g., if the requested stream doesn't exist),
    // the function returns nullptr and sets *size_out to 0.
    void *
    FindDumpStream(unsigned stream_number, size_t *size_out) const;

    // Isolate the data to keep Windows-specific types out of this header.  Can't
    // use the typical pimpl idiom because the implementation of this class also
    // needs access to public and protected members of the base class.
    class Data;
    std::unique_ptr<Data> m_data_up;
};

#endif  // liblldb_ProcessWinMiniDump_h_
