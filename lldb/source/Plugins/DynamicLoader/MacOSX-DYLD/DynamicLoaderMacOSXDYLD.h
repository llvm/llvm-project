//===-- DynamicLoaderMacOSXDYLD.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This is the DynamicLoader plugin for Darwin (macOS / iPhoneOS / tvOS /
// watchOS)
// platforms earlier than 2016, where lldb would read the "dyld_all_image_infos"
// dyld internal structure to understand where things were loaded and the
// solib loaded/unloaded notification function we put a breakpoint on gives us
// an array of (load address, mod time, file path) tuples.
//
// As of late 2016, the new DynamicLoaderMacOS plugin should be used, which uses
// dyld SPI functions to get the same information without reading internal dyld
// data structures.

#ifndef liblldb_DynamicLoaderMacOSXDYLD_h_
#define liblldb_DynamicLoaderMacOSXDYLD_h_

// C Includes
// C++ Includes
#include <mutex>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/StructuredData.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/SafeMachO.h"
#include "lldb/Utility/UUID.h"

#include "DynamicLoaderDarwin.h"

class DynamicLoaderMacOSXDYLD : public lldb_private::DynamicLoaderDarwin {
public:
  DynamicLoaderMacOSXDYLD(lldb_private::Process *process);

  virtual ~DynamicLoaderMacOSXDYLD() override;

  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static lldb_private::DynamicLoader *
  CreateInstance(lldb_private::Process *process, bool force);

  //------------------------------------------------------------------
  /// Called after attaching a process.
  ///
  /// Allow DynamicLoader plug-ins to execute some code after
  /// attaching to a process.
  //------------------------------------------------------------------
  bool ProcessDidExec() override;

  lldb_private::Status CanLoadImage() override;

  bool GetSharedCacheInformation(
      lldb::addr_t &base_address, lldb_private::UUID &uuid,
      lldb_private::LazyBool &using_shared_cache,
      lldb_private::LazyBool &private_shared_cache) override;

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

protected:
  void PutToLog(lldb_private::Log *log) const;

  void DoInitialImageFetch() override;

  bool NeedToDoInitialImageFetch() override;

  bool DidSetNotificationBreakpoint() override;

  void DoClear() override;

  bool ReadDYLDInfoFromMemoryAndSetNotificationCallback(lldb::addr_t addr);

  static bool
  NotifyBreakpointHit(void *baton,
                      lldb_private::StoppointCallbackContext *context,
                      lldb::user_id_t break_id, lldb::user_id_t break_loc_id);

  uint32_t AddrByteSize();

  bool ReadMachHeader(lldb::addr_t addr, llvm::MachO::mach_header *header,
                      lldb_private::DataExtractor *load_command_data);

  uint32_t ParseLoadCommands(const lldb_private::DataExtractor &data,
                             ImageInfo &dylib_info,
                             lldb_private::FileSpec *lc_id_dylinker);

  struct DYLDAllImageInfos {
    uint32_t version;
    uint32_t dylib_info_count;            // Version >= 1
    lldb::addr_t dylib_info_addr;         // Version >= 1
    lldb::addr_t notification;            // Version >= 1
    bool processDetachedFromSharedRegion; // Version >= 1
    bool libSystemInitialized;            // Version >= 2
    lldb::addr_t dyldImageLoadAddress;    // Version >= 2

    DYLDAllImageInfos()
        : version(0), dylib_info_count(0),
          dylib_info_addr(LLDB_INVALID_ADDRESS),
          notification(LLDB_INVALID_ADDRESS),
          processDetachedFromSharedRegion(false), libSystemInitialized(false),
          dyldImageLoadAddress(LLDB_INVALID_ADDRESS) {}

    void Clear() {
      version = 0;
      dylib_info_count = 0;
      dylib_info_addr = LLDB_INVALID_ADDRESS;
      notification = LLDB_INVALID_ADDRESS;
      processDetachedFromSharedRegion = false;
      libSystemInitialized = false;
      dyldImageLoadAddress = LLDB_INVALID_ADDRESS;
    }

    bool IsValid() const { return version >= 1 || version <= 6; }
  };

  static lldb::ByteOrder GetByteOrderFromMagic(uint32_t magic);

  bool SetNotificationBreakpoint() override;

  void ClearNotificationBreakpoint() override;

  // There is a little tricky bit where you might initially attach while dyld is
  // updating
  // the all_image_infos, and you can't read the infos, so you have to continue
  // and pick it
  // up when you hit the update breakpoint.  At that point, you need to run this
  // initialize
  // function, but when you do it that way you DON'T need to do the extra work
  // you would at
  // the breakpoint.
  // So this function will only do actual work if the image infos haven't been
  // read yet.
  // If it does do any work, then it will return true, and false otherwise.
  // That way you can
  // call it in the breakpoint action, and if it returns true you're done.
  bool InitializeFromAllImageInfos();

  bool ReadAllImageInfosStructure();

  bool AddModulesUsingImageInfosAddress(lldb::addr_t image_infos_addr,
                                        uint32_t image_infos_count);

  bool RemoveModulesUsingImageInfosAddress(lldb::addr_t image_infos_addr,
                                           uint32_t image_infos_count);

  void UpdateImageInfosHeaderAndLoadCommands(ImageInfo::collection &image_infos,
                                             uint32_t infos_count,
                                             bool update_executable);

  bool ReadImageInfos(lldb::addr_t image_infos_addr, uint32_t image_infos_count,
                      ImageInfo::collection &image_infos);

  lldb::addr_t m_dyld_all_image_infos_addr;
  DYLDAllImageInfos m_dyld_all_image_infos;
  uint32_t m_dyld_all_image_infos_stop_id;
  lldb::user_id_t m_break_id;
  mutable std::recursive_mutex m_mutex;
  bool m_process_image_addr_is_all_images_infos;

private:
  DISALLOW_COPY_AND_ASSIGN(DynamicLoaderMacOSXDYLD);
};

#endif // liblldb_DynamicLoaderMacOSXDYLD_h_
