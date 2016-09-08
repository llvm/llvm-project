//===-- DynamicLoaderDarwin.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DynamicLoaderDarwin_h_
#define liblldb_DynamicLoaderDarwin_h_

// C Includes
// C++ Includes
#include <map>
#include <mutex>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/StructuredData.h"
#include "lldb/Core/UUID.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/SafeMachO.h"

namespace lldb_private {

class DynamicLoaderDarwin : public lldb_private::DynamicLoader {
public:
  DynamicLoaderDarwin(lldb_private::Process *process);

  virtual ~DynamicLoaderDarwin() override;

  //------------------------------------------------------------------
  /// Called after attaching a process.
  ///
  /// Allow DynamicLoader plug-ins to execute some code after
  /// attaching to a process.
  //------------------------------------------------------------------
  void DidAttach() override;

  void DidLaunch() override;

  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(lldb_private::Thread &thread,
                                                  bool stop_others) override;

  size_t FindEquivalentSymbols(
      lldb_private::Symbol *original_symbol,
      lldb_private::ModuleList &module_list,
      lldb_private::SymbolContextList &equivalent_symbols) override;

  lldb::addr_t GetThreadLocalData(const lldb::ModuleSP module,
                                  const lldb::ThreadSP thread,
                                  lldb::addr_t tls_file_addr) override;

  bool AlwaysRelyOnEHUnwindInfo(lldb_private::SymbolContext &sym_ctx) override;

  virtual void DoInitialImageFetch() = 0;

  virtual bool NeedToDoInitialImageFetch() = 0;

protected:
  void PrivateInitialize(lldb_private::Process *process);

  void PrivateProcessStateChanged(lldb_private::Process *process,
                                  lldb::StateType state);

  void Clear(bool clear_process);

  // Clear method for classes derived from this one
  virtual void DoClear() = 0;

  void SetDYLDModule(lldb::ModuleSP &dyld_module_sp);

  lldb::ModuleSP GetDYLDModule();

  class Segment {
  public:
    Segment()
        : name(), vmaddr(LLDB_INVALID_ADDRESS), vmsize(0), fileoff(0),
          filesize(0), maxprot(0), initprot(0), nsects(0), flags(0) {}

    lldb_private::ConstString name;
    lldb::addr_t vmaddr;
    lldb::addr_t vmsize;
    lldb::addr_t fileoff;
    lldb::addr_t filesize;
    uint32_t maxprot;
    uint32_t initprot;
    uint32_t nsects;
    uint32_t flags;

    bool operator==(const Segment &rhs) const {
      return name == rhs.name && vmaddr == rhs.vmaddr && vmsize == rhs.vmsize;
    }

    void PutToLog(lldb_private::Log *log, lldb::addr_t slide) const;
  };

  struct ImageInfo {
    lldb::addr_t address;  // Address of mach header for this dylib
    lldb::addr_t slide;    // The amount to slide all segments by if there is a
                           // global slide.
    lldb::addr_t mod_date; // Modification date for this dylib
    lldb_private::FileSpec file_spec; // Resolved path for this dylib
    lldb_private::UUID
        uuid; // UUID for this dylib if it has one, else all zeros
    llvm::MachO::mach_header header; // The mach header for this image
    std::vector<Segment> segments;   // All segment vmaddr and vmsize pairs for
                                   // this executable (from memory of inferior)
    uint32_t load_stop_id; // The process stop ID that the sections for this
                           // image were loaded

    ImageInfo()
        : address(LLDB_INVALID_ADDRESS), slide(0), mod_date(0), file_spec(),
          uuid(), header(), segments(), load_stop_id(0) {}

    void Clear(bool load_cmd_data_only) {
      if (!load_cmd_data_only) {
        address = LLDB_INVALID_ADDRESS;
        slide = 0;
        mod_date = 0;
        file_spec.Clear();
        ::memset(&header, 0, sizeof(header));
      }
      uuid.Clear();
      segments.clear();
      load_stop_id = 0;
    }

    bool operator==(const ImageInfo &rhs) const {
      return address == rhs.address && slide == rhs.slide &&
             mod_date == rhs.mod_date && file_spec == rhs.file_spec &&
             uuid == rhs.uuid &&
             memcmp(&header, &rhs.header, sizeof(header)) == 0 &&
             segments == rhs.segments;
    }

    bool UUIDValid() const { return uuid.IsValid(); }

    uint32_t GetAddressByteSize() {
      if (header.cputype) {
        if (header.cputype & llvm::MachO::CPU_ARCH_ABI64)
          return 8;
        else
          return 4;
      }
      return 0;
    }

    lldb_private::ArchSpec GetArchitecture() const {
      return lldb_private::ArchSpec(lldb_private::eArchTypeMachO,
                                    header.cputype, header.cpusubtype);
    }

    const Segment *FindSegment(const lldb_private::ConstString &name) const;

    void PutToLog(lldb_private::Log *log) const;

    typedef std::vector<ImageInfo> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;
  };

  bool UpdateImageLoadAddress(lldb_private::Module *module, ImageInfo &info);

  bool UnloadModuleSections(lldb_private::Module *module, ImageInfo &info);

  ImageInfo *FindImageInfoForAddress(lldb::addr_t load_address);

  lldb::ModuleSP FindTargetModuleForImageInfo(ImageInfo &image_info,
                                              bool can_create,
                                              bool *did_create_ptr);

  void UnloadImages(const std::vector<lldb::addr_t> &solib_addresses);

  virtual bool SetNotificationBreakpoint() = 0;

  virtual void ClearNotificationBreakpoint() = 0;

  virtual bool DidSetNotificationBreakpoint() = 0;

  typedef std::map<uint64_t, lldb::addr_t> PthreadKeyToTLSMap;
  typedef std::map<lldb::user_id_t, PthreadKeyToTLSMap> ThreadIDToTLSMap;

  std::recursive_mutex &GetMutex() const { return m_mutex; }

  lldb::ModuleSP GetPThreadLibraryModule();

  lldb_private::Address GetPthreadSetSpecificAddress();

  bool JSONImageInformationIntoImageInfo(
      lldb_private::StructuredData::ObjectSP image_details,
      ImageInfo::collection &image_infos);

  // If image_infos contains / may contain dyld image, call this method
  // to keep our internal record keeping of the special dyld binary up-to-date.
  void UpdateDYLDImageInfoFromNewImageInfos(ImageInfo::collection &image_infos);

  // if image_info is a dyld binary, call this method
  void UpdateDYLDImageInfoFromNewImageInfo(ImageInfo &image_info);

  // If image_infos contains / may contain executable image, call this method
  // to keep our internal record keeping of the special dyld binary up-to-date.
  void AddExecutableModuleIfInImageInfos(ImageInfo::collection &image_infos);

  bool AddModulesUsingImageInfos(ImageInfo::collection &image_infos);

  lldb::ModuleWP m_dyld_module_wp;
  lldb::ModuleWP m_libpthread_module_wp;
  lldb_private::Address m_pthread_getspecific_addr;
  ThreadIDToTLSMap m_tid_to_tls_map;
  ImageInfo::collection
      m_dyld_image_infos;              // Current shared libraries information
  uint32_t m_dyld_image_infos_stop_id; // The process stop ID that
                                       // "m_dyld_image_infos" is valid for
  ImageInfo m_dyld;
  mutable std::recursive_mutex m_mutex;

private:
  DISALLOW_COPY_AND_ASSIGN(DynamicLoaderDarwin);
};

} // namespace lldb_private

#endif // liblldb_DynamicLoaderDarwin_h_
