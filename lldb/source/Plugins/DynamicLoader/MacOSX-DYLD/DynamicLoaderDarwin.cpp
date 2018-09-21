//===-- DynamicLoaderDarwin.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DynamicLoaderDarwin.h"

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanCallFunction.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"

//#define ENABLE_DEBUG_PRINTF // COMMENT THIS LINE OUT PRIOR TO CHECKIN
#ifdef ENABLE_DEBUG_PRINTF
#include <stdio.h>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

#ifndef __APPLE__
#include "Utility/UuidCompatibility.h"
#else
#include <uuid/uuid.h>
#endif

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
DynamicLoaderDarwin::DynamicLoaderDarwin(Process *process)
    : DynamicLoader(process), m_dyld_module_wp(), m_libpthread_module_wp(),
      m_pthread_getspecific_addr(), m_tid_to_tls_map(), m_dyld_image_infos(),
      m_dyld_image_infos_stop_id(UINT32_MAX), m_dyld(), m_mutex() {}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
DynamicLoaderDarwin::~DynamicLoaderDarwin() {}

//------------------------------------------------------------------
/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
//------------------------------------------------------------------
void DynamicLoaderDarwin::DidAttach() {
  PrivateInitialize(m_process);
  DoInitialImageFetch();
  SetNotificationBreakpoint();
}

//------------------------------------------------------------------
/// Called after attaching a process.
///
/// Allow DynamicLoader plug-ins to execute some code after
/// attaching to a process.
//------------------------------------------------------------------
void DynamicLoaderDarwin::DidLaunch() {
  PrivateInitialize(m_process);
  DoInitialImageFetch();
  SetNotificationBreakpoint();
}

//----------------------------------------------------------------------
// Clear out the state of this class.
//----------------------------------------------------------------------
void DynamicLoaderDarwin::Clear(bool clear_process) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (clear_process)
    m_process = NULL;
  m_dyld_image_infos.clear();
  m_dyld_image_infos_stop_id = UINT32_MAX;
  m_dyld.Clear(false);
}

ModuleSP DynamicLoaderDarwin::FindTargetModuleForImageInfo(
    ImageInfo &image_info, bool can_create, bool *did_create_ptr) {
  if (did_create_ptr)
    *did_create_ptr = false;

  Target &target = m_process->GetTarget();
  const ModuleList &target_images = target.GetImages();
  ModuleSpec module_spec(image_info.file_spec);
  module_spec.GetUUID() = image_info.uuid;
  ModuleSP module_sp(target_images.FindFirstModule(module_spec));

  if (module_sp && !module_spec.GetUUID().IsValid() &&
      !module_sp->GetUUID().IsValid()) {
    // No UUID, we must rely upon the cached module modification time and the
    // modification time of the file on disk
    if (module_sp->GetModificationTime() !=
        FileSystem::GetModificationTime(module_sp->GetFileSpec()))
      module_sp.reset();
  }

  if (!module_sp) {
    if (can_create) {
      module_sp = target.GetSharedModule(module_spec);
      if (!module_sp || module_sp->GetObjectFile() == NULL)
        module_sp = m_process->ReadModuleFromMemory(image_info.file_spec,
                                                    image_info.address);

      if (did_create_ptr)
        *did_create_ptr = (bool)module_sp;
    }
  }
  return module_sp;
}

void DynamicLoaderDarwin::UnloadImages(
    const std::vector<lldb::addr_t> &solib_addresses) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_process->GetStopID() == m_dyld_image_infos_stop_id)
    return;

  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  Target &target = m_process->GetTarget();
  if (log)
    log->Printf("Removing %" PRId64 " modules.",
                (uint64_t)solib_addresses.size());

  ModuleList unloaded_module_list;

  for (addr_t solib_addr : solib_addresses) {
    Address header;
    if (header.SetLoadAddress(solib_addr, &target)) {
      if (header.GetOffset() == 0) {
        ModuleSP module_to_remove(header.GetModule());
        if (module_to_remove.get()) {
          if (log)
            log->Printf("Removing module at address 0x%" PRIx64, solib_addr);
          // remove the sections from the Target
          UnloadSections(module_to_remove);
          // add this to the list of modules to remove
          unloaded_module_list.AppendIfNeeded(module_to_remove);
          // remove the entry from the m_dyld_image_infos
          ImageInfo::collection::iterator pos, end = m_dyld_image_infos.end();
          for (pos = m_dyld_image_infos.begin(); pos != end; pos++) {
            if (solib_addr == (*pos).address) {
              m_dyld_image_infos.erase(pos);
              break;
            }
          }
        }
      }
    }
  }

  if (unloaded_module_list.GetSize() > 0) {
    if (log) {
      log->PutCString("Unloaded:");
      unloaded_module_list.LogUUIDAndPaths(
          log, "DynamicLoaderDarwin::UnloadModules");
    }
    m_process->GetTarget().GetImages().Remove(unloaded_module_list);
    m_dyld_image_infos_stop_id = m_process->GetStopID();
  }
}

void DynamicLoaderDarwin::UnloadAllImages() {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  ModuleList unloaded_modules_list;

  Target &target = m_process->GetTarget();
  const ModuleList &target_modules = target.GetImages();
  std::lock_guard<std::recursive_mutex> guard(target_modules.GetMutex());

  size_t num_modules = target_modules.GetSize();
  ModuleSP dyld_sp(GetDYLDModule());

  for (size_t i = 0; i < num_modules; i++) {
    ModuleSP module_sp = target_modules.GetModuleAtIndexUnlocked(i);

    // Don't remove dyld - else we'll lose our breakpoint notifying us about
    // libraries being re-loaded...
    if (module_sp.get() != nullptr && module_sp.get() != dyld_sp.get()) {
      UnloadSections(module_sp);
      unloaded_modules_list.Append(module_sp);
    }
  }

  if (unloaded_modules_list.GetSize() != 0) {
    if (log) {
      log->PutCString("Unloaded:");
      unloaded_modules_list.LogUUIDAndPaths(
          log, "DynamicLoaderDarwin::UnloadAllImages");
    }
    target.GetImages().Remove(unloaded_modules_list);
    m_dyld_image_infos.clear();
    m_dyld_image_infos_stop_id = m_process->GetStopID();
  }
}

//----------------------------------------------------------------------
// Update the load addresses for all segments in MODULE using the updated INFO
// that is passed in.
//----------------------------------------------------------------------
bool DynamicLoaderDarwin::UpdateImageLoadAddress(Module *module,
                                                 ImageInfo &info) {
  bool changed = false;
  if (module) {
    ObjectFile *image_object_file = module->GetObjectFile();
    if (image_object_file) {
      SectionList *section_list = image_object_file->GetSectionList();
      if (section_list) {
        std::vector<uint32_t> inaccessible_segment_indexes;
        // We now know the slide amount, so go through all sections and update
        // the load addresses with the correct values.
        const size_t num_segments = info.segments.size();
        for (size_t i = 0; i < num_segments; ++i) {
          // Only load a segment if it has protections. Things like __PAGEZERO
          // don't have any protections, and they shouldn't be slid
          SectionSP section_sp(
              section_list->FindSectionByName(info.segments[i].name));

          if (info.segments[i].maxprot == 0) {
            inaccessible_segment_indexes.push_back(i);
          } else {
            const addr_t new_section_load_addr =
                info.segments[i].vmaddr + info.slide;
            static ConstString g_section_name_LINKEDIT("__LINKEDIT");

            if (section_sp) {
              // __LINKEDIT sections from files in the shared cache can overlap
              // so check to see what the segment name is and pass "false" so
              // we don't warn of overlapping "Section" objects, and "true" for
              // all other sections.
              const bool warn_multiple =
                  section_sp->GetName() != g_section_name_LINKEDIT;

              changed = m_process->GetTarget().SetSectionLoadAddress(
                  section_sp, new_section_load_addr, warn_multiple);
            } else {
              Host::SystemLog(
                  Host::eSystemLogWarning,
                  "warning: unable to find and load segment named '%s' at "
                  "0x%" PRIx64 " in '%s' in macosx dynamic loader plug-in.\n",
                  info.segments[i].name.AsCString("<invalid>"),
                  (uint64_t)new_section_load_addr,
                  image_object_file->GetFileSpec().GetPath().c_str());
            }
          }
        }

        // If the loaded the file (it changed) and we have segments that are
        // not readable or writeable, add them to the invalid memory region
        // cache for the process. This will typically only be the __PAGEZERO
        // segment in the main executable. We might be able to apply this more
        // generally to more sections that have no protections in the future,
        // but for now we are going to just do __PAGEZERO.
        if (changed && !inaccessible_segment_indexes.empty()) {
          for (uint32_t i = 0; i < inaccessible_segment_indexes.size(); ++i) {
            const uint32_t seg_idx = inaccessible_segment_indexes[i];
            SectionSP section_sp(
                section_list->FindSectionByName(info.segments[seg_idx].name));

            if (section_sp) {
              static ConstString g_pagezero_section_name("__PAGEZERO");
              if (g_pagezero_section_name == section_sp->GetName()) {
                // __PAGEZERO never slides...
                const lldb::addr_t vmaddr = info.segments[seg_idx].vmaddr;
                const lldb::addr_t vmsize = info.segments[seg_idx].vmsize;
                Process::LoadRange pagezero_range(vmaddr, vmsize);
                m_process->AddInvalidMemoryRegion(pagezero_range);
              }
            }
          }
        }
      }
    }
  }
  // We might have an in memory image that was loaded as soon as it was created
  if (info.load_stop_id == m_process->GetStopID())
    changed = true;
  else if (changed) {
    // Update the stop ID when this library was updated
    info.load_stop_id = m_process->GetStopID();
  }
  return changed;
}

//----------------------------------------------------------------------
// Unload the segments in MODULE using the INFO that is passed in.
//----------------------------------------------------------------------
bool DynamicLoaderDarwin::UnloadModuleSections(Module *module,
                                               ImageInfo &info) {
  bool changed = false;
  if (module) {
    ObjectFile *image_object_file = module->GetObjectFile();
    if (image_object_file) {
      SectionList *section_list = image_object_file->GetSectionList();
      if (section_list) {
        const size_t num_segments = info.segments.size();
        for (size_t i = 0; i < num_segments; ++i) {
          SectionSP section_sp(
              section_list->FindSectionByName(info.segments[i].name));
          if (section_sp) {
            const addr_t old_section_load_addr =
                info.segments[i].vmaddr + info.slide;
            if (m_process->GetTarget().SetSectionUnloaded(
                    section_sp, old_section_load_addr))
              changed = true;
          } else {
            Host::SystemLog(Host::eSystemLogWarning,
                            "warning: unable to find and unload segment named "
                            "'%s' in '%s' in macosx dynamic loader plug-in.\n",
                            info.segments[i].name.AsCString("<invalid>"),
                            image_object_file->GetFileSpec().GetPath().c_str());
          }
        }
      }
    }
  }
  return changed;
}

// Given a JSON dictionary (from debugserver, most likely) of binary images
// loaded in the inferior process, add the images to the ImageInfo collection.

bool DynamicLoaderDarwin::JSONImageInformationIntoImageInfo(
    StructuredData::ObjectSP image_details,
    ImageInfo::collection &image_infos) {
  StructuredData::ObjectSP images_sp =
      image_details->GetAsDictionary()->GetValueForKey("images");
  if (images_sp.get() == nullptr)
    return false;

  image_infos.resize(images_sp->GetAsArray()->GetSize());

  for (size_t i = 0; i < image_infos.size(); i++) {
    StructuredData::ObjectSP image_sp =
        images_sp->GetAsArray()->GetItemAtIndex(i);
    if (image_sp.get() == nullptr || image_sp->GetAsDictionary() == nullptr)
      return false;
    StructuredData::Dictionary *image = image_sp->GetAsDictionary();
    if (image->HasKey("load_address") == false ||
        image->HasKey("pathname") == false ||
        image->HasKey("mod_date") == false ||
        image->HasKey("mach_header") == false ||
        image->GetValueForKey("mach_header")->GetAsDictionary() == nullptr ||
        image->HasKey("segments") == false ||
        image->GetValueForKey("segments")->GetAsArray() == nullptr ||
        image->HasKey("uuid") == false) {
      return false;
    }
    image_infos[i].address =
        image->GetValueForKey("load_address")->GetAsInteger()->GetValue();
    image_infos[i].mod_date =
        image->GetValueForKey("mod_date")->GetAsInteger()->GetValue();
    image_infos[i].file_spec.SetFile(
        image->GetValueForKey("pathname")->GetAsString()->GetValue(), false,
        FileSpec::Style::native);

    StructuredData::Dictionary *mh =
        image->GetValueForKey("mach_header")->GetAsDictionary();
    image_infos[i].header.magic =
        mh->GetValueForKey("magic")->GetAsInteger()->GetValue();
    image_infos[i].header.cputype =
        mh->GetValueForKey("cputype")->GetAsInteger()->GetValue();
    image_infos[i].header.cpusubtype =
        mh->GetValueForKey("cpusubtype")->GetAsInteger()->GetValue();
    image_infos[i].header.filetype =
        mh->GetValueForKey("filetype")->GetAsInteger()->GetValue();

    if (image->HasKey("min_version_os_name")) {
      std::string os_name = image->GetValueForKey("min_version_os_name")
                                ->GetAsString()
                                ->GetValue();
      if (os_name == "macosx")
        image_infos[i].os_type = llvm::Triple::MacOSX;
      else if (os_name == "ios" || os_name == "iphoneos")
        image_infos[i].os_type = llvm::Triple::IOS;
      else if (os_name == "tvos")
        image_infos[i].os_type = llvm::Triple::TvOS;
      else if (os_name == "watchos")
        image_infos[i].os_type = llvm::Triple::WatchOS;
    }
    if (image->HasKey("min_version_os_sdk")) {
      image_infos[i].min_version_os_sdk =
          image->GetValueForKey("min_version_os_sdk")
              ->GetAsString()
              ->GetValue();
    }

    // Fields that aren't used by DynamicLoaderDarwin so debugserver doesn't
    // currently send them in the reply.

    if (mh->HasKey("flags"))
      image_infos[i].header.flags =
          mh->GetValueForKey("flags")->GetAsInteger()->GetValue();
    else
      image_infos[i].header.flags = 0;

    if (mh->HasKey("ncmds"))
      image_infos[i].header.ncmds =
          mh->GetValueForKey("ncmds")->GetAsInteger()->GetValue();
    else
      image_infos[i].header.ncmds = 0;

    if (mh->HasKey("sizeofcmds"))
      image_infos[i].header.sizeofcmds =
          mh->GetValueForKey("sizeofcmds")->GetAsInteger()->GetValue();
    else
      image_infos[i].header.sizeofcmds = 0;

    StructuredData::Array *segments =
        image->GetValueForKey("segments")->GetAsArray();
    uint32_t segcount = segments->GetSize();
    for (size_t j = 0; j < segcount; j++) {
      Segment segment;
      StructuredData::Dictionary *seg =
          segments->GetItemAtIndex(j)->GetAsDictionary();
      segment.name =
          ConstString(seg->GetValueForKey("name")->GetAsString()->GetValue());
      segment.vmaddr =
          seg->GetValueForKey("vmaddr")->GetAsInteger()->GetValue();
      segment.vmsize =
          seg->GetValueForKey("vmsize")->GetAsInteger()->GetValue();
      segment.fileoff =
          seg->GetValueForKey("fileoff")->GetAsInteger()->GetValue();
      segment.filesize =
          seg->GetValueForKey("filesize")->GetAsInteger()->GetValue();
      segment.maxprot =
          seg->GetValueForKey("maxprot")->GetAsInteger()->GetValue();

      // Fields that aren't used by DynamicLoaderDarwin so debugserver doesn't
      // currently send them in the reply.

      if (seg->HasKey("initprot"))
        segment.initprot =
            seg->GetValueForKey("initprot")->GetAsInteger()->GetValue();
      else
        segment.initprot = 0;

      if (seg->HasKey("flags"))
        segment.flags =
            seg->GetValueForKey("flags")->GetAsInteger()->GetValue();
      else
        segment.flags = 0;

      if (seg->HasKey("nsects"))
        segment.nsects =
            seg->GetValueForKey("nsects")->GetAsInteger()->GetValue();
      else
        segment.nsects = 0;

      image_infos[i].segments.push_back(segment);
    }

    image_infos[i].uuid.SetFromStringRef(
        image->GetValueForKey("uuid")->GetAsString()->GetValue());

    // All sections listed in the dyld image info structure will all either be
    // fixed up already, or they will all be off by a single slide amount that
    // is determined by finding the first segment that is at file offset zero
    // which also has bytes (a file size that is greater than zero) in the
    // object file.

    // Determine the slide amount (if any)
    const size_t num_sections = image_infos[i].segments.size();
    for (size_t k = 0; k < num_sections; ++k) {
      // Iterate through the object file sections to find the first section
      // that starts of file offset zero and that has bytes in the file...
      if ((image_infos[i].segments[k].fileoff == 0 &&
           image_infos[i].segments[k].filesize > 0) ||
          (image_infos[i].segments[k].name == ConstString("__TEXT"))) {
        image_infos[i].slide =
            image_infos[i].address - image_infos[i].segments[k].vmaddr;
        // We have found the slide amount, so we can exit this for loop.
        break;
      }
    }
  }

  return true;
}

void DynamicLoaderDarwin::UpdateSpecialBinariesFromNewImageInfos(
    ImageInfo::collection &image_infos) {
  uint32_t exe_idx = UINT32_MAX;
  uint32_t dyld_idx = UINT32_MAX;
  Target &target = m_process->GetTarget();
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  ConstString g_dyld_sim_filename("dyld_sim");

  ArchSpec target_arch = target.GetArchitecture();
  const size_t image_infos_size = image_infos.size();
  for (size_t i = 0; i < image_infos_size; i++) {
    if (image_infos[i].header.filetype == llvm::MachO::MH_DYLINKER) {
      // In a "simulator" process (an x86 process that is ios/tvos/watchos) we
      // will have two dyld modules -- a "dyld" that we want to keep track of,
      // and a "dyld_sim" which we don't need to keep track of here. If the
      // target is an x86 system and the OS of the dyld binary is
      // ios/tvos/watchos, then we are looking at dyld_sym.

      // debugserver has only recently (late 2016) started sending up the os
      // type for each binary it sees -- so if we don't have an os type, use a
      // filename check as our next best guess.
      if (image_infos[i].os_type == llvm::Triple::OSType::UnknownOS) {
        if (image_infos[i].file_spec.GetFilename() != g_dyld_sim_filename) {
          dyld_idx = i;
        }
      } else if (target_arch.GetTriple().getArch() == llvm::Triple::x86 ||
                 target_arch.GetTriple().getArch() == llvm::Triple::x86_64) {
        if (image_infos[i].os_type != llvm::Triple::OSType::IOS &&
            image_infos[i].os_type != llvm::Triple::TvOS &&
            image_infos[i].os_type != llvm::Triple::WatchOS) {
          dyld_idx = i;
        }
      }
      else {
        // catch-all for any other environment -- trust that dyld is actually
        // dyld
        dyld_idx = i;
      }
    } else if (image_infos[i].header.filetype == llvm::MachO::MH_EXECUTE) {
      exe_idx = i;
    }
  }

  if (exe_idx != UINT32_MAX) {
    const bool can_create = true;
    ModuleSP exe_module_sp(
        FindTargetModuleForImageInfo(image_infos[exe_idx], can_create, NULL));
    if (exe_module_sp) {
      if (log)
        log->Printf("Found executable module: %s",
                    exe_module_sp->GetFileSpec().GetPath().c_str());
      target.GetImages().AppendIfNeeded(exe_module_sp);
      UpdateImageLoadAddress(exe_module_sp.get(), image_infos[exe_idx]);
      if (exe_module_sp.get() != target.GetExecutableModulePointer()) {
        const bool get_dependent_images = false;
        target.SetExecutableModule(exe_module_sp, get_dependent_images);
      }
    }
  }

  if (dyld_idx != UINT32_MAX) {
    const bool can_create = true;
    ModuleSP dyld_sp =
        FindTargetModuleForImageInfo(image_infos[dyld_idx], can_create, NULL);
    if (dyld_sp.get()) {
      if (log)
        log->Printf("Found dyld module: %s",
                    dyld_sp->GetFileSpec().GetPath().c_str());
      target.GetImages().AppendIfNeeded(dyld_sp);
      UpdateImageLoadAddress(dyld_sp.get(), image_infos[dyld_idx]);
      SetDYLDModule(dyld_sp);
    }
  }
}

void DynamicLoaderDarwin::UpdateDYLDImageInfoFromNewImageInfo(
    ImageInfo &image_info) {
  if (image_info.header.filetype == llvm::MachO::MH_DYLINKER) {
    const bool can_create = true;
    ModuleSP dyld_sp =
        FindTargetModuleForImageInfo(image_info, can_create, NULL);
    if (dyld_sp.get()) {
      Target &target = m_process->GetTarget();
      target.GetImages().AppendIfNeeded(dyld_sp);
      UpdateImageLoadAddress(dyld_sp.get(), image_info);
      SetDYLDModule(dyld_sp);
    }
  }
}

void DynamicLoaderDarwin::SetDYLDModule(lldb::ModuleSP &dyld_module_sp) {
  m_dyld_module_wp = dyld_module_sp;
}

ModuleSP DynamicLoaderDarwin::GetDYLDModule() {
  ModuleSP dyld_sp(m_dyld_module_wp.lock());
  return dyld_sp;
}

bool DynamicLoaderDarwin::AddModulesUsingImageInfos(
    ImageInfo::collection &image_infos) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  // Now add these images to the main list.
  ModuleList loaded_module_list;
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  Target &target = m_process->GetTarget();
  ModuleList &target_images = target.GetImages();

  for (uint32_t idx = 0; idx < image_infos.size(); ++idx) {
    if (log) {
      log->Printf("Adding new image at address=0x%16.16" PRIx64 ".",
                  image_infos[idx].address);
      image_infos[idx].PutToLog(log);
    }

    m_dyld_image_infos.push_back(image_infos[idx]);

    ModuleSP image_module_sp(
        FindTargetModuleForImageInfo(image_infos[idx], true, NULL));

    if (image_module_sp) {
      ObjectFile *objfile = image_module_sp->GetObjectFile();
      if (objfile) {
        SectionList *sections = objfile->GetSectionList();
        if (sections) {
          ConstString commpage_dbstr("__commpage");
          Section *commpage_section =
              sections->FindSectionByName(commpage_dbstr).get();
          if (commpage_section) {
            ModuleSpec module_spec(objfile->GetFileSpec(),
                                   image_infos[idx].GetArchitecture());
            module_spec.GetObjectName() = commpage_dbstr;
            ModuleSP commpage_image_module_sp(
                target_images.FindFirstModule(module_spec));
            if (!commpage_image_module_sp) {
              module_spec.SetObjectOffset(objfile->GetFileOffset() +
                                          commpage_section->GetFileOffset());
              module_spec.SetObjectSize(objfile->GetByteSize());
              commpage_image_module_sp = target.GetSharedModule(module_spec);
              if (!commpage_image_module_sp ||
                  commpage_image_module_sp->GetObjectFile() == NULL) {
                commpage_image_module_sp = m_process->ReadModuleFromMemory(
                    image_infos[idx].file_spec, image_infos[idx].address);
                // Always load a memory image right away in the target in case
                // we end up trying to read the symbol table from memory... The
                // __LINKEDIT will need to be mapped so we can figure out where
                // the symbol table bits are...
                bool changed = false;
                UpdateImageLoadAddress(commpage_image_module_sp.get(),
                                       image_infos[idx]);
                target.GetImages().Append(commpage_image_module_sp);
                if (changed) {
                  image_infos[idx].load_stop_id = m_process->GetStopID();
                  loaded_module_list.AppendIfNeeded(commpage_image_module_sp);
                }
              }
            }
          }
        }
      }

      // UpdateImageLoadAddress will return true if any segments change load
      // address. We need to check this so we don't mention that all loaded
      // shared libraries are newly loaded each time we hit out dyld breakpoint
      // since dyld will list all shared libraries each time.
      if (UpdateImageLoadAddress(image_module_sp.get(), image_infos[idx])) {
        target_images.AppendIfNeeded(image_module_sp);
        loaded_module_list.AppendIfNeeded(image_module_sp);
      }
    }
  }

  if (loaded_module_list.GetSize() > 0) {
    if (log)
      loaded_module_list.LogUUIDAndPaths(log,
                                         "DynamicLoaderDarwin::ModulesDidLoad");
    m_process->GetTarget().ModulesDidLoad(loaded_module_list);
  }
  return true;
}

//----------------------------------------------------------------------
// On Mac OS X libobjc (the Objective-C runtime) has several critical dispatch
// functions written in hand-written assembly, and also have hand-written
// unwind information in the eh_frame section.  Normally we prefer analyzing
// the assembly instructions of a currently executing frame to unwind from that
// frame -- but on hand-written functions this profiling can fail.  We should
// use the eh_frame instructions for these functions all the time.
//
// As an aside, it would be better if the eh_frame entries had a flag (or were
// extensible so they could have an Apple-specific flag) which indicates that
// the instructions are asynchronous -- accurate at every instruction, instead
// of our normal default assumption that they are not.
//----------------------------------------------------------------------

bool DynamicLoaderDarwin::AlwaysRelyOnEHUnwindInfo(SymbolContext &sym_ctx) {
  ModuleSP module_sp;
  if (sym_ctx.symbol) {
    module_sp = sym_ctx.symbol->GetAddressRef().GetModule();
  }
  if (module_sp.get() == NULL && sym_ctx.function) {
    module_sp =
        sym_ctx.function->GetAddressRange().GetBaseAddress().GetModule();
  }
  if (module_sp.get() == NULL)
    return false;

  ObjCLanguageRuntime *objc_runtime = m_process->GetObjCLanguageRuntime();
  if (objc_runtime != NULL && objc_runtime->IsModuleObjCLibrary(module_sp)) {
    return true;
  }

  return false;
}

//----------------------------------------------------------------------
// Dump a Segment to the file handle provided.
//----------------------------------------------------------------------
void DynamicLoaderDarwin::Segment::PutToLog(Log *log,
                                            lldb::addr_t slide) const {
  if (log) {
    if (slide == 0)
      log->Printf("\t\t%16s [0x%16.16" PRIx64 " - 0x%16.16" PRIx64 ")",
                  name.AsCString(""), vmaddr + slide, vmaddr + slide + vmsize);
    else
      log->Printf("\t\t%16s [0x%16.16" PRIx64 " - 0x%16.16" PRIx64
                  ") slide = 0x%" PRIx64,
                  name.AsCString(""), vmaddr + slide, vmaddr + slide + vmsize,
                  slide);
  }
}

const DynamicLoaderDarwin::Segment *
DynamicLoaderDarwin::ImageInfo::FindSegment(const ConstString &name) const {
  const size_t num_segments = segments.size();
  for (size_t i = 0; i < num_segments; ++i) {
    if (segments[i].name == name)
      return &segments[i];
  }
  return NULL;
}

//----------------------------------------------------------------------
// Dump an image info structure to the file handle provided.
//----------------------------------------------------------------------
void DynamicLoaderDarwin::ImageInfo::PutToLog(Log *log) const {
  if (!log)
    return;
  if (address == LLDB_INVALID_ADDRESS) {
    LLDB_LOG(log, "modtime={0:x+8} uuid={1} path='{2}' (UNLOADED)", mod_date,
             uuid.GetAsString(), file_spec.GetPath());
  } else {
    LLDB_LOG(log, "address={0:x+16} modtime={1:x+8} uuid={2} path='{3}'",
             address, mod_date, uuid.GetAsString(), file_spec.GetPath());
    for (uint32_t i = 0; i < segments.size(); ++i)
      segments[i].PutToLog(log, slide);
  }
}

void DynamicLoaderDarwin::PrivateInitialize(Process *process) {
  DEBUG_PRINTF("DynamicLoaderDarwin::%s() process state = %s\n", __FUNCTION__,
               StateAsCString(m_process->GetState()));
  Clear(true);
  m_process = process;
  m_process->GetTarget().ClearAllLoadedSections();
}

//----------------------------------------------------------------------
// Member function that gets called when the process state changes.
//----------------------------------------------------------------------
void DynamicLoaderDarwin::PrivateProcessStateChanged(Process *process,
                                                     StateType state) {
  DEBUG_PRINTF("DynamicLoaderDarwin::%s(%s)\n", __FUNCTION__,
               StateAsCString(state));
  switch (state) {
  case eStateConnected:
  case eStateAttaching:
  case eStateLaunching:
  case eStateInvalid:
  case eStateUnloaded:
  case eStateExited:
  case eStateDetached:
    Clear(false);
    break;

  case eStateStopped:
    // Keep trying find dyld and set our notification breakpoint each time we
    // stop until we succeed
    if (!DidSetNotificationBreakpoint() && m_process->IsAlive()) {
      if (NeedToDoInitialImageFetch())
        DoInitialImageFetch();

      SetNotificationBreakpoint();
    }
    break;

  case eStateRunning:
  case eStateStepping:
  case eStateCrashed:
  case eStateSuspended:
    break;
  }
}

ThreadPlanSP
DynamicLoaderDarwin::GetStepThroughTrampolinePlan(Thread &thread,
                                                  bool stop_others) {
  ThreadPlanSP thread_plan_sp;
  StackFrame *current_frame = thread.GetStackFrameAtIndex(0).get();
  const SymbolContext &current_context =
      current_frame->GetSymbolContext(eSymbolContextSymbol);
  Symbol *current_symbol = current_context.symbol;
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_STEP));
  TargetSP target_sp(thread.CalculateTarget());

  if (current_symbol != NULL) {
    std::vector<Address> addresses;

    if (current_symbol->IsTrampoline()) {
      const ConstString &trampoline_name = current_symbol->GetMangled().GetName(
          current_symbol->GetLanguage(), Mangled::ePreferMangled);

      if (trampoline_name) {
        const ModuleList &images = target_sp->GetImages();

        SymbolContextList code_symbols;
        images.FindSymbolsWithNameAndType(trampoline_name, eSymbolTypeCode,
                                          code_symbols);
        size_t num_code_symbols = code_symbols.GetSize();

        if (num_code_symbols > 0) {
          for (uint32_t i = 0; i < num_code_symbols; i++) {
            SymbolContext context;
            AddressRange addr_range;
            if (code_symbols.GetContextAtIndex(i, context)) {
              context.GetAddressRange(eSymbolContextEverything, 0, false,
                                      addr_range);
              addresses.push_back(addr_range.GetBaseAddress());
              if (log) {
                addr_t load_addr =
                    addr_range.GetBaseAddress().GetLoadAddress(target_sp.get());

                log->Printf("Found a trampoline target symbol at 0x%" PRIx64
                            ".",
                            load_addr);
              }
            }
          }
        }

        SymbolContextList reexported_symbols;
        images.FindSymbolsWithNameAndType(
            trampoline_name, eSymbolTypeReExported, reexported_symbols);
        size_t num_reexported_symbols = reexported_symbols.GetSize();
        if (num_reexported_symbols > 0) {
          for (uint32_t i = 0; i < num_reexported_symbols; i++) {
            SymbolContext context;
            if (reexported_symbols.GetContextAtIndex(i, context)) {
              if (context.symbol) {
                Symbol *actual_symbol =
                    context.symbol->ResolveReExportedSymbol(*target_sp.get());
                if (actual_symbol) {
                  const Address actual_symbol_addr =
                      actual_symbol->GetAddress();
                  if (actual_symbol_addr.IsValid()) {
                    addresses.push_back(actual_symbol_addr);
                    if (log) {
                      lldb::addr_t load_addr =
                          actual_symbol_addr.GetLoadAddress(target_sp.get());
                      log->Printf(
                          "Found a re-exported symbol: %s at 0x%" PRIx64 ".",
                          actual_symbol->GetName().GetCString(), load_addr);
                    }
                  }
                }
              }
            }
          }
        }

        SymbolContextList indirect_symbols;
        images.FindSymbolsWithNameAndType(trampoline_name, eSymbolTypeResolver,
                                          indirect_symbols);
        size_t num_indirect_symbols = indirect_symbols.GetSize();
        if (num_indirect_symbols > 0) {
          for (uint32_t i = 0; i < num_indirect_symbols; i++) {
            SymbolContext context;
            AddressRange addr_range;
            if (indirect_symbols.GetContextAtIndex(i, context)) {
              context.GetAddressRange(eSymbolContextEverything, 0, false,
                                      addr_range);
              addresses.push_back(addr_range.GetBaseAddress());
              if (log) {
                addr_t load_addr =
                    addr_range.GetBaseAddress().GetLoadAddress(target_sp.get());

                log->Printf("Found an indirect target symbol at 0x%" PRIx64 ".",
                            load_addr);
              }
            }
          }
        }
      }
    } else if (current_symbol->GetType() == eSymbolTypeReExported) {
      // I am not sure we could ever end up stopped AT a re-exported symbol.
      // But just in case:

      const Symbol *actual_symbol =
          current_symbol->ResolveReExportedSymbol(*(target_sp.get()));
      if (actual_symbol) {
        Address target_addr(actual_symbol->GetAddress());
        if (target_addr.IsValid()) {
          if (log)
            log->Printf(
                "Found a re-exported symbol: %s pointing to: %s at 0x%" PRIx64
                ".",
                current_symbol->GetName().GetCString(),
                actual_symbol->GetName().GetCString(),
                target_addr.GetLoadAddress(target_sp.get()));
          addresses.push_back(target_addr.GetLoadAddress(target_sp.get()));
        }
      }
    }

    if (addresses.size() > 0) {
      // First check whether any of the addresses point to Indirect symbols,
      // and if they do, resolve them:
      std::vector<lldb::addr_t> load_addrs;
      for (Address address : addresses) {
        Symbol *symbol = address.CalculateSymbolContextSymbol();
        if (symbol && symbol->IsIndirect()) {
          Status error;
          Address symbol_address = symbol->GetAddress();
          addr_t resolved_addr = thread.GetProcess()->ResolveIndirectFunction(
              &symbol_address, error);
          if (error.Success()) {
            load_addrs.push_back(resolved_addr);
            if (log)
              log->Printf("ResolveIndirectFunction found resolved target for "
                          "%s at 0x%" PRIx64 ".",
                          symbol->GetName().GetCString(), resolved_addr);
          }
        } else {
          load_addrs.push_back(address.GetLoadAddress(target_sp.get()));
        }
      }
      thread_plan_sp.reset(
          new ThreadPlanRunToAddress(thread, load_addrs, stop_others));
    }
  } else {
    if (log)
      log->Printf("Could not find symbol for step through.");
  }

  return thread_plan_sp;
}

size_t DynamicLoaderDarwin::FindEquivalentSymbols(
    lldb_private::Symbol *original_symbol, lldb_private::ModuleList &images,
    lldb_private::SymbolContextList &equivalent_symbols) {
  const ConstString &trampoline_name = original_symbol->GetMangled().GetName(
      original_symbol->GetLanguage(), Mangled::ePreferMangled);
  if (!trampoline_name)
    return 0;

  size_t initial_size = equivalent_symbols.GetSize();

  static const char *resolver_name_regex = "(_gc|_non_gc|\\$[A-Za-z0-9\\$]+)$";
  std::string equivalent_regex_buf("^");
  equivalent_regex_buf.append(trampoline_name.GetCString());
  equivalent_regex_buf.append(resolver_name_regex);

  RegularExpression equivalent_name_regex(equivalent_regex_buf);
  const bool append = true;
  images.FindSymbolsMatchingRegExAndType(equivalent_name_regex, eSymbolTypeCode,
                                         equivalent_symbols, append);

  return equivalent_symbols.GetSize() - initial_size;
}

lldb::ModuleSP DynamicLoaderDarwin::GetPThreadLibraryModule() {
  ModuleSP module_sp = m_libpthread_module_wp.lock();
  if (!module_sp) {
    SymbolContextList sc_list;
    ModuleSpec module_spec;
    module_spec.GetFileSpec().GetFilename().SetCString(
        "libsystem_pthread.dylib");
    ModuleList module_list;
    if (m_process->GetTarget().GetImages().FindModules(module_spec,
                                                       module_list)) {
      if (module_list.GetSize() == 1) {
        module_sp = module_list.GetModuleAtIndex(0);
        if (module_sp)
          m_libpthread_module_wp = module_sp;
      }
    }
  }
  return module_sp;
}

Address DynamicLoaderDarwin::GetPthreadSetSpecificAddress() {
  if (!m_pthread_getspecific_addr.IsValid()) {
    ModuleSP module_sp = GetPThreadLibraryModule();
    if (module_sp) {
      lldb_private::SymbolContextList sc_list;
      module_sp->FindSymbolsWithNameAndType(ConstString("pthread_getspecific"),
                                            eSymbolTypeCode, sc_list);
      SymbolContext sc;
      if (sc_list.GetContextAtIndex(0, sc)) {
        if (sc.symbol)
          m_pthread_getspecific_addr = sc.symbol->GetAddress();
      }
    }
  }
  return m_pthread_getspecific_addr;
}

lldb::addr_t
DynamicLoaderDarwin::GetThreadLocalData(const lldb::ModuleSP module_sp,
                                        const lldb::ThreadSP thread_sp,
                                        lldb::addr_t tls_file_addr) {
  if (!thread_sp || !module_sp)
    return LLDB_INVALID_ADDRESS;

  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  const uint32_t addr_size = m_process->GetAddressByteSize();
  uint8_t buf[sizeof(lldb::addr_t) * 3];

  lldb_private::Address tls_addr;
  if (module_sp->ResolveFileAddress(tls_file_addr, tls_addr)) {
    Status error;
    const size_t tsl_data_size = addr_size * 3;
    Target &target = m_process->GetTarget();
    if (target.ReadMemory(tls_addr, false, buf, tsl_data_size, error) ==
        tsl_data_size) {
      const ByteOrder byte_order = m_process->GetByteOrder();
      DataExtractor data(buf, sizeof(buf), byte_order, addr_size);
      lldb::offset_t offset = addr_size; // Skip the first pointer
      const lldb::addr_t pthread_key = data.GetAddress(&offset);
      const lldb::addr_t tls_offset = data.GetAddress(&offset);
      if (pthread_key != 0) {
        // First check to see if we have already figured out the location of
        // TLS data for the pthread_key on a specific thread yet. If we have we
        // can re-use it since its location will not change unless the process
        // execs.
        const tid_t tid = thread_sp->GetID();
        auto tid_pos = m_tid_to_tls_map.find(tid);
        if (tid_pos != m_tid_to_tls_map.end()) {
          auto tls_pos = tid_pos->second.find(pthread_key);
          if (tls_pos != tid_pos->second.end()) {
            return tls_pos->second + tls_offset;
          }
        }
        StackFrameSP frame_sp = thread_sp->GetStackFrameAtIndex(0);
        if (frame_sp) {
          ClangASTContext *clang_ast_context =
              target.GetScratchClangASTContext();

          if (!clang_ast_context)
            return LLDB_INVALID_ADDRESS;

          CompilerType clang_void_ptr_type =
              clang_ast_context->GetBasicType(eBasicTypeVoid).GetPointerType();
          Address pthread_getspecific_addr = GetPthreadSetSpecificAddress();
          if (pthread_getspecific_addr.IsValid()) {
            EvaluateExpressionOptions options;

            lldb::ThreadPlanSP thread_plan_sp(new ThreadPlanCallFunction(
                *thread_sp, pthread_getspecific_addr, clang_void_ptr_type,
                llvm::ArrayRef<lldb::addr_t>(pthread_key), options));

            DiagnosticManager execution_errors;
            ExecutionContext exe_ctx(thread_sp);
            lldb::ExpressionResults results = m_process->RunThreadPlan(
                exe_ctx, thread_plan_sp, options, execution_errors);

            if (results == lldb::eExpressionCompleted) {
              lldb::ValueObjectSP result_valobj_sp =
                  thread_plan_sp->GetReturnValueObject();
              if (result_valobj_sp) {
                const lldb::addr_t pthread_key_data =
                    result_valobj_sp->GetValueAsUnsigned(0);
                if (pthread_key_data) {
                  m_tid_to_tls_map[tid].insert(
                      std::make_pair(pthread_key, pthread_key_data));
                  return pthread_key_data + tls_offset;
                }
              }
            }
          }
        }
      }
    }
  }
  return LLDB_INVALID_ADDRESS;
}

bool DynamicLoaderDarwin::UseDYLDSPI(Process *process) {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  bool use_new_spi_interface = false;

  llvm::VersionTuple version = process->GetHostOSVersion();
  if (!version.empty()) {
    const llvm::Triple::OSType os_type =
        process->GetTarget().GetArchitecture().GetTriple().getOS();

    // macOS 10.12 and newer
    if (os_type == llvm::Triple::MacOSX &&
        version >= llvm::VersionTuple(10, 12))
      use_new_spi_interface = true;

    // iOS 10 and newer
    if (os_type == llvm::Triple::IOS && version >= llvm::VersionTuple(10))
      use_new_spi_interface = true;

    // tvOS 10 and newer
    if (os_type == llvm::Triple::TvOS && version >= llvm::VersionTuple(10))
      use_new_spi_interface = true;

    // watchOS 3 and newer
    if (os_type == llvm::Triple::WatchOS && version >= llvm::VersionTuple(3))
      use_new_spi_interface = true;
  }

  if (log) {
    if (use_new_spi_interface)
      log->Printf(
          "DynamicLoaderDarwin::UseDYLDSPI: Use new DynamicLoader plugin");
    else
      log->Printf(
          "DynamicLoaderDarwin::UseDYLDSPI: Use old DynamicLoader plugin");
  }
  return use_new_spi_interface;
}
