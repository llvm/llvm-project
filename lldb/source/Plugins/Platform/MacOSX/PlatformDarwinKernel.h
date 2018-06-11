//===-- PlatformDarwinKernel.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformDarwinKernel_h_
#define liblldb_PlatformDarwinKernel_h_

#include "lldb/Utility/ConstString.h"

#if defined(__APPLE__) // This Plugin uses the Mac-specific
                       // source/Host/macosx/cfcpp utilities

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Utility/FileSpec.h"

#include "llvm/Support/FileSystem.h"

// Project includes
#include "PlatformDarwin.h"

class PlatformDarwinKernel : public PlatformDarwin {
public:
  //------------------------------------------------------------
  // Class Functions
  //------------------------------------------------------------
  static lldb::PlatformSP CreateInstance(bool force,
                                         const lldb_private::ArchSpec *arch);

  static void DebuggerInitialize(lldb_private::Debugger &debugger);

  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetDescriptionStatic();

  //------------------------------------------------------------
  // Class Methods
  //------------------------------------------------------------
  PlatformDarwinKernel(lldb_private::LazyBool is_ios_debug_session);

  virtual ~PlatformDarwinKernel();

  //------------------------------------------------------------
  // lldb_private::PluginInterface functions
  //------------------------------------------------------------
  lldb_private::ConstString GetPluginName() override {
    return GetPluginNameStatic();
  }

  uint32_t GetPluginVersion() override { return 1; }

  //------------------------------------------------------------
  // lldb_private::Platform functions
  //------------------------------------------------------------
  const char *GetDescription() override { return GetDescriptionStatic(); }

  void GetStatus(lldb_private::Stream &strm) override;

  lldb_private::Status
  GetSharedModule(const lldb_private::ModuleSpec &module_spec,
                  lldb_private::Process *process, lldb::ModuleSP &module_sp,
                  const lldb_private::FileSpecList *module_search_paths_ptr,
                  lldb::ModuleSP *old_module_sp_ptr,
                  bool *did_create_ptr) override;

  bool GetSupportedArchitectureAtIndex(uint32_t idx,
                                       lldb_private::ArchSpec &arch) override;

  bool SupportsModules() override { return false; }

  void CalculateTrapHandlerSymbolNames() override;

protected:
  // Map from kext bundle ID ("com.apple.filesystems.exfat") to FileSpec for the
  // kext bundle on
  // the host ("/System/Library/Extensions/exfat.kext/Contents/Info.plist").
  typedef std::multimap<lldb_private::ConstString, lldb_private::FileSpec>
      BundleIDToKextMap;
  typedef BundleIDToKextMap::iterator BundleIDToKextIterator;

  typedef std::vector<lldb_private::FileSpec> KernelBinaryCollection;

  // Array of directories that were searched for kext bundles (used only for
  // reporting to user)
  typedef std::vector<lldb_private::FileSpec> DirectoriesSearchedCollection;
  typedef DirectoriesSearchedCollection::iterator DirectoriesSearchedIterator;

  // Populate m_search_directories and m_search_directories_no_recursing vectors
  // of directories
  void CollectKextAndKernelDirectories();

  void GetUserSpecifiedDirectoriesToSearch();

  static void AddRootSubdirsToSearchPaths(PlatformDarwinKernel *thisp,
                                          const std::string &dir);

  void AddSDKSubdirsToSearchPaths(const std::string &dir);

  static lldb_private::FileSpec::EnumerateDirectoryResult
  FindKDKandSDKDirectoriesInDirectory(void *baton, llvm::sys::fs::file_type ft,
                                      const lldb_private::FileSpec &file_spec);

  void SearchForKextsAndKernelsRecursively();

  static lldb_private::FileSpec::EnumerateDirectoryResult
  GetKernelsAndKextsInDirectoryWithRecursion(
      void *baton, llvm::sys::fs::file_type ft,
      const lldb_private::FileSpec &file_spec);

  static lldb_private::FileSpec::EnumerateDirectoryResult
  GetKernelsAndKextsInDirectoryNoRecursion(
      void *baton, llvm::sys::fs::file_type ft,
      const lldb_private::FileSpec &file_spec);

  static lldb_private::FileSpec::EnumerateDirectoryResult
  GetKernelsAndKextsInDirectoryHelper(void *baton, llvm::sys::fs::file_type ft,
                                      const lldb_private::FileSpec &file_spec,
                                      bool recurse);

  static std::vector<lldb_private::FileSpec>
  SearchForExecutablesRecursively(const lldb_private::ConstString &dir);

  static void AddKextToMap(PlatformDarwinKernel *thisp,
                           const lldb_private::FileSpec &file_spec);

  // Returns true if there is a .dSYM bundle next to the kext, or next to the
  // binary inside the kext.
  static bool
  KextHasdSYMSibling(const lldb_private::FileSpec &kext_bundle_filepath);

  // Returns true if there is a .dSYM bundle next to the kernel
  static bool
  KernelHasdSYMSibling(const lldb_private::FileSpec &kext_bundle_filepath);

  lldb_private::Status
  ExamineKextForMatchingUUID(const lldb_private::FileSpec &kext_bundle_path,
                             const lldb_private::UUID &uuid,
                             const lldb_private::ArchSpec &arch,
                             lldb::ModuleSP &exe_module_sp);

  // Most of the ivars are assembled under FileSpec::EnumerateDirectory calls
  // where the
  // function being called for each file/directory must be static.  We'll pass a
  // this pointer
  // as a baton and access the ivars directly.  Toss-up whether this should just
  // be a struct
  // at this point.

public:
  BundleIDToKextMap m_name_to_kext_path_map_with_dsyms;    // multimap of
                                                           // CFBundleID to
                                                           // FileSpec on local
                                                           // filesystem, kexts
                                                           // with dSYMs next to
                                                           // them
  BundleIDToKextMap m_name_to_kext_path_map_without_dsyms; // multimap of
                                                           // CFBundleID to
                                                           // FileSpec on local
                                                           // filesystem, kexts
                                                           // without dSYMs next
                                                           // to them
  DirectoriesSearchedCollection
      m_search_directories; // list of directories we search for kexts/kernels
  DirectoriesSearchedCollection
      m_search_directories_no_recursing; // list of directories we search for
                                         // kexts/kernels, no recursion
  KernelBinaryCollection m_kernel_binaries_with_dsyms; // list of kernel
                                                       // binaries we found on
                                                       // local filesystem,
                                                       // without dSYMs next to
                                                       // them
  KernelBinaryCollection m_kernel_binaries_without_dsyms; // list of kernel
                                                          // binaries we found
                                                          // on local
                                                          // filesystem, with
                                                          // dSYMs next to them
  lldb_private::LazyBool m_ios_debug_session;

  DISALLOW_COPY_AND_ASSIGN(PlatformDarwinKernel);
};

#else // __APPLE__

// Since DynamicLoaderDarwinKernel is compiled in for all systems, and relies on
// PlatformDarwinKernel for the plug-in name, we compile just the plug-in name
// in
// here to avoid issues. We are tracking an internal bug to resolve this issue
// by
// either not compiling in DynamicLoaderDarwinKernel for non-apple builds, or to
// make
// PlatformDarwinKernel build on all systems. PlatformDarwinKernel is currently
// not
// compiled on other platforms due to the use of the Mac-specific
// source/Host/macosx/cfcpp utilities.

class PlatformDarwinKernel {
public:
  static lldb_private::ConstString GetPluginNameStatic();
};

#endif // __APPLE__

#endif // liblldb_PlatformDarwinKernel_h_
