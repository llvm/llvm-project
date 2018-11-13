//===-- ObjectContainerUniversalMachO.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjectContainerUniversalMachO_h_
#define liblldb_ObjectContainerUniversalMachO_h_

#include "lldb/Host/SafeMachO.h"
#include "lldb/Symbol/ObjectContainer.h"
#include "lldb/Utility/FileSpec.h"

class ObjectContainerUniversalMachO : public lldb_private::ObjectContainer {
public:
  ObjectContainerUniversalMachO(const lldb::ModuleSP &module_sp,
                                lldb::DataBufferSP &data_sp,
                                lldb::offset_t data_offset,
                                const lldb_private::FileSpec *file,
                                lldb::offset_t offset, lldb::offset_t length);

  ~ObjectContainerUniversalMachO() override;

  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static lldb_private::ObjectContainer *
  CreateInstance(const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
                 lldb::offset_t data_offset, const lldb_private::FileSpec *file,
                 lldb::offset_t offset, lldb::offset_t length);

  static size_t GetModuleSpecifications(const lldb_private::FileSpec &file,
                                        lldb::DataBufferSP &data_sp,
                                        lldb::offset_t data_offset,
                                        lldb::offset_t file_offset,
                                        lldb::offset_t length,
                                        lldb_private::ModuleSpecList &specs);

  static bool MagicBytesMatch(const lldb_private::DataExtractor &data);

  //------------------------------------------------------------------
  // Member Functions
  //------------------------------------------------------------------
  bool ParseHeader() override;

  void Dump(lldb_private::Stream *s) const override;

  size_t GetNumArchitectures() const override;

  bool GetArchitectureAtIndex(uint32_t cpu_idx,
                              lldb_private::ArchSpec &arch) const override;

  lldb::ObjectFileSP GetObjectFile(const lldb_private::FileSpec *file) override;

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

protected:
  llvm::MachO::fat_header m_header;
  std::vector<llvm::MachO::fat_arch> m_fat_archs;

  static bool ParseHeader(lldb_private::DataExtractor &data,
                          llvm::MachO::fat_header &header,
                          std::vector<llvm::MachO::fat_arch> &fat_archs);
};

#endif // liblldb_ObjectContainerUniversalMachO_h_
