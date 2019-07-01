//===-- NativeProcessELF.cpp ---------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeProcessELF.h"

#include "lldb/Utility/DataExtractor.h"

namespace lldb_private {

llvm::Optional<uint64_t>
NativeProcessELF::GetAuxValue(enum AuxVector::EntryType type) {
  if (m_aux_vector == nullptr) {
    auto buffer_or_error = GetAuxvData();
    if (!buffer_or_error)
      return llvm::None;
    DataExtractor auxv_data(buffer_or_error.get()->getBufferStart(),
                            buffer_or_error.get()->getBufferSize(),
                            GetByteOrder(), GetAddressByteSize());
    m_aux_vector = llvm::make_unique<AuxVector>(auxv_data);
  }

  return m_aux_vector->GetAuxValue(type);
}

lldb::addr_t NativeProcessELF::GetSharedLibraryInfoAddress() {
  if (!m_shared_library_info_addr.hasValue()) {
    if (GetAddressByteSize() == 8)
      m_shared_library_info_addr =
          GetELFImageInfoAddress<llvm::ELF::Elf64_Ehdr, llvm::ELF::Elf64_Phdr,
                                 llvm::ELF::Elf64_Dyn>();
    else
      m_shared_library_info_addr =
          GetELFImageInfoAddress<llvm::ELF::Elf32_Ehdr, llvm::ELF::Elf32_Phdr,
                                 llvm::ELF::Elf32_Dyn>();
  }

  return m_shared_library_info_addr.getValue();
}

template <typename ELF_EHDR, typename ELF_PHDR, typename ELF_DYN>
lldb::addr_t NativeProcessELF::GetELFImageInfoAddress() {
  llvm::Optional<uint64_t> maybe_phdr_addr =
      GetAuxValue(AuxVector::AUXV_AT_PHDR);
  llvm::Optional<uint64_t> maybe_phdr_entry_size =
      GetAuxValue(AuxVector::AUXV_AT_PHENT);
  llvm::Optional<uint64_t> maybe_phdr_num_entries =
      GetAuxValue(AuxVector::AUXV_AT_PHNUM);
  if (!maybe_phdr_addr || !maybe_phdr_entry_size || !maybe_phdr_num_entries)
    return LLDB_INVALID_ADDRESS;
  lldb::addr_t phdr_addr = *maybe_phdr_addr;
  size_t phdr_entry_size = *maybe_phdr_entry_size;
  size_t phdr_num_entries = *maybe_phdr_num_entries;

  // Find the PT_DYNAMIC segment (.dynamic section) in the program header and
  // what the load bias by calculating the difference of the program header
  // load address and its virtual address.
  lldb::offset_t load_bias;
  bool found_load_bias = false;
  lldb::addr_t dynamic_section_addr = 0;
  uint64_t dynamic_section_size = 0;
  bool found_dynamic_section = false;
  ELF_PHDR phdr_entry;
  for (size_t i = 0; i < phdr_num_entries; i++) {
    size_t bytes_read;
    auto error = ReadMemory(phdr_addr + i * phdr_entry_size, &phdr_entry,
                            sizeof(phdr_entry), bytes_read);
    if (!error.Success())
      return LLDB_INVALID_ADDRESS;
    if (phdr_entry.p_type == llvm::ELF::PT_PHDR) {
      load_bias = phdr_addr - phdr_entry.p_vaddr;
      found_load_bias = true;
    }

    if (phdr_entry.p_type == llvm::ELF::PT_DYNAMIC) {
      dynamic_section_addr = phdr_entry.p_vaddr;
      dynamic_section_size = phdr_entry.p_memsz;
      found_dynamic_section = true;
    }
  }

  if (!found_load_bias || !found_dynamic_section)
    return LLDB_INVALID_ADDRESS;

  // Find the DT_DEBUG entry in the .dynamic section
  dynamic_section_addr += load_bias;
  ELF_DYN dynamic_entry;
  size_t dynamic_num_entries = dynamic_section_size / sizeof(dynamic_entry);
  for (size_t i = 0; i < dynamic_num_entries; i++) {
    size_t bytes_read;
    auto error = ReadMemory(dynamic_section_addr + i * sizeof(dynamic_entry),
                            &dynamic_entry, sizeof(dynamic_entry), bytes_read);
    if (!error.Success())
      return LLDB_INVALID_ADDRESS;
    // Return the &DT_DEBUG->d_ptr which points to r_debug which contains the
    // link_map.
    if (dynamic_entry.d_tag == llvm::ELF::DT_DEBUG) {
      return dynamic_section_addr + i * sizeof(dynamic_entry) +
             sizeof(dynamic_entry.d_tag);
    }
  }

  return LLDB_INVALID_ADDRESS;
}

} // namespace lldb_private