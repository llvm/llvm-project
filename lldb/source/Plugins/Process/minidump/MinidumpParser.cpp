//===-- MinidumpParser.cpp ---------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "MinidumpParser.h"

// Other libraries and framework includes
// C includes
// C++ includes

using namespace lldb_private;
using namespace minidump;

llvm::Optional<MinidumpParser>
MinidumpParser::Create(const lldb::DataBufferSP &data_buf_sp) {
  if (data_buf_sp->GetByteSize() < sizeof(MinidumpHeader)) {
    return llvm::None;
  }

  llvm::ArrayRef<uint8_t> header_data(data_buf_sp->GetBytes(),
                                      sizeof(MinidumpHeader));
  const MinidumpHeader *header = MinidumpHeader::Parse(header_data);

  if (header == nullptr) {
    return llvm::None;
  }

  lldb::offset_t directory_list_offset = header->stream_directory_rva;
  // check if there is enough data for the parsing of the directory list
  if ((directory_list_offset +
       sizeof(MinidumpDirectory) * header->streams_count) >
      data_buf_sp->GetByteSize()) {
    return llvm::None;
  }

  const MinidumpDirectory *directory = nullptr;
  Error error;
  llvm::ArrayRef<uint8_t> directory_data(
      data_buf_sp->GetBytes() + directory_list_offset,
      sizeof(MinidumpDirectory) * header->streams_count);
  llvm::DenseMap<uint32_t, MinidumpLocationDescriptor> directory_map;

  for (uint32_t i = 0; i < header->streams_count; ++i) {
    error = consumeObject(directory_data, directory);
    if (error.Fail()) {
      return llvm::None;
    }
    directory_map[static_cast<const uint32_t>(directory->stream_type)] =
        directory->location;
  }

  MinidumpParser parser(data_buf_sp, header, directory_map);
  return llvm::Optional<MinidumpParser>(parser);
}

MinidumpParser::MinidumpParser(
    const lldb::DataBufferSP &data_buf_sp, const MinidumpHeader *header,
    const llvm::DenseMap<uint32_t, MinidumpLocationDescriptor> &directory_map)
    : m_data_sp(data_buf_sp), m_header(header), m_directory_map(directory_map) {
}

lldb::offset_t MinidumpParser::GetByteSize() {
  return m_data_sp->GetByteSize();
}

llvm::Optional<llvm::ArrayRef<uint8_t>>
MinidumpParser::GetStream(MinidumpStreamType stream_type) {
  auto iter = m_directory_map.find(static_cast<uint32_t>(stream_type));
  if (iter == m_directory_map.end())
    return llvm::None;

  // check if there is enough data
  if (iter->second.rva + iter->second.data_size > m_data_sp->GetByteSize())
    return llvm::None;

  llvm::ArrayRef<uint8_t> arr_ref(m_data_sp->GetBytes() + iter->second.rva,
                                  iter->second.data_size);
  return llvm::Optional<llvm::ArrayRef<uint8_t>>(arr_ref);
}

llvm::Optional<std::vector<const MinidumpThread *>>
MinidumpParser::GetThreads() {
  llvm::Optional<llvm::ArrayRef<uint8_t>> data =
      GetStream(MinidumpStreamType::ThreadList);

  if (!data)
    return llvm::None;

  return MinidumpThread::ParseThreadList(data.getValue());
}

const MinidumpSystemInfo *MinidumpParser::GetSystemInfo() {
  llvm::Optional<llvm::ArrayRef<uint8_t>> data =
      GetStream(MinidumpStreamType::SystemInfo);

  if (!data)
    return nullptr;

  return MinidumpSystemInfo::Parse(data.getValue());
}

ArchSpec MinidumpParser::GetArchitecture() {
  ArchSpec arch_spec;
  arch_spec.GetTriple().setOS(llvm::Triple::OSType::UnknownOS);
  arch_spec.GetTriple().setVendor(llvm::Triple::VendorType::UnknownVendor);
  arch_spec.GetTriple().setArch(llvm::Triple::ArchType::UnknownArch);

  // TODO should we add the OS type here, or somewhere else ?

  const MinidumpSystemInfo *system_info = GetSystemInfo();

  if (!system_info)
    return arch_spec;

  // TODO what to do about big endiand flavors of arm ?
  // TODO set the arm subarch stuff if the minidump has info about it

  const MinidumpCPUArchitecture arch =
      static_cast<const MinidumpCPUArchitecture>(
          static_cast<const uint32_t>(system_info->processor_arch));
  switch (arch) {
  case MinidumpCPUArchitecture::X86:
    arch_spec.GetTriple().setArch(llvm::Triple::ArchType::x86);
    break;
  case MinidumpCPUArchitecture::AMD64:
    arch_spec.GetTriple().setArch(llvm::Triple::ArchType::x86_64);
    break;
  case MinidumpCPUArchitecture::ARM:
    arch_spec.GetTriple().setArch(llvm::Triple::ArchType::arm);
    break;
  case MinidumpCPUArchitecture::ARM64:
    arch_spec.GetTriple().setArch(llvm::Triple::ArchType::aarch64);
    break;
  default:
    break;
  }

  return arch_spec;
}

const MinidumpMiscInfo *MinidumpParser::GetMiscInfo() {
  llvm::Optional<llvm::ArrayRef<uint8_t>> data =
      GetStream(MinidumpStreamType::MiscInfo);

  if (!data)
    return nullptr;

  return MinidumpMiscInfo::Parse(data.getValue());
}
