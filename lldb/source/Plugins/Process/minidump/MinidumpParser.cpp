//===-- MinidumpParser.cpp ---------------------------------------*- C++ -*-===//
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

  return MinidumpParser(data_buf_sp, header, std::move(directory_map));
}

MinidumpParser::MinidumpParser(
    const lldb::DataBufferSP &data_buf_sp, const MinidumpHeader *header,
    llvm::DenseMap<uint32_t, MinidumpLocationDescriptor> &&directory_map)
    : m_data_sp(data_buf_sp), m_header(header), m_directory_map(directory_map) {
}

lldb::offset_t MinidumpParser::GetByteSize() {
  return m_data_sp->GetByteSize();
}

llvm::ArrayRef<uint8_t>
MinidumpParser::GetStream(MinidumpStreamType stream_type) {
  auto iter = m_directory_map.find(static_cast<uint32_t>(stream_type));
  if (iter == m_directory_map.end())
    return {};

  // check if there is enough data
  if (iter->second.rva + iter->second.data_size > m_data_sp->GetByteSize())
    return {};

  return llvm::ArrayRef<uint8_t>(m_data_sp->GetBytes() + iter->second.rva,
                                 iter->second.data_size);
}

llvm::Optional<std::string> MinidumpParser::GetMinidumpString(uint32_t rva) {
  auto arr_ref = m_data_sp->GetData();
  if (rva > arr_ref.size())
    return llvm::None;
  arr_ref = arr_ref.drop_front(rva);
  return parseMinidumpString(arr_ref);
}

llvm::ArrayRef<MinidumpThread> MinidumpParser::GetThreads() {
  llvm::ArrayRef<uint8_t> data = GetStream(MinidumpStreamType::ThreadList);

  if (data.size() == 0)
    return llvm::None;

  return MinidumpThread::ParseThreadList(data);
}

const MinidumpSystemInfo *MinidumpParser::GetSystemInfo() {
  llvm::ArrayRef<uint8_t> data = GetStream(MinidumpStreamType::SystemInfo);

  if (data.size() == 0)
    return nullptr;

  return MinidumpSystemInfo::Parse(data);
}

ArchSpec MinidumpParser::GetArchitecture() {
  ArchSpec arch_spec;
  const MinidumpSystemInfo *system_info = GetSystemInfo();

  if (!system_info)
    return arch_spec;

  // TODO what to do about big endiand flavors of arm ?
  // TODO set the arm subarch stuff if the minidump has info about it

  llvm::Triple triple;
  triple.setVendor(llvm::Triple::VendorType::UnknownVendor);

  const MinidumpCPUArchitecture arch =
      static_cast<const MinidumpCPUArchitecture>(
          static_cast<const uint32_t>(system_info->processor_arch));

  switch (arch) {
  case MinidumpCPUArchitecture::X86:
    triple.setArch(llvm::Triple::ArchType::x86);
    break;
  case MinidumpCPUArchitecture::AMD64:
    triple.setArch(llvm::Triple::ArchType::x86_64);
    break;
  case MinidumpCPUArchitecture::ARM:
    triple.setArch(llvm::Triple::ArchType::arm);
    break;
  case MinidumpCPUArchitecture::ARM64:
    triple.setArch(llvm::Triple::ArchType::aarch64);
    break;
  default:
    triple.setArch(llvm::Triple::ArchType::UnknownArch);
    break;
  }

  const MinidumpOSPlatform os = static_cast<const MinidumpOSPlatform>(
      static_cast<const uint32_t>(system_info->platform_id));

  // TODO add all of the OSes that Minidump/breakpad distinguishes?
  switch (os) {
  case MinidumpOSPlatform::Win32S:
  case MinidumpOSPlatform::Win32Windows:
  case MinidumpOSPlatform::Win32NT:
  case MinidumpOSPlatform::Win32CE:
    triple.setOS(llvm::Triple::OSType::Win32);
    break;
  case MinidumpOSPlatform::Linux:
    triple.setOS(llvm::Triple::OSType::Linux);
    break;
  case MinidumpOSPlatform::MacOSX:
    triple.setOS(llvm::Triple::OSType::MacOSX);
    break;
  case MinidumpOSPlatform::Android:
    triple.setOS(llvm::Triple::OSType::Linux);
    triple.setEnvironment(llvm::Triple::EnvironmentType::Android);
    break;
  default:
    triple.setOS(llvm::Triple::OSType::UnknownOS);
    break;
  }

  arch_spec.SetTriple(triple);

  return arch_spec;
}

const MinidumpMiscInfo *MinidumpParser::GetMiscInfo() {
  llvm::ArrayRef<uint8_t> data = GetStream(MinidumpStreamType::MiscInfo);

  if (data.size() == 0)
    return nullptr;

  return MinidumpMiscInfo::Parse(data);
}

llvm::Optional<LinuxProcStatus> MinidumpParser::GetLinuxProcStatus() {
  llvm::ArrayRef<uint8_t> data = GetStream(MinidumpStreamType::LinuxProcStatus);

  if (data.size() == 0)
    return llvm::None;

  return LinuxProcStatus::Parse(data);
}

llvm::Optional<lldb::pid_t> MinidumpParser::GetPid() {
  const MinidumpMiscInfo *misc_info = GetMiscInfo();
  if (misc_info != nullptr) {
    return misc_info->GetPid();
  }

  llvm::Optional<LinuxProcStatus> proc_status = GetLinuxProcStatus();
  if (proc_status.hasValue()) {
    return proc_status->GetPid();
  }

  return llvm::None;
}

llvm::ArrayRef<MinidumpModule> MinidumpParser::GetModuleList() {
  llvm::ArrayRef<uint8_t> data = GetStream(MinidumpStreamType::ModuleList);

  if (data.size() == 0)
    return {};

  return MinidumpModule::ParseModuleList(data);
}

const MinidumpExceptionStream *MinidumpParser::GetExceptionStream() {
  llvm::ArrayRef<uint8_t> data = GetStream(MinidumpStreamType::Exception);

  if (data.size() == 0)
    return nullptr;

  return MinidumpExceptionStream::Parse(data);
}
