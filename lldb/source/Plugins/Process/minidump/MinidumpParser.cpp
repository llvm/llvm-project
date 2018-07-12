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
#include "NtStructures.h"
#include "RegisterContextMinidump_x86_32.h"

// Other libraries and framework includes
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/LLDBAssert.h"

// C includes
// C++ includes
#include <algorithm>
#include <map>
#include <vector>

using namespace lldb_private;
using namespace minidump;

llvm::Optional<MinidumpParser>
MinidumpParser::Create(const lldb::DataBufferSP &data_buf_sp) {
  if (data_buf_sp->GetByteSize() < sizeof(MinidumpHeader)) {
    return llvm::None;
  }
  return MinidumpParser(data_buf_sp);
}

MinidumpParser::MinidumpParser(const lldb::DataBufferSP &data_buf_sp)
    : m_data_sp(data_buf_sp) {}

llvm::ArrayRef<uint8_t> MinidumpParser::GetData() {
  return llvm::ArrayRef<uint8_t>(m_data_sp->GetBytes(),
                                 m_data_sp->GetByteSize());
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

UUID MinidumpParser::GetModuleUUID(const MinidumpModule *module) {
  auto cv_record =
      GetData().slice(module->CV_record.rva, module->CV_record.data_size);

  // Read the CV record signature
  const llvm::support::ulittle32_t *signature = nullptr;
  Status error = consumeObject(cv_record, signature);
  if (error.Fail())
    return UUID();

  const CvSignature cv_signature =
      static_cast<CvSignature>(static_cast<const uint32_t>(*signature));

  if (cv_signature == CvSignature::Pdb70) {
    // PDB70 record
    const CvRecordPdb70 *pdb70_uuid = nullptr;
    Status error = consumeObject(cv_record, pdb70_uuid);
    if (!error.Fail())
      return UUID::fromData(pdb70_uuid, sizeof(*pdb70_uuid));
  } else if (cv_signature == CvSignature::ElfBuildId)
    return UUID::fromData(cv_record);

  return UUID();
}

llvm::ArrayRef<MinidumpThread> MinidumpParser::GetThreads() {
  llvm::ArrayRef<uint8_t> data = GetStream(MinidumpStreamType::ThreadList);

  if (data.size() == 0)
    return llvm::None;

  return MinidumpThread::ParseThreadList(data);
}

llvm::ArrayRef<uint8_t>
MinidumpParser::GetThreadContext(const MinidumpThread &td) {
  if (td.thread_context.rva + td.thread_context.data_size > GetData().size())
    return {};

  return GetData().slice(td.thread_context.rva, td.thread_context.data_size);
}

llvm::ArrayRef<uint8_t>
MinidumpParser::GetThreadContextWow64(const MinidumpThread &td) {
  // On Windows, a 32-bit process can run on a 64-bit machine under WOW64. If
  // the minidump was captured with a 64-bit debugger, then the CONTEXT we just
  // grabbed from the mini_dump_thread is the one for the 64-bit "native"
  // process rather than the 32-bit "guest" process we care about.  In this
  // case, we can get the 32-bit CONTEXT from the TEB (Thread Environment
  // Block) of the 64-bit process.
  auto teb_mem = GetMemory(td.teb, sizeof(TEB64));
  if (teb_mem.empty())
    return {};

  const TEB64 *wow64teb;
  Status error = consumeObject(teb_mem, wow64teb);
  if (error.Fail())
    return {};

  // Slot 1 of the thread-local storage in the 64-bit TEB points to a structure
  // that includes the 32-bit CONTEXT (after a ULONG). See:
  // https://msdn.microsoft.com/en-us/library/ms681670.aspx
  auto context =
      GetMemory(wow64teb->tls_slots[1] + 4, sizeof(MinidumpContext_x86_32));
  if (context.size() < sizeof(MinidumpContext_x86_32))
    return {};

  return context;
  // NOTE:  We don't currently use the TEB for anything else.  If we
  // need it in the future, the 32-bit TEB is located according to the address
  // stored in the first slot of the 64-bit TEB (wow64teb.Reserved1[0]).
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

std::vector<const MinidumpModule *> MinidumpParser::GetFilteredModuleList() {
  llvm::ArrayRef<MinidumpModule> modules = GetModuleList();
  // map module_name -> pair(load_address, pointer to module struct in memory)
  llvm::StringMap<std::pair<uint64_t, const MinidumpModule *>> lowest_addr;

  std::vector<const MinidumpModule *> filtered_modules;

  llvm::Optional<std::string> name;
  std::string module_name;

  for (const auto &module : modules) {
    name = GetMinidumpString(module.module_name_rva);

    if (!name)
      continue;

    module_name = name.getValue();

    auto iter = lowest_addr.end();
    bool exists;
    std::tie(iter, exists) = lowest_addr.try_emplace(
        module_name, std::make_pair(module.base_of_image, &module));

    if (exists && module.base_of_image < iter->second.first)
      iter->second = std::make_pair(module.base_of_image, &module);
  }

  filtered_modules.reserve(lowest_addr.size());
  for (const auto &module : lowest_addr) {
    filtered_modules.push_back(module.second.second);
  }

  return filtered_modules;
}

const MinidumpExceptionStream *MinidumpParser::GetExceptionStream() {
  llvm::ArrayRef<uint8_t> data = GetStream(MinidumpStreamType::Exception);

  if (data.size() == 0)
    return nullptr;

  return MinidumpExceptionStream::Parse(data);
}

llvm::Optional<minidump::Range>
MinidumpParser::FindMemoryRange(lldb::addr_t addr) {
  llvm::ArrayRef<uint8_t> data = GetStream(MinidumpStreamType::MemoryList);
  llvm::ArrayRef<uint8_t> data64 = GetStream(MinidumpStreamType::Memory64List);

  if (data.empty() && data64.empty())
    return llvm::None;

  if (!data.empty()) {
    llvm::ArrayRef<MinidumpMemoryDescriptor> memory_list =
        MinidumpMemoryDescriptor::ParseMemoryList(data);

    if (memory_list.empty())
      return llvm::None;

    for (const auto &memory_desc : memory_list) {
      const MinidumpLocationDescriptor &loc_desc = memory_desc.memory;
      const lldb::addr_t range_start = memory_desc.start_of_memory_range;
      const size_t range_size = loc_desc.data_size;

      if (loc_desc.rva + loc_desc.data_size > GetData().size())
        return llvm::None;

      if (range_start <= addr && addr < range_start + range_size) {
        return minidump::Range(range_start,
                               GetData().slice(loc_desc.rva, range_size));
      }
    }
  }

  // Some Minidumps have a Memory64ListStream that captures all the heap memory
  // (full-memory Minidumps).  We can't exactly use the same loop as above,
  // because the Minidump uses slightly different data structures to describe
  // those

  if (!data64.empty()) {
    llvm::ArrayRef<MinidumpMemoryDescriptor64> memory64_list;
    uint64_t base_rva;
    std::tie(memory64_list, base_rva) =
        MinidumpMemoryDescriptor64::ParseMemory64List(data64);

    if (memory64_list.empty())
      return llvm::None;

    for (const auto &memory_desc64 : memory64_list) {
      const lldb::addr_t range_start = memory_desc64.start_of_memory_range;
      const size_t range_size = memory_desc64.data_size;

      if (base_rva + range_size > GetData().size())
        return llvm::None;

      if (range_start <= addr && addr < range_start + range_size) {
        return minidump::Range(range_start,
                               GetData().slice(base_rva, range_size));
      }
      base_rva += range_size;
    }
  }

  return llvm::None;
}

llvm::ArrayRef<uint8_t> MinidumpParser::GetMemory(lldb::addr_t addr,
                                                  size_t size) {
  // I don't have a sense of how frequently this is called or how many memory
  // ranges a Minidump typically has, so I'm not sure if searching for the
  // appropriate range linearly each time is stupid.  Perhaps we should build
  // an index for faster lookups.
  llvm::Optional<minidump::Range> range = FindMemoryRange(addr);
  if (!range)
    return {};

  // There's at least some overlap between the beginning of the desired range
  // (addr) and the current range.  Figure out where the overlap begins and how
  // much overlap there is.

  const size_t offset = addr - range->start;

  if (addr < range->start || offset >= range->range_ref.size())
    return {};

  const size_t overlap = std::min(size, range->range_ref.size() - offset);
  return range->range_ref.slice(offset, overlap);
}

llvm::Optional<MemoryRegionInfo>
MinidumpParser::GetMemoryRegionInfo(lldb::addr_t load_addr) {
  MemoryRegionInfo info;
  llvm::ArrayRef<uint8_t> data = GetStream(MinidumpStreamType::MemoryInfoList);
  if (data.empty())
    return llvm::None;

  std::vector<const MinidumpMemoryInfo *> mem_info_list =
      MinidumpMemoryInfo::ParseMemoryInfoList(data);
  if (mem_info_list.empty())
    return llvm::None;

  const auto yes = MemoryRegionInfo::eYes;
  const auto no = MemoryRegionInfo::eNo;

  const MinidumpMemoryInfo *next_entry = nullptr;
  for (const auto &entry : mem_info_list) {
    const auto head = entry->base_address;
    const auto tail = head + entry->region_size;

    if (head <= load_addr && load_addr < tail) {
      info.GetRange().SetRangeBase(
          (entry->state != uint32_t(MinidumpMemoryInfoState::MemFree))
              ? head
              : load_addr);
      info.GetRange().SetRangeEnd(tail);

      const uint32_t PageNoAccess =
          static_cast<uint32_t>(MinidumpMemoryProtectionContants::PageNoAccess);
      info.SetReadable((entry->protect & PageNoAccess) == 0 ? yes : no);

      const uint32_t PageWritable =
          static_cast<uint32_t>(MinidumpMemoryProtectionContants::PageWritable);
      info.SetWritable((entry->protect & PageWritable) != 0 ? yes : no);

      const uint32_t PageExecutable = static_cast<uint32_t>(
          MinidumpMemoryProtectionContants::PageExecutable);
      info.SetExecutable((entry->protect & PageExecutable) != 0 ? yes : no);

      const uint32_t MemFree =
          static_cast<uint32_t>(MinidumpMemoryInfoState::MemFree);
      info.SetMapped((entry->state != MemFree) ? yes : no);

      return info;
    } else if (head > load_addr &&
               (next_entry == nullptr || head < next_entry->base_address)) {
      // In case there is no region containing load_addr keep track of the
      // nearest region after load_addr so we can return the distance to it.
      next_entry = entry;
    }
  }

  // No containing region found. Create an unmapped region that extends to the
  // next region or LLDB_INVALID_ADDRESS
  info.GetRange().SetRangeBase(load_addr);
  info.GetRange().SetRangeEnd((next_entry != nullptr) ? next_entry->base_address
                                                      : LLDB_INVALID_ADDRESS);
  info.SetReadable(no);
  info.SetWritable(no);
  info.SetExecutable(no);
  info.SetMapped(no);

  // Note that the memory info list doesn't seem to contain ranges in kernel
  // space, so if you're walking a stack that has kernel frames, the stack may
  // appear truncated.
  return info;
}

Status MinidumpParser::Initialize() {
  Status error;

  lldbassert(m_directory_map.empty());

  llvm::ArrayRef<uint8_t> header_data(m_data_sp->GetBytes(),
                                      sizeof(MinidumpHeader));
  const MinidumpHeader *header = MinidumpHeader::Parse(header_data);
  if (header == nullptr) {
    error.SetErrorString("invalid minidump: can't parse the header");
    return error;
  }

  // A minidump without at least one stream is clearly ill-formed
  if (header->streams_count == 0) {
    error.SetErrorString("invalid minidump: no streams present");
    return error;
  }

  struct FileRange {
    uint32_t offset = 0;
    uint32_t size = 0;

    FileRange(uint32_t offset, uint32_t size) : offset(offset), size(size) {}
    uint32_t end() const { return offset + size; }
  };

  const uint32_t file_size = m_data_sp->GetByteSize();

  // Build a global minidump file map, checking for:
  // - overlapping streams/data structures
  // - truncation (streams pointing past the end of file)
  std::vector<FileRange> minidump_map;

  // Add the minidump header to the file map
  if (sizeof(MinidumpHeader) > file_size) {
    error.SetErrorString("invalid minidump: truncated header");
    return error;
  }
  minidump_map.emplace_back( 0, sizeof(MinidumpHeader) );

  // Add the directory entries to the file map
  FileRange directory_range(header->stream_directory_rva,
                            header->streams_count *
                                sizeof(MinidumpDirectory));
  if (directory_range.end() > file_size) {
    error.SetErrorString("invalid minidump: truncated streams directory");
    return error;
  }
  minidump_map.push_back(directory_range);

  // Parse stream directory entries
  llvm::ArrayRef<uint8_t> directory_data(
      m_data_sp->GetBytes() + directory_range.offset, directory_range.size);
  for (uint32_t i = 0; i < header->streams_count; ++i) {
    const MinidumpDirectory *directory_entry = nullptr;
    error = consumeObject(directory_data, directory_entry);
    if (error.Fail())
      return error;
    if (directory_entry->stream_type == 0) {
      // Ignore dummy streams (technically ill-formed, but a number of
      // existing minidumps seem to contain such streams)
      if (directory_entry->location.data_size == 0)
        continue;
      error.SetErrorString("invalid minidump: bad stream type");
      return error;
    }
    // Update the streams map, checking for duplicate stream types
    if (!m_directory_map
             .insert({directory_entry->stream_type, directory_entry->location})
             .second) {
      error.SetErrorString("invalid minidump: duplicate stream type");
      return error;
    }
    // Ignore the zero-length streams for layout checks
    if (directory_entry->location.data_size != 0) {
      minidump_map.emplace_back(directory_entry->location.rva,
                                directory_entry->location.data_size);
    }
  }

  // Sort the file map ranges by start offset
  std::sort(minidump_map.begin(), minidump_map.end(),
            [](const FileRange &a, const FileRange &b) {
              return a.offset < b.offset;
            });

  // Check for overlapping streams/data structures
  for (size_t i = 1; i < minidump_map.size(); ++i) {
    const auto &prev_range = minidump_map[i - 1];
    if (prev_range.end() > minidump_map[i].offset) {
      error.SetErrorString("invalid minidump: overlapping streams");
      return error;
    }
  }

  // Check for streams past the end of file
  const auto &last_range = minidump_map.back();
  if (last_range.end() > file_size) {
    error.SetErrorString("invalid minidump: truncated stream");
    return error;
  }

  return error;
}
