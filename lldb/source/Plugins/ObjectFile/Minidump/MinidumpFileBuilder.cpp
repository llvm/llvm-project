//===-- MinidumpFileBuilder.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinidumpFileBuilder.h"

#include "Plugins/Process/minidump/RegisterContextMinidump_ARM64.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_x86_64.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/Section.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/RegisterValue.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Minidump.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Error.h"

#include "Plugins/Process/minidump/MinidumpTypes.h"

#include <cinttypes>

using namespace lldb;
using namespace lldb_private;
using namespace llvm::minidump;

void MinidumpFileBuilder::AddDirectory(StreamType type, size_t stream_size) {
  LocationDescriptor loc;
  loc.DataSize = static_cast<llvm::support::ulittle32_t>(stream_size);
  // Stream will begin at the current end of data section
  loc.RVA = static_cast<llvm::support::ulittle32_t>(GetCurrentDataEndOffset());

  Directory dir;
  dir.Type = static_cast<llvm::support::little_t<StreamType>>(type);
  dir.Location = loc;

  m_directories.push_back(dir);
}

Status MinidumpFileBuilder::AddSystemInfo(const llvm::Triple &target_triple) {
  Status error;
  AddDirectory(StreamType::SystemInfo, sizeof(llvm::minidump::SystemInfo));

  llvm::minidump::ProcessorArchitecture arch;
  switch (target_triple.getArch()) {
  case llvm::Triple::ArchType::x86_64:
    arch = ProcessorArchitecture::AMD64;
    break;
  case llvm::Triple::ArchType::x86:
    arch = ProcessorArchitecture::X86;
    break;
  case llvm::Triple::ArchType::arm:
    arch = ProcessorArchitecture::ARM;
    break;
  case llvm::Triple::ArchType::aarch64:
    arch = ProcessorArchitecture::ARM64;
    break;
  case llvm::Triple::ArchType::mips64:
  case llvm::Triple::ArchType::mips64el:
  case llvm::Triple::ArchType::mips:
  case llvm::Triple::ArchType::mipsel:
    arch = ProcessorArchitecture::MIPS;
    break;
  case llvm::Triple::ArchType::ppc64:
  case llvm::Triple::ArchType::ppc:
  case llvm::Triple::ArchType::ppc64le:
    arch = ProcessorArchitecture::PPC;
    break;
  default:
    error.SetErrorStringWithFormat("Architecture %s not supported.",
                                   target_triple.getArchName().str().c_str());
    return error;
  };

  llvm::support::little_t<OSPlatform> platform_id;
  switch (target_triple.getOS()) {
  case llvm::Triple::OSType::Linux:
    if (target_triple.getEnvironment() ==
        llvm::Triple::EnvironmentType::Android)
      platform_id = OSPlatform::Android;
    else
      platform_id = OSPlatform::Linux;
    break;
  case llvm::Triple::OSType::Win32:
    platform_id = OSPlatform::Win32NT;
    break;
  case llvm::Triple::OSType::MacOSX:
    platform_id = OSPlatform::MacOSX;
    break;
  case llvm::Triple::OSType::IOS:
    platform_id = OSPlatform::IOS;
    break;
  default:
    error.SetErrorStringWithFormat("OS %s not supported.",
                                   target_triple.getOSName().str().c_str());
    return error;
  };

  llvm::minidump::SystemInfo sys_info;
  sys_info.ProcessorArch =
      static_cast<llvm::support::little_t<ProcessorArchitecture>>(arch);
  // Global offset to beginning of a csd_string in a data section
  sys_info.CSDVersionRVA = static_cast<llvm::support::ulittle32_t>(
      GetCurrentDataEndOffset() + sizeof(llvm::minidump::SystemInfo));
  sys_info.PlatformId = platform_id;
  m_data.AppendData(&sys_info, sizeof(llvm::minidump::SystemInfo));

  std::string csd_string;

  error = WriteString(csd_string, &m_data);
  if (error.Fail()) {
    error.SetErrorString("Unable to convert the csd string to UTF16.");
    return error;
  }

  return error;
}

Status WriteString(const std::string &to_write,
                   lldb_private::DataBufferHeap *buffer) {
  Status error;
  // let the StringRef eat also null termination char
  llvm::StringRef to_write_ref(to_write.c_str(), to_write.size() + 1);
  llvm::SmallVector<llvm::UTF16, 128> to_write_utf16;

  bool converted = convertUTF8ToUTF16String(to_write_ref, to_write_utf16);
  if (!converted) {
    error.SetErrorStringWithFormat(
        "Unable to convert the string to UTF16. Failed to convert %s",
        to_write.c_str());
    return error;
  }

  // size of the UTF16 string should be written without the null termination
  // character that is stored in 2 bytes
  llvm::support::ulittle32_t to_write_size(to_write_utf16.size_in_bytes() - 2);

  buffer->AppendData(&to_write_size, sizeof(llvm::support::ulittle32_t));
  buffer->AppendData(to_write_utf16.data(), to_write_utf16.size_in_bytes());

  return error;
}

llvm::Expected<uint64_t> getModuleFileSize(Target &target,
                                           const ModuleSP &mod) {
  SectionSP sect_sp = mod->GetObjectFile()->GetBaseAddress().GetSection();
  uint64_t SizeOfImage = 0;

  if (!sect_sp) {
    return llvm::createStringError(std::errc::operation_not_supported,
                                   "Couldn't obtain the section information.");
  }
  lldb::addr_t sect_addr = sect_sp->GetLoadBaseAddress(&target);
  // Use memory size since zero fill sections, like ".bss", will be smaller on
  // disk.
  lldb::addr_t sect_size = sect_sp->GetByteSize();
  // This will usually be zero, but make sure to calculate the BaseOfImage
  // offset.
  const lldb::addr_t base_sect_offset =
      mod->GetObjectFile()->GetBaseAddress().GetLoadAddress(&target) -
      sect_addr;
  SizeOfImage = sect_size - base_sect_offset;
  lldb::addr_t next_sect_addr = sect_addr + sect_size;
  Address sect_so_addr;
  target.ResolveLoadAddress(next_sect_addr, sect_so_addr);
  lldb::SectionSP next_sect_sp = sect_so_addr.GetSection();
  while (next_sect_sp &&
         next_sect_sp->GetLoadBaseAddress(&target) == next_sect_addr) {
    sect_size = sect_sp->GetByteSize();
    SizeOfImage += sect_size;
    next_sect_addr += sect_size;
    target.ResolveLoadAddress(next_sect_addr, sect_so_addr);
    next_sect_sp = sect_so_addr.GetSection();
  }

  return SizeOfImage;
}

// ModuleList stream consists of a number of modules, followed by an array
// of llvm::minidump::Module's structures. Every structure informs about a
// single module. Additional data of variable length, such as module's names,
// are stored just after the ModuleList stream. The llvm::minidump::Module
// structures point to this helper data by global offset.
Status MinidumpFileBuilder::AddModuleList(Target &target) {
  constexpr size_t minidump_module_size = sizeof(llvm::minidump::Module);
  Status error;

  const ModuleList &modules = target.GetImages();
  llvm::support::ulittle32_t modules_count =
      static_cast<llvm::support::ulittle32_t>(modules.GetSize());

  // This helps us with getting the correct global offset in minidump
  // file later, when we will be setting up offsets from the
  // the llvm::minidump::Module's structures into helper data
  size_t size_before = GetCurrentDataEndOffset();

  // This is the size of the main part of the ModuleList stream.
  // It consists of a module number and corresponding number of
  // structs describing individual modules
  size_t module_stream_size =
      sizeof(llvm::support::ulittle32_t) + modules_count * minidump_module_size;

  // Adding directory describing this stream.
  AddDirectory(StreamType::ModuleList, module_stream_size);

  m_data.AppendData(&modules_count, sizeof(llvm::support::ulittle32_t));

  // Temporary storage for the helper data (of variable length)
  // as these cannot be dumped to m_data before dumping entire
  // array of module structures.
  DataBufferHeap helper_data;

  for (size_t i = 0; i < modules_count; ++i) {
    ModuleSP mod = modules.GetModuleAtIndex(i);
    std::string module_name = mod->GetSpecificationDescription();
    auto maybe_mod_size = getModuleFileSize(target, mod);
    if (!maybe_mod_size) {
      error.SetErrorStringWithFormat("Unable to get the size of module %s.",
                                     module_name.c_str());
      return error;
    }

    uint64_t mod_size = std::move(*maybe_mod_size);

    llvm::support::ulittle32_t signature =
        static_cast<llvm::support::ulittle32_t>(
            static_cast<uint32_t>(minidump::CvSignature::ElfBuildId));
    auto uuid = mod->GetUUID().GetBytes();

    VSFixedFileInfo info;
    info.Signature = static_cast<llvm::support::ulittle32_t>(0u);
    info.StructVersion = static_cast<llvm::support::ulittle32_t>(0u);
    info.FileVersionHigh = static_cast<llvm::support::ulittle32_t>(0u);
    info.FileVersionLow = static_cast<llvm::support::ulittle32_t>(0u);
    info.ProductVersionHigh = static_cast<llvm::support::ulittle32_t>(0u);
    info.ProductVersionLow = static_cast<llvm::support::ulittle32_t>(0u);
    info.FileFlagsMask = static_cast<llvm::support::ulittle32_t>(0u);
    info.FileFlags = static_cast<llvm::support::ulittle32_t>(0u);
    info.FileOS = static_cast<llvm::support::ulittle32_t>(0u);
    info.FileType = static_cast<llvm::support::ulittle32_t>(0u);
    info.FileSubtype = static_cast<llvm::support::ulittle32_t>(0u);
    info.FileDateHigh = static_cast<llvm::support::ulittle32_t>(0u);
    info.FileDateLow = static_cast<llvm::support::ulittle32_t>(0u);

    LocationDescriptor ld;
    ld.DataSize = static_cast<llvm::support::ulittle32_t>(0u);
    ld.RVA = static_cast<llvm::support::ulittle32_t>(0u);

    // Setting up LocationDescriptor for uuid string. The global offset into
    // minidump file is calculated.
    LocationDescriptor ld_cv;
    ld_cv.DataSize = static_cast<llvm::support::ulittle32_t>(
        sizeof(llvm::support::ulittle32_t) + uuid.size());
    ld_cv.RVA = static_cast<llvm::support::ulittle32_t>(
        size_before + module_stream_size + helper_data.GetByteSize());

    helper_data.AppendData(&signature, sizeof(llvm::support::ulittle32_t));
    helper_data.AppendData(uuid.begin(), uuid.size());

    llvm::minidump::Module m;
    m.BaseOfImage = static_cast<llvm::support::ulittle64_t>(
        mod->GetObjectFile()->GetBaseAddress().GetLoadAddress(&target));
    m.SizeOfImage = static_cast<llvm::support::ulittle32_t>(mod_size);
    m.Checksum = static_cast<llvm::support::ulittle32_t>(0);
    m.TimeDateStamp =
        static_cast<llvm::support::ulittle32_t>(std::time(nullptr));
    m.ModuleNameRVA = static_cast<llvm::support::ulittle32_t>(
        size_before + module_stream_size + helper_data.GetByteSize());
    m.VersionInfo = info;
    m.CvRecord = ld_cv;
    m.MiscRecord = ld;

    error = WriteString(module_name, &helper_data);

    if (error.Fail())
      return error;

    m_data.AppendData(&m, sizeof(llvm::minidump::Module));
  }

  m_data.AppendData(helper_data.GetBytes(), helper_data.GetByteSize());
  return error;
}

uint16_t read_register_u16_raw(RegisterContext *reg_ctx,
                               llvm::StringRef reg_name) {
  const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(reg_name);
  if (!reg_info)
    return 0;
  lldb_private::RegisterValue reg_value;
  bool success = reg_ctx->ReadRegister(reg_info, reg_value);
  if (!success)
    return 0;
  return reg_value.GetAsUInt16();
}

uint32_t read_register_u32_raw(RegisterContext *reg_ctx,
                               llvm::StringRef reg_name) {
  const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(reg_name);
  if (!reg_info)
    return 0;
  lldb_private::RegisterValue reg_value;
  bool success = reg_ctx->ReadRegister(reg_info, reg_value);
  if (!success)
    return 0;
  return reg_value.GetAsUInt32();
}

uint64_t read_register_u64_raw(RegisterContext *reg_ctx,
                               llvm::StringRef reg_name) {
  const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(reg_name);
  if (!reg_info)
    return 0;
  lldb_private::RegisterValue reg_value;
  bool success = reg_ctx->ReadRegister(reg_info, reg_value);
  if (!success)
    return 0;
  return reg_value.GetAsUInt64();
}

llvm::support::ulittle16_t read_register_u16(RegisterContext *reg_ctx,
                                             llvm::StringRef reg_name) {
  return static_cast<llvm::support::ulittle16_t>(
      read_register_u16_raw(reg_ctx, reg_name));
}

llvm::support::ulittle32_t read_register_u32(RegisterContext *reg_ctx,
                                             llvm::StringRef reg_name) {
  return static_cast<llvm::support::ulittle32_t>(
      read_register_u32_raw(reg_ctx, reg_name));
}

llvm::support::ulittle64_t read_register_u64(RegisterContext *reg_ctx,
                                             llvm::StringRef reg_name) {
  return static_cast<llvm::support::ulittle64_t>(
      read_register_u64_raw(reg_ctx, reg_name));
}

void read_register_u128(RegisterContext *reg_ctx, llvm::StringRef reg_name,
                        uint8_t *dst) {
  const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(reg_name);
  if (reg_info) {
    lldb_private::RegisterValue reg_value;
    if (reg_ctx->ReadRegister(reg_info, reg_value)) {
      Status error;
      uint32_t bytes_copied = reg_value.GetAsMemoryData(
          *reg_info, dst, 16, lldb::ByteOrder::eByteOrderLittle, error);
      if (bytes_copied == 16)
        return;
    }
  }
  // If anything goes wrong, then zero out the register value.
  memset(dst, 0, 16);
}

lldb_private::minidump::MinidumpContext_x86_64
GetThreadContext_x86_64(RegisterContext *reg_ctx) {
  lldb_private::minidump::MinidumpContext_x86_64 thread_context = {};
  thread_context.p1_home = {};
  thread_context.context_flags = static_cast<uint32_t>(
      lldb_private::minidump::MinidumpContext_x86_64_Flags::x86_64_Flag |
      lldb_private::minidump::MinidumpContext_x86_64_Flags::Control |
      lldb_private::minidump::MinidumpContext_x86_64_Flags::Segments |
      lldb_private::minidump::MinidumpContext_x86_64_Flags::Integer);
  thread_context.rax = read_register_u64(reg_ctx, "rax");
  thread_context.rbx = read_register_u64(reg_ctx, "rbx");
  thread_context.rcx = read_register_u64(reg_ctx, "rcx");
  thread_context.rdx = read_register_u64(reg_ctx, "rdx");
  thread_context.rdi = read_register_u64(reg_ctx, "rdi");
  thread_context.rsi = read_register_u64(reg_ctx, "rsi");
  thread_context.rbp = read_register_u64(reg_ctx, "rbp");
  thread_context.rsp = read_register_u64(reg_ctx, "rsp");
  thread_context.r8 = read_register_u64(reg_ctx, "r8");
  thread_context.r9 = read_register_u64(reg_ctx, "r9");
  thread_context.r10 = read_register_u64(reg_ctx, "r10");
  thread_context.r11 = read_register_u64(reg_ctx, "r11");
  thread_context.r12 = read_register_u64(reg_ctx, "r12");
  thread_context.r13 = read_register_u64(reg_ctx, "r13");
  thread_context.r14 = read_register_u64(reg_ctx, "r14");
  thread_context.r15 = read_register_u64(reg_ctx, "r15");
  thread_context.rip = read_register_u64(reg_ctx, "rip");
  thread_context.eflags = read_register_u32(reg_ctx, "rflags");
  thread_context.cs = read_register_u16(reg_ctx, "cs");
  thread_context.fs = read_register_u16(reg_ctx, "fs");
  thread_context.gs = read_register_u16(reg_ctx, "gs");
  thread_context.ss = read_register_u16(reg_ctx, "ss");
  thread_context.ds = read_register_u16(reg_ctx, "ds");
  return thread_context;
}

minidump::RegisterContextMinidump_ARM64::Context
GetThreadContext_ARM64(RegisterContext *reg_ctx) {
  minidump::RegisterContextMinidump_ARM64::Context thread_context = {};
  thread_context.context_flags = static_cast<uint32_t>(
      minidump::RegisterContextMinidump_ARM64::Flags::ARM64_Flag |
      minidump::RegisterContextMinidump_ARM64::Flags::Integer |
      minidump::RegisterContextMinidump_ARM64::Flags::FloatingPoint);
  char reg_name[16];
  for (uint32_t i = 0; i < 31; ++i) {
    snprintf(reg_name, sizeof(reg_name), "x%u", i);
    thread_context.x[i] = read_register_u64(reg_ctx, reg_name);
  }
  // Work around a bug in debugserver where "sp" on arm64 doesn't have the alt
  // name set to "x31"
  thread_context.x[31] = read_register_u64(reg_ctx, "sp");
  thread_context.pc = read_register_u64(reg_ctx, "pc");
  thread_context.cpsr = read_register_u32(reg_ctx, "cpsr");
  thread_context.fpsr = read_register_u32(reg_ctx, "fpsr");
  thread_context.fpcr = read_register_u32(reg_ctx, "fpcr");
  for (uint32_t i = 0; i < 32; ++i) {
    snprintf(reg_name, sizeof(reg_name), "v%u", i);
    read_register_u128(reg_ctx, reg_name, &thread_context.v[i * 16]);
  }
  return thread_context;
}

class ArchThreadContexts {
  llvm::Triple::ArchType m_arch;
  union {
    lldb_private::minidump::MinidumpContext_x86_64 x86_64;
    lldb_private::minidump::RegisterContextMinidump_ARM64::Context arm64;
  };

public:
  ArchThreadContexts(llvm::Triple::ArchType arch) : m_arch(arch) {}

  bool prepareRegisterContext(RegisterContext *reg_ctx) {
    switch (m_arch) {
    case llvm::Triple::ArchType::x86_64:
      x86_64 = GetThreadContext_x86_64(reg_ctx);
      return true;
    case llvm::Triple::ArchType::aarch64:
      arm64 = GetThreadContext_ARM64(reg_ctx);
      return true;
    default:
      break;
    }
    return false;
  }

  const void *data() const { return &x86_64; }

  size_t size() const {
    switch (m_arch) {
    case llvm::Triple::ArchType::x86_64:
      return sizeof(x86_64);
    case llvm::Triple::ArchType::aarch64:
      return sizeof(arm64);
    default:
      break;
    }
    return 0;
  }
};

// Function returns start and size of the memory region that contains
// memory location pointed to by the current stack pointer.
llvm::Expected<std::pair<addr_t, addr_t>>
findStackHelper(const lldb::ProcessSP &process_sp, uint64_t rsp) {
  MemoryRegionInfo range_info;
  Status error = process_sp->GetMemoryRegionInfo(rsp, range_info);
  // Skip failed memory region requests or any regions with no permissions.
  if (error.Fail() || range_info.GetLLDBPermissions() == 0)
    return llvm::createStringError(
        std::errc::not_supported,
        "unable to load stack segment of the process");

  const addr_t addr = range_info.GetRange().GetRangeBase();
  const addr_t size = range_info.GetRange().GetByteSize();

  if (size == 0)
    return llvm::createStringError(std::errc::not_supported,
                                   "stack segment of the process is empty");

  return std::make_pair(addr, size);
}

Status MinidumpFileBuilder::AddThreadList(const lldb::ProcessSP &process_sp) {
  constexpr size_t minidump_thread_size = sizeof(llvm::minidump::Thread);
  lldb_private::ThreadList thread_list = process_sp->GetThreadList();

  // size of the entire thread stream consists of:
  // number of threads and threads array
  size_t thread_stream_size = sizeof(llvm::support::ulittle32_t) +
                              thread_list.GetSize() * minidump_thread_size;
  // save for the ability to set up RVA
  size_t size_before = GetCurrentDataEndOffset();

  AddDirectory(StreamType::ThreadList, thread_stream_size);

  llvm::support::ulittle32_t thread_count =
      static_cast<llvm::support::ulittle32_t>(thread_list.GetSize());
  m_data.AppendData(&thread_count, sizeof(llvm::support::ulittle32_t));

  DataBufferHeap helper_data;

  const uint32_t num_threads = thread_list.GetSize();

  for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    ThreadSP thread_sp(thread_list.GetThreadAtIndex(thread_idx));
    RegisterContextSP reg_ctx_sp(thread_sp->GetRegisterContext());
    Status error;

    if (!reg_ctx_sp) {
      error.SetErrorString("Unable to get the register context.");
      return error;
    }
    RegisterContext *reg_ctx = reg_ctx_sp.get();
    Target &target = process_sp->GetTarget();
    const ArchSpec &arch = target.GetArchitecture();
    ArchThreadContexts thread_context(arch.GetMachine());
    if (!thread_context.prepareRegisterContext(reg_ctx)) {
      error.SetErrorStringWithFormat(
          "architecture %s not supported.",
          arch.GetTriple().getArchName().str().c_str());
      return error;
    }
    uint64_t sp = reg_ctx->GetSP();
    auto expected_address_range = findStackHelper(process_sp, sp);

    if (!expected_address_range) {
      consumeError(expected_address_range.takeError());
      error.SetErrorString("Unable to get the stack address.");
      return error;
    }

    std::pair<uint64_t, uint64_t> range = std::move(*expected_address_range);
    uint64_t addr = range.first;
    uint64_t size = range.second;

    auto data_up = std::make_unique<DataBufferHeap>(size, 0);
    const size_t stack_bytes_read =
        process_sp->ReadMemory(addr, data_up->GetBytes(), size, error);

    if (error.Fail())
      return error;

    LocationDescriptor stack_memory;
    stack_memory.DataSize =
        static_cast<llvm::support::ulittle32_t>(stack_bytes_read);
    stack_memory.RVA = static_cast<llvm::support::ulittle32_t>(
        size_before + thread_stream_size + helper_data.GetByteSize());

    MemoryDescriptor stack;
    stack.StartOfMemoryRange = static_cast<llvm::support::ulittle64_t>(addr);
    stack.Memory = stack_memory;

    helper_data.AppendData(data_up->GetBytes(), stack_bytes_read);

    LocationDescriptor thread_context_memory_locator;
    thread_context_memory_locator.DataSize =
        static_cast<llvm::support::ulittle32_t>(thread_context.size());
    thread_context_memory_locator.RVA = static_cast<llvm::support::ulittle32_t>(
        size_before + thread_stream_size + helper_data.GetByteSize());
    // Cache thie thread context memory so we can reuse for exceptions.
    m_tid_to_reg_ctx[thread_sp->GetID()] = thread_context_memory_locator;

    helper_data.AppendData(thread_context.data(), thread_context.size());

    llvm::minidump::Thread t;
    t.ThreadId = static_cast<llvm::support::ulittle32_t>(thread_sp->GetID());
    t.SuspendCount = static_cast<llvm::support::ulittle32_t>(
        (thread_sp->GetState() == StateType::eStateSuspended) ? 1 : 0);
    t.PriorityClass = static_cast<llvm::support::ulittle32_t>(0);
    t.Priority = static_cast<llvm::support::ulittle32_t>(0);
    t.EnvironmentBlock = static_cast<llvm::support::ulittle64_t>(0);
    t.Stack = stack, t.Context = thread_context_memory_locator;

    m_data.AppendData(&t, sizeof(llvm::minidump::Thread));
  }

  m_data.AppendData(helper_data.GetBytes(), helper_data.GetByteSize());
  return Status();
}

void MinidumpFileBuilder::AddExceptions(const lldb::ProcessSP &process_sp) {
  lldb_private::ThreadList thread_list = process_sp->GetThreadList();

  const uint32_t num_threads = thread_list.GetSize();
  for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    ThreadSP thread_sp(thread_list.GetThreadAtIndex(thread_idx));
    StopInfoSP stop_info_sp = thread_sp->GetStopInfo();
    bool add_exception = false;
    if (stop_info_sp) {
      switch (stop_info_sp->GetStopReason()) {
      case eStopReasonSignal:
      case eStopReasonException:
        add_exception = true;
        break;
      default:
        break;
      }
    }
    if (add_exception) {
      constexpr size_t minidump_exception_size =
          sizeof(llvm::minidump::ExceptionStream);
      AddDirectory(StreamType::Exception, minidump_exception_size);
      StopInfoSP stop_info_sp = thread_sp->GetStopInfo();
      RegisterContextSP reg_ctx_sp(thread_sp->GetRegisterContext());
      Exception exp_record = {};
      exp_record.ExceptionCode =
          static_cast<llvm::support::ulittle32_t>(stop_info_sp->GetValue());
      exp_record.ExceptionFlags = static_cast<llvm::support::ulittle32_t>(0);
      exp_record.ExceptionRecord = static_cast<llvm::support::ulittle64_t>(0);
      exp_record.ExceptionAddress = reg_ctx_sp->GetPC();
      exp_record.NumberParameters = static_cast<llvm::support::ulittle32_t>(0);
      exp_record.UnusedAlignment = static_cast<llvm::support::ulittle32_t>(0);
      // exp_record.ExceptionInformation;

      ExceptionStream exp_stream;
      exp_stream.ThreadId =
          static_cast<llvm::support::ulittle32_t>(thread_sp->GetID());
      exp_stream.UnusedAlignment = static_cast<llvm::support::ulittle32_t>(0);
      exp_stream.ExceptionRecord = exp_record;
      auto Iter = m_tid_to_reg_ctx.find(thread_sp->GetID());
      if (Iter != m_tid_to_reg_ctx.end()) {
        exp_stream.ThreadContext = Iter->second;
      } else {
        exp_stream.ThreadContext.DataSize = 0;
        exp_stream.ThreadContext.RVA = 0;
      }
      m_data.AppendData(&exp_stream, minidump_exception_size);
    }
  }
}

lldb_private::Status
MinidumpFileBuilder::AddMemoryList(const lldb::ProcessSP &process_sp,
                                   lldb::SaveCoreStyle core_style) {
  Status error;
  Process::CoreFileMemoryRanges core_ranges;
  error = process_sp->CalculateCoreFileSaveRanges(core_style, core_ranges);
  if (error.Fail()) {
    error.SetErrorString("Process doesn't support getting memory region info.");
    return error;
  }

  DataBufferHeap helper_data;
  std::vector<MemoryDescriptor> mem_descriptors;
  for (const auto &core_range : core_ranges) {
    // Skip empty memory regions or any regions with no permissions.
    if (core_range.range.empty() || core_range.lldb_permissions == 0)
      continue;
    const addr_t addr = core_range.range.start();
    const addr_t size = core_range.range.size();
    auto data_up = std::make_unique<DataBufferHeap>(size, 0);
    const size_t bytes_read =
        process_sp->ReadMemory(addr, data_up->GetBytes(), size, error);
    if (bytes_read == 0)
      continue;
    // We have a good memory region with valid bytes to store.
    LocationDescriptor memory_dump;
    memory_dump.DataSize = static_cast<llvm::support::ulittle32_t>(bytes_read);
    memory_dump.RVA =
        static_cast<llvm::support::ulittle32_t>(GetCurrentDataEndOffset());
    MemoryDescriptor memory_desc;
    memory_desc.StartOfMemoryRange =
        static_cast<llvm::support::ulittle64_t>(addr);
    memory_desc.Memory = memory_dump;
    mem_descriptors.push_back(memory_desc);
    m_data.AppendData(data_up->GetBytes(), bytes_read);
  }

  AddDirectory(StreamType::MemoryList,
               sizeof(llvm::support::ulittle32_t) +
                   mem_descriptors.size() *
                       sizeof(llvm::minidump::MemoryDescriptor));
  llvm::support::ulittle32_t memory_ranges_num(mem_descriptors.size());

  m_data.AppendData(&memory_ranges_num, sizeof(llvm::support::ulittle32_t));
  for (auto memory_descriptor : mem_descriptors) {
    m_data.AppendData(&memory_descriptor,
                      sizeof(llvm::minidump::MemoryDescriptor));
  }

  return error;
}

void MinidumpFileBuilder::AddMiscInfo(const lldb::ProcessSP &process_sp) {
  AddDirectory(StreamType::MiscInfo,
               sizeof(lldb_private::minidump::MinidumpMiscInfo));

  lldb_private::minidump::MinidumpMiscInfo misc_info;
  misc_info.size = static_cast<llvm::support::ulittle32_t>(
      sizeof(lldb_private::minidump::MinidumpMiscInfo));
  // Default set flags1 to 0, in case that we will not be able to
  // get any information
  misc_info.flags1 = static_cast<llvm::support::ulittle32_t>(0);

  lldb_private::ProcessInstanceInfo process_info;
  process_sp->GetProcessInfo(process_info);
  if (process_info.ProcessIDIsValid()) {
    // Set flags1 to reflect that PID is filled in
    misc_info.flags1 =
        static_cast<llvm::support::ulittle32_t>(static_cast<uint32_t>(
            lldb_private::minidump::MinidumpMiscInfoFlags::ProcessID));
    misc_info.process_id =
        static_cast<llvm::support::ulittle32_t>(process_info.GetProcessID());
  }

  m_data.AppendData(&misc_info,
                    sizeof(lldb_private::minidump::MinidumpMiscInfo));
}

std::unique_ptr<llvm::MemoryBuffer>
getFileStreamHelper(const std::string &path) {
  auto maybe_stream = llvm::MemoryBuffer::getFileAsStream(path);
  if (!maybe_stream)
    return nullptr;
  return std::move(maybe_stream.get());
}

void MinidumpFileBuilder::AddLinuxFileStreams(
    const lldb::ProcessSP &process_sp) {
  std::vector<std::pair<StreamType, std::string>> files_with_stream_types = {
      {StreamType::LinuxCPUInfo, "/proc/cpuinfo"},
      {StreamType::LinuxLSBRelease, "/etc/lsb-release"},
  };

  lldb_private::ProcessInstanceInfo process_info;
  process_sp->GetProcessInfo(process_info);
  if (process_info.ProcessIDIsValid()) {
    lldb::pid_t pid = process_info.GetProcessID();
    std::string pid_str = std::to_string(pid);
    files_with_stream_types.push_back(
        {StreamType::LinuxProcStatus, "/proc/" + pid_str + "/status"});
    files_with_stream_types.push_back(
        {StreamType::LinuxCMDLine, "/proc/" + pid_str + "/cmdline"});
    files_with_stream_types.push_back(
        {StreamType::LinuxEnviron, "/proc/" + pid_str + "/environ"});
    files_with_stream_types.push_back(
        {StreamType::LinuxAuxv, "/proc/" + pid_str + "/auxv"});
    files_with_stream_types.push_back(
        {StreamType::LinuxMaps, "/proc/" + pid_str + "/maps"});
    files_with_stream_types.push_back(
        {StreamType::LinuxProcStat, "/proc/" + pid_str + "/stat"});
    files_with_stream_types.push_back(
        {StreamType::LinuxProcFD, "/proc/" + pid_str + "/fd"});
  }

  for (const auto &entry : files_with_stream_types) {
    StreamType stream = entry.first;
    std::string path = entry.second;
    auto memory_buffer = getFileStreamHelper(path);

    if (memory_buffer) {
      size_t size = memory_buffer->getBufferSize();
      if (size == 0)
        continue;
      AddDirectory(stream, size);
      m_data.AppendData(memory_buffer->getBufferStart(), size);
    }
  }
}

Status MinidumpFileBuilder::Dump(lldb::FileUP &core_file) const {
  constexpr size_t header_size = sizeof(llvm::minidump::Header);
  constexpr size_t directory_size = sizeof(llvm::minidump::Directory);

  // write header
  llvm::minidump::Header header;
  header.Signature = static_cast<llvm::support::ulittle32_t>(
      llvm::minidump::Header::MagicSignature);
  header.Version = static_cast<llvm::support::ulittle32_t>(
      llvm::minidump::Header::MagicVersion);
  header.NumberOfStreams =
      static_cast<llvm::support::ulittle32_t>(GetDirectoriesNum());
  header.StreamDirectoryRVA =
      static_cast<llvm::support::ulittle32_t>(GetCurrentDataEndOffset());
  header.Checksum = static_cast<llvm::support::ulittle32_t>(
      0u), // not used in most of the writers
      header.TimeDateStamp =
          static_cast<llvm::support::ulittle32_t>(std::time(nullptr));
  header.Flags =
      static_cast<llvm::support::ulittle64_t>(0u); // minidump normal flag

  Status error;
  size_t bytes_written;

  bytes_written = header_size;
  error = core_file->Write(&header, bytes_written);
  if (error.Fail() || bytes_written != header_size) {
    if (bytes_written != header_size)
      error.SetErrorStringWithFormat(
          "unable to write the header (written %zd/%zd)", bytes_written,
          header_size);
    return error;
  }

  // write data
  bytes_written = m_data.GetByteSize();
  error = core_file->Write(m_data.GetBytes(), bytes_written);
  if (error.Fail() || bytes_written != m_data.GetByteSize()) {
    if (bytes_written != m_data.GetByteSize())
      error.SetErrorStringWithFormat(
          "unable to write the data (written %zd/%" PRIu64 ")", bytes_written,
          m_data.GetByteSize());
    return error;
  }

  // write directories
  for (const Directory &dir : m_directories) {
    bytes_written = directory_size;
    error = core_file->Write(&dir, bytes_written);
    if (error.Fail() || bytes_written != directory_size) {
      if (bytes_written != directory_size)
        error.SetErrorStringWithFormat(
            "unable to write the directory (written %zd/%zd)", bytes_written,
            directory_size);
      return error;
    }
  }

  return error;
}

size_t MinidumpFileBuilder::GetDirectoriesNum() const {
  return m_directories.size();
}

size_t MinidumpFileBuilder::GetCurrentDataEndOffset() const {
  return sizeof(llvm::minidump::Header) + m_data.GetByteSize();
}
