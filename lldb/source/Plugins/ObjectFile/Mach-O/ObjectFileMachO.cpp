//===-- ObjectFileMachO.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/ADT/StringRef.h"

// Project includes
#include "Plugins/Process/Utility/RegisterContextDarwin_arm.h"
#include "Plugins/Process/Utility/RegisterContextDarwin_arm64.h"
#include "Plugins/Process/Utility/RegisterContextDarwin_i386.h"
#include "Plugins/Process/Utility/RegisterContextDarwin_x86_64.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"
#include "lldb/Utility/UUID.h"

#include "lldb/Utility/SafeMachO.h"

#include "llvm/Support/MemoryBuffer.h"

#include "ObjectFileMachO.h"

#if defined(__APPLE__) &&                                                      \
    (defined(__arm__) || defined(__arm64__) || defined(__aarch64__))
// GetLLDBSharedCacheUUID() needs to call dlsym()
#include <dlfcn.h>
#endif

#ifndef __APPLE__
#include "Utility/UuidCompatibility.h"
#else
#include <uuid/uuid.h>
#endif

#define THUMB_ADDRESS_BIT_MASK 0xfffffffffffffffeull
using namespace lldb;
using namespace lldb_private;
using namespace llvm::MachO;

// Some structure definitions needed for parsing the dyld shared cache files
// found on iOS devices.

struct lldb_copy_dyld_cache_header_v1 {
  char magic[16];         // e.g. "dyld_v0    i386", "dyld_v1   armv7", etc.
  uint32_t mappingOffset; // file offset to first dyld_cache_mapping_info
  uint32_t mappingCount;  // number of dyld_cache_mapping_info entries
  uint32_t imagesOffset;
  uint32_t imagesCount;
  uint64_t dyldBaseAddress;
  uint64_t codeSignatureOffset;
  uint64_t codeSignatureSize;
  uint64_t slideInfoOffset;
  uint64_t slideInfoSize;
  uint64_t localSymbolsOffset;
  uint64_t localSymbolsSize;
  uint8_t uuid[16]; // v1 and above, also recorded in dyld_all_image_infos v13
                    // and later
};

struct lldb_copy_dyld_cache_mapping_info {
  uint64_t address;
  uint64_t size;
  uint64_t fileOffset;
  uint32_t maxProt;
  uint32_t initProt;
};

struct lldb_copy_dyld_cache_local_symbols_info {
  uint32_t nlistOffset;
  uint32_t nlistCount;
  uint32_t stringsOffset;
  uint32_t stringsSize;
  uint32_t entriesOffset;
  uint32_t entriesCount;
};
struct lldb_copy_dyld_cache_local_symbols_entry {
  uint32_t dylibOffset;
  uint32_t nlistStartIndex;
  uint32_t nlistCount;
};

class RegisterContextDarwin_x86_64_Mach : public RegisterContextDarwin_x86_64 {
public:
  RegisterContextDarwin_x86_64_Mach(lldb_private::Thread &thread,
                                    const DataExtractor &data)
      : RegisterContextDarwin_x86_64(thread, 0) {
    SetRegisterDataFrom_LC_THREAD(data);
  }

  void InvalidateAllRegisters() override {
    // Do nothing... registers are always valid...
  }

  void SetRegisterDataFrom_LC_THREAD(const DataExtractor &data) {
    lldb::offset_t offset = 0;
    SetError(GPRRegSet, Read, -1);
    SetError(FPURegSet, Read, -1);
    SetError(EXCRegSet, Read, -1);
    bool done = false;

    while (!done) {
      int flavor = data.GetU32(&offset);
      if (flavor == 0)
        done = true;
      else {
        uint32_t i;
        uint32_t count = data.GetU32(&offset);
        switch (flavor) {
        case GPRRegSet:
          for (i = 0; i < count; ++i)
            (&gpr.rax)[i] = data.GetU64(&offset);
          SetError(GPRRegSet, Read, 0);
          done = true;

          break;
        case FPURegSet:
          // TODO: fill in FPU regs....
          // SetError (FPURegSet, Read, -1);
          done = true;

          break;
        case EXCRegSet:
          exc.trapno = data.GetU32(&offset);
          exc.err = data.GetU32(&offset);
          exc.faultvaddr = data.GetU64(&offset);
          SetError(EXCRegSet, Read, 0);
          done = true;
          break;
        case 7:
        case 8:
        case 9:
          // fancy flavors that encapsulate of the above flavors...
          break;

        default:
          done = true;
          break;
        }
      }
    }
  }

  static size_t WriteRegister(RegisterContext *reg_ctx, const char *name,
                              const char *alt_name, size_t reg_byte_size,
                              Stream &data) {
    const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(name);
    if (reg_info == NULL)
      reg_info = reg_ctx->GetRegisterInfoByName(alt_name);
    if (reg_info) {
      lldb_private::RegisterValue reg_value;
      if (reg_ctx->ReadRegister(reg_info, reg_value)) {
        if (reg_info->byte_size >= reg_byte_size)
          data.Write(reg_value.GetBytes(), reg_byte_size);
        else {
          data.Write(reg_value.GetBytes(), reg_info->byte_size);
          for (size_t i = 0, n = reg_byte_size - reg_info->byte_size; i < n;
               ++i)
            data.PutChar(0);
        }
        return reg_byte_size;
      }
    }
    // Just write zeros if all else fails
    for (size_t i = 0; i < reg_byte_size; ++i)
      data.PutChar(0);
    return reg_byte_size;
  }

  static bool Create_LC_THREAD(Thread *thread, Stream &data) {
    RegisterContextSP reg_ctx_sp(thread->GetRegisterContext());
    if (reg_ctx_sp) {
      RegisterContext *reg_ctx = reg_ctx_sp.get();

      data.PutHex32(GPRRegSet); // Flavor
      data.PutHex32(GPRWordCount);
      WriteRegister(reg_ctx, "rax", NULL, 8, data);
      WriteRegister(reg_ctx, "rbx", NULL, 8, data);
      WriteRegister(reg_ctx, "rcx", NULL, 8, data);
      WriteRegister(reg_ctx, "rdx", NULL, 8, data);
      WriteRegister(reg_ctx, "rdi", NULL, 8, data);
      WriteRegister(reg_ctx, "rsi", NULL, 8, data);
      WriteRegister(reg_ctx, "rbp", NULL, 8, data);
      WriteRegister(reg_ctx, "rsp", NULL, 8, data);
      WriteRegister(reg_ctx, "r8", NULL, 8, data);
      WriteRegister(reg_ctx, "r9", NULL, 8, data);
      WriteRegister(reg_ctx, "r10", NULL, 8, data);
      WriteRegister(reg_ctx, "r11", NULL, 8, data);
      WriteRegister(reg_ctx, "r12", NULL, 8, data);
      WriteRegister(reg_ctx, "r13", NULL, 8, data);
      WriteRegister(reg_ctx, "r14", NULL, 8, data);
      WriteRegister(reg_ctx, "r15", NULL, 8, data);
      WriteRegister(reg_ctx, "rip", NULL, 8, data);
      WriteRegister(reg_ctx, "rflags", NULL, 8, data);
      WriteRegister(reg_ctx, "cs", NULL, 8, data);
      WriteRegister(reg_ctx, "fs", NULL, 8, data);
      WriteRegister(reg_ctx, "gs", NULL, 8, data);

      //            // Write out the FPU registers
      //            const size_t fpu_byte_size = sizeof(FPU);
      //            size_t bytes_written = 0;
      //            data.PutHex32 (FPURegSet);
      //            data.PutHex32 (fpu_byte_size/sizeof(uint64_t));
      //            bytes_written += data.PutHex32(0); // uint32_t pad[0]
      //            bytes_written += data.PutHex32(0); // uint32_t pad[1]
      //            bytes_written += WriteRegister (reg_ctx, "fcw", "fctrl", 2,
      //            data);   // uint16_t    fcw;    // "fctrl"
      //            bytes_written += WriteRegister (reg_ctx, "fsw" , "fstat", 2,
      //            data);  // uint16_t    fsw;    // "fstat"
      //            bytes_written += WriteRegister (reg_ctx, "ftw" , "ftag", 1,
      //            data);   // uint8_t     ftw;    // "ftag"
      //            bytes_written += data.PutHex8  (0); // uint8_t pad1;
      //            bytes_written += WriteRegister (reg_ctx, "fop" , NULL, 2,
      //            data);     // uint16_t    fop;    // "fop"
      //            bytes_written += WriteRegister (reg_ctx, "fioff", "ip", 4,
      //            data);    // uint32_t    ip;     // "fioff"
      //            bytes_written += WriteRegister (reg_ctx, "fiseg", NULL, 2,
      //            data);    // uint16_t    cs;     // "fiseg"
      //            bytes_written += data.PutHex16 (0); // uint16_t    pad2;
      //            bytes_written += WriteRegister (reg_ctx, "dp", "fooff" , 4,
      //            data);   // uint32_t    dp;     // "fooff"
      //            bytes_written += WriteRegister (reg_ctx, "foseg", NULL, 2,
      //            data);    // uint16_t    ds;     // "foseg"
      //            bytes_written += data.PutHex16 (0); // uint16_t    pad3;
      //            bytes_written += WriteRegister (reg_ctx, "mxcsr", NULL, 4,
      //            data);    // uint32_t    mxcsr;
      //            bytes_written += WriteRegister (reg_ctx, "mxcsrmask", NULL,
      //            4, data);// uint32_t    mxcsrmask;
      //            bytes_written += WriteRegister (reg_ctx, "stmm0", NULL,
      //            sizeof(MMSReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "stmm1", NULL,
      //            sizeof(MMSReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "stmm2", NULL,
      //            sizeof(MMSReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "stmm3", NULL,
      //            sizeof(MMSReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "stmm4", NULL,
      //            sizeof(MMSReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "stmm5", NULL,
      //            sizeof(MMSReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "stmm6", NULL,
      //            sizeof(MMSReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "stmm7", NULL,
      //            sizeof(MMSReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm0" , NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm1" , NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm2" , NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm3" , NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm4" , NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm5" , NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm6" , NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm7" , NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm8" , NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm9" , NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm10", NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm11", NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm12", NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm13", NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm14", NULL,
      //            sizeof(XMMReg), data);
      //            bytes_written += WriteRegister (reg_ctx, "xmm15", NULL,
      //            sizeof(XMMReg), data);
      //
      //            // Fill rest with zeros
      //            for (size_t i=0, n = fpu_byte_size - bytes_written; i<n; ++
      //            i)
      //                data.PutChar(0);

      // Write out the EXC registers
      data.PutHex32(EXCRegSet);
      data.PutHex32(EXCWordCount);
      WriteRegister(reg_ctx, "trapno", NULL, 4, data);
      WriteRegister(reg_ctx, "err", NULL, 4, data);
      WriteRegister(reg_ctx, "faultvaddr", NULL, 8, data);
      return true;
    }
    return false;
  }

protected:
  int DoReadGPR(lldb::tid_t tid, int flavor, GPR &gpr) override { return 0; }

  int DoReadFPU(lldb::tid_t tid, int flavor, FPU &fpu) override { return 0; }

  int DoReadEXC(lldb::tid_t tid, int flavor, EXC &exc) override { return 0; }

  int DoWriteGPR(lldb::tid_t tid, int flavor, const GPR &gpr) override {
    return 0;
  }

  int DoWriteFPU(lldb::tid_t tid, int flavor, const FPU &fpu) override {
    return 0;
  }

  int DoWriteEXC(lldb::tid_t tid, int flavor, const EXC &exc) override {
    return 0;
  }
};

class RegisterContextDarwin_i386_Mach : public RegisterContextDarwin_i386 {
public:
  RegisterContextDarwin_i386_Mach(lldb_private::Thread &thread,
                                  const DataExtractor &data)
      : RegisterContextDarwin_i386(thread, 0) {
    SetRegisterDataFrom_LC_THREAD(data);
  }

  void InvalidateAllRegisters() override {
    // Do nothing... registers are always valid...
  }

  void SetRegisterDataFrom_LC_THREAD(const DataExtractor &data) {
    lldb::offset_t offset = 0;
    SetError(GPRRegSet, Read, -1);
    SetError(FPURegSet, Read, -1);
    SetError(EXCRegSet, Read, -1);
    bool done = false;

    while (!done) {
      int flavor = data.GetU32(&offset);
      if (flavor == 0)
        done = true;
      else {
        uint32_t i;
        uint32_t count = data.GetU32(&offset);
        switch (flavor) {
        case GPRRegSet:
          for (i = 0; i < count; ++i)
            (&gpr.eax)[i] = data.GetU32(&offset);
          SetError(GPRRegSet, Read, 0);
          done = true;

          break;
        case FPURegSet:
          // TODO: fill in FPU regs....
          // SetError (FPURegSet, Read, -1);
          done = true;

          break;
        case EXCRegSet:
          exc.trapno = data.GetU32(&offset);
          exc.err = data.GetU32(&offset);
          exc.faultvaddr = data.GetU32(&offset);
          SetError(EXCRegSet, Read, 0);
          done = true;
          break;
        case 7:
        case 8:
        case 9:
          // fancy flavors that encapsulate of the above flavors...
          break;

        default:
          done = true;
          break;
        }
      }
    }
  }

  static size_t WriteRegister(RegisterContext *reg_ctx, const char *name,
                              const char *alt_name, size_t reg_byte_size,
                              Stream &data) {
    const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(name);
    if (reg_info == NULL)
      reg_info = reg_ctx->GetRegisterInfoByName(alt_name);
    if (reg_info) {
      lldb_private::RegisterValue reg_value;
      if (reg_ctx->ReadRegister(reg_info, reg_value)) {
        if (reg_info->byte_size >= reg_byte_size)
          data.Write(reg_value.GetBytes(), reg_byte_size);
        else {
          data.Write(reg_value.GetBytes(), reg_info->byte_size);
          for (size_t i = 0, n = reg_byte_size - reg_info->byte_size; i < n;
               ++i)
            data.PutChar(0);
        }
        return reg_byte_size;
      }
    }
    // Just write zeros if all else fails
    for (size_t i = 0; i < reg_byte_size; ++i)
      data.PutChar(0);
    return reg_byte_size;
  }

  static bool Create_LC_THREAD(Thread *thread, Stream &data) {
    RegisterContextSP reg_ctx_sp(thread->GetRegisterContext());
    if (reg_ctx_sp) {
      RegisterContext *reg_ctx = reg_ctx_sp.get();

      data.PutHex32(GPRRegSet); // Flavor
      data.PutHex32(GPRWordCount);
      WriteRegister(reg_ctx, "eax", NULL, 4, data);
      WriteRegister(reg_ctx, "ebx", NULL, 4, data);
      WriteRegister(reg_ctx, "ecx", NULL, 4, data);
      WriteRegister(reg_ctx, "edx", NULL, 4, data);
      WriteRegister(reg_ctx, "edi", NULL, 4, data);
      WriteRegister(reg_ctx, "esi", NULL, 4, data);
      WriteRegister(reg_ctx, "ebp", NULL, 4, data);
      WriteRegister(reg_ctx, "esp", NULL, 4, data);
      WriteRegister(reg_ctx, "ss", NULL, 4, data);
      WriteRegister(reg_ctx, "eflags", NULL, 4, data);
      WriteRegister(reg_ctx, "eip", NULL, 4, data);
      WriteRegister(reg_ctx, "cs", NULL, 4, data);
      WriteRegister(reg_ctx, "ds", NULL, 4, data);
      WriteRegister(reg_ctx, "es", NULL, 4, data);
      WriteRegister(reg_ctx, "fs", NULL, 4, data);
      WriteRegister(reg_ctx, "gs", NULL, 4, data);

      // Write out the EXC registers
      data.PutHex32(EXCRegSet);
      data.PutHex32(EXCWordCount);
      WriteRegister(reg_ctx, "trapno", NULL, 4, data);
      WriteRegister(reg_ctx, "err", NULL, 4, data);
      WriteRegister(reg_ctx, "faultvaddr", NULL, 4, data);
      return true;
    }
    return false;
  }

protected:
  int DoReadGPR(lldb::tid_t tid, int flavor, GPR &gpr) override { return 0; }

  int DoReadFPU(lldb::tid_t tid, int flavor, FPU &fpu) override { return 0; }

  int DoReadEXC(lldb::tid_t tid, int flavor, EXC &exc) override { return 0; }

  int DoWriteGPR(lldb::tid_t tid, int flavor, const GPR &gpr) override {
    return 0;
  }

  int DoWriteFPU(lldb::tid_t tid, int flavor, const FPU &fpu) override {
    return 0;
  }

  int DoWriteEXC(lldb::tid_t tid, int flavor, const EXC &exc) override {
    return 0;
  }
};

class RegisterContextDarwin_arm_Mach : public RegisterContextDarwin_arm {
public:
  RegisterContextDarwin_arm_Mach(lldb_private::Thread &thread,
                                 const DataExtractor &data)
      : RegisterContextDarwin_arm(thread, 0) {
    SetRegisterDataFrom_LC_THREAD(data);
  }

  void InvalidateAllRegisters() override {
    // Do nothing... registers are always valid...
  }

  void SetRegisterDataFrom_LC_THREAD(const DataExtractor &data) {
    lldb::offset_t offset = 0;
    SetError(GPRRegSet, Read, -1);
    SetError(FPURegSet, Read, -1);
    SetError(EXCRegSet, Read, -1);
    bool done = false;

    while (!done) {
      int flavor = data.GetU32(&offset);
      uint32_t count = data.GetU32(&offset);
      lldb::offset_t next_thread_state = offset + (count * 4);
      switch (flavor) {
      case GPRAltRegSet:
      case GPRRegSet:
        for (uint32_t i = 0; i < count; ++i) {
          gpr.r[i] = data.GetU32(&offset);
        }

        // Note that gpr.cpsr is also copied by the above loop; this loop
        // technically extends one element past the end of the gpr.r[] array.

        SetError(GPRRegSet, Read, 0);
        offset = next_thread_state;
        break;

      case FPURegSet: {
        uint8_t *fpu_reg_buf = (uint8_t *)&fpu.floats.s[0];
        const int fpu_reg_buf_size = sizeof(fpu.floats);
        if (data.ExtractBytes(offset, fpu_reg_buf_size, eByteOrderLittle,
                              fpu_reg_buf) == fpu_reg_buf_size) {
          offset += fpu_reg_buf_size;
          fpu.fpscr = data.GetU32(&offset);
          SetError(FPURegSet, Read, 0);
        } else {
          done = true;
        }
      }
        offset = next_thread_state;
        break;

      case EXCRegSet:
        if (count == 3) {
          exc.exception = data.GetU32(&offset);
          exc.fsr = data.GetU32(&offset);
          exc.far = data.GetU32(&offset);
          SetError(EXCRegSet, Read, 0);
        }
        done = true;
        offset = next_thread_state;
        break;

      // Unknown register set flavor, stop trying to parse.
      default:
        done = true;
      }
    }
  }

  static size_t WriteRegister(RegisterContext *reg_ctx, const char *name,
                              const char *alt_name, size_t reg_byte_size,
                              Stream &data) {
    const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(name);
    if (reg_info == NULL)
      reg_info = reg_ctx->GetRegisterInfoByName(alt_name);
    if (reg_info) {
      lldb_private::RegisterValue reg_value;
      if (reg_ctx->ReadRegister(reg_info, reg_value)) {
        if (reg_info->byte_size >= reg_byte_size)
          data.Write(reg_value.GetBytes(), reg_byte_size);
        else {
          data.Write(reg_value.GetBytes(), reg_info->byte_size);
          for (size_t i = 0, n = reg_byte_size - reg_info->byte_size; i < n;
               ++i)
            data.PutChar(0);
        }
        return reg_byte_size;
      }
    }
    // Just write zeros if all else fails
    for (size_t i = 0; i < reg_byte_size; ++i)
      data.PutChar(0);
    return reg_byte_size;
  }

  static bool Create_LC_THREAD(Thread *thread, Stream &data) {
    RegisterContextSP reg_ctx_sp(thread->GetRegisterContext());
    if (reg_ctx_sp) {
      RegisterContext *reg_ctx = reg_ctx_sp.get();

      data.PutHex32(GPRRegSet); // Flavor
      data.PutHex32(GPRWordCount);
      WriteRegister(reg_ctx, "r0", NULL, 4, data);
      WriteRegister(reg_ctx, "r1", NULL, 4, data);
      WriteRegister(reg_ctx, "r2", NULL, 4, data);
      WriteRegister(reg_ctx, "r3", NULL, 4, data);
      WriteRegister(reg_ctx, "r4", NULL, 4, data);
      WriteRegister(reg_ctx, "r5", NULL, 4, data);
      WriteRegister(reg_ctx, "r6", NULL, 4, data);
      WriteRegister(reg_ctx, "r7", NULL, 4, data);
      WriteRegister(reg_ctx, "r8", NULL, 4, data);
      WriteRegister(reg_ctx, "r9", NULL, 4, data);
      WriteRegister(reg_ctx, "r10", NULL, 4, data);
      WriteRegister(reg_ctx, "r11", NULL, 4, data);
      WriteRegister(reg_ctx, "r12", NULL, 4, data);
      WriteRegister(reg_ctx, "sp", NULL, 4, data);
      WriteRegister(reg_ctx, "lr", NULL, 4, data);
      WriteRegister(reg_ctx, "pc", NULL, 4, data);
      WriteRegister(reg_ctx, "cpsr", NULL, 4, data);

      // Write out the EXC registers
      //            data.PutHex32 (EXCRegSet);
      //            data.PutHex32 (EXCWordCount);
      //            WriteRegister (reg_ctx, "exception", NULL, 4, data);
      //            WriteRegister (reg_ctx, "fsr", NULL, 4, data);
      //            WriteRegister (reg_ctx, "far", NULL, 4, data);
      return true;
    }
    return false;
  }

protected:
  int DoReadGPR(lldb::tid_t tid, int flavor, GPR &gpr) override { return -1; }

  int DoReadFPU(lldb::tid_t tid, int flavor, FPU &fpu) override { return -1; }

  int DoReadEXC(lldb::tid_t tid, int flavor, EXC &exc) override { return -1; }

  int DoReadDBG(lldb::tid_t tid, int flavor, DBG &dbg) override { return -1; }

  int DoWriteGPR(lldb::tid_t tid, int flavor, const GPR &gpr) override {
    return 0;
  }

  int DoWriteFPU(lldb::tid_t tid, int flavor, const FPU &fpu) override {
    return 0;
  }

  int DoWriteEXC(lldb::tid_t tid, int flavor, const EXC &exc) override {
    return 0;
  }

  int DoWriteDBG(lldb::tid_t tid, int flavor, const DBG &dbg) override {
    return -1;
  }
};

class RegisterContextDarwin_arm64_Mach : public RegisterContextDarwin_arm64 {
public:
  RegisterContextDarwin_arm64_Mach(lldb_private::Thread &thread,
                                   const DataExtractor &data)
      : RegisterContextDarwin_arm64(thread, 0) {
    SetRegisterDataFrom_LC_THREAD(data);
  }

  void InvalidateAllRegisters() override {
    // Do nothing... registers are always valid...
  }

  void SetRegisterDataFrom_LC_THREAD(const DataExtractor &data) {
    lldb::offset_t offset = 0;
    SetError(GPRRegSet, Read, -1);
    SetError(FPURegSet, Read, -1);
    SetError(EXCRegSet, Read, -1);
    bool done = false;
    while (!done) {
      int flavor = data.GetU32(&offset);
      uint32_t count = data.GetU32(&offset);
      lldb::offset_t next_thread_state = offset + (count * 4);
      switch (flavor) {
      case GPRRegSet:
        // x0-x29 + fp + lr + sp + pc (== 33 64-bit registers) plus cpsr (1
        // 32-bit register)
        if (count >= (33 * 2) + 1) {
          for (uint32_t i = 0; i < 29; ++i)
            gpr.x[i] = data.GetU64(&offset);
          gpr.fp = data.GetU64(&offset);
          gpr.lr = data.GetU64(&offset);
          gpr.sp = data.GetU64(&offset);
          gpr.pc = data.GetU64(&offset);
          gpr.cpsr = data.GetU32(&offset);
          SetError(GPRRegSet, Read, 0);
        }
        offset = next_thread_state;
        break;
      case FPURegSet: {
        uint8_t *fpu_reg_buf = (uint8_t *)&fpu.v[0];
        const int fpu_reg_buf_size = sizeof(fpu);
        if (fpu_reg_buf_size == count * sizeof(uint32_t) &&
            data.ExtractBytes(offset, fpu_reg_buf_size, eByteOrderLittle,
                              fpu_reg_buf) == fpu_reg_buf_size) {
          SetError(FPURegSet, Read, 0);
        } else {
          done = true;
        }
      }
        offset = next_thread_state;
        break;
      case EXCRegSet:
        if (count == 4) {
          exc.far = data.GetU64(&offset);
          exc.esr = data.GetU32(&offset);
          exc.exception = data.GetU32(&offset);
          SetError(EXCRegSet, Read, 0);
        }
        offset = next_thread_state;
        break;
      default:
        done = true;
        break;
      }
    }
  }

  static size_t WriteRegister(RegisterContext *reg_ctx, const char *name,
                              const char *alt_name, size_t reg_byte_size,
                              Stream &data) {
    const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(name);
    if (reg_info == NULL)
      reg_info = reg_ctx->GetRegisterInfoByName(alt_name);
    if (reg_info) {
      lldb_private::RegisterValue reg_value;
      if (reg_ctx->ReadRegister(reg_info, reg_value)) {
        if (reg_info->byte_size >= reg_byte_size)
          data.Write(reg_value.GetBytes(), reg_byte_size);
        else {
          data.Write(reg_value.GetBytes(), reg_info->byte_size);
          for (size_t i = 0, n = reg_byte_size - reg_info->byte_size; i < n;
               ++i)
            data.PutChar(0);
        }
        return reg_byte_size;
      }
    }
    // Just write zeros if all else fails
    for (size_t i = 0; i < reg_byte_size; ++i)
      data.PutChar(0);
    return reg_byte_size;
  }

  static bool Create_LC_THREAD(Thread *thread, Stream &data) {
    RegisterContextSP reg_ctx_sp(thread->GetRegisterContext());
    if (reg_ctx_sp) {
      RegisterContext *reg_ctx = reg_ctx_sp.get();

      data.PutHex32(GPRRegSet); // Flavor
      data.PutHex32(GPRWordCount);
      WriteRegister(reg_ctx, "x0", NULL, 8, data);
      WriteRegister(reg_ctx, "x1", NULL, 8, data);
      WriteRegister(reg_ctx, "x2", NULL, 8, data);
      WriteRegister(reg_ctx, "x3", NULL, 8, data);
      WriteRegister(reg_ctx, "x4", NULL, 8, data);
      WriteRegister(reg_ctx, "x5", NULL, 8, data);
      WriteRegister(reg_ctx, "x6", NULL, 8, data);
      WriteRegister(reg_ctx, "x7", NULL, 8, data);
      WriteRegister(reg_ctx, "x8", NULL, 8, data);
      WriteRegister(reg_ctx, "x9", NULL, 8, data);
      WriteRegister(reg_ctx, "x10", NULL, 8, data);
      WriteRegister(reg_ctx, "x11", NULL, 8, data);
      WriteRegister(reg_ctx, "x12", NULL, 8, data);
      WriteRegister(reg_ctx, "x13", NULL, 8, data);
      WriteRegister(reg_ctx, "x14", NULL, 8, data);
      WriteRegister(reg_ctx, "x15", NULL, 8, data);
      WriteRegister(reg_ctx, "x16", NULL, 8, data);
      WriteRegister(reg_ctx, "x17", NULL, 8, data);
      WriteRegister(reg_ctx, "x18", NULL, 8, data);
      WriteRegister(reg_ctx, "x19", NULL, 8, data);
      WriteRegister(reg_ctx, "x20", NULL, 8, data);
      WriteRegister(reg_ctx, "x21", NULL, 8, data);
      WriteRegister(reg_ctx, "x22", NULL, 8, data);
      WriteRegister(reg_ctx, "x23", NULL, 8, data);
      WriteRegister(reg_ctx, "x24", NULL, 8, data);
      WriteRegister(reg_ctx, "x25", NULL, 8, data);
      WriteRegister(reg_ctx, "x26", NULL, 8, data);
      WriteRegister(reg_ctx, "x27", NULL, 8, data);
      WriteRegister(reg_ctx, "x28", NULL, 8, data);
      WriteRegister(reg_ctx, "fp", NULL, 8, data);
      WriteRegister(reg_ctx, "lr", NULL, 8, data);
      WriteRegister(reg_ctx, "sp", NULL, 8, data);
      WriteRegister(reg_ctx, "pc", NULL, 8, data);
      WriteRegister(reg_ctx, "cpsr", NULL, 4, data);

      // Write out the EXC registers
      //            data.PutHex32 (EXCRegSet);
      //            data.PutHex32 (EXCWordCount);
      //            WriteRegister (reg_ctx, "far", NULL, 8, data);
      //            WriteRegister (reg_ctx, "esr", NULL, 4, data);
      //            WriteRegister (reg_ctx, "exception", NULL, 4, data);
      return true;
    }
    return false;
  }

protected:
  int DoReadGPR(lldb::tid_t tid, int flavor, GPR &gpr) override { return -1; }

  int DoReadFPU(lldb::tid_t tid, int flavor, FPU &fpu) override { return -1; }

  int DoReadEXC(lldb::tid_t tid, int flavor, EXC &exc) override { return -1; }

  int DoReadDBG(lldb::tid_t tid, int flavor, DBG &dbg) override { return -1; }

  int DoWriteGPR(lldb::tid_t tid, int flavor, const GPR &gpr) override {
    return 0;
  }

  int DoWriteFPU(lldb::tid_t tid, int flavor, const FPU &fpu) override {
    return 0;
  }

  int DoWriteEXC(lldb::tid_t tid, int flavor, const EXC &exc) override {
    return 0;
  }

  int DoWriteDBG(lldb::tid_t tid, int flavor, const DBG &dbg) override {
    return -1;
  }
};

static uint32_t MachHeaderSizeFromMagic(uint32_t magic) {
  switch (magic) {
  case MH_MAGIC:
  case MH_CIGAM:
    return sizeof(struct mach_header);

  case MH_MAGIC_64:
  case MH_CIGAM_64:
    return sizeof(struct mach_header_64);
    break;

  default:
    break;
  }
  return 0;
}

#define MACHO_NLIST_ARM_SYMBOL_IS_THUMB 0x0008

void ObjectFileMachO::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
      CreateMemoryInstance, GetModuleSpecifications, SaveCore);
}

void ObjectFileMachO::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString ObjectFileMachO::GetPluginNameStatic() {
  static ConstString g_name("mach-o");
  return g_name;
}

const char *ObjectFileMachO::GetPluginDescriptionStatic() {
  return "Mach-o object file reader (32 and 64 bit)";
}

ObjectFile *ObjectFileMachO::CreateInstance(const lldb::ModuleSP &module_sp,
                                            DataBufferSP &data_sp,
                                            lldb::offset_t data_offset,
                                            const FileSpec *file,
                                            lldb::offset_t file_offset,
                                            lldb::offset_t length) {
  if (!data_sp) {
    data_sp = MapFileData(*file, length, file_offset);
    if (!data_sp)
      return nullptr;
    data_offset = 0;
  }

  if (!ObjectFileMachO::MagicBytesMatch(data_sp, data_offset, length))
    return nullptr;

  // Update the data to contain the entire file if it doesn't already
  if (data_sp->GetByteSize() < length) {
    data_sp = MapFileData(*file, length, file_offset);
    if (!data_sp)
      return nullptr;
    data_offset = 0;
  }
  auto objfile_ap = llvm::make_unique<ObjectFileMachO>(
      module_sp, data_sp, data_offset, file, file_offset, length);
  if (!objfile_ap || !objfile_ap->ParseHeader())
    return nullptr;

  return objfile_ap.release();
}

ObjectFile *ObjectFileMachO::CreateMemoryInstance(
    const lldb::ModuleSP &module_sp, DataBufferSP &data_sp,
    const ProcessSP &process_sp, lldb::addr_t header_addr) {
  if (ObjectFileMachO::MagicBytesMatch(data_sp, 0, data_sp->GetByteSize())) {
    std::unique_ptr<ObjectFile> objfile_ap(
        new ObjectFileMachO(module_sp, data_sp, process_sp, header_addr));
    if (objfile_ap.get() && objfile_ap->ParseHeader())
      return objfile_ap.release();
  }
  return NULL;
}

size_t ObjectFileMachO::GetModuleSpecifications(
    const lldb_private::FileSpec &file, lldb::DataBufferSP &data_sp,
    lldb::offset_t data_offset, lldb::offset_t file_offset,
    lldb::offset_t length, lldb_private::ModuleSpecList &specs) {
  const size_t initial_count = specs.GetSize();

  if (ObjectFileMachO::MagicBytesMatch(data_sp, 0, data_sp->GetByteSize())) {
    DataExtractor data;
    data.SetData(data_sp);
    llvm::MachO::mach_header header;
    if (ParseHeader(data, &data_offset, header)) {
      size_t header_and_load_cmds =
          header.sizeofcmds + MachHeaderSizeFromMagic(header.magic);
      if (header_and_load_cmds >= data_sp->GetByteSize()) {
        data_sp = MapFileData(file, header_and_load_cmds, file_offset);
        data.SetData(data_sp);
        data_offset = MachHeaderSizeFromMagic(header.magic);
      }
      if (data_sp) {
        ModuleSpec spec;
        spec.GetFileSpec() = file;
        spec.SetObjectOffset(file_offset);
        spec.SetObjectSize(length);

        if (GetArchitecture(header, data, data_offset,
                            spec.GetArchitecture())) {
          if (spec.GetArchitecture().IsValid()) {
            GetUUID(header, data, data_offset, spec.GetUUID());
            specs.Append(spec);
          }
        }
      }
    }
  }
  return specs.GetSize() - initial_count;
}

const ConstString &ObjectFileMachO::GetSegmentNameTEXT() {
  static ConstString g_segment_name_TEXT("__TEXT");
  return g_segment_name_TEXT;
}

const ConstString &ObjectFileMachO::GetSegmentNameDATA() {
  static ConstString g_segment_name_DATA("__DATA");
  return g_segment_name_DATA;
}

const ConstString &ObjectFileMachO::GetSegmentNameDATA_DIRTY() {
  static ConstString g_segment_name("__DATA_DIRTY");
  return g_segment_name;
}

const ConstString &ObjectFileMachO::GetSegmentNameDATA_CONST() {
  static ConstString g_segment_name("__DATA_CONST");
  return g_segment_name;
}

const ConstString &ObjectFileMachO::GetSegmentNameOBJC() {
  static ConstString g_segment_name_OBJC("__OBJC");
  return g_segment_name_OBJC;
}

const ConstString &ObjectFileMachO::GetSegmentNameLINKEDIT() {
  static ConstString g_section_name_LINKEDIT("__LINKEDIT");
  return g_section_name_LINKEDIT;
}

const ConstString &ObjectFileMachO::GetSectionNameEHFrame() {
  static ConstString g_section_name_eh_frame("__eh_frame");
  return g_section_name_eh_frame;
}

bool ObjectFileMachO::MagicBytesMatch(DataBufferSP &data_sp,
                                      lldb::addr_t data_offset,
                                      lldb::addr_t data_length) {
  DataExtractor data;
  data.SetData(data_sp, data_offset, data_length);
  lldb::offset_t offset = 0;
  uint32_t magic = data.GetU32(&offset);
  return MachHeaderSizeFromMagic(magic) != 0;
}

ObjectFileMachO::ObjectFileMachO(const lldb::ModuleSP &module_sp,
                                 DataBufferSP &data_sp,
                                 lldb::offset_t data_offset,
                                 const FileSpec *file,
                                 lldb::offset_t file_offset,
                                 lldb::offset_t length)
    : ObjectFile(module_sp, file, file_offset, length, data_sp, data_offset),
      m_mach_segments(), m_mach_sections(), m_entry_point_address(),
      m_thread_context_offsets(), m_thread_context_offsets_valid(false),
      m_reexported_dylibs(), m_allow_assembly_emulation_unwind_plans(true) {
  ::memset(&m_header, 0, sizeof(m_header));
  ::memset(&m_dysymtab, 0, sizeof(m_dysymtab));
}

ObjectFileMachO::ObjectFileMachO(const lldb::ModuleSP &module_sp,
                                 lldb::DataBufferSP &header_data_sp,
                                 const lldb::ProcessSP &process_sp,
                                 lldb::addr_t header_addr)
    : ObjectFile(module_sp, process_sp, header_addr, header_data_sp),
      m_mach_segments(), m_mach_sections(), m_entry_point_address(),
      m_thread_context_offsets(), m_thread_context_offsets_valid(false),
      m_reexported_dylibs(), m_allow_assembly_emulation_unwind_plans(true) {
  ::memset(&m_header, 0, sizeof(m_header));
  ::memset(&m_dysymtab, 0, sizeof(m_dysymtab));
}

bool ObjectFileMachO::ParseHeader(DataExtractor &data,
                                  lldb::offset_t *data_offset_ptr,
                                  llvm::MachO::mach_header &header) {
  data.SetByteOrder(endian::InlHostByteOrder());
  // Leave magic in the original byte order
  header.magic = data.GetU32(data_offset_ptr);
  bool can_parse = false;
  bool is_64_bit = false;
  switch (header.magic) {
  case MH_MAGIC:
    data.SetByteOrder(endian::InlHostByteOrder());
    data.SetAddressByteSize(4);
    can_parse = true;
    break;

  case MH_MAGIC_64:
    data.SetByteOrder(endian::InlHostByteOrder());
    data.SetAddressByteSize(8);
    can_parse = true;
    is_64_bit = true;
    break;

  case MH_CIGAM:
    data.SetByteOrder(endian::InlHostByteOrder() == eByteOrderBig
                          ? eByteOrderLittle
                          : eByteOrderBig);
    data.SetAddressByteSize(4);
    can_parse = true;
    break;

  case MH_CIGAM_64:
    data.SetByteOrder(endian::InlHostByteOrder() == eByteOrderBig
                          ? eByteOrderLittle
                          : eByteOrderBig);
    data.SetAddressByteSize(8);
    is_64_bit = true;
    can_parse = true;
    break;

  default:
    break;
  }

  if (can_parse) {
    data.GetU32(data_offset_ptr, &header.cputype, 6);
    if (is_64_bit)
      *data_offset_ptr += 4;
    return true;
  } else {
    memset(&header, 0, sizeof(header));
  }
  return false;
}

bool ObjectFileMachO::ParseHeader() {
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    bool can_parse = false;
    lldb::offset_t offset = 0;
    m_data.SetByteOrder(endian::InlHostByteOrder());
    // Leave magic in the original byte order
    m_header.magic = m_data.GetU32(&offset);
    switch (m_header.magic) {
    case MH_MAGIC:
      m_data.SetByteOrder(endian::InlHostByteOrder());
      m_data.SetAddressByteSize(4);
      can_parse = true;
      break;

    case MH_MAGIC_64:
      m_data.SetByteOrder(endian::InlHostByteOrder());
      m_data.SetAddressByteSize(8);
      can_parse = true;
      break;

    case MH_CIGAM:
      m_data.SetByteOrder(endian::InlHostByteOrder() == eByteOrderBig
                              ? eByteOrderLittle
                              : eByteOrderBig);
      m_data.SetAddressByteSize(4);
      can_parse = true;
      break;

    case MH_CIGAM_64:
      m_data.SetByteOrder(endian::InlHostByteOrder() == eByteOrderBig
                              ? eByteOrderLittle
                              : eByteOrderBig);
      m_data.SetAddressByteSize(8);
      can_parse = true;
      break;

    default:
      break;
    }

    if (can_parse) {
      m_data.GetU32(&offset, &m_header.cputype, 6);

      ArchSpec mach_arch;

      if (GetArchitecture(mach_arch)) {
        // Check if the module has a required architecture
        const ArchSpec &module_arch = module_sp->GetArchitecture();
        if (module_arch.IsValid() && !module_arch.IsCompatibleMatch(mach_arch))
          return false;

        if (SetModulesArchitecture(mach_arch)) {
          const size_t header_and_lc_size =
              m_header.sizeofcmds + MachHeaderSizeFromMagic(m_header.magic);
          if (m_data.GetByteSize() < header_and_lc_size) {
            DataBufferSP data_sp;
            ProcessSP process_sp(m_process_wp.lock());
            if (process_sp) {
              data_sp =
                  ReadMemory(process_sp, m_memory_addr, header_and_lc_size);
            } else {
              // Read in all only the load command data from the file on disk
              data_sp = MapFileData(m_file, header_and_lc_size, m_file_offset);
              if (data_sp->GetByteSize() != header_and_lc_size)
                return false;
            }
            if (data_sp)
              m_data.SetData(data_sp);
          }
        }
        return true;
      }
    } else {
      memset(&m_header, 0, sizeof(struct mach_header));
    }
  }
  return false;
}

ByteOrder ObjectFileMachO::GetByteOrder() const {
  return m_data.GetByteOrder();
}

bool ObjectFileMachO::IsExecutable() const {
  return m_header.filetype == MH_EXECUTE;
}

uint32_t ObjectFileMachO::GetAddressByteSize() const {
  return m_data.GetAddressByteSize();
}

AddressClass ObjectFileMachO::GetAddressClass(lldb::addr_t file_addr) {
  Symtab *symtab = GetSymtab();
  if (symtab) {
    Symbol *symbol = symtab->FindSymbolContainingFileAddress(file_addr);
    if (symbol) {
      if (symbol->ValueIsAddress()) {
        SectionSP section_sp(symbol->GetAddressRef().GetSection());
        if (section_sp) {
          const lldb::SectionType section_type = section_sp->GetType();
          switch (section_type) {
          case eSectionTypeInvalid:
            return eAddressClassUnknown;

          case eSectionTypeCode:
            if (m_header.cputype == llvm::MachO::CPU_TYPE_ARM) {
              // For ARM we have a bit in the n_desc field of the symbol that
              // tells us ARM/Thumb which is bit 0x0008.
              if (symbol->GetFlags() & MACHO_NLIST_ARM_SYMBOL_IS_THUMB)
                return eAddressClassCodeAlternateISA;
            }
            return eAddressClassCode;

          case eSectionTypeContainer:
            return eAddressClassUnknown;

          case eSectionTypeData:
          case eSectionTypeDataCString:
          case eSectionTypeDataCStringPointers:
          case eSectionTypeDataSymbolAddress:
          case eSectionTypeData4:
          case eSectionTypeData8:
          case eSectionTypeData16:
          case eSectionTypeDataPointers:
          case eSectionTypeZeroFill:
          case eSectionTypeDataObjCMessageRefs:
          case eSectionTypeDataObjCCFStrings:
          case eSectionTypeGoSymtab:
            return eAddressClassData;

          case eSectionTypeDebug:
          case eSectionTypeDWARFDebugAbbrev:
          case eSectionTypeDWARFDebugAddr:
          case eSectionTypeDWARFDebugAranges:
          case eSectionTypeDWARFDebugCuIndex:
          case eSectionTypeDWARFDebugFrame:
          case eSectionTypeDWARFDebugInfo:
          case eSectionTypeDWARFDebugLine:
          case eSectionTypeDWARFDebugLoc:
          case eSectionTypeDWARFDebugMacInfo:
          case eSectionTypeDWARFDebugMacro:
          case eSectionTypeDWARFDebugNames:
          case eSectionTypeDWARFDebugPubNames:
          case eSectionTypeDWARFDebugPubTypes:
          case eSectionTypeDWARFDebugRanges:
          case eSectionTypeDWARFDebugStr:
          case eSectionTypeDWARFDebugStrOffsets:
          case eSectionTypeDWARFDebugTypes:
          case eSectionTypeDWARFAppleNames:
          case eSectionTypeDWARFAppleTypes:
          case eSectionTypeDWARFAppleNamespaces:
          case eSectionTypeDWARFAppleObjC:
          case eSectionTypeDWARFGNUDebugAltLink:
            return eAddressClassDebug;

          case eSectionTypeEHFrame:
          case eSectionTypeARMexidx:
          case eSectionTypeARMextab:
          case eSectionTypeCompactUnwind:
            return eAddressClassRuntime;

          case eSectionTypeAbsoluteAddress:
          case eSectionTypeELFSymbolTable:
          case eSectionTypeELFDynamicSymbols:
          case eSectionTypeELFRelocationEntries:
          case eSectionTypeELFDynamicLinkInfo:
          case eSectionTypeOther:
            return eAddressClassUnknown;
          }
        }
      }

      const SymbolType symbol_type = symbol->GetType();
      switch (symbol_type) {
      case eSymbolTypeAny:
        return eAddressClassUnknown;
      case eSymbolTypeAbsolute:
        return eAddressClassUnknown;

      case eSymbolTypeCode:
      case eSymbolTypeTrampoline:
      case eSymbolTypeResolver:
        if (m_header.cputype == llvm::MachO::CPU_TYPE_ARM) {
          // For ARM we have a bit in the n_desc field of the symbol that tells
          // us ARM/Thumb which is bit 0x0008.
          if (symbol->GetFlags() & MACHO_NLIST_ARM_SYMBOL_IS_THUMB)
            return eAddressClassCodeAlternateISA;
        }
        return eAddressClassCode;

      case eSymbolTypeData:
        return eAddressClassData;
      case eSymbolTypeRuntime:
        return eAddressClassRuntime;
      case eSymbolTypeException:
        return eAddressClassRuntime;
      case eSymbolTypeSourceFile:
        return eAddressClassDebug;
      case eSymbolTypeHeaderFile:
        return eAddressClassDebug;
      case eSymbolTypeObjectFile:
        return eAddressClassDebug;
      case eSymbolTypeCommonBlock:
        return eAddressClassDebug;
      case eSymbolTypeBlock:
        return eAddressClassDebug;
      case eSymbolTypeLocal:
        return eAddressClassData;
      case eSymbolTypeParam:
        return eAddressClassData;
      case eSymbolTypeVariable:
        return eAddressClassData;
      case eSymbolTypeVariableType:
        return eAddressClassDebug;
      case eSymbolTypeLineEntry:
        return eAddressClassDebug;
      case eSymbolTypeLineHeader:
        return eAddressClassDebug;
      case eSymbolTypeScopeBegin:
        return eAddressClassDebug;
      case eSymbolTypeScopeEnd:
        return eAddressClassDebug;
      case eSymbolTypeAdditional:
        return eAddressClassUnknown;
      case eSymbolTypeCompiler:
        return eAddressClassDebug;
      case eSymbolTypeInstrumentation:
        return eAddressClassDebug;
      case eSymbolTypeUndefined:
        return eAddressClassUnknown;
      case eSymbolTypeObjCClass:
        return eAddressClassRuntime;
      case eSymbolTypeObjCMetaClass:
        return eAddressClassRuntime;
      case eSymbolTypeObjCIVar:
        return eAddressClassRuntime;
      case eSymbolTypeReExported:
        return eAddressClassRuntime;
      }
    }
  }
  return eAddressClassUnknown;
}

Symtab *ObjectFileMachO::GetSymtab() {
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    if (m_symtab_ap.get() == NULL) {
      m_symtab_ap.reset(new Symtab(this));
      std::lock_guard<std::recursive_mutex> symtab_guard(
          m_symtab_ap->GetMutex());
      ParseSymtab();
      m_symtab_ap->Finalize();
    }
  }
  return m_symtab_ap.get();
}

bool ObjectFileMachO::IsStripped() {
  if (m_dysymtab.cmd == 0) {
    ModuleSP module_sp(GetModule());
    if (module_sp) {
      lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
      for (uint32_t i = 0; i < m_header.ncmds; ++i) {
        const lldb::offset_t load_cmd_offset = offset;

        load_command lc;
        if (m_data.GetU32(&offset, &lc.cmd, 2) == NULL)
          break;
        if (lc.cmd == LC_DYSYMTAB) {
          m_dysymtab.cmd = lc.cmd;
          m_dysymtab.cmdsize = lc.cmdsize;
          if (m_data.GetU32(&offset, &m_dysymtab.ilocalsym,
                            (sizeof(m_dysymtab) / sizeof(uint32_t)) - 2) ==
              NULL) {
            // Clear m_dysymtab if we were unable to read all items from the
            // load command
            ::memset(&m_dysymtab, 0, sizeof(m_dysymtab));
          }
        }
        offset = load_cmd_offset + lc.cmdsize;
      }
    }
  }
  if (m_dysymtab.cmd)
    return m_dysymtab.nlocalsym <= 1;
  return false;
}

ObjectFileMachO::EncryptedFileRanges ObjectFileMachO::GetEncryptedFileRanges() {
  EncryptedFileRanges result;
  lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);

  encryption_info_command encryption_cmd;
  for (uint32_t i = 0; i < m_header.ncmds; ++i) {
    const lldb::offset_t load_cmd_offset = offset;
    if (m_data.GetU32(&offset, &encryption_cmd, 2) == NULL)
      break;

    // LC_ENCRYPTION_INFO and LC_ENCRYPTION_INFO_64 have the same sizes for the
    // 3 fields we care about, so treat them the same.
    if (encryption_cmd.cmd == LC_ENCRYPTION_INFO ||
        encryption_cmd.cmd == LC_ENCRYPTION_INFO_64) {
      if (m_data.GetU32(&offset, &encryption_cmd.cryptoff, 3)) {
        if (encryption_cmd.cryptid != 0) {
          EncryptedFileRanges::Entry entry;
          entry.SetRangeBase(encryption_cmd.cryptoff);
          entry.SetByteSize(encryption_cmd.cryptsize);
          result.Append(entry);
        }
      }
    }
    offset = load_cmd_offset + encryption_cmd.cmdsize;
  }

  return result;
}

void ObjectFileMachO::SanitizeSegmentCommand(segment_command_64 &seg_cmd,
                                             uint32_t cmd_idx) {
  if (m_length == 0 || seg_cmd.filesize == 0)
    return;

  if (seg_cmd.fileoff > m_length) {
    // We have a load command that says it extends past the end of the file.
    // This is likely a corrupt file.  We don't have any way to return an error
    // condition here (this method was likely invoked from something like
    // ObjectFile::GetSectionList()), so we just null out the section contents,
    // and dump a message to stdout.  The most common case here is core file
    // debugging with a truncated file.
    const char *lc_segment_name =
        seg_cmd.cmd == LC_SEGMENT_64 ? "LC_SEGMENT_64" : "LC_SEGMENT";
    GetModule()->ReportWarning(
        "load command %u %s has a fileoff (0x%" PRIx64
        ") that extends beyond the end of the file (0x%" PRIx64
        "), ignoring this section",
        cmd_idx, lc_segment_name, seg_cmd.fileoff, m_length);

    seg_cmd.fileoff = 0;
    seg_cmd.filesize = 0;
  }

  if (seg_cmd.fileoff + seg_cmd.filesize > m_length) {
    // We have a load command that says it extends past the end of the file.
    // This is likely a corrupt file.  We don't have any way to return an error
    // condition here (this method was likely invoked from something like
    // ObjectFile::GetSectionList()), so we just null out the section contents,
    // and dump a message to stdout.  The most common case here is core file
    // debugging with a truncated file.
    const char *lc_segment_name =
        seg_cmd.cmd == LC_SEGMENT_64 ? "LC_SEGMENT_64" : "LC_SEGMENT";
    GetModule()->ReportWarning(
        "load command %u %s has a fileoff + filesize (0x%" PRIx64
        ") that extends beyond the end of the file (0x%" PRIx64
        "), the segment will be truncated to match",
        cmd_idx, lc_segment_name, seg_cmd.fileoff + seg_cmd.filesize, m_length);

    // Truncate the length
    seg_cmd.filesize = m_length - seg_cmd.fileoff;
  }
}

static uint32_t GetSegmentPermissions(const segment_command_64 &seg_cmd) {
  uint32_t result = 0;
  if (seg_cmd.initprot & VM_PROT_READ)
    result |= ePermissionsReadable;
  if (seg_cmd.initprot & VM_PROT_WRITE)
    result |= ePermissionsWritable;
  if (seg_cmd.initprot & VM_PROT_EXECUTE)
    result |= ePermissionsExecutable;
  return result;
}

static lldb::SectionType GetSectionType(uint32_t flags,
                                        ConstString section_name) {

  if (flags & (S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS))
    return eSectionTypeCode;

  uint32_t mach_sect_type = flags & SECTION_TYPE;
  static ConstString g_sect_name_objc_data("__objc_data");
  static ConstString g_sect_name_objc_msgrefs("__objc_msgrefs");
  static ConstString g_sect_name_objc_selrefs("__objc_selrefs");
  static ConstString g_sect_name_objc_classrefs("__objc_classrefs");
  static ConstString g_sect_name_objc_superrefs("__objc_superrefs");
  static ConstString g_sect_name_objc_const("__objc_const");
  static ConstString g_sect_name_objc_classlist("__objc_classlist");
  static ConstString g_sect_name_cfstring("__cfstring");

  static ConstString g_sect_name_dwarf_debug_abbrev("__debug_abbrev");
  static ConstString g_sect_name_dwarf_debug_aranges("__debug_aranges");
  static ConstString g_sect_name_dwarf_debug_frame("__debug_frame");
  static ConstString g_sect_name_dwarf_debug_info("__debug_info");
  static ConstString g_sect_name_dwarf_debug_line("__debug_line");
  static ConstString g_sect_name_dwarf_debug_loc("__debug_loc");
  static ConstString g_sect_name_dwarf_debug_macinfo("__debug_macinfo");
  static ConstString g_sect_name_dwarf_debug_names("__debug_names");
  static ConstString g_sect_name_dwarf_debug_pubnames("__debug_pubnames");
  static ConstString g_sect_name_dwarf_debug_pubtypes("__debug_pubtypes");
  static ConstString g_sect_name_dwarf_debug_ranges("__debug_ranges");
  static ConstString g_sect_name_dwarf_debug_str("__debug_str");
  static ConstString g_sect_name_dwarf_debug_types("__debug_types");
  static ConstString g_sect_name_dwarf_apple_names("__apple_names");
  static ConstString g_sect_name_dwarf_apple_types("__apple_types");
  static ConstString g_sect_name_dwarf_apple_namespaces("__apple_namespac");
  static ConstString g_sect_name_dwarf_apple_objc("__apple_objc");
  static ConstString g_sect_name_eh_frame("__eh_frame");
  static ConstString g_sect_name_compact_unwind("__unwind_info");
  static ConstString g_sect_name_text("__text");
  static ConstString g_sect_name_data("__data");
  static ConstString g_sect_name_go_symtab("__gosymtab");

  if (section_name == g_sect_name_dwarf_debug_abbrev)
    return eSectionTypeDWARFDebugAbbrev;
  if (section_name == g_sect_name_dwarf_debug_aranges)
    return eSectionTypeDWARFDebugAranges;
  if (section_name == g_sect_name_dwarf_debug_frame)
    return eSectionTypeDWARFDebugFrame;
  if (section_name == g_sect_name_dwarf_debug_info)
    return eSectionTypeDWARFDebugInfo;
  if (section_name == g_sect_name_dwarf_debug_line)
    return eSectionTypeDWARFDebugLine;
  if (section_name == g_sect_name_dwarf_debug_loc)
    return eSectionTypeDWARFDebugLoc;
  if (section_name == g_sect_name_dwarf_debug_macinfo)
    return eSectionTypeDWARFDebugMacInfo;
  if (section_name == g_sect_name_dwarf_debug_names)
    return eSectionTypeDWARFDebugNames;
  if (section_name == g_sect_name_dwarf_debug_pubnames)
    return eSectionTypeDWARFDebugPubNames;
  if (section_name == g_sect_name_dwarf_debug_pubtypes)
    return eSectionTypeDWARFDebugPubTypes;
  if (section_name == g_sect_name_dwarf_debug_ranges)
    return eSectionTypeDWARFDebugRanges;
  if (section_name == g_sect_name_dwarf_debug_str)
    return eSectionTypeDWARFDebugStr;
  if (section_name == g_sect_name_dwarf_debug_types)
    return eSectionTypeDWARFDebugTypes;
  if (section_name == g_sect_name_dwarf_apple_names)
    return eSectionTypeDWARFAppleNames;
  if (section_name == g_sect_name_dwarf_apple_types)
    return eSectionTypeDWARFAppleTypes;
  if (section_name == g_sect_name_dwarf_apple_namespaces)
    return eSectionTypeDWARFAppleNamespaces;
  if (section_name == g_sect_name_dwarf_apple_objc)
    return eSectionTypeDWARFAppleObjC;
  if (section_name == g_sect_name_objc_selrefs)
    return eSectionTypeDataCStringPointers;
  if (section_name == g_sect_name_objc_msgrefs)
    return eSectionTypeDataObjCMessageRefs;
  if (section_name == g_sect_name_eh_frame)
    return eSectionTypeEHFrame;
  if (section_name == g_sect_name_compact_unwind)
    return eSectionTypeCompactUnwind;
  if (section_name == g_sect_name_cfstring)
    return eSectionTypeDataObjCCFStrings;
  if (section_name == g_sect_name_go_symtab)
    return eSectionTypeGoSymtab;
  if (section_name == g_sect_name_objc_data ||
      section_name == g_sect_name_objc_classrefs ||
      section_name == g_sect_name_objc_superrefs ||
      section_name == g_sect_name_objc_const ||
      section_name == g_sect_name_objc_classlist) {
    return eSectionTypeDataPointers;
  }

  switch (mach_sect_type) {
  // TODO: categorize sections by other flags for regular sections
  case S_REGULAR:
    if (section_name == g_sect_name_text)
      return eSectionTypeCode;
    if (section_name == g_sect_name_data)
      return eSectionTypeData;
    return eSectionTypeOther;
  case S_ZEROFILL:
    return eSectionTypeZeroFill;
  case S_CSTRING_LITERALS: // section with only literal C strings
    return eSectionTypeDataCString;
  case S_4BYTE_LITERALS: // section with only 4 byte literals
    return eSectionTypeData4;
  case S_8BYTE_LITERALS: // section with only 8 byte literals
    return eSectionTypeData8;
  case S_LITERAL_POINTERS: // section with only pointers to literals
    return eSectionTypeDataPointers;
  case S_NON_LAZY_SYMBOL_POINTERS: // section with only non-lazy symbol pointers
    return eSectionTypeDataPointers;
  case S_LAZY_SYMBOL_POINTERS: // section with only lazy symbol pointers
    return eSectionTypeDataPointers;
  case S_SYMBOL_STUBS: // section with only symbol stubs, byte size of stub in
                       // the reserved2 field
    return eSectionTypeCode;
  case S_MOD_INIT_FUNC_POINTERS: // section with only function pointers for
                                 // initialization
    return eSectionTypeDataPointers;
  case S_MOD_TERM_FUNC_POINTERS: // section with only function pointers for
                                 // termination
    return eSectionTypeDataPointers;
  case S_COALESCED:
    return eSectionTypeOther;
  case S_GB_ZEROFILL:
    return eSectionTypeZeroFill;
  case S_INTERPOSING: // section with only pairs of function pointers for
                      // interposing
    return eSectionTypeCode;
  case S_16BYTE_LITERALS: // section with only 16 byte literals
    return eSectionTypeData16;
  case S_DTRACE_DOF:
    return eSectionTypeDebug;
  case S_LAZY_DYLIB_SYMBOL_POINTERS:
    return eSectionTypeDataPointers;
  default:
    return eSectionTypeOther;
  }
}

struct ObjectFileMachO::SegmentParsingContext {
  const EncryptedFileRanges EncryptedRanges;
  lldb_private::SectionList &UnifiedList;
  uint32_t NextSegmentIdx = 0;
  uint32_t NextSectionIdx = 0;
  bool FileAddressesChanged = false;

  SegmentParsingContext(EncryptedFileRanges EncryptedRanges,
                        lldb_private::SectionList &UnifiedList)
      : EncryptedRanges(std::move(EncryptedRanges)), UnifiedList(UnifiedList) {}
};

void ObjectFileMachO::ProcessSegmentCommand(const load_command &load_cmd_,
                                            lldb::offset_t offset,
                                            uint32_t cmd_idx,
                                            SegmentParsingContext &context) {
  segment_command_64 load_cmd;
  memcpy(&load_cmd, &load_cmd_, sizeof(load_cmd_));

  if (!m_data.GetU8(&offset, (uint8_t *)load_cmd.segname, 16))
    return;

  ModuleSP module_sp = GetModule();
  const bool is_core = GetType() == eTypeCoreFile;
  const bool is_dsym = (m_header.filetype == MH_DSYM);
  bool add_section = true;
  bool add_to_unified = true;
  ConstString const_segname(
      load_cmd.segname,
      std::min<size_t>(strlen(load_cmd.segname), sizeof(load_cmd.segname)));

  SectionSP unified_section_sp(
      context.UnifiedList.FindSectionByName(const_segname));
  if (is_dsym && unified_section_sp) {
    if (const_segname == GetSegmentNameLINKEDIT()) {
      // We need to keep the __LINKEDIT segment private to this object file
      // only
      add_to_unified = false;
    } else {
      // This is the dSYM file and this section has already been created by the
      // object file, no need to create it.
      add_section = false;
    }
  }
  load_cmd.vmaddr = m_data.GetAddress(&offset);
  load_cmd.vmsize = m_data.GetAddress(&offset);
  load_cmd.fileoff = m_data.GetAddress(&offset);
  load_cmd.filesize = m_data.GetAddress(&offset);
  if (!m_data.GetU32(&offset, &load_cmd.maxprot, 4))
    return;

  SanitizeSegmentCommand(load_cmd, cmd_idx);

  const uint32_t segment_permissions = GetSegmentPermissions(load_cmd);
  const bool segment_is_encrypted =
      (load_cmd.flags & SG_PROTECTED_VERSION_1) != 0;

  // Keep a list of mach segments around in case we need to get at data that
  // isn't stored in the abstracted Sections.
  m_mach_segments.push_back(load_cmd);

  // Use a segment ID of the segment index shifted left by 8 so they never
  // conflict with any of the sections.
  SectionSP segment_sp;
  if (add_section && (const_segname || is_core)) {
    segment_sp.reset(new Section(
        module_sp, // Module to which this section belongs
        this,      // Object file to which this sections belongs
        ++context.NextSegmentIdx
            << 8, // Section ID is the 1 based segment index
        // shifted right by 8 bits as not to collide with any of the 256
        // section IDs that are possible
        const_segname,         // Name of this section
        eSectionTypeContainer, // This section is a container of other
        // sections.
        load_cmd.vmaddr, // File VM address == addresses as they are
        // found in the object file
        load_cmd.vmsize,  // VM size in bytes of this section
        load_cmd.fileoff, // Offset to the data for this section in
        // the file
        load_cmd.filesize, // Size in bytes of this section as found
        // in the file
        0,                // Segments have no alignment information
        load_cmd.flags)); // Flags for this section

    segment_sp->SetIsEncrypted(segment_is_encrypted);
    m_sections_ap->AddSection(segment_sp);
    segment_sp->SetPermissions(segment_permissions);
    if (add_to_unified)
      context.UnifiedList.AddSection(segment_sp);
  } else if (unified_section_sp) {
    if (is_dsym && unified_section_sp->GetFileAddress() != load_cmd.vmaddr) {
      // Check to see if the module was read from memory?
      if (module_sp->GetObjectFile()->GetHeaderAddress().IsValid()) {
        // We have a module that is in memory and needs to have its file
        // address adjusted. We need to do this because when we load a file
        // from memory, its addresses will be slid already, yet the addresses
        // in the new symbol file will still be unslid.  Since everything is
        // stored as section offset, this shouldn't cause any problems.

        // Make sure we've parsed the symbol table from the ObjectFile before
        // we go around changing its Sections.
        module_sp->GetObjectFile()->GetSymtab();
        // eh_frame would present the same problems but we parse that on a per-
        // function basis as-needed so it's more difficult to remove its use of
        // the Sections.  Realistically, the environments where this code path
        // will be taken will not have eh_frame sections.

        unified_section_sp->SetFileAddress(load_cmd.vmaddr);

        // Notify the module that the section addresses have been changed once
        // we're done so any file-address caches can be updated.
        context.FileAddressesChanged = true;
      }
    }
    m_sections_ap->AddSection(unified_section_sp);
  }

  struct section_64 sect64;
  ::memset(&sect64, 0, sizeof(sect64));
  // Push a section into our mach sections for the section at index zero
  // (NO_SECT) if we don't have any mach sections yet...
  if (m_mach_sections.empty())
    m_mach_sections.push_back(sect64);
  uint32_t segment_sect_idx;
  const lldb::user_id_t first_segment_sectID = context.NextSectionIdx + 1;

  const uint32_t num_u32s = load_cmd.cmd == LC_SEGMENT ? 7 : 8;
  for (segment_sect_idx = 0; segment_sect_idx < load_cmd.nsects;
       ++segment_sect_idx) {
    if (m_data.GetU8(&offset, (uint8_t *)sect64.sectname,
                     sizeof(sect64.sectname)) == NULL)
      break;
    if (m_data.GetU8(&offset, (uint8_t *)sect64.segname,
                     sizeof(sect64.segname)) == NULL)
      break;
    sect64.addr = m_data.GetAddress(&offset);
    sect64.size = m_data.GetAddress(&offset);

    if (m_data.GetU32(&offset, &sect64.offset, num_u32s) == NULL)
      break;

    // Keep a list of mach sections around in case we need to get at data that
    // isn't stored in the abstracted Sections.
    m_mach_sections.push_back(sect64);

    if (add_section) {
      ConstString section_name(
          sect64.sectname,
          std::min<size_t>(strlen(sect64.sectname), sizeof(sect64.sectname)));
      if (!const_segname) {
        // We have a segment with no name so we need to conjure up segments
        // that correspond to the section's segname if there isn't already such
        // a section. If there is such a section, we resize the section so that
        // it spans all sections.  We also mark these sections as fake so
        // address matches don't hit if they land in the gaps between the child
        // sections.
        const_segname.SetTrimmedCStringWithLength(sect64.segname,
                                                  sizeof(sect64.segname));
        segment_sp = context.UnifiedList.FindSectionByName(const_segname);
        if (segment_sp.get()) {
          Section *segment = segment_sp.get();
          // Grow the section size as needed.
          const lldb::addr_t sect64_min_addr = sect64.addr;
          const lldb::addr_t sect64_max_addr = sect64_min_addr + sect64.size;
          const lldb::addr_t curr_seg_byte_size = segment->GetByteSize();
          const lldb::addr_t curr_seg_min_addr = segment->GetFileAddress();
          const lldb::addr_t curr_seg_max_addr =
              curr_seg_min_addr + curr_seg_byte_size;
          if (sect64_min_addr >= curr_seg_min_addr) {
            const lldb::addr_t new_seg_byte_size =
                sect64_max_addr - curr_seg_min_addr;
            // Only grow the section size if needed
            if (new_seg_byte_size > curr_seg_byte_size)
              segment->SetByteSize(new_seg_byte_size);
          } else {
            // We need to change the base address of the segment and adjust the
            // child section offsets for all existing children.
            const lldb::addr_t slide_amount =
                sect64_min_addr - curr_seg_min_addr;
            segment->Slide(slide_amount, false);
            segment->GetChildren().Slide(-slide_amount, false);
            segment->SetByteSize(curr_seg_max_addr - sect64_min_addr);
          }

          // Grow the section size as needed.
          if (sect64.offset) {
            const lldb::addr_t segment_min_file_offset =
                segment->GetFileOffset();
            const lldb::addr_t segment_max_file_offset =
                segment_min_file_offset + segment->GetFileSize();

            const lldb::addr_t section_min_file_offset = sect64.offset;
            const lldb::addr_t section_max_file_offset =
                section_min_file_offset + sect64.size;
            const lldb::addr_t new_file_offset =
                std::min(section_min_file_offset, segment_min_file_offset);
            const lldb::addr_t new_file_size =
                std::max(section_max_file_offset, segment_max_file_offset) -
                new_file_offset;
            segment->SetFileOffset(new_file_offset);
            segment->SetFileSize(new_file_size);
          }
        } else {
          // Create a fake section for the section's named segment
          segment_sp.reset(new Section(
              segment_sp, // Parent section
              module_sp,  // Module to which this section belongs
              this,       // Object file to which this section belongs
              ++context.NextSegmentIdx
                  << 8, // Section ID is the 1 based segment index
              // shifted right by 8 bits as not to
              // collide with any of the 256 section IDs
              // that are possible
              const_segname,         // Name of this section
              eSectionTypeContainer, // This section is a container of
              // other sections.
              sect64.addr, // File VM address == addresses as they are
              // found in the object file
              sect64.size,   // VM size in bytes of this section
              sect64.offset, // Offset to the data for this section in
              // the file
              sect64.offset ? sect64.size : 0, // Size in bytes of
              // this section as
              // found in the file
              sect64.align,
              load_cmd.flags)); // Flags for this section
          segment_sp->SetIsFake(true);
          segment_sp->SetPermissions(segment_permissions);
          m_sections_ap->AddSection(segment_sp);
          if (add_to_unified)
            context.UnifiedList.AddSection(segment_sp);
          segment_sp->SetIsEncrypted(segment_is_encrypted);
        }
      }
      assert(segment_sp.get());

      lldb::SectionType sect_type = GetSectionType(sect64.flags, section_name);

      SectionSP section_sp(new Section(
          segment_sp, module_sp, this, ++context.NextSectionIdx, section_name,
          sect_type, sect64.addr - segment_sp->GetFileAddress(), sect64.size,
          sect64.offset, sect64.offset == 0 ? 0 : sect64.size, sect64.align,
          sect64.flags));
      // Set the section to be encrypted to match the segment

      bool section_is_encrypted = false;
      if (!segment_is_encrypted && load_cmd.filesize != 0)
        section_is_encrypted = context.EncryptedRanges.FindEntryThatContains(
                                   sect64.offset) != NULL;

      section_sp->SetIsEncrypted(segment_is_encrypted || section_is_encrypted);
      section_sp->SetPermissions(segment_permissions);
      segment_sp->GetChildren().AddSection(section_sp);

      if (segment_sp->IsFake()) {
        segment_sp.reset();
        const_segname.Clear();
      }
    }
  }
  if (segment_sp && is_dsym) {
    if (first_segment_sectID <= context.NextSectionIdx) {
      lldb::user_id_t sect_uid;
      for (sect_uid = first_segment_sectID; sect_uid <= context.NextSectionIdx;
           ++sect_uid) {
        SectionSP curr_section_sp(
            segment_sp->GetChildren().FindSectionByID(sect_uid));
        SectionSP next_section_sp;
        if (sect_uid + 1 <= context.NextSectionIdx)
          next_section_sp =
              segment_sp->GetChildren().FindSectionByID(sect_uid + 1);

        if (curr_section_sp.get()) {
          if (curr_section_sp->GetByteSize() == 0) {
            if (next_section_sp.get() != NULL)
              curr_section_sp->SetByteSize(next_section_sp->GetFileAddress() -
                                           curr_section_sp->GetFileAddress());
            else
              curr_section_sp->SetByteSize(load_cmd.vmsize);
          }
        }
      }
    }
  }
}

void ObjectFileMachO::ProcessDysymtabCommand(const load_command &load_cmd,
                                             lldb::offset_t offset) {
  m_dysymtab.cmd = load_cmd.cmd;
  m_dysymtab.cmdsize = load_cmd.cmdsize;
  m_data.GetU32(&offset, &m_dysymtab.ilocalsym,
                (sizeof(m_dysymtab) / sizeof(uint32_t)) - 2);
}

void ObjectFileMachO::CreateSections(SectionList &unified_section_list) {
  if (m_sections_ap)
    return;

  m_sections_ap.reset(new SectionList());

  lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
  // bool dump_sections = false;
  ModuleSP module_sp(GetModule());

  offset = MachHeaderSizeFromMagic(m_header.magic);

  SegmentParsingContext context(GetEncryptedFileRanges(), unified_section_list);
  struct load_command load_cmd;
  for (uint32_t i = 0; i < m_header.ncmds; ++i) {
    const lldb::offset_t load_cmd_offset = offset;
    if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
      break;

    if (load_cmd.cmd == LC_SEGMENT || load_cmd.cmd == LC_SEGMENT_64)
      ProcessSegmentCommand(load_cmd, offset, i, context);
    else if (load_cmd.cmd == LC_DYSYMTAB)
      ProcessDysymtabCommand(load_cmd, offset);

    offset = load_cmd_offset + load_cmd.cmdsize;
  }

  if (context.FileAddressesChanged && module_sp)
    module_sp->SectionFileAddressesChanged();
}

class MachSymtabSectionInfo {
public:
  MachSymtabSectionInfo(SectionList *section_list)
      : m_section_list(section_list), m_section_infos() {
    // Get the number of sections down to a depth of 1 to include all segments
    // and their sections, but no other sections that may be added for debug
    // map or
    m_section_infos.resize(section_list->GetNumSections(1));
  }

  SectionSP GetSection(uint8_t n_sect, addr_t file_addr) {
    if (n_sect == 0)
      return SectionSP();
    if (n_sect < m_section_infos.size()) {
      if (!m_section_infos[n_sect].section_sp) {
        SectionSP section_sp(m_section_list->FindSectionByID(n_sect));
        m_section_infos[n_sect].section_sp = section_sp;
        if (section_sp) {
          m_section_infos[n_sect].vm_range.SetBaseAddress(
              section_sp->GetFileAddress());
          m_section_infos[n_sect].vm_range.SetByteSize(
              section_sp->GetByteSize());
        } else {
          Host::SystemLog(Host::eSystemLogError,
                          "error: unable to find section for section %u\n",
                          n_sect);
        }
      }
      if (m_section_infos[n_sect].vm_range.Contains(file_addr)) {
        // Symbol is in section.
        return m_section_infos[n_sect].section_sp;
      } else if (m_section_infos[n_sect].vm_range.GetByteSize() == 0 &&
                 m_section_infos[n_sect].vm_range.GetBaseAddress() ==
                     file_addr) {
        // Symbol is in section with zero size, but has the same start address
        // as the section. This can happen with linker symbols (symbols that
        // start with the letter 'l' or 'L'.
        return m_section_infos[n_sect].section_sp;
      }
    }
    return m_section_list->FindSectionContainingFileAddress(file_addr);
  }

protected:
  struct SectionInfo {
    SectionInfo() : vm_range(), section_sp() {}

    VMRange vm_range;
    SectionSP section_sp;
  };
  SectionList *m_section_list;
  std::vector<SectionInfo> m_section_infos;
};

struct TrieEntry {
  TrieEntry()
      : name(), address(LLDB_INVALID_ADDRESS), flags(0), other(0),
        import_name() {}

  void Clear() {
    name.Clear();
    address = LLDB_INVALID_ADDRESS;
    flags = 0;
    other = 0;
    import_name.Clear();
  }

  void Dump() const {
    printf("0x%16.16llx 0x%16.16llx 0x%16.16llx \"%s\"",
           static_cast<unsigned long long>(address),
           static_cast<unsigned long long>(flags),
           static_cast<unsigned long long>(other), name.GetCString());
    if (import_name)
      printf(" -> \"%s\"\n", import_name.GetCString());
    else
      printf("\n");
  }
  ConstString name;
  uint64_t address;
  uint64_t flags;
  uint64_t other;
  ConstString import_name;
};

struct TrieEntryWithOffset {
  lldb::offset_t nodeOffset;
  TrieEntry entry;

  TrieEntryWithOffset(lldb::offset_t offset) : nodeOffset(offset), entry() {}

  void Dump(uint32_t idx) const {
    printf("[%3u] 0x%16.16llx: ", idx,
           static_cast<unsigned long long>(nodeOffset));
    entry.Dump();
  }

  bool operator<(const TrieEntryWithOffset &other) const {
    return (nodeOffset < other.nodeOffset);
  }
};

static bool ParseTrieEntries(DataExtractor &data, lldb::offset_t offset,
                             const bool is_arm,
                             std::vector<llvm::StringRef> &nameSlices,
                             std::set<lldb::addr_t> &resolver_addresses,
                             std::vector<TrieEntryWithOffset> &output) {
  if (!data.ValidOffset(offset))
    return true;

  const uint64_t terminalSize = data.GetULEB128(&offset);
  lldb::offset_t children_offset = offset + terminalSize;
  if (terminalSize != 0) {
    TrieEntryWithOffset e(offset);
    e.entry.flags = data.GetULEB128(&offset);
    const char *import_name = NULL;
    if (e.entry.flags & EXPORT_SYMBOL_FLAGS_REEXPORT) {
      e.entry.address = 0;
      e.entry.other = data.GetULEB128(&offset); // dylib ordinal
      import_name = data.GetCStr(&offset);
    } else {
      e.entry.address = data.GetULEB128(&offset);
      if (e.entry.flags & EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER) {
        e.entry.other = data.GetULEB128(&offset);
        uint64_t resolver_addr = e.entry.other;
        if (is_arm)
          resolver_addr &= THUMB_ADDRESS_BIT_MASK;
        resolver_addresses.insert(resolver_addr);
      } else
        e.entry.other = 0;
    }
    // Only add symbols that are reexport symbols with a valid import name
    if (EXPORT_SYMBOL_FLAGS_REEXPORT & e.entry.flags && import_name &&
        import_name[0]) {
      std::string name;
      if (!nameSlices.empty()) {
        for (auto name_slice : nameSlices)
          name.append(name_slice.data(), name_slice.size());
      }
      if (name.size() > 1) {
        // Skip the leading '_'
        e.entry.name.SetCStringWithLength(name.c_str() + 1, name.size() - 1);
      }
      if (import_name) {
        // Skip the leading '_'
        e.entry.import_name.SetCString(import_name + 1);
      }
      output.push_back(e);
    }
  }

  const uint8_t childrenCount = data.GetU8(&children_offset);
  for (uint8_t i = 0; i < childrenCount; ++i) {
    const char *cstr = data.GetCStr(&children_offset);
    if (cstr)
      nameSlices.push_back(llvm::StringRef(cstr));
    else
      return false; // Corrupt data
    lldb::offset_t childNodeOffset = data.GetULEB128(&children_offset);
    if (childNodeOffset) {
      if (!ParseTrieEntries(data, childNodeOffset, is_arm, nameSlices,
                            resolver_addresses, output)) {
        return false;
      }
    }
    nameSlices.pop_back();
  }
  return true;
}

// Read the UUID out of a dyld_shared_cache file on-disk.
UUID ObjectFileMachO::GetSharedCacheUUID(FileSpec dyld_shared_cache,
                                         const ByteOrder byte_order,
                                         const uint32_t addr_byte_size) {
  UUID dsc_uuid;
  DataBufferSP DscData = MapFileData(
      dyld_shared_cache, sizeof(struct lldb_copy_dyld_cache_header_v1), 0);
  if (!DscData)
    return dsc_uuid;
  DataExtractor dsc_header_data(DscData, byte_order, addr_byte_size);

  char version_str[7];
  lldb::offset_t offset = 0;
  memcpy(version_str, dsc_header_data.GetData(&offset, 6), 6);
  version_str[6] = '\0';
  if (strcmp(version_str, "dyld_v") == 0) {
    offset = offsetof(struct lldb_copy_dyld_cache_header_v1, uuid);
    uint8_t uuid_bytes[sizeof(uuid_t)];
    memcpy(uuid_bytes, dsc_header_data.GetData(&offset, sizeof(uuid_t)),
           sizeof(uuid_t));
    dsc_uuid.SetBytes(uuid_bytes);
  }
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_SYMBOLS));
  if (log && dsc_uuid.IsValid()) {
    log->Printf("Shared cache %s has UUID %s", dyld_shared_cache.GetPath().c_str(), 
                dsc_uuid.GetAsString().c_str());
  }
  return dsc_uuid;
}

size_t ObjectFileMachO::ParseSymtab() {
  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, "ObjectFileMachO::ParseSymtab () module = %s",
                     m_file.GetFilename().AsCString(""));
  ModuleSP module_sp(GetModule());
  if (!module_sp)
    return 0;

  struct symtab_command symtab_load_command = {0, 0, 0, 0, 0, 0};
  struct linkedit_data_command function_starts_load_command = {0, 0, 0, 0};
  struct dyld_info_command dyld_info = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  typedef AddressDataArray<lldb::addr_t, bool, 100> FunctionStarts;
  FunctionStarts function_starts;
  lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
  uint32_t i;
  FileSpecList dylib_files;
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_SYMBOLS));
  static const llvm::StringRef g_objc_v2_prefix_class("_OBJC_CLASS_$_");
  static const llvm::StringRef g_objc_v2_prefix_metaclass("_OBJC_METACLASS_$_");
  static const llvm::StringRef g_objc_v2_prefix_ivar("_OBJC_IVAR_$_");

  for (i = 0; i < m_header.ncmds; ++i) {
    const lldb::offset_t cmd_offset = offset;
    // Read in the load command and load command size
    struct load_command lc;
    if (m_data.GetU32(&offset, &lc, 2) == NULL)
      break;
    // Watch for the symbol table load command
    switch (lc.cmd) {
    case LC_SYMTAB:
      symtab_load_command.cmd = lc.cmd;
      symtab_load_command.cmdsize = lc.cmdsize;
      // Read in the rest of the symtab load command
      if (m_data.GetU32(&offset, &symtab_load_command.symoff, 4) ==
          0) // fill in symoff, nsyms, stroff, strsize fields
        return 0;
      if (symtab_load_command.symoff == 0) {
        if (log)
          module_sp->LogMessage(log, "LC_SYMTAB.symoff == 0");
        return 0;
      }

      if (symtab_load_command.stroff == 0) {
        if (log)
          module_sp->LogMessage(log, "LC_SYMTAB.stroff == 0");
        return 0;
      }

      if (symtab_load_command.nsyms == 0) {
        if (log)
          module_sp->LogMessage(log, "LC_SYMTAB.nsyms == 0");
        return 0;
      }

      if (symtab_load_command.strsize == 0) {
        if (log)
          module_sp->LogMessage(log, "LC_SYMTAB.strsize == 0");
        return 0;
      }
      break;

    case LC_DYLD_INFO:
    case LC_DYLD_INFO_ONLY:
      if (m_data.GetU32(&offset, &dyld_info.rebase_off, 10)) {
        dyld_info.cmd = lc.cmd;
        dyld_info.cmdsize = lc.cmdsize;
      } else {
        memset(&dyld_info, 0, sizeof(dyld_info));
      }
      break;

    case LC_LOAD_DYLIB:
    case LC_LOAD_WEAK_DYLIB:
    case LC_REEXPORT_DYLIB:
    case LC_LOADFVMLIB:
    case LC_LOAD_UPWARD_DYLIB: {
      uint32_t name_offset = cmd_offset + m_data.GetU32(&offset);
      const char *path = m_data.PeekCStr(name_offset);
      if (path) {
        FileSpec file_spec(path, false);
        // Strip the path if there is @rpath, @executable, etc so we just use
        // the basename
        if (path[0] == '@')
          file_spec.GetDirectory().Clear();

        if (lc.cmd == LC_REEXPORT_DYLIB) {
          m_reexported_dylibs.AppendIfUnique(file_spec);
        }

        dylib_files.Append(file_spec);
      }
    } break;

    case LC_FUNCTION_STARTS:
      function_starts_load_command.cmd = lc.cmd;
      function_starts_load_command.cmdsize = lc.cmdsize;
      if (m_data.GetU32(&offset, &function_starts_load_command.dataoff, 2) ==
          NULL) // fill in symoff, nsyms, stroff, strsize fields
        memset(&function_starts_load_command, 0,
               sizeof(function_starts_load_command));
      break;

    default:
      break;
    }
    offset = cmd_offset + lc.cmdsize;
  }

  if (symtab_load_command.cmd) {
    Symtab *symtab = m_symtab_ap.get();
    SectionList *section_list = GetSectionList();
    if (section_list == NULL)
      return 0;

    const uint32_t addr_byte_size = m_data.GetAddressByteSize();
    const ByteOrder byte_order = m_data.GetByteOrder();
    bool bit_width_32 = addr_byte_size == 4;
    const size_t nlist_byte_size =
        bit_width_32 ? sizeof(struct nlist) : sizeof(struct nlist_64);

    DataExtractor nlist_data(NULL, 0, byte_order, addr_byte_size);
    DataExtractor strtab_data(NULL, 0, byte_order, addr_byte_size);
    DataExtractor function_starts_data(NULL, 0, byte_order, addr_byte_size);
    DataExtractor indirect_symbol_index_data(NULL, 0, byte_order,
                                             addr_byte_size);
    DataExtractor dyld_trie_data(NULL, 0, byte_order, addr_byte_size);

    const addr_t nlist_data_byte_size =
        symtab_load_command.nsyms * nlist_byte_size;
    const addr_t strtab_data_byte_size = symtab_load_command.strsize;
    addr_t strtab_addr = LLDB_INVALID_ADDRESS;

    ProcessSP process_sp(m_process_wp.lock());
    Process *process = process_sp.get();

    uint32_t memory_module_load_level = eMemoryModuleLoadLevelComplete;

    if (process && m_header.filetype != llvm::MachO::MH_OBJECT) {
      Target &target = process->GetTarget();

      memory_module_load_level = target.GetMemoryModuleLoadLevel();

      SectionSP linkedit_section_sp(
          section_list->FindSectionByName(GetSegmentNameLINKEDIT()));
      // Reading mach file from memory in a process or core file...

      if (linkedit_section_sp) {
        addr_t linkedit_load_addr =
            linkedit_section_sp->GetLoadBaseAddress(&target);
        if (linkedit_load_addr == LLDB_INVALID_ADDRESS) {
          // We might be trying to access the symbol table before the
          // __LINKEDIT's load address has been set in the target. We can't
          // fail to read the symbol table, so calculate the right address
          // manually
          linkedit_load_addr = CalculateSectionLoadAddressForMemoryImage(
              m_memory_addr, GetMachHeaderSection(), linkedit_section_sp.get());
        }

        const addr_t linkedit_file_offset =
            linkedit_section_sp->GetFileOffset();
        const addr_t symoff_addr = linkedit_load_addr +
                                   symtab_load_command.symoff -
                                   linkedit_file_offset;
        strtab_addr = linkedit_load_addr + symtab_load_command.stroff -
                      linkedit_file_offset;

        bool data_was_read = false;

#if defined(__APPLE__) &&                                                      \
    (defined(__arm__) || defined(__arm64__) || defined(__aarch64__))
        if (m_header.flags & 0x80000000u &&
            process->GetAddressByteSize() == sizeof(void *)) {
          // This mach-o memory file is in the dyld shared cache. If this
          // program is not remote and this is iOS, then this process will
          // share the same shared cache as the process we are debugging and we
          // can read the entire __LINKEDIT from the address space in this
          // process. This is a needed optimization that is used for local iOS
          // debugging only since all shared libraries in the shared cache do
          // not have corresponding files that exist in the file system of the
          // device. They have been combined into a single file. This means we
          // always have to load these files from memory. All of the symbol and
          // string tables from all of the __LINKEDIT sections from the shared
          // libraries in the shared cache have been merged into a single large
          // symbol and string table. Reading all of this symbol and string
          // table data across can slow down debug launch times, so we optimize
          // this by reading the memory for the __LINKEDIT section from this
          // process.

          UUID lldb_shared_cache;
          addr_t lldb_shared_cache_addr;
          GetLLDBSharedCacheUUID (lldb_shared_cache_addr, lldb_shared_cache);
          UUID process_shared_cache;
          addr_t process_shared_cache_addr;
          GetProcessSharedCacheUUID(process, process_shared_cache_addr, process_shared_cache);
          bool use_lldb_cache = true;
          if (lldb_shared_cache.IsValid() && process_shared_cache.IsValid() &&
              (lldb_shared_cache != process_shared_cache
               || process_shared_cache_addr != lldb_shared_cache_addr)) {
            use_lldb_cache = false;
          }

          PlatformSP platform_sp(target.GetPlatform());
          if (platform_sp && platform_sp->IsHost() && use_lldb_cache) {
            data_was_read = true;
            nlist_data.SetData((void *)symoff_addr, nlist_data_byte_size,
                               eByteOrderLittle);
            strtab_data.SetData((void *)strtab_addr, strtab_data_byte_size,
                                eByteOrderLittle);
            if (function_starts_load_command.cmd) {
              const addr_t func_start_addr =
                  linkedit_load_addr + function_starts_load_command.dataoff -
                  linkedit_file_offset;
              function_starts_data.SetData(
                  (void *)func_start_addr,
                  function_starts_load_command.datasize, eByteOrderLittle);
            }
          }
        }
#endif

        if (!data_was_read) {
          // Always load dyld - the dynamic linker - from memory if we didn't
          // find a binary anywhere else. lldb will not register
          // dylib/framework/bundle loads/unloads if we don't have the dyld
          // symbols, we force dyld to load from memory despite the user's
          // target.memory-module-load-level setting.
          if (memory_module_load_level == eMemoryModuleLoadLevelComplete ||
              m_header.filetype == llvm::MachO::MH_DYLINKER) {
            DataBufferSP nlist_data_sp(
                ReadMemory(process_sp, symoff_addr, nlist_data_byte_size));
            if (nlist_data_sp)
              nlist_data.SetData(nlist_data_sp, 0,
                                 nlist_data_sp->GetByteSize());
            // Load strings individually from memory when loading from memory
            // since shared cache string tables contain strings for all symbols
            // from all shared cached libraries DataBufferSP strtab_data_sp
            // (ReadMemory (process_sp, strtab_addr,
            // strtab_data_byte_size));
            // if (strtab_data_sp)
            //    strtab_data.SetData (strtab_data_sp, 0,
            //    strtab_data_sp->GetByteSize());
            if (m_dysymtab.nindirectsyms != 0) {
              const addr_t indirect_syms_addr = linkedit_load_addr +
                                                m_dysymtab.indirectsymoff -
                                                linkedit_file_offset;
              DataBufferSP indirect_syms_data_sp(
                  ReadMemory(process_sp, indirect_syms_addr,
                             m_dysymtab.nindirectsyms * 4));
              if (indirect_syms_data_sp)
                indirect_symbol_index_data.SetData(
                    indirect_syms_data_sp, 0,
                    indirect_syms_data_sp->GetByteSize());
            }
          } else if (memory_module_load_level >=
                     eMemoryModuleLoadLevelPartial) {
            if (function_starts_load_command.cmd) {
              const addr_t func_start_addr =
                  linkedit_load_addr + function_starts_load_command.dataoff -
                  linkedit_file_offset;
              DataBufferSP func_start_data_sp(
                  ReadMemory(process_sp, func_start_addr,
                             function_starts_load_command.datasize));
              if (func_start_data_sp)
                function_starts_data.SetData(func_start_data_sp, 0,
                                             func_start_data_sp->GetByteSize());
            }
          }
        }
      }
    } else {
      nlist_data.SetData(m_data, symtab_load_command.symoff,
                         nlist_data_byte_size);
      strtab_data.SetData(m_data, symtab_load_command.stroff,
                          strtab_data_byte_size);

      if (dyld_info.export_size > 0) {
        dyld_trie_data.SetData(m_data, dyld_info.export_off,
                               dyld_info.export_size);
      }

      if (m_dysymtab.nindirectsyms != 0) {
        indirect_symbol_index_data.SetData(m_data, m_dysymtab.indirectsymoff,
                                           m_dysymtab.nindirectsyms * 4);
      }
      if (function_starts_load_command.cmd) {
        function_starts_data.SetData(m_data,
                                     function_starts_load_command.dataoff,
                                     function_starts_load_command.datasize);
      }
    }

    if (nlist_data.GetByteSize() == 0 &&
        memory_module_load_level == eMemoryModuleLoadLevelComplete) {
      if (log)
        module_sp->LogMessage(log, "failed to read nlist data");
      return 0;
    }

    const bool have_strtab_data = strtab_data.GetByteSize() > 0;
    if (!have_strtab_data) {
      if (process) {
        if (strtab_addr == LLDB_INVALID_ADDRESS) {
          if (log)
            module_sp->LogMessage(log, "failed to locate the strtab in memory");
          return 0;
        }
      } else {
        if (log)
          module_sp->LogMessage(log, "failed to read strtab data");
        return 0;
      }
    }

    const ConstString &g_segment_name_TEXT = GetSegmentNameTEXT();
    const ConstString &g_segment_name_DATA = GetSegmentNameDATA();
    const ConstString &g_segment_name_DATA_DIRTY = GetSegmentNameDATA_DIRTY();
    const ConstString &g_segment_name_DATA_CONST = GetSegmentNameDATA_CONST();
    const ConstString &g_segment_name_OBJC = GetSegmentNameOBJC();
    const ConstString &g_section_name_eh_frame = GetSectionNameEHFrame();
    SectionSP text_section_sp(
        section_list->FindSectionByName(g_segment_name_TEXT));
    SectionSP data_section_sp(
        section_list->FindSectionByName(g_segment_name_DATA));
    SectionSP data_dirty_section_sp(
        section_list->FindSectionByName(g_segment_name_DATA_DIRTY));
    SectionSP data_const_section_sp(
        section_list->FindSectionByName(g_segment_name_DATA_CONST));
    SectionSP objc_section_sp(
        section_list->FindSectionByName(g_segment_name_OBJC));
    SectionSP eh_frame_section_sp;
    if (text_section_sp.get())
      eh_frame_section_sp = text_section_sp->GetChildren().FindSectionByName(
          g_section_name_eh_frame);
    else
      eh_frame_section_sp =
          section_list->FindSectionByName(g_section_name_eh_frame);

    const bool is_arm = (m_header.cputype == llvm::MachO::CPU_TYPE_ARM);

    // lldb works best if it knows the start address of all functions in a
    // module. Linker symbols or debug info are normally the best source of
    // information for start addr / size but they may be stripped in a released
    // binary. Two additional sources of information exist in Mach-O binaries:
    //    LC_FUNCTION_STARTS - a list of ULEB128 encoded offsets of each
    //    function's start address in the
    //                         binary, relative to the text section.
    //    eh_frame           - the eh_frame FDEs have the start addr & size of
    //    each function
    //  LC_FUNCTION_STARTS is the fastest source to read in, and is present on
    //  all modern binaries.
    //  Binaries built to run on older releases may need to use eh_frame
    //  information.

    if (text_section_sp && function_starts_data.GetByteSize()) {
      FunctionStarts::Entry function_start_entry;
      function_start_entry.data = false;
      lldb::offset_t function_start_offset = 0;
      function_start_entry.addr = text_section_sp->GetFileAddress();
      uint64_t delta;
      while ((delta = function_starts_data.GetULEB128(&function_start_offset)) >
             0) {
        // Now append the current entry
        function_start_entry.addr += delta;
        function_starts.Append(function_start_entry);
      }
    } else {
      // If m_type is eTypeDebugInfo, then this is a dSYM - it will have the
      // load command claiming an eh_frame but it doesn't actually have the
      // eh_frame content.  And if we have a dSYM, we don't need to do any of
      // this fill-in-the-missing-symbols works anyway - the debug info should
      // give us all the functions in the module.
      if (text_section_sp.get() && eh_frame_section_sp.get() &&
          m_type != eTypeDebugInfo) {
        DWARFCallFrameInfo eh_frame(*this, eh_frame_section_sp,
                                    DWARFCallFrameInfo::EH);
        DWARFCallFrameInfo::FunctionAddressAndSizeVector functions;
        eh_frame.GetFunctionAddressAndSizeVector(functions);
        addr_t text_base_addr = text_section_sp->GetFileAddress();
        size_t count = functions.GetSize();
        for (size_t i = 0; i < count; ++i) {
          const DWARFCallFrameInfo::FunctionAddressAndSizeVector::Entry *func =
              functions.GetEntryAtIndex(i);
          if (func) {
            FunctionStarts::Entry function_start_entry;
            function_start_entry.addr = func->base - text_base_addr;
            function_starts.Append(function_start_entry);
          }
        }
      }
    }

    const size_t function_starts_count = function_starts.GetSize();

    // For user process binaries (executables, dylibs, frameworks, bundles), if
    // we don't have LC_FUNCTION_STARTS/eh_frame section in this binary, we're
    // going to assume the binary has been stripped.  Don't allow assembly
    // language instruction emulation because we don't know proper function
    // start boundaries.
    //
    // For all other types of binaries (kernels, stand-alone bare board
    // binaries, kexts), they may not have LC_FUNCTION_STARTS / eh_frame
    // sections - we should not make any assumptions about them based on that.
    if (function_starts_count == 0 && CalculateStrata() == eStrataUser) {
      m_allow_assembly_emulation_unwind_plans = false;
      Log *unwind_or_symbol_log(lldb_private::GetLogIfAnyCategoriesSet(
          LIBLLDB_LOG_SYMBOLS | LIBLLDB_LOG_UNWIND));

      if (unwind_or_symbol_log)
        module_sp->LogMessage(
            unwind_or_symbol_log,
            "no LC_FUNCTION_STARTS, will not allow assembly profiled unwinds");
    }

    const user_id_t TEXT_eh_frame_sectID =
        eh_frame_section_sp.get() ? eh_frame_section_sp->GetID()
                                  : static_cast<user_id_t>(NO_SECT);

    lldb::offset_t nlist_data_offset = 0;

    uint32_t N_SO_index = UINT32_MAX;

    MachSymtabSectionInfo section_info(section_list);
    std::vector<uint32_t> N_FUN_indexes;
    std::vector<uint32_t> N_NSYM_indexes;
    std::vector<uint32_t> N_INCL_indexes;
    std::vector<uint32_t> N_BRAC_indexes;
    std::vector<uint32_t> N_COMM_indexes;
    typedef std::multimap<uint64_t, uint32_t> ValueToSymbolIndexMap;
    typedef std::map<uint32_t, uint32_t> NListIndexToSymbolIndexMap;
    typedef std::map<const char *, uint32_t> ConstNameToSymbolIndexMap;
    ValueToSymbolIndexMap N_FUN_addr_to_sym_idx;
    ValueToSymbolIndexMap N_STSYM_addr_to_sym_idx;
    ConstNameToSymbolIndexMap N_GSYM_name_to_sym_idx;
    // Any symbols that get merged into another will get an entry in this map
    // so we know
    NListIndexToSymbolIndexMap m_nlist_idx_to_sym_idx;
    uint32_t nlist_idx = 0;
    Symbol *symbol_ptr = NULL;

    uint32_t sym_idx = 0;
    Symbol *sym = NULL;
    size_t num_syms = 0;
    std::string memory_symbol_name;
    uint32_t unmapped_local_symbols_found = 0;

    std::vector<TrieEntryWithOffset> trie_entries;
    std::set<lldb::addr_t> resolver_addresses;

    if (dyld_trie_data.GetByteSize() > 0) {
      std::vector<llvm::StringRef> nameSlices;
      ParseTrieEntries(dyld_trie_data, 0, is_arm, nameSlices,
                       resolver_addresses, trie_entries);

      ConstString text_segment_name("__TEXT");
      SectionSP text_segment_sp =
          GetSectionList()->FindSectionByName(text_segment_name);
      if (text_segment_sp) {
        const lldb::addr_t text_segment_file_addr =
            text_segment_sp->GetFileAddress();
        if (text_segment_file_addr != LLDB_INVALID_ADDRESS) {
          for (auto &e : trie_entries)
            e.entry.address += text_segment_file_addr;
        }
      }
    }

    typedef std::set<ConstString> IndirectSymbols;
    IndirectSymbols indirect_symbol_names;

#if defined(__APPLE__) &&                                                      \
    (defined(__arm__) || defined(__arm64__) || defined(__aarch64__))

    // Some recent builds of the dyld_shared_cache (hereafter: DSC) have been
    // optimized by moving LOCAL symbols out of the memory mapped portion of
    // the DSC. The symbol information has all been retained, but it isn't
    // available in the normal nlist data. However, there *are* duplicate
    // entries of *some*
    // LOCAL symbols in the normal nlist data. To handle this situation
    // correctly, we must first attempt
    // to parse any DSC unmapped symbol information. If we find any, we set a
    // flag that tells the normal nlist parser to ignore all LOCAL symbols.

    if (m_header.flags & 0x80000000u) {
      // Before we can start mapping the DSC, we need to make certain the
      // target process is actually using the cache we can find.

      // Next we need to determine the correct path for the dyld shared cache.

      ArchSpec header_arch;
      GetArchitecture(header_arch);
      char dsc_path[PATH_MAX];
      char dsc_path_development[PATH_MAX];

      snprintf(
          dsc_path, sizeof(dsc_path), "%s%s%s",
          "/System/Library/Caches/com.apple.dyld/", /* IPHONE_DYLD_SHARED_CACHE_DIR
                                                       */
          "dyld_shared_cache_", /* DYLD_SHARED_CACHE_BASE_NAME */
          header_arch.GetArchitectureName());

      snprintf(
          dsc_path_development, sizeof(dsc_path), "%s%s%s%s",
          "/System/Library/Caches/com.apple.dyld/", /* IPHONE_DYLD_SHARED_CACHE_DIR
                                                       */
          "dyld_shared_cache_", /* DYLD_SHARED_CACHE_BASE_NAME */
          header_arch.GetArchitectureName(), ".development");

      FileSpec dsc_nondevelopment_filespec(dsc_path, false);
      FileSpec dsc_development_filespec(dsc_path_development, false);
      FileSpec dsc_filespec;

      UUID dsc_uuid;
      UUID process_shared_cache_uuid;
      addr_t process_shared_cache_base_addr;

      if (process) {
        GetProcessSharedCacheUUID(process, process_shared_cache_base_addr, process_shared_cache_uuid);
      }

      // First see if we can find an exact match for the inferior process
      // shared cache UUID in the development or non-development shared caches
      // on disk.
      if (process_shared_cache_uuid.IsValid()) {
        if (dsc_development_filespec.Exists()) {
          UUID dsc_development_uuid = GetSharedCacheUUID(
              dsc_development_filespec, byte_order, addr_byte_size);
          if (dsc_development_uuid.IsValid() &&
              dsc_development_uuid == process_shared_cache_uuid) {
            dsc_filespec = dsc_development_filespec;
            dsc_uuid = dsc_development_uuid;
          }
        }
        if (!dsc_uuid.IsValid() && dsc_nondevelopment_filespec.Exists()) {
          UUID dsc_nondevelopment_uuid = GetSharedCacheUUID(
              dsc_nondevelopment_filespec, byte_order, addr_byte_size);
          if (dsc_nondevelopment_uuid.IsValid() &&
              dsc_nondevelopment_uuid == process_shared_cache_uuid) {
            dsc_filespec = dsc_nondevelopment_filespec;
            dsc_uuid = dsc_nondevelopment_uuid;
          }
        }
      }

      // Failing a UUID match, prefer the development dyld_shared cache if both
      // are present.
      if (!dsc_filespec.Exists()) {
        if (dsc_development_filespec.Exists()) {
          dsc_filespec = dsc_development_filespec;
        } else {
          dsc_filespec = dsc_nondevelopment_filespec;
        }
      }

      /* The dyld_cache_header has a pointer to the
         dyld_cache_local_symbols_info structure (localSymbolsOffset).
         The dyld_cache_local_symbols_info structure gives us three things:
           1. The start and count of the nlist records in the dyld_shared_cache
         file
           2. The start and size of the strings for these nlist records
           3. The start and count of dyld_cache_local_symbols_entry entries

         There is one dyld_cache_local_symbols_entry per dylib/framework in the
         dyld shared cache.
         The "dylibOffset" field is the Mach-O header of this dylib/framework in
         the dyld shared cache.
         The dyld_cache_local_symbols_entry also lists the start of this
         dylib/framework's nlist records
         and the count of how many nlist records there are for this
         dylib/framework.
      */

      // Process the dyld shared cache header to find the unmapped symbols

      DataBufferSP dsc_data_sp = MapFileData(
          dsc_filespec, sizeof(struct lldb_copy_dyld_cache_header_v1), 0);
      if (!dsc_uuid.IsValid()) {
        dsc_uuid = GetSharedCacheUUID(dsc_filespec, byte_order, addr_byte_size);
      }
      if (dsc_data_sp) {
        DataExtractor dsc_header_data(dsc_data_sp, byte_order, addr_byte_size);

        bool uuid_match = true;
        if (dsc_uuid.IsValid() && process) {
          if (process_shared_cache_uuid.IsValid() &&
              dsc_uuid != process_shared_cache_uuid) {
            // The on-disk dyld_shared_cache file is not the same as the one in
            // this process' memory, don't use it.
            uuid_match = false;
            ModuleSP module_sp(GetModule());
            if (module_sp)
              module_sp->ReportWarning("process shared cache does not match "
                                       "on-disk dyld_shared_cache file, some "
                                       "symbol names will be missing.");
          }
        }

        offset = offsetof(struct lldb_copy_dyld_cache_header_v1, mappingOffset);

        uint32_t mappingOffset = dsc_header_data.GetU32(&offset);

        // If the mappingOffset points to a location inside the header, we've
        // opened an old dyld shared cache, and should not proceed further.
        if (uuid_match &&
            mappingOffset >= sizeof(struct lldb_copy_dyld_cache_header_v1)) {

          DataBufferSP dsc_mapping_info_data_sp = MapFileData(
              dsc_filespec, sizeof(struct lldb_copy_dyld_cache_mapping_info),
              mappingOffset);

          DataExtractor dsc_mapping_info_data(dsc_mapping_info_data_sp,
                                              byte_order, addr_byte_size);
          offset = 0;

          // The File addresses (from the in-memory Mach-O load commands) for
          // the shared libraries in the shared library cache need to be
          // adjusted by an offset to match up with the dylibOffset identifying
          // field in the dyld_cache_local_symbol_entry's.  This offset is
          // recorded in mapping_offset_value.
          const uint64_t mapping_offset_value =
              dsc_mapping_info_data.GetU64(&offset);

          offset = offsetof(struct lldb_copy_dyld_cache_header_v1,
                            localSymbolsOffset);
          uint64_t localSymbolsOffset = dsc_header_data.GetU64(&offset);
          uint64_t localSymbolsSize = dsc_header_data.GetU64(&offset);

          if (localSymbolsOffset && localSymbolsSize) {
            // Map the local symbols
            DataBufferSP dsc_local_symbols_data_sp =
                MapFileData(dsc_filespec, localSymbolsSize, localSymbolsOffset);

            if (dsc_local_symbols_data_sp) {
              DataExtractor dsc_local_symbols_data(dsc_local_symbols_data_sp,
                                                   byte_order, addr_byte_size);

              offset = 0;

              typedef std::map<ConstString, uint16_t> UndefinedNameToDescMap;
              typedef std::map<uint32_t, ConstString> SymbolIndexToName;
              UndefinedNameToDescMap undefined_name_to_desc;
              SymbolIndexToName reexport_shlib_needs_fixup;

              // Read the local_symbols_infos struct in one shot
              struct lldb_copy_dyld_cache_local_symbols_info local_symbols_info;
              dsc_local_symbols_data.GetU32(&offset,
                                            &local_symbols_info.nlistOffset, 6);

              SectionSP text_section_sp(
                  section_list->FindSectionByName(GetSegmentNameTEXT()));

              uint32_t header_file_offset =
                  (text_section_sp->GetFileAddress() - mapping_offset_value);

              offset = local_symbols_info.entriesOffset;
              for (uint32_t entry_index = 0;
                   entry_index < local_symbols_info.entriesCount;
                   entry_index++) {
                struct lldb_copy_dyld_cache_local_symbols_entry
                    local_symbols_entry;
                local_symbols_entry.dylibOffset =
                    dsc_local_symbols_data.GetU32(&offset);
                local_symbols_entry.nlistStartIndex =
                    dsc_local_symbols_data.GetU32(&offset);
                local_symbols_entry.nlistCount =
                    dsc_local_symbols_data.GetU32(&offset);

                if (header_file_offset == local_symbols_entry.dylibOffset) {
                  unmapped_local_symbols_found = local_symbols_entry.nlistCount;

                  // The normal nlist code cannot correctly size the Symbols
                  // array, we need to allocate it here.
                  sym = symtab->Resize(
                      symtab_load_command.nsyms + m_dysymtab.nindirectsyms +
                      unmapped_local_symbols_found - m_dysymtab.nlocalsym);
                  num_syms = symtab->GetNumSymbols();

                  nlist_data_offset =
                      local_symbols_info.nlistOffset +
                      (nlist_byte_size * local_symbols_entry.nlistStartIndex);
                  uint32_t string_table_offset =
                      local_symbols_info.stringsOffset;

                  for (uint32_t nlist_index = 0;
                       nlist_index < local_symbols_entry.nlistCount;
                       nlist_index++) {
                    /////////////////////////////
                    {
                      struct nlist_64 nlist;
                      if (!dsc_local_symbols_data.ValidOffsetForDataOfSize(
                              nlist_data_offset, nlist_byte_size))
                        break;

                      nlist.n_strx = dsc_local_symbols_data.GetU32_unchecked(
                          &nlist_data_offset);
                      nlist.n_type = dsc_local_symbols_data.GetU8_unchecked(
                          &nlist_data_offset);
                      nlist.n_sect = dsc_local_symbols_data.GetU8_unchecked(
                          &nlist_data_offset);
                      nlist.n_desc = dsc_local_symbols_data.GetU16_unchecked(
                          &nlist_data_offset);
                      nlist.n_value =
                          dsc_local_symbols_data.GetAddress_unchecked(
                              &nlist_data_offset);

                      SymbolType type = eSymbolTypeInvalid;
                      const char *symbol_name = dsc_local_symbols_data.PeekCStr(
                          string_table_offset + nlist.n_strx);

                      if (symbol_name == NULL) {
                        // No symbol should be NULL, even the symbols with no
                        // string values should have an offset zero which
                        // points to an empty C-string
                        Host::SystemLog(
                            Host::eSystemLogError,
                            "error: DSC unmapped local symbol[%u] has invalid "
                            "string table offset 0x%x in %s, ignoring symbol\n",
                            entry_index, nlist.n_strx,
                            module_sp->GetFileSpec().GetPath().c_str());
                        continue;
                      }
                      if (symbol_name[0] == '\0')
                        symbol_name = NULL;

                      const char *symbol_name_non_abi_mangled = NULL;

                      SectionSP symbol_section;
                      uint32_t symbol_byte_size = 0;
                      bool add_nlist = true;
                      bool is_debug = ((nlist.n_type & N_STAB) != 0);
                      bool demangled_is_synthesized = false;
                      bool is_gsym = false;
                      bool set_value = true;

                      assert(sym_idx < num_syms);

                      sym[sym_idx].SetDebug(is_debug);

                      if (is_debug) {
                        switch (nlist.n_type) {
                        case N_GSYM:
                          // global symbol: name,,NO_SECT,type,0
                          // Sometimes the N_GSYM value contains the address.

                          // FIXME: In the .o files, we have a GSYM and a debug
                          // symbol for all the ObjC data.  They
                          // have the same address, but we want to ensure that
                          // we always find only the real symbol, 'cause we
                          // don't currently correctly attribute the
                          // GSYM one to the ObjCClass/Ivar/MetaClass
                          // symbol type.  This is a temporary hack to make
                          // sure the ObjectiveC symbols get treated correctly.
                          // To do this right, we should coalesce all the GSYM
                          // & global symbols that have the same address.

                          is_gsym = true;
                          sym[sym_idx].SetExternal(true);

                          if (symbol_name && symbol_name[0] == '_' &&
                              symbol_name[1] == 'O') {
                            llvm::StringRef symbol_name_ref(symbol_name);
                            if (symbol_name_ref.startswith(
                                    g_objc_v2_prefix_class)) {
                              symbol_name_non_abi_mangled = symbol_name + 1;
                              symbol_name =
                                  symbol_name + g_objc_v2_prefix_class.size();
                              type = eSymbolTypeObjCClass;
                              demangled_is_synthesized = true;

                            } else if (symbol_name_ref.startswith(
                                           g_objc_v2_prefix_metaclass)) {
                              symbol_name_non_abi_mangled = symbol_name + 1;
                              symbol_name = symbol_name +
                                            g_objc_v2_prefix_metaclass.size();
                              type = eSymbolTypeObjCMetaClass;
                              demangled_is_synthesized = true;
                            } else if (symbol_name_ref.startswith(
                                           g_objc_v2_prefix_ivar)) {
                              symbol_name_non_abi_mangled = symbol_name + 1;
                              symbol_name =
                                  symbol_name + g_objc_v2_prefix_ivar.size();
                              type = eSymbolTypeObjCIVar;
                              demangled_is_synthesized = true;
                            }
                          } else {
                            if (nlist.n_value != 0)
                              symbol_section = section_info.GetSection(
                                  nlist.n_sect, nlist.n_value);
                            type = eSymbolTypeData;
                          }
                          break;

                        case N_FNAME:
                          // procedure name (f77 kludge): name,,NO_SECT,0,0
                          type = eSymbolTypeCompiler;
                          break;

                        case N_FUN:
                          // procedure: name,,n_sect,linenumber,address
                          if (symbol_name) {
                            type = eSymbolTypeCode;
                            symbol_section = section_info.GetSection(
                                nlist.n_sect, nlist.n_value);

                            N_FUN_addr_to_sym_idx.insert(
                                std::make_pair(nlist.n_value, sym_idx));
                            // We use the current number of symbols in the
                            // symbol table in lieu of using nlist_idx in case
                            // we ever start trimming entries out
                            N_FUN_indexes.push_back(sym_idx);
                          } else {
                            type = eSymbolTypeCompiler;

                            if (!N_FUN_indexes.empty()) {
                              // Copy the size of the function into the
                              // original
                              // STAB entry so we don't have
                              // to hunt for it later
                              symtab->SymbolAtIndex(N_FUN_indexes.back())
                                  ->SetByteSize(nlist.n_value);
                              N_FUN_indexes.pop_back();
                              // We don't really need the end function STAB as
                              // it contains the size which we already placed
                              // with the original symbol, so don't add it if
                              // we want a minimal symbol table
                              add_nlist = false;
                            }
                          }
                          break;

                        case N_STSYM:
                          // static symbol: name,,n_sect,type,address
                          N_STSYM_addr_to_sym_idx.insert(
                              std::make_pair(nlist.n_value, sym_idx));
                          symbol_section = section_info.GetSection(
                              nlist.n_sect, nlist.n_value);
                          if (symbol_name && symbol_name[0]) {
                            type = ObjectFile::GetSymbolTypeFromName(
                                symbol_name + 1, eSymbolTypeData);
                          }
                          break;

                        case N_LCSYM:
                          // .lcomm symbol: name,,n_sect,type,address
                          symbol_section = section_info.GetSection(
                              nlist.n_sect, nlist.n_value);
                          type = eSymbolTypeCommonBlock;
                          break;

                        case N_BNSYM:
                          // We use the current number of symbols in the symbol
                          // table in lieu of using nlist_idx in case we ever
                          // start trimming entries out Skip these if we want
                          // minimal symbol tables
                          add_nlist = false;
                          break;

                        case N_ENSYM:
                          // Set the size of the N_BNSYM to the terminating
                          // index of this N_ENSYM so that we can always skip
                          // the entire symbol if we need to navigate more
                          // quickly at the source level when parsing STABS
                          // Skip these if we want minimal symbol tables
                          add_nlist = false;
                          break;

                        case N_OPT:
                          // emitted with gcc2_compiled and in gcc source
                          type = eSymbolTypeCompiler;
                          break;

                        case N_RSYM:
                          // register sym: name,,NO_SECT,type,register
                          type = eSymbolTypeVariable;
                          break;

                        case N_SLINE:
                          // src line: 0,,n_sect,linenumber,address
                          symbol_section = section_info.GetSection(
                              nlist.n_sect, nlist.n_value);
                          type = eSymbolTypeLineEntry;
                          break;

                        case N_SSYM:
                          // structure elt: name,,NO_SECT,type,struct_offset
                          type = eSymbolTypeVariableType;
                          break;

                        case N_SO:
                          // source file name
                          type = eSymbolTypeSourceFile;
                          if (symbol_name == NULL) {
                            add_nlist = false;
                            if (N_SO_index != UINT32_MAX) {
                              // Set the size of the N_SO to the terminating
                              // index of this N_SO so that we can always skip
                              // the entire N_SO if we need to navigate more
                              // quickly at the source level when parsing STABS
                              symbol_ptr = symtab->SymbolAtIndex(N_SO_index);
                              symbol_ptr->SetByteSize(sym_idx);
                              symbol_ptr->SetSizeIsSibling(true);
                            }
                            N_NSYM_indexes.clear();
                            N_INCL_indexes.clear();
                            N_BRAC_indexes.clear();
                            N_COMM_indexes.clear();
                            N_FUN_indexes.clear();
                            N_SO_index = UINT32_MAX;
                          } else {
                            // We use the current number of symbols in the
                            // symbol table in lieu of using nlist_idx in case
                            // we ever start trimming entries out
                            const bool N_SO_has_full_path =
                                symbol_name[0] == '/';
                            if (N_SO_has_full_path) {
                              if ((N_SO_index == sym_idx - 1) &&
                                  ((sym_idx - 1) < num_syms)) {
                                // We have two consecutive N_SO entries where
                                // the first contains a directory and the
                                // second contains a full path.
                                sym[sym_idx - 1].GetMangled().SetValue(
                                    ConstString(symbol_name), false);
                                m_nlist_idx_to_sym_idx[nlist_idx] = sym_idx - 1;
                                add_nlist = false;
                              } else {
                                // This is the first entry in a N_SO that
                                // contains a directory or
                                // a full path to the source file
                                N_SO_index = sym_idx;
                              }
                            } else if ((N_SO_index == sym_idx - 1) &&
                                       ((sym_idx - 1) < num_syms)) {
                              // This is usually the second N_SO entry that
                              // contains just the filename, so here we combine
                              // it with the first one if we are minimizing the
                              // symbol table
                              const char *so_path =
                                  sym[sym_idx - 1]
                                      .GetMangled()
                                      .GetDemangledName(
                                          lldb::eLanguageTypeUnknown)
                                      .AsCString();
                              if (so_path && so_path[0]) {
                                std::string full_so_path(so_path);
                                const size_t double_slash_pos =
                                    full_so_path.find("//");
                                if (double_slash_pos != std::string::npos) {
                                  // The linker has been generating bad N_SO
                                  // entries with doubled up paths
                                  // in the format "%s%s" where the first
                                  // string in the DW_AT_comp_dir, and the
                                  // second is the directory for the source
                                  // file so you end up with a path that looks
                                  // like "/tmp/src//tmp/src/"
                                  FileSpec so_dir(so_path, false);
                                  if (!so_dir.Exists()) {
                                    so_dir.SetFile(
                                        &full_so_path[double_slash_pos + 1],
                                        false);
                                    if (so_dir.Exists()) {
                                      // Trim off the incorrect path
                                      full_so_path.erase(0,
                                                         double_slash_pos + 1);
                                    }
                                  }
                                }
                                if (*full_so_path.rbegin() != '/')
                                  full_so_path += '/';
                                full_so_path += symbol_name;
                                sym[sym_idx - 1].GetMangled().SetValue(
                                    ConstString(full_so_path.c_str()), false);
                                add_nlist = false;
                                m_nlist_idx_to_sym_idx[nlist_idx] = sym_idx - 1;
                              }
                            } else {
                              // This could be a relative path to a N_SO
                              N_SO_index = sym_idx;
                            }
                          }
                          break;

                        case N_OSO:
                          // object file name: name,,0,0,st_mtime
                          type = eSymbolTypeObjectFile;
                          break;

                        case N_LSYM:
                          // local sym: name,,NO_SECT,type,offset
                          type = eSymbolTypeLocal;
                          break;

                        //----------------------------------------------------------------------
                        // INCL scopes
                        //----------------------------------------------------------------------
                        case N_BINCL:
                          // include file beginning: name,,NO_SECT,0,sum We use
                          // the current number of symbols in the symbol table
                          // in lieu of using nlist_idx in case we ever start
                          // trimming entries out
                          N_INCL_indexes.push_back(sym_idx);
                          type = eSymbolTypeScopeBegin;
                          break;

                        case N_EINCL:
                          // include file end: name,,NO_SECT,0,0
                          // Set the size of the N_BINCL to the terminating
                          // index of this N_EINCL so that we can always skip
                          // the entire symbol if we need to navigate more
                          // quickly at the source level when parsing STABS
                          if (!N_INCL_indexes.empty()) {
                            symbol_ptr =
                                symtab->SymbolAtIndex(N_INCL_indexes.back());
                            symbol_ptr->SetByteSize(sym_idx + 1);
                            symbol_ptr->SetSizeIsSibling(true);
                            N_INCL_indexes.pop_back();
                          }
                          type = eSymbolTypeScopeEnd;
                          break;

                        case N_SOL:
                          // #included file name: name,,n_sect,0,address
                          type = eSymbolTypeHeaderFile;

                          // We currently don't use the header files on darwin
                          add_nlist = false;
                          break;

                        case N_PARAMS:
                          // compiler parameters: name,,NO_SECT,0,0
                          type = eSymbolTypeCompiler;
                          break;

                        case N_VERSION:
                          // compiler version: name,,NO_SECT,0,0
                          type = eSymbolTypeCompiler;
                          break;

                        case N_OLEVEL:
                          // compiler -O level: name,,NO_SECT,0,0
                          type = eSymbolTypeCompiler;
                          break;

                        case N_PSYM:
                          // parameter: name,,NO_SECT,type,offset
                          type = eSymbolTypeVariable;
                          break;

                        case N_ENTRY:
                          // alternate entry: name,,n_sect,linenumber,address
                          symbol_section = section_info.GetSection(
                              nlist.n_sect, nlist.n_value);
                          type = eSymbolTypeLineEntry;
                          break;

                        //----------------------------------------------------------------------
                        // Left and Right Braces
                        //----------------------------------------------------------------------
                        case N_LBRAC:
                          // left bracket: 0,,NO_SECT,nesting level,address We
                          // use the current number of symbols in the symbol
                          // table in lieu of using nlist_idx in case we ever
                          // start trimming entries out
                          symbol_section = section_info.GetSection(
                              nlist.n_sect, nlist.n_value);
                          N_BRAC_indexes.push_back(sym_idx);
                          type = eSymbolTypeScopeBegin;
                          break;

                        case N_RBRAC:
                          // right bracket: 0,,NO_SECT,nesting level,address
                          // Set the size of the N_LBRAC to the terminating
                          // index of this N_RBRAC so that we can always skip
                          // the entire symbol if we need to navigate more
                          // quickly at the source level when parsing STABS
                          symbol_section = section_info.GetSection(
                              nlist.n_sect, nlist.n_value);
                          if (!N_BRAC_indexes.empty()) {
                            symbol_ptr =
                                symtab->SymbolAtIndex(N_BRAC_indexes.back());
                            symbol_ptr->SetByteSize(sym_idx + 1);
                            symbol_ptr->SetSizeIsSibling(true);
                            N_BRAC_indexes.pop_back();
                          }
                          type = eSymbolTypeScopeEnd;
                          break;

                        case N_EXCL:
                          // deleted include file: name,,NO_SECT,0,sum
                          type = eSymbolTypeHeaderFile;
                          break;

                        //----------------------------------------------------------------------
                        // COMM scopes
                        //----------------------------------------------------------------------
                        case N_BCOMM:
                          // begin common: name,,NO_SECT,0,0
                          // We use the current number of symbols in the symbol
                          // table in lieu of using nlist_idx in case we ever
                          // start trimming entries out
                          type = eSymbolTypeScopeBegin;
                          N_COMM_indexes.push_back(sym_idx);
                          break;

                        case N_ECOML:
                          // end common (local name): 0,,n_sect,0,address
                          symbol_section = section_info.GetSection(
                              nlist.n_sect, nlist.n_value);
                        // Fall through

                        case N_ECOMM:
                          // end common: name,,n_sect,0,0
                          // Set the size of the N_BCOMM to the terminating
                          // index of this N_ECOMM/N_ECOML so that we can
                          // always skip the entire symbol if we need to
                          // navigate more quickly at the source level when
                          // parsing STABS
                          if (!N_COMM_indexes.empty()) {
                            symbol_ptr =
                                symtab->SymbolAtIndex(N_COMM_indexes.back());
                            symbol_ptr->SetByteSize(sym_idx + 1);
                            symbol_ptr->SetSizeIsSibling(true);
                            N_COMM_indexes.pop_back();
                          }
                          type = eSymbolTypeScopeEnd;
                          break;

                        case N_LENG:
                          // second stab entry with length information
                          type = eSymbolTypeAdditional;
                          break;

                        default:
                          break;
                        }
                      } else {
                        // uint8_t n_pext    = N_PEXT & nlist.n_type;
                        uint8_t n_type = N_TYPE & nlist.n_type;
                        sym[sym_idx].SetExternal((N_EXT & nlist.n_type) != 0);

                        switch (n_type) {
                        case N_INDR: {
                          const char *reexport_name_cstr =
                              strtab_data.PeekCStr(nlist.n_value);
                          if (reexport_name_cstr && reexport_name_cstr[0]) {
                            type = eSymbolTypeReExported;
                            ConstString reexport_name(
                                reexport_name_cstr +
                                ((reexport_name_cstr[0] == '_') ? 1 : 0));
                            sym[sym_idx].SetReExportedSymbolName(reexport_name);
                            set_value = false;
                            reexport_shlib_needs_fixup[sym_idx] = reexport_name;
                            indirect_symbol_names.insert(
                                ConstString(symbol_name +
                                            ((symbol_name[0] == '_') ? 1 : 0)));
                          } else
                            type = eSymbolTypeUndefined;
                        } break;

                        case N_UNDF:
                          if (symbol_name && symbol_name[0]) {
                            ConstString undefined_name(
                                symbol_name +
                                ((symbol_name[0] == '_') ? 1 : 0));
                            undefined_name_to_desc[undefined_name] =
                                nlist.n_desc;
                          }
                        // Fall through
                        case N_PBUD:
                          type = eSymbolTypeUndefined;
                          break;

                        case N_ABS:
                          type = eSymbolTypeAbsolute;
                          break;

                        case N_SECT: {
                          symbol_section = section_info.GetSection(
                              nlist.n_sect, nlist.n_value);

                          if (symbol_section == NULL) {
                            // TODO: warn about this?
                            add_nlist = false;
                            break;
                          }

                          if (TEXT_eh_frame_sectID == nlist.n_sect) {
                            type = eSymbolTypeException;
                          } else {
                            uint32_t section_type =
                                symbol_section->Get() & SECTION_TYPE;

                            switch (section_type) {
                            case S_CSTRING_LITERALS:
                              type = eSymbolTypeData;
                              break; // section with only literal C strings
                            case S_4BYTE_LITERALS:
                              type = eSymbolTypeData;
                              break; // section with only 4 byte literals
                            case S_8BYTE_LITERALS:
                              type = eSymbolTypeData;
                              break; // section with only 8 byte literals
                            case S_LITERAL_POINTERS:
                              type = eSymbolTypeTrampoline;
                              break; // section with only pointers to literals
                            case S_NON_LAZY_SYMBOL_POINTERS:
                              type = eSymbolTypeTrampoline;
                              break; // section with only non-lazy symbol
                                     // pointers
                            case S_LAZY_SYMBOL_POINTERS:
                              type = eSymbolTypeTrampoline;
                              break; // section with only lazy symbol pointers
                            case S_SYMBOL_STUBS:
                              type = eSymbolTypeTrampoline;
                              break; // section with only symbol stubs, byte
                                     // size of stub in the reserved2 field
                            case S_MOD_INIT_FUNC_POINTERS:
                              type = eSymbolTypeCode;
                              break; // section with only function pointers for
                                     // initialization
                            case S_MOD_TERM_FUNC_POINTERS:
                              type = eSymbolTypeCode;
                              break; // section with only function pointers for
                                     // termination
                            case S_INTERPOSING:
                              type = eSymbolTypeTrampoline;
                              break; // section with only pairs of function
                                     // pointers for interposing
                            case S_16BYTE_LITERALS:
                              type = eSymbolTypeData;
                              break; // section with only 16 byte literals
                            case S_DTRACE_DOF:
                              type = eSymbolTypeInstrumentation;
                              break;
                            case S_LAZY_DYLIB_SYMBOL_POINTERS:
                              type = eSymbolTypeTrampoline;
                              break;
                            default:
                              switch (symbol_section->GetType()) {
                              case lldb::eSectionTypeCode:
                                type = eSymbolTypeCode;
                                break;
                              case eSectionTypeData:
                              case eSectionTypeDataCString: // Inlined C string
                                                            // data
                              case eSectionTypeDataCStringPointers: // Pointers
                                                                    // to C
                                                                    // string
                                                                    // data
                              case eSectionTypeDataSymbolAddress: // Address of
                                                                  // a symbol in
                                                                  // the symbol
                                                                  // table
                              case eSectionTypeData4:
                              case eSectionTypeData8:
                              case eSectionTypeData16:
                                type = eSymbolTypeData;
                                break;
                              default:
                                break;
                              }
                              break;
                            }

                            if (type == eSymbolTypeInvalid) {
                              const char *symbol_sect_name =
                                  symbol_section->GetName().AsCString();
                              if (symbol_section->IsDescendant(
                                      text_section_sp.get())) {
                                if (symbol_section->IsClear(
                                        S_ATTR_PURE_INSTRUCTIONS |
                                        S_ATTR_SELF_MODIFYING_CODE |
                                        S_ATTR_SOME_INSTRUCTIONS))
                                  type = eSymbolTypeData;
                                else
                                  type = eSymbolTypeCode;
                              } else if (symbol_section->IsDescendant(
                                             data_section_sp.get()) ||
                                         symbol_section->IsDescendant(
                                             data_dirty_section_sp.get()) ||
                                         symbol_section->IsDescendant(
                                             data_const_section_sp.get())) {
                                if (symbol_sect_name &&
                                    ::strstr(symbol_sect_name, "__objc") ==
                                        symbol_sect_name) {
                                  type = eSymbolTypeRuntime;

                                  if (symbol_name) {
                                    llvm::StringRef symbol_name_ref(
                                        symbol_name);
                                    if (symbol_name_ref.startswith("_OBJC_")) {
                                      static const llvm::StringRef
                                          g_objc_v2_prefix_class(
                                              "_OBJC_CLASS_$_");
                                      static const llvm::StringRef
                                          g_objc_v2_prefix_metaclass(
                                              "_OBJC_METACLASS_$_");
                                      static const llvm::StringRef
                                          g_objc_v2_prefix_ivar(
                                              "_OBJC_IVAR_$_");
                                      if (symbol_name_ref.startswith(
                                              g_objc_v2_prefix_class)) {
                                        symbol_name_non_abi_mangled =
                                            symbol_name + 1;
                                        symbol_name =
                                            symbol_name +
                                            g_objc_v2_prefix_class.size();
                                        type = eSymbolTypeObjCClass;
                                        demangled_is_synthesized = true;
                                      } else if (
                                          symbol_name_ref.startswith(
                                              g_objc_v2_prefix_metaclass)) {
                                        symbol_name_non_abi_mangled =
                                            symbol_name + 1;
                                        symbol_name =
                                            symbol_name +
                                            g_objc_v2_prefix_metaclass.size();
                                        type = eSymbolTypeObjCMetaClass;
                                        demangled_is_synthesized = true;
                                      } else if (symbol_name_ref.startswith(
                                                     g_objc_v2_prefix_ivar)) {
                                        symbol_name_non_abi_mangled =
                                            symbol_name + 1;
                                        symbol_name =
                                            symbol_name +
                                            g_objc_v2_prefix_ivar.size();
                                        type = eSymbolTypeObjCIVar;
                                        demangled_is_synthesized = true;
                                      }
                                    }
                                  }
                                } else if (symbol_sect_name &&
                                           ::strstr(symbol_sect_name,
                                                    "__gcc_except_tab") ==
                                               symbol_sect_name) {
                                  type = eSymbolTypeException;
                                } else {
                                  type = eSymbolTypeData;
                                }
                              } else if (symbol_sect_name &&
                                         ::strstr(symbol_sect_name,
                                                  "__IMPORT") ==
                                             symbol_sect_name) {
                                type = eSymbolTypeTrampoline;
                              } else if (symbol_section->IsDescendant(
                                             objc_section_sp.get())) {
                                type = eSymbolTypeRuntime;
                                if (symbol_name && symbol_name[0] == '.') {
                                  llvm::StringRef symbol_name_ref(symbol_name);
                                  static const llvm::StringRef
                                      g_objc_v1_prefix_class(
                                          ".objc_class_name_");
                                  if (symbol_name_ref.startswith(
                                          g_objc_v1_prefix_class)) {
                                    symbol_name_non_abi_mangled = symbol_name;
                                    symbol_name = symbol_name +
                                                  g_objc_v1_prefix_class.size();
                                    type = eSymbolTypeObjCClass;
                                    demangled_is_synthesized = true;
                                  }
                                }
                              }
                            }
                          }
                        } break;
                        }
                      }

                      if (add_nlist) {
                        uint64_t symbol_value = nlist.n_value;
                        if (symbol_name_non_abi_mangled) {
                          sym[sym_idx].GetMangled().SetMangledName(
                              ConstString(symbol_name_non_abi_mangled));
                          sym[sym_idx].GetMangled().SetDemangledName(
                              ConstString(symbol_name));
                        } else {
                          bool symbol_name_is_mangled = false;

                          if (symbol_name && symbol_name[0] == '_') {
                            symbol_name_is_mangled = symbol_name[1] == '_';
                            symbol_name++; // Skip the leading underscore
                          }

                          if (symbol_name) {
                            ConstString const_symbol_name(symbol_name);
                            sym[sym_idx].GetMangled().SetValue(
                                const_symbol_name, symbol_name_is_mangled);
                            if (is_gsym && is_debug) {
                              const char *gsym_name =
                                  sym[sym_idx]
                                      .GetMangled()
                                      .GetName(lldb::eLanguageTypeUnknown,
                                               Mangled::ePreferMangled)
                                      .GetCString();
                              if (gsym_name)
                                N_GSYM_name_to_sym_idx[gsym_name] = sym_idx;
                            }
                          }
                        }
                        if (symbol_section) {
                          const addr_t section_file_addr =
                              symbol_section->GetFileAddress();
                          if (symbol_byte_size == 0 &&
                              function_starts_count > 0) {
                            addr_t symbol_lookup_file_addr = nlist.n_value;
                            // Do an exact address match for non-ARM addresses,
                            // else get the closest since the symbol might be a
                            // thumb symbol which has an address with bit zero
                            // set
                            FunctionStarts::Entry *func_start_entry =
                                function_starts.FindEntry(
                                    symbol_lookup_file_addr, !is_arm);
                            if (is_arm && func_start_entry) {
                              // Verify that the function start address is the
                              // symbol address (ARM) or the symbol address + 1
                              // (thumb)
                              if (func_start_entry->addr !=
                                      symbol_lookup_file_addr &&
                                  func_start_entry->addr !=
                                      (symbol_lookup_file_addr + 1)) {
                                // Not the right entry, NULL it out...
                                func_start_entry = NULL;
                              }
                            }
                            if (func_start_entry) {
                              func_start_entry->data = true;

                              addr_t symbol_file_addr = func_start_entry->addr;
                              uint32_t symbol_flags = 0;
                              if (is_arm) {
                                if (symbol_file_addr & 1)
                                  symbol_flags =
                                      MACHO_NLIST_ARM_SYMBOL_IS_THUMB;
                                symbol_file_addr &= THUMB_ADDRESS_BIT_MASK;
                              }

                              const FunctionStarts::Entry
                                  *next_func_start_entry =
                                      function_starts.FindNextEntry(
                                          func_start_entry);
                              const addr_t section_end_file_addr =
                                  section_file_addr +
                                  symbol_section->GetByteSize();
                              if (next_func_start_entry) {
                                addr_t next_symbol_file_addr =
                                    next_func_start_entry->addr;
                                // Be sure the clear the Thumb address bit when
                                // we calculate the size from the current and
                                // next address
                                if (is_arm)
                                  next_symbol_file_addr &=
                                      THUMB_ADDRESS_BIT_MASK;
                                symbol_byte_size = std::min<lldb::addr_t>(
                                    next_symbol_file_addr - symbol_file_addr,
                                    section_end_file_addr - symbol_file_addr);
                              } else {
                                symbol_byte_size =
                                    section_end_file_addr - symbol_file_addr;
                              }
                            }
                          }
                          symbol_value -= section_file_addr;
                        }

                        if (is_debug == false) {
                          if (type == eSymbolTypeCode) {
                            // See if we can find a N_FUN entry for any code
                            // symbols. If we do find a match, and the name
                            // matches, then we can merge the two into just the
                            // function symbol to avoid duplicate entries in
                            // the symbol table
                            std::pair<ValueToSymbolIndexMap::const_iterator,
                                      ValueToSymbolIndexMap::const_iterator>
                                range;
                            range = N_FUN_addr_to_sym_idx.equal_range(
                                nlist.n_value);
                            if (range.first != range.second) {
                              bool found_it = false;
                              for (ValueToSymbolIndexMap::const_iterator pos =
                                       range.first;
                                   pos != range.second; ++pos) {
                                if (sym[sym_idx].GetMangled().GetName(
                                        lldb::eLanguageTypeUnknown,
                                        Mangled::ePreferMangled) ==
                                    sym[pos->second].GetMangled().GetName(
                                        lldb::eLanguageTypeUnknown,
                                        Mangled::ePreferMangled)) {
                                  m_nlist_idx_to_sym_idx[nlist_idx] =
                                      pos->second;
                                  // We just need the flags from the linker
                                  // symbol, so put these flags
                                  // into the N_FUN flags to avoid duplicate
                                  // symbols in the symbol table
                                  sym[pos->second].SetExternal(
                                      sym[sym_idx].IsExternal());
                                  sym[pos->second].SetFlags(nlist.n_type << 16 |
                                                            nlist.n_desc);
                                  if (resolver_addresses.find(nlist.n_value) !=
                                      resolver_addresses.end())
                                    sym[pos->second].SetType(
                                        eSymbolTypeResolver);
                                  sym[sym_idx].Clear();
                                  found_it = true;
                                  break;
                                }
                              }
                              if (found_it)
                                continue;
                            } else {
                              if (resolver_addresses.find(nlist.n_value) !=
                                  resolver_addresses.end())
                                type = eSymbolTypeResolver;
                            }
                          } else if (type == eSymbolTypeData ||
                                     type == eSymbolTypeObjCClass ||
                                     type == eSymbolTypeObjCMetaClass ||
                                     type == eSymbolTypeObjCIVar) {
                            // See if we can find a N_STSYM entry for any data
                            // symbols. If we do find a match, and the name
                            // matches, then we can merge the two into just the
                            // Static symbol to avoid duplicate entries in the
                            // symbol table
                            std::pair<ValueToSymbolIndexMap::const_iterator,
                                      ValueToSymbolIndexMap::const_iterator>
                                range;
                            range = N_STSYM_addr_to_sym_idx.equal_range(
                                nlist.n_value);
                            if (range.first != range.second) {
                              bool found_it = false;
                              for (ValueToSymbolIndexMap::const_iterator pos =
                                       range.first;
                                   pos != range.second; ++pos) {
                                if (sym[sym_idx].GetMangled().GetName(
                                        lldb::eLanguageTypeUnknown,
                                        Mangled::ePreferMangled) ==
                                    sym[pos->second].GetMangled().GetName(
                                        lldb::eLanguageTypeUnknown,
                                        Mangled::ePreferMangled)) {
                                  m_nlist_idx_to_sym_idx[nlist_idx] =
                                      pos->second;
                                  // We just need the flags from the linker
                                  // symbol, so put these flags
                                  // into the N_STSYM flags to avoid duplicate
                                  // symbols in the symbol table
                                  sym[pos->second].SetExternal(
                                      sym[sym_idx].IsExternal());
                                  sym[pos->second].SetFlags(nlist.n_type << 16 |
                                                            nlist.n_desc);
                                  sym[sym_idx].Clear();
                                  found_it = true;
                                  break;
                                }
                              }
                              if (found_it)
                                continue;
                            } else {
                              const char *gsym_name =
                                  sym[sym_idx]
                                      .GetMangled()
                                      .GetName(lldb::eLanguageTypeUnknown,
                                               Mangled::ePreferMangled)
                                      .GetCString();
                              if (gsym_name) {
                                // Combine N_GSYM stab entries with the non
                                // stab symbol
                                ConstNameToSymbolIndexMap::const_iterator pos =
                                    N_GSYM_name_to_sym_idx.find(gsym_name);
                                if (pos != N_GSYM_name_to_sym_idx.end()) {
                                  const uint32_t GSYM_sym_idx = pos->second;
                                  m_nlist_idx_to_sym_idx[nlist_idx] =
                                      GSYM_sym_idx;
                                  // Copy the address, because often the N_GSYM
                                  // address has an invalid address of zero
                                  // when the global is a common symbol
                                  sym[GSYM_sym_idx].GetAddressRef().SetSection(
                                      symbol_section);
                                  sym[GSYM_sym_idx].GetAddressRef().SetOffset(
                                      symbol_value);
                                  // We just need the flags from the linker
                                  // symbol, so put these flags
                                  // into the N_GSYM flags to avoid duplicate
                                  // symbols in the symbol table
                                  sym[GSYM_sym_idx].SetFlags(
                                      nlist.n_type << 16 | nlist.n_desc);
                                  sym[sym_idx].Clear();
                                  continue;
                                }
                              }
                            }
                          }
                        }

                        sym[sym_idx].SetID(nlist_idx);
                        sym[sym_idx].SetType(type);
                        if (set_value) {
                          sym[sym_idx].GetAddressRef().SetSection(
                              symbol_section);
                          sym[sym_idx].GetAddressRef().SetOffset(symbol_value);
                        }
                        sym[sym_idx].SetFlags(nlist.n_type << 16 |
                                              nlist.n_desc);

                        if (symbol_byte_size > 0)
                          sym[sym_idx].SetByteSize(symbol_byte_size);

                        if (demangled_is_synthesized)
                          sym[sym_idx].SetDemangledNameIsSynthesized(true);
                        ++sym_idx;
                      } else {
                        sym[sym_idx].Clear();
                      }
                    }
                    /////////////////////////////
                  }
                  break; // No more entries to consider
                }
              }

              for (const auto &pos : reexport_shlib_needs_fixup) {
                const auto undef_pos = undefined_name_to_desc.find(pos.second);
                if (undef_pos != undefined_name_to_desc.end()) {
                  const uint8_t dylib_ordinal =
                      llvm::MachO::GET_LIBRARY_ORDINAL(undef_pos->second);
                  if (dylib_ordinal > 0 &&
                      dylib_ordinal < dylib_files.GetSize())
                    sym[pos.first].SetReExportedSymbolSharedLibrary(
                        dylib_files.GetFileSpecAtIndex(dylib_ordinal - 1));
                }
              }
            }
          }
        }
      }
    }

    // Must reset this in case it was mutated above!
    nlist_data_offset = 0;
#endif

    if (nlist_data.GetByteSize() > 0) {

      // If the sym array was not created while parsing the DSC unmapped
      // symbols, create it now.
      if (sym == NULL) {
        sym = symtab->Resize(symtab_load_command.nsyms +
                             m_dysymtab.nindirectsyms);
        num_syms = symtab->GetNumSymbols();
      }

      if (unmapped_local_symbols_found) {
        assert(m_dysymtab.ilocalsym == 0);
        nlist_data_offset += (m_dysymtab.nlocalsym * nlist_byte_size);
        nlist_idx = m_dysymtab.nlocalsym;
      } else {
        nlist_idx = 0;
      }

      typedef std::map<ConstString, uint16_t> UndefinedNameToDescMap;
      typedef std::map<uint32_t, ConstString> SymbolIndexToName;
      UndefinedNameToDescMap undefined_name_to_desc;
      SymbolIndexToName reexport_shlib_needs_fixup;
      for (; nlist_idx < symtab_load_command.nsyms; ++nlist_idx) {
        struct nlist_64 nlist;
        if (!nlist_data.ValidOffsetForDataOfSize(nlist_data_offset,
                                                 nlist_byte_size))
          break;

        nlist.n_strx = nlist_data.GetU32_unchecked(&nlist_data_offset);
        nlist.n_type = nlist_data.GetU8_unchecked(&nlist_data_offset);
        nlist.n_sect = nlist_data.GetU8_unchecked(&nlist_data_offset);
        nlist.n_desc = nlist_data.GetU16_unchecked(&nlist_data_offset);
        nlist.n_value = nlist_data.GetAddress_unchecked(&nlist_data_offset);

        SymbolType type = eSymbolTypeInvalid;
        const char *symbol_name = NULL;

        if (have_strtab_data) {
          symbol_name = strtab_data.PeekCStr(nlist.n_strx);

          if (symbol_name == NULL) {
            // No symbol should be NULL, even the symbols with no string values
            // should have an offset zero which points to an empty C-string
            Host::SystemLog(Host::eSystemLogError,
                            "error: symbol[%u] has invalid string table offset "
                            "0x%x in %s, ignoring symbol\n",
                            nlist_idx, nlist.n_strx,
                            module_sp->GetFileSpec().GetPath().c_str());
            continue;
          }
          if (symbol_name[0] == '\0')
            symbol_name = NULL;
        } else {
          const addr_t str_addr = strtab_addr + nlist.n_strx;
          Status str_error;
          if (process->ReadCStringFromMemory(str_addr, memory_symbol_name,
                                             str_error))
            symbol_name = memory_symbol_name.c_str();
        }
        const char *symbol_name_non_abi_mangled = NULL;

        SectionSP symbol_section;
        lldb::addr_t symbol_byte_size = 0;
        bool add_nlist = true;
        bool is_gsym = false;
        bool is_debug = ((nlist.n_type & N_STAB) != 0);
        bool demangled_is_synthesized = false;
        bool set_value = true;
        assert(sym_idx < num_syms);

        sym[sym_idx].SetDebug(is_debug);

        if (is_debug) {
          switch (nlist.n_type) {
          case N_GSYM:
            // global symbol: name,,NO_SECT,type,0
            // Sometimes the N_GSYM value contains the address.

            // FIXME: In the .o files, we have a GSYM and a debug symbol for all
            // the ObjC data.  They
            // have the same address, but we want to ensure that we always find
            // only the real symbol, 'cause we don't currently correctly
            // attribute the GSYM one to the ObjCClass/Ivar/MetaClass symbol
            // type.  This is a temporary hack to make sure the ObjectiveC
            // symbols get treated correctly.  To do this right, we should
            // coalesce all the GSYM & global symbols that have the same
            // address.
            is_gsym = true;
            sym[sym_idx].SetExternal(true);

            if (symbol_name && symbol_name[0] == '_' && symbol_name[1] == 'O') {
              llvm::StringRef symbol_name_ref(symbol_name);
              if (symbol_name_ref.startswith(g_objc_v2_prefix_class)) {
                symbol_name_non_abi_mangled = symbol_name + 1;
                symbol_name = symbol_name + g_objc_v2_prefix_class.size();
                type = eSymbolTypeObjCClass;
                demangled_is_synthesized = true;

              } else if (symbol_name_ref.startswith(
                             g_objc_v2_prefix_metaclass)) {
                symbol_name_non_abi_mangled = symbol_name + 1;
                symbol_name = symbol_name + g_objc_v2_prefix_metaclass.size();
                type = eSymbolTypeObjCMetaClass;
                demangled_is_synthesized = true;
              } else if (symbol_name_ref.startswith(g_objc_v2_prefix_ivar)) {
                symbol_name_non_abi_mangled = symbol_name + 1;
                symbol_name = symbol_name + g_objc_v2_prefix_ivar.size();
                type = eSymbolTypeObjCIVar;
                demangled_is_synthesized = true;
              }
            } else {
              if (nlist.n_value != 0)
                symbol_section =
                    section_info.GetSection(nlist.n_sect, nlist.n_value);
              type = eSymbolTypeData;
            }
            break;

          case N_FNAME:
            // procedure name (f77 kludge): name,,NO_SECT,0,0
            type = eSymbolTypeCompiler;
            break;

          case N_FUN:
            // procedure: name,,n_sect,linenumber,address
            if (symbol_name) {
              type = eSymbolTypeCode;
              symbol_section =
                  section_info.GetSection(nlist.n_sect, nlist.n_value);

              N_FUN_addr_to_sym_idx.insert(
                  std::make_pair(nlist.n_value, sym_idx));
              // We use the current number of symbols in the symbol table in
              // lieu of using nlist_idx in case we ever start trimming entries
              // out
              N_FUN_indexes.push_back(sym_idx);
            } else {
              type = eSymbolTypeCompiler;

              if (!N_FUN_indexes.empty()) {
                // Copy the size of the function into the original STAB entry
                // so we don't have to hunt for it later
                symtab->SymbolAtIndex(N_FUN_indexes.back())
                    ->SetByteSize(nlist.n_value);
                N_FUN_indexes.pop_back();
                // We don't really need the end function STAB as it contains
                // the size which we already placed with the original symbol,
                // so don't add it if we want a minimal symbol table
                add_nlist = false;
              }
            }
            break;

          case N_STSYM:
            // static symbol: name,,n_sect,type,address
            N_STSYM_addr_to_sym_idx.insert(
                std::make_pair(nlist.n_value, sym_idx));
            symbol_section =
                section_info.GetSection(nlist.n_sect, nlist.n_value);
            if (symbol_name && symbol_name[0]) {
              type = ObjectFile::GetSymbolTypeFromName(symbol_name + 1,
                                                       eSymbolTypeData);
            }
            break;

          case N_LCSYM:
            // .lcomm symbol: name,,n_sect,type,address
            symbol_section =
                section_info.GetSection(nlist.n_sect, nlist.n_value);
            type = eSymbolTypeCommonBlock;
            break;

          case N_BNSYM:
            // We use the current number of symbols in the symbol table in lieu
            // of using nlist_idx in case we ever start trimming entries out
            // Skip these if we want minimal symbol tables
            add_nlist = false;
            break;

          case N_ENSYM:
            // Set the size of the N_BNSYM to the terminating index of this
            // N_ENSYM so that we can always skip the entire symbol if we need
            // to navigate more quickly at the source level when parsing STABS
            // Skip these if we want minimal symbol tables
            add_nlist = false;
            break;

          case N_OPT:
            // emitted with gcc2_compiled and in gcc source
            type = eSymbolTypeCompiler;
            break;

          case N_RSYM:
            // register sym: name,,NO_SECT,type,register
            type = eSymbolTypeVariable;
            break;

          case N_SLINE:
            // src line: 0,,n_sect,linenumber,address
            symbol_section =
                section_info.GetSection(nlist.n_sect, nlist.n_value);
            type = eSymbolTypeLineEntry;
            break;

          case N_SSYM:
            // structure elt: name,,NO_SECT,type,struct_offset
            type = eSymbolTypeVariableType;
            break;

          case N_SO:
            // source file name
            type = eSymbolTypeSourceFile;
            if (symbol_name == NULL) {
              add_nlist = false;
              if (N_SO_index != UINT32_MAX) {
                // Set the size of the N_SO to the terminating index of this
                // N_SO so that we can always skip the entire N_SO if we need
                // to navigate more quickly at the source level when parsing
                // STABS
                symbol_ptr = symtab->SymbolAtIndex(N_SO_index);
                symbol_ptr->SetByteSize(sym_idx);
                symbol_ptr->SetSizeIsSibling(true);
              }
              N_NSYM_indexes.clear();
              N_INCL_indexes.clear();
              N_BRAC_indexes.clear();
              N_COMM_indexes.clear();
              N_FUN_indexes.clear();
              N_SO_index = UINT32_MAX;
            } else {
              // We use the current number of symbols in the symbol table in
              // lieu of using nlist_idx in case we ever start trimming entries
              // out
              const bool N_SO_has_full_path = symbol_name[0] == '/';
              if (N_SO_has_full_path) {
                if ((N_SO_index == sym_idx - 1) && ((sym_idx - 1) < num_syms)) {
                  // We have two consecutive N_SO entries where the first
                  // contains a directory and the second contains a full path.
                  sym[sym_idx - 1].GetMangled().SetValue(
                      ConstString(symbol_name), false);
                  m_nlist_idx_to_sym_idx[nlist_idx] = sym_idx - 1;
                  add_nlist = false;
                } else {
                  // This is the first entry in a N_SO that contains a
                  // directory or a full path to the source file
                  N_SO_index = sym_idx;
                }
              } else if ((N_SO_index == sym_idx - 1) &&
                         ((sym_idx - 1) < num_syms)) {
                // This is usually the second N_SO entry that contains just the
                // filename, so here we combine it with the first one if we are
                // minimizing the symbol table
                const char *so_path =
                    sym[sym_idx - 1]
                        .GetMangled()
                        .GetDemangledName(lldb::eLanguageTypeUnknown)
                        .AsCString();
                if (so_path && so_path[0]) {
                  std::string full_so_path(so_path);
                  const size_t double_slash_pos = full_so_path.find("//");
                  if (double_slash_pos != std::string::npos) {
                    // The linker has been generating bad N_SO entries with
                    // doubled up paths in the format "%s%s" where the first
                    // string in the DW_AT_comp_dir, and the second is the
                    // directory for the source file so you end up with a path
                    // that looks like "/tmp/src//tmp/src/"
                    FileSpec so_dir(so_path, false);
                    if (!so_dir.Exists()) {
                      so_dir.SetFile(&full_so_path[double_slash_pos + 1], false,
                                     FileSpec::Style::native);
                      if (so_dir.Exists()) {
                        // Trim off the incorrect path
                        full_so_path.erase(0, double_slash_pos + 1);
                      }
                    }
                  }
                  if (*full_so_path.rbegin() != '/')
                    full_so_path += '/';
                  full_so_path += symbol_name;
                  sym[sym_idx - 1].GetMangled().SetValue(
                      ConstString(full_so_path.c_str()), false);
                  add_nlist = false;
                  m_nlist_idx_to_sym_idx[nlist_idx] = sym_idx - 1;
                }
              } else {
                // This could be a relative path to a N_SO
                N_SO_index = sym_idx;
              }
            }
            break;

          case N_OSO:
            // object file name: name,,0,0,st_mtime
            type = eSymbolTypeObjectFile;
            break;

          case N_LSYM:
            // local sym: name,,NO_SECT,type,offset
            type = eSymbolTypeLocal;
            break;

          //----------------------------------------------------------------------
          // INCL scopes
          //----------------------------------------------------------------------
          case N_BINCL:
            // include file beginning: name,,NO_SECT,0,sum We use the current
            // number of symbols in the symbol table in lieu of using nlist_idx
            // in case we ever start trimming entries out
            N_INCL_indexes.push_back(sym_idx);
            type = eSymbolTypeScopeBegin;
            break;

          case N_EINCL:
            // include file end: name,,NO_SECT,0,0
            // Set the size of the N_BINCL to the terminating index of this
            // N_EINCL so that we can always skip the entire symbol if we need
            // to navigate more quickly at the source level when parsing STABS
            if (!N_INCL_indexes.empty()) {
              symbol_ptr = symtab->SymbolAtIndex(N_INCL_indexes.back());
              symbol_ptr->SetByteSize(sym_idx + 1);
              symbol_ptr->SetSizeIsSibling(true);
              N_INCL_indexes.pop_back();
            }
            type = eSymbolTypeScopeEnd;
            break;

          case N_SOL:
            // #included file name: name,,n_sect,0,address
            type = eSymbolTypeHeaderFile;

            // We currently don't use the header files on darwin
            add_nlist = false;
            break;

          case N_PARAMS:
            // compiler parameters: name,,NO_SECT,0,0
            type = eSymbolTypeCompiler;
            break;

          case N_VERSION:
            // compiler version: name,,NO_SECT,0,0
            type = eSymbolTypeCompiler;
            break;

          case N_OLEVEL:
            // compiler -O level: name,,NO_SECT,0,0
            type = eSymbolTypeCompiler;
            break;

          case N_PSYM:
            // parameter: name,,NO_SECT,type,offset
            type = eSymbolTypeVariable;
            break;

          case N_ENTRY:
            // alternate entry: name,,n_sect,linenumber,address
            symbol_section =
                section_info.GetSection(nlist.n_sect, nlist.n_value);
            type = eSymbolTypeLineEntry;
            break;

          //----------------------------------------------------------------------
          // Left and Right Braces
          //----------------------------------------------------------------------
          case N_LBRAC:
            // left bracket: 0,,NO_SECT,nesting level,address We use the
            // current number of symbols in the symbol table in lieu of using
            // nlist_idx in case we ever start trimming entries out
            symbol_section =
                section_info.GetSection(nlist.n_sect, nlist.n_value);
            N_BRAC_indexes.push_back(sym_idx);
            type = eSymbolTypeScopeBegin;
            break;

          case N_RBRAC:
            // right bracket: 0,,NO_SECT,nesting level,address Set the size of
            // the N_LBRAC to the terminating index of this N_RBRAC so that we
            // can always skip the entire symbol if we need to navigate more
            // quickly at the source level when parsing STABS
            symbol_section =
                section_info.GetSection(nlist.n_sect, nlist.n_value);
            if (!N_BRAC_indexes.empty()) {
              symbol_ptr = symtab->SymbolAtIndex(N_BRAC_indexes.back());
              symbol_ptr->SetByteSize(sym_idx + 1);
              symbol_ptr->SetSizeIsSibling(true);
              N_BRAC_indexes.pop_back();
            }
            type = eSymbolTypeScopeEnd;
            break;

          case N_EXCL:
            // deleted include file: name,,NO_SECT,0,sum
            type = eSymbolTypeHeaderFile;
            break;

          //----------------------------------------------------------------------
          // COMM scopes
          //----------------------------------------------------------------------
          case N_BCOMM:
            // begin common: name,,NO_SECT,0,0
            // We use the current number of symbols in the symbol table in lieu
            // of using nlist_idx in case we ever start trimming entries out
            type = eSymbolTypeScopeBegin;
            N_COMM_indexes.push_back(sym_idx);
            break;

          case N_ECOML:
            // end common (local name): 0,,n_sect,0,address
            symbol_section =
                section_info.GetSection(nlist.n_sect, nlist.n_value);
            LLVM_FALLTHROUGH;

          case N_ECOMM:
            // end common: name,,n_sect,0,0
            // Set the size of the N_BCOMM to the terminating index of this
            // N_ECOMM/N_ECOML so that we can always skip the entire symbol if
            // we need to navigate more quickly at the source level when
            // parsing STABS
            if (!N_COMM_indexes.empty()) {
              symbol_ptr = symtab->SymbolAtIndex(N_COMM_indexes.back());
              symbol_ptr->SetByteSize(sym_idx + 1);
              symbol_ptr->SetSizeIsSibling(true);
              N_COMM_indexes.pop_back();
            }
            type = eSymbolTypeScopeEnd;
            break;

          case N_LENG:
            // second stab entry with length information
            type = eSymbolTypeAdditional;
            break;

          default:
            break;
          }
        } else {
          // uint8_t n_pext    = N_PEXT & nlist.n_type;
          uint8_t n_type = N_TYPE & nlist.n_type;
          sym[sym_idx].SetExternal((N_EXT & nlist.n_type) != 0);

          switch (n_type) {
          case N_INDR: {
            const char *reexport_name_cstr =
                strtab_data.PeekCStr(nlist.n_value);
            if (reexport_name_cstr && reexport_name_cstr[0]) {
              type = eSymbolTypeReExported;
              ConstString reexport_name(
                  reexport_name_cstr +
                  ((reexport_name_cstr[0] == '_') ? 1 : 0));
              sym[sym_idx].SetReExportedSymbolName(reexport_name);
              set_value = false;
              reexport_shlib_needs_fixup[sym_idx] = reexport_name;
              indirect_symbol_names.insert(
                  ConstString(symbol_name + ((symbol_name[0] == '_') ? 1 : 0)));
            } else
              type = eSymbolTypeUndefined;
          } break;

          case N_UNDF:
            if (symbol_name && symbol_name[0]) {
              ConstString undefined_name(symbol_name +
                                         ((symbol_name[0] == '_') ? 1 : 0));
              undefined_name_to_desc[undefined_name] = nlist.n_desc;
            }
            LLVM_FALLTHROUGH;

          case N_PBUD:
            type = eSymbolTypeUndefined;
            break;

          case N_ABS:
            type = eSymbolTypeAbsolute;
            break;

          case N_SECT: {
            symbol_section =
                section_info.GetSection(nlist.n_sect, nlist.n_value);

            if (!symbol_section) {
              // TODO: warn about this?
              add_nlist = false;
              break;
            }

            if (TEXT_eh_frame_sectID == nlist.n_sect) {
              type = eSymbolTypeException;
            } else {
              uint32_t section_type = symbol_section->Get() & SECTION_TYPE;

              switch (section_type) {
              case S_CSTRING_LITERALS:
                type = eSymbolTypeData;
                break; // section with only literal C strings
              case S_4BYTE_LITERALS:
                type = eSymbolTypeData;
                break; // section with only 4 byte literals
              case S_8BYTE_LITERALS:
                type = eSymbolTypeData;
                break; // section with only 8 byte literals
              case S_LITERAL_POINTERS:
                type = eSymbolTypeTrampoline;
                break; // section with only pointers to literals
              case S_NON_LAZY_SYMBOL_POINTERS:
                type = eSymbolTypeTrampoline;
                break; // section with only non-lazy symbol pointers
              case S_LAZY_SYMBOL_POINTERS:
                type = eSymbolTypeTrampoline;
                break; // section with only lazy symbol pointers
              case S_SYMBOL_STUBS:
                type = eSymbolTypeTrampoline;
                break; // section with only symbol stubs, byte size of stub in
                       // the reserved2 field
              case S_MOD_INIT_FUNC_POINTERS:
                type = eSymbolTypeCode;
                break; // section with only function pointers for initialization
              case S_MOD_TERM_FUNC_POINTERS:
                type = eSymbolTypeCode;
                break; // section with only function pointers for termination
              case S_INTERPOSING:
                type = eSymbolTypeTrampoline;
                break; // section with only pairs of function pointers for
                       // interposing
              case S_16BYTE_LITERALS:
                type = eSymbolTypeData;
                break; // section with only 16 byte literals
              case S_DTRACE_DOF:
                type = eSymbolTypeInstrumentation;
                break;
              case S_LAZY_DYLIB_SYMBOL_POINTERS:
                type = eSymbolTypeTrampoline;
                break;
              default:
                switch (symbol_section->GetType()) {
                case lldb::eSectionTypeCode:
                  type = eSymbolTypeCode;
                  break;
                case eSectionTypeData:
                case eSectionTypeDataCString:         // Inlined C string data
                case eSectionTypeDataCStringPointers: // Pointers to C string
                                                      // data
                case eSectionTypeDataSymbolAddress:   // Address of a symbol in
                                                      // the symbol table
                case eSectionTypeData4:
                case eSectionTypeData8:
                case eSectionTypeData16:
                  type = eSymbolTypeData;
                  break;
                default:
                  break;
                }
                break;
              }

              if (type == eSymbolTypeInvalid) {
                const char *symbol_sect_name =
                    symbol_section->GetName().AsCString();
                if (symbol_section->IsDescendant(text_section_sp.get())) {
                  if (symbol_section->IsClear(S_ATTR_PURE_INSTRUCTIONS |
                                              S_ATTR_SELF_MODIFYING_CODE |
                                              S_ATTR_SOME_INSTRUCTIONS))
                    type = eSymbolTypeData;
                  else
                    type = eSymbolTypeCode;
                } else if (symbol_section->IsDescendant(
                               data_section_sp.get()) ||
                           symbol_section->IsDescendant(
                               data_dirty_section_sp.get()) ||
                           symbol_section->IsDescendant(
                               data_const_section_sp.get())) {
                  if (symbol_sect_name &&
                      ::strstr(symbol_sect_name, "__objc") ==
                          symbol_sect_name) {
                    type = eSymbolTypeRuntime;

                    if (symbol_name) {
                      llvm::StringRef symbol_name_ref(symbol_name);
                      if (symbol_name_ref.startswith("_OBJC_")) {
                        static const llvm::StringRef g_objc_v2_prefix_class(
                            "_OBJC_CLASS_$_");
                        static const llvm::StringRef g_objc_v2_prefix_metaclass(
                            "_OBJC_METACLASS_$_");
                        static const llvm::StringRef g_objc_v2_prefix_ivar(
                            "_OBJC_IVAR_$_");
                        if (symbol_name_ref.startswith(
                                g_objc_v2_prefix_class)) {
                          symbol_name_non_abi_mangled = symbol_name + 1;
                          symbol_name =
                              symbol_name + g_objc_v2_prefix_class.size();
                          type = eSymbolTypeObjCClass;
                          demangled_is_synthesized = true;
                        } else if (symbol_name_ref.startswith(
                                       g_objc_v2_prefix_metaclass)) {
                          symbol_name_non_abi_mangled = symbol_name + 1;
                          symbol_name =
                              symbol_name + g_objc_v2_prefix_metaclass.size();
                          type = eSymbolTypeObjCMetaClass;
                          demangled_is_synthesized = true;
                        } else if (symbol_name_ref.startswith(
                                       g_objc_v2_prefix_ivar)) {
                          symbol_name_non_abi_mangled = symbol_name + 1;
                          symbol_name =
                              symbol_name + g_objc_v2_prefix_ivar.size();
                          type = eSymbolTypeObjCIVar;
                          demangled_is_synthesized = true;
                        }
                      }
                    }
                  } else if (symbol_sect_name &&
                             ::strstr(symbol_sect_name, "__gcc_except_tab") ==
                                 symbol_sect_name) {
                    type = eSymbolTypeException;
                  } else {
                    type = eSymbolTypeData;
                  }
                } else if (symbol_sect_name &&
                           ::strstr(symbol_sect_name, "__IMPORT") ==
                               symbol_sect_name) {
                  type = eSymbolTypeTrampoline;
                } else if (symbol_section->IsDescendant(
                               objc_section_sp.get())) {
                  type = eSymbolTypeRuntime;
                  if (symbol_name && symbol_name[0] == '.') {
                    llvm::StringRef symbol_name_ref(symbol_name);
                    static const llvm::StringRef g_objc_v1_prefix_class(
                        ".objc_class_name_");
                    if (symbol_name_ref.startswith(g_objc_v1_prefix_class)) {
                      symbol_name_non_abi_mangled = symbol_name;
                      symbol_name = symbol_name + g_objc_v1_prefix_class.size();
                      type = eSymbolTypeObjCClass;
                      demangled_is_synthesized = true;
                    }
                  }
                }
              }
            }
          } break;
          }
        }

        if (add_nlist) {
          uint64_t symbol_value = nlist.n_value;

          if (symbol_name_non_abi_mangled) {
            sym[sym_idx].GetMangled().SetMangledName(
                ConstString(symbol_name_non_abi_mangled));
            sym[sym_idx].GetMangled().SetDemangledName(
                ConstString(symbol_name));
          } else {
            bool symbol_name_is_mangled = false;

            if (symbol_name && symbol_name[0] == '_') {
              symbol_name_is_mangled = symbol_name[1] == '_';
              symbol_name++; // Skip the leading underscore
            }

            if (symbol_name) {
              ConstString const_symbol_name(symbol_name);
              sym[sym_idx].GetMangled().SetValue(const_symbol_name,
                                                 symbol_name_is_mangled);
            }
          }

          if (is_gsym) {
            const char *gsym_name = sym[sym_idx]
                                        .GetMangled()
                                        .GetName(lldb::eLanguageTypeUnknown,
                                                 Mangled::ePreferMangled)
                                        .GetCString();
            if (gsym_name)
              N_GSYM_name_to_sym_idx[gsym_name] = sym_idx;
          }

          if (symbol_section) {
            const addr_t section_file_addr = symbol_section->GetFileAddress();
            if (symbol_byte_size == 0 && function_starts_count > 0) {
              addr_t symbol_lookup_file_addr = nlist.n_value;
              // Do an exact address match for non-ARM addresses, else get the
              // closest since the symbol might be a thumb symbol which has an
              // address with bit zero set
              FunctionStarts::Entry *func_start_entry =
                  function_starts.FindEntry(symbol_lookup_file_addr, !is_arm);
              if (is_arm && func_start_entry) {
                // Verify that the function start address is the symbol address
                // (ARM) or the symbol address + 1 (thumb)
                if (func_start_entry->addr != symbol_lookup_file_addr &&
                    func_start_entry->addr != (symbol_lookup_file_addr + 1)) {
                  // Not the right entry, NULL it out...
                  func_start_entry = NULL;
                }
              }
              if (func_start_entry) {
                func_start_entry->data = true;

                addr_t symbol_file_addr = func_start_entry->addr;
                if (is_arm)
                  symbol_file_addr &= THUMB_ADDRESS_BIT_MASK;

                const FunctionStarts::Entry *next_func_start_entry =
                    function_starts.FindNextEntry(func_start_entry);
                const addr_t section_end_file_addr =
                    section_file_addr + symbol_section->GetByteSize();
                if (next_func_start_entry) {
                  addr_t next_symbol_file_addr = next_func_start_entry->addr;
                  // Be sure the clear the Thumb address bit when we calculate
                  // the size from the current and next address
                  if (is_arm)
                    next_symbol_file_addr &= THUMB_ADDRESS_BIT_MASK;
                  symbol_byte_size = std::min<lldb::addr_t>(
                      next_symbol_file_addr - symbol_file_addr,
                      section_end_file_addr - symbol_file_addr);
                } else {
                  symbol_byte_size = section_end_file_addr - symbol_file_addr;
                }
              }
            }
            symbol_value -= section_file_addr;
          }

          if (is_debug == false) {
            if (type == eSymbolTypeCode) {
              // See if we can find a N_FUN entry for any code symbols. If we
              // do find a match, and the name matches, then we can merge the
              // two into just the function symbol to avoid duplicate entries
              // in the symbol table
              std::pair<ValueToSymbolIndexMap::const_iterator,
                        ValueToSymbolIndexMap::const_iterator>
                  range;
              range = N_FUN_addr_to_sym_idx.equal_range(nlist.n_value);
              if (range.first != range.second) {
                bool found_it = false;
                for (ValueToSymbolIndexMap::const_iterator pos = range.first;
                     pos != range.second; ++pos) {
                  if (sym[sym_idx].GetMangled().GetName(
                          lldb::eLanguageTypeUnknown,
                          Mangled::ePreferMangled) ==
                      sym[pos->second].GetMangled().GetName(
                          lldb::eLanguageTypeUnknown,
                          Mangled::ePreferMangled)) {
                    m_nlist_idx_to_sym_idx[nlist_idx] = pos->second;
                    // We just need the flags from the linker symbol, so put
                    // these flags into the N_FUN flags to avoid duplicate
                    // symbols in the symbol table
                    sym[pos->second].SetExternal(sym[sym_idx].IsExternal());
                    sym[pos->second].SetFlags(nlist.n_type << 16 |
                                              nlist.n_desc);
                    if (resolver_addresses.find(nlist.n_value) !=
                        resolver_addresses.end())
                      sym[pos->second].SetType(eSymbolTypeResolver);
                    sym[sym_idx].Clear();
                    found_it = true;
                    break;
                  }
                }
                if (found_it)
                  continue;
              } else {
                if (resolver_addresses.find(nlist.n_value) !=
                    resolver_addresses.end())
                  type = eSymbolTypeResolver;
              }
            } else if (type == eSymbolTypeData ||
                       type == eSymbolTypeObjCClass ||
                       type == eSymbolTypeObjCMetaClass ||
                       type == eSymbolTypeObjCIVar) {
              // See if we can find a N_STSYM entry for any data symbols. If we
              // do find a match, and the name matches, then we can merge the
              // two into just the Static symbol to avoid duplicate entries in
              // the symbol table
              std::pair<ValueToSymbolIndexMap::const_iterator,
                        ValueToSymbolIndexMap::const_iterator>
                  range;
              range = N_STSYM_addr_to_sym_idx.equal_range(nlist.n_value);
              if (range.first != range.second) {
                bool found_it = false;
                for (ValueToSymbolIndexMap::const_iterator pos = range.first;
                     pos != range.second; ++pos) {
                  if (sym[sym_idx].GetMangled().GetName(
                          lldb::eLanguageTypeUnknown,
                          Mangled::ePreferMangled) ==
                      sym[pos->second].GetMangled().GetName(
                          lldb::eLanguageTypeUnknown,
                          Mangled::ePreferMangled)) {
                    m_nlist_idx_to_sym_idx[nlist_idx] = pos->second;
                    // We just need the flags from the linker symbol, so put
                    // these flags into the N_STSYM flags to avoid duplicate
                    // symbols in the symbol table
                    sym[pos->second].SetExternal(sym[sym_idx].IsExternal());
                    sym[pos->second].SetFlags(nlist.n_type << 16 |
                                              nlist.n_desc);
                    sym[sym_idx].Clear();
                    found_it = true;
                    break;
                  }
                }
                if (found_it)
                  continue;
              } else {
                // Combine N_GSYM stab entries with the non stab symbol
                const char *gsym_name = sym[sym_idx]
                                            .GetMangled()
                                            .GetName(lldb::eLanguageTypeUnknown,
                                                     Mangled::ePreferMangled)
                                            .GetCString();
                if (gsym_name) {
                  ConstNameToSymbolIndexMap::const_iterator pos =
                      N_GSYM_name_to_sym_idx.find(gsym_name);
                  if (pos != N_GSYM_name_to_sym_idx.end()) {
                    const uint32_t GSYM_sym_idx = pos->second;
                    m_nlist_idx_to_sym_idx[nlist_idx] = GSYM_sym_idx;
                    // Copy the address, because often the N_GSYM address has
                    // an invalid address of zero when the global is a common
                    // symbol
                    sym[GSYM_sym_idx].GetAddressRef().SetSection(
                        symbol_section);
                    sym[GSYM_sym_idx].GetAddressRef().SetOffset(symbol_value);
                    // We just need the flags from the linker symbol, so put
                    // these flags into the N_GSYM flags to avoid duplicate
                    // symbols in the symbol table
                    sym[GSYM_sym_idx].SetFlags(nlist.n_type << 16 |
                                               nlist.n_desc);
                    sym[sym_idx].Clear();
                    continue;
                  }
                }
              }
            }
          }

          sym[sym_idx].SetID(nlist_idx);
          sym[sym_idx].SetType(type);
          if (set_value) {
            sym[sym_idx].GetAddressRef().SetSection(symbol_section);
            sym[sym_idx].GetAddressRef().SetOffset(symbol_value);
          }
          sym[sym_idx].SetFlags(nlist.n_type << 16 | nlist.n_desc);

          if (symbol_byte_size > 0)
            sym[sym_idx].SetByteSize(symbol_byte_size);

          if (demangled_is_synthesized)
            sym[sym_idx].SetDemangledNameIsSynthesized(true);

          ++sym_idx;
        } else {
          sym[sym_idx].Clear();
        }
      }

      for (const auto &pos : reexport_shlib_needs_fixup) {
        const auto undef_pos = undefined_name_to_desc.find(pos.second);
        if (undef_pos != undefined_name_to_desc.end()) {
          const uint8_t dylib_ordinal =
              llvm::MachO::GET_LIBRARY_ORDINAL(undef_pos->second);
          if (dylib_ordinal > 0 && dylib_ordinal < dylib_files.GetSize())
            sym[pos.first].SetReExportedSymbolSharedLibrary(
                dylib_files.GetFileSpecAtIndex(dylib_ordinal - 1));
        }
      }
    }

    uint32_t synthetic_sym_id = symtab_load_command.nsyms;

    if (function_starts_count > 0) {
      uint32_t num_synthetic_function_symbols = 0;
      for (i = 0; i < function_starts_count; ++i) {
        if (function_starts.GetEntryRef(i).data == false)
          ++num_synthetic_function_symbols;
      }

      if (num_synthetic_function_symbols > 0) {
        if (num_syms < sym_idx + num_synthetic_function_symbols) {
          num_syms = sym_idx + num_synthetic_function_symbols;
          sym = symtab->Resize(num_syms);
        }
        for (i = 0; i < function_starts_count; ++i) {
          const FunctionStarts::Entry *func_start_entry =
              function_starts.GetEntryAtIndex(i);
          if (func_start_entry->data == false) {
            addr_t symbol_file_addr = func_start_entry->addr;
            uint32_t symbol_flags = 0;
            if (is_arm) {
              if (symbol_file_addr & 1)
                symbol_flags = MACHO_NLIST_ARM_SYMBOL_IS_THUMB;
              symbol_file_addr &= THUMB_ADDRESS_BIT_MASK;
            }
            Address symbol_addr;
            if (module_sp->ResolveFileAddress(symbol_file_addr, symbol_addr)) {
              SectionSP symbol_section(symbol_addr.GetSection());
              uint32_t symbol_byte_size = 0;
              if (symbol_section) {
                const addr_t section_file_addr =
                    symbol_section->GetFileAddress();
                const FunctionStarts::Entry *next_func_start_entry =
                    function_starts.FindNextEntry(func_start_entry);
                const addr_t section_end_file_addr =
                    section_file_addr + symbol_section->GetByteSize();
                if (next_func_start_entry) {
                  addr_t next_symbol_file_addr = next_func_start_entry->addr;
                  if (is_arm)
                    next_symbol_file_addr &= THUMB_ADDRESS_BIT_MASK;
                  symbol_byte_size = std::min<lldb::addr_t>(
                      next_symbol_file_addr - symbol_file_addr,
                      section_end_file_addr - symbol_file_addr);
                } else {
                  symbol_byte_size = section_end_file_addr - symbol_file_addr;
                }
                sym[sym_idx].SetID(synthetic_sym_id++);
                sym[sym_idx].GetMangled().SetDemangledName(
                    GetNextSyntheticSymbolName());
                sym[sym_idx].SetType(eSymbolTypeCode);
                sym[sym_idx].SetIsSynthetic(true);
                sym[sym_idx].GetAddressRef() = symbol_addr;
                if (symbol_flags)
                  sym[sym_idx].SetFlags(symbol_flags);
                if (symbol_byte_size)
                  sym[sym_idx].SetByteSize(symbol_byte_size);
                ++sym_idx;
              }
            }
          }
        }
      }
    }

    // Trim our symbols down to just what we ended up with after removing any
    // symbols.
    if (sym_idx < num_syms) {
      num_syms = sym_idx;
      sym = symtab->Resize(num_syms);
    }

    // Now synthesize indirect symbols
    if (m_dysymtab.nindirectsyms != 0) {
      if (indirect_symbol_index_data.GetByteSize()) {
        NListIndexToSymbolIndexMap::const_iterator end_index_pos =
            m_nlist_idx_to_sym_idx.end();

        for (uint32_t sect_idx = 1; sect_idx < m_mach_sections.size();
             ++sect_idx) {
          if ((m_mach_sections[sect_idx].flags & SECTION_TYPE) ==
              S_SYMBOL_STUBS) {
            uint32_t symbol_stub_byte_size =
                m_mach_sections[sect_idx].reserved2;
            if (symbol_stub_byte_size == 0)
              continue;

            const uint32_t num_symbol_stubs =
                m_mach_sections[sect_idx].size / symbol_stub_byte_size;

            if (num_symbol_stubs == 0)
              continue;

            const uint32_t symbol_stub_index_offset =
                m_mach_sections[sect_idx].reserved1;
            for (uint32_t stub_idx = 0; stub_idx < num_symbol_stubs;
                 ++stub_idx) {
              const uint32_t symbol_stub_index =
                  symbol_stub_index_offset + stub_idx;
              const lldb::addr_t symbol_stub_addr =
                  m_mach_sections[sect_idx].addr +
                  (stub_idx * symbol_stub_byte_size);
              lldb::offset_t symbol_stub_offset = symbol_stub_index * 4;
              if (indirect_symbol_index_data.ValidOffsetForDataOfSize(
                      symbol_stub_offset, 4)) {
                const uint32_t stub_sym_id =
                    indirect_symbol_index_data.GetU32(&symbol_stub_offset);
                if (stub_sym_id & (INDIRECT_SYMBOL_ABS | INDIRECT_SYMBOL_LOCAL))
                  continue;

                NListIndexToSymbolIndexMap::const_iterator index_pos =
                    m_nlist_idx_to_sym_idx.find(stub_sym_id);
                Symbol *stub_symbol = NULL;
                if (index_pos != end_index_pos) {
                  // We have a remapping from the original nlist index to a
                  // current symbol index, so just look this up by index
                  stub_symbol = symtab->SymbolAtIndex(index_pos->second);
                } else {
                  // We need to lookup a symbol using the original nlist symbol
                  // index since this index is coming from the S_SYMBOL_STUBS
                  stub_symbol = symtab->FindSymbolByID(stub_sym_id);
                }

                if (stub_symbol) {
                  Address so_addr(symbol_stub_addr, section_list);

                  if (stub_symbol->GetType() == eSymbolTypeUndefined) {
                    // Change the external symbol into a trampoline that makes
                    // sense These symbols were N_UNDF N_EXT, and are useless
                    // to us, so we can re-use them so we don't have to make up
                    // a synthetic symbol for no good reason.
                    if (resolver_addresses.find(symbol_stub_addr) ==
                        resolver_addresses.end())
                      stub_symbol->SetType(eSymbolTypeTrampoline);
                    else
                      stub_symbol->SetType(eSymbolTypeResolver);
                    stub_symbol->SetExternal(false);
                    stub_symbol->GetAddressRef() = so_addr;
                    stub_symbol->SetByteSize(symbol_stub_byte_size);
                  } else {
                    // Make a synthetic symbol to describe the trampoline stub
                    Mangled stub_symbol_mangled_name(stub_symbol->GetMangled());
                    if (sym_idx >= num_syms) {
                      sym = symtab->Resize(++num_syms);
                      stub_symbol = NULL; // this pointer no longer valid
                    }
                    sym[sym_idx].SetID(synthetic_sym_id++);
                    sym[sym_idx].GetMangled() = stub_symbol_mangled_name;
                    if (resolver_addresses.find(symbol_stub_addr) ==
                        resolver_addresses.end())
                      sym[sym_idx].SetType(eSymbolTypeTrampoline);
                    else
                      sym[sym_idx].SetType(eSymbolTypeResolver);
                    sym[sym_idx].SetIsSynthetic(true);
                    sym[sym_idx].GetAddressRef() = so_addr;
                    sym[sym_idx].SetByteSize(symbol_stub_byte_size);
                    ++sym_idx;
                  }
                } else {
                  if (log)
                    log->Warning("symbol stub referencing symbol table symbol "
                                 "%u that isn't in our minimal symbol table, "
                                 "fix this!!!",
                                 stub_sym_id);
                }
              }
            }
          }
        }
      }
    }

    if (!trie_entries.empty()) {
      for (const auto &e : trie_entries) {
        if (e.entry.import_name) {
          // Only add indirect symbols from the Trie entries if we didn't have
          // a N_INDR nlist entry for this already
          if (indirect_symbol_names.find(e.entry.name) ==
              indirect_symbol_names.end()) {
            // Make a synthetic symbol to describe re-exported symbol.
            if (sym_idx >= num_syms)
              sym = symtab->Resize(++num_syms);
            sym[sym_idx].SetID(synthetic_sym_id++);
            sym[sym_idx].GetMangled() = Mangled(e.entry.name);
            sym[sym_idx].SetType(eSymbolTypeReExported);
            sym[sym_idx].SetIsSynthetic(true);
            sym[sym_idx].SetReExportedSymbolName(e.entry.import_name);
            if (e.entry.other > 0 && e.entry.other <= dylib_files.GetSize()) {
              sym[sym_idx].SetReExportedSymbolSharedLibrary(
                  dylib_files.GetFileSpecAtIndex(e.entry.other - 1));
            }
            ++sym_idx;
          }
        }
      }
    }

    //        StreamFile s(stdout, false);
    //        s.Printf ("Symbol table before CalculateSymbolSizes():\n");
    //        symtab->Dump(&s, NULL, eSortOrderNone);
    // Set symbol byte sizes correctly since mach-o nlist entries don't have
    // sizes
    symtab->CalculateSymbolSizes();

    //        s.Printf ("Symbol table after CalculateSymbolSizes():\n");
    //        symtab->Dump(&s, NULL, eSortOrderNone);

    return symtab->GetNumSymbols();
  }
  return 0;
}

void ObjectFileMachO::Dump(Stream *s) {
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    s->Printf("%p: ", static_cast<void *>(this));
    s->Indent();
    if (m_header.magic == MH_MAGIC_64 || m_header.magic == MH_CIGAM_64)
      s->PutCString("ObjectFileMachO64");
    else
      s->PutCString("ObjectFileMachO32");

    ArchSpec header_arch;
    GetArchitecture(header_arch);

    *s << ", file = '" << m_file
       << "', triple = " << header_arch.GetTriple().getTriple() << "\n";

    SectionList *sections = GetSectionList();
    if (sections)
      sections->Dump(s, NULL, true, UINT32_MAX);

    if (m_symtab_ap.get())
      m_symtab_ap->Dump(s, NULL, eSortOrderNone);
  }
}

bool ObjectFileMachO::GetUUID(const llvm::MachO::mach_header &header,
                              const lldb_private::DataExtractor &data,
                              lldb::offset_t lc_offset,
                              lldb_private::UUID &uuid) {
  uint32_t i;
  struct uuid_command load_cmd;

  lldb::offset_t offset = lc_offset;
  for (i = 0; i < header.ncmds; ++i) {
    const lldb::offset_t cmd_offset = offset;
    if (data.GetU32(&offset, &load_cmd, 2) == NULL)
      break;

    if (load_cmd.cmd == LC_UUID) {
      const uint8_t *uuid_bytes = data.PeekData(offset, 16);

      if (uuid_bytes) {
        // OpenCL on Mac OS X uses the same UUID for each of its object files.
        // We pretend these object files have no UUID to prevent crashing.

        const uint8_t opencl_uuid[] = {0x8c, 0x8e, 0xb3, 0x9b, 0x3b, 0xa8,
                                       0x4b, 0x16, 0xb6, 0xa4, 0x27, 0x63,
                                       0xbb, 0x14, 0xf0, 0x0d};

        if (!memcmp(uuid_bytes, opencl_uuid, 16))
          return false;

        uuid.SetBytes(uuid_bytes);
        return true;
      }
      return false;
    }
    offset = cmd_offset + load_cmd.cmdsize;
  }
  return false;
}

static const char *GetOSName(uint32_t cmd) {
  switch (cmd) {
  case llvm::MachO::LC_VERSION_MIN_IPHONEOS:
    return "ios";
  case llvm::MachO::LC_VERSION_MIN_MACOSX:
    return "macosx";
  case llvm::MachO::LC_VERSION_MIN_TVOS:
    return "tvos";
  case llvm::MachO::LC_VERSION_MIN_WATCHOS:
    return "watchos";
  default:
    llvm_unreachable("unexpected LC_VERSION load command");
  }
}

bool ObjectFileMachO::GetArchitecture(const llvm::MachO::mach_header &header,
                                      const lldb_private::DataExtractor &data,
                                      lldb::offset_t lc_offset,
                                      ArchSpec &arch) {
  arch.SetArchitecture(eArchTypeMachO, header.cputype, header.cpusubtype);

  if (arch.IsValid()) {
    llvm::Triple &triple = arch.GetTriple();

    // Set OS to an unspecified unknown or a "*" so it can match any OS
    triple.setOS(llvm::Triple::UnknownOS);
    triple.setOSName(llvm::StringRef());

    if (header.filetype == MH_PRELOAD) {
      if (header.cputype == CPU_TYPE_ARM) {
        // If this is a 32-bit arm binary, and it's a standalone binary, force
        // the Vendor to Apple so we don't accidentally pick up the generic
        // armv7 ABI at runtime.  Apple's armv7 ABI always uses r7 for the
        // frame pointer register; most other armv7 ABIs use a combination of
        // r7 and r11.
        triple.setVendor(llvm::Triple::Apple);
      } else {
        // Set vendor to an unspecified unknown or a "*" so it can match any
        // vendor This is required for correct behavior of EFI debugging on
        // x86_64
        triple.setVendor(llvm::Triple::UnknownVendor);
        triple.setVendorName(llvm::StringRef());
      }
      return true;
    } else {
      struct load_command load_cmd;

      lldb::offset_t offset = lc_offset;
      for (uint32_t i = 0; i < header.ncmds; ++i) {
        const lldb::offset_t cmd_offset = offset;
        if (data.GetU32(&offset, &load_cmd, 2) == NULL)
          break;

        uint32_t major, minor, patch;
        struct version_min_command version_min;

        llvm::SmallString<16> os_name;
        llvm::raw_svector_ostream os(os_name);

        switch (load_cmd.cmd) {
        case llvm::MachO::LC_VERSION_MIN_IPHONEOS:
        case llvm::MachO::LC_VERSION_MIN_MACOSX:
        case llvm::MachO::LC_VERSION_MIN_TVOS:
        case llvm::MachO::LC_VERSION_MIN_WATCHOS:
          if (load_cmd.cmdsize != sizeof(version_min))
            break;
          data.ExtractBytes(cmd_offset,
                            sizeof(version_min), data.GetByteOrder(),
                            &version_min);
          major = version_min.version >> 16;
          minor = (version_min.version >> 8) & 0xffu;
          patch = version_min.version & 0xffu;
          os << GetOSName(load_cmd.cmd) << major << '.' << minor << '.'
             << patch;
          triple.setOSName(os.str());
          return true;
        default:
          break;
        }

        offset = cmd_offset + load_cmd.cmdsize;
      }

      if (header.filetype != MH_KEXT_BUNDLE) {
        // We didn't find a LC_VERSION_MIN load command and this isn't a KEXT
        // so lets not say our Vendor is Apple, leave it as an unspecified
        // unknown
        triple.setVendor(llvm::Triple::UnknownVendor);
        triple.setVendorName(llvm::StringRef());
      }
    }
  }
  return arch.IsValid();
}

bool ObjectFileMachO::GetUUID(lldb_private::UUID *uuid) {
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
    return GetUUID(m_header, m_data, offset, *uuid);
  }
  return false;
}

uint32_t ObjectFileMachO::GetDependentModules(FileSpecList &files) {
  uint32_t count = 0;
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    struct load_command load_cmd;
    lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
    std::vector<std::string> rpath_paths;
    std::vector<std::string> rpath_relative_paths;
    std::vector<std::string> at_exec_relative_paths;
    const bool resolve_path = false; // Don't resolve the dependent file paths
                                     // since they may not reside on this
                                     // system
    uint32_t i;
    for (i = 0; i < m_header.ncmds; ++i) {
      const uint32_t cmd_offset = offset;
      if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
        break;

      switch (load_cmd.cmd) {
      case LC_RPATH:
      case LC_LOAD_DYLIB:
      case LC_LOAD_WEAK_DYLIB:
      case LC_REEXPORT_DYLIB:
      case LC_LOAD_DYLINKER:
      case LC_LOADFVMLIB:
      case LC_LOAD_UPWARD_DYLIB: {
        uint32_t name_offset = cmd_offset + m_data.GetU32(&offset);
        const char *path = m_data.PeekCStr(name_offset);
        if (path) {
          if (load_cmd.cmd == LC_RPATH)
            rpath_paths.push_back(path);
          else {
            if (path[0] == '@') {
              if (strncmp(path, "@rpath", strlen("@rpath")) == 0)
                rpath_relative_paths.push_back(path + strlen("@rpath"));
              else if (strncmp(path, "@executable_path", 
                       strlen("@executable_path")) == 0)
                at_exec_relative_paths.push_back(path 
                                                 + strlen("@executable_path"));
            } else {
              FileSpec file_spec(path, resolve_path);
              if (files.AppendIfUnique(file_spec))
                count++;
            }
          }
        }
      } break;

      default:
        break;
      }
      offset = cmd_offset + load_cmd.cmdsize;
    }

    FileSpec this_file_spec(m_file);
    this_file_spec.ResolvePath();
    
    if (!rpath_paths.empty()) {
      // Fixup all LC_RPATH values to be absolute paths
      std::string loader_path("@loader_path");
      std::string executable_path("@executable_path");
      for (auto &rpath : rpath_paths) {
        if (rpath.find(loader_path) == 0) {
          rpath.erase(0, loader_path.size());
          rpath.insert(0, this_file_spec.GetDirectory().GetCString());
        } else if (rpath.find(executable_path) == 0) {
          rpath.erase(0, executable_path.size());
          rpath.insert(0, this_file_spec.GetDirectory().GetCString());
        }
      }

      for (const auto &rpath_relative_path : rpath_relative_paths) {
        for (const auto &rpath : rpath_paths) {
          std::string path = rpath;
          path += rpath_relative_path;
          // It is OK to resolve this path because we must find a file on disk
          // for us to accept it anyway if it is rpath relative.
          FileSpec file_spec(path, true);
          if (file_spec.Exists() && files.AppendIfUnique(file_spec)) {
            count++;
            break;
          }
        }
      }
    }

    // We may have @executable_paths but no RPATHS.  Figure those out here.
    // Only do this if this object file is the executable.  We have no way to
    // get back to the actual executable otherwise, so we won't get the right
    // path.
    if (!at_exec_relative_paths.empty() && CalculateType() == eTypeExecutable) {
      FileSpec exec_dir = this_file_spec.CopyByRemovingLastPathComponent();
      for (const auto &at_exec_relative_path : at_exec_relative_paths) {
        FileSpec file_spec = 
            exec_dir.CopyByAppendingPathComponent(at_exec_relative_path);
        if (file_spec.Exists() && files.AppendIfUnique(file_spec))
          count++;
      }
    }
  }
  return count;
}

lldb_private::Address ObjectFileMachO::GetEntryPointAddress() {
  // If the object file is not an executable it can't hold the entry point.
  // m_entry_point_address is initialized to an invalid address, so we can just
  // return that. If m_entry_point_address is valid it means we've found it
  // already, so return the cached value.

  if (!IsExecutable() || m_entry_point_address.IsValid())
    return m_entry_point_address;

  // Otherwise, look for the UnixThread or Thread command.  The data for the
  // Thread command is given in /usr/include/mach-o.h, but it is basically:
  //
  //  uint32_t flavor  - this is the flavor argument you would pass to
  //  thread_get_state
  //  uint32_t count   - this is the count of longs in the thread state data
  //  struct XXX_thread_state state - this is the structure from
  //  <machine/thread_status.h> corresponding to the flavor.
  //  <repeat this trio>
  //
  // So we just keep reading the various register flavors till we find the GPR
  // one, then read the PC out of there.
  // FIXME: We will need to have a "RegisterContext data provider" class at some
  // point that can get all the registers
  // out of data in this form & attach them to a given thread.  That should
  // underlie the MacOS X User process plugin, and we'll also need it for the
  // MacOS X Core File process plugin.  When we have that we can also use it
  // here.
  //
  // For now we hard-code the offsets and flavors we need:
  //
  //

  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    struct load_command load_cmd;
    lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
    uint32_t i;
    lldb::addr_t start_address = LLDB_INVALID_ADDRESS;
    bool done = false;

    for (i = 0; i < m_header.ncmds; ++i) {
      const lldb::offset_t cmd_offset = offset;
      if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
        break;

      switch (load_cmd.cmd) {
      case LC_UNIXTHREAD:
      case LC_THREAD: {
        while (offset < cmd_offset + load_cmd.cmdsize) {
          uint32_t flavor = m_data.GetU32(&offset);
          uint32_t count = m_data.GetU32(&offset);
          if (count == 0) {
            // We've gotten off somehow, log and exit;
            return m_entry_point_address;
          }

          switch (m_header.cputype) {
          case llvm::MachO::CPU_TYPE_ARM:
            if (flavor == 1 ||
                flavor == 9) // ARM_THREAD_STATE/ARM_THREAD_STATE32 from
                             // mach/arm/thread_status.h
            {
              offset += 60; // This is the offset of pc in the GPR thread state
                            // data structure.
              start_address = m_data.GetU32(&offset);
              done = true;
            }
            break;
          case llvm::MachO::CPU_TYPE_ARM64:
            if (flavor == 6) // ARM_THREAD_STATE64 from mach/arm/thread_status.h
            {
              offset += 256; // This is the offset of pc in the GPR thread state
                             // data structure.
              start_address = m_data.GetU64(&offset);
              done = true;
            }
            break;
          case llvm::MachO::CPU_TYPE_I386:
            if (flavor ==
                1) // x86_THREAD_STATE32 from mach/i386/thread_status.h
            {
              offset += 40; // This is the offset of eip in the GPR thread state
                            // data structure.
              start_address = m_data.GetU32(&offset);
              done = true;
            }
            break;
          case llvm::MachO::CPU_TYPE_X86_64:
            if (flavor ==
                4) // x86_THREAD_STATE64 from mach/i386/thread_status.h
            {
              offset += 16 * 8; // This is the offset of rip in the GPR thread
                                // state data structure.
              start_address = m_data.GetU64(&offset);
              done = true;
            }
            break;
          default:
            return m_entry_point_address;
          }
          // Haven't found the GPR flavor yet, skip over the data for this
          // flavor:
          if (done)
            break;
          offset += count * 4;
        }
      } break;
      case LC_MAIN: {
        ConstString text_segment_name("__TEXT");
        uint64_t entryoffset = m_data.GetU64(&offset);
        SectionSP text_segment_sp =
            GetSectionList()->FindSectionByName(text_segment_name);
        if (text_segment_sp) {
          done = true;
          start_address = text_segment_sp->GetFileAddress() + entryoffset;
        }
      } break;

      default:
        break;
      }
      if (done)
        break;

      // Go to the next load command:
      offset = cmd_offset + load_cmd.cmdsize;
    }

    if (start_address != LLDB_INVALID_ADDRESS) {
      // We got the start address from the load commands, so now resolve that
      // address in the sections of this ObjectFile:
      if (!m_entry_point_address.ResolveAddressUsingFileSections(
              start_address, GetSectionList())) {
        m_entry_point_address.Clear();
      }
    } else {
      // We couldn't read the UnixThread load command - maybe it wasn't there.
      // As a fallback look for the "start" symbol in the main executable.

      ModuleSP module_sp(GetModule());

      if (module_sp) {
        SymbolContextList contexts;
        SymbolContext context;
        if (module_sp->FindSymbolsWithNameAndType(ConstString("start"),
                                                  eSymbolTypeCode, contexts)) {
          if (contexts.GetContextAtIndex(0, context))
            m_entry_point_address = context.symbol->GetAddress();
        }
      }
    }
  }

  return m_entry_point_address;
}

lldb_private::Address ObjectFileMachO::GetHeaderAddress() {
  lldb_private::Address header_addr;
  SectionList *section_list = GetSectionList();
  if (section_list) {
    SectionSP text_segment_sp(
        section_list->FindSectionByName(GetSegmentNameTEXT()));
    if (text_segment_sp) {
      header_addr.SetSection(text_segment_sp);
      header_addr.SetOffset(0);
    }
  }
  return header_addr;
}

uint32_t ObjectFileMachO::GetNumThreadContexts() {
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    if (!m_thread_context_offsets_valid) {
      m_thread_context_offsets_valid = true;
      lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
      FileRangeArray::Entry file_range;
      thread_command thread_cmd;
      for (uint32_t i = 0; i < m_header.ncmds; ++i) {
        const uint32_t cmd_offset = offset;
        if (m_data.GetU32(&offset, &thread_cmd, 2) == NULL)
          break;

        if (thread_cmd.cmd == LC_THREAD) {
          file_range.SetRangeBase(offset);
          file_range.SetByteSize(thread_cmd.cmdsize - 8);
          m_thread_context_offsets.Append(file_range);
        }
        offset = cmd_offset + thread_cmd.cmdsize;
      }
    }
  }
  return m_thread_context_offsets.GetSize();
}

std::string ObjectFileMachO::GetIdentifierString() {
  std::string result;
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());

    // First, look over the load commands for an LC_NOTE load command with
    // data_owner string "kern ver str" & use that if found.
    lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
    for (uint32_t i = 0; i < m_header.ncmds; ++i) {
      const uint32_t cmd_offset = offset;
      load_command lc;
      if (m_data.GetU32(&offset, &lc.cmd, 2) == NULL)
          break;
      if (lc.cmd == LC_NOTE)
      {
          char data_owner[17];
          m_data.CopyData (offset, 16, data_owner);
          data_owner[16] = '\0';
          offset += 16;
          uint64_t fileoff = m_data.GetU64_unchecked (&offset);
          uint64_t size = m_data.GetU64_unchecked (&offset);

          // "kern ver str" has a uint32_t version and then a nul terminated
          // c-string.
          if (strcmp ("kern ver str", data_owner) == 0)
          {
              offset = fileoff;
              uint32_t version;
              if (m_data.GetU32 (&offset, &version, 1) != nullptr)
              {
                  if (version == 1)
                  {
                      uint32_t strsize = size - sizeof (uint32_t);
                      char *buf = (char*) malloc (strsize);
                      if (buf)
                      {
                          m_data.CopyData (offset, strsize, buf);
                          buf[strsize - 1] = '\0';
                          result = buf;
                          if (buf)
                              free (buf);
                          return result;
                      }
                  }
              }
          }
      }
      offset = cmd_offset + lc.cmdsize;
    }

    // Second, make a pass over the load commands looking for an obsolete
    // LC_IDENT load command.
    offset = MachHeaderSizeFromMagic(m_header.magic);
    for (uint32_t i = 0; i < m_header.ncmds; ++i) {
      const uint32_t cmd_offset = offset;
      struct ident_command ident_command;
      if (m_data.GetU32(&offset, &ident_command, 2) == NULL)
        break;
      if (ident_command.cmd == LC_IDENT && ident_command.cmdsize != 0) {
        char *buf = (char *) malloc (ident_command.cmdsize);
        if (buf != nullptr 
            && m_data.CopyData (offset, ident_command.cmdsize, buf) == ident_command.cmdsize) {
          buf[ident_command.cmdsize - 1] = '\0';
          result = buf;
        }
        if (buf)
          free (buf);
      }
      offset = cmd_offset + ident_command.cmdsize;
    }

  }
  return result;
}

bool ObjectFileMachO::GetCorefileMainBinaryInfo (addr_t &address, UUID &uuid) {
  address = LLDB_INVALID_ADDRESS;
  uuid.Clear();
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
    for (uint32_t i = 0; i < m_header.ncmds; ++i) {
      const uint32_t cmd_offset = offset;
      load_command lc;
      if (m_data.GetU32(&offset, &lc.cmd, 2) == NULL)
          break;
      if (lc.cmd == LC_NOTE)
      {
          char data_owner[17];
          memset (data_owner, 0, sizeof (data_owner));
          m_data.CopyData (offset, 16, data_owner);
          offset += 16;
          uint64_t fileoff = m_data.GetU64_unchecked (&offset);
          uint64_t size = m_data.GetU64_unchecked (&offset);

          // "main bin spec" (main binary specification) data payload is
          // formatted:
          //    uint32_t version       [currently 1]
          //    uint32_t type          [0 == unspecified, 1 == kernel, 2 == user process]
          //    uint64_t address       [ UINT64_MAX if address not specified ]
          //    uuid_t   uuid          [ all zero's if uuid not specified ]
          //    uint32_t log2_pagesize [ process page size in log base 2, e.g. 4k pages are 12.  0 for unspecified ]

          if (strcmp ("main bin spec", data_owner) == 0 && size >= 32)
          {
              offset = fileoff;
              uint32_t version;
              if (m_data.GetU32 (&offset, &version, 1) != nullptr && version == 1)
              {
                  uint32_t type = 0;
                  uuid_t raw_uuid;
                  memset (raw_uuid, 0, sizeof (uuid_t));

                  if (m_data.GetU32 (&offset, &type, 1)
                      && m_data.GetU64 (&offset, &address, 1)
                      && m_data.CopyData (offset, sizeof (uuid_t), raw_uuid) != 0
                      && uuid.SetBytes (raw_uuid, sizeof (uuid_t)))
                  {
                      return true;
                  }
              }
          }
      }
      offset = cmd_offset + lc.cmdsize;
    }
  }
  return false;
}

lldb::RegisterContextSP
ObjectFileMachO::GetThreadContextAtIndex(uint32_t idx,
                                         lldb_private::Thread &thread) {
  lldb::RegisterContextSP reg_ctx_sp;

  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    if (!m_thread_context_offsets_valid)
      GetNumThreadContexts();

    const FileRangeArray::Entry *thread_context_file_range =
        m_thread_context_offsets.GetEntryAtIndex(idx);
    if (thread_context_file_range) {

      DataExtractor data(m_data, thread_context_file_range->GetRangeBase(),
                         thread_context_file_range->GetByteSize());

      switch (m_header.cputype) {
      case llvm::MachO::CPU_TYPE_ARM64:
        reg_ctx_sp.reset(new RegisterContextDarwin_arm64_Mach(thread, data));
        break;

      case llvm::MachO::CPU_TYPE_ARM:
        reg_ctx_sp.reset(new RegisterContextDarwin_arm_Mach(thread, data));
        break;

      case llvm::MachO::CPU_TYPE_I386:
        reg_ctx_sp.reset(new RegisterContextDarwin_i386_Mach(thread, data));
        break;

      case llvm::MachO::CPU_TYPE_X86_64:
        reg_ctx_sp.reset(new RegisterContextDarwin_x86_64_Mach(thread, data));
        break;
      }
    }
  }
  return reg_ctx_sp;
}

ObjectFile::Type ObjectFileMachO::CalculateType() {
  switch (m_header.filetype) {
  case MH_OBJECT: // 0x1u
    if (GetAddressByteSize() == 4) {
      // 32 bit kexts are just object files, but they do have a valid
      // UUID load command.
      UUID uuid;
      if (GetUUID(&uuid)) {
        // this checking for the UUID load command is not enough we could
        // eventually look for the symbol named "OSKextGetCurrentIdentifier" as
        // this is required of kexts
        if (m_strata == eStrataInvalid)
          m_strata = eStrataKernel;
        return eTypeSharedLibrary;
      }
    }
    return eTypeObjectFile;

  case MH_EXECUTE:
    return eTypeExecutable; // 0x2u
  case MH_FVMLIB:
    return eTypeSharedLibrary; // 0x3u
  case MH_CORE:
    return eTypeCoreFile; // 0x4u
  case MH_PRELOAD:
    return eTypeSharedLibrary; // 0x5u
  case MH_DYLIB:
    return eTypeSharedLibrary; // 0x6u
  case MH_DYLINKER:
    return eTypeDynamicLinker; // 0x7u
  case MH_BUNDLE:
    return eTypeSharedLibrary; // 0x8u
  case MH_DYLIB_STUB:
    return eTypeStubLibrary; // 0x9u
  case MH_DSYM:
    return eTypeDebugInfo; // 0xAu
  case MH_KEXT_BUNDLE:
    return eTypeSharedLibrary; // 0xBu
  default:
    break;
  }
  return eTypeUnknown;
}

ObjectFile::Strata ObjectFileMachO::CalculateStrata() {
  switch (m_header.filetype) {
  case MH_OBJECT: // 0x1u
  {
    // 32 bit kexts are just object files, but they do have a valid
    // UUID load command.
    UUID uuid;
    if (GetUUID(&uuid)) {
      // this checking for the UUID load command is not enough we could
      // eventually look for the symbol named "OSKextGetCurrentIdentifier" as
      // this is required of kexts
      if (m_type == eTypeInvalid)
        m_type = eTypeSharedLibrary;

      return eStrataKernel;
    }
  }
    return eStrataUnknown;

  case MH_EXECUTE: // 0x2u
    // Check for the MH_DYLDLINK bit in the flags
    if (m_header.flags & MH_DYLDLINK) {
      return eStrataUser;
    } else {
      SectionList *section_list = GetSectionList();
      if (section_list) {
        static ConstString g_kld_section_name("__KLD");
        if (section_list->FindSectionByName(g_kld_section_name))
          return eStrataKernel;
      }
    }
    return eStrataRawImage;

  case MH_FVMLIB:
    return eStrataUser; // 0x3u
  case MH_CORE:
    return eStrataUnknown; // 0x4u
  case MH_PRELOAD:
    return eStrataRawImage; // 0x5u
  case MH_DYLIB:
    return eStrataUser; // 0x6u
  case MH_DYLINKER:
    return eStrataUser; // 0x7u
  case MH_BUNDLE:
    return eStrataUser; // 0x8u
  case MH_DYLIB_STUB:
    return eStrataUser; // 0x9u
  case MH_DSYM:
    return eStrataUnknown; // 0xAu
  case MH_KEXT_BUNDLE:
    return eStrataKernel; // 0xBu
  default:
    break;
  }
  return eStrataUnknown;
}

uint32_t ObjectFileMachO::GetVersion(uint32_t *versions,
                                     uint32_t num_versions) {
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    struct dylib_command load_cmd;
    lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
    uint32_t version_cmd = 0;
    uint64_t version = 0;
    uint32_t i;
    for (i = 0; i < m_header.ncmds; ++i) {
      const lldb::offset_t cmd_offset = offset;
      if (m_data.GetU32(&offset, &load_cmd, 2) == NULL)
        break;

      if (load_cmd.cmd == LC_ID_DYLIB) {
        if (version_cmd == 0) {
          version_cmd = load_cmd.cmd;
          if (m_data.GetU32(&offset, &load_cmd.dylib, 4) == NULL)
            break;
          version = load_cmd.dylib.current_version;
        }
        break; // Break for now unless there is another more complete version
               // number load command in the future.
      }
      offset = cmd_offset + load_cmd.cmdsize;
    }

    if (version_cmd == LC_ID_DYLIB) {
      if (versions != NULL && num_versions > 0) {
        if (num_versions > 0)
          versions[0] = (version & 0xFFFF0000ull) >> 16;
        if (num_versions > 1)
          versions[1] = (version & 0x0000FF00ull) >> 8;
        if (num_versions > 2)
          versions[2] = (version & 0x000000FFull);
        // Fill in an remaining version numbers with invalid values
        for (i = 3; i < num_versions; ++i)
          versions[i] = UINT32_MAX;
      }
      // The LC_ID_DYLIB load command has a version with 3 version numbers in
      // it, so always return 3
      return 3;
    }
  }
  return false;
}

bool ObjectFileMachO::GetArchitecture(ArchSpec &arch) {
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    return GetArchitecture(m_header, m_data,
                           MachHeaderSizeFromMagic(m_header.magic), arch);
  }
  return false;
}

void ObjectFileMachO::GetProcessSharedCacheUUID(Process *process, addr_t &base_addr, UUID &uuid) {
  uuid.Clear();
  base_addr = LLDB_INVALID_ADDRESS;
  if (process && process->GetDynamicLoader()) {
    DynamicLoader *dl = process->GetDynamicLoader();
    LazyBool using_shared_cache;
    LazyBool private_shared_cache;
    dl->GetSharedCacheInformation(base_addr, uuid, using_shared_cache,
                                  private_shared_cache);
  }
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_SYMBOLS | LIBLLDB_LOG_PROCESS));
  if (log)
    log->Printf("inferior process shared cache has a UUID of %s, base address 0x%" PRIx64 , uuid.GetAsString().c_str(), base_addr);
}

// From dyld SPI header dyld_process_info.h
typedef void *dyld_process_info;
struct lldb_copy__dyld_process_cache_info {
  uuid_t cacheUUID;          // UUID of cache used by process
  uint64_t cacheBaseAddress; // load address of dyld shared cache
  bool noCache;              // process is running without a dyld cache
  bool privateCache; // process is using a private copy of its dyld cache
};

// #including mach/mach.h pulls in machine.h & CPU_TYPE_ARM etc conflicts with llvm
// enum definitions llvm::MachO::CPU_TYPE_ARM turning them into compile errors.
// So we need to use the actual underlying types of task_t and kern_return_t
// below.
extern "C" unsigned int /*task_t*/ mach_task_self(); 

void ObjectFileMachO::GetLLDBSharedCacheUUID(addr_t &base_addr, UUID &uuid) {
  uuid.Clear();
  base_addr = LLDB_INVALID_ADDRESS;

#if defined(__APPLE__) &&                                                      \
    (defined(__arm__) || defined(__arm64__) || defined(__aarch64__))
  uint8_t *(*dyld_get_all_image_infos)(void);
  dyld_get_all_image_infos =
      (uint8_t * (*)())dlsym(RTLD_DEFAULT, "_dyld_get_all_image_infos");
  if (dyld_get_all_image_infos) {
    uint8_t *dyld_all_image_infos_address = dyld_get_all_image_infos();
    if (dyld_all_image_infos_address) {
      uint32_t *version = (uint32_t *)
          dyld_all_image_infos_address; // version <mach-o/dyld_images.h>
      if (*version >= 13) {
        uuid_t *sharedCacheUUID_address = 0;
        int wordsize = sizeof(uint8_t *);
        if (wordsize == 8) {
          sharedCacheUUID_address =
              (uuid_t *)((uint8_t *)dyld_all_image_infos_address +
                         160); // sharedCacheUUID <mach-o/dyld_images.h>
          if (*version >= 15)
            base_addr = *(uint64_t *) ((uint8_t *) dyld_all_image_infos_address 
                          + 176); // sharedCacheBaseAddress <mach-o/dyld_images.h>
        } else {
          sharedCacheUUID_address =
              (uuid_t *)((uint8_t *)dyld_all_image_infos_address +
                         84); // sharedCacheUUID <mach-o/dyld_images.h>
          if (*version >= 15) {
            base_addr = 0;
            base_addr = *(uint32_t *) ((uint8_t *) dyld_all_image_infos_address 
                          + 100); // sharedCacheBaseAddress <mach-o/dyld_images.h>
          }
        }
        uuid.SetBytes(sharedCacheUUID_address);
      }
    }
  } else {
    // Exists in macOS 10.12 and later, iOS 10.0 and later - dyld SPI
    dyld_process_info (*dyld_process_info_create)(unsigned int /* task_t */ task, uint64_t timestamp, unsigned int /*kern_return_t*/ *kernelError);
    void (*dyld_process_info_get_cache)(void *info, void *cacheInfo);
    void (*dyld_process_info_release)(dyld_process_info info);

    dyld_process_info_create = (void *(*)(unsigned int /* task_t */, uint64_t, unsigned int /*kern_return_t*/ *))
               dlsym (RTLD_DEFAULT, "_dyld_process_info_create");
    dyld_process_info_get_cache = (void (*)(void *, void *))
               dlsym (RTLD_DEFAULT, "_dyld_process_info_get_cache");
    dyld_process_info_release = (void (*)(void *))
               dlsym (RTLD_DEFAULT, "_dyld_process_info_release");

    if (dyld_process_info_create && dyld_process_info_get_cache) {
      unsigned int /*kern_return_t */ kern_ret;
		  dyld_process_info process_info = dyld_process_info_create(::mach_task_self(), 0, &kern_ret);
      if (process_info) {
        struct lldb_copy__dyld_process_cache_info sc_info;
        memset (&sc_info, 0, sizeof (struct lldb_copy__dyld_process_cache_info));
        dyld_process_info_get_cache (process_info, &sc_info);
        if (sc_info.cacheBaseAddress != 0) {
          base_addr = sc_info.cacheBaseAddress;
          uuid.SetBytes (sc_info.cacheUUID);
        }
        dyld_process_info_release (process_info);
      }
    }
  }
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_SYMBOLS | LIBLLDB_LOG_PROCESS));
  if (log && uuid.IsValid())
    log->Printf("lldb's in-memory shared cache has a UUID of %s base address of 0x%" PRIx64, uuid.GetAsString().c_str(), base_addr);
#endif
}

uint32_t ObjectFileMachO::GetMinimumOSVersion(uint32_t *versions,
                                              uint32_t num_versions) {
  if (m_min_os_versions.empty()) {
    lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
    bool success = false;
    for (uint32_t i = 0; success == false && i < m_header.ncmds; ++i) {
      const lldb::offset_t load_cmd_offset = offset;

      version_min_command lc;
      if (m_data.GetU32(&offset, &lc.cmd, 2) == NULL)
        break;
      if (lc.cmd == llvm::MachO::LC_VERSION_MIN_MACOSX ||
          lc.cmd == llvm::MachO::LC_VERSION_MIN_IPHONEOS ||
          lc.cmd == llvm::MachO::LC_VERSION_MIN_TVOS ||
          lc.cmd == llvm::MachO::LC_VERSION_MIN_WATCHOS) {
        if (m_data.GetU32(&offset, &lc.version,
                          (sizeof(lc) / sizeof(uint32_t)) - 2)) {
          const uint32_t xxxx = lc.version >> 16;
          const uint32_t yy = (lc.version >> 8) & 0xffu;
          const uint32_t zz = lc.version & 0xffu;
          if (xxxx) {
            m_min_os_versions.push_back(xxxx);
            m_min_os_versions.push_back(yy);
            m_min_os_versions.push_back(zz);
          }
          success = true;
        }
      }
      offset = load_cmd_offset + lc.cmdsize;
    }

    if (success == false) {
      // Push an invalid value so we don't keep trying to
      m_min_os_versions.push_back(UINT32_MAX);
    }
  }

  if (m_min_os_versions.size() > 1 || m_min_os_versions[0] != UINT32_MAX) {
    if (versions != NULL && num_versions > 0) {
      for (size_t i = 0; i < num_versions; ++i) {
        if (i < m_min_os_versions.size())
          versions[i] = m_min_os_versions[i];
        else
          versions[i] = 0;
      }
    }
    return m_min_os_versions.size();
  }
  // Call the superclasses version that will empty out the data
  return ObjectFile::GetMinimumOSVersion(versions, num_versions);
}

uint32_t ObjectFileMachO::GetSDKVersion(uint32_t *versions,
                                        uint32_t num_versions) {
  if (m_sdk_versions.empty()) {
    lldb::offset_t offset = MachHeaderSizeFromMagic(m_header.magic);
    bool success = false;
    for (uint32_t i = 0; success == false && i < m_header.ncmds; ++i) {
      const lldb::offset_t load_cmd_offset = offset;

      version_min_command lc;
      if (m_data.GetU32(&offset, &lc.cmd, 2) == NULL)
        break;
      if (lc.cmd == llvm::MachO::LC_VERSION_MIN_MACOSX ||
          lc.cmd == llvm::MachO::LC_VERSION_MIN_IPHONEOS ||
          lc.cmd == llvm::MachO::LC_VERSION_MIN_TVOS ||
          lc.cmd == llvm::MachO::LC_VERSION_MIN_WATCHOS) {
        if (m_data.GetU32(&offset, &lc.version,
                          (sizeof(lc) / sizeof(uint32_t)) - 2)) {
          const uint32_t xxxx = lc.sdk >> 16;
          const uint32_t yy = (lc.sdk >> 8) & 0xffu;
          const uint32_t zz = lc.sdk & 0xffu;
          if (xxxx) {
            m_sdk_versions.push_back(xxxx);
            m_sdk_versions.push_back(yy);
            m_sdk_versions.push_back(zz);
          }
          success = true;
        }
      }
      offset = load_cmd_offset + lc.cmdsize;
    }

    if (success == false) {
      // Push an invalid value so we don't keep trying to
      m_sdk_versions.push_back(UINT32_MAX);
    }
  }

  if (m_sdk_versions.size() > 1 || m_sdk_versions[0] != UINT32_MAX) {
    if (versions != NULL && num_versions > 0) {
      for (size_t i = 0; i < num_versions; ++i) {
        if (i < m_sdk_versions.size())
          versions[i] = m_sdk_versions[i];
        else
          versions[i] = 0;
      }
    }
    return m_sdk_versions.size();
  }
  // Call the superclasses version that will empty out the data
  return ObjectFile::GetSDKVersion(versions, num_versions);
}

bool ObjectFileMachO::GetIsDynamicLinkEditor() {
  return m_header.filetype == llvm::MachO::MH_DYLINKER;
}

bool ObjectFileMachO::AllowAssemblyEmulationUnwindPlans() {
  return m_allow_assembly_emulation_unwind_plans;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString ObjectFileMachO::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ObjectFileMachO::GetPluginVersion() { return 1; }

Section *ObjectFileMachO::GetMachHeaderSection() {
  // Find the first address of the mach header which is the first non-zero file
  // sized section whose file offset is zero. This is the base file address of
  // the mach-o file which can be subtracted from the vmaddr of the other
  // segments found in memory and added to the load address
  ModuleSP module_sp = GetModule();
  if (module_sp) {
    SectionList *section_list = GetSectionList();
    if (section_list) {
      lldb::addr_t mach_base_file_addr = LLDB_INVALID_ADDRESS;
      const size_t num_sections = section_list->GetSize();

      for (size_t sect_idx = 0; sect_idx < num_sections &&
                                mach_base_file_addr == LLDB_INVALID_ADDRESS;
           ++sect_idx) {
        Section *section = section_list->GetSectionAtIndex(sect_idx).get();
        if (section && section->GetFileSize() > 0 &&
            section->GetFileOffset() == 0 &&
            section->IsThreadSpecific() == false &&
            module_sp.get() == section->GetModule().get()) {
          return section;
        }
      }
    }
  }
  return nullptr;
}

lldb::addr_t ObjectFileMachO::CalculateSectionLoadAddressForMemoryImage(
    lldb::addr_t mach_header_load_address, const Section *mach_header_section,
    const Section *section) {
  ModuleSP module_sp = GetModule();
  if (module_sp && mach_header_section && section &&
      mach_header_load_address != LLDB_INVALID_ADDRESS) {
    lldb::addr_t mach_header_file_addr = mach_header_section->GetFileAddress();
    if (mach_header_file_addr != LLDB_INVALID_ADDRESS) {
      if (section && section->GetFileSize() > 0 &&
          section->IsThreadSpecific() == false &&
          module_sp.get() == section->GetModule().get()) {
        // Ignore __LINKEDIT and __DWARF segments
        if (section->GetName() == GetSegmentNameLINKEDIT()) {
          // Only map __LINKEDIT if we have an in memory image and this isn't a
          // kernel binary like a kext or mach_kernel.
          const bool is_memory_image = (bool)m_process_wp.lock();
          const Strata strata = GetStrata();
          if (is_memory_image == false || strata == eStrataKernel)
            return LLDB_INVALID_ADDRESS;
        }
        return section->GetFileAddress() - mach_header_file_addr +
               mach_header_load_address;
      }
    }
  }
  return LLDB_INVALID_ADDRESS;
}

bool ObjectFileMachO::SetLoadAddress(Target &target, lldb::addr_t value,
                                     bool value_is_offset) {
  ModuleSP module_sp = GetModule();
  if (module_sp) {
    size_t num_loaded_sections = 0;
    SectionList *section_list = GetSectionList();
    if (section_list) {
      const size_t num_sections = section_list->GetSize();

      if (value_is_offset) {
        // "value" is an offset to apply to each top level segment
        for (size_t sect_idx = 0; sect_idx < num_sections; ++sect_idx) {
          // Iterate through the object file sections to find all of the
          // sections that size on disk (to avoid __PAGEZERO) and load them
          SectionSP section_sp(section_list->GetSectionAtIndex(sect_idx));
          if (section_sp && section_sp->GetFileSize() > 0 &&
              section_sp->IsThreadSpecific() == false &&
              module_sp.get() == section_sp->GetModule().get()) {
            // Ignore __LINKEDIT and __DWARF segments
            if (section_sp->GetName() == GetSegmentNameLINKEDIT()) {
              // Only map __LINKEDIT if we have an in memory image and this
              // isn't a kernel binary like a kext or mach_kernel.
              const bool is_memory_image = (bool)m_process_wp.lock();
              const Strata strata = GetStrata();
              if (is_memory_image == false || strata == eStrataKernel)
                continue;
            }
            if (target.GetSectionLoadList().SetSectionLoadAddress(
                    section_sp, section_sp->GetFileAddress() + value))
              ++num_loaded_sections;
          }
        }
      } else {
        // "value" is the new base address of the mach_header, adjust each
        // section accordingly

        Section *mach_header_section = GetMachHeaderSection();
        if (mach_header_section) {
          for (size_t sect_idx = 0; sect_idx < num_sections; ++sect_idx) {
            SectionSP section_sp(section_list->GetSectionAtIndex(sect_idx));

            lldb::addr_t section_load_addr =
                CalculateSectionLoadAddressForMemoryImage(
                    value, mach_header_section, section_sp.get());
            if (section_load_addr != LLDB_INVALID_ADDRESS) {
              if (target.GetSectionLoadList().SetSectionLoadAddress(
                      section_sp, section_load_addr))
                ++num_loaded_sections;
            }
          }
        }
      }
    }
    return num_loaded_sections > 0;
  }
  return false;
}

bool ObjectFileMachO::SaveCore(const lldb::ProcessSP &process_sp,
                               const FileSpec &outfile, Status &error) {
  if (process_sp) {
    Target &target = process_sp->GetTarget();
    const ArchSpec target_arch = target.GetArchitecture();
    const llvm::Triple &target_triple = target_arch.GetTriple();
    if (target_triple.getVendor() == llvm::Triple::Apple &&
        (target_triple.getOS() == llvm::Triple::MacOSX ||
         target_triple.getOS() == llvm::Triple::IOS ||
         target_triple.getOS() == llvm::Triple::WatchOS ||
         target_triple.getOS() == llvm::Triple::TvOS)) {
      bool make_core = false;
      switch (target_arch.GetMachine()) {
      case llvm::Triple::aarch64:
      case llvm::Triple::arm:
      case llvm::Triple::thumb:
      case llvm::Triple::x86:
      case llvm::Triple::x86_64:
        make_core = true;
        break;
      default:
        error.SetErrorStringWithFormat("unsupported core architecture: %s",
                                       target_triple.str().c_str());
        break;
      }

      if (make_core) {
        std::vector<segment_command_64> segment_load_commands;
        //                uint32_t range_info_idx = 0;
        MemoryRegionInfo range_info;
        Status range_error = process_sp->GetMemoryRegionInfo(0, range_info);
        const uint32_t addr_byte_size = target_arch.GetAddressByteSize();
        const ByteOrder byte_order = target_arch.GetByteOrder();
        if (range_error.Success()) {
          while (range_info.GetRange().GetRangeBase() != LLDB_INVALID_ADDRESS) {
            const addr_t addr = range_info.GetRange().GetRangeBase();
            const addr_t size = range_info.GetRange().GetByteSize();

            if (size == 0)
              break;

            // Calculate correct protections
            uint32_t prot = 0;
            if (range_info.GetReadable() == MemoryRegionInfo::eYes)
              prot |= VM_PROT_READ;
            if (range_info.GetWritable() == MemoryRegionInfo::eYes)
              prot |= VM_PROT_WRITE;
            if (range_info.GetExecutable() == MemoryRegionInfo::eYes)
              prot |= VM_PROT_EXECUTE;

            //                        printf ("[%3u] [0x%16.16" PRIx64 " -
            //                        0x%16.16" PRIx64 ") %c%c%c\n",
            //                                range_info_idx,
            //                                addr,
            //                                size,
            //                                (prot & VM_PROT_READ   ) ? 'r' :
            //                                '-',
            //                                (prot & VM_PROT_WRITE  ) ? 'w' :
            //                                '-',
            //                                (prot & VM_PROT_EXECUTE) ? 'x' :
            //                                '-');

            if (prot != 0) {
              uint32_t cmd_type = LC_SEGMENT_64;
              uint32_t segment_size = sizeof(segment_command_64);
              if (addr_byte_size == 4) {
                cmd_type = LC_SEGMENT;
                segment_size = sizeof(segment_command);
              }
              segment_command_64 segment = {
                  cmd_type,     // uint32_t cmd;
                  segment_size, // uint32_t cmdsize;
                  {0},          // char segname[16];
                  addr, // uint64_t vmaddr;    // uint32_t for 32-bit Mach-O
                  size, // uint64_t vmsize;    // uint32_t for 32-bit Mach-O
                  0,    // uint64_t fileoff;   // uint32_t for 32-bit Mach-O
                  size, // uint64_t filesize;  // uint32_t for 32-bit Mach-O
                  prot, // uint32_t maxprot;
                  prot, // uint32_t initprot;
                  0,    // uint32_t nsects;
                  0};   // uint32_t flags;
              segment_load_commands.push_back(segment);
            } else {
              // No protections and a size of 1 used to be returned from old
              // debugservers when we asked about a region that was past the
              // last memory region and it indicates the end...
              if (size == 1)
                break;
            }

            range_error = process_sp->GetMemoryRegionInfo(
                range_info.GetRange().GetRangeEnd(), range_info);
            if (range_error.Fail())
              break;
          }

          StreamString buffer(Stream::eBinary, addr_byte_size, byte_order);

          mach_header_64 mach_header;
          if (addr_byte_size == 8) {
            mach_header.magic = MH_MAGIC_64;
          } else {
            mach_header.magic = MH_MAGIC;
          }
          mach_header.cputype = target_arch.GetMachOCPUType();
          mach_header.cpusubtype = target_arch.GetMachOCPUSubType();
          mach_header.filetype = MH_CORE;
          mach_header.ncmds = segment_load_commands.size();
          mach_header.flags = 0;
          mach_header.reserved = 0;
          ThreadList &thread_list = process_sp->GetThreadList();
          const uint32_t num_threads = thread_list.GetSize();

          // Make an array of LC_THREAD data items. Each one contains the
          // contents of the LC_THREAD load command. The data doesn't contain
          // the load command + load command size, we will add the load command
          // and load command size as we emit the data.
          std::vector<StreamString> LC_THREAD_datas(num_threads);
          for (auto &LC_THREAD_data : LC_THREAD_datas) {
            LC_THREAD_data.GetFlags().Set(Stream::eBinary);
            LC_THREAD_data.SetAddressByteSize(addr_byte_size);
            LC_THREAD_data.SetByteOrder(byte_order);
          }
          for (uint32_t thread_idx = 0; thread_idx < num_threads;
               ++thread_idx) {
            ThreadSP thread_sp(thread_list.GetThreadAtIndex(thread_idx));
            if (thread_sp) {
              switch (mach_header.cputype) {
              case llvm::MachO::CPU_TYPE_ARM64:
                RegisterContextDarwin_arm64_Mach::Create_LC_THREAD(
                    thread_sp.get(), LC_THREAD_datas[thread_idx]);
                break;

              case llvm::MachO::CPU_TYPE_ARM:
                RegisterContextDarwin_arm_Mach::Create_LC_THREAD(
                    thread_sp.get(), LC_THREAD_datas[thread_idx]);
                break;

              case llvm::MachO::CPU_TYPE_I386:
                RegisterContextDarwin_i386_Mach::Create_LC_THREAD(
                    thread_sp.get(), LC_THREAD_datas[thread_idx]);
                break;

              case llvm::MachO::CPU_TYPE_X86_64:
                RegisterContextDarwin_x86_64_Mach::Create_LC_THREAD(
                    thread_sp.get(), LC_THREAD_datas[thread_idx]);
                break;
              }
            }
          }

          // The size of the load command is the size of the segments...
          if (addr_byte_size == 8) {
            mach_header.sizeofcmds = segment_load_commands.size() *
                                     sizeof(struct segment_command_64);
          } else {
            mach_header.sizeofcmds =
                segment_load_commands.size() * sizeof(struct segment_command);
          }

          // and the size of all LC_THREAD load command
          for (const auto &LC_THREAD_data : LC_THREAD_datas) {
            ++mach_header.ncmds;
            mach_header.sizeofcmds += 8 + LC_THREAD_data.GetSize();
          }

          printf("mach_header: 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x "
                 "0x%8.8x 0x%8.8x\n",
                 mach_header.magic, mach_header.cputype, mach_header.cpusubtype,
                 mach_header.filetype, mach_header.ncmds,
                 mach_header.sizeofcmds, mach_header.flags,
                 mach_header.reserved);

          // Write the mach header
          buffer.PutHex32(mach_header.magic);
          buffer.PutHex32(mach_header.cputype);
          buffer.PutHex32(mach_header.cpusubtype);
          buffer.PutHex32(mach_header.filetype);
          buffer.PutHex32(mach_header.ncmds);
          buffer.PutHex32(mach_header.sizeofcmds);
          buffer.PutHex32(mach_header.flags);
          if (addr_byte_size == 8) {
            buffer.PutHex32(mach_header.reserved);
          }

          // Skip the mach header and all load commands and align to the next
          // 0x1000 byte boundary
          addr_t file_offset = buffer.GetSize() + mach_header.sizeofcmds;
          if (file_offset & 0x00000fff) {
            file_offset += 0x00001000ull;
            file_offset &= (~0x00001000ull + 1);
          }

          for (auto &segment : segment_load_commands) {
            segment.fileoff = file_offset;
            file_offset += segment.filesize;
          }

          // Write out all of the LC_THREAD load commands
          for (const auto &LC_THREAD_data : LC_THREAD_datas) {
            const size_t LC_THREAD_data_size = LC_THREAD_data.GetSize();
            buffer.PutHex32(LC_THREAD);
            buffer.PutHex32(8 + LC_THREAD_data_size); // cmd + cmdsize + data
            buffer.Write(LC_THREAD_data.GetString().data(),
                         LC_THREAD_data_size);
          }

          // Write out all of the segment load commands
          for (const auto &segment : segment_load_commands) {
            printf("0x%8.8x 0x%8.8x [0x%16.16" PRIx64 " - 0x%16.16" PRIx64
                   ") [0x%16.16" PRIx64 " 0x%16.16" PRIx64
                   ") 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x]\n",
                   segment.cmd, segment.cmdsize, segment.vmaddr,
                   segment.vmaddr + segment.vmsize, segment.fileoff,
                   segment.filesize, segment.maxprot, segment.initprot,
                   segment.nsects, segment.flags);

            buffer.PutHex32(segment.cmd);
            buffer.PutHex32(segment.cmdsize);
            buffer.PutRawBytes(segment.segname, sizeof(segment.segname));
            if (addr_byte_size == 8) {
              buffer.PutHex64(segment.vmaddr);
              buffer.PutHex64(segment.vmsize);
              buffer.PutHex64(segment.fileoff);
              buffer.PutHex64(segment.filesize);
            } else {
              buffer.PutHex32(static_cast<uint32_t>(segment.vmaddr));
              buffer.PutHex32(static_cast<uint32_t>(segment.vmsize));
              buffer.PutHex32(static_cast<uint32_t>(segment.fileoff));
              buffer.PutHex32(static_cast<uint32_t>(segment.filesize));
            }
            buffer.PutHex32(segment.maxprot);
            buffer.PutHex32(segment.initprot);
            buffer.PutHex32(segment.nsects);
            buffer.PutHex32(segment.flags);
          }

          File core_file;
          std::string core_file_path(outfile.GetPath());
          error = core_file.Open(core_file_path.c_str(),
                                 File::eOpenOptionWrite |
                                     File::eOpenOptionTruncate |
                                     File::eOpenOptionCanCreate);
          if (error.Success()) {
            // Read 1 page at a time
            uint8_t bytes[0x1000];
            // Write the mach header and load commands out to the core file
            size_t bytes_written = buffer.GetString().size();
            error = core_file.Write(buffer.GetString().data(), bytes_written);
            if (error.Success()) {
              // Now write the file data for all memory segments in the process
              for (const auto &segment : segment_load_commands) {
                if (core_file.SeekFromStart(segment.fileoff) == -1) {
                  error.SetErrorStringWithFormat(
                      "unable to seek to offset 0x%" PRIx64 " in '%s'",
                      segment.fileoff, core_file_path.c_str());
                  break;
                }

                printf("Saving %" PRId64
                       " bytes of data for memory region at 0x%" PRIx64 "\n",
                       segment.vmsize, segment.vmaddr);
                addr_t bytes_left = segment.vmsize;
                addr_t addr = segment.vmaddr;
                Status memory_read_error;
                while (bytes_left > 0 && error.Success()) {
                  const size_t bytes_to_read =
                      bytes_left > sizeof(bytes) ? sizeof(bytes) : bytes_left;
                  const size_t bytes_read = process_sp->ReadMemory(
                      addr, bytes, bytes_to_read, memory_read_error);
                  if (bytes_read == bytes_to_read) {
                    size_t bytes_written = bytes_read;
                    error = core_file.Write(bytes, bytes_written);
                    bytes_left -= bytes_read;
                    addr += bytes_read;
                  } else {
                    // Some pages within regions are not readable, those should
                    // be zero filled
                    memset(bytes, 0, bytes_to_read);
                    size_t bytes_written = bytes_to_read;
                    error = core_file.Write(bytes, bytes_written);
                    bytes_left -= bytes_to_read;
                    addr += bytes_to_read;
                  }
                }
              }
            }
          }
        } else {
          error.SetErrorString(
              "process doesn't support getting memory region info");
        }
      }
      return true; // This is the right plug to handle saving core files for
                   // this process
    }
  }
  return false;
}
