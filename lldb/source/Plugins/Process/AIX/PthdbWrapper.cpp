//===-- PthdbWrapper.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements the libpthdebug callback table (pthdb_callbacks_t) and the
/// EnsurePthdbSession() helper called by NativeProcessAIX.
///
/// The nine static callback functions receive a NativeProcessAIX* packed into
/// the opaque pthdb_user_t handle.  They use only the public interface of
/// NativeProcessAIX (ReadMemory, WriteMemory, PtraceWrapper, the symbol-cache
/// accessors, and GetDelegate) so this file has no access to private members.
///
//===----------------------------------------------------------------------===//

#include "PthdbWrapper.h"
#include "NativeProcessAIX.h"

#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/StringRef.h"

// Pull in GPR_PPC64 / FPR_PPC64 / VMX_PPC64 / VSX_PPC64 register structs
// needed by cb_read_regs / cb_write_regs and PtraceWrapper.
#define DECLARE_REGISTER_INFOS_PPC64_STRUCT
#include "Plugins/Process/Utility/RegisterInfos_ppc64.h"
#undef DECLARE_REGISTER_INFOS_PPC64_STRUCT

using namespace lldb_private;
using namespace lldb_private::process_aix;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// File-static list of symbol names that libpthdebug has ever requested.
/// Used for diagnostics / debugging.
static std::vector<std::string> g_pthdb_requested_symbol_names;

/// Recover the NativeProcessAIX instance from the opaque pthdb user handle.
static inline NativeProcessAIX *PthdbUserToProcess(pthdb_user_t user) {
  return reinterpret_cast<NativeProcessAIX *>(static_cast<uintptr_t>(user));
}

// ---------------------------------------------------------------------------
// libpthdebug callbacks
// ---------------------------------------------------------------------------

/// Resolve libpthreads symbol names to their load addresses.
/// Mirrors callback_symbol_addrs() in pthreaded.c.
static int cb_symbol_addrs(pthdb_user_t user, pthdb_symbol_t *symbols,
                            int count) {
  Log *log = GetLog(POSIXLog::Process);
  if (!symbols || count <= 0)
    return 1;

  NativeProcessAIX *process = PthdbUserToProcess(user);

  for (int i = 0; i < count; ++i) {
    if (!symbols[i].name || symbols[i].name[0] == '\0') {
      symbols[i].addr = 0;
      continue;
    }

    llvm::StringRef name(symbols[i].name);

    // Record the symbol name for diagnostics if we haven't seen it before.
    bool known = false;
    for (const auto &s : g_pthdb_requested_symbol_names) {
      if (s == name.str()) {
        known = true;
        break;
      }
    }
    if (!known)
      g_pthdb_requested_symbol_names.push_back(name.str());

    if (!process) {
      // Probe call with no real process — leave address as 0.
      symbols[i].addr = 0;
      continue;
    }

    // Check cache first; resolve via delegate if not yet cached.
    lldb::addr_t addr = process->LookupPthdbgSymbol(name);
    if (addr == LLDB_INVALID_ADDRESS) {
      addr = process->GetDelegate().SymbolLookup(name);
      LLDB_LOG(log, "cb_symbol_addrs: looking up '{0}' -> {1:x}", name, addr);
      if (addr != LLDB_INVALID_ADDRESS)
        process->StorePthdbgSymbol(name, addr);
    }

    symbols[i].addr =
        (addr != LLDB_INVALID_ADDRESS) ? static_cast<pthdb_addr_t>(addr) : 0;

    LLDB_LOG(log, "cb_symbol_addrs: '{0}' -> {1:x}", name,
             static_cast<uint64_t>(symbols[i].addr));
  }

  return 0;
}

/// Read bytes from the inferior's address space.
static int cb_read_data(pthdb_user_t user, void *buf, pthdb_addr_t addr,
                        size_t len) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "cb_read_data: addr={0:x} len={1}",
           static_cast<uint64_t>(addr), len);
  if (!buf || !addr || !len)
    return 1;
  NativeProcessAIX *process = PthdbUserToProcess(user);
  if (!process)
    return 1;
  size_t bytes_read = 0;
  Status error =
      process->ReadMemory(static_cast<lldb::addr_t>(addr), buf, len, bytes_read);
  int rc = (error.Success() && bytes_read == len) ? 0 : 1;
  LLDB_LOG(log, "cb_read_data: rc={0}", rc);
  return rc;
}

/// Write bytes to the inferior's address space.
static int cb_write_data(pthdb_user_t user, void *buf, pthdb_addr_t addr,
                         size_t len) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "cb_write_data: addr={0:x} len={1}",
           static_cast<uint64_t>(addr), len);
  if (!buf || !addr || !len)
    return 1;
  NativeProcessAIX *process = PthdbUserToProcess(user);
  if (!process)
    return 1;
  size_t bytes_written = 0;
  Status error =
      process->WriteMemory(static_cast<lldb::addr_t>(addr), buf, len, bytes_written);
  int rc = (error.Success() && bytes_written == len) ? 0 : 1;
  LLDB_LOG(log, "cb_write_data: rc={0}", rc);
  return rc;
}

/// Read registers for a kernel thread into a pthdb_context_t.
static int cb_read_regs(pthdb_user_t user, pthdb_tid_t tid,
                        unsigned long long flags, pthdb_context_t *context) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "cb_read_regs: tid={0} flags={1:x}",
           static_cast<uint64_t>(tid), flags);
  if (!tid || !flags || !context)
    return 1;

  NativeProcessAIX *process = PthdbUserToProcess(user);
  const bool is32 = process &&
                    process->GetArchitecture().GetAddressByteSize() == 4;
  lldb::tid_t ktid = static_cast<lldb::tid_t>(tid);

  if (flags & PTHDB_FLAG_GPRS) {
    if (is32) {
      GPR_PPC gpr{};
      Status s = NativeProcessAIX::PtraceWrapper(PTT_READ_GPRS, ktid, nullptr,
                                                 &gpr, sizeof(gpr));
      if (s.Fail())
        return 1;

      // TODO: Create a common function for this to avoid redundant code
      for (int i = 0; i < 32; ++i)
        context->gpr[i] = (&gpr.r0)[i];
      context->iar = gpr.pc;
      context->lr  = gpr.lr;
      context->ctr = gpr.ctr;
      context->cr  = gpr.cr;
      context->xer = gpr.xer;
      context->msr = gpr.msr;
    } else {
      GPR_PPC64 gpr{};
      Status s = NativeProcessAIX::PtraceWrapper(PTT_READ_GPRS, ktid, nullptr,
                                                 &gpr, sizeof(gpr));
      if (s.Fail())
        return 1;
      for (int i = 0; i < 32; ++i)
        context->gpr[i] = (&gpr.r0)[i];
      context->iar = gpr.pc;
      context->lr  = gpr.lr;
      context->ctr = gpr.ctr;
      context->cr  = gpr.cr;
      context->xer = gpr.xer;
      context->msr = gpr.msr;
    }
  }

  if (flags & PTHDB_FLAG_FPRS) {
    FPR_PPC64 fpr{};
    Status s = NativeProcessAIX::PtraceWrapper(PTT_READ_FPRS, ktid, nullptr,
                                               &fpr, sizeof(fpr));
    if (s.Fail())
      return 1;
    static_assert(sizeof(context->fpr[0]) == sizeof(uint64_t),
                  "FPR element size mismatch");
    const uint64_t *fpr_arr = reinterpret_cast<const uint64_t *>(&fpr);
    for (int i = 0; i < 32; ++i)
      ::memcpy(&context->fpr[i], &fpr_arr[i], sizeof(double));
    context->fpscr = static_cast<unsigned int>(fpr.fpscr);
  }

  if (flags & PTHDB_FLAG_VRS) {
    VMX_PPC64 vmx{};
    Status s = NativeProcessAIX::PtraceWrapper(PTT_READ_VEC, ktid, nullptr,
                                               &vmx, sizeof(vmx));
    if (s.Fail())
      return 1;
    // VMX copy intentionally left as no-op (layout differences); expand later.
  }

  if (flags & PTHDB_FLAG_VSRS) {
    VSX_PPC64 vsx{};
    Status s = NativeProcessAIX::PtraceWrapper(PTT_READ_VSX, ktid, nullptr,
                                               &vsx, sizeof(vsx));
    if (s.Fail())
      return 1;
    // VSX copy intentionally left as no-op; expand later.
  }

  LLDB_LOG(log, "cb_read_regs: done tid={0}", static_cast<uint64_t>(tid));
  return 0;
}

/// Write registers from a pthdb_context_t to a kernel thread.
static int cb_write_regs(pthdb_user_t user, pthdb_tid_t tid,
                         unsigned long long flags, pthdb_context_t *context) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "cb_write_regs: tid={0} flags={1:x}",
           static_cast<uint64_t>(tid), flags);
  if (!tid || !flags || !context)
    return 1;

  NativeProcessAIX *process = PthdbUserToProcess(user);
  const bool is32 = process &&
                    process->GetArchitecture().GetAddressByteSize() == 4;
  lldb::tid_t ktid = static_cast<lldb::tid_t>(tid);

  if (flags & PTHDB_FLAG_GPRS) {
    if (is32) {
      GPR_PPC gpr{};
      // TODO: Create a common function for this to avoid redundant code
      for (int i = 0; i < 32; ++i)
        (&gpr.r0)[i] = static_cast<uint32_t>(context->gpr[i]);
      gpr.pc  = static_cast<uint32_t>(context->iar);
      gpr.lr  = static_cast<uint32_t>(context->lr);
      gpr.ctr = static_cast<uint32_t>(context->ctr);
      gpr.cr  = static_cast<uint32_t>(context->cr);
      gpr.xer = static_cast<uint32_t>(context->xer);
      gpr.msr = static_cast<uint32_t>(context->msr);
      Status s = NativeProcessAIX::PtraceWrapper(PTT_WRITE_GPRS, ktid, nullptr,
                                                 &gpr, sizeof(gpr));
      if (s.Fail())
        return 1;
    } else {
      GPR_PPC64 gpr{};
      for (int i = 0; i < 32; ++i)
        (&gpr.r0)[i] = context->gpr[i];
      gpr.pc  = context->iar;
      gpr.lr  = context->lr;
      gpr.ctr = context->ctr;
      gpr.cr  = static_cast<uint32_t>(context->cr);
      gpr.xer = static_cast<uint32_t>(context->xer);
      gpr.msr = context->msr;
      Status s = NativeProcessAIX::PtraceWrapper(PTT_WRITE_GPRS, ktid, nullptr,
                                                 &gpr, sizeof(gpr));
      if (s.Fail())
        return 1;
    }
  }

  if (flags & PTHDB_FLAG_FPRS) {
    FPR_PPC64 fpr{};
    uint64_t *fpr_arr = reinterpret_cast<uint64_t *>(&fpr);
    for (int i = 0; i < 32; ++i)
      ::memcpy(&fpr_arr[i], &context->fpr[i], sizeof(double));
    fpr.fpscr = context->fpscr;
    Status s = NativeProcessAIX::PtraceWrapper(PTT_WRITE_FPRS, ktid, nullptr,
                                               &fpr, sizeof(fpr));
    if (s.Fail())
      return 1;
  }

  // VRS / VSRS write intentionally left as no-op; expand later.

  LLDB_LOG(log, "cb_write_regs: done tid={0}", static_cast<uint64_t>(tid));
  return 0;
}

/// Allocate a buffer for libpthdebug's internal use.
static int cb_alloc(pthdb_user_t /*user*/, size_t len, void **bufp) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "cb_alloc: len={0}", len);
  if (!len || !bufp)
    return 1;
  *bufp = ::malloc(len);
  int rc = (*bufp != nullptr) ? 0 : 1;
  LLDB_LOG(log, "cb_alloc: rc={0} buf={1}", rc, *bufp);
  return rc;
}

/// Resize a buffer for libpthdebug's internal use.
static int cb_realloc(pthdb_user_t /*user*/, void *buf, size_t len,
                      void **bufp) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "cb_realloc: buf={0} len={1}", buf, len);
  if (!buf || !len || !bufp)
    return 1;
  *bufp = ::realloc(buf, len);
  int rc = (*bufp != nullptr) ? 0 : 1;
  LLDB_LOG(log, "cb_realloc: rc={0} newbuf={1}", rc, *bufp);
  return rc;
}

/// Free a buffer allocated for libpthdebug.
static int cb_dealloc(pthdb_user_t /*user*/, void *buf) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "cb_dealloc: buf={0}", buf);
  if (!buf)
    return 1;
  ::free(buf);
  return 0;
}

/// Forward libpthdebug diagnostic messages to the LLDB log.
static int cb_print(pthdb_user_t /*user*/, char *str) {
  if (!str)
    return 1;
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "libpthdebug: {0}", str);
  return 0;
}

// ---------------------------------------------------------------------------
// Public globals / helpers
// ---------------------------------------------------------------------------

/// The single, process-wide callbacks table passed to pthdb_session_init().
pthdb_callbacks_t lldb_private::process_aix::g_pthdb_callbacks = {
    cb_symbol_addrs, cb_read_data,  cb_write_data, cb_read_regs,
    cb_write_regs,   cb_alloc,      cb_realloc,    cb_dealloc,
    cb_print};

/// Lazily initialise (and optionally update) the libpthdebug session.
/// Mirrors pthreads_active() from pthreaded.c.
bool lldb_private::process_aix::EnsurePthdbSession(NativeProcessAIX &process,
                                                    bool update) {
  Log *log = GetLog(POSIXLog::Process);

  if (process.IsPthdbNotPthreaded())
    return false;

  if (!process.IsPthdbInitialized()) {
    // Probe: check whether the inferior uses pthreads.
    // pthdb_session_pthreaded() resolves __n_pthreads via cb_symbol_addrs,
    // reads it, and returns PTHDB_SUCCESS only when __n_pthreads > 0.
    char *pthreaded_sym = nullptr;
    int probe_rc = pthdb_session_pthreaded(
        static_cast<pthdb_user_t>(reinterpret_cast<uintptr_t>(&process)),
        /*flags=*/0, &g_pthdb_callbacks, &pthreaded_sym);

    if (probe_rc == PTHDB_NOT_PTHREADED) {
      LLDB_LOG(log, "EnsurePthdbSession: inferior is not pthreaded");
      process.SetPthdbNotPthreaded();
      return false;
    }

    if (probe_rc != PTHDB_SUCCESS) {
      LLDB_LOG(log,
               "EnsurePthdbSession: pthdb_session_pthreaded failed (rc={0})",
               probe_rc);
      // Don't mark as not-pthreaded; symbols may not yet be resolved.
      return false;
    }

    LLDB_LOG(log,
             "EnsurePthdbSession: inferior is pthreaded, initialising session");

    // Real session init.  cb_symbol_addrs resolves each name via
    // GetDelegate().SymbolLookup() and caches it in the process.
    const pthdb_exec_mode_t pem =
        (process.GetArchitecture().GetAddressByteSize() == 4) ? PEM_32BIT
                                                              : PEM_64BIT;
    pthdb_session_t session = 0;
    int init_rc = pthdb_session_init(
        static_cast<pthdb_user_t>(reinterpret_cast<uintptr_t>(&process)),
        pem, PTHDB_FLAG_REGS | PTHDB_FLAG_HOLD, &g_pthdb_callbacks,
        &session);

    if (init_rc == PTHDB_NOT_PTHREADED) {
      LLDB_LOG(log, "EnsurePthdbSession: pthdb_session_init: not pthreaded");
      process.SetPthdbNotPthreaded();
      return false;
    }

    if (init_rc != PTHDB_SUCCESS) {
      LLDB_LOG(log,
               "EnsurePthdbSession: pthdb_session_init failed (rc={0})",
               init_rc);
      return false;
    }

    process.SetPthdbSession(session);
    LLDB_LOG(log, "EnsurePthdbSession: session initialised successfully");
  }

  if (update) {
    int upd_rc = pthdb_session_update(process.GetPthdbSession());
    if (upd_rc != PTHDB_SUCCESS)
      LLDB_LOG(log,
               "EnsurePthdbSession: pthdb_session_update failed (rc={0})",
               upd_rc);
  }

  return true;
}
