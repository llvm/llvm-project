//===-- SoftwareBreakpoint.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/SoftwareBreakpoint.h"

#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Host/Debug.h"
#include "lldb/Host/Mutex.h"

#include "lldb/Host/common/NativeProcessProtocol.h"

using namespace lldb_private;

// -------------------------------------------------------------------
// static members
// -------------------------------------------------------------------

Error SoftwareBreakpoint::CreateSoftwareBreakpoint(
    NativeProcessProtocol &process, lldb::addr_t addr, size_t size_hint,
    NativeBreakpointSP &breakpoint_sp) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
  if (log)
    log->Printf("SoftwareBreakpoint::%s addr = 0x%" PRIx64, __FUNCTION__, addr);

  // Validate the address.
  if (addr == LLDB_INVALID_ADDRESS)
    return Error("SoftwareBreakpoint::%s invalid load address specified.",
                 __FUNCTION__);

  // Ask the NativeProcessProtocol subclass to fill in the correct software
  // breakpoint
  // trap for the breakpoint site.
  size_t bp_opcode_size = 0;
  const uint8_t *bp_opcode_bytes = NULL;
  Error error = process.GetSoftwareBreakpointTrapOpcode(
      size_hint, bp_opcode_size, bp_opcode_bytes);

  if (error.Fail()) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s failed to retrieve software "
                  "breakpoint trap opcode: %s",
                  __FUNCTION__, error.AsCString());
    return error;
  }

  // Validate size of trap opcode.
  if (bp_opcode_size == 0) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s failed to retrieve any trap opcodes",
                  __FUNCTION__);
    return Error("SoftwareBreakpoint::GetSoftwareBreakpointTrapOpcode() "
                 "returned zero, unable to get breakpoint trap for address "
                 "0x%" PRIx64,
                 addr);
  }

  if (bp_opcode_size > MAX_TRAP_OPCODE_SIZE) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s cannot support %lu trapcode bytes, "
                  "max size is %lu",
                  __FUNCTION__, bp_opcode_size, MAX_TRAP_OPCODE_SIZE);
    return Error("SoftwareBreakpoint::GetSoftwareBreakpointTrapOpcode() "
                 "returned too many trap opcode bytes: requires %lu but we "
                 "only support a max of %lu",
                 bp_opcode_size, MAX_TRAP_OPCODE_SIZE);
  }

  // Validate that we received opcodes.
  if (!bp_opcode_bytes) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s failed to retrieve trap opcode bytes",
                  __FUNCTION__);
    return Error("SoftwareBreakpoint::GetSoftwareBreakpointTrapOpcode() "
                 "returned NULL trap opcode bytes, unable to get breakpoint "
                 "trap for address 0x%" PRIx64,
                 addr);
  }

  // Enable the breakpoint.
  uint8_t saved_opcode_bytes[MAX_TRAP_OPCODE_SIZE];
  error = EnableSoftwareBreakpoint(process, addr, bp_opcode_size,
                                   bp_opcode_bytes, saved_opcode_bytes);
  if (error.Fail()) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s: failed to enable new breakpoint at "
                  "0x%" PRIx64 ": %s",
                  __FUNCTION__, addr, error.AsCString());
    return error;
  }

  if (log)
    log->Printf("SoftwareBreakpoint::%s addr = 0x%" PRIx64 " -- SUCCESS",
                __FUNCTION__, addr);

  // Set the breakpoint and verified it was written properly.  Now
  // create a breakpoint remover that understands how to undo this
  // breakpoint.
  breakpoint_sp.reset(new SoftwareBreakpoint(process, addr, saved_opcode_bytes,
                                             bp_opcode_bytes, bp_opcode_size));
  return Error();
}

Error SoftwareBreakpoint::EnableSoftwareBreakpoint(
    NativeProcessProtocol &process, lldb::addr_t addr, size_t bp_opcode_size,
    const uint8_t *bp_opcode_bytes, uint8_t *saved_opcode_bytes) {
  assert(bp_opcode_size <= MAX_TRAP_OPCODE_SIZE &&
         "bp_opcode_size out of valid range");
  assert(bp_opcode_bytes && "bp_opcode_bytes is NULL");
  assert(saved_opcode_bytes && "saved_opcode_bytes is NULL");

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
  if (log)
    log->Printf("SoftwareBreakpoint::%s addr = 0x%" PRIx64, __FUNCTION__, addr);

  // Save the original opcodes by reading them so we can restore later.
  size_t bytes_read = 0;

  Error error =
      process.ReadMemory(addr, saved_opcode_bytes, bp_opcode_size, bytes_read);
  if (error.Fail()) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s failed to read memory while "
                  "attempting to set breakpoint: %s",
                  __FUNCTION__, error.AsCString());
    return error;
  }

  // Ensure we read as many bytes as we expected.
  if (bytes_read != bp_opcode_size) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s failed to read memory while "
                  "attempting to set breakpoint: attempted to read %lu bytes "
                  "but only read %" PRIu64,
                  __FUNCTION__, bp_opcode_size, (uint64_t)bytes_read);
    return Error("SoftwareBreakpoint::%s failed to read memory while "
                 "attempting to set breakpoint: attempted to read %lu bytes "
                 "but only read %" PRIu64,
                 __FUNCTION__, bp_opcode_size, (uint64_t)bytes_read);
  }

  // Log what we read.
  if (log) {
    int i = 0;
    for (const uint8_t *read_byte = saved_opcode_bytes;
         read_byte < saved_opcode_bytes + bp_opcode_size; ++read_byte) {
      log->Printf("SoftwareBreakpoint::%s addr = 0x%" PRIx64
                  " ovewriting byte index %d (was 0x%hhx)",
                  __FUNCTION__, addr, i++, *read_byte);
    }
  }

  // Write a software breakpoint in place of the original opcode.
  size_t bytes_written = 0;
  error =
      process.WriteMemory(addr, bp_opcode_bytes, bp_opcode_size, bytes_written);
  if (error.Fail()) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s failed to write memory while "
                  "attempting to set breakpoint: %s",
                  __FUNCTION__, error.AsCString());
    return error;
  }

  // Ensure we wrote as many bytes as we expected.
  if (bytes_written != bp_opcode_size) {
    error.SetErrorStringWithFormat(
        "SoftwareBreakpoint::%s failed write memory while attempting to set "
        "breakpoint: attempted to write %lu bytes but only wrote %" PRIu64,
        __FUNCTION__, bp_opcode_size, (uint64_t)bytes_written);
    if (log)
      log->PutCString(error.AsCString());
    return error;
  }

  uint8_t verify_bp_opcode_bytes[MAX_TRAP_OPCODE_SIZE];
  size_t verify_bytes_read = 0;
  error = process.ReadMemory(addr, verify_bp_opcode_bytes, bp_opcode_size,
                             verify_bytes_read);
  if (error.Fail()) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s failed to read memory while "
                  "attempting to verify the breakpoint set: %s",
                  __FUNCTION__, error.AsCString());
    return error;
  }

  // Ensure we read as many verification bytes as we expected.
  if (verify_bytes_read != bp_opcode_size) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s failed to read memory while "
                  "attempting to verify breakpoint: attempted to read %lu "
                  "bytes but only read %" PRIu64,
                  __FUNCTION__, bp_opcode_size, (uint64_t)verify_bytes_read);
    return Error("SoftwareBreakpoint::%s failed to read memory while "
                 "attempting to verify breakpoint: attempted to read %lu bytes "
                 "but only read %" PRIu64,
                 __FUNCTION__, bp_opcode_size, (uint64_t)verify_bytes_read);
  }

  if (::memcmp(bp_opcode_bytes, verify_bp_opcode_bytes, bp_opcode_size) != 0) {
    if (log)
      log->Printf("SoftwareBreakpoint::%s: verification of software breakpoint "
                  "writing failed - trap opcodes not successfully read back "
                  "after writing when setting breakpoint at 0x%" PRIx64,
                  __FUNCTION__, addr);
    return Error("SoftwareBreakpoint::%s: verification of software breakpoint "
                 "writing failed - trap opcodes not successfully read back "
                 "after writing when setting breakpoint at 0x%" PRIx64,
                 __FUNCTION__, addr);
  }

  if (log)
    log->Printf("SoftwareBreakpoint::%s addr = 0x%" PRIx64 " -- SUCCESS",
                __FUNCTION__, addr);

  return Error();
}

// -------------------------------------------------------------------
// instance-level members
// -------------------------------------------------------------------

SoftwareBreakpoint::SoftwareBreakpoint(NativeProcessProtocol &process,
                                       lldb::addr_t addr,
                                       const uint8_t *saved_opcodes,
                                       const uint8_t *trap_opcodes,
                                       size_t opcode_size)
    : NativeBreakpoint(addr), m_process(process), m_saved_opcodes(),
      m_trap_opcodes(), m_opcode_size(opcode_size) {
  assert(opcode_size > 0 && "setting software breakpoint with no trap opcodes");
  assert(opcode_size <= MAX_TRAP_OPCODE_SIZE && "trap opcode size too large");

  ::memcpy(m_saved_opcodes, saved_opcodes, opcode_size);
  ::memcpy(m_trap_opcodes, trap_opcodes, opcode_size);
}

Error SoftwareBreakpoint::DoEnable() {
  return EnableSoftwareBreakpoint(m_process, m_addr, m_opcode_size,
                                  m_trap_opcodes, m_saved_opcodes);
}

Error SoftwareBreakpoint::DoDisable() {
  Error error;
  assert(m_addr && (m_addr != LLDB_INVALID_ADDRESS) &&
         "can't remove a software breakpoint for an invalid address");

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
  if (log)
    log->Printf("SoftwareBreakpoint::%s addr = 0x%" PRIx64, __FUNCTION__,
                m_addr);

  assert((m_opcode_size > 0) &&
         "cannot restore opcodes when there are no opcodes");

  if (m_opcode_size > 0) {
    // Clear a software breakpoint instruction
    uint8_t curr_break_op[MAX_TRAP_OPCODE_SIZE];
    bool break_op_found = false;
    assert(m_opcode_size <= sizeof(curr_break_op));

    // Read the breakpoint opcode
    size_t bytes_read = 0;
    error =
        m_process.ReadMemory(m_addr, curr_break_op, m_opcode_size, bytes_read);
    if (error.Success() && bytes_read < m_opcode_size) {
      error.SetErrorStringWithFormat(
          "SoftwareBreakpointr::%s addr=0x%" PRIx64
          ": tried to read %lu bytes but only read %" PRIu64,
          __FUNCTION__, m_addr, m_opcode_size, (uint64_t)bytes_read);
    }
    if (error.Success()) {
      bool verify = false;
      // Make sure the breakpoint opcode exists at this address
      if (::memcmp(curr_break_op, m_trap_opcodes, m_opcode_size) == 0) {
        break_op_found = true;
        // We found a valid breakpoint opcode at this address, now restore
        // the saved opcode.
        size_t bytes_written = 0;
        error = m_process.WriteMemory(m_addr, m_saved_opcodes, m_opcode_size,
                                      bytes_written);
        if (error.Success() && bytes_written < m_opcode_size) {
          error.SetErrorStringWithFormat(
              "SoftwareBreakpoint::%s addr=0x%" PRIx64
              ": tried to write %lu bytes but only wrote %" PRIu64,
              __FUNCTION__, m_addr, m_opcode_size, (uint64_t)bytes_written);
        }
        if (error.Success()) {
          verify = true;
        }
      } else {
        error.SetErrorString(
            "Original breakpoint trap is no longer in memory.");
        // Set verify to true and so we can check if the original opcode has
        // already been restored
        verify = true;
      }

      if (verify) {
        uint8_t verify_opcode[MAX_TRAP_OPCODE_SIZE];
        assert(m_opcode_size <= sizeof(verify_opcode));
        // Verify that our original opcode made it back to the inferior

        size_t verify_bytes_read = 0;
        error = m_process.ReadMemory(m_addr, verify_opcode, m_opcode_size,
                                     verify_bytes_read);
        if (error.Success() && verify_bytes_read < m_opcode_size) {
          error.SetErrorStringWithFormat(
              "SoftwareBreakpoint::%s addr=0x%" PRIx64
              ": tried to read %lu verification bytes but only read %" PRIu64,
              __FUNCTION__, m_addr, m_opcode_size, (uint64_t)verify_bytes_read);
        }
        if (error.Success()) {
          // compare the memory we just read with the original opcode
          if (::memcmp(m_saved_opcodes, verify_opcode, m_opcode_size) == 0) {
            // SUCCESS
            if (log) {
              int i = 0;
              for (const uint8_t *verify_byte = verify_opcode;
                   verify_byte < verify_opcode + m_opcode_size; ++verify_byte) {
                log->Printf("SoftwareBreakpoint::%s addr = 0x%" PRIx64
                            " replaced byte index %d with 0x%hhx",
                            __FUNCTION__, m_addr, i++, *verify_byte);
              }
              log->Printf("SoftwareBreakpoint::%s addr = 0x%" PRIx64
                          " -- SUCCESS",
                          __FUNCTION__, m_addr);
            }
            return error;
          } else {
            if (break_op_found)
              error.SetErrorString("Failed to restore original opcode.");
          }
        } else
          error.SetErrorString("Failed to read memory to verify that "
                               "breakpoint trap was restored.");
      }
    }
  }

  if (log && error.Fail())
    log->Printf("SoftwareBreakpoint::%s addr = 0x%" PRIx64 " -- FAILED: %s",
                __FUNCTION__, m_addr, error.AsCString());
  return error;
}

bool SoftwareBreakpoint::IsSoftwareBreakpoint() const { return true; }
