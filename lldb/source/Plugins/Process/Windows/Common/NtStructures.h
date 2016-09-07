//===-- NtStructures.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_Common_NtStructures_h_
#define liblldb_Plugins_Process_Windows_Common_NtStructures_h_

#include "lldb/Host/windows/windows.h"

// This describes the layout of a TEB (Thread Environment Block) for a 64-bit
// process.  It's adapted from the 32-bit TEB in winternl.h.  Currently, we care
// only about the position of the TlsSlots.
struct TEB64 {
  ULONG64 Reserved1[12];
  ULONG64 ProcessEnvironmentBlock;
  ULONG64 Reserved2[399];
  BYTE Reserved3[1952];
  ULONG64 TlsSlots[64];
  BYTE Reserved4[8];
  ULONG64 Reserved5[26];
  ULONG64 ReservedForOle; // Windows 2000 only
  ULONG64 Reserved6[4];
  ULONG64 TlsExpansionSlots;
};

#endif
