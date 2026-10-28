//===-- RegisterUtilities.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERUTILITIES_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERUTILITIES_H

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "lldb/Utility/DataExtractor.h"
#include "llvm/BinaryFormat/ELF.h"

namespace lldb_private {
/// Core files PT_NOTE segment descriptor types

namespace NETBSD {
enum { NT_PROCINFO = 1, NT_AUXV = 2 };

/* Size in bytes */
enum { NT_PROCINFO_SIZE = 160 };

/* Size in bytes */
enum {
  NT_PROCINFO_CPI_VERSION_SIZE = 4,
  NT_PROCINFO_CPI_CPISIZE_SIZE = 4,
  NT_PROCINFO_CPI_SIGNO_SIZE = 4,
  NT_PROCINFO_CPI_SIGCODE_SIZE = 4,
  NT_PROCINFO_CPI_SIGPEND_SIZE = 16,
  NT_PROCINFO_CPI_SIGMASK_SIZE = 16,
  NT_PROCINFO_CPI_SIGIGNORE_SIZE = 16,
  NT_PROCINFO_CPI_SIGCATCH_SIZE = 16,
  NT_PROCINFO_CPI_PID_SIZE = 4,
  NT_PROCINFO_CPI_PPID_SIZE = 4,
  NT_PROCINFO_CPI_PGRP_SIZE = 4,
  NT_PROCINFO_CPI_SID_SIZE = 4,
  NT_PROCINFO_CPI_RUID_SIZE = 4,
  NT_PROCINFO_CPI_EUID_SIZE = 4,
  NT_PROCINFO_CPI_SVUID_SIZE = 4,
  NT_PROCINFO_CPI_RGID_SIZE = 4,
  NT_PROCINFO_CPI_EGID_SIZE = 4,
  NT_PROCINFO_CPI_SVGID_SIZE = 4,
  NT_PROCINFO_CPI_NLWPS_SIZE = 4,
  NT_PROCINFO_CPI_NAME_SIZE = 32,
  NT_PROCINFO_CPI_SIGLWP_SIZE = 4,
};

namespace AARCH64 {
enum { NT_REGS = 32, NT_FPREGS = 34 };
}

namespace AMD64 {
enum { NT_REGS = 33, NT_FPREGS = 35 };
}

namespace I386 {
enum { NT_REGS = 33, NT_FPREGS = 35 };
}

} // namespace NETBSD

namespace OPENBSD {
enum {
  NT_PROCINFO = 10,
  NT_AUXV = 11,
  NT_REGS = 20,
  NT_FPREGS = 21,
};
}

namespace RISCV32 {
enum {
  // 'NT_CSREGMAP' is a new ELF core-file note type defined for RISC-V that
  // encodes a sparse set of Control and Status Registers (CSRs) and their
  // values, instead of dumping all 4,096 possible CSRs. This note records only
  // the CSRs that are relevant (e.g. implemented or non-default) as a series of
  // key–value pairs in a RISC-V core dump image. This keeps core files compact
  // by omitting the many unimplemented or unused CSRs out of the 12-bit CSR
  // address space (which allows up to 4,096 CSR indices).
  //
  // The format of the 'NT_CSREGMAP' note is as follows:
  //
  // ELF Note Header
  //
  //    As with all core notes, the entry begins with the standard ELF note
  //    header fields –
  //
  //    namesz
  //
  //        Length of the note name (including null terminator). For
  //        'NT_CSREGMAP'the note name is the usual core dump note name (e.g.
  //        "CORE" for OS-generated core files).
  //
  //    descsz
  //
  //        Length of the note descriptor (payload) in bytes.
  //
  //    note type
  //
  //        An integer tag identifying the note’s type. For 'NT_CSREGMAP', the
  //        note type field is set to NT_CSREGMAP (a new enum value distinct
  //        from other note types).
  //
  // Note Payload
  //
  //    The descriptor payload (desc) contains a sequence of CSR entries encoded
  //    in binary format. All multi-byte values use the target’s endianness
  //    (typically little-endian for RISC-V). The layout of the payload is as
  //    follows:
  //
  //    CSR entries
  //
  //        A list of N key-value pairs structured as follows:
  //
  //        Key (CSR Identifier)
  //
  //            A 32-bit unsigned integer identifying which CSR the entry
  //            represents. This corresponds to the CSR’s address/index in the
  //            RISC-V CSR address space (0–4,095). Ideally, each CSR would
  //            appear at most once; if by error the note has duplicate keys,
  //            then the consumer may use the first or the last occurrence but
  //            in a well-formed core dump image, duplicates won’t occur. The
  //            current implementation uses the first occurrence and skips
  //            subsequent occurrences.
  //
  //        Value (CSR Content)
  //
  //            An XLEN-sized value that was held in the CSR identified by 'Key'
  //            at the time that the core dump image was generated. This will be
  //            a 32-bit value for a 32-bit RISC-V core dump image and a 64-bit
  //            value for a 64-bit RISC-V core dump image.
  //
  //        For instance, if the machine status register, 'mstatus' (CSR 0x300)
  //        had the value '0x00001800' at the time that the core dump image was
  //        generated, then the entry would have key '0x300' and value
  //        '0x00001800'.
  //
  //        The entries appear consecutively in the note data. There is no fixed
  //        ordering requirement for the entries, but they may be sorted by CSR
  //        number for consistency (this is up to the producer of the core dump
  //        image).
  //
  //        For a 32-bit RISC-V core dump image, each entry is 4 bytes (key) + 4
  //        bytes (value) = 8 bytes. For a 64-bit core dump image, each entry is
  //        4 bytes (key) + 8 bytes (value) = 12 bytes. The 'descsz' field in
  //        the note header tells the consumer how big the payload is, so a
  //        consumer can calculate the number of entries, N, as N = descsz / 8
  //        for a 32-bit RISC-V core dump image, or N = descz / 12 for a 64-bit
  //        RISC-V core dump image.
  //
  //        [Illustration] Suppose we only want to save three CSRs (say, CSRs
  //        'mstatus' (0x300), 'mtvec' (0x305), and 'mscratch' (0x344)) in a
  //        32-bit RISC-V core dump image. Then, here's what the raw layout of
  //        the 'NT_CSREGMAP' note’s descriptor payload might look like:
  //
  //        |----------------|-------------|
  //        | Offset (bytes) |    Value    |
  //        |----------------|-------------|
  //        |      0–3       | 0x00000300  |
  //        |----------------|-------------|
  //        |      4–7       | 0x00001800  |
  //        |----------------|-------------|
  //        |      8–11      | 0x00000305  |
  //        |----------------|-------------|
  //        |     12–15      | 0x00000004  |
  //        |----------------|-------------|
  //        |     16–19      | 0x00000344  |
  //        |----------------|-------------|
  //        |     20–23      | 0x00000000  |
  //        |----------------|-------------|
  //
  //        In this illustration, 'descsz' would be 3 * 8 bytes = 24 bytes.
  //
  // When a consumer reads a core dump image with an 'NT_CSREGMAP' note, it will
  // parse the note to populate the target’s register set with the recorded CSR
  // values. Only the listed CSRs will be updated; any CSR not present in the
  // note will be assumed to be unavailable.
  NT_CSREGMAP = 20,
};
}

struct CoreNote {
  ELFNote info;
  DataExtractor data;
};

// A structure describing how to find a register set in a core file from a given
// OS.
struct RegsetDesc {
  // OS to which this entry applies to. Must not be UnknownOS.
  llvm::Triple::OSType OS;

  // Architecture to which this entry applies to. Can be UnknownArch, in which
  // case it applies to all architectures of a given OS.
  llvm::Triple::ArchType Arch;

  // The note type under which the register set can be found.
  uint32_t Note;
};

// Returns the register set in Notes which corresponds to the specified Triple
// according to the list of register set descriptions in RegsetDescs. The list
// is scanned linearly, so you can use a more specific entry (e.g. linux-i386)
// to override a more general entry (e.g. general linux), as long as you place
// it earlier in the list. If a register set is not found, it returns an empty
// DataExtractor.
DataExtractor getRegset(llvm::ArrayRef<CoreNote> Notes,
                        const llvm::Triple &Triple,
                        llvm::ArrayRef<RegsetDesc> RegsetDescs);

constexpr RegsetDesc FPR_Desc[] = {
    // FreeBSD/i386 core NT_FPREGSET is x87 FSAVE result but the XSAVE dump
    // starts with FXSAVE struct, so use that instead if available.
    {llvm::Triple::FreeBSD, llvm::Triple::x86, llvm::ELF::NT_X86_XSTATE},
    {llvm::Triple::FreeBSD, llvm::Triple::UnknownArch, llvm::ELF::NT_FPREGSET},
    // In a i386 core file NT_FPREGSET is present, but it's not the result
    // of the FXSAVE instruction like in 64 bit files.
    // The result from FXSAVE is in NT_PRXFPREG for i386 core files
    {llvm::Triple::Linux, llvm::Triple::x86, llvm::ELF::NT_PRXFPREG},
    {llvm::Triple::Linux, llvm::Triple::UnknownArch, llvm::ELF::NT_FPREGSET},
    {llvm::Triple::NetBSD, llvm::Triple::aarch64, NETBSD::AARCH64::NT_FPREGS},
    {llvm::Triple::NetBSD, llvm::Triple::x86, NETBSD::I386::NT_FPREGS},
    {llvm::Triple::NetBSD, llvm::Triple::x86_64, NETBSD::AMD64::NT_FPREGS},
    {llvm::Triple::OpenBSD, llvm::Triple::UnknownArch, OPENBSD::NT_FPREGS},
    // Bare-metal 32-bit RISC-V debug target.
    {llvm::Triple::UnknownOS, llvm::Triple::riscv32, llvm::ELF::NT_FPREGSET},
};

constexpr RegsetDesc AARCH64_SVE_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::aarch64, llvm::ELF::NT_ARM_SVE},
};

constexpr RegsetDesc AARCH64_SSVE_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::aarch64, llvm::ELF::NT_ARM_SSVE},
};

constexpr RegsetDesc AARCH64_ZA_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::aarch64, llvm::ELF::NT_ARM_ZA},
};

constexpr RegsetDesc AARCH64_ZT_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::aarch64, llvm::ELF::NT_ARM_ZT},
};

constexpr RegsetDesc AARCH64_PAC_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::aarch64, llvm::ELF::NT_ARM_PAC_MASK},
};

constexpr RegsetDesc AARCH64_TLS_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::aarch64, llvm::ELF::NT_ARM_TLS},
};

constexpr RegsetDesc AARCH64_MTE_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::aarch64,
     llvm::ELF::NT_ARM_TAGGED_ADDR_CTRL},
};

constexpr RegsetDesc AARCH64_FPMR_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::aarch64, llvm::ELF::NT_ARM_FPMR},
};

constexpr RegsetDesc AARCH64_GCS_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::aarch64, llvm::ELF::NT_ARM_GCS},
};

constexpr RegsetDesc AARCH64_POE_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::aarch64, llvm::ELF::NT_ARM_POE},
};

constexpr RegsetDesc ARM_VFP_Desc[] = {
    {llvm::Triple::FreeBSD, llvm::Triple::arm, llvm::ELF::NT_ARM_VFP},
    {llvm::Triple::Linux, llvm::Triple::arm, llvm::ELF::NT_ARM_VFP},
};

constexpr RegsetDesc PPC_VMX_Desc[] = {
    {llvm::Triple::FreeBSD, llvm::Triple::UnknownArch, llvm::ELF::NT_PPC_VMX},
    {llvm::Triple::Linux, llvm::Triple::UnknownArch, llvm::ELF::NT_PPC_VMX},
};

constexpr RegsetDesc PPC_VSX_Desc[] = {
    {llvm::Triple::Linux, llvm::Triple::UnknownArch, llvm::ELF::NT_PPC_VSX},
};

constexpr RegsetDesc RISCV32_CSREGMAP_Desc[] = {
    {llvm::Triple::UnknownOS, llvm::Triple::riscv32, RISCV32::NT_CSREGMAP},
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERUTILITIES_H
