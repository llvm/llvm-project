//===-- lib/Archive/ArchiveInternals.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Internal implementation header for LLVM Archive files.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_ARCHIVE_ARCHIVEINTERNALS_H
#define LIB_ARCHIVE_ARCHIVEINTERNALS_H

#include "llvm/Bitcode/Archive.h"
#include "llvm/System/TimeValue.h"
#include "llvm/ADT/StringExtras.h"

#include <cstring>

#define ARFILE_MAGIC "!<arch>\n"                   ///< magic string
#define ARFILE_MAGIC_LEN (sizeof(ARFILE_MAGIC)-1)  ///< length of magic string
#define ARFILE_SVR4_SYMTAB_NAME "/               " ///< SVR4 symtab entry name
#define ARFILE_LLVM_SYMTAB_NAME "#_LLVM_SYM_TAB_#" ///< LLVM symtab entry name
#define ARFILE_BSD4_SYMTAB_NAME "__.SYMDEF SORTED" ///< BSD4 symtab entry name
#define ARFILE_STRTAB_NAME      "//              " ///< Name of string table
#define ARFILE_PAD "\n"                            ///< inter-file align padding
#define ARFILE_MEMBER_MAGIC "`\n"                  ///< fmag field magic #

namespace llvm {

  /// The ArchiveMemberHeader structure is used internally for bitcode
  /// archives.
  /// The header precedes each file member in the archive. This structure is
  /// defined using character arrays for direct and correct interpretation
  /// regardless of the endianess of the machine that produced it.
  /// @brief Archive File Member Header
  class ArchiveMemberHeader {
    /// @name Data
    /// @{
    public:
      char name[16];  ///< Name of the file member.
      char date[12];  ///< File date, decimal seconds since Epoch
      char uid[6];    ///< user id in ASCII decimal
      char gid[6];    ///< group id in ASCII decimal
      char mode[8];   ///< file mode in ASCII octal
      char size[10];  ///< file size in ASCII decimal
      char fmag[2];   ///< Always contains ARFILE_MAGIC_TERMINATOR

    /// @}
    /// @name Methods
    /// @{
    public:
    void init() {
      memset(name,' ',16);
      memset(date,' ',12);
      memset(uid,' ',6);
      memset(gid,' ',6);
      memset(mode,' ',8);
      memset(size,' ',10);
      fmag[0] = '`';
      fmag[1] = '\n';
    }

    bool checkSignature() {
      return 0 == memcmp(fmag, ARFILE_MEMBER_MAGIC,2);
    }
  };
  
  // Get just the externally visible defined symbols from the bitcode
  bool GetBitcodeSymbols(const sys::Path& fName,
                          std::vector<std::string>& symbols,
                          std::string* ErrMsg);
  
  ModuleProvider* GetBitcodeSymbols(const unsigned char*Buffer,unsigned Length,
                                    const std::string& ModuleID,
                                    std::vector<std::string>& symbols,
                                    std::string* ErrMsg);
}

#endif

// vim: sw=2 ai
