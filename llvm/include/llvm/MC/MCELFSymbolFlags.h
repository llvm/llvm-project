//===- MCELFSymbolFlags.h - ELF Symbol Flags ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SymbolFlags used for the ELF target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELFSYMBOLFLAGS_H
#define LLVM_MC_MCELFSYMBOLFLAGS_H

#include "llvm/Support/ELF.h"

// Because all the symbol flags need to be stored in the MCSymbolData
// 'flags' variable we need to provide shift constants per flag type.

namespace llvm {
  enum {
    ELF_STT_Shift   = 0, // Shift value for STT_* flags.
    ELF_STB_Shift   = 4, // Shift value for STB_* flags.
    ELF_STV_Shift   = 8, // Shift value for STV_* flags.
    ELF_Other_Shift = 10 // Shift value for other flags.
  };

  enum SymbolFlags {
    ELF_STB_Local     = (ELF::STB_LOCAL     << ELF_STB_Shift),
      ELF_STB_Global    = (ELF::STB_GLOBAL    << ELF_STB_Shift),
      ELF_STB_Weak      = (ELF::STB_WEAK      << ELF_STB_Shift),
      ELF_STB_Loproc    = (ELF::STB_LOPROC    << ELF_STB_Shift),
      ELF_STB_Hiproc    = (ELF::STB_HIPROC    << ELF_STB_Shift),

      ELF_STT_Notype    = (ELF::STT_NOTYPE    << ELF_STT_Shift),
      ELF_STT_Object    = (ELF::STT_OBJECT    << ELF_STT_Shift),
      ELF_STT_Func      = (ELF::STT_FUNC      << ELF_STT_Shift),
      ELF_STT_Section   = (ELF::STT_SECTION   << ELF_STT_Shift),
      ELF_STT_File      = (ELF::STT_FILE      << ELF_STT_Shift),
      ELF_STT_Common    = (ELF::STT_COMMON    << ELF_STT_Shift),
      ELF_STT_Tls       = (ELF::STT_TLS       << ELF_STT_Shift),
      ELF_STT_Loproc    = (ELF::STT_LOPROC    << ELF_STT_Shift),
      ELF_STT_Hiproc    = (ELF::STT_HIPROC    << ELF_STT_Shift),

      ELF_STV_Default   = (ELF::STV_DEFAULT   << ELF_STV_Shift),
      ELF_STV_Internal  = (ELF::STV_INTERNAL  << ELF_STV_Shift),
      ELF_STV_Hidden    = (ELF::STV_HIDDEN    << ELF_STV_Shift),
      ELF_STV_Protected = (ELF::STV_PROTECTED << ELF_STV_Shift),

      ELF_Other_Weakref = (1                  << ELF_Other_Shift),
      ELF_Other_ThumbFunc = (2                << ELF_Other_Shift)
  };

} // end namespace llvm

#endif
