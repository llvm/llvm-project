//===- MCMachOSymbolFlags.h - MachO Symbol Flags ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SymbolFlags used for the MachO target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCMACHOSYMBOLFLAGS_H
#define LLVM_MC_MCMACHOSYMBOLFLAGS_H

// These flags are mostly used in MCMachOStreamer.cpp but also needed in
// MachObjectWriter.cpp to test for Weak Definitions of symbols to emit
// the correct relocation information.

namespace llvm {
  /// SymbolFlags - We store the value for the 'desc' symbol field in the lowest
  /// 16 bits of the implementation defined flags.
  enum SymbolFlags { // See <mach-o/nlist.h>.
    SF_DescFlagsMask                        = 0xFFFF,

    // Reference type flags.
    SF_ReferenceTypeMask                    = 0x0007,
    SF_ReferenceTypeUndefinedNonLazy        = 0x0000,
    SF_ReferenceTypeUndefinedLazy           = 0x0001,
    SF_ReferenceTypeDefined                 = 0x0002,
    SF_ReferenceTypePrivateDefined          = 0x0003,
    SF_ReferenceTypePrivateUndefinedNonLazy = 0x0004,
    SF_ReferenceTypePrivateUndefinedLazy    = 0x0005,

    // Other 'desc' flags.
    SF_ThumbFunc                            = 0x0008,
    SF_NoDeadStrip                          = 0x0020,
    SF_WeakReference                        = 0x0040,
    SF_WeakDefinition                       = 0x0080,
    SF_SymbolResolver                       = 0x0100
  };

} // end namespace llvm

#endif
