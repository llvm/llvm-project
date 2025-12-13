//===- MCSFrame.h - Machine Code SFrame support ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of MCSFrameEmitter to support emitting
// sframe unwinding info from .cfi_* directives. It relies on FDEs and CIEs
// created for Dwarf frame info, but emits that info in a different format.
//
// See https://sourceware.org/binutils/docs-2.41/sframe-spec.html
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSFRAME_H
#define LLVM_MC_MCSFRAME_H

#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace llvm {

class MCContext;
class MCObjectStreamer;
class MCFragment;

class MCSFrameEmitter {
public:
  // Emit the sframe section.
  //
  // \param Streamer - Emit into this stream.
  static void emit(MCObjectStreamer &Streamer);

  // Encode the FRE's function offset.
  //
  // \param C - Context.
  // \param Offset - Offset to encode.
  // \param Out - Destination of the encoding.
  // \param FDEFrag - Frag that specifies the encoding format.
  static void encodeFuncOffset(MCContext &C, uint64_t Offset,
                               SmallVectorImpl<char> &Out, MCFragment *FDEFrag);
};

} // end namespace llvm
#endif // LLVM_MC_MCSFRAME_H
