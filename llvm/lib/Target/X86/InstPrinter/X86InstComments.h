//=- X86InstComments.h - Generate verbose-asm comments for instrs -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines functionality used to emit comments about X86 instructions to
// an output stream for -fverbose-asm.
//
//===----------------------------------------------------------------------===//

#ifndef X86_INST_COMMENTS_H
#define X86_INST_COMMENTS_H

namespace llvm {
  class MCInst;
  class raw_ostream;
  void EmitAnyX86InstComments(const MCInst *MI, raw_ostream &OS,
                              const char *(*getRegName)(unsigned));
}

#endif
