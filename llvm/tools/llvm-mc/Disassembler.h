//===- Disassembler.h - Text File Disassembler ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the disassembler of strings of bytes written in
// hexadecimal, from standard input or from a file.
//
//===----------------------------------------------------------------------===//

#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H

#include <string>

namespace llvm {

class MemoryBuffer;
class Target;
class raw_ostream;
class SourceMgr;
class MCSubtargetInfo;
class MCStreamer;

class Disassembler {
public:
  static int disassemble(const Target &T,
                         const std::string &Triple,
                         MCSubtargetInfo &STI,
                         MCStreamer &Streamer,
                         MemoryBuffer &Buffer,
                         SourceMgr &SM,
                         raw_ostream &Out);

  static int disassembleEnhanced(const std::string &tripleString,
                                 MemoryBuffer &buffer,
                                 SourceMgr &SM,
                                 raw_ostream &Out);
};

} // namespace llvm

#endif
