//===-- NVPTXAsmStreamer.cpp - NVPTX assembly text output ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVPTXAsmStreamer.h"

using namespace llvm;

NVPTXAsmStreamer::NVPTXAsmStreamer(MCContext &Context,
                                   std::unique_ptr<formatted_raw_ostream> os,
                                   std::unique_ptr<MCInstPrinter> printer,
                                   std::unique_ptr<MCCodeEmitter> emitter,
                                   std::unique_ptr<MCAsmBackend> asmbackend)
    : MCAsmBaseStreamer(Context), OSOwner(std::move(os)), OS(*OSOwner),
      MAI(Context.getAsmInfo()), InstPrinter(std::move(printer)),
      Assembler(std::make_unique<MCAssembler>(
          Context, std::move(asmbackend), std::move(emitter),
          (asmbackend) ? asmbackend->createObjectWriter(NullStream) : nullptr)),
      CommentStream(CommentToEmit) {
  assert(InstPrinter);
  if (Assembler->getBackendPtr())
    setAllowAutoPadding(Assembler->getBackend().allowAutoPadding());

  Context.setUseNamesOnTempLabels(true);

  auto *TO = Context.getTargetOptions();
  IsVerboseAsm = TO->AsmVerbose;
  if (IsVerboseAsm)
    InstPrinter->setCommentStream(CommentStream);
  ShowInst = TO->ShowMCInst;
  switch (TO->MCUseDwarfDirectory) {
  case MCTargetOptions::DisableDwarfDirectory:
    UseDwarfDirectory = false;
    break;
  case MCTargetOptions::EnableDwarfDirectory:
    UseDwarfDirectory = true;
    break;
  case MCTargetOptions::DefaultDwarfDirectory:
    UseDwarfDirectory = Context.getAsmInfo()->enableDwarfFileDirectoryDefault();
    break;
  }
}

void NVPTXAsmStreamer::EmitCommentsAndEOL() {
  if (CommentToEmit.empty() && CommentStream.GetNumBytesInBuffer() == 0) {
    OS << '\n';
    return;
  }

  StringRef Comments = CommentToEmit;

  assert(Comments.back() == '\n' && "Comment array not newline terminated");
  do {
    // Emit a line of comments.
    OS.PadToColumn(MAI->getCommentColumn());
    size_t Position = Comments.find('\n');
    OS << MAI->getCommentString() << ' ' << Comments.substr(0, Position)
       << '\n';

    Comments = Comments.substr(Position + 1);
  } while (!Comments.empty());

  CommentToEmit.clear();
}

void NVPTXAsmStreamer::emitExplicitComments() {
  StringRef Comments = ExplicitCommentToEmit;
  if (!Comments.empty())
    OS << Comments;
  ExplicitCommentToEmit.clear();
}

void NVPTXAsmStreamer::emitRawTextImpl(StringRef String) {
  String.consume_back("\n");
  OS << String;
  EmitEOL();
}
