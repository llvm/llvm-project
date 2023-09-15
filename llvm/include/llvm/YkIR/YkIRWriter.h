#ifndef __YKIRWRITER_H
#define __YKIRWRITER_H

#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
  void embedYkIR(MCContext &Ctx, MCStreamer &OutStreamer, Module &M);
} // namespace llvm

#endif
