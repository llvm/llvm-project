//===- COFFObjcopy.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "COFFObjcopy.h"
#include "Buffer.h"
#include "CopyConfig.h"
#include "Object.h"
#include "Reader.h"
#include "Writer.h"
#include "llvm-objcopy.h"

#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Errc.h"
#include <cassert>

namespace llvm {
namespace objcopy {
namespace coff {

using namespace object;
using namespace COFF;

static bool isDebugSection(const Section &Sec) {
  return Sec.Name.startswith(".debug");
}

static Error handleArgs(const CopyConfig &Config, Object &Obj) {
  // Perform the actual section removals.
  Obj.removeSections([&Config](const Section &Sec) {
    // Contrary to --only-keep-debug, --only-section fully removes sections that
    // aren't mentioned.
    if (!Config.OnlySection.empty() &&
        !is_contained(Config.OnlySection, Sec.Name))
      return true;

    if (Config.StripDebug || Config.StripAll || Config.StripAllGNU ||
        Config.DiscardAll || Config.StripUnneeded) {
      if (isDebugSection(Sec) &&
          (Sec.Header.Characteristics & IMAGE_SCN_MEM_DISCARDABLE) != 0)
        return true;
    }

    if (is_contained(Config.ToRemove, Sec.Name))
      return true;

    return false;
  });

  if (Config.OnlyKeepDebug) {
    // For --only-keep-debug, we keep all other sections, but remove their
    // content. The VirtualSize field in the section header is kept intact.
    Obj.truncateSections([](const Section &Sec) {
      return !isDebugSection(Sec) && Sec.Name != ".buildid" &&
             ((Sec.Header.Characteristics &
               (IMAGE_SCN_CNT_CODE | IMAGE_SCN_CNT_INITIALIZED_DATA)) != 0);
    });
  }

  // StripAll removes all symbols and thus also removes all relocations.
  if (Config.StripAll || Config.StripAllGNU)
    for (Section &Sec : Obj.getMutableSections())
      Sec.Relocs.clear();

  // If we need to do per-symbol removals, initialize the Referenced field.
  if (Config.StripUnneeded || Config.DiscardAll ||
      !Config.SymbolsToRemove.empty())
    if (Error E = Obj.markSymbols())
      return E;

  // Actually do removals of symbols.
  Obj.removeSymbols([&](const Symbol &Sym) {
    // For StripAll, all relocations have been stripped and we remove all
    // symbols.
    if (Config.StripAll || Config.StripAllGNU)
      return true;

    if (is_contained(Config.SymbolsToRemove, Sym.Name)) {
      // Explicitly removing a referenced symbol is an error.
      if (Sym.Referenced)
        reportError(Config.OutputFilename,
                    createStringError(llvm::errc::invalid_argument,
                                      "not stripping symbol '%s' because it is "
                                      "named in a relocation.",
                                      Sym.Name.str().c_str()));
      return true;
    }

    if (!Sym.Referenced) {
      // With --strip-unneeded, GNU objcopy removes all unreferenced local
      // symbols, and any unreferenced undefined external.
      if (Config.StripUnneeded &&
          (Sym.Sym.StorageClass == IMAGE_SYM_CLASS_STATIC ||
           Sym.Sym.SectionNumber == 0))
        return true;

      // GNU objcopy keeps referenced local symbols and external symbols
      // if --discard-all is set, similar to what --strip-unneeded does,
      // but undefined local symbols are kept when --discard-all is set.
      if (Config.DiscardAll && Sym.Sym.StorageClass == IMAGE_SYM_CLASS_STATIC &&
          Sym.Sym.SectionNumber != 0)
        return true;
    }

    return false;
  });
  return Error::success();
}

void executeObjcopyOnBinary(const CopyConfig &Config,
                            COFFObjectFile &In, Buffer &Out) {
  COFFReader Reader(In);
  Expected<std::unique_ptr<Object>> ObjOrErr = Reader.create();
  if (!ObjOrErr)
    reportError(Config.InputFilename, ObjOrErr.takeError());
  Object *Obj = ObjOrErr->get();
  assert(Obj && "Unable to deserialize COFF object");
  if (Error E = handleArgs(Config, *Obj))
    reportError(Config.InputFilename, std::move(E));
  COFFWriter Writer(*Obj, Out);
  if (Error E = Writer.write())
    reportError(Config.OutputFilename, std::move(E));
}

} // end namespace coff
} // end namespace objcopy
} // end namespace llvm
