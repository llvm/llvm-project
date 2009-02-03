//===-- OcamlGCPrinter.cpp - Ocaml frametable emitter ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements printing the assembly code for an Ocaml frametable.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCs.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/GCMetadataPrinter.h"
#include "llvm/Module.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {

  class VISIBILITY_HIDDEN OcamlGCMetadataPrinter : public GCMetadataPrinter {
  public:
    void beginAssembly(raw_ostream &OS, AsmPrinter &AP,
                       const TargetAsmInfo &TAI);

    void finishAssembly(raw_ostream &OS, AsmPrinter &AP,
                        const TargetAsmInfo &TAI);
  };

}

static GCMetadataPrinterRegistry::Add<OcamlGCMetadataPrinter>
Y("ocaml", "ocaml 3.10-compatible collector");

void llvm::linkOcamlGCPrinter() { }

static void EmitCamlGlobal(const Module &M, raw_ostream &OS, AsmPrinter &AP,
                           const TargetAsmInfo &TAI, const char *Id) {
  const std::string &MId = M.getModuleIdentifier();

  std::string Mangled;
  Mangled += TAI.getGlobalPrefix();
  Mangled += "caml";
  size_t Letter = Mangled.size();
  Mangled.append(MId.begin(), std::find(MId.begin(), MId.end(), '.'));
  Mangled += "__";
  Mangled += Id;

  // Capitalize the first letter of the module name.
  Mangled[Letter] = toupper(Mangled[Letter]);

  if (const char *GlobalDirective = TAI.getGlobalDirective())
    OS << GlobalDirective << Mangled << "\n";
  OS << Mangled << ":\n";
}

void OcamlGCMetadataPrinter::beginAssembly(raw_ostream &OS, AsmPrinter &AP,
                                           const TargetAsmInfo &TAI) {
  AP.SwitchToSection(TAI.getTextSection());
  EmitCamlGlobal(getModule(), OS, AP, TAI, "code_begin");

  AP.SwitchToSection(TAI.getDataSection());
  EmitCamlGlobal(getModule(), OS, AP, TAI, "data_begin");
}

/// emitAssembly - Print the frametable. The ocaml frametable format is thus:
///
///   extern "C" struct align(sizeof(intptr_t)) {
///     uint16_t NumDescriptors;
///     struct align(sizeof(intptr_t)) {
///       void *ReturnAddress;
///       uint16_t FrameSize;
///       uint16_t NumLiveOffsets;
///       uint16_t LiveOffsets[NumLiveOffsets];
///     } Descriptors[NumDescriptors];
///   } caml${module}__frametable;
///
/// Note that this precludes programs from stack frames larger than 64K
/// (FrameSize and LiveOffsets would overflow). FrameTablePrinter will abort if
/// either condition is detected in a function which uses the GC.
///
void OcamlGCMetadataPrinter::finishAssembly(raw_ostream &OS, AsmPrinter &AP,
                                            const TargetAsmInfo &TAI) {
  const char *AddressDirective;
  int AddressAlignLog;
  if (AP.TM.getTargetData()->getPointerSize() == sizeof(int32_t)) {
    AddressDirective = TAI.getData32bitsDirective();
    AddressAlignLog = 2;
  } else {
    AddressDirective = TAI.getData64bitsDirective();
    AddressAlignLog = 3;
  }

  AP.SwitchToSection(TAI.getTextSection());
  EmitCamlGlobal(getModule(), OS, AP, TAI, "code_end");

  AP.SwitchToSection(TAI.getDataSection());
  EmitCamlGlobal(getModule(), OS, AP, TAI, "data_end");

  OS << AddressDirective << 0; // FIXME: Why does ocaml emit this??
  AP.EOL();

  AP.SwitchToSection(TAI.getDataSection());
  EmitCamlGlobal(getModule(), OS, AP, TAI, "frametable");

  for (iterator I = begin(), IE = end(); I != IE; ++I) {
    GCFunctionInfo &FI = **I;

    uint64_t FrameSize = FI.getFrameSize();
    if (FrameSize >= 1<<16) {
      cerr << "Function '" << FI.getFunction().getNameStart()
           << "' is too large for the ocaml GC! "
           << "Frame size " << FrameSize << " >= 65536.\n";
      cerr << "(" << uintptr_t(&FI) << ")\n";
      abort(); // Very rude!
    }

    OS << "\t" << TAI.getCommentString() << " live roots for "
       << FI.getFunction().getNameStart() << "\n";

    for (GCFunctionInfo::iterator J = FI.begin(), JE = FI.end(); J != JE; ++J) {
      size_t LiveCount = FI.live_size(J);
      if (LiveCount >= 1<<16) {
        cerr << "Function '" << FI.getFunction().getNameStart()
             << "' is too large for the ocaml GC! "
             << "Live root count " << LiveCount << " >= 65536.\n";
        abort(); // Very rude!
      }

      OS << AddressDirective
         << TAI.getPrivateGlobalPrefix() << "label" << J->Num;
      AP.EOL("call return address");

      AP.EmitInt16(FrameSize);
      AP.EOL("stack frame size");

      AP.EmitInt16(LiveCount);
      AP.EOL("live root count");

      for (GCFunctionInfo::live_iterator K = FI.live_begin(J),
                                         KE = FI.live_end(J); K != KE; ++K) {
        assert(K->StackOffset < 1<<16 &&
               "GC root stack offset is outside of fixed stack frame and out "
               "of range for ocaml GC!");

        OS << "\t.word\t" << K->StackOffset;
        AP.EOL("stack offset");
      }

      AP.EmitAlignment(AddressAlignLog);
    }
  }
}
