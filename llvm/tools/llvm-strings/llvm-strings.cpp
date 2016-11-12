//===-- llvm-strings.cpp - Printable String dumping utility ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like binutils "strings", that is, it
// prints out printable strings in a binary, objdump, or archive file.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include <cctype>
#include <string>

using namespace llvm;
using namespace llvm::object;

static cl::list<std::string> InputFileNames(cl::Positional,
                                            cl::desc("<input object files>"),
                                            cl::ZeroOrMore);

static void dump(raw_ostream &OS, StringRef Contents) {
  const char *P = nullptr, *E = nullptr, *S = nullptr;
  for (P = Contents.begin(), E = Contents.end(); P < E; ++P) {
    if (std::isgraph(*P) || std::isblank(*P)) {
      if (S == nullptr)
        S = P;
    } else if (S) {
      if (P - S > 3)
        OS << StringRef(S, P - S) << '\n';
      S = nullptr;
    }
  }
  if (S && E - S > 3)
    OS << StringRef(S, E - S) << '\n';
}

namespace {
class Strings {
  LLVMContext Context;
  raw_ostream &OS;

  void dump(const ObjectFile *O) {
    for (const auto &S : O->sections()) {
      StringRef Contents;
      if (!S.getContents(Contents))
        ::dump(OS, Contents);
    }
  }

  void dump(const Archive *A) {
    Error E = Error::success();
    for (auto &Element : A->children(E)) {
      if (Expected<std::unique_ptr<Binary>> Child =
              Element.getAsBinary(&Context)) {
        dump(dyn_cast<ObjectFile>(&**Child));
      } else {
        if (auto E = isNotObjectErrorInvalidFileType(Child.takeError())) {
          errs() << A->getFileName();
          if (Expected<StringRef> Name = Element.getName())
            errs() << '(' << *Name << ')';
          logAllUnhandledErrors(std::move(E), errs(), "");
          errs() << '\n';
        }
      }
    }
    (void)static_cast<bool>(E);
  }

public:
  Strings(raw_ostream &S) : OS(S) {}

  void scan(StringRef File) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
        MemoryBuffer::getFileOrSTDIN(File);
    if (std::error_code EC = Buffer.getError()) {
      errs() << File << ": " << EC.message() << '\n';
      return;
    }

    if (Expected<std::unique_ptr<Binary>> B =
            createBinary(Buffer.get()->getMemBufferRef(), &Context)) {
      if (auto *A = dyn_cast<Archive>(&**B))
        return dump(A);
      if (auto *O = dyn_cast<ObjectFile>(&**B))
        return dump(O);
      ::dump(OS, Buffer.get()->getMemBufferRef().getBuffer());
    } else {
      consumeError(B.takeError());
      ::dump(OS, Buffer.get()->getMemBufferRef().getBuffer());
    }
  }
};
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "llvm string dumper\n");

  if (InputFileNames.empty())
    InputFileNames.push_back("-");

  Strings S(llvm::outs());
  std::for_each(InputFileNames.begin(), InputFileNames.end(),
                [&S](StringRef F) { S.scan(F); });
  return EXIT_SUCCESS;
}

