//===-- llvm-lto2: test harness for the resolution-based LTO interface ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program takes in a list of bitcode files, links them and performs
// link-time optimization according to the provided symbol resolutions using the
// resolution-based LTO interface, and outputs one or more object files.
//
// This program is intended to eventually replace llvm-lto which uses the legacy
// LTO interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTO.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace lto;
using namespace object;

static cl::list<std::string> InputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input bitcode files>"));

static cl::opt<std::string> OutputFilename("o", cl::Required,
                                           cl::desc("Output filename"),
                                           cl::value_desc("filename"));

static cl::opt<bool> SaveTemps("save-temps", cl::desc("Save temporary files"));

static cl::list<std::string> SymbolResolutions(
    "r",
    cl::desc("Specify a symbol resolution: filename,symbolname,resolution\n"
             "where \"resolution\" is a sequence (which may be empty) of the\n"
             "following characters:\n"
             " p - prevailing: the linker has chosen this definition of the\n"
             "     symbol\n"
             " l - local: the definition of this symbol is unpreemptable at\n"
             "     runtime and is known to be in this linkage unit\n"
             " x - externally visible: the definition of this symbol is\n"
             "     visible outside of the LTO unit\n"
             "A resolution for each symbol must be specified."),
    cl::ZeroOrMore);

static void check(Error E, std::string Msg) {
  if (!E)
    return;
  handleAllErrors(std::move(E), [&](ErrorInfoBase &EIB) {
    errs() << "llvm-lto: " << Msg << ": " << EIB.message().c_str() << '\n';
  });
  exit(1);
}

template <typename T> static T check(Expected<T> E, std::string Msg) {
  if (E)
    return std::move(*E);
  check(E.takeError(), Msg);
  return T();
}

static void check(std::error_code EC, std::string Msg) {
  check(errorCodeToError(EC), Msg);
}

template <typename T> static T check(ErrorOr<T> E, std::string Msg) {
  if (E)
    return std::move(*E);
  check(E.getError(), Msg);
  return T();
}

int main(int argc, char **argv) {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  cl::ParseCommandLineOptions(argc, argv, "Resolution-based LTO test harness");

  std::map<std::pair<std::string, std::string>, SymbolResolution>
      CommandLineResolutions;
  for (std::string R : SymbolResolutions) {
    StringRef Rest = R;
    StringRef FileName, SymbolName;
    std::tie(FileName, Rest) = Rest.split(',');
    if (Rest.empty()) {
      llvm::errs() << "invalid resolution: " << R << '\n';
      return 1;
    }
    std::tie(SymbolName, Rest) = Rest.split(',');
    SymbolResolution Res;
    for (char C : Rest) {
      if (C == 'p')
        Res.Prevailing = true;
      else if (C == 'l')
        Res.FinalDefinitionInLinkageUnit = true;
      else if (C == 'x')
        Res.VisibleToRegularObj = true;
      else
        llvm::errs() << "invalid character " << C << " in resolution: " << R
                     << '\n';
    }
    CommandLineResolutions[{FileName, SymbolName}] = Res;
  }

  std::vector<std::unique_ptr<MemoryBuffer>> MBs;

  Config Conf;
  Conf.DiagHandler = [](const DiagnosticInfo &) {
    exit(1);
  };

  if (SaveTemps)
    check(Conf.addSaveTemps(OutputFilename), "Config::addSaveTemps failed");

  LTO Lto(std::move(Conf));

  bool HasErrors = false;
  for (std::string F : InputFilenames) {
    std::unique_ptr<MemoryBuffer> MB = check(MemoryBuffer::getFile(F), F);
    std::unique_ptr<InputFile> Input =
        check(InputFile::create(MB->getMemBufferRef()), F);

    std::vector<SymbolResolution> Res;
    for (const InputFile::Symbol &Sym : Input->symbols()) {
      auto I = CommandLineResolutions.find({F, Sym.getName()});
      if (I == CommandLineResolutions.end()) {
        llvm::errs() << argv[0] << ": missing symbol resolution for " << F
                     << ',' << Sym.getName() << '\n';
        HasErrors = true;
      } else {
        Res.push_back(I->second);
        CommandLineResolutions.erase(I);
      }
    }

    if (HasErrors)
      continue;

    MBs.push_back(std::move(MB));
    check(Lto.add(std::move(Input), Res), F);
  }

  if (!CommandLineResolutions.empty()) {
    HasErrors = true;
    for (auto UnusedRes : CommandLineResolutions)
      llvm::errs() << argv[0] << ": unused symbol resolution for "
                   << UnusedRes.first.first << ',' << UnusedRes.first.second
                   << '\n';
  }
  if (HasErrors)
    return 1;

  auto AddStream = [&](size_t Task) {
    std::string Path = OutputFilename + "." + utostr(Task);
    std::error_code EC;
    auto S = llvm::make_unique<raw_fd_ostream>(Path, EC, sys::fs::F_None);
    check(EC, Path);
    return S;
  };

  check(Lto.run(AddStream), "LTO::run failed");
}
