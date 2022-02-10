//===-- llvm-dwp.cpp - Split DWARF merging tool for llvm ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility for merging DWARF 5 Split DWARF .dwo files into .dwp (DWARF
// package files).
//
//===----------------------------------------------------------------------===//
#include "llvm/DWP/DWP.h"
#include "llvm/DWP/DWPError.h"
#include "llvm/DWP/DWPStringPool.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace llvm::object;

static mc::RegisterMCTargetOptionsFlags MCTargetOptionsFlags;

cl::OptionCategory DwpCategory("Specific Options");
static cl::list<std::string> InputFiles(cl::Positional, cl::ZeroOrMore,
                                        cl::desc("<input files>"),
                                        cl::cat(DwpCategory));

static cl::list<std::string> ExecFilenames(
    "e", cl::ZeroOrMore,
    cl::desc("Specify the executable/library files to get the list of *.dwo from"),
    cl::value_desc("filename"), cl::cat(DwpCategory));

static cl::opt<std::string> OutputFilename(cl::Required, "o",
                                           cl::desc("Specify the output file."),
                                           cl::value_desc("filename"),
                                           cl::cat(DwpCategory));

static Expected<SmallVector<std::string, 16>>
getDWOFilenames(StringRef ExecFilename) {
  auto ErrOrObj = object::ObjectFile::createObjectFile(ExecFilename);
  if (!ErrOrObj)
    return ErrOrObj.takeError();

  const ObjectFile &Obj = *ErrOrObj.get().getBinary();
  std::unique_ptr<DWARFContext> DWARFCtx = DWARFContext::create(Obj);

  SmallVector<std::string, 16> DWOPaths;
  for (const auto &CU : DWARFCtx->compile_units()) {
    const DWARFDie &Die = CU->getUnitDIE();
    std::string DWOName = dwarf::toString(
        Die.find({dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}), "");
    if (DWOName.empty())
      continue;
    std::string DWOCompDir =
        dwarf::toString(Die.find(dwarf::DW_AT_comp_dir), "");
    if (!DWOCompDir.empty()) {
      SmallString<16> DWOPath(std::move(DWOName));
      sys::fs::make_absolute(DWOCompDir, DWOPath);
      DWOPaths.emplace_back(DWOPath.data(), DWOPath.size());
    } else {
      DWOPaths.push_back(std::move(DWOName));
    }
  }
  return std::move(DWOPaths);
}

static int error(const Twine &Error, const Twine &Context) {
  errs() << Twine("while processing ") + Context + ":\n";
  errs() << Twine("error: ") + Error + "\n";
  return 1;
}

static Expected<Triple> readTargetTriple(StringRef FileName) {
  auto ErrOrObj = object::ObjectFile::createObjectFile(FileName);
  if (!ErrOrObj)
    return ErrOrObj.takeError();

  return ErrOrObj->getBinary()->makeTriple();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  cl::HideUnrelatedOptions({&DwpCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "merge split dwarf (.dwo) files\n");

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  std::vector<std::string> DWOFilenames = InputFiles;
  for (const auto &ExecFilename : ExecFilenames) {
    auto DWOs = getDWOFilenames(ExecFilename);
    if (!DWOs) {
      logAllUnhandledErrors(DWOs.takeError(), WithColor::error());
      return 1;
    }
    DWOFilenames.insert(DWOFilenames.end(),
                        std::make_move_iterator(DWOs->begin()),
                        std::make_move_iterator(DWOs->end()));
  }

  if (DWOFilenames.empty())
    return 0;

  std::string ErrorStr;
  StringRef Context = "dwarf streamer init";

  auto ErrOrTriple = readTargetTriple(DWOFilenames.front());
  if (!ErrOrTriple) {
    logAllUnhandledErrors(ErrOrTriple.takeError(), WithColor::error());
    return 1;
  }

  // Get the target.
  const Target *TheTarget =
      TargetRegistry::lookupTarget("", *ErrOrTriple, ErrorStr);
  if (!TheTarget)
    return error(ErrorStr, Context);
  std::string TripleName = ErrOrTriple->getTriple();

  // Create all the MC Objects.
  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    return error(Twine("no register info for target ") + TripleName, Context);

  MCTargetOptions MCOptions = llvm::mc::InitMCTargetOptionsFromFlags();
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!MAI)
    return error("no asm info for target " + TripleName, Context);

  std::unique_ptr<MCSubtargetInfo> MSTI(
      TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return error("no subtarget info for target " + TripleName, Context);

  MCContext MC(*ErrOrTriple, MAI.get(), MRI.get(), MSTI.get());
  std::unique_ptr<MCObjectFileInfo> MOFI(
      TheTarget->createMCObjectFileInfo(MC, /*PIC=*/false));
  MC.setObjectFileInfo(MOFI.get());

  MCTargetOptions Options;
  auto MAB = TheTarget->createMCAsmBackend(*MSTI, *MRI, Options);
  if (!MAB)
    return error("no asm backend for target " + TripleName, Context);

  std::unique_ptr<MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    return error("no instr info info for target " + TripleName, Context);

  MCCodeEmitter *MCE = TheTarget->createMCCodeEmitter(*MII, *MRI, MC);
  if (!MCE)
    return error("no code emitter for target " + TripleName, Context);

  // Create the output file.
  std::error_code EC;
  ToolOutputFile OutFile(OutputFilename, EC, sys::fs::OF_None);
  Optional<buffer_ostream> BOS;
  raw_pwrite_stream *OS;
  if (EC)
    return error(Twine(OutputFilename) + ": " + EC.message(), Context);
  if (OutFile.os().supportsSeeking()) {
    OS = &OutFile.os();
  } else {
    BOS.emplace(OutFile.os());
    OS = BOS.getPointer();
  }

  std::unique_ptr<MCStreamer> MS(TheTarget->createMCObjectStreamer(
      *ErrOrTriple, MC, std::unique_ptr<MCAsmBackend>(MAB),
      MAB->createObjectWriter(*OS), std::unique_ptr<MCCodeEmitter>(MCE), *MSTI,
      MCOptions.MCRelaxAll, MCOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));
  if (!MS)
    return error("no object streamer for target " + TripleName, Context);

  if (auto Err = write(*MS, DWOFilenames)) {
    logAllUnhandledErrors(std::move(Err), WithColor::error());
    return 1;
  }

  MS->Finish();
  OutFile.keep();
  return 0;
}
