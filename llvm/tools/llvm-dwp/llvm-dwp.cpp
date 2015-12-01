#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Options.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/TargetSelect.h"
#include <memory>
#include <list>
#include <unordered_set>

using namespace llvm;
using namespace cl;

OptionCategory DwpCategory("Specific Options");
static list<std::string> InputFiles(Positional, OneOrMore,
                                    desc("<input files>"), cat(DwpCategory));

static opt<std::string> OutputFilename(Required, "o", desc("Specify the output file."),
                                      value_desc("filename"), cat(DwpCategory));

static int error(const Twine &Error, const Twine &Context) {
  errs() << Twine("while processing ") + Context + ":\n";
  errs() << Twine("error: ") + Error + "\n";
  return 1;
}

static std::error_code writeSection(MCStreamer &Out, MCSection *OutSection,
                                    const object::SectionRef &Sym) {
  StringRef Contents;
  if (auto Err = Sym.getContents(Contents))
    return Err;
  Out.SwitchSection(OutSection);
  Out.EmitBytes(Contents);
  return std::error_code();
}

static std::error_code write(MCStreamer &Out, ArrayRef<std::string> Inputs) {
  for (const auto &Input : Inputs) {
    auto ErrOrObj = object::ObjectFile::createObjectFile(Input);
    if (!ErrOrObj)
      return ErrOrObj.getError();
    const auto *Obj = ErrOrObj->getBinary();
    for (const auto &Section : Obj->sections()) {
      const auto &MCOFI = *Out.getContext().getObjectFileInfo();
      static const StringMap<MCSection *> KnownSections = {
          {"debug_info.dwo", MCOFI.getDwarfInfoDWOSection()},
          {"debug_types.dwo", MCOFI.getDwarfTypesDWOSection()},
          {"debug_str_offsets.dwo", MCOFI.getDwarfStrOffDWOSection()},
          {"debug_str.dwo", MCOFI.getDwarfStrDWOSection()},
          {"debug_loc.dwo", MCOFI.getDwarfLocDWOSection()},
          {"debug_abbrev.dwo", MCOFI.getDwarfAbbrevDWOSection()}};
      StringRef Name;
      if (std::error_code Err = Section.getName(Name))
        return Err;
      if (MCSection *OutSection =
              KnownSections.lookup(Name.substr(Name.find_first_not_of("._"))))
        if (auto Err = writeSection(Out, OutSection, Section))
          return Err;
    }
  }
  return std::error_code();
}

int main(int argc, char** argv) {

  ParseCommandLineOptions(argc, argv, "merge split dwarf (.dwo) files");

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  std::string ErrorStr;
  StringRef Context = "dwarf streamer init";

  Triple TheTriple("x86_64-linux-gnu");

  // Get the target.
  const Target *TheTarget =
      TargetRegistry::lookupTarget("", TheTriple, ErrorStr);
  if (!TheTarget)
    return error(ErrorStr, Context);
  std::string TripleName = TheTriple.getTriple();

  // Create all the MC Objects.
  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    return error(Twine("no register info for target ") + TripleName, Context);

  std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TripleName));
  if (!MAI)
    return error("no asm info for target " + TripleName, Context);

  MCObjectFileInfo MOFI;
  MCContext MC(MAI.get(), MRI.get(), &MOFI);
  MOFI.InitMCObjectFileInfo(TheTriple, Reloc::Default, CodeModel::Default,
                             MC);

  auto MAB = TheTarget->createMCAsmBackend(*MRI, TripleName, "");
  if (!MAB)
    return error("no asm backend for target " + TripleName, Context);

  std::unique_ptr<MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    return error("no instr info info for target " + TripleName, Context);

  std::unique_ptr<MCSubtargetInfo> MSTI(
      TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!MSTI)
    return error("no subtarget info for target " + TripleName, Context);

  MCCodeEmitter *MCE = TheTarget->createMCCodeEmitter(*MII, *MRI, MC);
  if (!MCE)
    return error("no code emitter for target " + TripleName, Context);

  // Create the output file.
  std::error_code EC;
  raw_fd_ostream OutFile(OutputFilename, EC, sys::fs::F_None);
  if (EC)
    return error(Twine(OutputFilename) + ": " + EC.message(), Context);

  std::unique_ptr<MCStreamer> MS(TheTarget->createMCObjectStreamer(
      TheTriple, MC, *MAB, OutFile, MCE, *MSTI, false,
      /*DWARFMustBeAtTheEnd*/ false));
  if (!MS)
    return error("no object streamer for target " + TripleName, Context);

  if (auto Err = write(*MS, InputFiles))
    return error(Err.message(), "Writing DWP file");

  MS->Finish();
}
