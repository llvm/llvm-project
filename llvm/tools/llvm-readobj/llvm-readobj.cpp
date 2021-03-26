//===- llvm-readobj.cpp - Dump contents of an Object File -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a tool similar to readelf, except it works on multiple object file
// formats. The main purpose of this tool is to provide detailed output suitable
// for FileCheck.
//
// Flags should be similar to readelf where supported, but the output format
// does not need to be identical. The point is to not make users learn yet
// another set of flags.
//
// Output should be specialized for each format where appropriate.
//
//===----------------------------------------------------------------------===//

#include "llvm-readobj.h"
#include "ObjDumper.h"
#include "WindowsResourceDumper.h"
#include "llvm/DebugInfo/CodeView/GlobalTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/MergingTypeTableBuilder.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Object/WindowsResource.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::object;

namespace opts {
  cl::list<std::string> InputFilenames(cl::Positional,
    cl::desc("<input object files>"),
    cl::ZeroOrMore);

  // --all, -a
  cl::opt<bool>
      All("all",
          cl::desc("Equivalent to setting: --file-headers, --program-headers, "
                   "--section-headers, --symbols, --relocations, "
                   "--dynamic-table, --notes, --version-info, --unwind, "
                   "--section-groups and --elf-hash-histogram."));
  cl::alias AllShort("a", cl::desc("Alias for --all"), cl::aliasopt(All));

  // --dependent-libraries
  cl::opt<bool>
      DependentLibraries("dependent-libraries",
                         cl::desc("Display the dependent libraries section"));

  // --headers, -e
  cl::opt<bool>
      Headers("headers",
          cl::desc("Equivalent to setting: --file-headers, --program-headers, "
                   "--section-headers"));
  cl::alias HeadersShort("e", cl::desc("Alias for --headers"),
     cl::aliasopt(Headers));

  // --wide, -W
  cl::opt<bool>
      WideOutput("wide", cl::desc("Ignored for compatibility with GNU readelf"),
                 cl::Hidden);
  cl::alias WideOutputShort("W",
    cl::desc("Alias for --wide"),
    cl::aliasopt(WideOutput));

  // --file-headers, --file-header, -h
  cl::opt<bool> FileHeaders("file-headers",
    cl::desc("Display file headers "));
  cl::alias FileHeadersShort("h", cl::desc("Alias for --file-headers"),
                             cl::aliasopt(FileHeaders), cl::NotHidden);
  cl::alias FileHeadersSingular("file-header",
                                cl::desc("Alias for --file-headers"),
                                cl::aliasopt(FileHeaders));

  // --section-headers, --sections, -S
  // Also -s in llvm-readobj mode.
  cl::opt<bool> SectionHeaders("section-headers",
                               cl::desc("Display all section headers."));
  cl::alias SectionsShortUpper("S", cl::desc("Alias for --section-headers"),
                               cl::aliasopt(SectionHeaders), cl::NotHidden);
  cl::alias SectionHeadersAlias("sections",
                                cl::desc("Alias for --section-headers"),
                                cl::aliasopt(SectionHeaders), cl::NotHidden);

  // --section-relocations
  // Also --sr in llvm-readobj mode.
  cl::opt<bool> SectionRelocations("section-relocations",
    cl::desc("Display relocations for each section shown."));

  // --section-symbols
  // Also --st in llvm-readobj mode.
  cl::opt<bool> SectionSymbols("section-symbols",
    cl::desc("Display symbols for each section shown."));

  // --section-data
  // Also --sd in llvm-readobj mode.
  cl::opt<bool> SectionData("section-data",
    cl::desc("Display section data for each section shown."));

  // --section-mapping
  cl::opt<cl::boolOrDefault>
      SectionMapping("section-mapping",
                     cl::desc("Display the section to segment mapping."));

  // --relocations, --relocs, -r
  cl::opt<bool> Relocations("relocations",
    cl::desc("Display the relocation entries in the file"));
  cl::alias RelocationsShort("r", cl::desc("Alias for --relocations"),
                             cl::aliasopt(Relocations), cl::NotHidden);
  cl::alias RelocationsGNU("relocs", cl::desc("Alias for --relocations"),
                           cl::aliasopt(Relocations));

  // --notes, -n
  cl::opt<bool> Notes("notes", cl::desc("Display the ELF notes in the file"));
  cl::alias NotesShort("n", cl::desc("Alias for --notes"), cl::aliasopt(Notes));

  // --dyn-relocations
  cl::opt<bool> DynRelocs("dyn-relocations",
    cl::desc("Display the dynamic relocation entries in the file"));

  // --section-details
  // Also -t in llvm-readelf mode.
  cl::opt<bool> SectionDetails("section-details",
                               cl::desc("Display the section details"));

  // --symbols
  // Also -s in llvm-readelf mode, or -t in llvm-readobj mode.
  cl::opt<bool>
      Symbols("symbols",
              cl::desc("Display the symbol table. Also display the dynamic "
                       "symbol table when using GNU output style for ELF"));
  cl::alias SymbolsGNU("syms", cl::desc("Alias for --symbols"),
                       cl::aliasopt(Symbols));

  // --dyn-symbols, --dyn-syms
  // Also --dt in llvm-readobj mode.
  cl::opt<bool> DynamicSymbols("dyn-symbols",
    cl::desc("Display the dynamic symbol table"));
  cl::alias DynSymsGNU("dyn-syms", cl::desc("Alias for --dyn-symbols"),
                       cl::aliasopt(DynamicSymbols));

  // --hash-symbols
  cl::opt<bool> HashSymbols(
      "hash-symbols",
      cl::desc("Display the dynamic symbols derived from the hash section"));

  // --unwind, -u
  cl::opt<bool> UnwindInfo("unwind",
    cl::desc("Display unwind information"));
  cl::alias UnwindInfoShort("u",
    cl::desc("Alias for --unwind"),
    cl::aliasopt(UnwindInfo));

  // --dynamic-table, --dynamic, -d
  cl::opt<bool> DynamicTable("dynamic-table",
    cl::desc("Display the ELF .dynamic section table"));
  cl::alias DynamicTableShort("d", cl::desc("Alias for --dynamic-table"),
                              cl::aliasopt(DynamicTable), cl::NotHidden);
  cl::alias DynamicTableAlias("dynamic", cl::desc("Alias for --dynamic-table"),
                              cl::aliasopt(DynamicTable));

  // --needed-libs
  cl::opt<bool> NeededLibraries("needed-libs",
    cl::desc("Display the needed libraries"));

  // --program-headers, --segments, -l
  cl::opt<bool> ProgramHeaders("program-headers",
    cl::desc("Display ELF program headers"));
  cl::alias ProgramHeadersShort("l", cl::desc("Alias for --program-headers"),
                                cl::aliasopt(ProgramHeaders), cl::NotHidden);
  cl::alias SegmentsAlias("segments", cl::desc("Alias for --program-headers"),
                          cl::aliasopt(ProgramHeaders));

  // --string-dump, -p
  cl::list<std::string> StringDump(
      "string-dump", cl::value_desc("number|name"),
      cl::desc("Display the specified section(s) as a list of strings"),
      cl::ZeroOrMore);
  cl::alias StringDumpShort("p", cl::desc("Alias for --string-dump"),
                            cl::aliasopt(StringDump), cl::Prefix);

  // --hex-dump, -x
  cl::list<std::string>
      HexDump("hex-dump", cl::value_desc("number|name"),
              cl::desc("Display the specified section(s) as hexadecimal bytes"),
              cl::ZeroOrMore);
  cl::alias HexDumpShort("x", cl::desc("Alias for --hex-dump"),
                         cl::aliasopt(HexDump), cl::Prefix);

  // --demangle, -C
  cl::opt<bool> Demangle("demangle",
                         cl::desc("Demangle symbol names in output"));
  cl::alias DemangleShort("C", cl::desc("Alias for --demangle"),
                          cl::aliasopt(Demangle), cl::NotHidden);

  // --hash-table
  cl::opt<bool> HashTable("hash-table",
    cl::desc("Display ELF hash table"));

  // --gnu-hash-table
  cl::opt<bool> GnuHashTable("gnu-hash-table",
    cl::desc("Display ELF .gnu.hash section"));

  // --expand-relocs
  cl::opt<bool> ExpandRelocs("expand-relocs",
    cl::desc("Expand each shown relocation to multiple lines"));

  // --raw-relr
  cl::opt<bool> RawRelr("raw-relr",
    cl::desc("Do not decode relocations in SHT_RELR section, display raw contents"));

  // --codeview
  cl::opt<bool> CodeView("codeview",
                         cl::desc("Display CodeView debug information"));

  // --codeview-merged-types
  cl::opt<bool>
      CodeViewMergedTypes("codeview-merged-types",
                          cl::desc("Display the merged CodeView type stream"));

  // --codeview-ghash
  cl::opt<bool> CodeViewEnableGHash(
      "codeview-ghash",
      cl::desc(
          "Enable global hashing for CodeView type stream de-duplication"));

  // --codeview-subsection-bytes
  cl::opt<bool> CodeViewSubsectionBytes(
      "codeview-subsection-bytes",
      cl::desc("Dump raw contents of codeview debug sections and records"));

  // --arch-specific
  cl::opt<bool> ArchSpecificInfo("arch-specific",
                              cl::desc("Displays architecture-specific information, if there is any."));
  cl::alias ArchSpecifcInfoShort("A", cl::desc("Alias for --arch-specific"),
                                 cl::aliasopt(ArchSpecificInfo), cl::NotHidden);

  // --coff-imports
  cl::opt<bool>
  COFFImports("coff-imports", cl::desc("Display the PE/COFF import table"));

  // --coff-exports
  cl::opt<bool>
  COFFExports("coff-exports", cl::desc("Display the PE/COFF export table"));

  // --coff-directives
  cl::opt<bool>
  COFFDirectives("coff-directives",
                 cl::desc("Display the PE/COFF .drectve section"));

  // --coff-basereloc
  cl::opt<bool>
  COFFBaseRelocs("coff-basereloc",
                 cl::desc("Display the PE/COFF .reloc section"));

  // --coff-debug-directory
  cl::opt<bool>
  COFFDebugDirectory("coff-debug-directory",
                     cl::desc("Display the PE/COFF debug directory"));

  // --coff-tls-directory
  cl::opt<bool> COFFTLSDirectory("coff-tls-directory",
                                 cl::desc("Display the PE/COFF TLS directory"));

  // --coff-resources
  cl::opt<bool> COFFResources("coff-resources",
                              cl::desc("Display the PE/COFF .rsrc section"));

  // --coff-load-config
  cl::opt<bool>
  COFFLoadConfig("coff-load-config",
                 cl::desc("Display the PE/COFF load config"));

  // --elf-linker-options
  cl::opt<bool>
  ELFLinkerOptions("elf-linker-options",
                   cl::desc("Display the ELF .linker-options section"));

  // --macho-data-in-code
  cl::opt<bool>
  MachODataInCode("macho-data-in-code",
                  cl::desc("Display MachO Data in Code command"));

  // --macho-indirect-symbols
  cl::opt<bool>
  MachOIndirectSymbols("macho-indirect-symbols",
                  cl::desc("Display MachO indirect symbols"));

  // --macho-linker-options
  cl::opt<bool>
  MachOLinkerOptions("macho-linker-options",
                  cl::desc("Display MachO linker options"));

  // --macho-segment
  cl::opt<bool>
  MachOSegment("macho-segment",
                  cl::desc("Display MachO Segment command"));

  // --macho-version-min
  cl::opt<bool>
  MachOVersionMin("macho-version-min",
                  cl::desc("Display MachO version min command"));

  // --macho-dysymtab
  cl::opt<bool>
  MachODysymtab("macho-dysymtab",
                  cl::desc("Display MachO Dysymtab command"));

  // --stackmap
  cl::opt<bool>
  PrintStackMap("stackmap",
                cl::desc("Display contents of stackmap section"));

  // --stack-sizes
  cl::opt<bool>
      PrintStackSizes("stack-sizes",
                      cl::desc("Display contents of all stack sizes sections"));

  // --version-info, -V
  cl::opt<bool>
      VersionInfo("version-info",
                  cl::desc("Display ELF version sections (if present)"));
  cl::alias VersionInfoShort("V", cl::desc("Alias for -version-info"),
                             cl::aliasopt(VersionInfo));

  // --elf-section-groups, --section-groups, -g
  cl::opt<bool> SectionGroups("elf-section-groups",
                              cl::desc("Display ELF section group contents"));
  cl::alias SectionGroupsAlias("section-groups",
                               cl::desc("Alias for -elf-sections-groups"),
                               cl::aliasopt(SectionGroups));
  cl::alias SectionGroupsShort("g", cl::desc("Alias for -elf-sections-groups"),
                               cl::aliasopt(SectionGroups));

  // --elf-hash-histogram, --histogram, -I
  cl::opt<bool> HashHistogram(
      "elf-hash-histogram",
      cl::desc("Display bucket list histogram for hash sections"));
  cl::alias HashHistogramShort("I", cl::desc("Alias for -elf-hash-histogram"),
                               cl::aliasopt(HashHistogram));
  cl::alias HistogramAlias("histogram",
                           cl::desc("Alias for --elf-hash-histogram"),
                           cl::aliasopt(HashHistogram));

  // --cg-profile
  cl::opt<bool> CGProfile("cg-profile",
                          cl::desc("Display callgraph profile section"));
  cl::alias ELFCGProfile("elf-cg-profile", cl::desc("Alias for --cg-profile"),
                         cl::aliasopt(CGProfile));

  // --bb-addr-map
  cl::opt<bool> BBAddrMap("bb-addr-map",
                          cl::desc("Display the BB address map section"));

  // -addrsig
  cl::opt<bool> Addrsig("addrsig",
                        cl::desc("Display address-significance table"));

  // -elf-output-style
  cl::opt<OutputStyleTy>
      Output("elf-output-style", cl::desc("Specify ELF dump style"),
             cl::values(clEnumVal(LLVM, "LLVM default style"),
                        clEnumVal(GNU, "GNU readelf style")),
             cl::init(LLVM));

  cl::extrahelp
      HelpResponse("\nPass @FILE as argument to read options from FILE.\n");
} // namespace opts

static StringRef ToolName;

namespace llvm {

LLVM_ATTRIBUTE_NORETURN static void error(Twine Msg) {
  // Flush the standard output to print the error at a
  // proper place.
  fouts().flush();
  WithColor::error(errs(), ToolName) << Msg << "\n";
  exit(1);
}

LLVM_ATTRIBUTE_NORETURN void reportError(Error Err, StringRef Input) {
  assert(Err);
  if (Input == "-")
    Input = "<stdin>";
  handleAllErrors(createFileError(Input, std::move(Err)),
                  [&](const ErrorInfoBase &EI) { error(EI.message()); });
  llvm_unreachable("error() call should never return");
}

void reportWarning(Error Err, StringRef Input) {
  assert(Err);
  if (Input == "-")
    Input = "<stdin>";

  // Flush the standard output to print the warning at a
  // proper place.
  fouts().flush();
  handleAllErrors(
      createFileError(Input, std::move(Err)), [&](const ErrorInfoBase &EI) {
        WithColor::warning(errs(), ToolName) << EI.message() << "\n";
      });
}

} // namespace llvm

namespace {
struct ReadObjTypeTableBuilder {
  ReadObjTypeTableBuilder()
      : Allocator(), IDTable(Allocator), TypeTable(Allocator),
        GlobalIDTable(Allocator), GlobalTypeTable(Allocator) {}

  llvm::BumpPtrAllocator Allocator;
  llvm::codeview::MergingTypeTableBuilder IDTable;
  llvm::codeview::MergingTypeTableBuilder TypeTable;
  llvm::codeview::GlobalTypeTableBuilder GlobalIDTable;
  llvm::codeview::GlobalTypeTableBuilder GlobalTypeTable;
  std::vector<OwningBinary<Binary>> Binaries;
};
} // namespace
static ReadObjTypeTableBuilder CVTypes;

/// Creates an format-specific object file dumper.
static Expected<std::unique_ptr<ObjDumper>>
createDumper(const ObjectFile &Obj, ScopedPrinter &Writer) {
  if (const COFFObjectFile *COFFObj = dyn_cast<COFFObjectFile>(&Obj))
    return createCOFFDumper(*COFFObj, Writer);

  if (const ELFObjectFileBase *ELFObj = dyn_cast<ELFObjectFileBase>(&Obj))
    return createELFDumper(*ELFObj, Writer);

  if (const MachOObjectFile *MachOObj = dyn_cast<MachOObjectFile>(&Obj))
    return createMachODumper(*MachOObj, Writer);

  if (const WasmObjectFile *WasmObj = dyn_cast<WasmObjectFile>(&Obj))
    return createWasmDumper(*WasmObj, Writer);

  if (const XCOFFObjectFile *XObj = dyn_cast<XCOFFObjectFile>(&Obj))
    return createXCOFFDumper(*XObj, Writer);

  return createStringError(errc::invalid_argument,
                           "unsupported object file format");
}

/// Dumps the specified object file.
static void dumpObject(ObjectFile &Obj, ScopedPrinter &Writer,
                       const Archive *A = nullptr) {
  std::string FileStr =
      A ? Twine(A->getFileName() + "(" + Obj.getFileName() + ")").str()
        : Obj.getFileName().str();

  std::string ContentErrString;
  if (Error ContentErr = Obj.initContent())
    ContentErrString = "unable to continue dumping, the file is corrupt: " +
                       toString(std::move(ContentErr));

  ObjDumper *Dumper;
  Expected<std::unique_ptr<ObjDumper>> DumperOrErr = createDumper(Obj, Writer);
  if (!DumperOrErr)
    reportError(DumperOrErr.takeError(), FileStr);
  Dumper = (*DumperOrErr).get();

  if (opts::Output == opts::LLVM || opts::InputFilenames.size() > 1 || A) {
    Writer.startLine() << "\n";
    Writer.printString("File", FileStr);
  }
  if (opts::Output == opts::LLVM) {
    Writer.printString("Format", Obj.getFileFormatName());
    Writer.printString("Arch", Triple::getArchTypeName(Obj.getArch()));
    Writer.printString(
        "AddressSize",
        std::string(formatv("{0}bit", 8 * Obj.getBytesInAddress())));
    Dumper->printLoadName();
  }

  if (opts::FileHeaders)
    Dumper->printFileHeaders();

  // This is only used for ELF currently. In some cases, when an object is
  // corrupt (e.g. truncated), we can't dump anything except the file header.
  if (!ContentErrString.empty())
    reportError(createError(ContentErrString), FileStr);

  if (opts::SectionDetails || opts::SectionHeaders) {
    if (opts::Output == opts::GNU && opts::SectionDetails)
      Dumper->printSectionDetails();
    else
      Dumper->printSectionHeaders();
  }

  if (opts::HashSymbols)
    Dumper->printHashSymbols();
  if (opts::ProgramHeaders || opts::SectionMapping == cl::BOU_TRUE)
    Dumper->printProgramHeaders(opts::ProgramHeaders, opts::SectionMapping);
  if (opts::DynamicTable)
    Dumper->printDynamicTable();
  if (opts::NeededLibraries)
    Dumper->printNeededLibraries();
  if (opts::Relocations)
    Dumper->printRelocations();
  if (opts::DynRelocs)
    Dumper->printDynamicRelocations();
  if (opts::UnwindInfo)
    Dumper->printUnwindInfo();
  if (opts::Symbols || opts::DynamicSymbols)
    Dumper->printSymbols(opts::Symbols, opts::DynamicSymbols);
  if (!opts::StringDump.empty())
    Dumper->printSectionsAsString(Obj, opts::StringDump);
  if (!opts::HexDump.empty())
    Dumper->printSectionsAsHex(Obj, opts::HexDump);
  if (opts::HashTable)
    Dumper->printHashTable();
  if (opts::GnuHashTable)
    Dumper->printGnuHashTable();
  if (opts::VersionInfo)
    Dumper->printVersionInfo();
  if (Obj.isELF()) {
    if (opts::DependentLibraries)
      Dumper->printDependentLibs();
    if (opts::ELFLinkerOptions)
      Dumper->printELFLinkerOptions();
    if (opts::ArchSpecificInfo)
      Dumper->printArchSpecificInfo();
    if (opts::SectionGroups)
      Dumper->printGroupSections();
    if (opts::HashHistogram)
      Dumper->printHashHistograms();
    if (opts::CGProfile)
      Dumper->printCGProfile();
    if (opts::BBAddrMap)
      Dumper->printBBAddrMaps();
    if (opts::Addrsig)
      Dumper->printAddrsig();
    if (opts::Notes)
      Dumper->printNotes();
  }
  if (Obj.isCOFF()) {
    if (opts::COFFImports)
      Dumper->printCOFFImports();
    if (opts::COFFExports)
      Dumper->printCOFFExports();
    if (opts::COFFDirectives)
      Dumper->printCOFFDirectives();
    if (opts::COFFBaseRelocs)
      Dumper->printCOFFBaseReloc();
    if (opts::COFFDebugDirectory)
      Dumper->printCOFFDebugDirectory();
    if (opts::COFFTLSDirectory)
      Dumper->printCOFFTLSDirectory();
    if (opts::COFFResources)
      Dumper->printCOFFResources();
    if (opts::COFFLoadConfig)
      Dumper->printCOFFLoadConfig();
    if (opts::CGProfile)
      Dumper->printCGProfile();
    if (opts::Addrsig)
      Dumper->printAddrsig();
    if (opts::CodeView)
      Dumper->printCodeViewDebugInfo();
    if (opts::CodeViewMergedTypes)
      Dumper->mergeCodeViewTypes(CVTypes.IDTable, CVTypes.TypeTable,
                                 CVTypes.GlobalIDTable, CVTypes.GlobalTypeTable,
                                 opts::CodeViewEnableGHash);
  }
  if (Obj.isMachO()) {
    if (opts::MachODataInCode)
      Dumper->printMachODataInCode();
    if (opts::MachOIndirectSymbols)
      Dumper->printMachOIndirectSymbols();
    if (opts::MachOLinkerOptions)
      Dumper->printMachOLinkerOptions();
    if (opts::MachOSegment)
      Dumper->printMachOSegment();
    if (opts::MachOVersionMin)
      Dumper->printMachOVersionMin();
    if (opts::MachODysymtab)
      Dumper->printMachODysymtab();
  }
  if (opts::PrintStackMap)
    Dumper->printStackMap();
  if (opts::PrintStackSizes)
    Dumper->printStackSizes();
}

/// Dumps each object file in \a Arc;
static void dumpArchive(const Archive *Arc, ScopedPrinter &Writer) {
  Error Err = Error::success();
  for (auto &Child : Arc->children(Err)) {
    Expected<std::unique_ptr<Binary>> ChildOrErr = Child.getAsBinary();
    if (!ChildOrErr) {
      if (auto E = isNotObjectErrorInvalidFileType(ChildOrErr.takeError()))
        reportError(std::move(E), Arc->getFileName());
      continue;
    }

    Binary *Bin = ChildOrErr->get();
    if (ObjectFile *Obj = dyn_cast<ObjectFile>(Bin))
      dumpObject(*Obj, Writer, Arc);
    else if (COFFImportFile *Imp = dyn_cast<COFFImportFile>(Bin))
      dumpCOFFImportFile(Imp, Writer);
    else
      reportWarning(createStringError(errc::invalid_argument,
                                      Bin->getFileName() +
                                          " has an unsupported file type"),
                    Arc->getFileName());
  }
  if (Err)
    reportError(std::move(Err), Arc->getFileName());
}

/// Dumps each object file in \a MachO Universal Binary;
static void dumpMachOUniversalBinary(const MachOUniversalBinary *UBinary,
                                     ScopedPrinter &Writer) {
  for (const MachOUniversalBinary::ObjectForArch &Obj : UBinary->objects()) {
    Expected<std::unique_ptr<MachOObjectFile>> ObjOrErr = Obj.getAsObjectFile();
    if (ObjOrErr)
      dumpObject(*ObjOrErr.get(), Writer);
    else if (auto E = isNotObjectErrorInvalidFileType(ObjOrErr.takeError()))
      reportError(ObjOrErr.takeError(), UBinary->getFileName());
    else if (Expected<std::unique_ptr<Archive>> AOrErr = Obj.getAsArchive())
      dumpArchive(&*AOrErr.get(), Writer);
  }
}

/// Dumps \a WinRes, Windows Resource (.res) file;
static void dumpWindowsResourceFile(WindowsResource *WinRes,
                                    ScopedPrinter &Printer) {
  WindowsRes::Dumper Dumper(WinRes, Printer);
  if (auto Err = Dumper.printData())
    reportError(std::move(Err), WinRes->getFileName());
}


/// Opens \a File and dumps it.
static void dumpInput(StringRef File, ScopedPrinter &Writer) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(File, /*IsText=*/false,
                                   /*RequiresNullTerminator=*/false);
  if (std::error_code EC = FileOrErr.getError())
    return reportError(errorCodeToError(EC), File);

  std::unique_ptr<MemoryBuffer> &Buffer = FileOrErr.get();
  file_magic Type = identify_magic(Buffer->getBuffer());
  if (Type == file_magic::bitcode) {
    reportWarning(createStringError(errc::invalid_argument,
                                    "bitcode files are not supported"),
                  File);
    return;
  }

  Expected<std::unique_ptr<Binary>> BinaryOrErr = createBinary(
      Buffer->getMemBufferRef(), /*Context=*/nullptr, /*InitContent=*/false);
  if (!BinaryOrErr)
    reportError(BinaryOrErr.takeError(), File);

  std::unique_ptr<Binary> Bin = std::move(*BinaryOrErr);
  if (Archive *Arc = dyn_cast<Archive>(Bin.get()))
    dumpArchive(Arc, Writer);
  else if (MachOUniversalBinary *UBinary =
               dyn_cast<MachOUniversalBinary>(Bin.get()))
    dumpMachOUniversalBinary(UBinary, Writer);
  else if (ObjectFile *Obj = dyn_cast<ObjectFile>(Bin.get()))
    dumpObject(*Obj, Writer);
  else if (COFFImportFile *Import = dyn_cast<COFFImportFile>(Bin.get()))
    dumpCOFFImportFile(Import, Writer);
  else if (WindowsResource *WinRes = dyn_cast<WindowsResource>(Bin.get()))
    dumpWindowsResourceFile(WinRes, Writer);
  else
    llvm_unreachable("unrecognized file type");

  CVTypes.Binaries.push_back(
      OwningBinary<Binary>(std::move(Bin), std::move(Buffer)));
}

/// Registers aliases that should only be allowed by readobj.
static void registerReadobjAliases() {
  // -s has meant --sections for a very long time in llvm-readobj despite
  // meaning --symbols in readelf.
  static cl::alias SectionsShort("s", cl::desc("Alias for --section-headers"),
                                 cl::aliasopt(opts::SectionHeaders),
                                 cl::NotHidden);

  // llvm-readelf reserves it for --section-details.
  static cl::alias SymbolsShort("t", cl::desc("Alias for --symbols"),
                                cl::aliasopt(opts::Symbols), cl::NotHidden);

  // The following two-letter aliases are only provided for readobj, as readelf
  // allows single-letter args to be grouped together.
  static cl::alias SectionRelocationsShort(
      "sr", cl::desc("Alias for --section-relocations"),
      cl::aliasopt(opts::SectionRelocations));
  static cl::alias SectionDataShort("sd", cl::desc("Alias for --section-data"),
                                    cl::aliasopt(opts::SectionData));
  static cl::alias SectionSymbolsShort("st",
                                       cl::desc("Alias for --section-symbols"),
                                       cl::aliasopt(opts::SectionSymbols));
  static cl::alias DynamicSymbolsShort("dt",
                                       cl::desc("Alias for --dyn-symbols"),
                                       cl::aliasopt(opts::DynamicSymbols));
}

/// Registers aliases that should only be allowed by readelf.
static void registerReadelfAliases() {
  // -s is here because for readobj it means --sections.
  static cl::alias SymbolsShort("s", cl::desc("Alias for --symbols"),
                                cl::aliasopt(opts::Symbols), cl::NotHidden,
                                cl::Grouping);

  // -t is here because for readobj it is an alias for --symbols.
  static cl::alias SectionDetailsShort(
      "t", cl::desc("Alias for --section-details"),
      cl::aliasopt(opts::SectionDetails), cl::NotHidden);

  // Allow all single letter flags to be grouped together.
  for (auto &OptEntry : cl::getRegisteredOptions()) {
    StringRef ArgName = OptEntry.getKey();
    cl::Option *Option = OptEntry.getValue();
    if (ArgName.size() == 1)
      apply(Option, cl::Grouping);
  }
}

int main(int argc, const char *argv[]) {
  InitLLVM X(argc, argv);
  ToolName = argv[0];

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  if (sys::path::stem(argv[0]).contains("readelf")) {
    opts::Output = opts::GNU;
    registerReadelfAliases();
  } else {
    registerReadobjAliases();
  }

  cl::ParseCommandLineOptions(argc, argv, "LLVM Object Reader\n");

  // Default to print error if no filename is specified.
  if (opts::InputFilenames.empty()) {
    error("no input files specified");
  }

  if (opts::All) {
    opts::FileHeaders = true;
    opts::ProgramHeaders = true;
    opts::SectionHeaders = true;
    opts::Symbols = true;
    opts::Relocations = true;
    opts::DynamicTable = true;
    opts::Notes = true;
    opts::VersionInfo = true;
    opts::UnwindInfo = true;
    opts::SectionGroups = true;
    opts::HashHistogram = true;
    if (opts::Output == opts::LLVM) {
      opts::Addrsig = true;
      opts::PrintStackSizes = true;
    }
  }

  if (opts::Headers) {
    opts::FileHeaders = true;
    opts::ProgramHeaders = true;
    opts::SectionHeaders = true;
  }

  ScopedPrinter Writer(fouts());
  for (const std::string &I : opts::InputFilenames)
    dumpInput(I, Writer);

  if (opts::CodeViewMergedTypes) {
    if (opts::CodeViewEnableGHash)
      dumpCodeViewMergedTypes(Writer, CVTypes.GlobalIDTable.records(),
                              CVTypes.GlobalTypeTable.records());
    else
      dumpCodeViewMergedTypes(Writer, CVTypes.IDTable.records(),
                              CVTypes.TypeTable.records());
  }

  return 0;
}
