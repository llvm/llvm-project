//===-- llvm-ar.cpp - LLVM archive librarian utility ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Builds up (relatively) standard unix archive files (.a) containing LLVM
// bitcode or other files.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/LibDriver/LibDriver.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <memory>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using namespace llvm;

// The name this program was invoked as.
static StringRef ToolName;

// Show the error message and exit.
LLVM_ATTRIBUTE_NORETURN static void fail(Twine Error) {
  outs() << ToolName << ": " << Error << ".\n";
  exit(1);
}

static void failIfError(std::error_code EC, Twine Context = "") {
  if (!EC)
    return;

  std::string ContextStr = Context.str();
  if (ContextStr == "")
    fail(EC.message());
  fail(Context + ": " + EC.message());
}

static void failIfError(Error E, Twine Context = "") {
  if (!E)
    return;

  handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EIB) {
    std::string ContextStr = Context.str();
    if (ContextStr == "")
      fail(EIB.message());
    fail(Context + ": " + EIB.message());
  });
}

// llvm-ar/llvm-ranlib remaining positional arguments.
static cl::list<std::string>
    RestOfArgs(cl::Positional, cl::ZeroOrMore,
               cl::desc("[relpos] [count] <archive-file> [members]..."));

static cl::opt<bool> MRI("M", cl::desc(""));
static cl::opt<std::string> Plugin("plugin", cl::desc("plugin (ignored for compatibility"));

namespace {
enum Format { Default, GNU, BSD };
}

static cl::opt<Format>
    FormatOpt("format", cl::desc("Archive format to create"),
              cl::values(clEnumValN(Default, "default", "default"),
                         clEnumValN(GNU, "gnu", "gnu"),
                         clEnumValN(BSD, "bsd", "bsd")));

static std::string Options;

// Provide additional help output explaining the operations and modifiers of
// llvm-ar. This object instructs the CommandLine library to print the text of
// the constructor when the --help option is given.
static cl::extrahelp MoreHelp(
  "\nOPERATIONS:\n"
  "  d[NsS]       - delete file(s) from the archive\n"
  "  m[abiSs]     - move file(s) in the archive\n"
  "  p[kN]        - print file(s) found in the archive\n"
  "  q[ufsS]      - quick append file(s) to the archive\n"
  "  r[abfiuRsS]  - replace or insert file(s) into the archive\n"
  "  t            - display contents of archive\n"
  "  x[No]        - extract file(s) from the archive\n"
  "\nMODIFIERS (operation specific):\n"
  "  [a] - put file(s) after [relpos]\n"
  "  [b] - put file(s) before [relpos] (same as [i])\n"
  "  [i] - put file(s) before [relpos] (same as [b])\n"
  "  [o] - preserve original dates\n"
  "  [s] - create an archive index (cf. ranlib)\n"
  "  [S] - do not build a symbol table\n"
  "  [T] - create a thin archive\n"
  "  [u] - update only files newer than archive contents\n"
  "\nMODIFIERS (generic):\n"
  "  [c] - do not warn if the library had to be created\n"
  "  [v] - be verbose about actions taken\n"
);

// This enumeration delineates the kinds of operations on an archive
// that are permitted.
enum ArchiveOperation {
  Print,            ///< Print the contents of the archive
  Delete,           ///< Delete the specified members
  Move,             ///< Move members to end or as given by {a,b,i} modifiers
  QuickAppend,      ///< Quickly append to end of archive
  ReplaceOrInsert,  ///< Replace or Insert members
  DisplayTable,     ///< Display the table of contents
  Extract,          ///< Extract files back to file system
  CreateSymTab      ///< Create a symbol table in an existing archive
};

// Modifiers to follow operation to vary behavior
static bool AddAfter = false;      ///< 'a' modifier
static bool AddBefore = false;     ///< 'b' modifier
static bool Create = false;        ///< 'c' modifier
static bool OriginalDates = false; ///< 'o' modifier
static bool OnlyUpdate = false;    ///< 'u' modifier
static bool Verbose = false;       ///< 'v' modifier
static bool Symtab = true;         ///< 's' modifier
static bool Deterministic = true;  ///< 'D' and 'U' modifiers
static bool Thin = false;          ///< 'T' modifier

// Relative Positional Argument (for insert/move). This variable holds
// the name of the archive member to which the 'a', 'b' or 'i' modifier
// refers. Only one of 'a', 'b' or 'i' can be specified so we only need
// one variable.
static std::string RelPos;

// This variable holds the name of the archive file as given on the
// command line.
static std::string ArchiveName;

// This variable holds the list of member files to proecess, as given
// on the command line.
static std::vector<StringRef> Members;

// Show the error message, the help message and exit.
LLVM_ATTRIBUTE_NORETURN static void
show_help(const std::string &msg) {
  errs() << ToolName << ": " << msg << "\n\n";
  cl::PrintHelpMessage();
  std::exit(1);
}

// Extract the member filename from the command line for the [relpos] argument
// associated with a, b, and i modifiers
static void getRelPos() {
  if(RestOfArgs.size() == 0)
    show_help("Expected [relpos] for a, b, or i modifier");
  RelPos = RestOfArgs[0];
  RestOfArgs.erase(RestOfArgs.begin());
}

static void getOptions() {
  if(RestOfArgs.size() == 0)
    show_help("Expected options");
  Options = RestOfArgs[0];
  RestOfArgs.erase(RestOfArgs.begin());
}

// Get the archive file name from the command line
static void getArchive() {
  if(RestOfArgs.size() == 0)
    show_help("An archive name must be specified");
  ArchiveName = RestOfArgs[0];
  RestOfArgs.erase(RestOfArgs.begin());
}

// Copy over remaining items in RestOfArgs to our Members vector
static void getMembers() {
  for (auto &Arg : RestOfArgs)
    Members.push_back(Arg);
}

static void runMRIScript();

// Parse the command line options as presented and return the operation
// specified. Process all modifiers and check to make sure that constraints on
// modifier/operation pairs have not been violated.
static ArchiveOperation parseCommandLine() {
  if (MRI) {
    if (!RestOfArgs.empty())
      fail("Cannot mix -M and other options");
    runMRIScript();
  }

  getOptions();

  // Keep track of number of operations. We can only specify one
  // per execution.
  unsigned NumOperations = 0;

  // Keep track of the number of positional modifiers (a,b,i). Only
  // one can be specified.
  unsigned NumPositional = 0;

  // Keep track of which operation was requested
  ArchiveOperation Operation;

  bool MaybeJustCreateSymTab = false;

  for(unsigned i=0; i<Options.size(); ++i) {
    switch(Options[i]) {
    case 'd': ++NumOperations; Operation = Delete; break;
    case 'm': ++NumOperations; Operation = Move ; break;
    case 'p': ++NumOperations; Operation = Print; break;
    case 'q': ++NumOperations; Operation = QuickAppend; break;
    case 'r': ++NumOperations; Operation = ReplaceOrInsert; break;
    case 't': ++NumOperations; Operation = DisplayTable; break;
    case 'x': ++NumOperations; Operation = Extract; break;
    case 'c': Create = true; break;
    case 'l': /* accepted but unused */ break;
    case 'o': OriginalDates = true; break;
    case 's':
      Symtab = true;
      MaybeJustCreateSymTab = true;
      break;
    case 'S':
      Symtab = false;
      break;
    case 'u': OnlyUpdate = true; break;
    case 'v': Verbose = true; break;
    case 'a':
      getRelPos();
      AddAfter = true;
      NumPositional++;
      break;
    case 'b':
      getRelPos();
      AddBefore = true;
      NumPositional++;
      break;
    case 'i':
      getRelPos();
      AddBefore = true;
      NumPositional++;
      break;
    case 'D':
      Deterministic = true;
      break;
    case 'U':
      Deterministic = false;
      break;
    case 'T':
      Thin = true;
      break;
    default:
      cl::PrintHelpMessage();
    }
  }

  // At this point, the next thing on the command line must be
  // the archive name.
  getArchive();

  // Everything on the command line at this point is a member.
  getMembers();

 if (NumOperations == 0 && MaybeJustCreateSymTab) {
    NumOperations = 1;
    Operation = CreateSymTab;
    if (!Members.empty())
      show_help("The s operation takes only an archive as argument");
  }

  // Perform various checks on the operation/modifier specification
  // to make sure we are dealing with a legal request.
  if (NumOperations == 0)
    show_help("You must specify at least one of the operations");
  if (NumOperations > 1)
    show_help("Only one operation may be specified");
  if (NumPositional > 1)
    show_help("You may only specify one of a, b, and i modifiers");
  if (AddAfter || AddBefore) {
    if (Operation != Move && Operation != ReplaceOrInsert)
      show_help("The 'a', 'b' and 'i' modifiers can only be specified with "
            "the 'm' or 'r' operations");
  }
  if (OriginalDates && Operation != Extract)
    show_help("The 'o' modifier is only applicable to the 'x' operation");
  if (OnlyUpdate && Operation != ReplaceOrInsert)
    show_help("The 'u' modifier is only applicable to the 'r' operation");

  // Return the parsed operation to the caller
  return Operation;
}

// Implements the 'p' operation. This function traverses the archive
// looking for members that match the path list.
static void doPrint(StringRef Name, const object::Archive::Child &C) {
  if (Verbose)
    outs() << "Printing " << Name << "\n";

  Expected<StringRef> DataOrErr = C.getBuffer();
  failIfError(DataOrErr.takeError());
  StringRef Data = *DataOrErr;
  outs().write(Data.data(), Data.size());
}

// Utility function for printing out the file mode when the 't' operation is in
// verbose mode.
static void printMode(unsigned mode) {
  outs() << ((mode & 004) ? "r" : "-");
  outs() << ((mode & 002) ? "w" : "-");
  outs() << ((mode & 001) ? "x" : "-");
}

// Implement the 't' operation. This function prints out just
// the file names of each of the members. However, if verbose mode is requested
// ('v' modifier) then the file type, permission mode, user, group, size, and
// modification time are also printed.
static void doDisplayTable(StringRef Name, const object::Archive::Child &C) {
  if (Verbose) {
    Expected<sys::fs::perms> ModeOrErr = C.getAccessMode();
    failIfError(ModeOrErr.takeError());
    sys::fs::perms Mode = ModeOrErr.get();
    printMode((Mode >> 6) & 007);
    printMode((Mode >> 3) & 007);
    printMode(Mode & 007);
    Expected<unsigned> UIDOrErr = C.getUID();
    failIfError(UIDOrErr.takeError());
    outs() << ' ' << UIDOrErr.get();
    Expected<unsigned> GIDOrErr = C.getGID();
    failIfError(GIDOrErr.takeError());
    outs() << '/' << GIDOrErr.get();
    Expected<uint64_t> Size = C.getSize();
    failIfError(Size.takeError());
    outs() << ' ' << format("%6llu", Size.get());
    auto ModTimeOrErr = C.getLastModified();
    failIfError(ModTimeOrErr.takeError());
    outs() << ' ' << ModTimeOrErr.get();
    outs() << ' ';
  }

  if (C.getParent()->isThin()) {
    outs() << sys::path::parent_path(ArchiveName);
    outs() << '/';
  }
  outs() << Name << "\n";
}

// Implement the 'x' operation. This function extracts files back to the file
// system.
static void doExtract(StringRef Name, const object::Archive::Child &C) {
  // Retain the original mode.
  Expected<sys::fs::perms> ModeOrErr = C.getAccessMode();
  failIfError(ModeOrErr.takeError());
  sys::fs::perms Mode = ModeOrErr.get();

  int FD;
  failIfError(sys::fs::openFileForWrite(Name, FD, sys::fs::F_None, Mode), Name);

  {
    raw_fd_ostream file(FD, false);

    // Get the data and its length
    Expected<StringRef> BufOrErr = C.getBuffer();
    failIfError(BufOrErr.takeError());
    StringRef Data = BufOrErr.get();

    // Write the data.
    file.write(Data.data(), Data.size());
  }

  // If we're supposed to retain the original modification times, etc. do so
  // now.
  if (OriginalDates) {
    auto ModTimeOrErr = C.getLastModified();
    failIfError(ModTimeOrErr.takeError());
    failIfError(
        sys::fs::setLastModificationAndAccessTime(FD, ModTimeOrErr.get()));
  }

  if (close(FD))
    fail("Could not close the file");
}

static bool shouldCreateArchive(ArchiveOperation Op) {
  switch (Op) {
  case Print:
  case Delete:
  case Move:
  case DisplayTable:
  case Extract:
  case CreateSymTab:
    return false;

  case QuickAppend:
  case ReplaceOrInsert:
    return true;
  }

  llvm_unreachable("Missing entry in covered switch.");
}

static void performReadOperation(ArchiveOperation Operation,
                                 object::Archive *OldArchive) {
  if (Operation == Extract && OldArchive->isThin())
    fail("extracting from a thin archive is not supported");

  bool Filter = !Members.empty();
  {
    Error Err = Error::success();
    for (auto &C : OldArchive->children(Err)) {
      Expected<StringRef> NameOrErr = C.getName();
      failIfError(NameOrErr.takeError());
      StringRef Name = NameOrErr.get();

      if (Filter) {
        auto I = find(Members, Name);
        if (I == Members.end())
          continue;
        Members.erase(I);
      }

      switch (Operation) {
      default:
        llvm_unreachable("Not a read operation");
      case Print:
        doPrint(Name, C);
        break;
      case DisplayTable:
        doDisplayTable(Name, C);
        break;
      case Extract:
        doExtract(Name, C);
        break;
      }
    }
    failIfError(std::move(Err));
  }

  if (Members.empty())
    return;
  for (StringRef Name : Members)
    errs() << Name << " was not found\n";
  std::exit(1);
}

static void addMember(std::vector<NewArchiveMember> &Members,
                      StringRef FileName, int Pos = -1) {
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getFile(FileName, Deterministic);
  failIfError(NMOrErr.takeError(), FileName);
  if (Pos == -1)
    Members.push_back(std::move(*NMOrErr));
  else
    Members[Pos] = std::move(*NMOrErr);
}

static void addMember(std::vector<NewArchiveMember> &Members,
                      const object::Archive::Child &M, int Pos = -1) {
  if (Thin && !M.getParent()->isThin())
    fail("Cannot convert a regular archive to a thin one");
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getOldMember(M, Deterministic);
  failIfError(NMOrErr.takeError());
  if (Pos == -1)
    Members.push_back(std::move(*NMOrErr));
  else
    Members[Pos] = std::move(*NMOrErr);
}

enum InsertAction {
  IA_AddOldMember,
  IA_AddNewMeber,
  IA_Delete,
  IA_MoveOldMember,
  IA_MoveNewMember
};

static InsertAction computeInsertAction(ArchiveOperation Operation,
                                        const object::Archive::Child &Member,
                                        StringRef Name,
                                        std::vector<StringRef>::iterator &Pos) {
  if (Operation == QuickAppend || Members.empty())
    return IA_AddOldMember;

  auto MI = find_if(Members, [Name](StringRef Path) {
    return Name == sys::path::filename(Path);
  });

  if (MI == Members.end())
    return IA_AddOldMember;

  Pos = MI;

  if (Operation == Delete)
    return IA_Delete;

  if (Operation == Move)
    return IA_MoveOldMember;

  if (Operation == ReplaceOrInsert) {
    StringRef PosName = sys::path::filename(RelPos);
    if (!OnlyUpdate) {
      if (PosName.empty())
        return IA_AddNewMeber;
      return IA_MoveNewMember;
    }

    // We could try to optimize this to a fstat, but it is not a common
    // operation.
    sys::fs::file_status Status;
    failIfError(sys::fs::status(*MI, Status), *MI);
    auto ModTimeOrErr = Member.getLastModified();
    failIfError(ModTimeOrErr.takeError());
    if (Status.getLastModificationTime() < ModTimeOrErr.get()) {
      if (PosName.empty())
        return IA_AddOldMember;
      return IA_MoveOldMember;
    }

    if (PosName.empty())
      return IA_AddNewMeber;
    return IA_MoveNewMember;
  }
  llvm_unreachable("No such operation");
}

// We have to walk this twice and computing it is not trivial, so creating an
// explicit std::vector is actually fairly efficient.
static std::vector<NewArchiveMember>
computeNewArchiveMembers(ArchiveOperation Operation,
                         object::Archive *OldArchive) {
  std::vector<NewArchiveMember> Ret;
  std::vector<NewArchiveMember> Moved;
  int InsertPos = -1;
  StringRef PosName = sys::path::filename(RelPos);
  if (OldArchive) {
    Error Err = Error::success();
    for (auto &Child : OldArchive->children(Err)) {
      int Pos = Ret.size();
      Expected<StringRef> NameOrErr = Child.getName();
      failIfError(NameOrErr.takeError());
      StringRef Name = NameOrErr.get();
      if (Name == PosName) {
        assert(AddAfter || AddBefore);
        if (AddBefore)
          InsertPos = Pos;
        else
          InsertPos = Pos + 1;
      }

      std::vector<StringRef>::iterator MemberI = Members.end();
      InsertAction Action =
          computeInsertAction(Operation, Child, Name, MemberI);
      switch (Action) {
      case IA_AddOldMember:
        addMember(Ret, Child);
        break;
      case IA_AddNewMeber:
        addMember(Ret, *MemberI);
        break;
      case IA_Delete:
        break;
      case IA_MoveOldMember:
        addMember(Moved, Child);
        break;
      case IA_MoveNewMember:
        addMember(Moved, *MemberI);
        break;
      }
      if (MemberI != Members.end())
        Members.erase(MemberI);
    }
    failIfError(std::move(Err));
  }

  if (Operation == Delete)
    return Ret;

  if (!RelPos.empty() && InsertPos == -1)
    fail("Insertion point not found");

  if (RelPos.empty())
    InsertPos = Ret.size();

  assert(unsigned(InsertPos) <= Ret.size());
  int Pos = InsertPos;
  for (auto &M : Moved) {
    Ret.insert(Ret.begin() + Pos, std::move(M));
    ++Pos;
  }

  for (unsigned I = 0; I != Members.size(); ++I)
    Ret.insert(Ret.begin() + InsertPos, NewArchiveMember());
  Pos = InsertPos;
  for (auto &Member : Members) {
    addMember(Ret, Member, Pos);
    ++Pos;
  }

  return Ret;
}

static object::Archive::Kind getDefaultForHost() {
  return Triple(sys::getProcessTriple()).isOSDarwin() ? object::Archive::K_BSD
                                                      : object::Archive::K_GNU;
}

static object::Archive::Kind getKindFromMember(const NewArchiveMember &Member) {
  Expected<std::unique_ptr<object::ObjectFile>> OptionalObject =
      object::ObjectFile::createObjectFile(Member.Buf->getMemBufferRef());

  if (OptionalObject)
    return isa<object::MachOObjectFile>(**OptionalObject)
               ? object::Archive::K_BSD
               : object::Archive::K_GNU;

  // squelch the error in case we had a non-object file
  consumeError(OptionalObject.takeError());
  return getDefaultForHost();
}

static void
performWriteOperation(ArchiveOperation Operation,
                      object::Archive *OldArchive,
                      std::unique_ptr<MemoryBuffer> OldArchiveBuf,
                      std::vector<NewArchiveMember> *NewMembersP) {
  std::vector<NewArchiveMember> NewMembers;
  if (!NewMembersP)
    NewMembers = computeNewArchiveMembers(Operation, OldArchive);

  object::Archive::Kind Kind;
  switch (FormatOpt) {
  case Default:
    if (Thin)
      Kind = object::Archive::K_GNU;
    else if (OldArchive)
      Kind = OldArchive->kind();
    else if (NewMembersP)
      Kind = NewMembersP->size() ? getKindFromMember(NewMembersP->front())
                                 : getDefaultForHost();
    else
      Kind = NewMembers.size() ? getKindFromMember(NewMembers.front())
                               : getDefaultForHost();
    break;
  case GNU:
    Kind = object::Archive::K_GNU;
    break;
  case BSD:
    if (Thin)
      fail("Only the gnu format has a thin mode");
    Kind = object::Archive::K_BSD;
    break;
  }

  std::pair<StringRef, std::error_code> Result =
      writeArchive(ArchiveName, NewMembersP ? *NewMembersP : NewMembers, Symtab,
                   Kind, Deterministic, Thin, std::move(OldArchiveBuf));
  failIfError(Result.second, Result.first);
}

static void createSymbolTable(object::Archive *OldArchive) {
  // When an archive is created or modified, if the s option is given, the
  // resulting archive will have a current symbol table. If the S option
  // is given, it will have no symbol table.
  // In summary, we only need to update the symbol table if we have none.
  // This is actually very common because of broken build systems that think
  // they have to run ranlib.
  if (OldArchive->hasSymbolTable())
    return;

  performWriteOperation(CreateSymTab, OldArchive, nullptr, nullptr);
}

static void performOperation(ArchiveOperation Operation,
                             object::Archive *OldArchive,
                             std::unique_ptr<MemoryBuffer> OldArchiveBuf,
                             std::vector<NewArchiveMember> *NewMembers) {
  switch (Operation) {
  case Print:
  case DisplayTable:
  case Extract:
    performReadOperation(Operation, OldArchive);
    return;

  case Delete:
  case Move:
  case QuickAppend:
  case ReplaceOrInsert:
    performWriteOperation(Operation, OldArchive, std::move(OldArchiveBuf),
                          NewMembers);
    return;
  case CreateSymTab:
    createSymbolTable(OldArchive);
    return;
  }
  llvm_unreachable("Unknown operation.");
}

static int performOperation(ArchiveOperation Operation,
                            std::vector<NewArchiveMember> *NewMembers) {
  // Create or open the archive object.
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
      MemoryBuffer::getFile(ArchiveName, -1, false);
  std::error_code EC = Buf.getError();
  if (EC && EC != errc::no_such_file_or_directory)
    fail("error opening '" + ArchiveName + "': " + EC.message() + "!");

  if (!EC) {
    Error Err = Error::success();
    object::Archive Archive(Buf.get()->getMemBufferRef(), Err);
    EC = errorToErrorCode(std::move(Err));
    failIfError(EC,
                "error loading '" + ArchiveName + "': " + EC.message() + "!");
    performOperation(Operation, &Archive, std::move(Buf.get()), NewMembers);
    return 0;
  }

  assert(EC == errc::no_such_file_or_directory);

  if (!shouldCreateArchive(Operation)) {
    failIfError(EC, Twine("error loading '") + ArchiveName + "'");
  } else {
    if (!Create) {
      // Produce a warning if we should and we're creating the archive
      errs() << ToolName << ": creating " << ArchiveName << "\n";
    }
  }

  performOperation(Operation, nullptr, nullptr, NewMembers);
  return 0;
}

static void runMRIScript() {
  enum class MRICommand { AddLib, AddMod, Create, Save, End, Invalid };

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getSTDIN();
  failIfError(Buf.getError());
  const MemoryBuffer &Ref = *Buf.get();
  bool Saved = false;
  std::vector<NewArchiveMember> NewMembers;
  std::vector<std::unique_ptr<MemoryBuffer>> ArchiveBuffers;
  std::vector<std::unique_ptr<object::Archive>> Archives;

  for (line_iterator I(Ref, /*SkipBlanks*/ true, ';'), E; I != E; ++I) {
    StringRef Line = *I;
    StringRef CommandStr, Rest;
    std::tie(CommandStr, Rest) = Line.split(' ');
    Rest = Rest.trim();
    if (!Rest.empty() && Rest.front() == '"' && Rest.back() == '"')
      Rest = Rest.drop_front().drop_back();
    auto Command = StringSwitch<MRICommand>(CommandStr.lower())
                       .Case("addlib", MRICommand::AddLib)
                       .Case("addmod", MRICommand::AddMod)
                       .Case("create", MRICommand::Create)
                       .Case("save", MRICommand::Save)
                       .Case("end", MRICommand::End)
                       .Default(MRICommand::Invalid);

    switch (Command) {
    case MRICommand::AddLib: {
      auto BufOrErr = MemoryBuffer::getFile(Rest, -1, false);
      failIfError(BufOrErr.getError(), "Could not open library");
      ArchiveBuffers.push_back(std::move(*BufOrErr));
      auto LibOrErr =
          object::Archive::create(ArchiveBuffers.back()->getMemBufferRef());
      failIfError(errorToErrorCode(LibOrErr.takeError()),
                  "Could not parse library");
      Archives.push_back(std::move(*LibOrErr));
      object::Archive &Lib = *Archives.back();
      {
        Error Err = Error::success();
        for (auto &Member : Lib.children(Err))
          addMember(NewMembers, Member);
        failIfError(std::move(Err));
      }
      break;
    }
    case MRICommand::AddMod:
      addMember(NewMembers, Rest);
      break;
    case MRICommand::Create:
      Create = true;
      if (!ArchiveName.empty())
        fail("Editing multiple archives not supported");
      if (Saved)
        fail("File already saved");
      ArchiveName = Rest;
      break;
    case MRICommand::Save:
      Saved = true;
      break;
    case MRICommand::End:
      break;
    case MRICommand::Invalid:
      fail("Unknown command: " + CommandStr);
    }
  }

  // Nothing to do if not saved.
  if (Saved)
    performOperation(ReplaceOrInsert, &NewMembers);
  exit(0);
}

static int ar_main() {
  // Do our own parsing of the command line because the CommandLine utility
  // can't handle the grouped positional parameters without a dash.
  ArchiveOperation Operation = parseCommandLine();
  return performOperation(Operation, nullptr);
}

static int ranlib_main() {
  if (RestOfArgs.size() != 1)
    fail(ToolName + " takes just one archive as an argument");
  ArchiveName = RestOfArgs[0];
  return performOperation(CreateSymTab, nullptr);
}

int main(int argc, char **argv) {
  ToolName = argv[0];
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  StringRef Stem = sys::path::stem(ToolName);
  if (Stem.find("ranlib") == StringRef::npos &&
      Stem.find("lib") != StringRef::npos)
    return libDriverMain(makeArrayRef(argv, argc));

  // Have the command line options parsed and handle things
  // like --help and --version.
  cl::ParseCommandLineOptions(argc, argv,
    "LLVM Archiver (llvm-ar)\n\n"
    "  This program archives bitcode files into single libraries\n"
  );

  if (Stem.find("ranlib") != StringRef::npos)
    return ranlib_main();
  if (Stem.find("ar") != StringRef::npos)
    return ar_main();
  fail("Not ranlib, ar or lib!");
}
