//===- ArchiveWriter.cpp - ar File Format implementation --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the writeArchive function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ArchiveWriter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using namespace llvm;

NewArchiveMember::NewArchiveMember(MemoryBufferRef BufRef)
    : Buf(MemoryBuffer::getMemBuffer(BufRef, false)),
      MemberName(BufRef.getBufferIdentifier()) {}

Expected<NewArchiveMember>
NewArchiveMember::getOldMember(const object::Archive::Child &OldMember,
                               bool Deterministic) {
  Expected<llvm::MemoryBufferRef> BufOrErr = OldMember.getMemoryBufferRef();
  if (!BufOrErr)
    return BufOrErr.takeError();

  NewArchiveMember M;
  assert(M.IsNew == false);
  M.Buf = MemoryBuffer::getMemBuffer(*BufOrErr, false);
  M.MemberName = M.Buf->getBufferIdentifier();
  if (!Deterministic) {
    auto ModTimeOrErr = OldMember.getLastModified();
    if (!ModTimeOrErr)
      return ModTimeOrErr.takeError();
    M.ModTime = ModTimeOrErr.get();
    Expected<unsigned> UIDOrErr = OldMember.getUID();
    if (!UIDOrErr)
      return UIDOrErr.takeError();
    M.UID = UIDOrErr.get();
    Expected<unsigned> GIDOrErr = OldMember.getGID();
    if (!GIDOrErr)
      return GIDOrErr.takeError();
    M.GID = GIDOrErr.get();
    Expected<sys::fs::perms> AccessModeOrErr = OldMember.getAccessMode();
    if (!AccessModeOrErr)
      return AccessModeOrErr.takeError();
    M.Perms = AccessModeOrErr.get();
  }
  return std::move(M);
}

Expected<NewArchiveMember> NewArchiveMember::getFile(StringRef FileName,
                                                     bool Deterministic) {
  sys::fs::file_status Status;
  int FD;
  if (auto EC = sys::fs::openFileForRead(FileName, FD))
    return errorCodeToError(EC);
  assert(FD != -1);

  if (auto EC = sys::fs::status(FD, Status))
    return errorCodeToError(EC);

  // Opening a directory doesn't make sense. Let it fail.
  // Linux cannot open directories with open(2), although
  // cygwin and *bsd can.
  if (Status.type() == sys::fs::file_type::directory_file)
    return errorCodeToError(make_error_code(errc::is_a_directory));

  ErrorOr<std::unique_ptr<MemoryBuffer>> MemberBufferOrErr =
      MemoryBuffer::getOpenFile(FD, FileName, Status.getSize(), false);
  if (!MemberBufferOrErr)
    return errorCodeToError(MemberBufferOrErr.getError());

  if (close(FD) != 0)
    return errorCodeToError(std::error_code(errno, std::generic_category()));

  NewArchiveMember M;
  M.IsNew = true;
  M.Buf = std::move(*MemberBufferOrErr);
  M.MemberName = M.Buf->getBufferIdentifier();
  if (!Deterministic) {
    M.ModTime = std::chrono::time_point_cast<std::chrono::seconds>(
        Status.getLastModificationTime());
    M.UID = Status.getUser();
    M.GID = Status.getGroup();
    M.Perms = Status.permissions();
  }
  return std::move(M);
}

template <typename T>
static void printWithSpacePadding(raw_fd_ostream &OS, T Data, unsigned Size,
                                  bool MayTruncate = false) {
  uint64_t OldPos = OS.tell();
  OS << Data;
  unsigned SizeSoFar = OS.tell() - OldPos;
  if (Size > SizeSoFar) {
    OS.indent(Size - SizeSoFar);
  } else if (Size < SizeSoFar) {
    assert(MayTruncate && "Data doesn't fit in Size");
    // Some of the data this is used for (like UID) can be larger than the
    // space available in the archive format. Truncate in that case.
    OS.seek(OldPos + Size);
  }
}

static bool isBSDLike(object::Archive::Kind Kind) {
  switch (Kind) {
  case object::Archive::K_GNU:
    return false;
  case object::Archive::K_BSD:
  case object::Archive::K_DARWIN:
    return true;
  case object::Archive::K_MIPS64:
  case object::Archive::K_DARWIN64:
  case object::Archive::K_COFF:
    break;
  }
  llvm_unreachable("not supported for writting");
}

static void print32(raw_ostream &Out, object::Archive::Kind Kind,
                    uint32_t Val) {
  if (isBSDLike(Kind))
    support::endian::Writer<support::little>(Out).write(Val);
  else
    support::endian::Writer<support::big>(Out).write(Val);
}

static void printRestOfMemberHeader(
    raw_fd_ostream &Out, const sys::TimePoint<std::chrono::seconds> &ModTime,
    unsigned UID, unsigned GID, unsigned Perms, unsigned Size) {
  printWithSpacePadding(Out, sys::toTimeT(ModTime), 12);
  printWithSpacePadding(Out, UID, 6, true);
  printWithSpacePadding(Out, GID, 6, true);
  printWithSpacePadding(Out, format("%o", Perms), 8);
  printWithSpacePadding(Out, Size, 10);
  Out << "`\n";
}

static void
printGNUSmallMemberHeader(raw_fd_ostream &Out, StringRef Name,
                          const sys::TimePoint<std::chrono::seconds> &ModTime,
                          unsigned UID, unsigned GID, unsigned Perms,
                          unsigned Size) {
  printWithSpacePadding(Out, Twine(Name) + "/", 16);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms, Size);
}

static void
printBSDMemberHeader(raw_fd_ostream &Out, StringRef Name,
                     const sys::TimePoint<std::chrono::seconds> &ModTime,
                     unsigned UID, unsigned GID, unsigned Perms,
                     unsigned Size) {
  uint64_t PosAfterHeader = Out.tell() + 60 + Name.size();
  // Pad so that even 64 bit object files are aligned.
  unsigned Pad = OffsetToAlignment(PosAfterHeader, 8);
  unsigned NameWithPadding = Name.size() + Pad;
  printWithSpacePadding(Out, Twine("#1/") + Twine(NameWithPadding), 16);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms,
                          NameWithPadding + Size);
  Out << Name;
  assert(PosAfterHeader == Out.tell());
  while (Pad--)
    Out.write(uint8_t(0));
}

static bool useStringTable(bool Thin, StringRef Name) {
  return Thin || Name.size() >= 16 || Name.contains('/');
}

static void
printMemberHeader(raw_fd_ostream &Out, object::Archive::Kind Kind, bool Thin,
                  StringRef Name,
                  std::vector<unsigned>::iterator &StringMapIndexIter,
                  const sys::TimePoint<std::chrono::seconds> &ModTime,
                  unsigned UID, unsigned GID, unsigned Perms, unsigned Size) {
  if (isBSDLike(Kind))
    return printBSDMemberHeader(Out, Name, ModTime, UID, GID, Perms, Size);
  if (!useStringTable(Thin, Name))
    return printGNUSmallMemberHeader(Out, Name, ModTime, UID, GID, Perms, Size);
  Out << '/';
  printWithSpacePadding(Out, *StringMapIndexIter++, 15);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms, Size);
}

// Compute the relative path from From to To.
static std::string computeRelativePath(StringRef From, StringRef To) {
  if (sys::path::is_absolute(From) || sys::path::is_absolute(To))
    return To;

  StringRef DirFrom = sys::path::parent_path(From);
  auto FromI = sys::path::begin(DirFrom);
  auto ToI = sys::path::begin(To);
  while (*FromI == *ToI) {
    ++FromI;
    ++ToI;
  }

  SmallString<128> Relative;
  for (auto FromE = sys::path::end(DirFrom); FromI != FromE; ++FromI)
    sys::path::append(Relative, "..");

  for (auto ToE = sys::path::end(To); ToI != ToE; ++ToI)
    sys::path::append(Relative, *ToI);

#ifdef LLVM_ON_WIN32
  // Replace backslashes with slashes so that the path is portable between *nix
  // and Windows.
  std::replace(Relative.begin(), Relative.end(), '\\', '/');
#endif

  return Relative.str();
}

static void writeStringTable(raw_fd_ostream &Out, StringRef ArcName,
                             ArrayRef<NewArchiveMember> Members,
                             std::vector<unsigned> &StringMapIndexes,
                             bool Thin) {
  unsigned StartOffset = 0;
  for (const NewArchiveMember &M : Members) {
    StringRef Path = M.Buf->getBufferIdentifier();
    StringRef Name = M.MemberName;
    if (!useStringTable(Thin, Name))
      continue;
    if (StartOffset == 0) {
      printWithSpacePadding(Out, "//", 58);
      Out << "`\n";
      StartOffset = Out.tell();
    }
    StringMapIndexes.push_back(Out.tell() - StartOffset);

    if (Thin) {
      if (M.IsNew)
        Out << computeRelativePath(ArcName, Path);
      else
        Out << M.Buf->getBufferIdentifier();
    } else
      Out << Name;

    Out << "/\n";
  }
  if (StartOffset == 0)
    return;
  if (Out.tell() % 2)
    Out << '\n';
  int Pos = Out.tell();
  Out.seek(StartOffset - 12);
  printWithSpacePadding(Out, Pos - StartOffset, 10);
  Out.seek(Pos);
}

static sys::TimePoint<std::chrono::seconds> now(bool Deterministic) {
  using namespace std::chrono;

  if (!Deterministic)
    return time_point_cast<seconds>(system_clock::now());
  return sys::TimePoint<seconds>();
}

// Returns the offset of the first reference to a member offset.
static ErrorOr<unsigned>
writeSymbolTable(raw_fd_ostream &Out, object::Archive::Kind Kind,
                 ArrayRef<NewArchiveMember> Members,
                 std::vector<unsigned> &MemberOffsetRefs, bool Deterministic) {
  unsigned HeaderStartOffset = 0;
  unsigned BodyStartOffset = 0;
  SmallString<128> NameBuf;
  raw_svector_ostream NameOS(NameBuf);
  LLVMContext Context;
  for (unsigned MemberNum = 0, N = Members.size(); MemberNum < N; ++MemberNum) {
    MemoryBufferRef MemberBuffer = Members[MemberNum].Buf->getMemBufferRef();
    Expected<std::unique_ptr<object::SymbolicFile>> ObjOrErr =
        object::SymbolicFile::createSymbolicFile(
            MemberBuffer, llvm::file_magic::unknown, &Context);
    if (!ObjOrErr) {
      // FIXME: check only for "not an object file" errors.
      consumeError(ObjOrErr.takeError());
      continue;
    }
    object::SymbolicFile &Obj = *ObjOrErr.get();

    if (!HeaderStartOffset) {
      HeaderStartOffset = Out.tell();
      if (isBSDLike(Kind))
        printBSDMemberHeader(Out, "__.SYMDEF", now(Deterministic), 0, 0, 0, 0);
      else
        printGNUSmallMemberHeader(Out, "", now(Deterministic), 0, 0, 0, 0);
      BodyStartOffset = Out.tell();
      print32(Out, Kind, 0); // number of entries or bytes
    }

    for (const object::BasicSymbolRef &S : Obj.symbols()) {
      uint32_t Symflags = S.getFlags();
      if (Symflags & object::SymbolRef::SF_FormatSpecific)
        continue;
      if (!(Symflags & object::SymbolRef::SF_Global))
        continue;
      if (Symflags & object::SymbolRef::SF_Undefined &&
          !(Symflags & object::SymbolRef::SF_Indirect))
        continue;

      unsigned NameOffset = NameOS.tell();
      if (auto EC = S.printName(NameOS))
        return EC;
      NameOS << '\0';
      MemberOffsetRefs.push_back(MemberNum);
      if (isBSDLike(Kind))
        print32(Out, Kind, NameOffset);
      print32(Out, Kind, 0); // member offset
    }
  }

  if (HeaderStartOffset == 0)
    return 0;

  // ld64 prefers the cctools type archive which pads its string table to a
  // boundary of sizeof(int32_t).
  if (isBSDLike(Kind))
    for (unsigned P = OffsetToAlignment(NameOS.tell(), sizeof(int32_t)); P--;)
      NameOS << '\0';

  StringRef StringTable = NameOS.str();
  if (isBSDLike(Kind))
    print32(Out, Kind, StringTable.size()); // byte count of the string table
  Out << StringTable;
  // If there are no symbols, emit an empty symbol table, to satisfy Solaris
  // tools, older versions of which expect a symbol table in a non-empty
  // archive, regardless of whether there are any symbols in it.
  if (StringTable.size() == 0)
    print32(Out, Kind, 0);

  // ld64 requires the next member header to start at an offset that is
  // 4 bytes aligned.
  unsigned Pad = OffsetToAlignment(Out.tell(), 4);
  while (Pad--)
    Out.write(uint8_t(0));

  // Patch up the size of the symbol table now that we know how big it is.
  unsigned Pos = Out.tell();
  const unsigned MemberHeaderSize = 60;
  Out.seek(HeaderStartOffset + 48); // offset of the size field.
  printWithSpacePadding(Out, Pos - MemberHeaderSize - HeaderStartOffset, 10);

  // Patch up the number of symbols.
  Out.seek(BodyStartOffset);
  unsigned NumSyms = MemberOffsetRefs.size();
  if (isBSDLike(Kind))
    print32(Out, Kind, NumSyms * 8);
  else
    print32(Out, Kind, NumSyms);

  Out.seek(Pos);
  return BodyStartOffset + 4;
}

std::error_code
llvm::writeArchive(StringRef ArcName, std::vector<NewArchiveMember> &NewMembers,
                   bool WriteSymtab, object::Archive::Kind Kind,
                   bool Deterministic, bool Thin,
                   std::unique_ptr<MemoryBuffer> OldArchiveBuf) {
  assert((!Thin || !isBSDLike(Kind)) && "Only the gnu format has a thin mode");
  SmallString<128> TmpArchive;
  int TmpArchiveFD;
  if (auto EC = sys::fs::createUniqueFile(ArcName + ".temp-archive-%%%%%%%.a",
                                          TmpArchiveFD, TmpArchive))
    return EC;

  tool_output_file Output(TmpArchive, TmpArchiveFD);
  raw_fd_ostream &Out = Output.os();
  if (Thin)
    Out << "!<thin>\n";
  else
    Out << "!<arch>\n";

  std::vector<unsigned> MemberOffsetRefs;

  unsigned MemberReferenceOffset = 0;
  if (WriteSymtab) {
    ErrorOr<unsigned> MemberReferenceOffsetOrErr = writeSymbolTable(
        Out, Kind, NewMembers, MemberOffsetRefs, Deterministic);
    if (auto EC = MemberReferenceOffsetOrErr.getError())
      return EC;
    MemberReferenceOffset = MemberReferenceOffsetOrErr.get();
  }

  std::vector<unsigned> StringMapIndexes;
  if (!isBSDLike(Kind))
    writeStringTable(Out, ArcName, NewMembers, StringMapIndexes, Thin);

  std::vector<unsigned>::iterator StringMapIndexIter = StringMapIndexes.begin();
  std::vector<unsigned> MemberOffset;
  for (const NewArchiveMember &M : NewMembers) {
    MemoryBufferRef File = M.Buf->getMemBufferRef();
    unsigned Padding = 0;

    unsigned Pos = Out.tell();
    MemberOffset.push_back(Pos);

    // ld64 expects the members to be 8-byte aligned for 64-bit content and at
    // least 4-byte aligned for 32-bit content.  Opt for the larger encoding
    // uniformly.  This matches the behaviour with cctools and ensures that ld64
    // is happy with archives that we generate.
    if (Kind == object::Archive::K_DARWIN)
      Padding = OffsetToAlignment(M.Buf->getBufferSize(), 8);

    printMemberHeader(Out, Kind, Thin, M.MemberName, StringMapIndexIter,
                      M.ModTime, M.UID, M.GID, M.Perms,
                      M.Buf->getBufferSize() + Padding);

    if (!Thin)
      Out << File.getBuffer();

    while (Padding--)
      Out << '\n';
    if (Out.tell() % 2)
      Out << '\n';
  }

  if (MemberReferenceOffset) {
    Out.seek(MemberReferenceOffset);
    for (unsigned MemberNum : MemberOffsetRefs) {
      if (isBSDLike(Kind))
        Out.seek(Out.tell() + 4); // skip over the string offset
      print32(Out, Kind, MemberOffset[MemberNum]);
    }
  }

  Output.keep();
  Out.close();

  // At this point, we no longer need whatever backing memory
  // was used to generate the NewMembers. On Windows, this buffer
  // could be a mapped view of the file we want to replace (if
  // we're updating an existing archive, say). In that case, the
  // rename would still succeed, but it would leave behind a
  // temporary file (actually the original file renamed) because
  // a file cannot be deleted while there's a handle open on it,
  // only renamed. So by freeing this buffer, this ensures that
  // the last open handle on the destination file, if any, is
  // closed before we attempt to rename.
  OldArchiveBuf.reset();

  sys::fs::rename(TmpArchive, ArcName);
  return std::error_code();
}
