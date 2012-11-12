//===- lib/Linker/Linker.cpp - Basic Linker functionality  ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains basic Linker functionality that all usages will need.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Module.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
using namespace llvm;

Linker::Linker(StringRef progname, StringRef modname,
               LLVMContext& C, unsigned flags):
  Context(C),
  Composite(new Module(modname, C)),
  LibPaths(),
  Flags(flags),
  Error(),
  ProgramName(progname) { }

Linker::Linker(StringRef progname, Module* aModule, unsigned flags) :
  Context(aModule->getContext()),
  Composite(aModule),
  LibPaths(),
  Flags(flags),
  Error(),
  ProgramName(progname) { }

Linker::~Linker() {
  delete Composite;
}

bool
Linker::error(StringRef message) {
  Error = message;
  if (!(Flags&QuietErrors))
    errs() << ProgramName << ": error: " << message << "\n";
  return true;
}

bool
Linker::warning(StringRef message) {
  Error = message;
  if (!(Flags&QuietWarnings))
    errs() << ProgramName << ": warning: " << message << "\n";
  return false;
}

void
Linker::verbose(StringRef message) {
  if (Flags&Verbose)
    errs() << "  " << message << "\n";
}

void
Linker::addPath(const sys::Path& path) {
  LibPaths.push_back(path);
}

void
Linker::addPaths(const std::vector<std::string>& paths) {
  for (unsigned i = 0, e = paths.size(); i != e; ++i)
    LibPaths.push_back(sys::Path(paths[i]));
}

void
Linker::addSystemPaths() {
  sys::Path::GetBitcodeLibraryPaths(LibPaths);
  LibPaths.insert(LibPaths.begin(),sys::Path("./"));
}

Module*
Linker::releaseModule() {
  Module* result = Composite;
  LibPaths.clear();
  Error.clear();
  Composite = 0;
  Flags = 0;
  return result;
}

// LoadObject - Read in and parse the bitcode file named by FN and return the
// module it contains (wrapped in an auto_ptr), or auto_ptr<Module>() and set
// Error if an error occurs.
std::auto_ptr<Module>
Linker::LoadObject(const sys::Path &FN) {
  std::string ParseErrorMessage;
  Module *Result = 0;

  OwningPtr<MemoryBuffer> Buffer;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(FN.c_str(), Buffer))
    ParseErrorMessage = "Error reading file '" + FN.str() + "'" + ": "
                      + ec.message();
  else
    Result = ParseBitcodeFile(Buffer.get(), Context, &ParseErrorMessage);

  if (Result)
    return std::auto_ptr<Module>(Result);
  Error = "Bitcode file '" + FN.str() + "' could not be loaded";
  if (ParseErrorMessage.size())
    Error += ": " + ParseErrorMessage;
  return std::auto_ptr<Module>();
}

// IsLibrary - Determine if "Name" is a library in "Directory". Return
// a non-empty sys::Path if its found, an empty one otherwise.
static inline sys::Path IsLibrary(StringRef Name,
                                  const sys::Path &Directory) {

  sys::Path FullPath(Directory);

  // Try the libX.a form
  FullPath.appendComponent(("lib" + Name).str());
  FullPath.appendSuffix("a");
  if (FullPath.isArchive())
    return FullPath;

  // Try the libX.bca form
  FullPath.eraseSuffix();
  FullPath.appendSuffix("bca");
  if (FullPath.isArchive())
    return FullPath;

  // Try the libX.so (or .dylib) form
  FullPath.eraseSuffix();
  FullPath.appendSuffix(sys::Path::GetDLLSuffix());
  if (FullPath.isDynamicLibrary())  // Native shared library?
    return FullPath;
  if (FullPath.isBitcodeFile())    // .so file containing bitcode?
    return FullPath;

  // Try libX form, to make it possible to add dependency on the
  // specific version of .so, like liblzma.so.1.0.0
  FullPath.eraseSuffix();
  if (FullPath.isDynamicLibrary())  // Native shared library?
    return FullPath;
  if (FullPath.isBitcodeFile())    // .so file containing bitcode?
    return FullPath;

  // Not found .. fall through

  // Indicate that the library was not found in the directory.
  FullPath.clear();
  return FullPath;
}

/// FindLib - Try to convert Filename into the name of a file that we can open,
/// if it does not already name a file we can open, by first trying to open
/// Filename, then libFilename.[suffix] for each of a set of several common
/// library suffixes, in each of the directories in LibPaths. Returns an empty
/// Path if no matching file can be found.
///
sys::Path
Linker::FindLib(StringRef Filename) {
  // Determine if the pathname can be found as it stands.
  sys::Path FilePath(Filename);
  if (FilePath.canRead() &&
      (FilePath.isArchive() || FilePath.isDynamicLibrary()))
    return FilePath;

  // Iterate over the directories in Paths to see if we can find the library
  // there.
  for (unsigned Index = 0; Index != LibPaths.size(); ++Index) {
    sys::Path Directory(LibPaths[Index]);
    sys::Path FullPath = IsLibrary(Filename, Directory);
    if (!FullPath.isEmpty())
      return FullPath;
  }
  return sys::Path();
}
