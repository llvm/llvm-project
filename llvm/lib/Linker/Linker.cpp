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
#include "llvm/Config/config.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Streams.h"
using namespace llvm;

Linker::Linker(const std::string& progname, const std::string& modname,
               unsigned flags)
  : Composite(0)
  , LibPaths()
  , Flags(flags)
  , Error()
  , ProgramName(progname)
{
  Composite = new Module(modname);
}

Linker::Linker(const std::string& progname, Module* aModule, unsigned flags)
  : Composite(aModule)
  , LibPaths()
  , Flags(flags)
  , Error()
  , ProgramName(progname)
{
}

Linker::~Linker() {
  delete Composite;
}

bool
Linker::error(const std::string& message) {
  Error = message;
  if (!(Flags&QuietErrors))
    cerr << ProgramName << ": error: " << message << "\n";
  return true;
}

bool
Linker::warning(const std::string& message) {
  Error = message;
  if (!(Flags&QuietWarnings))
    cerr << ProgramName << ": warning: " << message << "\n";
  return false;
}

void
Linker::verbose(const std::string& message) {
  if (Flags&Verbose)
    cerr << "  " << message << "\n";
}

void
Linker::addPath(const sys::Path& path) {
  LibPaths.push_back(path);
}

void
Linker::addPaths(const std::vector<std::string>& paths) {
  for (unsigned i = 0; i != paths.size(); ++i) {
    sys::Path aPath;
    aPath.set(paths[i]);
    LibPaths.push_back(aPath);
  }
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
  
  const std::string &FNS = FN.toString();
  std::auto_ptr<MemoryBuffer> Buffer(MemoryBuffer::getFileOrSTDIN(FNS.c_str()));
  if (Buffer.get())
    Result = ParseBitcodeFile(Buffer.get(), &ParseErrorMessage);
  else
    ParseErrorMessage = "Error reading file '" + FNS + "'";
    
  if (Result)
    return std::auto_ptr<Module>(Result);
  Error = "Bitcode file '" + FN.toString() + "' could not be loaded";
  if (ParseErrorMessage.size())
    Error += ": " + ParseErrorMessage;
  return std::auto_ptr<Module>();
}

// IsLibrary - Determine if "Name" is a library in "Directory". Return
// a non-empty sys::Path if its found, an empty one otherwise.
static inline sys::Path IsLibrary(const std::string& Name,
                                  const sys::Path& Directory) {

  sys::Path FullPath(Directory);

  // Try the libX.a form
  FullPath.appendComponent("lib" + Name);
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
  FullPath.appendSuffix(&(LTDL_SHLIB_EXT[1]));
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
Linker::FindLib(const std::string &Filename) {
  // Determine if the pathname can be found as it stands.
  sys::Path FilePath(Filename);
  if (FilePath.canRead() &&
      (FilePath.isArchive() || FilePath.isDynamicLibrary()))
    return FilePath;

  // Iterate over the directories in Paths to see if we can find the library
  // there.
  for (unsigned Index = 0; Index != LibPaths.size(); ++Index) {
    sys::Path Directory(LibPaths[Index]);
    sys::Path FullPath = IsLibrary(Filename,Directory);
    if (!FullPath.isEmpty())
      return FullPath;
  }
  return sys::Path();
}
