//===- lib/Linker/LinkItems.cpp - Link LLVM objects and libraries ---------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains routines to handle linking together LLVM bytecode files,
// and to handle annoying things like static libraries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/Archive.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Config/config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Timer.h"
#include "llvm/System/Signals.h"
#include "llvm/Support/SystemUtils.h"
#include <algorithm>
#include <fstream>
#include <memory>
#include <set>
using namespace llvm;

static bool 
LinkOneLibrary(const char*progname, Module* HeadModule, 
               const std::string& Lib, 
               const std::vector<std::string>& LibPaths,
               bool Verbose, bool Native) {

  // String in which to receive error messages.
  std::string ErrorMessage;

  // Determine where this library lives.
  std::string Pathname = FindLib(Lib, LibPaths);
  if (Pathname.empty()) {
    // If the pathname does not exist, then simply return if we're doing a 
    // native link and give a warning if we're doing a bytecode link.
    if (!Native) {
      std::cerr << progname << ": warning: Cannot find library '"
                << Lib << "'\n";
      return false;
    }
  }

  // A user may specify an ar archive without -l, perhaps because it
  // is not installed as a library. Detect that and link the library.
  if (IsArchive(Pathname)) {
    if (Verbose)
      std::cerr << "Trying to link archive '" << Pathname << "' (-l"
                << Lib << ")\n";

    if (LinkInArchive(HeadModule, Pathname, &ErrorMessage, Verbose)) {
      std::cerr << progname << ": " << ErrorMessage
                << ": Error linking in archive '" << Pathname << "' (-l"
                << Lib << ")\n";
      return true;
    }
  } else {
      std::cerr << progname << ": WARNING: Supposed library -l"
                << Lib << " isn't a library.\n";
  }
  return false;
}

// LinkItems - preserve link order for an arbitrary set of linkage items.
Module* 
llvm::LinkItems(const char *progname, const LinkItemList& Items,
                const std::vector<std::string>& LibPaths,
                bool Verbose, bool Native) {

  // Construct the HeadModule to contain the result of the linkage
  std::auto_ptr<Module> HeadModule(new Module(progname));

  // Construct a mutable path list we can add paths to. This list will always
  // have LLVM_LIB_SEARCH_PATH at the end so we place it there now.
  std::vector<std::string> MyLibPaths(LibPaths);
  MyLibPaths.insert(MyLibPaths.begin(),".");
  char* SearchPath = getenv("LLVM_LIB_SEARCH_PATH");
  if (SearchPath)
    MyLibPaths.push_back(SearchPath);

  // For each linkage item ...
  for (LinkItemList::const_iterator I = Items.begin(), E = Items.end(); 
       I != E; ++I) {
    if (I->second) {
      // Link in the library suggested.
      if (LinkOneLibrary(progname,HeadModule.get(),I->first,MyLibPaths,
                     Verbose,Native))
        return 0;
    } else {
      std::vector<std::string> Files;
      Files.push_back(I->first);
      if (LinkFiles(progname,HeadModule.get(),Files,Verbose))
        return 0;
    }
  }

  // At this point we have processed all the link items provided to us. Since
  // we have an aggregated module at this point, the dependent libraries in
  // that module should also be aggregated with duplicates eliminated. This is
  // now the time to process the dependent libraries to resolve any remaining
  // symbols.
  const Module::LibraryListType& DepLibs = HeadModule->getLibraries();
  for (Module::LibraryListType::const_iterator I = DepLibs.begin(), 
      E = DepLibs.end(); I != E; ++I) {
    if(LinkOneLibrary(progname,HeadModule.get(),*I,MyLibPaths,Verbose,Native))
      return 0;
  }

  return HeadModule.release();
}

// BuildLinkItems -- This function
void llvm::BuildLinkItems(
  LinkItemList& Items,
  const cl::list<std::string>& Files,
  const cl::list<std::string>& Libraries) {

  // Build the list of linkage items for LinkItems. 

  cl::list<std::string>::const_iterator fileIt = Files.begin();
  cl::list<std::string>::const_iterator libIt  = Libraries.begin();

  int libPos = -1, filePos = -1;
  while ( 1 ) {
    if (libIt != Libraries.end())
      libPos = Libraries.getPosition(libIt - Libraries.begin());
    else
      libPos = -1;
    if (fileIt != Files.end())
      filePos = Files.getPosition(fileIt - Files.begin());
    else
      filePos = -1;

    if (filePos != -1 && (libPos == -1 || filePos < libPos)) {
      // Add a source file
      Items.push_back(std::make_pair(*fileIt++, false));
    } else if (libPos != -1 && (filePos == -1 || libPos < filePos)) {
      // Add a library
      Items.push_back(std::make_pair(*libIt++, true));
    } else {
        break; // we're done with the list
    }
  }
}
