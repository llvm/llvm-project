//===- lib/Linker/LinkArchives.cpp - Link LLVM objects and libraries ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains routines to handle linking together LLVM bitcode files,
// and to handle annoying things like static libraries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Module.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Bitcode/Archive.h"
#include <memory>
#include <set>
using namespace llvm;

/// GetAllUndefinedSymbols - calculates the set of undefined symbols that still
/// exist in an LLVM module. This is a bit tricky because there may be two
/// symbols with the same name but different LLVM types that will be resolved to
/// each other but aren't currently (thus we need to treat it as resolved).
///
/// Inputs:
///  M - The module in which to find undefined symbols.
///
/// Outputs:
///  UndefinedSymbols - A set of C++ strings containing the name of all
///                     undefined symbols.
///
static void
GetAllUndefinedSymbols(Module *M, std::set<std::string> &UndefinedSymbols) {
  std::set<std::string> DefinedSymbols;
  UndefinedSymbols.clear();

  // If the program doesn't define a main, try pulling one in from a .a file.
  // This is needed for programs where the main function is defined in an
  // archive, such f2c'd programs.
  Function *Main = M->getFunction("main");
  if (Main == 0 || Main->isDeclaration())
    UndefinedSymbols.insert("main");

  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (I->hasName()) {
      if (I->isDeclaration())
        UndefinedSymbols.insert(I->getName());
      else if (!I->hasLocalLinkage()) {
        assert(!I->hasDLLImportLinkage()
               && "Found dllimported non-external symbol!");
        DefinedSymbols.insert(I->getName());
      }      
    }

  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
    if (I->hasName()) {
      if (I->isDeclaration())
        UndefinedSymbols.insert(I->getName());
      else if (!I->hasLocalLinkage()) {
        assert(!I->hasDLLImportLinkage()
               && "Found dllimported non-external symbol!");
        DefinedSymbols.insert(I->getName());
      }      
    }

  for (Module::alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    if (I->hasName())
      DefinedSymbols.insert(I->getName());

  // Prune out any defined symbols from the undefined symbols set...
  for (std::set<std::string>::iterator I = UndefinedSymbols.begin();
       I != UndefinedSymbols.end(); )
    if (DefinedSymbols.count(*I))
      UndefinedSymbols.erase(I++);  // This symbol really is defined!
    else
      ++I; // Keep this symbol in the undefined symbols list
}

/// LinkInArchive - opens an archive library and link in all objects which
/// provide symbols that are currently undefined.
///
/// Inputs:
///  Filename - The pathname of the archive.
///
/// Return Value:
///  TRUE  - An error occurred.
///  FALSE - No errors.
bool
Linker::LinkInArchive(const sys::Path &Filename, bool &is_native) {
  // Make sure this is an archive file we're dealing with
  if (!Filename.isArchive())
    return error("File '" + Filename.str() + "' is not an archive.");

  // Open the archive file
  verbose("Linking archive file '" + Filename.str() + "'");

  // Find all of the symbols currently undefined in the bitcode program.
  // If all the symbols are defined, the program is complete, and there is
  // no reason to link in any archive files.
  std::set<std::string> UndefinedSymbols;
  GetAllUndefinedSymbols(Composite, UndefinedSymbols);

  if (UndefinedSymbols.empty()) {
    verbose("No symbols undefined, skipping library '" + Filename.str() + "'");
    return false;  // No need to link anything in!
  }

  std::string ErrMsg;
  std::auto_ptr<Archive> AutoArch (
    Archive::OpenAndLoadSymbols(Filename, Context, &ErrMsg));

  Archive* arch = AutoArch.get();

  if (!arch)
    return error("Cannot read archive '" + Filename.str() +
                 "': " + ErrMsg);
  if (!arch->isBitcodeArchive()) {
    is_native = true;
    return false;
  }
  is_native = false;

  // Save a set of symbols that are not defined by the archive. Since we're
  // entering a loop, there's no point searching for these multiple times. This
  // variable is used to "set_subtract" from the set of undefined symbols.
  std::set<std::string> NotDefinedByArchive;

  // Save the current set of undefined symbols, because we may have to make
  // multiple passes over the archive:
  std::set<std::string> CurrentlyUndefinedSymbols;

  do {
    CurrentlyUndefinedSymbols = UndefinedSymbols;

    // Find the modules we need to link into the target module.  Note that arch
    // keeps ownership of these modules and may return the same Module* from a
    // subsequent call.
    SmallVector<Module*, 16> Modules;
    if (!arch->findModulesDefiningSymbols(UndefinedSymbols, Modules, &ErrMsg))
      return error("Cannot find symbols in '" + Filename.str() + 
                   "': " + ErrMsg);

    // If we didn't find any more modules to link this time, we are done
    // searching this archive.
    if (Modules.empty())
      break;

    // Any symbols remaining in UndefinedSymbols after
    // findModulesDefiningSymbols are ones that the archive does not define. So
    // we add them to the NotDefinedByArchive variable now.
    NotDefinedByArchive.insert(UndefinedSymbols.begin(),
        UndefinedSymbols.end());

    // Loop over all the Modules that we got back from the archive
    for (SmallVectorImpl<Module*>::iterator I=Modules.begin(), E=Modules.end();
         I != E; ++I) {

      // Get the module we must link in.
      std::string moduleErrorMsg;
      Module* aModule = *I;
      if (aModule != NULL) {
        if (aModule->MaterializeAll(&moduleErrorMsg))
          return error("Could not load a module: " + moduleErrorMsg);

        verbose("  Linking in module: " + aModule->getModuleIdentifier());

        // Link it in
        if (LinkInModule(aModule, &moduleErrorMsg))
          return error("Cannot link in module '" +
                       aModule->getModuleIdentifier() + "': " + moduleErrorMsg);
      } 
    }
    
    // Get the undefined symbols from the aggregate module. This recomputes the
    // symbols we still need after the new modules have been linked in.
    GetAllUndefinedSymbols(Composite, UndefinedSymbols);

    // At this point we have two sets of undefined symbols: UndefinedSymbols
    // which holds the undefined symbols from all the modules, and
    // NotDefinedByArchive which holds symbols we know the archive doesn't
    // define. There's no point searching for symbols that we won't find in the
    // archive so we subtract these sets.
    set_subtract(UndefinedSymbols, NotDefinedByArchive);

    // If there's no symbols left, no point in continuing to search the
    // archive.
    if (UndefinedSymbols.empty())
      break;
  } while (CurrentlyUndefinedSymbols != UndefinedSymbols);

  return false;
}
