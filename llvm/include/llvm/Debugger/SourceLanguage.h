//===- SourceLanguage.h - Interact with source languages --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the abstract SourceLanguage interface, which is used by the
// LLVM debugger to parse source-language expressions and render program objects
// into a human readable string.  In general, these classes perform all of the
// analysis and interpretation of the language-specific debugger information.
//
// This interface is designed to be completely stateless, so all methods are
// const.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGGER_SOURCELANGUAGE_H
#define LLVM_DEBUGGER_SOURCELANGUAGE_H

#include <string>

namespace llvm {
  class GlobalVariable;
  class SourceFileInfo;
  class SourceFunctionInfo;
  class ProgramInfo;
  class RuntimeInfo;

  struct SourceLanguage {
    virtual ~SourceLanguage() {}

    /// getSourceLanguageName - This method is used to implement the 'show
    /// language' command in the debugger.
    virtual const char *getSourceLanguageName() const = 0;

    //===------------------------------------------------------------------===//
    // Methods used to implement debugger hooks.
    //

    /// printInfo - Implementing this method allows the debugger to use
    /// language-specific 'info' extensions, e.g., 'info selectors' for objc.
    /// This method should return true if the specified string is recognized.
    ///
    virtual bool printInfo(const std::string &What) const {
      return false;
    }

    /// lookupFunction - Given a textual function name, return the
    /// SourceFunctionInfo descriptor for that function, or null if it cannot be
    /// found.  If the program is currently running, the RuntimeInfo object
    /// provides information about the current evaluation context, otherwise it
    /// will be null.
    ///
    virtual SourceFunctionInfo *lookupFunction(const std::string &FunctionName,
                                               ProgramInfo &PI,
                                               RuntimeInfo *RI = 0) const {
      return 0;
    }


    //===------------------------------------------------------------------===//
    // Methods used to parse various pieces of program information.
    //

    /// createSourceFileInfo - This method can be implemented by the front-end
    /// if it needs to keep track of information beyond what the debugger
    /// requires.
    virtual SourceFileInfo *
    createSourceFileInfo(const GlobalVariable *Desc, ProgramInfo &PI) const;

    /// createSourceFunctionInfo - This method can be implemented by the derived
    /// SourceLanguage if it needs to keep track of more information than the
    /// SourceFunctionInfo has.
    virtual SourceFunctionInfo *
    createSourceFunctionInfo(const GlobalVariable *Desc, ProgramInfo &PI) const;


    //===------------------------------------------------------------------===//
    // Static methods used to get instances of various source languages.
    //

    /// get - This method returns a source-language instance for the specified
    /// Dwarf 3 language identifier.  If the language is unknown, an object is
    /// returned that can support some minimal operations, but is not terribly
    /// bright.
    static const SourceLanguage &get(unsigned ID);

    /// get*Instance() - These methods return specific instances of languages.
    ///
    static const SourceLanguage &getCFamilyInstance();
    static const SourceLanguage &getCPlusPlusInstance();
    static const SourceLanguage &getUnknownLanguageInstance();
  };
}

#endif
