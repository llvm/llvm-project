//===- ProgramInfo.h - Information about the loaded program -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various pieces of information about the currently loaded
// program.  One instance of this object is created every time a program is
// loaded, and destroyed every time it is unloaded.
//
// The various pieces of information gathered about the source program are all
// designed to be extended by various SourceLanguage implementations.  This
// allows source languages to keep any extended information that they support in
// the derived class portions of the class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGGER_PROGRAMINFO_H
#define LLVM_DEBUGGER_PROGRAMINFO_H

#include "llvm/System/TimeValue.h"
#include <string>
#include <map>
#include <vector>

namespace llvm {
  class GlobalVariable;
  class Module;
  class SourceFile;
  class SourceLanguage;
  class ProgramInfo;

  /// SourceLanguageCache - SourceLanguage implementations are allowed to cache
  /// stuff in the ProgramInfo object.  The only requirement we have on these
  /// instances is that they are destroyable.
  struct SourceLanguageCache {
    virtual ~SourceLanguageCache() {}
  };

  /// SourceFileInfo - One instance of this structure is created for each
  /// source file in the program.
  ///
  class SourceFileInfo {
    /// BaseName - The filename of the source file.
    std::string BaseName;

    /// Directory - The working directory of this source file when it was
    /// compiled.
    std::string Directory;

    /// Version - The version of the LLVM debug information that this file was
    /// compiled with.
    unsigned Version;

    /// Language - The source language that the file was compiled with.  This
    /// pointer is never null.
    ///
    const SourceLanguage *Language;

    /// Descriptor - The LLVM Global Variable which describes the source file.
    ///
    const GlobalVariable *Descriptor;

    /// SourceText - The body of this source file, or null if it has not yet
    /// been loaded.
    mutable SourceFile *SourceText;
  public:
    SourceFileInfo(const GlobalVariable *Desc, const SourceLanguage &Lang);
    ~SourceFileInfo();

    const std::string &getBaseName() const { return BaseName; }
    const std::string &getDirectory() const { return Directory; }
    unsigned getDebugVersion() const { return Version; }
    const GlobalVariable *getDescriptor() const { return Descriptor; }
    SourceFile &getSourceText() const;

    const SourceLanguage &getLanguage() const { return *Language; }
  };


  /// SourceFunctionInfo - An instance of this class is used to represent each
  /// source function in the program.
  ///
  class SourceFunctionInfo {
    /// Name - This contains an abstract name that is potentially useful to the
    /// end-user.  If there is no explicit support for the current language,
    /// then this string is used to identify the function.
    std::string Name;

    /// Descriptor - The descriptor for this function.
    ///
    const GlobalVariable *Descriptor;

    /// SourceFile - The file that this function is defined in.
    ///
    const SourceFileInfo *SourceFile;

    /// LineNo, ColNo - The location of the first stop-point in the function.
    /// These are computed on demand.
    mutable unsigned LineNo, ColNo;

  public:
    SourceFunctionInfo(ProgramInfo &PI, const GlobalVariable *Desc);
    virtual ~SourceFunctionInfo() {}

    /// getSymbolicName - Return a human-readable symbolic name to identify the
    /// function (for example, in stack traces).
    virtual std::string getSymbolicName() const { return Name; }

    /// getDescriptor - This returns the descriptor for the function.
    ///
    const GlobalVariable *getDescriptor() const { return Descriptor; }

    /// getSourceFile - This returns the source file that defines the function.
    ///
    const SourceFileInfo &getSourceFile() const { return *SourceFile; }

    /// getSourceLocation - This method returns the location of the first
    /// stopping point in the function.  If the body of the function cannot be
    /// found, this returns zeros for both values.
    void getSourceLocation(unsigned &LineNo, unsigned &ColNo) const;
  };


  /// ProgramInfo - This object contains information about the loaded program.
  /// When a new program is loaded, an instance of this class is created.  When
  /// the program is unloaded, the instance is destroyed.  This object basically
  /// manages the lazy computation of information useful for the debugger.
  class ProgramInfo {
    Module *M;

    /// ProgramTimeStamp - This is the timestamp of the executable file that we
    /// currently have loaded into the debugger.
    sys::TimeValue ProgramTimeStamp;

    /// SourceFiles - This map is used to transform source file descriptors into
    /// their corresponding SourceFileInfo objects.  This mapping owns the
    /// memory for the SourceFileInfo objects.
    ///
    bool SourceFilesIsComplete;
    std::map<const GlobalVariable*, SourceFileInfo*> SourceFiles;

    /// SourceFileIndex - Mapping from source file basenames to the information
    /// about the file.  Note that there can be filename collisions, so this is
    /// a multimap.  This map is populated incrementally as the user interacts
    /// with the program, through the getSourceFileFromDesc method.  If ALL of
    /// the source files are needed, the getSourceFiles() method scans the
    /// entire program looking for them.
    ///
    std::multimap<std::string, SourceFileInfo*> SourceFileIndex;

    /// SourceFunctions - This map contains entries functions in the source
    /// program.  If SourceFunctionsIsComplete is true, then this is ALL of the
    /// functions in the program are in this map.
    bool SourceFunctionsIsComplete;
    std::map<const GlobalVariable*, SourceFunctionInfo*> SourceFunctions;

    /// LanguageCaches - Each source language is permitted to keep a per-program
    /// cache of information specific to whatever it needs.  This vector is
    /// effectively a small map from the languages that are active in the
    /// program to their caches.  This can be accessed by the language by the
    /// "getLanguageCache" method.
    std::vector<std::pair<const SourceLanguage*,
                          SourceLanguageCache*> > LanguageCaches;
  public:
    ProgramInfo(Module *m);
    ~ProgramInfo();

    /// getProgramTimeStamp - Return the time-stamp of the program when it was
    /// loaded.
    sys::TimeValue getProgramTimeStamp() const { return ProgramTimeStamp; }

    //===------------------------------------------------------------------===//
    // Interfaces to the source code files that make up the program.
    //

    /// getSourceFile - Return source file information for the specified source
    /// file descriptor object, adding it to the collection as needed.  This
    /// method always succeeds (is unambiguous), and is always efficient.
    ///
    const SourceFileInfo &getSourceFile(const GlobalVariable *Desc);

    /// getSourceFile - Look up the file with the specified name.  If there is
    /// more than one match for the specified filename, prompt the user to pick
    /// one.  If there is no source file that matches the specified name, throw
    /// an exception indicating that we can't find the file.  Otherwise, return
    /// the file information for that file.
    ///
    /// If the source file hasn't been discovered yet in the program, this
    /// method might have to index the whole program by calling the
    /// getSourceFiles() method.
    ///
    const SourceFileInfo &getSourceFile(const std::string &Filename);

    /// getSourceFiles - Index all of the source files in the program and return
    /// them.  This information is lazily computed the first time that it is
    /// requested.  Since this information can take a long time to compute, the
    /// user is given a chance to cancel it.  If this occurs, an exception is
    /// thrown.
    const std::map<const GlobalVariable*, SourceFileInfo*> &
    getSourceFiles(bool RequiresCompleteMap = true);

    //===------------------------------------------------------------------===//
    // Interfaces to the functions that make up the program.
    //

    /// getFunction - Return source function information for the specified
    /// function descriptor object, adding it to the collection as needed.  This
    /// method always succeeds (is unambiguous), and is always efficient.
    ///
    const SourceFunctionInfo &getFunction(const GlobalVariable *Desc);

    /// getSourceFunctions - Index all of the functions in the program and
    /// return them.  This information is lazily computed the first time that it
    /// is requested.  Since this information can take a long time to compute,
    /// the user is given a chance to cancel it.  If this occurs, an exception
    /// is thrown.
    const std::map<const GlobalVariable*, SourceFunctionInfo*> &
    getSourceFunctions(bool RequiresCompleteMap = true);

    /// addSourceFunctionsRead - Return true if the source functions map is
    /// complete: that is, all functions in the program have been read in.
    bool allSourceFunctionsRead() const { return SourceFunctionsIsComplete; }

    /// getLanguageCache - This method is used to build per-program caches of
    /// information, such as the functions or types visible to the program.
    /// This can be used by SourceLanguage implementations because it requires
    /// an accessible [sl]::CacheType typedef, where [sl] is the C++ type of the
    /// source-language subclass.
    template<typename SL>
    typename SL::CacheType &getLanguageCache(const SL *L) {
      for (unsigned i = 0, e = LanguageCaches.size(); i != e; ++i)
        if (LanguageCaches[i].first == L)
          return *(typename SL::CacheType*)LanguageCaches[i].second;
      typename SL::CacheType *NewCache = L->createSourceLanguageCache(*this);
      LanguageCaches.push_back(std::make_pair(L, NewCache));
      return *NewCache;
    }
  };

} // end namespace llvm

#endif
