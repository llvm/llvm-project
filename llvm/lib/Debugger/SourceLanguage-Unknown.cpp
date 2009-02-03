//===-- SourceLanguage-Unknown.cpp - Implement itf for unknown languages --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// If the LLVM debugger does not have a module for a particular language, it
// falls back on using this one to perform the source-language interface.  This
// interface is not wonderful, but it gets the job done.
//
//===----------------------------------------------------------------------===//

#include "llvm/Debugger/SourceLanguage.h"
#include "llvm/Debugger/ProgramInfo.h"
#include "llvm/Support/Streams.h"
#include <cassert>
#include <ostream>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Implement the SourceLanguage cache for the Unknown language.
//

namespace {
  /// SLUCache - This cache allows for efficient lookup of source functions by
  /// name.
  ///
  struct SLUCache : public SourceLanguageCache {
    ProgramInfo &PI;
    std::multimap<std::string, SourceFunctionInfo*> FunctionMap;
  public:
    SLUCache(ProgramInfo &pi);

    typedef std::multimap<std::string, SourceFunctionInfo*>::const_iterator
       fm_iterator;

    std::pair<fm_iterator, fm_iterator>
    getFunction(const std::string &Name) const {
      return FunctionMap.equal_range(Name);
    }

    SourceFunctionInfo *addSourceFunction(SourceFunctionInfo *SF) {
      FunctionMap.insert(std::make_pair(SF->getSymbolicName(), SF));
      return SF;
    }
  };
}

SLUCache::SLUCache(ProgramInfo &pi) : PI(pi) {
}


//===----------------------------------------------------------------------===//
// Implement SourceLanguageUnknown class, which is used to handle unrecognized
// languages.
//

namespace {
  static struct SLU : public SourceLanguage {
    //===------------------------------------------------------------------===//
    // Implement the miscellaneous methods...
    //
    virtual const char *getSourceLanguageName() const {
      return "unknown";
    }

    /// lookupFunction - Given a textual function name, return the
    /// SourceFunctionInfo descriptor for that function, or null if it cannot be
    /// found.  If the program is currently running, the RuntimeInfo object
    /// provides information about the current evaluation context, otherwise it
    /// will be null.
    ///
    virtual SourceFunctionInfo *lookupFunction(const std::string &FunctionName,
                                               ProgramInfo &PI,
                                               RuntimeInfo *RI = 0) const;

    //===------------------------------------------------------------------===//
    // We do use a cache for information...
    //
    typedef SLUCache CacheType;
    SLUCache *createSourceLanguageCache(ProgramInfo &PI) const {
      return new SLUCache(PI);
    }

    /// createSourceFunctionInfo - Create the new object and inform the cache of
    /// the new function.
    virtual SourceFunctionInfo *
    createSourceFunctionInfo(const GlobalVariable *Desc, ProgramInfo &PI) const;

  } TheUnknownSourceLanguageInstance;
}

const SourceLanguage &SourceLanguage::getUnknownLanguageInstance() {
  return TheUnknownSourceLanguageInstance;
}


SourceFunctionInfo *
SLU::createSourceFunctionInfo(const GlobalVariable *Desc,
                              ProgramInfo &PI) const {
  SourceFunctionInfo *Result = new SourceFunctionInfo(PI, Desc);
  return PI.getLanguageCache(this).addSourceFunction(Result);
}


/// lookupFunction - Given a textual function name, return the
/// SourceFunctionInfo descriptor for that function, or null if it cannot be
/// found.  If the program is currently running, the RuntimeInfo object
/// provides information about the current evaluation context, otherwise it will
/// be null.
///
SourceFunctionInfo *SLU::lookupFunction(const std::string &FunctionName,
                                        ProgramInfo &PI, RuntimeInfo *RI) const{
  SLUCache &Cache = PI.getLanguageCache(this);
  std::pair<SLUCache::fm_iterator, SLUCache::fm_iterator> IP
    = Cache.getFunction(FunctionName);

  if (IP.first == IP.second) {
    if (PI.allSourceFunctionsRead())
      return 0;  // Nothing found

    // Otherwise, we might be able to find the function if we read all of them
    // in.  Do so now.
    PI.getSourceFunctions();
    assert(PI.allSourceFunctionsRead() && "Didn't read in all functions?");
    return lookupFunction(FunctionName, PI, RI);
  }

  SourceFunctionInfo *Found = IP.first->second;
  ++IP.first;
  if (IP.first != IP.second)
    cout << "Whoa, found multiple functions with the same name.  I should"
         << " ask the user which one to use: FIXME!\n";
  return Found;
}
