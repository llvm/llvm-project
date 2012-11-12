//===-- JIT.h - Abstract Execution Engine Interface -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common functionality for JITEventListener implementations
//
//===----------------------------------------------------------------------===//

#ifndef EVENT_LISTENER_COMMON_H
#define EVENT_LISTENER_COMMON_H

#include "llvm/DebugInfo.h"
#include "llvm/Metadata.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/Path.h"

namespace llvm {

namespace jitprofiling {

class FilenameCache {
  // Holds the filename of each Scope, so that we can pass a null-terminated
  // string into oprofile.  Use an AssertingVH rather than a ValueMap because we
  // shouldn't be modifying any MDNodes while this map is alive.
  DenseMap<AssertingVH<MDNode>, std::string> Filenames;
  DenseMap<AssertingVH<MDNode>, std::string> Paths;

 public:
  const char *getFilename(MDNode *Scope) {
    std::string &Filename = Filenames[Scope];
    if (Filename.empty()) {
      DIScope DIScope(Scope);
      Filename = DIScope.getFilename();
    }
    return Filename.c_str();
  }

  const char *getFullPath(MDNode *Scope) {
    std::string &P = Paths[Scope];
    if (P.empty()) {
      DIScope DIScope(Scope);
      StringRef DirName = DIScope.getDirectory();
      StringRef FileName = DIScope.getFilename();
      SmallString<256> FullPath;
      if (DirName != "." && DirName != "") {
        FullPath = DirName;
      }
      if (FileName != "") {
        sys::path::append(FullPath, FileName);
      }
      P = FullPath.str();
    }
    return P.c_str();
  }
};

} // namespace jitprofiling

} // namespace llvm

#endif //EVENT_LISTENER_COMMON_H
