//===-- BlackList.cpp - blacklist for sanitizers --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility class for instrumentation passes (like AddressSanitizer
// or ThreadSanitizer) to avoid instrumenting some functions or global
// variables based on a user-supplied blacklist.
//
//===----------------------------------------------------------------------===//

#include <utility>
#include <string>

#include "BlackList.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

namespace llvm {

BlackList::BlackList(const StringRef Path) {
  // Validate and open blacklist file.
  if (!Path.size()) return;
  OwningPtr<MemoryBuffer> File;
  if (error_code EC = MemoryBuffer::getFile(Path, File)) {
    report_fatal_error("Can't open blacklist file: " + Path + ": " +
                       EC.message());
  }

  // Iterate through each line in the blacklist file.
  SmallVector<StringRef, 16> Lines;
  SplitString(File.take()->getBuffer(), Lines, "\n\r");
  StringMap<std::string> Regexps;
  for (SmallVector<StringRef, 16>::iterator I = Lines.begin(), E = Lines.end();
       I != E; ++I) {
    // Ignore empty lines and lines starting with "#"
    if (I->empty() || I->startswith("#"))
      continue;
    // Get our prefix and unparsed regexp.
    std::pair<StringRef, StringRef> SplitLine = I->split(":");
    StringRef Prefix = SplitLine.first;
    std::string Regexp = SplitLine.second;

    // Replace * with .*
    for (size_t pos = 0; (pos = Regexp.find("*", pos)) != std::string::npos;
         pos += strlen(".*")) {
      Regexp.replace(pos, strlen("*"), ".*");
    }

    // Check that the regexp is valid.
    Regex CheckRE(Regexp);
    std::string Error;
    if (!CheckRE.isValid(Error)) {
      report_fatal_error("malformed blacklist regex: " + SplitLine.second +
          ": " + Error);
    }

    // Add this regexp into the proper group by its prefix.
    if (Regexps[Prefix].size())
      Regexps[Prefix] += "|";
    Regexps[Prefix] += Regexp;
  }

  // Iterate through each of the prefixes, and create Regexs for them.
  for (StringMap<std::string>::iterator I = Regexps.begin(), E = Regexps.end();
       I != E; ++I) {
    Entries[I->getKey()] = new Regex(I->getValue());
  }
}

bool BlackList::isIn(const Function &F) {
  return isIn(*F.getParent()) || inSection("fun", F.getName());
}

bool BlackList::isIn(const GlobalVariable &G) {
  return isIn(*G.getParent()) || inSection("global", G.getName());
}

bool BlackList::isIn(const Module &M) {
  return inSection("src", M.getModuleIdentifier());
}

bool BlackList::isInInit(const GlobalVariable &G) {
  return isIn(*G.getParent()) || inSection("global-init", G.getName());
}

bool BlackList::inSection(const StringRef Section,
                                  const StringRef Query) {
  Regex *FunctionRegex = Entries[Section];
  return FunctionRegex ? FunctionRegex->match(Query) : false;
}

}  // namespace llvm
