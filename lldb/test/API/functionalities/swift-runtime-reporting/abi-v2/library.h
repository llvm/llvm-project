// library.h
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------

#include <stdint.h>

// From swift's include/swift/Runtime/Debug.h file.
struct RuntimeErrorDetails {
  uintptr_t version;
  const char *errorType;
  const char *currentStackDescription;
  uintptr_t framesToSkip;
  void *memoryAddress;

  struct Thread {
    const char *description;
    uint64_t threadID;
    uintptr_t numFrames;
    void **frames;
  };

  uintptr_t numExtraThreads;
  Thread *threads;

  struct FixIt {
    const char *filename;
    uintptr_t startLine;
    uintptr_t startColumn;
    uintptr_t endLine;
    uintptr_t endColumn;
    const char *replacementText;
  };

  struct Note {
    const char *description;
    uintptr_t numFixIts;
    FixIt *fixIts;
  };

  uintptr_t numFixIts;
  FixIt *fixIts;

  uintptr_t numNotes;
  Note *notes;
};

enum: uintptr_t {
  RuntimeErrorFlagNone = 0,
  RuntimeErrorFlagFatal = 1 << 0
};

extern "C"
void _swift_runtime_on_report(uintptr_t flags, const char *message, RuntimeErrorDetails *details);
