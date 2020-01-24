//===-- CFCData.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CFCData.h"

// CFCData constructor
CFCData::CFCData(CFDataRef data) : CFCReleaser<CFDataRef>(data) {}

// CFCData copy constructor
CFCData::CFCData(const CFCData &rhs) : CFCReleaser<CFDataRef>(rhs) {}

// CFCData copy constructor
CFCData &CFCData::operator=(const CFCData &rhs)

{
  if (this != &rhs)
    *this = rhs;
  return *this;
}

// Destructor
CFCData::~CFCData() {}

CFIndex CFCData::GetLength() const {
  CFDataRef data = get();
  if (data)
    return CFDataGetLength(data);
  return 0;
}

const uint8_t *CFCData::GetBytePtr() const {
  CFDataRef data = get();
  if (data)
    return CFDataGetBytePtr(data);
  return NULL;
}

CFDataRef CFCData::Serialize(CFPropertyListRef plist,
                             CFPropertyListFormat format) {
  CFAllocatorRef alloc = kCFAllocatorDefault;
  reset();
  CFCReleaser<CFWriteStreamRef> stream(
      ::CFWriteStreamCreateWithAllocatedBuffers(alloc, alloc));
  ::CFWriteStreamOpen(stream.get());
  CFIndex len =
      ::CFPropertyListWriteToStream(plist, stream.get(), format, NULL);
  if (len > 0)
    reset((CFDataRef)::CFWriteStreamCopyProperty(stream.get(),
                                                 kCFStreamPropertyDataWritten));
  ::CFWriteStreamClose(stream.get());
  return get();
}
