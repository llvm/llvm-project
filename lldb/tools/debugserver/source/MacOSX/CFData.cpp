//===-- CFData.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 1/16/08.
//
//===----------------------------------------------------------------------===//

#include "CFData.h"

//----------------------------------------------------------------------
// CFData constructor
//----------------------------------------------------------------------
CFData::CFData(CFDataRef data) :
    CFReleaser<CFDataRef>(data)
{

}

//----------------------------------------------------------------------
// CFData copy constructor
//----------------------------------------------------------------------
CFData::CFData(const CFData& rhs) :
    CFReleaser<CFDataRef>(rhs)
{

}

//----------------------------------------------------------------------
// CFData copy constructor
//----------------------------------------------------------------------
CFData&
CFData::operator=(const CFData& rhs)

{
    *this = rhs;
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CFData::~CFData()
{
}


CFIndex
CFData::GetLength() const
{
    CFDataRef data = get();
    if (data)
        return CFDataGetLength (data);
    return 0;
}


const uint8_t*
CFData::GetBytePtr() const
{
    CFDataRef data = get();
    if (data)
        return CFDataGetBytePtr (data);
    return NULL;
}

CFDataRef
CFData::Serialize(CFPropertyListRef plist, CFPropertyListFormat format)
{
    CFAllocatorRef alloc = kCFAllocatorDefault;
    reset();
    CFReleaser<CFWriteStreamRef> stream (::CFWriteStreamCreateWithAllocatedBuffers (alloc, alloc));
    ::CFWriteStreamOpen (stream.get());
    CFIndex len = ::CFPropertyListWriteToStream (plist, stream.get(), format, NULL);
    if (len > 0)
        reset((CFDataRef)::CFWriteStreamCopyProperty (stream.get(), kCFStreamPropertyDataWritten));
    ::CFWriteStreamClose (stream.get());
    return get();
}

