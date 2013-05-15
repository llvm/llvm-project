//===-- CFCMutableSet.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CFCMutableSet.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

//----------------------------------------------------------------------
// CFCString constructor
//----------------------------------------------------------------------
CFCMutableSet::CFCMutableSet(CFMutableSetRef s) :
    CFCReleaser<CFMutableSetRef> (s)
{
}

//----------------------------------------------------------------------
// CFCMutableSet copy constructor
//----------------------------------------------------------------------
CFCMutableSet::CFCMutableSet(const CFCMutableSet& rhs) :
    CFCReleaser<CFMutableSetRef> (rhs)
{
}

//----------------------------------------------------------------------
// CFCMutableSet copy constructor
//----------------------------------------------------------------------
const CFCMutableSet&
CFCMutableSet::operator=(const CFCMutableSet& rhs)
{
    if (this != &rhs)
        *this = rhs;
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CFCMutableSet::~CFCMutableSet()
{
}


CFIndex
CFCMutableSet::GetCount() const
{
    CFMutableSetRef set = get();
    if (set)
        return ::CFSetGetCount (set);
    return 0;
}

CFIndex
CFCMutableSet::GetCountOfValue(const void *value) const
{
    CFMutableSetRef set = get();
    if (set)
        return ::CFSetGetCountOfValue (set, value);
    return 0;
}

const void *
CFCMutableSet::GetValue(const void *value) const
{
    CFMutableSetRef set = get();
    if (set)
        return ::CFSetGetValue(set, value);
    return NULL;
}


const void *
CFCMutableSet::AddValue(const void *value, bool can_create)
{
    CFMutableSetRef set = get();
    if (set == NULL)
    {
        if (can_create == false)
            return NULL;
        set = ::CFSetCreateMutable(kCFAllocatorDefault, 0, &kCFTypeSetCallBacks);
        reset ( set );
    }
    if (set != NULL)
    {
        ::CFSetAddValue(set, value);
        return value;
    }
    return NULL;
}

void
CFCMutableSet::RemoveValue(const void *value)
{
    CFMutableSetRef set = get();
    if (set)
        ::CFSetRemoveValue(set, value);
}

void
CFCMutableSet::RemoveAllValues()
{
    CFMutableSetRef set = get();
    if (set)
        ::CFSetRemoveAllValues(set);
}

