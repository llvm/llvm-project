//===-- CFCMutableSet.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CoreFoundationCPP_CFMutableSet_h_
#define CoreFoundationCPP_CFMutableSet_h_

#include "CFCReleaser.h"

class CFCMutableSet : public CFCReleaser<CFMutableSetRef>
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CFCMutableSet(CFMutableSetRef s = NULL);
    CFCMutableSet(const CFCMutableSet& rhs);
    virtual ~CFCMutableSet();

    //------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------
    const CFCMutableSet&
    operator=(const CFCMutableSet& rhs);


    CFIndex GetCount() const;
    CFIndex GetCountOfValue(const void *value) const;
    const void * GetValue(const void *value) const;
    const void * AddValue(const void *value, bool can_create);
    void RemoveValue(const void *value);
    void RemoveAllValues();



protected:
    //------------------------------------------------------------------
    // Classes that inherit from CFCMutableSet can see and modify these
    //------------------------------------------------------------------

private:
    //------------------------------------------------------------------
    // For CFCMutableSet only
    //------------------------------------------------------------------

};

#endif  // CoreFoundationCPP_CFMutableSet_h_
