//===-- DNBRuntimeAction.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 10/8/07.
//
//===----------------------------------------------------------------------===//

#ifndef __DNBRuntimeAction_h__
#define __DNBRuntimeAction_h__

class DNBRuntimeAction
{
    virtual void Initialize (nub_process_t pid) = 0;
    virtual void ProcessStateChanged (nub_state_t state) = 0;
    virtual void SharedLibraryStateChanged (DNBExecutableImageInfo *image_infos, nub_size_t num_image_infos) = 0;
};


#endif // #ifndef __DNBRuntimeAction_h__
