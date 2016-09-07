//===-- HasAVX.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HasAVX_h
#define HasAVX_h

#if defined(__i386__) || defined(__x86_64__)

#ifdef __cplusplus
extern "C" {
#endif

int HasAVX();

#ifdef __cplusplus
}
#endif

#endif

#endif
