/*===- llvm/System/Solaris.h ------------------------------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*
 *
 * This file contains portability fixes for Solaris hosts.
 *
 *===----------------------------------------------------------------------===*/

#ifndef LLVM_SYSTEM_SOLARIS_H
#define LLVM_SYSTEM_SOLARIS_H

#include <sys/types.h>
#include <sys/regset.h>

#undef CS
#undef DS
#undef ES
#undef FS
#undef GS
#undef SS
#undef EAX
#undef ECX
#undef EDX
#undef EBX
#undef ESP
#undef EBP
#undef ESI
#undef EDI
#undef EIP
#undef UESP
#undef EFL
#undef ERR
#undef TRAPNO

#endif
