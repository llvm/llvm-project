 //===- llvm/System/Unix/Host.inc -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the UNIX Host support.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/ADT/StringRef.h"
#include "Unix.h"
#include <sys/utsname.h>
#include <string>

using namespace llvm;

static std::string getOSVersion() {
  struct utsname info;

  if (uname(&info))
    return "";

  return info.release;
}

std::string sys::getHostTriple() {
  // FIXME: Derive directly instead of relying on the autoconf generated
  // variable.

  StringRef HostTripleString(LLVM_HOSTTRIPLE);
  std::pair<StringRef, StringRef> ArchSplit = HostTripleString.split('-');
  
  // Normalize the arch, since the host triple may not actually match the host.
  std::string Arch = ArchSplit.first;

  // It would be nice to do this in terms of llvm::Triple, but that is in
  // Support which is layered above us.
#if defined(__x86_64__)
  Arch = "x86_64";
#elif defined(__i386__)
  Arch = "i386";
#elif defined(__ppc64__)
  Arch = "powerpc64";
#elif defined(__ppc__)
  Arch = "powerpc";
#elif defined(__arm__)

  // FIXME: We need to pick the right ARM triple (which involves querying the
  // chip). However, for now this is most important for LLVM arch selection, so
  // we only need to make sure to distinguish ARM and Thumb.
#  if defined(__thumb__)
  Arch = "thumb";
#  else
  Arch = "arm";
#  endif

#else

  // FIXME: When enough auto-detection is in place, this should just
  // #error. Then at least the arch selection is done, and we only need the OS
  // etc selection to kill off the use of LLVM_HOSTTRIPLE.

#endif

  std::string Triple(Arch);
  Triple += '-';
  Triple += ArchSplit.second;

  // Force i<N>86 to i386.
  if (Triple[0] == 'i' && isdigit(Triple[1]) && 
      Triple[2] == '8' && Triple[3] == '6')
    Triple[1] = '3';

  // On darwin, we want to update the version to match that of the
  // host.    
  std::string::size_type DarwinDashIdx = Triple.find("-darwin");
  if (DarwinDashIdx != std::string::npos) {
    Triple.resize(DarwinDashIdx + strlen("-darwin"));
    
    // Only add the major part of the os version.
    std::string Version = getOSVersion();
    Triple += Version.substr(0, Version.find('.'));
  }

  return Triple;
}
