//===--- Triple.cpp - Target triple helper class --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"

#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstring>
using namespace llvm;

//

const char *Triple::getArchTypeName(ArchType Kind) {
  switch (Kind) {
  case InvalidArch: return "<invalid>";
  case UnknownArch: return "unknown";
    
  case alpha:   return "alpha";
  case arm:     return "arm";
  case bfin:    return "bfin";
  case cellspu: return "cellspu";
  case mips:    return "mips";
  case mipsel:  return "mipsel";
  case msp430:  return "msp430";
  case pic16:   return "pic16";
  case ppc64:   return "powerpc64";
  case ppc:     return "powerpc";
  case sparc:   return "sparc";
  case systemz: return "s390x";
  case tce:     return "tce";
  case thumb:   return "thumb";
  case x86:     return "i386";
  case x86_64:  return "x86_64";
  case xcore:   return "xcore";
  }

  return "<invalid>";
}

const char *Triple::getVendorTypeName(VendorType Kind) {
  switch (Kind) {
  case UnknownVendor: return "unknown";

  case Apple: return "apple";
  case PC: return "pc";
  }

  return "<invalid>";
}

const char *Triple::getOSTypeName(OSType Kind) {
  switch (Kind) {
  case UnknownOS: return "unknown";

  case AuroraUX: return "auroraux";
  case Cygwin: return "cygwin";
  case Darwin: return "darwin";
  case DragonFly: return "dragonfly";
  case FreeBSD: return "freebsd";
  case Linux: return "linux";
  case MinGW32: return "mingw32";
  case MinGW64: return "mingw64";
  case NetBSD: return "netbsd";
  case OpenBSD: return "openbsd";
  case Solaris: return "solaris";
  case Win32: return "win32";
  }

  return "<invalid>";
}

Triple::ArchType Triple::getArchTypeForLLVMName(const StringRef &Name) {
  if (Name == "alpha")
    return alpha;
  if (Name == "arm")
    return arm;
  if (Name == "bfin")
    return bfin;
  if (Name == "cellspu")
    return cellspu;
  if (Name == "mips")
    return mips;
  if (Name == "mipsel")
    return mipsel;
  if (Name == "msp430")
    return msp430;
  if (Name == "pic16")
    return pic16;
  if (Name == "ppc64")
    return ppc64;
  if (Name == "ppc")
    return ppc;
  if (Name == "sparc")
    return sparc;
  if (Name == "systemz")
    return systemz;
  if (Name == "tce")
    return tce;
  if (Name == "thumb")
    return thumb;
  if (Name == "x86")
    return x86;
  if (Name == "x86-64")
    return x86_64;
  if (Name == "xcore")
    return xcore;

  return UnknownArch;
}

//

void Triple::Parse() const {
  assert(!isInitialized() && "Invalid parse call.");

  StringRef ArchName = getArchName();
  StringRef VendorName = getVendorName();
  StringRef OSName = getOSName();

  if (ArchName.size() == 4 && ArchName[0] == 'i' && 
      ArchName[2] == '8' && ArchName[3] == '6' && 
      ArchName[1] - '3' < 6) // i[3-9]86
    Arch = x86;
  else if (ArchName == "amd64" || ArchName == "x86_64")
    Arch = x86_64;
  else if (ArchName == "bfin")
    Arch = bfin;
  else if (ArchName == "pic16")
    Arch = pic16;
  else if (ArchName == "powerpc")
    Arch = ppc;
  else if (ArchName == "powerpc64")
    Arch = ppc64;
  else if (ArchName == "arm" ||
           ArchName.startswith("armv") ||
           ArchName == "xscale")
    Arch = arm;
  else if (ArchName == "thumb" ||
           ArchName.startswith("thumbv"))
    Arch = thumb;
  else if (ArchName.startswith("alpha"))
    Arch = alpha;
  else if (ArchName == "spu" || ArchName == "cellspu")
    Arch = cellspu;
  else if (ArchName == "msp430")
    Arch = msp430;
  else if (ArchName == "mips" || ArchName == "mipsallegrex")
    Arch = mips;
  else if (ArchName == "mipsel" || ArchName == "mipsallegrexel" ||
           ArchName == "psp")
    Arch = mipsel;
  else if (ArchName == "sparc")
    Arch = sparc;
  else if (ArchName == "s390x")
    Arch = systemz;
  else if (ArchName == "tce")
    Arch = tce;
  else
    Arch = UnknownArch;


  // Handle some exceptional cases where the OS / environment components are
  // stuck into the vendor field.
  if (StringRef(getTriple()).count('-') == 1) {
    StringRef VendorName = getVendorName();

    if (VendorName.startswith("mingw32")) { // 'i386-mingw32', etc.
      Vendor = PC;
      OS = MinGW32;
      return;
    }

    // arm-elf is another example, but we don't currently parse anything about
    // the environment.
  }

  if (VendorName == "apple")
    Vendor = Apple;
  else if (VendorName == "pc")
    Vendor = PC;
  else
    Vendor = UnknownVendor;

  if (OSName.startswith("auroraux"))
    OS = AuroraUX;
  else if (OSName.startswith("cygwin"))
    OS = Cygwin;
  else if (OSName.startswith("darwin"))
    OS = Darwin;
  else if (OSName.startswith("dragonfly"))
    OS = DragonFly;
  else if (OSName.startswith("freebsd"))
    OS = FreeBSD;
  else if (OSName.startswith("linux"))
    OS = Linux;
  else if (OSName.startswith("mingw32"))
    OS = MinGW32;
  else if (OSName.startswith("mingw64"))
    OS = MinGW64;
  else if (OSName.startswith("netbsd"))
    OS = NetBSD;
  else if (OSName.startswith("openbsd"))
    OS = OpenBSD;
  else if (OSName.startswith("solaris"))
    OS = Solaris;
  else if (OSName.startswith("win32"))
    OS = Win32;
  else
    OS = UnknownOS;

  assert(isInitialized() && "Failed to initialize!");
}

StringRef Triple::getArchName() const {
  return StringRef(Data).split('-').first;           // Isolate first component
}

StringRef Triple::getVendorName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  return Tmp.split('-').first;                       // Isolate second component
}

StringRef Triple::getOSName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  Tmp = Tmp.split('-').second;                       // Strip second component
  return Tmp.split('-').first;                       // Isolate third component
}

StringRef Triple::getEnvironmentName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  Tmp = Tmp.split('-').second;                       // Strip second component
  return Tmp.split('-').second;                      // Strip third component
}

StringRef Triple::getOSAndEnvironmentName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  return Tmp.split('-').second;                      // Strip second component
}

static unsigned EatNumber(StringRef &Str) {
  assert(!Str.empty() && Str[0] >= '0' && Str[0] <= '9' && "Not a number");
  unsigned Result = Str[0]-'0';
  
  // Eat the digit.
  Str = Str.substr(1);
  
  // Handle "darwin11".
  if (Result == 1 && !Str.empty() && Str[0] >= '0' && Str[0] <= '9') {
    Result = Result*10 + (Str[0] - '0');
    // Eat the digit.
    Str = Str.substr(1);
  }
  
  return Result;
}

/// getDarwinNumber - Parse the 'darwin number' out of the specific target
/// triple.  For example, if we have darwin8.5 return 8,5,0.  If any entry is
/// not defined, return 0's.  This requires that the triple have an OSType of
/// darwin before it is called.
void Triple::getDarwinNumber(unsigned &Maj, unsigned &Min,
                             unsigned &Revision) const {
  assert(getOS() == Darwin && "Not a darwin target triple!");
  StringRef OSName = getOSName();
  assert(OSName.startswith("darwin") && "Unknown darwin target triple!");
  
  // Strip off "darwin".
  OSName = OSName.substr(6);
  
  Maj = Min = Revision = 0;

  if (OSName.empty() || OSName[0] < '0' || OSName[0] > '9')
    return;

  // The major version is the first digit.
  Maj = EatNumber(OSName);
  if (OSName.empty()) return;
  
  // Handle minor version: 10.4.9 -> darwin8.9.
  if (OSName[0] != '.')
    return;
  
  // Eat the '.'.
  OSName = OSName.substr(1);

  if (OSName.empty() || OSName[0] < '0' || OSName[0] > '9')
    return;
  
  Min = EatNumber(OSName);
  if (OSName.empty()) return;

  // Handle revision darwin8.9.1
  if (OSName[0] != '.')
    return;
  
  // Eat the '.'.
  OSName = OSName.substr(1);
  
  if (OSName.empty() || OSName[0] < '0' || OSName[0] > '9')
    return;

  Revision = EatNumber(OSName);
}

void Triple::setTriple(const Twine &Str) {
  Data = Str.str();
  Arch = InvalidArch;
}

void Triple::setArch(ArchType Kind) {
  setArchName(getArchTypeName(Kind));
}

void Triple::setVendor(VendorType Kind) {
  setVendorName(getVendorTypeName(Kind));
}

void Triple::setOS(OSType Kind) {
  setOSName(getOSTypeName(Kind));
}

void Triple::setArchName(const StringRef &Str) {
  // Work around a miscompilation bug in gcc 4.0.3.
  Twine a = getVendorName() + "-" + getOSAndEnvironmentName();
  Twine b = Str + "-" + a;
  setTriple(b);
}

void Triple::setVendorName(const StringRef &Str) {
  setTriple(getArchName() + "-" + Str + "-" + getOSAndEnvironmentName());
}

void Triple::setOSName(const StringRef &Str) {
  if (hasEnvironment())
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str +
              "-" + getEnvironmentName());
  else
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

void Triple::setEnvironmentName(const StringRef &Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + getOSName() + 
            "-" + Str);
}

void Triple::setOSAndEnvironmentName(const StringRef &Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}
