//===--- Triple.cpp - Target triple helper class --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstring>
using namespace llvm;

const char *Triple::getArchTypeName(ArchType Kind) {
  switch (Kind) {
  case UnknownArch: return "unknown";

  case arm:     return "arm";
  case cellspu: return "cellspu";
  case hexagon: return "hexagon";
  case mips:    return "mips";
  case mipsel:  return "mipsel";
  case mips64:  return "mips64";
  case mips64el:return "mips64el";
  case msp430:  return "msp430";
  case ppc64:   return "powerpc64";
  case ppc:     return "powerpc";
  case r600:    return "r600";
  case sparc:   return "sparc";
  case sparcv9: return "sparcv9";
  case tce:     return "tce";
  case thumb:   return "thumb";
  case x86:     return "i386";
  case x86_64:  return "x86_64";
  case xcore:   return "xcore";
  case mblaze:  return "mblaze";
  case nvptx:   return "nvptx";
  case nvptx64: return "nvptx64";
  case le32:    return "le32";
  case amdil:   return "amdil";
  case spir:    return "spir";
  }

  llvm_unreachable("Invalid ArchType!");
}

const char *Triple::getArchTypePrefix(ArchType Kind) {
  switch (Kind) {
  default:
    return 0;

  case arm:
  case thumb:   return "arm";

  case cellspu: return "spu";

  case ppc64:
  case ppc:     return "ppc";

  case mblaze:  return "mblaze";

  case mips:
  case mipsel:
  case mips64:
  case mips64el:return "mips";

  case hexagon: return "hexagon";

  case r600:    return "r600";

  case sparcv9:
  case sparc:   return "sparc";

  case x86:
  case x86_64:  return "x86";

  case xcore:   return "xcore";

  case nvptx:   return "nvptx";
  case nvptx64: return "nvptx";
  case le32:    return "le32";
  case amdil:   return "amdil";
  case spir:    return "spir";
  }
}

const char *Triple::getVendorTypeName(VendorType Kind) {
  switch (Kind) {
  case UnknownVendor: return "unknown";

  case Apple: return "apple";
  case PC: return "pc";
  case SCEI: return "scei";
  case BGP: return "bgp";
  case BGQ: return "bgq";
  case Freescale: return "fsl";
  case IBM: return "ibm";
  }

  llvm_unreachable("Invalid VendorType!");
}

const char *Triple::getOSTypeName(OSType Kind) {
  switch (Kind) {
  case UnknownOS: return "unknown";

  case AuroraUX: return "auroraux";
  case Cygwin: return "cygwin";
  case Darwin: return "darwin";
  case DragonFly: return "dragonfly";
  case FreeBSD: return "freebsd";
  case IOS: return "ios";
  case KFreeBSD: return "kfreebsd";
  case Linux: return "linux";
  case Lv2: return "lv2";
  case MacOSX: return "macosx";
  case MinGW32: return "mingw32";
  case NetBSD: return "netbsd";
  case OpenBSD: return "openbsd";
  case Solaris: return "solaris";
  case Win32: return "win32";
  case Haiku: return "haiku";
  case Minix: return "minix";
  case RTEMS: return "rtems";
  case NativeClient: return "nacl";
  case CNK: return "cnk";
  case Bitrig: return "bitrig";
  case AIX: return "aix";
  }

  llvm_unreachable("Invalid OSType");
}

const char *Triple::getEnvironmentTypeName(EnvironmentType Kind) {
  switch (Kind) {
  case UnknownEnvironment: return "unknown";
  case GNU: return "gnu";
  case GNUEABIHF: return "gnueabihf";
  case GNUEABI: return "gnueabi";
  case EABI: return "eabi";
  case MachO: return "macho";
  case Android: return "android";
  case ELF: return "elf";
  }

  llvm_unreachable("Invalid EnvironmentType!");
}

Triple::ArchType Triple::getArchTypeForLLVMName(StringRef Name) {
  return StringSwitch<Triple::ArchType>(Name)
    .Case("arm", arm)
    .Case("cellspu", cellspu)
    .Case("mips", mips)
    .Case("mipsel", mipsel)
    .Case("mips64", mips64)
    .Case("mips64el", mips64el)
    .Case("msp430", msp430)
    .Case("ppc64", ppc64)
    .Case("ppc32", ppc)
    .Case("ppc", ppc)
    .Case("mblaze", mblaze)
    .Case("r600", r600)
    .Case("hexagon", hexagon)
    .Case("sparc", sparc)
    .Case("sparcv9", sparcv9)
    .Case("tce", tce)
    .Case("thumb", thumb)
    .Case("x86", x86)
    .Case("x86-64", x86_64)
    .Case("xcore", xcore)
    .Case("nvptx", nvptx)
    .Case("nvptx64", nvptx64)
    .Case("le32", le32)
    .Case("amdil", amdil)
    .Case("spir", spir)
    .Default(UnknownArch);
}

// Returns architecture name that is understood by the target assembler.
const char *Triple::getArchNameForAssembler() {
  if (!isOSDarwin() && getVendor() != Triple::Apple)
    return NULL;

  return StringSwitch<const char*>(getArchName())
    .Case("i386", "i386")
    .Case("x86_64", "x86_64")
    .Case("powerpc", "ppc")
    .Case("powerpc64", "ppc64")
    .Cases("mblaze", "microblaze", "mblaze")
    .Case("arm", "arm")
    .Cases("armv4t", "thumbv4t", "armv4t")
    .Cases("armv5", "armv5e", "thumbv5", "thumbv5e", "armv5")
    .Cases("armv6", "thumbv6", "armv6")
    .Cases("armv7", "thumbv7", "armv7")
    .Case("r600", "r600")
    .Case("nvptx", "nvptx")
    .Case("nvptx64", "nvptx64")
    .Case("le32", "le32")
    .Case("amdil", "amdil")
    .Case("spir", "spir")
    .Default(NULL);
}

static Triple::ArchType parseArch(StringRef ArchName) {
  return StringSwitch<Triple::ArchType>(ArchName)
    .Cases("i386", "i486", "i586", "i686", Triple::x86)
    // FIXME: Do we need to support these?
    .Cases("i786", "i886", "i986", Triple::x86)
    .Cases("amd64", "x86_64", Triple::x86_64)
    .Case("powerpc", Triple::ppc)
    .Cases("powerpc64", "ppu", Triple::ppc64)
    .Case("mblaze", Triple::mblaze)
    .Cases("arm", "xscale", Triple::arm)
    // FIXME: It would be good to replace these with explicit names for all the
    // various suffixes supported.
    .StartsWith("armv", Triple::arm)
    .Case("thumb", Triple::thumb)
    .StartsWith("thumbv", Triple::thumb)
    .Cases("spu", "cellspu", Triple::cellspu)
    .Case("msp430", Triple::msp430)
    .Cases("mips", "mipseb", "mipsallegrex", Triple::mips)
    .Cases("mipsel", "mipsallegrexel", Triple::mipsel)
    .Cases("mips64", "mips64eb", Triple::mips64)
    .Case("mips64el", Triple::mips64el)
    .Case("r600", Triple::r600)
    .Case("hexagon", Triple::hexagon)
    .Case("sparc", Triple::sparc)
    .Case("sparcv9", Triple::sparcv9)
    .Case("tce", Triple::tce)
    .Case("xcore", Triple::xcore)
    .Case("nvptx", Triple::nvptx)
    .Case("nvptx64", Triple::nvptx64)
    .Case("le32", Triple::le32)
    .Case("amdil", Triple::amdil)
    .Case("spir", Triple::spir)
    .Default(Triple::UnknownArch);
}

static Triple::VendorType parseVendor(StringRef VendorName) {
  return StringSwitch<Triple::VendorType>(VendorName)
    .Case("apple", Triple::Apple)
    .Case("pc", Triple::PC)
    .Case("scei", Triple::SCEI)
    .Case("bgp", Triple::BGP)
    .Case("bgq", Triple::BGQ)
    .Case("fsl", Triple::Freescale)
    .Case("ibm", Triple::IBM)
    .Default(Triple::UnknownVendor);
}

static Triple::OSType parseOS(StringRef OSName) {
  return StringSwitch<Triple::OSType>(OSName)
    .StartsWith("auroraux", Triple::AuroraUX)
    .StartsWith("cygwin", Triple::Cygwin)
    .StartsWith("darwin", Triple::Darwin)
    .StartsWith("dragonfly", Triple::DragonFly)
    .StartsWith("freebsd", Triple::FreeBSD)
    .StartsWith("ios", Triple::IOS)
    .StartsWith("kfreebsd", Triple::KFreeBSD)
    .StartsWith("linux", Triple::Linux)
    .StartsWith("lv2", Triple::Lv2)
    .StartsWith("macosx", Triple::MacOSX)
    .StartsWith("mingw32", Triple::MinGW32)
    .StartsWith("netbsd", Triple::NetBSD)
    .StartsWith("openbsd", Triple::OpenBSD)
    .StartsWith("solaris", Triple::Solaris)
    .StartsWith("win32", Triple::Win32)
    .StartsWith("haiku", Triple::Haiku)
    .StartsWith("minix", Triple::Minix)
    .StartsWith("rtems", Triple::RTEMS)
    .StartsWith("nacl", Triple::NativeClient)
    .StartsWith("cnk", Triple::CNK)
    .StartsWith("bitrig", Triple::Bitrig)
    .StartsWith("aix", Triple::AIX)
    .Default(Triple::UnknownOS);
}

static Triple::EnvironmentType parseEnvironment(StringRef EnvironmentName) {
  return StringSwitch<Triple::EnvironmentType>(EnvironmentName)
    .StartsWith("eabi", Triple::EABI)
    .StartsWith("gnueabihf", Triple::GNUEABIHF)
    .StartsWith("gnueabi", Triple::GNUEABI)
    .StartsWith("gnu", Triple::GNU)
    .StartsWith("macho", Triple::MachO)
    .StartsWith("android", Triple::Android)
    .StartsWith("elf", Triple::ELF)
    .Default(Triple::UnknownEnvironment);
}

/// \brief Construct a triple from the string representation provided.
///
/// This stores the string representation and parses the various pieces into
/// enum members.
Triple::Triple(const Twine &Str)
    : Data(Str.str()),
      Arch(parseArch(getArchName())),
      Vendor(parseVendor(getVendorName())),
      OS(parseOS(getOSName())),
      Environment(parseEnvironment(getEnvironmentName())) {
}

/// \brief Construct a triple from string representations of the architecture,
/// vendor, and OS.
///
/// This joins each argument into a canonical string representation and parses
/// them into enum members. It leaves the environment unknown and omits it from
/// the string representation.
Triple::Triple(const Twine &ArchStr, const Twine &VendorStr, const Twine &OSStr)
    : Data((ArchStr + Twine('-') + VendorStr + Twine('-') + OSStr).str()),
      Arch(parseArch(ArchStr.str())),
      Vendor(parseVendor(VendorStr.str())),
      OS(parseOS(OSStr.str())),
      Environment() {
}

/// \brief Construct a triple from string representations of the architecture,
/// vendor, OS, and environment.
///
/// This joins each argument into a canonical string representation and parses
/// them into enum members.
Triple::Triple(const Twine &ArchStr, const Twine &VendorStr, const Twine &OSStr,
               const Twine &EnvironmentStr)
    : Data((ArchStr + Twine('-') + VendorStr + Twine('-') + OSStr + Twine('-') +
            EnvironmentStr).str()),
      Arch(parseArch(ArchStr.str())),
      Vendor(parseVendor(VendorStr.str())),
      OS(parseOS(OSStr.str())),
      Environment(parseEnvironment(EnvironmentStr.str())) {
}

std::string Triple::normalize(StringRef Str) {
  // Parse into components.
  SmallVector<StringRef, 4> Components;
  Str.split(Components, "-");

  // If the first component corresponds to a known architecture, preferentially
  // use it for the architecture.  If the second component corresponds to a
  // known vendor, preferentially use it for the vendor, etc.  This avoids silly
  // component movement when a component parses as (eg) both a valid arch and a
  // valid os.
  ArchType Arch = UnknownArch;
  if (Components.size() > 0)
    Arch = parseArch(Components[0]);
  VendorType Vendor = UnknownVendor;
  if (Components.size() > 1)
    Vendor = parseVendor(Components[1]);
  OSType OS = UnknownOS;
  if (Components.size() > 2)
    OS = parseOS(Components[2]);
  EnvironmentType Environment = UnknownEnvironment;
  if (Components.size() > 3)
    Environment = parseEnvironment(Components[3]);

  // Note which components are already in their final position.  These will not
  // be moved.
  bool Found[4];
  Found[0] = Arch != UnknownArch;
  Found[1] = Vendor != UnknownVendor;
  Found[2] = OS != UnknownOS;
  Found[3] = Environment != UnknownEnvironment;

  // If they are not there already, permute the components into their canonical
  // positions by seeing if they parse as a valid architecture, and if so moving
  // the component to the architecture position etc.
  for (unsigned Pos = 0; Pos != array_lengthof(Found); ++Pos) {
    if (Found[Pos])
      continue; // Already in the canonical position.

    for (unsigned Idx = 0; Idx != Components.size(); ++Idx) {
      // Do not reparse any components that already matched.
      if (Idx < array_lengthof(Found) && Found[Idx])
        continue;

      // Does this component parse as valid for the target position?
      bool Valid = false;
      StringRef Comp = Components[Idx];
      switch (Pos) {
      default: llvm_unreachable("unexpected component type!");
      case 0:
        Arch = parseArch(Comp);
        Valid = Arch != UnknownArch;
        break;
      case 1:
        Vendor = parseVendor(Comp);
        Valid = Vendor != UnknownVendor;
        break;
      case 2:
        OS = parseOS(Comp);
        Valid = OS != UnknownOS;
        break;
      case 3:
        Environment = parseEnvironment(Comp);
        Valid = Environment != UnknownEnvironment;
        break;
      }
      if (!Valid)
        continue; // Nope, try the next component.

      // Move the component to the target position, pushing any non-fixed
      // components that are in the way to the right.  This tends to give
      // good results in the common cases of a forgotten vendor component
      // or a wrongly positioned environment.
      if (Pos < Idx) {
        // Insert left, pushing the existing components to the right.  For
        // example, a-b-i386 -> i386-a-b when moving i386 to the front.
        StringRef CurrentComponent(""); // The empty component.
        // Replace the component we are moving with an empty component.
        std::swap(CurrentComponent, Components[Idx]);
        // Insert the component being moved at Pos, displacing any existing
        // components to the right.
        for (unsigned i = Pos; !CurrentComponent.empty(); ++i) {
          // Skip over any fixed components.
          while (i < array_lengthof(Found) && Found[i])
            ++i;
          // Place the component at the new position, getting the component
          // that was at this position - it will be moved right.
          std::swap(CurrentComponent, Components[i]);
        }
      } else if (Pos > Idx) {
        // Push right by inserting empty components until the component at Idx
        // reaches the target position Pos.  For example, pc-a -> -pc-a when
        // moving pc to the second position.
        do {
          // Insert one empty component at Idx.
          StringRef CurrentComponent(""); // The empty component.
          for (unsigned i = Idx; i < Components.size();) {
            // Place the component at the new position, getting the component
            // that was at this position - it will be moved right.
            std::swap(CurrentComponent, Components[i]);
            // If it was placed on top of an empty component then we are done.
            if (CurrentComponent.empty())
              break;
            // Advance to the next component, skipping any fixed components.
            while (++i < array_lengthof(Found) && Found[i])
              ;
          }
          // The last component was pushed off the end - append it.
          if (!CurrentComponent.empty())
            Components.push_back(CurrentComponent);

          // Advance Idx to the component's new position.
          while (++Idx < array_lengthof(Found) && Found[Idx])
            ;
        } while (Idx < Pos); // Add more until the final position is reached.
      }
      assert(Pos < Components.size() && Components[Pos] == Comp &&
             "Component moved wrong!");
      Found[Pos] = true;
      break;
    }
  }

  // Special case logic goes here.  At this point Arch, Vendor and OS have the
  // correct values for the computed components.

  // Stick the corrected components back together to form the normalized string.
  std::string Normalized;
  for (unsigned i = 0, e = Components.size(); i != e; ++i) {
    if (i) Normalized += '-';
    Normalized += Components[i];
  }
  return Normalized;
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
  unsigned Result = 0;

  do {
    // Consume the leading digit.
    Result = Result*10 + (Str[0] - '0');

    // Eat the digit.
    Str = Str.substr(1);
  } while (!Str.empty() && Str[0] >= '0' && Str[0] <= '9');

  return Result;
}

void Triple::getOSVersion(unsigned &Major, unsigned &Minor,
                          unsigned &Micro) const {
  StringRef OSName = getOSName();

  // Assume that the OS portion of the triple starts with the canonical name.
  StringRef OSTypeName = getOSTypeName(getOS());
  if (OSName.startswith(OSTypeName))
    OSName = OSName.substr(OSTypeName.size());

  // Any unset version defaults to 0.
  Major = Minor = Micro = 0;

  // Parse up to three components.
  unsigned *Components[3] = { &Major, &Minor, &Micro };
  for (unsigned i = 0; i != 3; ++i) {
    if (OSName.empty() || OSName[0] < '0' || OSName[0] > '9')
      break;

    // Consume the leading number.
    *Components[i] = EatNumber(OSName);

    // Consume the separator, if present.
    if (OSName.startswith("."))
      OSName = OSName.substr(1);
  }
}

bool Triple::getMacOSXVersion(unsigned &Major, unsigned &Minor,
                              unsigned &Micro) const {
  getOSVersion(Major, Minor, Micro);

  switch (getOS()) {
  default: llvm_unreachable("unexpected OS for Darwin triple");
  case Darwin:
    // Default to darwin8, i.e., MacOSX 10.4.
    if (Major == 0)
      Major = 8;
    // Darwin version numbers are skewed from OS X versions.
    if (Major < 4)
      return false;
    Micro = 0;
    Minor = Major - 4;
    Major = 10;
    break;
  case MacOSX:
    // Default to 10.4.
    if (Major == 0) {
      Major = 10;
      Minor = 4;
    }
    if (Major != 10)
      return false;
    break;
  case IOS:
    // Ignore the version from the triple.  This is only handled because the
    // the clang driver combines OS X and IOS support into a common Darwin
    // toolchain that wants to know the OS X version number even when targeting
    // IOS.
    Major = 10;
    Minor = 4;
    Micro = 0;
    break;
  }
  return true;
}

void Triple::getiOSVersion(unsigned &Major, unsigned &Minor,
                           unsigned &Micro) const {
  switch (getOS()) {
  default: llvm_unreachable("unexpected OS for Darwin triple");
  case Darwin:
  case MacOSX:
    // Ignore the version from the triple.  This is only handled because the
    // the clang driver combines OS X and IOS support into a common Darwin
    // toolchain that wants to know the iOS version number even when targeting
    // OS X.
    Major = 3;
    Minor = 0;
    Micro = 0;
    break;
  case IOS:
    getOSVersion(Major, Minor, Micro);
    // Default to 3.0.
    if (Major == 0)
      Major = 3;
    break;
  }
}

void Triple::setTriple(const Twine &Str) {
  *this = Triple(Str);
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

void Triple::setEnvironment(EnvironmentType Kind) {
  setEnvironmentName(getEnvironmentTypeName(Kind));
}

void Triple::setArchName(StringRef Str) {
  // Work around a miscompilation bug for Twines in gcc 4.0.3.
  SmallString<64> Triple;
  Triple += Str;
  Triple += "-";
  Triple += getVendorName();
  Triple += "-";
  Triple += getOSAndEnvironmentName();
  setTriple(Triple.str());
}

void Triple::setVendorName(StringRef Str) {
  setTriple(getArchName() + "-" + Str + "-" + getOSAndEnvironmentName());
}

void Triple::setOSName(StringRef Str) {
  if (hasEnvironment())
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str +
              "-" + getEnvironmentName());
  else
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

void Triple::setEnvironmentName(StringRef Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + getOSName() +
            "-" + Str);
}

void Triple::setOSAndEnvironmentName(StringRef Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

static unsigned getArchPointerBitWidth(llvm::Triple::ArchType Arch) {
  switch (Arch) {
  case llvm::Triple::spir:
  case llvm::Triple::UnknownArch:
    return 0;

  case llvm::Triple::msp430:
    return 16;

  case llvm::Triple::amdil:
  case llvm::Triple::arm:
  case llvm::Triple::cellspu:
  case llvm::Triple::hexagon:
  case llvm::Triple::le32:
  case llvm::Triple::mblaze:
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::nvptx:
  case llvm::Triple::ppc:
  case llvm::Triple::r600:
  case llvm::Triple::sparc:
  case llvm::Triple::tce:
  case llvm::Triple::thumb:
  case llvm::Triple::x86:
  case llvm::Triple::xcore:
    return 32;

  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
  case llvm::Triple::nvptx64:
  case llvm::Triple::ppc64:
  case llvm::Triple::sparcv9:
  case llvm::Triple::x86_64:
    return 64;
  }
  llvm_unreachable("Invalid architecture value");
}

bool Triple::isArch64Bit() const {
  return getArchPointerBitWidth(getArch()) == 64;
}

bool Triple::isArch32Bit() const {
  return getArchPointerBitWidth(getArch()) == 32;
}

bool Triple::isArch16Bit() const {
  return getArchPointerBitWidth(getArch()) == 16;
}

Triple Triple::get32BitArchVariant() const {
  Triple T(*this);
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::msp430:
    T.setArch(UnknownArch);
    break;

  case Triple::amdil:
  case Triple::spir:
  case Triple::arm:
  case Triple::cellspu:
  case Triple::hexagon:
  case Triple::le32:
  case Triple::mblaze:
  case Triple::mips:
  case Triple::mipsel:
  case Triple::nvptx:
  case Triple::ppc:
  case Triple::r600:
  case Triple::sparc:
  case Triple::tce:
  case Triple::thumb:
  case Triple::x86:
  case Triple::xcore:
    // Already 32-bit.
    break;

  case Triple::mips64:    T.setArch(Triple::mips);    break;
  case Triple::mips64el:  T.setArch(Triple::mipsel);  break;
  case Triple::nvptx64:   T.setArch(Triple::nvptx);   break;
  case Triple::ppc64:     T.setArch(Triple::ppc);   break;
  case Triple::sparcv9:   T.setArch(Triple::sparc);   break;
  case Triple::x86_64:    T.setArch(Triple::x86);     break;
  }
  return T;
}

Triple Triple::get64BitArchVariant() const {
  Triple T(*this);
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::amdil:
  case Triple::arm:
  case Triple::cellspu:
  case Triple::hexagon:
  case Triple::le32:
  case Triple::mblaze:
  case Triple::msp430:
  case Triple::r600:
  case Triple::tce:
  case Triple::thumb:
  case Triple::xcore:
    T.setArch(UnknownArch);
    break;

  case Triple::spir:
  case Triple::mips64:
  case Triple::mips64el:
  case Triple::nvptx64:
  case Triple::ppc64:
  case Triple::sparcv9:
  case Triple::x86_64:
    // Already 64-bit.
    break;

  case Triple::mips:    T.setArch(Triple::mips64);    break;
  case Triple::mipsel:  T.setArch(Triple::mips64el);  break;
  case Triple::nvptx:   T.setArch(Triple::nvptx64);   break;
  case Triple::ppc:     T.setArch(Triple::ppc64);     break;
  case Triple::sparc:   T.setArch(Triple::sparcv9);   break;
  case Triple::x86:     T.setArch(Triple::x86_64);    break;
  }
  return T;
}
