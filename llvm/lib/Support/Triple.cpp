//===--- Triple.cpp - Target triple helper class --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstring>
using namespace llvm;

const char *Triple::getArchTypeName(ArchType Kind) {
  switch (Kind) {
  case UnknownArch: return "unknown";

  case aarch64:     return "aarch64";
  case aarch64_be:  return "aarch64_be";
  case arm:         return "arm";
  case armeb:       return "armeb";
  case hexagon:     return "hexagon";
  case mips:        return "mips";
  case mipsel:      return "mipsel";
  case mips64:      return "mips64";
  case mips64el:    return "mips64el";
  case msp430:      return "msp430";
  case ppc64:       return "powerpc64";
  case ppc64le:     return "powerpc64le";
  case ppc:         return "powerpc";
  case r600:        return "r600";
  case amdgcn:      return "amdgcn";
  case sparc:       return "sparc";
  case sparcv9:     return "sparcv9";
  case systemz:     return "s390x";
  case tce:         return "tce";
  case thumb:       return "thumb";
  case thumbeb:     return "thumbeb";
  case x86:         return "i386";
  case x86_64:      return "x86_64";
  case xcore:       return "xcore";
  case nvptx:       return "nvptx";
  case nvptx64:     return "nvptx64";
  case le32:        return "le32";
  case le64:        return "le64";
  case amdil:       return "amdil";
  case amdil64:     return "amdil64";
  case hsail:       return "hsail";
  case hsail64:     return "hsail64";
  case spir:        return "spir";
  case spir64:      return "spir64";
  case kalimba:     return "kalimba";
  }

  llvm_unreachable("Invalid ArchType!");
}

const char *Triple::getArchTypePrefix(ArchType Kind) {
  switch (Kind) {
  default:
    return nullptr;

  case aarch64:
  case aarch64_be:  return "aarch64";

  case arm:
  case armeb:
  case thumb:
  case thumbeb:     return "arm";

  case ppc64:
  case ppc64le:
  case ppc:         return "ppc";

  case mips:
  case mipsel:
  case mips64:
  case mips64el:    return "mips";

  case hexagon:     return "hexagon";

  case amdgcn:
  case r600:        return "amdgpu";

  case sparcv9:
  case sparc:       return "sparc";

  case systemz:     return "systemz";

  case x86:
  case x86_64:      return "x86";

  case xcore:       return "xcore";

  case nvptx:       return "nvptx";
  case nvptx64:     return "nvptx";

  case le32:        return "le32";
  case le64:        return "le64";

  case amdil:
  case amdil64:     return "amdil";

  case hsail:
  case hsail64:     return "hsail";

  case spir:
  case spir64:      return "spir";
  case kalimba:     return "kalimba";
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
  case ImaginationTechnologies: return "img";
  case MipsTechnologies: return "mti";
  case NVIDIA: return "nvidia";
  case CSR: return "csr";
  }

  llvm_unreachable("Invalid VendorType!");
}

const char *Triple::getOSTypeName(OSType Kind) {
  switch (Kind) {
  case UnknownOS: return "unknown";

  case Darwin: return "darwin";
  case DragonFly: return "dragonfly";
  case FreeBSD: return "freebsd";
  case IOS: return "ios";
  case KFreeBSD: return "kfreebsd";
  case Linux: return "linux";
  case Lv2: return "lv2";
  case MacOSX: return "macosx";
  case NetBSD: return "netbsd";
  case OpenBSD: return "openbsd";
  case Solaris: return "solaris";
  case Win32: return "windows";
  case Haiku: return "haiku";
  case Minix: return "minix";
  case RTEMS: return "rtems";
  case NaCl: return "nacl";
  case CNK: return "cnk";
  case Bitrig: return "bitrig";
  case AIX: return "aix";
  case CUDA: return "cuda";
  case NVCL: return "nvcl";
  case AMDHSA: return "amdhsa";
  }

  llvm_unreachable("Invalid OSType");
}

const char *Triple::getEnvironmentTypeName(EnvironmentType Kind) {
  switch (Kind) {
  case UnknownEnvironment: return "unknown";
  case GNU: return "gnu";
  case GNUEABIHF: return "gnueabihf";
  case GNUEABI: return "gnueabi";
  case GNUX32: return "gnux32";
  case CODE16: return "code16";
  case EABI: return "eabi";
  case EABIHF: return "eabihf";
  case Android: return "android";
  case MSVC: return "msvc";
  case Itanium: return "itanium";
  case Cygnus: return "cygnus";
  }

  llvm_unreachable("Invalid EnvironmentType!");
}

Triple::ArchType Triple::getArchTypeForLLVMName(StringRef Name) {
  return StringSwitch<Triple::ArchType>(Name)
    .Case("aarch64", aarch64)
    .Case("aarch64_be", aarch64_be)
    .Case("arm64", aarch64) // "arm64" is an alias for "aarch64"
    .Case("arm", arm)
    .Case("armeb", armeb)
    .Case("mips", mips)
    .Case("mipsel", mipsel)
    .Case("mips64", mips64)
    .Case("mips64el", mips64el)
    .Case("msp430", msp430)
    .Case("ppc64", ppc64)
    .Case("ppc32", ppc)
    .Case("ppc", ppc)
    .Case("ppc64le", ppc64le)
    .Case("r600", r600)
    .Case("amdgcn", amdgcn)
    .Case("hexagon", hexagon)
    .Case("sparc", sparc)
    .Case("sparcv9", sparcv9)
    .Case("systemz", systemz)
    .Case("tce", tce)
    .Case("thumb", thumb)
    .Case("thumbeb", thumbeb)
    .Case("x86", x86)
    .Case("x86-64", x86_64)
    .Case("xcore", xcore)
    .Case("nvptx", nvptx)
    .Case("nvptx64", nvptx64)
    .Case("le32", le32)
    .Case("le64", le64)
    .Case("amdil", amdil)
    .Case("amdil64", amdil64)
    .Case("hsail", hsail)
    .Case("hsail64", hsail64)
    .Case("spir", spir)
    .Case("spir64", spir64)
    .Case("kalimba", kalimba)
    .Default(UnknownArch);
}

static Triple::ArchType parseARMArch(StringRef ArchName) {
  size_t offset = StringRef::npos;
  Triple::ArchType arch = Triple::UnknownArch;
  bool isThumb = ArchName.startswith("thumb");

  if (ArchName.equals("arm"))
    return Triple::arm;
  if (ArchName.equals("armeb"))
    return Triple::armeb;
  if (ArchName.equals("thumb"))
    return Triple::thumb;
  if (ArchName.equals("thumbeb"))
    return Triple::thumbeb;
  if (ArchName.equals("arm64") || ArchName.equals("aarch64"))
    return Triple::aarch64;
  if (ArchName.equals("aarch64_be"))
    return Triple::aarch64_be;

  if (ArchName.startswith("armv")) {
    offset = 3;
    if (ArchName.endswith("eb")) {
      arch = Triple::armeb;
      ArchName = ArchName.substr(0, ArchName.size() - 2);
    } else
      arch = Triple::arm;
  } else if (ArchName.startswith("armebv")) {
    offset = 5;
    arch = Triple::armeb;
  } else if (ArchName.startswith("thumbv")) {
    offset = 5;
    if (ArchName.endswith("eb")) {
      arch = Triple::thumbeb;
      ArchName = ArchName.substr(0, ArchName.size() - 2);
    } else
      arch = Triple::thumb;
  } else if (ArchName.startswith("thumbebv")) {
    offset = 7;
    arch = Triple::thumbeb;
  }
  return StringSwitch<Triple::ArchType>(ArchName.substr(offset))
    .Cases("v2", "v2a", isThumb ? Triple::UnknownArch : arch)
    .Cases("v3", "v3m", isThumb ? Triple::UnknownArch : arch)
    .Cases("v4", "v4t", arch)
    .Cases("v5", "v5e", "v5t", "v5te", "v5tej", arch)
    .Cases("v6", "v6j", "v6k", "v6m", arch)
    .Cases("v6t2", "v6z", "v6zk", arch)
    .Cases("v7", "v7a", "v7em", "v7l", arch)
    .Cases("v7m", "v7r", "v7s", arch)
    .Cases("v8", "v8a", arch)
    .Default(Triple::UnknownArch);
}

static Triple::ArchType parseArch(StringRef ArchName) {
  Triple::ArchType ARMArch(parseARMArch(ArchName));

  return StringSwitch<Triple::ArchType>(ArchName)
    .Cases("i386", "i486", "i586", "i686", Triple::x86)
    // FIXME: Do we need to support these?
    .Cases("i786", "i886", "i986", Triple::x86)
    .Cases("amd64", "x86_64", "x86_64h", Triple::x86_64)
    .Case("powerpc", Triple::ppc)
    .Cases("powerpc64", "ppu", Triple::ppc64)
    .Case("powerpc64le", Triple::ppc64le)
    .Case("xscale", Triple::arm)
    .Case("xscaleeb", Triple::armeb)
    .StartsWith("arm", ARMArch)
    .StartsWith("thumb", ARMArch)
    .StartsWith("aarch64", ARMArch)
    .Case("msp430", Triple::msp430)
    .Cases("mips", "mipseb", "mipsallegrex", Triple::mips)
    .Cases("mipsel", "mipsallegrexel", Triple::mipsel)
    .Cases("mips64", "mips64eb", Triple::mips64)
    .Case("mips64el", Triple::mips64el)
    .Case("r600", Triple::r600)
    .Case("amdgcn", Triple::amdgcn)
    .Case("hexagon", Triple::hexagon)
    .Case("s390x", Triple::systemz)
    .Case("sparc", Triple::sparc)
    .Cases("sparcv9", "sparc64", Triple::sparcv9)
    .Case("tce", Triple::tce)
    .Case("xcore", Triple::xcore)
    .Case("nvptx", Triple::nvptx)
    .Case("nvptx64", Triple::nvptx64)
    .Case("le32", Triple::le32)
    .Case("le64", Triple::le64)
    .Case("amdil", Triple::amdil)
    .Case("amdil64", Triple::amdil64)
    .Case("hsail", Triple::hsail)
    .Case("hsail64", Triple::hsail64)
    .Case("spir", Triple::spir)
    .Case("spir64", Triple::spir64)
    .StartsWith("kalimba", Triple::kalimba)
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
    .Case("img", Triple::ImaginationTechnologies)
    .Case("mti", Triple::MipsTechnologies)
    .Case("nvidia", Triple::NVIDIA)
    .Case("csr", Triple::CSR)
    .Default(Triple::UnknownVendor);
}

static Triple::OSType parseOS(StringRef OSName) {
  return StringSwitch<Triple::OSType>(OSName)
    .StartsWith("darwin", Triple::Darwin)
    .StartsWith("dragonfly", Triple::DragonFly)
    .StartsWith("freebsd", Triple::FreeBSD)
    .StartsWith("ios", Triple::IOS)
    .StartsWith("kfreebsd", Triple::KFreeBSD)
    .StartsWith("linux", Triple::Linux)
    .StartsWith("lv2", Triple::Lv2)
    .StartsWith("macosx", Triple::MacOSX)
    .StartsWith("netbsd", Triple::NetBSD)
    .StartsWith("openbsd", Triple::OpenBSD)
    .StartsWith("solaris", Triple::Solaris)
    .StartsWith("win32", Triple::Win32)
    .StartsWith("windows", Triple::Win32)
    .StartsWith("haiku", Triple::Haiku)
    .StartsWith("minix", Triple::Minix)
    .StartsWith("rtems", Triple::RTEMS)
    .StartsWith("nacl", Triple::NaCl)
    .StartsWith("cnk", Triple::CNK)
    .StartsWith("bitrig", Triple::Bitrig)
    .StartsWith("aix", Triple::AIX)
    .StartsWith("cuda", Triple::CUDA)
    .StartsWith("nvcl", Triple::NVCL)
    .StartsWith("amdhsa", Triple::AMDHSA)
    .Default(Triple::UnknownOS);
}

static Triple::EnvironmentType parseEnvironment(StringRef EnvironmentName) {
  return StringSwitch<Triple::EnvironmentType>(EnvironmentName)
    .StartsWith("eabihf", Triple::EABIHF)
    .StartsWith("eabi", Triple::EABI)
    .StartsWith("gnueabihf", Triple::GNUEABIHF)
    .StartsWith("gnueabi", Triple::GNUEABI)
    .StartsWith("gnux32", Triple::GNUX32)
    .StartsWith("code16", Triple::CODE16)
    .StartsWith("gnu", Triple::GNU)
    .StartsWith("android", Triple::Android)
    .StartsWith("msvc", Triple::MSVC)
    .StartsWith("itanium", Triple::Itanium)
    .StartsWith("cygnus", Triple::Cygnus)
    .Default(Triple::UnknownEnvironment);
}

static Triple::ObjectFormatType parseFormat(StringRef EnvironmentName) {
  return StringSwitch<Triple::ObjectFormatType>(EnvironmentName)
    .EndsWith("coff", Triple::COFF)
    .EndsWith("elf", Triple::ELF)
    .EndsWith("macho", Triple::MachO)
    .Default(Triple::UnknownObjectFormat);
}

static Triple::SubArchType parseSubArch(StringRef SubArchName) {
  if (SubArchName.endswith("eb"))
    SubArchName = SubArchName.substr(0, SubArchName.size() - 2);

  return StringSwitch<Triple::SubArchType>(SubArchName)
    .EndsWith("v8", Triple::ARMSubArch_v8)
    .EndsWith("v8a", Triple::ARMSubArch_v8)
    .EndsWith("v7", Triple::ARMSubArch_v7)
    .EndsWith("v7a", Triple::ARMSubArch_v7)
    .EndsWith("v7em", Triple::ARMSubArch_v7em)
    .EndsWith("v7l", Triple::ARMSubArch_v7)
    .EndsWith("v7m", Triple::ARMSubArch_v7m)
    .EndsWith("v7r", Triple::ARMSubArch_v7)
    .EndsWith("v7s", Triple::ARMSubArch_v7s)
    .EndsWith("v6", Triple::ARMSubArch_v6)
    .EndsWith("v6m", Triple::ARMSubArch_v6m)
    .EndsWith("v6t2", Triple::ARMSubArch_v6t2)
    .EndsWith("v5", Triple::ARMSubArch_v5)
    .EndsWith("v5e", Triple::ARMSubArch_v5)
    .EndsWith("v5t", Triple::ARMSubArch_v5)
    .EndsWith("v5te", Triple::ARMSubArch_v5te)
    .EndsWith("v4t", Triple::ARMSubArch_v4t)
    .EndsWith("kalimba3", Triple::KalimbaSubArch_v3)
    .EndsWith("kalimba4", Triple::KalimbaSubArch_v4)
    .EndsWith("kalimba5", Triple::KalimbaSubArch_v5)
    .Default(Triple::NoSubArch);
}

static const char *getObjectFormatTypeName(Triple::ObjectFormatType Kind) {
  switch (Kind) {
  case Triple::UnknownObjectFormat: return "";
  case Triple::COFF: return "coff";
  case Triple::ELF: return "elf";
  case Triple::MachO: return "macho";
  }
  llvm_unreachable("unknown object format type");
}

static Triple::ObjectFormatType getDefaultFormat(const Triple &T) {
  if (T.isOSDarwin())
    return Triple::MachO;
  else if (T.isOSWindows())
    return Triple::COFF;
  return Triple::ELF;
}

/// \brief Construct a triple from the string representation provided.
///
/// This stores the string representation and parses the various pieces into
/// enum members.
Triple::Triple(const Twine &Str)
    : Data(Str.str()),
      Arch(parseArch(getArchName())),
      SubArch(parseSubArch(getArchName())),
      Vendor(parseVendor(getVendorName())),
      OS(parseOS(getOSName())),
      Environment(parseEnvironment(getEnvironmentName())),
      ObjectFormat(parseFormat(getEnvironmentName())) {
  if (ObjectFormat == Triple::UnknownObjectFormat)
    ObjectFormat = getDefaultFormat(*this);
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
      SubArch(parseSubArch(ArchStr.str())),
      Vendor(parseVendor(VendorStr.str())),
      OS(parseOS(OSStr.str())),
      Environment(), ObjectFormat(Triple::UnknownObjectFormat) {
  ObjectFormat = getDefaultFormat(*this);
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
      SubArch(parseSubArch(ArchStr.str())),
      Vendor(parseVendor(VendorStr.str())),
      OS(parseOS(OSStr.str())),
      Environment(parseEnvironment(EnvironmentStr.str())),
      ObjectFormat(parseFormat(EnvironmentStr.str())) {
  if (ObjectFormat == Triple::UnknownObjectFormat)
    ObjectFormat = getDefaultFormat(*this);
}

std::string Triple::normalize(StringRef Str) {
  bool IsMinGW32 = false;
  bool IsCygwin = false;

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
  if (Components.size() > 2) {
    OS = parseOS(Components[2]);
    IsCygwin = Components[2].startswith("cygwin");
    IsMinGW32 = Components[2].startswith("mingw");
  }
  EnvironmentType Environment = UnknownEnvironment;
  if (Components.size() > 3)
    Environment = parseEnvironment(Components[3]);
  ObjectFormatType ObjectFormat = UnknownObjectFormat;
  if (Components.size() > 4)
    ObjectFormat = parseFormat(Components[4]);

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
        IsCygwin = Comp.startswith("cygwin");
        IsMinGW32 = Comp.startswith("mingw");
        Valid = OS != UnknownOS || IsCygwin || IsMinGW32;
        break;
      case 3:
        Environment = parseEnvironment(Comp);
        Valid = Environment != UnknownEnvironment;
        if (!Valid) {
          ObjectFormat = parseFormat(Comp);
          Valid = ObjectFormat != UnknownObjectFormat;
        }
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

  if (OS == Triple::Win32) {
    Components.resize(4);
    Components[2] = "windows";
    if (Environment == UnknownEnvironment) {
      if (ObjectFormat == UnknownObjectFormat || ObjectFormat == Triple::COFF)
        Components[3] = "msvc";
      else
        Components[3] = getObjectFormatTypeName(ObjectFormat);
    }
  } else if (IsMinGW32) {
    Components.resize(4);
    Components[2] = "windows";
    Components[3] = "gnu";
  } else if (IsCygwin) {
    Components.resize(4);
    Components[2] = "windows";
    Components[3] = "cygnus";
  }
  if (IsMinGW32 || IsCygwin ||
      (OS == Triple::Win32 && Environment != UnknownEnvironment)) {
    if (ObjectFormat != UnknownObjectFormat && ObjectFormat != Triple::COFF) {
      Components.resize(5);
      Components[4] = getObjectFormatTypeName(ObjectFormat);
    }
  }

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
    Major = 5;
    Minor = 0;
    Micro = 0;
    break;
  case IOS:
    getOSVersion(Major, Minor, Micro);
    // Default to 5.0 (or 7.0 for arm64).
    if (Major == 0)
      Major = (getArch() == aarch64) ? 7 : 5;
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

void Triple::setObjectFormat(ObjectFormatType Kind) {
  if (Environment == UnknownEnvironment)
    return setEnvironmentName(getObjectFormatTypeName(Kind));

  setEnvironmentName((getEnvironmentTypeName(Environment) + Twine("-") +
                      getObjectFormatTypeName(Kind)).str());
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
  case llvm::Triple::UnknownArch:
    return 0;

  case llvm::Triple::msp430:
    return 16;

  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::hexagon:
  case llvm::Triple::le32:
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::nvptx:
  case llvm::Triple::ppc:
  case llvm::Triple::r600:
  case llvm::Triple::sparc:
  case llvm::Triple::tce:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
  case llvm::Triple::x86:
  case llvm::Triple::xcore:
  case llvm::Triple::amdil:
  case llvm::Triple::hsail:
  case llvm::Triple::spir:
  case llvm::Triple::kalimba:
    return 32;

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::amdgcn:
  case llvm::Triple::le64:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
  case llvm::Triple::nvptx64:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
  case llvm::Triple::sparcv9:
  case llvm::Triple::systemz:
  case llvm::Triple::x86_64:
  case llvm::Triple::amdil64:
  case llvm::Triple::hsail64:
  case llvm::Triple::spir64:
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
  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::amdgcn:
  case Triple::msp430:
  case Triple::systemz:
  case Triple::ppc64le:
    T.setArch(UnknownArch);
    break;

  case Triple::amdil:
  case Triple::hsail:
  case Triple::spir:
  case Triple::arm:
  case Triple::armeb:
  case Triple::hexagon:
  case Triple::kalimba:
  case Triple::le32:
  case Triple::mips:
  case Triple::mipsel:
  case Triple::nvptx:
  case Triple::ppc:
  case Triple::r600:
  case Triple::sparc:
  case Triple::tce:
  case Triple::thumb:
  case Triple::thumbeb:
  case Triple::x86:
  case Triple::xcore:
    // Already 32-bit.
    break;

  case Triple::le64:      T.setArch(Triple::le32);    break;
  case Triple::mips64:    T.setArch(Triple::mips);    break;
  case Triple::mips64el:  T.setArch(Triple::mipsel);  break;
  case Triple::nvptx64:   T.setArch(Triple::nvptx);   break;
  case Triple::ppc64:     T.setArch(Triple::ppc);     break;
  case Triple::sparcv9:   T.setArch(Triple::sparc);   break;
  case Triple::x86_64:    T.setArch(Triple::x86);     break;
  case Triple::amdil64:   T.setArch(Triple::amdil);   break;
  case Triple::hsail64:   T.setArch(Triple::hsail);   break;
  case Triple::spir64:    T.setArch(Triple::spir);    break;
  }
  return T;
}

Triple Triple::get64BitArchVariant() const {
  Triple T(*this);
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::arm:
  case Triple::armeb:
  case Triple::hexagon:
  case Triple::kalimba:
  case Triple::msp430:
  case Triple::r600:
  case Triple::tce:
  case Triple::thumb:
  case Triple::thumbeb:
  case Triple::xcore:
    T.setArch(UnknownArch);
    break;

  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::le64:
  case Triple::amdil64:
  case Triple::amdgcn:
  case Triple::hsail64:
  case Triple::spir64:
  case Triple::mips64:
  case Triple::mips64el:
  case Triple::nvptx64:
  case Triple::ppc64:
  case Triple::ppc64le:
  case Triple::sparcv9:
  case Triple::systemz:
  case Triple::x86_64:
    // Already 64-bit.
    break;

  case Triple::le32:    T.setArch(Triple::le64);      break;
  case Triple::mips:    T.setArch(Triple::mips64);    break;
  case Triple::mipsel:  T.setArch(Triple::mips64el);  break;
  case Triple::nvptx:   T.setArch(Triple::nvptx64);   break;
  case Triple::ppc:     T.setArch(Triple::ppc64);     break;
  case Triple::sparc:   T.setArch(Triple::sparcv9);   break;
  case Triple::x86:     T.setArch(Triple::x86_64);    break;
  case Triple::amdil:   T.setArch(Triple::amdil64);   break;
  case Triple::hsail:   T.setArch(Triple::hsail64);   break;
  case Triple::spir:    T.setArch(Triple::spir64);    break;
  }
  return T;
}

// FIXME: tblgen this.
const char *Triple::getARMCPUForArch(StringRef MArch) const {
  if (MArch.empty())
    MArch = getArchName();

  switch (getOS()) {
  case llvm::Triple::FreeBSD:
  case llvm::Triple::NetBSD:
    if (MArch == "armv6")
      return "arm1176jzf-s";
    break;
  case llvm::Triple::Win32:
    // FIXME: this is invalid for WindowsCE
    return "cortex-a9";
  default:
    break;
  }

  const char *result = nullptr;
  size_t offset = StringRef::npos;
  if (MArch.startswith("arm"))
    offset = 3;
  if (MArch.startswith("thumb"))
    offset = 5;
  if (offset != StringRef::npos && MArch.substr(offset, 2) == "eb")
    offset += 2;
  if (MArch.endswith("eb"))
    MArch = MArch.substr(0, MArch.size() - 2);
  if (offset != StringRef::npos)
    result = llvm::StringSwitch<const char *>(MArch.substr(offset))
      .Cases("v2", "v2a", "arm2")
      .Case("v3", "arm6")
      .Case("v3m", "arm7m")
      .Case("v4", "strongarm")
      .Case("v4t", "arm7tdmi")
      .Cases("v5", "v5t", "arm10tdmi")
      .Cases("v5e", "v5te", "arm1022e")
      .Case("v5tej", "arm926ej-s")
      .Cases("v6", "v6k", "arm1136jf-s")
      .Case("v6j", "arm1136j-s")
      .Cases("v6z", "v6zk", "arm1176jzf-s")
      .Case("v6t2", "arm1156t2-s")
      .Cases("v6m", "v6-m", "cortex-m0")
      .Cases("v7", "v7a", "v7-a", "v7l", "v7-l", "cortex-a8")
      .Cases("v7s", "v7-s", "swift")
      .Cases("v7r", "v7-r", "cortex-r4")
      .Cases("v7m", "v7-m", "cortex-m3")
      .Cases("v7em", "v7e-m", "cortex-m4")
      .Cases("v8", "v8a", "v8-a", "cortex-a53")
      .Default(nullptr);
  else
    result = llvm::StringSwitch<const char *>(MArch)
      .Case("ep9312", "ep9312")
      .Case("iwmmxt", "iwmmxt")
      .Case("xscale", "xscale")
      .Default(nullptr);

  if (result)
    return result;

  // If all else failed, return the most base CPU with thumb interworking
  // supported by LLVM.
  // FIXME: Should warn once that we're falling back.
  switch (getOS()) {
  case llvm::Triple::NetBSD:
    switch (getEnvironment()) {
    case llvm::Triple::GNUEABIHF:
    case llvm::Triple::GNUEABI:
    case llvm::Triple::EABIHF:
    case llvm::Triple::EABI:
      return "arm926ej-s";
    default:
      return "strongarm";
    }
  default:
    switch (getEnvironment()) {
    case llvm::Triple::EABIHF:
    case llvm::Triple::GNUEABIHF:
      return "arm1176jzf-s";
    default:
      return "arm7tdmi";
    }
  }
}
