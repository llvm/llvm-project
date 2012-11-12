//===-- Path.cpp - Implement OS Path Concept --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system Path concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Config/config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Endian.h"
#include <cassert>
#include <cstring>
#include <ostream>
using namespace llvm;
using namespace sys;
namespace {
using support::ulittle32_t;
}

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

bool Path::operator==(const Path &that) const {
  return path == that.path;
}

bool Path::operator<(const Path& that) const {
  return path < that.path;
}

LLVMFileType
sys::IdentifyFileType(const char *magic, unsigned length) {
  assert(magic && "Invalid magic number string");
  assert(length >=4 && "Invalid magic number length");
  switch ((unsigned char)magic[0]) {
    case 0xDE:  // 0x0B17C0DE = BC wraper
      if (magic[1] == (char)0xC0 && magic[2] == (char)0x17 &&
          magic[3] == (char)0x0B)
        return Bitcode_FileType;
      break;
    case 'B':
      if (magic[1] == 'C' && magic[2] == (char)0xC0 && magic[3] == (char)0xDE)
        return Bitcode_FileType;
      break;
    case '!':
      if (length >= 8)
        if (memcmp(magic,"!<arch>\n",8) == 0)
          return Archive_FileType;
      break;

    case '\177':
      if (magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F') {
        bool Data2MSB = magic[5] == 2;
        unsigned high = Data2MSB ? 16 : 17;
        unsigned low  = Data2MSB ? 17 : 16;
        if (length >= 18 && magic[high] == 0)
          switch (magic[low]) {
            default: break;
            case 1: return ELF_Relocatable_FileType;
            case 2: return ELF_Executable_FileType;
            case 3: return ELF_SharedObject_FileType;
            case 4: return ELF_Core_FileType;
          }
      }
      break;

    case 0xCA:
      if (magic[1] == char(0xFE) && magic[2] == char(0xBA) &&
          magic[3] == char(0xBE)) {
        // This is complicated by an overlap with Java class files.
        // See the Mach-O section in /usr/share/file/magic for details.
        if (length >= 8 && magic[7] < 43)
          // FIXME: Universal Binary of any type.
          return Mach_O_DynamicallyLinkedSharedLib_FileType;
      }
      break;

      // The two magic numbers for mach-o are:
      // 0xfeedface - 32-bit mach-o
      // 0xfeedfacf - 64-bit mach-o
    case 0xFE:
    case 0xCE:
    case 0xCF: {
      uint16_t type = 0;
      if (magic[0] == char(0xFE) && magic[1] == char(0xED) &&
          magic[2] == char(0xFA) &&
          (magic[3] == char(0xCE) || magic[3] == char(0xCF))) {
        /* Native endian */
        if (length >= 16) type = magic[14] << 8 | magic[15];
      } else if ((magic[0] == char(0xCE) || magic[0] == char(0xCF)) &&
                 magic[1] == char(0xFA) && magic[2] == char(0xED) &&
                 magic[3] == char(0xFE)) {
        /* Reverse endian */
        if (length >= 14) type = magic[13] << 8 | magic[12];
      }
      switch (type) {
        default: break;
        case 1: return Mach_O_Object_FileType;
        case 2: return Mach_O_Executable_FileType;
        case 3: return Mach_O_FixedVirtualMemorySharedLib_FileType;
        case 4: return Mach_O_Core_FileType;
        case 5: return Mach_O_PreloadExecutable_FileType;
        case 6: return Mach_O_DynamicallyLinkedSharedLib_FileType;
        case 7: return Mach_O_DynamicLinker_FileType;
        case 8: return Mach_O_Bundle_FileType;
        case 9: return Mach_O_DynamicallyLinkedSharedLibStub_FileType;
        case 10: return Mach_O_DSYMCompanion_FileType;
      }
      break;
    }
    case 0xF0: // PowerPC Windows
    case 0x83: // Alpha 32-bit
    case 0x84: // Alpha 64-bit
    case 0x66: // MPS R4000 Windows
    case 0x50: // mc68K
    case 0x4c: // 80386 Windows
      if (magic[1] == 0x01)
        return COFF_FileType;

    case 0x90: // PA-RISC Windows
    case 0x68: // mc68K Windows
      if (magic[1] == 0x02)
        return COFF_FileType;
      break;

    case 0x4d: // Possible MS-DOS stub on Windows PE file
      if (magic[1] == 0x5a) {
        uint32_t off = *reinterpret_cast<const ulittle32_t *>(magic + 0x3c);
        // PE/COFF file, either EXE or DLL.
        if (off < length && memcmp(magic + off, "PE\0\0",4) == 0)
          return COFF_FileType;
      }
      break;

    case 0x64: // x86-64 Windows.
      if (magic[1] == char(0x86))
        return COFF_FileType;
      break;

    default:
      break;
  }
  return Unknown_FileType;
}

bool
Path::isArchive() const {
  fs::file_magic type;
  if (fs::identify_magic(str(), type))
    return false;
  return type == fs::file_magic::archive;
}

bool
Path::isDynamicLibrary() const {
  fs::file_magic type;
  if (fs::identify_magic(str(), type))
    return false;
  switch (type) {
    default: return false;
    case fs::file_magic::macho_fixed_virtual_memory_shared_lib:
    case fs::file_magic::macho_dynamically_linked_shared_lib:
    case fs::file_magic::macho_dynamically_linked_shared_lib_stub:
    case fs::file_magic::elf_shared_object:
    case fs::file_magic::pecoff_executable:  return true;
  }
}

bool
Path::isObjectFile() const {
  fs::file_magic type;
  if (fs::identify_magic(str(), type) || type == fs::file_magic::unknown)
    return false;
  return true;
}

Path
Path::FindLibrary(std::string& name) {
  std::vector<sys::Path> LibPaths;
  GetSystemLibraryPaths(LibPaths);
  for (unsigned i = 0; i < LibPaths.size(); ++i) {
    sys::Path FullPath(LibPaths[i]);
    FullPath.appendComponent("lib" + name + LTDL_SHLIB_EXT);
    if (FullPath.isDynamicLibrary())
      return FullPath;
    FullPath.eraseSuffix();
    FullPath.appendSuffix("a");
    if (FullPath.isArchive())
      return FullPath;
  }
  return sys::Path();
}

StringRef Path::GetDLLSuffix() {
  return &(LTDL_SHLIB_EXT[1]);
}

void
Path::appendSuffix(StringRef suffix) {
  if (!suffix.empty()) {
    path.append(".");
    path.append(suffix);
  }
}

bool
Path::isBitcodeFile() const {
  fs::file_magic type;
  if (fs::identify_magic(str(), type))
    return false;
  return type == fs::file_magic::bitcode;
}

bool Path::hasMagicNumber(StringRef Magic) const {
  std::string actualMagic;
  if (getMagicNumber(actualMagic, static_cast<unsigned>(Magic.size())))
    return Magic == actualMagic;
  return false;
}

static void getPathList(const char*path, std::vector<Path>& Paths) {
  const char* at = path;
  const char* delim = strchr(at, PathSeparator);
  Path tmpPath;
  while (delim != 0) {
    std::string tmp(at, size_t(delim-at));
    if (tmpPath.set(tmp))
      if (tmpPath.canRead())
        Paths.push_back(tmpPath);
    at = delim + 1;
    delim = strchr(at, PathSeparator);
  }

  if (*at != 0)
    if (tmpPath.set(std::string(at)))
      if (tmpPath.canRead())
        Paths.push_back(tmpPath);
}

static StringRef getDirnameCharSep(StringRef path, const char *Sep) {
  assert(Sep[0] != '\0' && Sep[1] == '\0' &&
         "Sep must be a 1-character string literal.");
  if (path.empty())
    return ".";

  // If the path is all slashes, return a single slash.
  // Otherwise, remove all trailing slashes.

  signed pos = static_cast<signed>(path.size()) - 1;

  while (pos >= 0 && path[pos] == Sep[0])
    --pos;

  if (pos < 0)
    return path[0] == Sep[0] ? Sep : ".";

  // Any slashes left?
  signed i = 0;

  while (i < pos && path[i] != Sep[0])
    ++i;

  if (i == pos) // No slashes?  Return "."
    return ".";

  // There is at least one slash left.  Remove all trailing non-slashes.
  while (pos >= 0 && path[pos] != Sep[0])
    --pos;

  // Remove any trailing slashes.
  while (pos >= 0 && path[pos] == Sep[0])
    --pos;

  if (pos < 0)
    return path[0] == Sep[0] ? Sep : ".";

  return path.substr(0, pos+1);
}

// Include the truly platform-specific parts of this class.
#if defined(LLVM_ON_UNIX)
#include "Unix/Path.inc"
#endif
#if defined(LLVM_ON_WIN32)
#include "Windows/Path.inc"
#endif
