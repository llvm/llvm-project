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

#include "llvm/System/Path.h"
#include "llvm/Config/config.h"
#include <cassert>
#include <cstring>
#include <ostream>
using namespace llvm;
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

bool Path::operator==(const Path &that) const {
  return path == that.path;
}

bool Path::operator!=(const Path &that) const {
  return path != that.path;
}

bool Path::operator<(const Path& that) const {
  return path < that.path;
}

std::ostream& llvm::operator<<(std::ostream &strm, const sys::Path &aPath) {
  strm << aPath.toString();
  return strm;
}

Path
Path::GetLLVMConfigDir() {
  Path result;
#ifdef LLVM_ETCDIR
  if (result.set(LLVM_ETCDIR))
    return result;
#endif
  return GetLLVMDefaultConfigDir();
}

LLVMFileType
sys::IdentifyFileType(const char *magic, unsigned length) {
  assert(magic && "Invalid magic number string");
  assert(length >=4 && "Invalid magic number length");
  switch (magic[0]) {
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
        if (length >= 18 && magic[17] == 0)
          switch (magic[16]) {
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

    case 0xFE:
    case 0xCE: {
      uint16_t type = 0;
      if (magic[0] == char(0xFE) && magic[1] == char(0xED) && 
          magic[2] == char(0xFA) && magic[3] == char(0xCE)) {
        /* Native endian */
        if (length >= 16) type = magic[14] << 8 | magic[15];
      } else if (magic[0] == char(0xCE) && magic[1] == char(0xFA) && 
                 magic[2] == char(0xED) && magic[3] == char(0xFE)) {
        /* Reverse endian */
        if (length >= 14) type = magic[13] << 8 | magic[12];
      }
      switch (type) {
        default: break;      
        case 1: return Mach_O_Object_FileType; 
        case 2: return Mach_O_Executable_FileType;
        case 3: return Mach_O_FixedVirtualMemorySharedLib_FileType;
        case 4: return Mach_O_Core_FileType;
        case 5: return Mach_O_PreloadExectuable_FileType;
        case 6: return Mach_O_DynamicallyLinkedSharedLib_FileType;
        case 7: return Mach_O_DynamicLinker_FileType;
        case 8: return Mach_O_Bundle_FileType;
        case 9: return Mach_O_DynamicallyLinkedSharedLibStub_FileType;
        case 10: break; // FIXME: MH_DSYM companion file with only debug.
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

    default:
      break;
  }
  return Unknown_FileType;
}

bool
Path::isArchive() const {
  if (canRead())
    return hasMagicNumber("!<arch>\012");
  return false;
}

bool
Path::isDynamicLibrary() const {
  if (canRead()) {
    std::string Magic;
    if (getMagicNumber(Magic, 64))
      switch (IdentifyFileType(Magic.c_str(),
                               static_cast<unsigned>(Magic.length()))) {
        default: return false;
        case Mach_O_FixedVirtualMemorySharedLib_FileType:
        case Mach_O_DynamicallyLinkedSharedLib_FileType:
        case Mach_O_DynamicallyLinkedSharedLibStub_FileType:
        case ELF_SharedObject_FileType:
        case COFF_FileType:  return true;
      }
  }
  return false;
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

std::string Path::GetDLLSuffix() {
  return LTDL_SHLIB_EXT;
}

bool
Path::isBitcodeFile() const {
  std::string actualMagic;
  if (!getMagicNumber(actualMagic, 4))
    return false;
  LLVMFileType FT =
    IdentifyFileType(actualMagic.c_str(),
                     static_cast<unsigned>(actualMagic.length()));
  return FT == Bitcode_FileType;
}

bool Path::hasMagicNumber(const std::string &Magic) const {
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

static std::string getDirnameCharSep(const std::string& path, char Sep) {
  
  if (path.empty())
    return ".";
  
  // If the path is all slashes, return a single slash.
  // Otherwise, remove all trailing slashes.
  
  signed pos = static_cast<signed>(path.size()) - 1;
  
  while (pos >= 0 && path[pos] == Sep)
    --pos;
  
  if (pos < 0)
    return path[0] == Sep ? std::string(1, Sep) : std::string(".");
  
  // Any slashes left?
  signed i = 0;
  
  while (i < pos && path[i] != Sep)
    ++i;
  
  if (i == pos) // No slashes?  Return "."
    return ".";
  
  // There is at least one slash left.  Remove all trailing non-slashes.  
  while (pos >= 0 && path[pos] != Sep)
    --pos;
  
  // Remove any trailing slashes.
  while (pos >= 0 && path[pos] == Sep)
    --pos;
  
  if (pos < 0)
    return path[0] == Sep ? std::string(1, Sep) : std::string(".");
  
  return path.substr(0, pos+1);
}

// Include the truly platform-specific parts of this class.
#if defined(LLVM_ON_UNIX)
#include "Unix/Path.inc"
#endif
#if defined(LLVM_ON_WIN32)
#include "Win32/Path.inc"
#endif

