//===- tools/dsymutil/SymbolMap.cpp ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolMap.h"
#include "DebugMap.h"
#include "MachOUtils.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <uuid/uuid.h>
#endif

namespace llvm {
namespace dsymutil {

StringRef SymbolMapTranslator::operator()(StringRef Input) {
  if (!Input.startswith("__hidden#") && !Input.startswith("___hidden#"))
    return Input;

  bool MightNeedUnderscore = false;
  StringRef Line = Input.drop_front(sizeof("__hidden#") - 1);
  if (Line[0] == '#') {
    Line = Line.drop_front();
    MightNeedUnderscore = true;
  }

  std::size_t LineNumber = std::numeric_limits<std::size_t>::max();
  Line.split('_').first.getAsInteger(10, LineNumber);
  if (LineNumber >= UnobfuscatedStrings.size()) {
    WithColor::warning() << "reference to a unexisting unobfuscated string "
                         << Input << ": symbol map mismatch?\n"
                         << Line << '\n';
    return Input;
  }

  const std::string &Translation = UnobfuscatedStrings[LineNumber];
  if (!MightNeedUnderscore || !MangleNames)
    return Translation;

  // Objective-C symbols for the MachO symbol table start with a \1. Please see
  // `CGObjCCommonMac::GetNameForMethod` in clang.
  if (Translation[0] == 1)
    return StringRef(Translation).drop_front();

  // We need permanent storage for the string we are about to create. Just
  // append it to the vector containing translations. This should only happen
  // during MachO symbol table translation, thus there should be no risk on
  // exponential growth.
  UnobfuscatedStrings.emplace_back("_" + Translation);
  return UnobfuscatedStrings.back();
}

SymbolMapTranslator SymbolMapLoader::Load(StringRef InputFile,
                                          const DebugMap &Map) const {
  if (SymbolMap.empty())
    return {};

  std::string SymbolMapPath = SymbolMap;

#if __APPLE__
  // Look through the UUID Map.
  if (sys::fs::is_directory(SymbolMapPath) && !Map.getUUID().empty()) {
    uuid_string_t UUIDString;
    uuid_unparse_upper((const uint8_t *)Map.getUUID().data(), UUIDString);

    SmallString<256> PlistPath(
        sys::path::parent_path(sys::path::parent_path(InputFile)));
    sys::path::append(PlistPath, StringRef(UUIDString).str() + ".plist");

    CFStringRef plistFile = CFStringCreateWithCString(
        kCFAllocatorDefault, PlistPath.c_str(), kCFStringEncodingUTF8);
    CFURLRef fileURL = CFURLCreateWithFileSystemPath(
        kCFAllocatorDefault, plistFile, kCFURLPOSIXPathStyle, false);
    CFReadStreamRef resourceData =
        CFReadStreamCreateWithFile(kCFAllocatorDefault, fileURL);
    if (resourceData) {
      CFReadStreamOpen(resourceData);
      CFDictionaryRef plist = (CFDictionaryRef)CFPropertyListCreateWithStream(
          kCFAllocatorDefault, resourceData, 0, kCFPropertyListImmutable,
          nullptr, nullptr);

      if (plist) {
        if (CFDictionaryContainsKey(plist, CFSTR("DBGOriginalUUID"))) {
          CFStringRef OldUUID = (CFStringRef)CFDictionaryGetValue(
              plist, CFSTR("DBGOriginalUUID"));

          StringRef UUID(CFStringGetCStringPtr(OldUUID, kCFStringEncodingUTF8));
          SmallString<256> BCSymbolMapPath(SymbolMapPath);
          sys::path::append(BCSymbolMapPath, UUID.str() + ".bcsymbolmap");
          SymbolMapPath = BCSymbolMapPath.str();
        }
        CFRelease(plist);
      }
      CFReadStreamClose(resourceData);
      CFRelease(resourceData);
    }
    CFRelease(fileURL);
    CFRelease(plistFile);
  }
#endif

  if (sys::fs::is_directory(SymbolMapPath)) {
    SymbolMapPath += (Twine("/") + sys::path::filename(InputFile) + "-" +
                      MachOUtils::getArchName(Map.getTriple().getArchName()) +
                      ".bcsymbolmap")
                         .str();
  }

  auto ErrOrMemBuffer = MemoryBuffer::getFile(SymbolMapPath);
  if (auto EC = ErrOrMemBuffer.getError()) {
    WithColor::warning() << SymbolMapPath << ": " << EC.message()
                         << ": not unobfuscating.\n";
    return {};
  }

  std::vector<std::string> UnobfuscatedStrings;
  auto &MemBuf = **ErrOrMemBuffer;
  StringRef Data(MemBuf.getBufferStart(),
                 MemBuf.getBufferEnd() - MemBuf.getBufferStart());
  StringRef LHS;
  std::tie(LHS, Data) = Data.split('\n');
  bool MangleNames = false;

  // Check version string first.
  if (!LHS.startswith("BCSymbolMap Version:")) {
    // Version string not present, warns but try to parse it.
    WithColor::warning() << SymbolMapPath
                         << " is missing version string: assuming 1.0.\n";
    UnobfuscatedStrings.emplace_back(LHS);
  } else if (LHS.equals("BCSymbolMap Version: 1.0")) {
    MangleNames = true;
  } else if (LHS.equals("BCSymbolMap Version: 2.0")) {
    MangleNames = false;
  } else {
    StringRef VersionNum;
    std::tie(LHS, VersionNum) = LHS.split(':');
    WithColor::warning() << SymbolMapPath
                         << " has unsupported symbol map version" << VersionNum
                         << ": not unobfuscating.\n";
    return {};
  }

  while (!Data.empty()) {
    std::tie(LHS, Data) = Data.split('\n');
    UnobfuscatedStrings.emplace_back(LHS);
  }

  return SymbolMapTranslator(std::move(UnobfuscatedStrings), MangleNames);
}

} // namespace dsymutil
} // namespace llvm
