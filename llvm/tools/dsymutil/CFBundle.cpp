//===- tools/dsymutil/CFBundle.cpp - CFBundle helper ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CFBundle.h"

#ifdef __APPLE__
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <CoreFoundation/CoreFoundation.h>
#include <assert.h>
#include <glob.h>
#include <memory>

namespace llvm {
namespace dsymutil {

/// Deleter that calls CFRelease rather than deleting the pointer.
template <typename T> struct CFDeleter {
  void operator()(T *P) {
    if (P)
      ::CFRelease(P);
  }
};

/// This helper owns any CoreFoundation pointer and will call CFRelease() on
/// any valid pointer it owns unless that pointer is explicitly released using
/// the release() member function.
template <typename T>
using CFReleaser =
    std::unique_ptr<typename std::remove_pointer<T>::type,
                    CFDeleter<typename std::remove_pointer<T>::type>>;

/// RAII wrapper around CFBundleRef.
class CFString : public CFReleaser<CFStringRef> {
public:
  CFString(CFStringRef CFStr = nullptr) : CFReleaser<CFStringRef>(CFStr) {}

  const char *UTF8(std::string &Str) const {
    return CFString::UTF8(get(), Str);
  }

  CFIndex GetLength() const {
    if (CFStringRef Str = get())
      return CFStringGetLength(Str);
    return 0;
  }

  static const char *UTF8(CFStringRef CFStr, std::string &Str);
};

/// Static function that puts a copy of the UTF8 contents of CFStringRef into
/// std::string and returns the C string pointer that is contained in the
/// std::string when successful, nullptr otherwise.
///
/// This allows the std::string parameter to own the extracted string, and also
/// allows that string to be returned as a C string pointer that can be used.
const char *CFString::UTF8(CFStringRef CFStr, std::string &Str) {
  if (!CFStr)
    return nullptr;

  const CFStringEncoding Encoding = kCFStringEncodingUTF8;
  CFIndex MaxUTF8StrLength = CFStringGetLength(CFStr);
  MaxUTF8StrLength =
      CFStringGetMaximumSizeForEncoding(MaxUTF8StrLength, Encoding);
  if (MaxUTF8StrLength > 0) {
    Str.resize(MaxUTF8StrLength);
    if (!Str.empty() &&
        CFStringGetCString(CFStr, &Str[0], Str.size(), Encoding)) {
      Str.resize(strlen(Str.c_str()));
      return Str.c_str();
    }
  }

  return nullptr;
}

/// RAII wrapper around CFBundleRef.
class CFBundle : public CFReleaser<CFBundleRef> {
public:
  CFBundle(const char *Path = nullptr) : CFReleaser<CFBundleRef>() {
    if (Path && Path[0])
      SetFromPath(Path);
  }

  CFBundle(CFURLRef url)
      : CFReleaser<CFBundleRef>(url ? ::CFBundleCreate(nullptr, url)
                                    : nullptr) {}

  /// Return the bundle identifier.
  CFStringRef GetIdentifier() const {
    if (CFBundleRef bundle = get())
      return ::CFBundleGetIdentifier(bundle);
    return nullptr;
  }

  /// Return value for key.
  CFTypeRef GetValueForInfoDictionaryKey(CFStringRef key) const {
    if (CFBundleRef bundle = get())
      return ::CFBundleGetValueForInfoDictionaryKey(bundle, key);
    return nullptr;
  }

private:
  /// Update this instance with a new bundle created from the given path.
  bool SetFromPath(const char *Path);
};

bool CFBundle::SetFromPath(const char *InPath) {
  // Release our old bundle and URL.
  reset();

  if (InPath && InPath[0]) {
    char ResolvedPath[PATH_MAX];
    const char *Path = ::realpath(InPath, ResolvedPath);
    if (Path == nullptr)
      Path = InPath;

    CFAllocatorRef Allocator = kCFAllocatorDefault;
    // Make our Bundle URL.
    CFReleaser<CFURLRef> BundleURL(::CFURLCreateFromFileSystemRepresentation(
        Allocator, (const UInt8 *)Path, strlen(Path), false));
    if (BundleURL.get()) {
      CFIndex LastLength = LONG_MAX;

      while (BundleURL.get() != nullptr) {
        // Check the Path range and make sure we didn't make it to just "/",
        // ".", or "..".
        CFRange rangeIncludingSeparators;
        CFRange range = ::CFURLGetByteRangeForComponent(
            BundleURL.get(), kCFURLComponentPath, &rangeIncludingSeparators);
        if (range.length > LastLength)
          break;

        reset(::CFBundleCreate(Allocator, BundleURL.get()));
        if (get() != nullptr) {
          if (GetIdentifier() != nullptr)
            break;
          reset();
        }
        BundleURL.reset(::CFURLCreateCopyDeletingLastPathComponent(
            Allocator, BundleURL.get()));

        LastLength = range.length;
      }
    }
  }

  return get() != nullptr;
}

#endif

/// On Darwin, try and find the original executable's Info.plist information
/// using CoreFoundation calls by creating a URL for the executable and
/// chopping off the last Path component. The CFBundle can then get the
/// identifier and grab any needed information from it directly. Return default
/// CFBundleInfo on other platforms.
CFBundleInfo getBundleInfo(StringRef ExePath) {
  CFBundleInfo BundleInfo;

#ifdef __APPLE__
  if (ExePath.empty() || !sys::fs::exists(ExePath))
    return BundleInfo;

  auto PrintError = [&](CFTypeID TypeID) {
    CFString TypeIDCFStr(::CFCopyTypeIDDescription(TypeID));
    std::string TypeIDStr;
    errs() << "The Info.plist key \"CFBundleShortVersionString\" is"
           << "a " << TypeIDCFStr.UTF8(TypeIDStr)
           << ", but it should be a string in: " << ExePath << ".\n";
  };

  CFBundle Bundle(ExePath.data());
  if (CFStringRef BundleID = Bundle.GetIdentifier()) {
    CFString::UTF8(BundleID, BundleInfo.IDStr);
    if (CFTypeRef TypeRef =
            Bundle.GetValueForInfoDictionaryKey(CFSTR("CFBundleVersion"))) {
      CFTypeID TypeID = ::CFGetTypeID(TypeRef);
      if (TypeID == ::CFStringGetTypeID())
        CFString::UTF8((CFStringRef)TypeRef, BundleInfo.VersionStr);
      else
        PrintError(TypeID);
    }
    if (CFTypeRef TypeRef = Bundle.GetValueForInfoDictionaryKey(
            CFSTR("CFBundleShortVersionString"))) {
      CFTypeID TypeID = ::CFGetTypeID(TypeRef);
      if (TypeID == ::CFStringGetTypeID())
        CFString::UTF8((CFStringRef)TypeRef, BundleInfo.ShortVersionStr);
      else
        PrintError(TypeID);
    }
  }
#endif

  return BundleInfo;
}

} // end namespace dsymutil
} // end namespace llvm
