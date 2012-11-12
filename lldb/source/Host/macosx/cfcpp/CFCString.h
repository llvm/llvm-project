//===-- CFCString.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CoreFoundationCPP_CFString_h_
#define CoreFoundationCPP_CFString_h_

#include <iosfwd>

#include "CFCReleaser.h"

class CFCString : public CFCReleaser<CFStringRef>
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
                        CFCString (CFStringRef cf_str = NULL);
                        CFCString (const char *s, CFStringEncoding encoding = kCFStringEncodingUTF8);
                        CFCString (const CFCString& rhs);
                        CFCString& operator= (const CFCString& rhs);
                        virtual ~CFCString ();

        const char *    GetFileSystemRepresentation (std::string& str);
        CFStringRef     SetFileSystemRepresentation (const char *path);
        CFStringRef     SetFileSystemRepresentationFromCFType (CFTypeRef cf_type);
        CFStringRef     SetFileSystemRepresentationAndExpandTilde (const char *path);
        const char *    UTF8 (std::string& str);
        CFIndex         GetLength() const;
        static const char *UTF8 (CFStringRef cf_str, std::string& str);
        static const char *FileSystemRepresentation (CFStringRef cf_str, std::string& str);
        static const char *ExpandTildeInPath(const char* path, std::string &expanded_path);

};

#endif // #ifndef CoreFoundationCPP_CFString_h_
