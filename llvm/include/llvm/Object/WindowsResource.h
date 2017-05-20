//===-- WindowsResource.h ---------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This file declares the .res file class.  .res files are intermediate
// products of the typical resource-compilation process on Windows.  This
// process is as follows:
//
// .rc file(s) ---(rc.exe)---> .res file(s) ---(cvtres.exe)---> COFF file
//
// .rc files are human-readable scripts that list all resources a program uses.
//
// They are compiled into .res files, which are a list of the resources in
// binary form.
//
// Finally the data stored in the .res is compiled into a COFF file, where it
// is organized in a directory tree structure for optimized access by the
// program during runtime.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/ms648007(v=vs.85).aspx
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_INCLUDE_LLVM_OBJECT_RESFILE_H
#define LLVM_INCLUDE_LLVM_OBJECT_RESFILE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace object {

class WindowsResource;

class ResourceEntryRef {
public:
  Error moveNext(bool &End);

private:
  friend class WindowsResource;

  ResourceEntryRef(BinaryStreamRef Ref, const WindowsResource *Owner,
                   Error &Err);
  Error loadNext();

  BinaryStreamReader Reader;
  BinaryStreamRef HeaderBytes;
  BinaryStreamRef DataBytes;
  const WindowsResource *OwningRes = nullptr;
};

class WindowsResource : public Binary {
public:
  ~WindowsResource() override;
  Expected<ResourceEntryRef> getHeadEntry();

  static bool classof(const Binary *V) { return V->isWinRes(); }

  static Expected<std::unique_ptr<WindowsResource>>
  createWindowsResource(MemoryBufferRef Source);

private:
  friend class ResourceEntryRef;

  WindowsResource(MemoryBufferRef Source);

  BinaryByteStream BBS;
};

} // namespace object
} // namespace llvm

#endif
