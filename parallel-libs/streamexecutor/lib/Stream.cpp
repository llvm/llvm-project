//===-- Stream.cpp - General stream implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation details for a general stream object.
///
//===----------------------------------------------------------------------===//

#include <cassert>

#include "streamexecutor/Stream.h"

namespace streamexecutor {

Stream::Stream(PlatformDevice *D, const void *PlatformStreamHandle)
    : PDevice(D), PlatformStreamHandle(PlatformStreamHandle),
      ErrorMessageMutex(llvm::make_unique<llvm::sys::RWMutex>()) {
  assert(D != nullptr &&
         "cannot construct a stream object with a null platform device");
  assert(PlatformStreamHandle != nullptr &&
         "cannot construct a stream object with a null platform stream handle");
}

Stream::Stream(Stream &&Other)
    : PDevice(Other.PDevice), PlatformStreamHandle(Other.PlatformStreamHandle),
      ErrorMessageMutex(std::move(Other.ErrorMessageMutex)),
      ErrorMessage(std::move(Other.ErrorMessage)) {
  Other.PDevice = nullptr;
  Other.PlatformStreamHandle = nullptr;
}

Stream &Stream::operator=(Stream &&Other) {
  PDevice = Other.PDevice;
  PlatformStreamHandle = Other.PlatformStreamHandle;
  ErrorMessageMutex = std::move(Other.ErrorMessageMutex);
  ErrorMessage = std::move(Other.ErrorMessage);
  Other.PDevice = nullptr;
  Other.PlatformStreamHandle = nullptr;
  return *this;
}

Stream::~Stream() {
  if (PlatformStreamHandle)
    // TODO(jhen): Handle error condition here.
    consumeError(PDevice->destroyStream(PlatformStreamHandle));
}

} // namespace streamexecutor
