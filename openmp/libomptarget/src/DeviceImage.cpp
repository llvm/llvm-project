//===-- DeviceImage.cpp - Representation of the device code/image ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "DeviceImage.h"

#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Shared/Utils.h"

#include "llvm/Support/Error.h"

DeviceImageTy::DeviceImageTy(__tgt_device_image &TgtDeviceImage)
    : Image(TgtDeviceImage) {
  llvm::StringRef ImageStr(
      static_cast<char *>(Image.ImageStart),
      llvm::omp::target::getPtrDiff(Image.ImageEnd, Image.ImageStart));

  auto BinaryOrErr =
      llvm::object::OffloadBinary::create(llvm::MemoryBufferRef(ImageStr, ""));

  if (!BinaryOrErr) {
    consumeError(BinaryOrErr.takeError());
    return;
  }

  Binary = std::move(*BinaryOrErr);
  void *Begin = const_cast<void *>(
      static_cast<const void *>(Binary->getImage().bytes_begin()));
  void *End = const_cast<void *>(
      static_cast<const void *>(Binary->getImage().bytes_end()));

  Image = __tgt_device_image{Begin, End, Image.EntriesBegin, Image.EntriesEnd};
  ImageInfo = __tgt_image_info{Binary->getArch().data()};
}
