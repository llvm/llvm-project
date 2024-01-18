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

#include "OffloadEntry.h"
#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Shared/Utils.h"

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include <memory>

__tgt_bin_desc *OffloadEntryTy::getBinaryDescription() const {
  return &DeviceImage.getBinaryDesc();
}

DeviceImageTy::DeviceImageTy(__tgt_bin_desc &BinaryDesc,
                             __tgt_device_image &TgtDeviceImage)
    : BinaryDesc(&BinaryDesc), Image(TgtDeviceImage) {

  for (__tgt_offload_entry &Entry :
       llvm::make_range(Image.EntriesBegin, Image.EntriesEnd))
    OffloadEntries.emplace_back(std::make_unique<OffloadEntryTy>(*this, Entry));

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
