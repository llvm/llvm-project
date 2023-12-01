//===-- DeviceImage.h - Representation of the device code/image -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICE_IMAGE_H
#define OMPTARGET_DEVICE_IMAGE_H

#include "Shared/APITypes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/OffloadBinary.h"

class DeviceImageTy {

  std::unique_ptr<llvm::object::OffloadBinary> Binary;

  __tgt_device_image Image;
  __tgt_image_info ImageInfo;

public:
  DeviceImageTy(__tgt_device_image &Image);

  __tgt_device_image &getExecutableImage() { return Image; }
  __tgt_image_info &getImageInfo() { return ImageInfo; }

  llvm::StringRef
  getArch(llvm::StringRef DefaultArch = llvm::StringRef()) const {
    return ImageInfo.Arch ? ImageInfo.Arch : DefaultArch;
  }
};

#endif // OMPTARGET_DEVICE_IMAGE_H
