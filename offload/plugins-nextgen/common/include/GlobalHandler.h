//===- GlobalHandler.h - Target independent global & enviroment handling --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target independent global handler and environment manager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_GLOBALHANDLER_H
#define LLVM_OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_GLOBALHANDLER_H

#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Object/ELFObjectFile.h"

#include "Shared/Debug.h"
#include "Shared/Utils.h"

#include "omptarget.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

class DeviceImageTy;
struct GenericDeviceTy;

using namespace llvm::object;

/// Common abstraction for globals that live on the host and device.
/// It simply encapsulates the symbol name, symbol size, and symbol address
/// (which might be host or device depending on the context).
class GlobalTy {
  // NOTE: Maybe we can have a pointer to the offload entry name instead of
  // holding a private copy of the name as a std::string.
  std::string Name;
  uint32_t Size;
  void *Ptr;

public:
  GlobalTy(const std::string &Name, uint32_t Size, void *Ptr = nullptr)
      : Name(Name), Size(Size), Ptr(Ptr) {}

  const std::string &getName() const { return Name; }
  uint32_t getSize() const { return Size; }
  void *getPtr() const { return Ptr; }

  void setSize(int32_t S) { Size = S; }
  void setPtr(void *P) { Ptr = P; }
};

/// Subclass of GlobalTy that holds the memory for a global of \p Ty.
template <typename Ty> class StaticGlobalTy : public GlobalTy {
  Ty Data;

public:
  template <typename... Args>
  StaticGlobalTy(const std::string &Name, Args &&...args)
      : GlobalTy(Name, sizeof(Ty), &Data),
        Data(Ty{std::forward<Args>(args)...}) {}

  template <typename... Args>
  StaticGlobalTy(const char *Name, Args &&...args)
      : GlobalTy(Name, sizeof(Ty), &Data),
        Data(Ty{std::forward<Args>(args)...}) {}

  template <typename... Args>
  StaticGlobalTy(const char *Name, const char *Suffix, Args &&...args)
      : GlobalTy(std::string(Name) + Suffix, sizeof(Ty), &Data),
        Data(Ty{std::forward<Args>(args)...}) {}

  Ty &getValue() { return Data; }
  const Ty &getValue() const { return Data; }
  void setValue(const Ty &V) { Data = V; }
};

/// Helper class to do the heavy lifting when it comes to moving globals between
/// host and device. Through the GenericDeviceTy we access memcpy DtoH and HtoD,
/// which means the only things specialized by the subclass is the retrival of
/// global metadata (size, addr) from the device.
/// \see getGlobalMetadataFromDevice
class GenericGlobalHandlerTy {
  /// Actually move memory between host and device. See readGlobalFromDevice and
  /// writeGlobalToDevice for the interface description.
  Error moveGlobalBetweenDeviceAndHost(GenericDeviceTy &Device,
                                       DeviceImageTy &Image,
                                       const GlobalTy &HostGlobal,
                                       bool Device2Host);

  /// Actually move memory between host and device. See readGlobalFromDevice and
  /// writeGlobalToDevice for the interface description.
  Error moveGlobalBetweenDeviceAndHost(GenericDeviceTy &Device,
                                       const GlobalTy &HostGlobal,
                                       const GlobalTy &DeviceGlobal,
                                       bool Device2Host);

public:
  virtual ~GenericGlobalHandlerTy() {}

  /// Helper function for getting an ELF from a device image.
  Expected<std::unique_ptr<ObjectFile>> getELFObjectFile(DeviceImageTy &Image);

  /// Returns whether the symbol named \p SymName is present in the given \p
  /// Image.
  bool isSymbolInImage(GenericDeviceTy &Device, DeviceImageTy &Image,
                       StringRef SymName);

  /// Get the address and size of a global in the image. Address and size are
  /// return in \p ImageGlobal, the global name is passed in \p ImageGlobal.
  Error getGlobalMetadataFromImage(GenericDeviceTy &Device,
                                   DeviceImageTy &Image, GlobalTy &ImageGlobal);

  /// Read the memory associated with a global from the image and store it on
  /// the host. The name, size, and destination are defined by \p HostGlobal.
  Error readGlobalFromImage(GenericDeviceTy &Device, DeviceImageTy &Image,
                            const GlobalTy &HostGlobal);

  /// Get the address and size of a global from the device. Address is return in
  /// \p DeviceGlobal, the global name and expected size are passed in
  /// \p DeviceGlobal.
  virtual Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                            DeviceImageTy &Image,
                                            GlobalTy &DeviceGlobal) = 0;

  /// Copy the memory associated with a global from the device to its
  /// counterpart on the host. The name, size, and destination are defined by
  /// \p HostGlobal. The origin is defined by \p DeviceGlobal.
  Error readGlobalFromDevice(GenericDeviceTy &Device,
                             const GlobalTy &HostGlobal,
                             const GlobalTy &DeviceGlobal) {
    return moveGlobalBetweenDeviceAndHost(Device, HostGlobal, DeviceGlobal,
                                          /*D2H=*/true);
  }

  /// Copy the memory associated with a global from the device to its
  /// counterpart on the host. The name, size, and destination are defined by
  /// \p HostGlobal. The origin is automatically resolved.
  Error readGlobalFromDevice(GenericDeviceTy &Device, DeviceImageTy &Image,
                             const GlobalTy &HostGlobal) {
    return moveGlobalBetweenDeviceAndHost(Device, Image, HostGlobal,
                                          /*D2H=*/true);
  }

  /// Copy the memory associated with a global from the host to its counterpart
  /// on the device. The name, size, and origin are defined by \p HostGlobal.
  /// The destination is defined by \p DeviceGlobal.
  Error writeGlobalToDevice(GenericDeviceTy &Device, const GlobalTy &HostGlobal,
                            const GlobalTy &DeviceGlobal) {
    return moveGlobalBetweenDeviceAndHost(Device, HostGlobal, DeviceGlobal,
                                          /*D2H=*/false);
  }

  /// Copy the memory associated with a global from the host to its counterpart
  /// on the device. The name, size, and origin are defined by \p HostGlobal.
  /// The destination is automatically resolved.
  Error writeGlobalToDevice(GenericDeviceTy &Device, DeviceImageTy &Image,
                            const GlobalTy &HostGlobal) {
    return moveGlobalBetweenDeviceAndHost(Device, Image, HostGlobal,
                                          /*D2H=*/false);
  }
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // LLVM_OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_GLOBALHANDLER_H
