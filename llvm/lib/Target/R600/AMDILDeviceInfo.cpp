//===-- AMDILDeviceInfo.cpp - AMDILDeviceInfo class -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// \brief Function that creates DeviceInfo from a device name and other information.
//
//==-----------------------------------------------------------------------===//
#include "AMDILDevices.h"
#include "AMDGPUSubtarget.h"

using namespace llvm;
namespace llvm {
namespace AMDGPUDeviceInfo {

AMDGPUDevice* getDeviceFromName(const std::string &deviceName,
                                AMDGPUSubtarget *ptr,
                                bool is64bit, bool is64on32bit) {
  if (deviceName.c_str()[2] == '7') {
    switch (deviceName.c_str()[3]) {
    case '1':
      return new AMDGPU710Device(ptr);
    case '7':
      return new AMDGPU770Device(ptr);
    default:
      return new AMDGPU7XXDevice(ptr);
    }
  } else if (deviceName == "cypress") {
#if DEBUG
    assert(!is64bit && "This device does not support 64bit pointers!");
    assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
    return new AMDGPUCypressDevice(ptr);
  } else if (deviceName == "juniper") {
#if DEBUG
    assert(!is64bit && "This device does not support 64bit pointers!");
    assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
    return new AMDGPUEvergreenDevice(ptr);
  } else if (deviceName == "redwood" || deviceName == "sumo") {
#if DEBUG
    assert(!is64bit && "This device does not support 64bit pointers!");
    assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
    return new AMDGPURedwoodDevice(ptr);
  } else if (deviceName == "cedar") {
#if DEBUG
    assert(!is64bit && "This device does not support 64bit pointers!");
    assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
    return new AMDGPUCedarDevice(ptr);
  } else if (deviceName == "barts" || deviceName == "turks") {
#if DEBUG
    assert(!is64bit && "This device does not support 64bit pointers!");
    assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
    return new AMDGPUNIDevice(ptr);
  } else if (deviceName == "cayman") {
#if DEBUG
    assert(!is64bit && "This device does not support 64bit pointers!");
    assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
    return new AMDGPUCaymanDevice(ptr);
  } else if (deviceName == "caicos") {
#if DEBUG
    assert(!is64bit && "This device does not support 64bit pointers!");
    assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
    return new AMDGPUNIDevice(ptr);
  } else if (deviceName == "SI" ||
             deviceName == "tahiti" || deviceName == "pitcairn" ||
             deviceName == "verde"  || deviceName == "oland" ||
	     deviceName == "hainan") {
    return new AMDGPUSIDevice(ptr);
  } else {
#if DEBUG
    assert(!is64bit && "This device does not support 64bit pointers!");
    assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
    return new AMDGPU7XXDevice(ptr);
  }
}
} // End namespace AMDGPUDeviceInfo
} // End namespace llvm
