//==------ device_selector.hpp - SYCL device selector ---------*- C++ --*---==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.6.1 Device selection class

namespace cl {
namespace sycl {

// Forward declarations
class device;

class device_selector {
public:
  virtual ~device_selector() = default;

  device select_device() const;

  virtual int operator()(const device &device) const = 0;
};

class default_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class gpu_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class cpu_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class accelerator_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class host_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

} // namespace sycl
} // namespace cl
