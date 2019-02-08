//==------ device_selector.hpp - SYCL device selector ---------*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
