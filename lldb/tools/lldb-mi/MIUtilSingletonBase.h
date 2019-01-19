//===-- MIUtilSingletonBase.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace MI {

//   MI::ISingleton base class usage:
//
//   class CMIDerivedClass
//       : public MI::ISingleton< CMIDerivedClass >
//   {
//       friend MI::ISingleton< CMIDerivedClass >;
//
//   // Overridden:
//   public:
//       // From MI::ISingleton
//       bool Initialize() override;
//       bool Shutdown() override;
//   };

//++
//============================================================================
// Details: Base class for the singleton pattern.
// Gotchas: Derived class must specify MI::ISingleton<> as a friend class.
//--
template <typename T> class ISingleton {
  // Statics:
public:
  // Return an instance of the derived class
  static T &Instance() {
    // This will fail if the derived class has not
    // declared itself to be a friend of MI::ISingleton
    static T instance;

    return instance;
  }

  // Overrideable:
public:
  virtual bool Initialize() = 0;
  virtual bool Shutdown() = 0;
  //
  /* dtor */ virtual ~ISingleton() {}
};

} // namespace MI
