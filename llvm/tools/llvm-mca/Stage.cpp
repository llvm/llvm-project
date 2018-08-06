//===---------------------- Stage.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a stage.
/// A chain of stages compose an instruction pipeline.
///
//===----------------------------------------------------------------------===//

#include "Stage.h"

namespace mca {

// Pin the vtable here in the implementation file.
Stage::Stage() {}

void Stage::addListener(HWEventListener *Listener) {
  Listeners.insert(Listener);
}

} // namespace mca
