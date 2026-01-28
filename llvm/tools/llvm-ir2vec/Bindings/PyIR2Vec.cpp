//===- PyIR2Vec.cpp - Python Bindings for IR2Vec ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(ir2vec, m) { m.doc() = "Python bindings for IR2Vec"; }
