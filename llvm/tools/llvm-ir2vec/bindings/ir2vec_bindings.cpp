//===- ir2vec_bindings.cpp - Python Bindings for IR2Vec ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(py_ir2vec, m) { m.doc() = "Python bindings for IR2Vec"; }
