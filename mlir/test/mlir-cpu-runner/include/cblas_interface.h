//===- cblas_interface.h - Simple Blas subset interface -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CPU_RUNNER_CBLAS_INTERFACE_H_
#define MLIR_CPU_RUNNER_CBLAS_INTERFACE_H_

#include "mlir_runner_utils.h"

#ifdef _WIN32
#ifndef MLIR_CBLAS_INTERFACE_EXPORT
#ifdef cblas_interface_EXPORTS
/* We are building this library */
#define MLIR_CBLAS_INTERFACE_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define MLIR_CBLAS_INTERFACE_EXPORT __declspec(dllimport)
#endif // cblas_interface_EXPORTS
#endif // MLIR_CBLAS_INTERFACE_EXPORT
#else
#define MLIR_CBLAS_INTERFACE_EXPORT
#endif // _WIN32

extern "C" MLIR_CBLAS_INTERFACE_EXPORT void
linalg_fill_viewf32_f32(StridedMemRefType<float, 0> *X, float f);

extern "C" MLIR_CBLAS_INTERFACE_EXPORT void
linalg_fill_viewsxf32_f32(StridedMemRefType<float, 1> *X, float f);

extern "C" MLIR_CBLAS_INTERFACE_EXPORT void
linalg_fill_viewsxsxf32_f32(StridedMemRefType<float, 2> *X, float f);

extern "C" MLIR_CBLAS_INTERFACE_EXPORT void
linalg_copy_viewf32_viewf32(StridedMemRefType<float, 0> *I,
                            StridedMemRefType<float, 0> *O);

extern "C" MLIR_CBLAS_INTERFACE_EXPORT void
linalg_copy_viewsxf32_viewsxf32(StridedMemRefType<float, 1> *I,
                                StridedMemRefType<float, 1> *O);

extern "C" MLIR_CBLAS_INTERFACE_EXPORT void
linalg_copy_viewsxsxf32_viewsxsxf32(StridedMemRefType<float, 2> *I,
                                    StridedMemRefType<float, 2> *O);

extern "C" MLIR_CBLAS_INTERFACE_EXPORT void
linalg_dot_viewsxf32_viewsxf32_viewf32(StridedMemRefType<float, 1> *X,
                                       StridedMemRefType<float, 1> *Y,
                                       StridedMemRefType<float, 0> *Z);

extern "C" MLIR_CBLAS_INTERFACE_EXPORT void
linalg_matmul_viewsxsxf32_viewsxsxf32_viewsxsxf32(
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    StridedMemRefType<float, 2> *C);

#endif // MLIR_CPU_RUNNER_CBLAS_INTERFACE_H_
