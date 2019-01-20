//===- llvm_dep.go - creates LLVM dependency ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file ensures that the LLVM libraries are built before using the
// bindings.
//
//===----------------------------------------------------------------------===//

// +build !byollvm

package llvm

var _ run_build_sh
