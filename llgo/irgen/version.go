//===- version.go - version info ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file specifies the Go version supported by the IR generator.
//
//===----------------------------------------------------------------------===//

package irgen

const (
	goVersion = "go1.4.2"
)

// GoVersion returns the version of Go that we are targeting.
func GoVersion() string {
	return goVersion
}
