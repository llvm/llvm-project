//===- version.go - version info ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
