//===- targets.go - target data -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions for retrieving target-specific data.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"fmt"
	"strings"

	"llvm.org/llvm/bindings/go/llvm"
)

// llvmDataLayout returns the data layout string
// representation for the specified LLVM triple.
func llvmDataLayout(triple string) (string, error) {
	// Triples are several fields separated by '-' characters.
	// The first field is the architecture. The architecture's
	// canonical form may include a '-' character, which would
	// have been translated to '_' for inclusion in a triple.
	arch := parseArch(triple[:strings.IndexRune(triple, '-')])
	for target := llvm.FirstTarget(); target.C != nil; target = target.NextTarget() {
		if arch == target.Name() {
			machine := target.CreateTargetMachine(
				triple, "", "",
				llvm.CodeGenLevelDefault,
				llvm.RelocDefault,
				llvm.CodeModelDefault,
			)
			targetData := machine.CreateTargetData()
			targetDataLayout := targetData.String()
			targetData.Dispose()
			machine.Dispose()
			return targetDataLayout, nil
		}
	}
	return "", fmt.Errorf("Invalid target triple: %s", triple)
}

// Based on parseArch from LLVM's lib/Support/Triple.cpp.
// This is used to match the target machine type.
func parseArch(arch string) string {
	switch arch {
	case "i386", "i486", "i586", "i686", "i786", "i886", "i986":
		return "x86"
	case "amd64", "x86_64":
		return "x86-64"
	case "powerpc":
		return "ppc"
	case "powerpc64", "ppu":
		return "ppc64"
	case "mblaze":
		return "mblaze"
	case "arm", "xscale":
		return "arm"
	case "thumb":
		return "thumb"
	case "spu", "cellspu":
		return "cellspu"
	case "msp430":
		return "msp430"
	case "mips", "mipseb", "mipsallegrex":
		return "mips"
	case "mipsel", "mipsallegrexel":
		return "mipsel"
	case "mips64", "mips64eb":
		return "mips64"
	case "mipsel64":
		return "mipsel64"
	case "r600", "hexagon", "sparc", "sparcv9", "tce",
		"xcore", "nvptx", "nvptx64", "le32", "amdil":
		return arch
	}
	if strings.HasPrefix(arch, "armv") {
		return "arm"
	} else if strings.HasPrefix(arch, "thumbv") {
		return "thumb"
	}
	return "unknown"
}
