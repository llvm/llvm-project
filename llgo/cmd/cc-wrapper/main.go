//===- main.go - Clang compiler wrapper for building libgo ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a wrapper for Clang that passes invocations with -fdump-go-spec to
// GCC, and rewrites -fplan9-extensions to -fms-extensions. It is intended to
// go away once libgo's build no longer uses these flags.
//
//===----------------------------------------------------------------------===//

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
)

func runproc(name string, argv []string) {
	path, err := exec.LookPath(name)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cc-wrapper: could not find %s: %v\n", name, err)
		os.Exit(1)
	}

	proc, err := os.StartProcess(path, append([]string{name}, argv...), &os.ProcAttr{
		Files: []*os.File{os.Stdin, os.Stdout, os.Stderr},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "cc-wrapper: could not start %s: %v\n", name, err)
		os.Exit(1)
	}

	state, err := proc.Wait()
	if err != nil {
		fmt.Fprintf(os.Stderr, "cc-wrapper: could not wait for %s: %v\n", name, err)
		os.Exit(1)
	}

	if state.Success() {
		os.Exit(0)
	} else {
		os.Exit(1)
	}
}

func main() {
	newargs := make([]string, len(os.Args)-1)
	for i, arg := range os.Args[1:] {
		switch {
		case strings.HasPrefix(arg, "-fdump-go-spec"):
			runproc("gcc", os.Args[1:])

		case arg == "-fplan9-extensions":
			newargs[i] = "-fms-extensions"
			newargs = append(newargs, "-Wno-microsoft")

		default:
			newargs[i] = arg
		}
	}

	ccargs := strings.Split(os.Getenv("REAL_CC"), "@SPACE@")
	runproc(ccargs[0], append(ccargs[1:], newargs...))
}
