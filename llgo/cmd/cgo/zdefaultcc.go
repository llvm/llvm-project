//===- zdefaultcc.go - default compiler locations -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a default location for cc.
//
//===----------------------------------------------------------------------===//

package main

import (
	"path/filepath"
	"os"
	"os/exec"
)

var defaultCC string

func getInstPrefix() (string, error) {
	path, err := exec.LookPath(os.Args[0])
	if err != nil {
		return "", err
	}

	path, err = filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}

	prefix := filepath.Join(path, "..", "..", "..", "..")
	return prefix, nil
}

func init() {
	prefix, err := getInstPrefix()
	if err != nil {
		panic(err.Error())
	}

	defaultCC = filepath.Join(prefix, "bin", "clang")
}
