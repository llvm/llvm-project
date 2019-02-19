//===- zdefaultcc.go - default compiler locations -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
