//===- context.go - Build context utilities for llgo ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Build context utilities for llgo.
//
//===----------------------------------------------------------------------===//

package build

import (
	"errors"
	"go/build"
	"regexp"
	"strings"
)

type Context struct {
	build.Context

	// LLVM triple
	Triple string
}

// ContextFromTriple returns a new go/build.Context with GOOS and GOARCH
// configured from the given triple.
func ContextFromTriple(triple string) (*Context, error) {
	goos, goarch, err := parseTriple(triple)
	if err != nil {
		return nil, err
	}
	ctx := &Context{Context: build.Default, Triple: triple}
	ctx.GOOS = goos
	ctx.GOARCH = goarch
	ctx.BuildTags = append(ctx.BuildTags, "llgo")
	if triple == "pnacl" {
		ctx.BuildTags = append(ctx.BuildTags, "pnacl")
	}
	return ctx, nil
}

func parseTriple(triple string) (goos string, goarch string, err error) {
	if strings.ToLower(triple) == "pnacl" {
		return "nacl", "le32", nil
	}

	type REs struct{ re, out string }
	// reference: http://llvm.org/docs/doxygen/html/Triple_8cpp_source.html
	goarchREs := []REs{
		{"amd64|x86_64", "amd64"},
		{"i[3-9]86", "386"},
		{"xscale|((arm|thumb)(v.*)?)", "arm"},
	}
	goosREs := []REs{
		{"linux.*", "linux"},
		{"(darwin|macosx|ios).*", "darwin"},
		{"k?freebsd.*", "freebsd"},
		{"netbsd.*", "netbsd"},
		{"openbsd.*", "openbsd"},
	}
	match := func(list []REs, s string) string {
		for _, t := range list {
			if matched, _ := regexp.MatchString(t.re, s); matched {
				return t.out
			}
		}
		return ""
	}

	s := strings.Split(triple, "-")
	switch l := len(s); l {
	default:
		return "", "", errors.New("triple should be made up of 2, 3, or 4 parts.")
	case 2, 3: // ARCHITECTURE-(VENDOR-)OPERATING_SYSTEM
		goarch = s[0]
		goos = s[l-1]
	case 4: // ARCHITECTURE-VENDOR-OPERATING_SYSTEM-ENVIRONMENT
		goarch = s[0]
		goos = s[2]
	}
	goarch = match(goarchREs, goarch)
	if goarch == "" {
		return "", "", errors.New("unknown architecture in triple")
	}
	goos = match(goosREs, goos)
	if goos == "" {
		return "", "", errors.New("unknown OS in triple")
	}
	return goos, goarch, nil
}
