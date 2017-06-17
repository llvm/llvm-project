//===- git-r.go - svn revisions to git revisions --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a small program for mapping svn revisions to git revisions in the
// monorepo.
//
// To set up:
// 1) http://llvm.org/docs/GettingStarted.html#for-developers-to-work-with-a-git-monorepo
//    and make sure to follow the instructions for fetching commit notes.
// 2) go build
// 3) cp git-r ~/bin
//
// To use:
// $ git r 1
// 09c4b68e68c4fcff64b00e1ac077c4f4a524cbcc
//
//===----------------------------------------------------------------------===//

package main

import (
	"bufio"
	"bytes"
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

func git(args ...string) (*bytes.Buffer, error) {
	cmd := exec.Command("git", args...)

	var b bytes.Buffer
	cmd.Stdout = &b
	err := cmd.Run()

	return &b, err
}

func mkrevmap() []string {
	revs, err := git("grep", "git-svn-rev", "refs/notes/commits")
	if err != nil {
		panic(err)
	}

	var revmap []string

	scanner := bufio.NewScanner(revs)
	for scanner.Scan() {
		// refs/notes/commits:00/0b/d4acb454290301c140a1d9c4f7a45aa2fa9c:git-svn-rev: 37235

		bits := strings.Split(scanner.Text(), ":")
		gitrev := strings.Replace(bits[1], "/", "", -1)
		svnrev := bits[3][1:]

		svnrevn, err := strconv.Atoi(svnrev)
		if err != nil {
			panic(err)
		}

		if svnrevn >= len(revmap) {
			newrevmap := make([]string, svnrevn+1)
			copy(newrevmap, revmap)
			revmap = newrevmap
		}
		revmap[svnrevn] = gitrev
	}

	return revmap
}

type revmap struct {
	Noterev string
	Revs    []string
}

func writerevmap(path string, rmap *revmap, svnrev int) {
	noterevbuf, err := git("rev-parse", "refs/notes/commits")
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: could not find refs/notes/commits, see instructions:\n", os.Args[0])
		fmt.Fprintln(os.Stderr, "http://llvm.org/docs/GettingStarted.html#for-developers-to-work-with-a-git-monorepo")
		os.Exit(1)
	}
	noterev := noterevbuf.String()
	noterev = noterev[:len(noterev)-1]

	if rmap == nil || rmap.Noterev != noterev {
		var newrmap revmap
		newrmap.Revs = mkrevmap()
		newrmap.Noterev = noterev

		f, err := os.Create(path)
		if err != nil {
			panic(err)
		}

		enc := gob.NewEncoder(f)
		err = enc.Encode(newrmap)
		if err != nil {
			os.Remove(path)
			panic(err)
		}

		rmap = &newrmap
	}

	if svnrev >= len(rmap.Revs) || rmap.Revs[svnrev] == "" {
		fmt.Fprintf(os.Stderr, "%s: %d: unknown revision\n", os.Args[0], svnrev)
		os.Exit(1)
	}

	fmt.Println(rmap.Revs[svnrev])
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "%s: expected a single argument\n", os.Args[0])
		os.Exit(1)
	}
	svnrev, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %s: expected an integer argument\n", os.Args[0], os.Args[1])
		os.Exit(1)
	}

	gitdirbuf, err := git("rev-parse", "--git-common-dir")
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: not in a git repository\n", os.Args[0])
		os.Exit(1)
	}

	gitdir := gitdirbuf.String()
	gitdir = gitdir[:len(gitdir)-1]
	err = os.Chdir(gitdir)
	if err != nil {
		panic(err)
	}

	mappath := "git-svn-revmap-cache"
	f, err := os.Open(mappath)
	if err != nil {
		writerevmap(mappath, nil, svnrev)
		return
	}

	dec := gob.NewDecoder(f)
	var rmap revmap
	err = dec.Decode(&rmap)
	if err != nil {
		writerevmap(mappath, nil, svnrev)
		return
	}

	if svnrev < len(rmap.Revs) && rmap.Revs[svnrev] != "" {
		fmt.Println(rmap.Revs[svnrev])
		return
	}

	writerevmap(mappath, &rmap, svnrev)
}
