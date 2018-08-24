//===- parser.go - parser wrapper -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions for calling the parser in an appropriate way for
// llgo.
//
//===----------------------------------------------------------------------===//

package driver

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
)

func parseFile(fset *token.FileSet, filename string) (*ast.File, error) {
	// Retain comments; this is important for annotation processing.
	mode := parser.DeclarationErrors | parser.ParseComments
	return parser.ParseFile(fset, filename, nil, mode)
}

func ParseFiles(fset *token.FileSet, filenames []string) ([]*ast.File, error) {
	files := make([]*ast.File, len(filenames))
	for i, filename := range filenames {
		file, err := parseFile(fset, filename)
		if _, ok := err.(scanner.ErrorList); ok {
			return nil, err
		} else if err != nil {
			return nil, fmt.Errorf("%q: %v", filename, err)
		}
		files[i] = file
	}
	return files, nil
}
