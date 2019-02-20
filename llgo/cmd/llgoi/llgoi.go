//===- llgoi.go - llgo-based Go REPL --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is llgoi, a Go REPL based on llgo and the LLVM JIT.
//
//===----------------------------------------------------------------------===//

package main

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/scanner"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"
	"unsafe"

	"llvm.org/llgo/driver"
	"llvm.org/llgo/irgen"
	"llvm.org/llgo/third_party/gotools/go/types"
	"llvm.org/llgo/third_party/liner"
	"llvm.org/llvm/bindings/go/llvm"
)

// /* Force exporting __morestack if it's available, so that it is
//    available to the engine when linking with libLLVM.so. */
//
// void *__morestack __attribute__((weak));
import "C"

func getInstPrefix() (string, error) {
	path, err := exec.LookPath(os.Args[0])
	if err != nil {
		return "", err
	}

	path, err = filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}

	prefix := filepath.Join(path, "..", "..")
	return prefix, nil
}

func llvmVersion() string {
	return strings.Replace(llvm.Version, "svn", "", 1)
}

type line struct {
	line     string
	isStmt   bool
	declName string
	assigns  []string

	parens, bracks, braces int
}

type interp struct {
	engine llvm.ExecutionEngine

	liner       *liner.State
	pendingLine line

	copts irgen.CompilerOptions

	imports []*types.Package
	scope   map[string]types.Object

	modules map[string]llvm.Module
	pkgmap  map[string]*types.Package
	pkgnum  int
}

func (in *interp) makeCompilerOptions() error {
	prefix, err := getInstPrefix()
	if err != nil {
		return err
	}

	importPaths := []string{filepath.Join(prefix, "lib", "go", "llgo-"+llvmVersion())}
	in.copts = irgen.CompilerOptions{
		TargetTriple:  llvm.DefaultTargetTriple(),
		ImportPaths:   importPaths,
		GenerateDebug: true,
		Packages:      in.pkgmap,
	}
	err = in.copts.MakeImporter()
	if err != nil {
		return err
	}

	origImporter := in.copts.Importer
	in.copts.Importer = func(pkgmap map[string]*types.Package, pkgpath string) (*types.Package, error) {
		if pkg, ok := pkgmap[pkgpath]; ok && pkg.Complete() {
			return pkg, nil
		}
		return origImporter(pkgmap, pkgpath)
	}
	return nil
}

func (in *interp) init() error {
	in.liner = liner.NewLiner()
	in.scope = make(map[string]types.Object)
	in.pkgmap = make(map[string]*types.Package)
	in.modules = make(map[string]llvm.Module)

	err := in.makeCompilerOptions()
	if err != nil {
		return err
	}

	return nil
}

func (in *interp) dispose() {
	in.liner.Close()
	in.engine.Dispose()
}

func (in *interp) loadSourcePackageFromCode(pkgcode, pkgpath string, copts irgen.CompilerOptions) (*types.Package, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "<input>", pkgcode, parser.DeclarationErrors|parser.ParseComments)
	if err != nil {
		return nil, err
	}
	files := []*ast.File{file}
	return in.loadSourcePackage(fset, files, pkgpath, copts)
}

func (in *interp) loadSourcePackage(fset *token.FileSet, files []*ast.File, pkgpath string, copts irgen.CompilerOptions) (_ *types.Package, resultErr error) {
	compiler, err := irgen.NewCompiler(copts)
	if err != nil {
		return nil, err
	}

	module, err := compiler.Compile(fset, files, pkgpath)
	if err != nil {
		return nil, err
	}
	in.modules[pkgpath] = module.Module

	if in.engine.C != nil {
		in.engine.AddModule(module.Module)
	} else {
		options := llvm.NewMCJITCompilerOptions()
		in.engine, err = llvm.NewMCJITCompiler(module.Module, options)
		if err != nil {
			return nil, err
		}
	}

	var importFunc func()
	importAddress := in.getPackageSymbol(pkgpath, ".import$descriptor")
	*(*unsafe.Pointer)(unsafe.Pointer(&importFunc)) = importAddress

	defer func() {
		p := recover()
		if p != nil {
			resultErr = fmt.Errorf("panic: %v\n%v", p, string(debug.Stack()))
		}
	}()
	importFunc()
	in.pkgmap[pkgpath] = module.Package

	return module.Package, nil
}

func (in *interp) getPackageSymbol(pkgpath, name string) unsafe.Pointer {
	symbolName := irgen.ManglePackagePath(pkgpath) + "." + name
	global := in.modules[pkgpath].NamedGlobal(symbolName)
	if global.IsNil() {
		return nil
	}
	return in.engine.PointerToGlobal(global)
}

func (in *interp) augmentPackageScope(pkg *types.Package) {
	for _, obj := range in.scope {
		pkg.Scope().Insert(obj)
	}
}

func (l *line) append(str string, assigns []string) {
	var s scanner.Scanner
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(str))
	s.Init(file, []byte(str), nil, 0)

	_, tok, _ := s.Scan()
	if l.line == "" {
		switch tok {
		case token.FOR, token.GO, token.IF, token.LBRACE, token.SELECT, token.SWITCH:
			l.isStmt = true
		case token.CONST, token.FUNC, token.TYPE, token.VAR:
			var lit string
			_, tok, lit = s.Scan()
			if tok == token.IDENT {
				l.declName = lit
			}
		}
	}

	for tok != token.EOF {
		switch tok {
		case token.LPAREN:
			l.parens++
		case token.RPAREN:
			l.parens--
		case token.LBRACE:
			l.braces++
		case token.RBRACE:
			l.braces--
		case token.LBRACK:
			l.bracks++
		case token.RBRACK:
			l.bracks--
		case token.DEC, token.INC,
			token.ASSIGN, token.ADD_ASSIGN, token.SUB_ASSIGN,
			token.MUL_ASSIGN, token.QUO_ASSIGN, token.REM_ASSIGN,
			token.AND_ASSIGN, token.OR_ASSIGN, token.XOR_ASSIGN,
			token.SHL_ASSIGN, token.SHR_ASSIGN, token.AND_NOT_ASSIGN:
			if l.parens == 0 && l.bracks == 0 && l.braces == 0 {
				l.isStmt = true
			}
		}
		_, tok, _ = s.Scan()
	}

	if l.line == "" {
		l.assigns = assigns
	}
	l.line += str
}

func (l *line) ready() bool {
	return l.parens <= 0 && l.bracks <= 0 && l.braces <= 0
}

func (in *interp) readExprLine(str string, assigns []string) ([]interface{}, error) {
	in.pendingLine.append(str, assigns)
	if !in.pendingLine.ready() {
		return nil, nil
	}
	results, err := in.interpretLine(in.pendingLine)
	in.pendingLine = line{}
	return results, err
}

func (in *interp) interpretLine(l line) ([]interface{}, error) {
	pkgname := fmt.Sprintf("input%05d", in.pkgnum)
	in.pkgnum++

	pkg := types.NewPackage(pkgname, pkgname)
	scope := pkg.Scope()

	for _, imppkg := range in.imports {
		obj := types.NewPkgName(token.NoPos, pkg, imppkg.Name(), imppkg)
		scope.Insert(obj)
	}

	in.augmentPackageScope(pkg)

	var tv types.TypeAndValue
	if l.declName == "" && !l.isStmt {
		var err error
		tv, err = types.Eval(l.line, pkg, scope)
		if err != nil {
			return nil, err
		}
	}

	var code bytes.Buffer
	fmt.Fprintf(&code, "package %s\n", pkgname)

	for _, pkg := range in.imports {
		fmt.Fprintf(&code, "import %q\n", pkg.Path())
	}

	if l.declName != "" {
		code.WriteString(l.line)
	} else if !l.isStmt && tv.IsValue() {
		var typs []types.Type
		if tuple, ok := tv.Type.(*types.Tuple); ok {
			typs = make([]types.Type, tuple.Len())
			for i := range typs {
				typs[i] = tuple.At(i).Type()
			}
		} else {
			typs = []types.Type{tv.Type}
		}
		if len(l.assigns) == 2 && tv.HasOk() {
			typs = append(typs, types.Typ[types.Bool])
		}
		if len(l.assigns) != 0 && len(l.assigns) != len(typs) {
			return nil, errors.New("return value mismatch")
		}

		code.WriteString("var ")
		for i := range typs {
			if i != 0 {
				code.WriteString(", ")
			}
			if len(l.assigns) != 0 && l.assigns[i] != "" {
				if _, ok := in.scope[l.assigns[i]]; ok {
					fmt.Fprintf(&code, "__llgoiV%d", i)
				} else {
					code.WriteString(l.assigns[i])
				}
			} else {
				fmt.Fprintf(&code, "__llgoiV%d", i)
			}
		}
		fmt.Fprintf(&code, " = %s\n", l.line)

		code.WriteString("func init() {\n")
		varnames := make([]string, len(typs))
		for i := range typs {
			var varname string
			if len(l.assigns) != 0 && l.assigns[i] != "" {
				if _, ok := in.scope[l.assigns[i]]; ok {
					fmt.Fprintf(&code, "\t%s = __llgoiV%d\n", l.assigns[i], i)
				}
				varname = l.assigns[i]
			} else {
				varname = fmt.Sprintf("__llgoiV%d", i)
			}
			varnames[i] = varname
		}
		code.WriteString("}\n\n")

		code.WriteString("func __llgoiResults() []interface{} {\n")
		code.WriteString("\treturn []interface{}{\n")
		for _, varname := range varnames {
			fmt.Fprintf(&code, "\t\t%s,\n", varname)
		}
		code.WriteString("\t}\n")
		code.WriteString("}\n")
	} else {
		if len(l.assigns) != 0 {
			return nil, errors.New("return value mismatch")
		}

		fmt.Fprintf(&code, "func init() {\n\t%s}", l.line)
	}

	copts := in.copts
	copts.PackageCreated = in.augmentPackageScope
	copts.DisableUnusedImportCheck = true
	pkg, err := in.loadSourcePackageFromCode(code.String(), pkgname, copts)
	if err != nil {
		return nil, err
	}
	in.imports = append(in.imports, pkg)

	var results []interface{}
	llgoiResultsAddress := in.getPackageSymbol(pkgname, "__llgoiResults$descriptor")
	if llgoiResultsAddress != nil {
		var resultsFunc func() []interface{}
		*(*unsafe.Pointer)(unsafe.Pointer(&resultsFunc)) = llgoiResultsAddress
		results = resultsFunc()
	}

	for _, assign := range l.assigns {
		if assign != "" {
			if _, ok := in.scope[assign]; !ok {
				in.scope[assign] = pkg.Scope().Lookup(assign)
			}
		}
	}

	if l.declName != "" {
		in.scope[l.declName] = pkg.Scope().Lookup(l.declName)
	}

	return results, nil
}

func (in *interp) maybeReadAssignment(line string, s *scanner.Scanner, initial string, base int) (bool, error) {
	if initial == "_" {
		initial = ""
	}
	assigns := []string{initial}

	pos, tok, lit := s.Scan()
	for tok == token.COMMA {
		pos, tok, lit = s.Scan()
		if tok != token.IDENT {
			return false, nil
		}

		if lit == "_" {
			lit = ""
		}
		assigns = append(assigns, lit)

		pos, tok, lit = s.Scan()
	}

	if tok != token.DEFINE {
		return false, nil
	}

	// It's an assignment statement, there are no results.
	_, err := in.readExprLine(line[int(pos)-base+2:], assigns)
	return true, err
}

func (in *interp) loadPackage(pkgpath string) (*types.Package, error) {
	pkg, err := in.copts.Importer(in.pkgmap, pkgpath)
	if err == nil {
		return pkg, nil
	}

	buildpkg, err := build.Import(pkgpath, ".", 0)
	if err != nil {
		return nil, err
	}
	if len(buildpkg.CgoFiles) != 0 {
		return nil, fmt.Errorf("%s: cannot load cgo package", pkgpath)
	}

	for _, imp := range buildpkg.Imports {
		_, err := in.loadPackage(imp)
		if err != nil {
			return nil, err
		}
	}

	inputs := make([]string, len(buildpkg.GoFiles))
	for i, file := range buildpkg.GoFiles {
		inputs[i] = filepath.Join(buildpkg.Dir, file)
	}

	fset := token.NewFileSet()
	files, err := driver.ParseFiles(fset, inputs)
	if err != nil {
		return nil, err
	}

	return in.loadSourcePackage(fset, files, pkgpath, in.copts)
}

// readLine accumulates lines of input, including trailing newlines,
// executing statements as they are completed.
func (in *interp) readLine(line string) ([]interface{}, error) {
	if !in.pendingLine.ready() {
		return in.readExprLine(line, nil)
	}

	var s scanner.Scanner
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(line))
	s.Init(file, []byte(line), nil, 0)

	_, tok, lit := s.Scan()
	switch tok {
	case token.EOF:
		return nil, nil

	case token.IMPORT:
		_, tok, lit = s.Scan()
		if tok != token.STRING {
			return nil, errors.New("expected string literal")
		}
		pkgpath, err := strconv.Unquote(lit)
		if err != nil {
			return nil, err
		}
		pkg, err := in.loadPackage(pkgpath)
		if err != nil {
			return nil, err
		}
		in.imports = append(in.imports, pkg)
		return nil, nil

	case token.IDENT:
		ok, err := in.maybeReadAssignment(line, &s, lit, file.Base())
		if err != nil {
			return nil, err
		}
		if ok {
			return nil, nil
		}
		fallthrough

	default:
		return in.readExprLine(line, nil)
	}
}

// printResult prints a value that was the result of an expression evaluated
// by the interpreter.
func printResult(w io.Writer, v interface{}) {
	// TODO the result should be formatted in Go syntax, without
	// package qualifiers for types defined within the interpreter.
	fmt.Fprintf(w, "%+v", v)
}

// formatHistory reformats the provided Go source by collapsing all lines
// and adding semicolons where required, suitable for adding to line history.
func formatHistory(input []byte) string {
	var buf bytes.Buffer
	var s scanner.Scanner
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(input))
	s.Init(file, input, nil, 0)
	pos, tok, lit := s.Scan()
	for tok != token.EOF {
		if int(pos)-1 > buf.Len() {
			n := int(pos) - 1 - buf.Len()
			buf.WriteString(strings.Repeat(" ", n))
		}
		var semicolon bool
		if tok == token.SEMICOLON {
			semicolon = true
		} else if lit != "" {
			buf.WriteString(lit)
		} else {
			buf.WriteString(tok.String())
		}
		pos, tok, lit = s.Scan()
		if semicolon {
			switch tok {
			case token.RBRACE, token.RPAREN, token.EOF:
			default:
				buf.WriteRune(';')
			}
		}
	}
	return buf.String()
}

func main() {
	llvm.LinkInMCJIT()
	llvm.InitializeNativeTarget()
	llvm.InitializeNativeAsmPrinter()

	var in interp
	err := in.init()
	if err != nil {
		panic(err)
	}
	defer in.dispose()

	var buf bytes.Buffer
	for {
		if in.pendingLine.ready() && buf.Len() > 0 {
			history := formatHistory(buf.Bytes())
			in.liner.AppendHistory(history)
			buf.Reset()
		}
		prompt := "(llgo) "
		if !in.pendingLine.ready() {
			prompt = strings.Repeat(" ", len(prompt))
		}
		line, err := in.liner.Prompt(prompt)
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		if line == "" {
			continue
		}
		buf.WriteString(line + "\n")
		results, err := in.readLine(line + "\n")
		if err != nil {
			fmt.Println(err)
		}
		for _, result := range results {
			printResult(os.Stdout, result)
			fmt.Println()
		}
	}

	if liner.TerminalSupported() {
		fmt.Println()
	}
}
