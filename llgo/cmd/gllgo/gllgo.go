//===- gllgo.go - gccgo-like driver for llgo ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is llgo's driver. It has a gccgo-like interface in order to easily
// interoperate with the "go" command and the libgo build system.
//
//===----------------------------------------------------------------------===//

package main

/*
#include "config.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"go/scanner"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"llvm.org/llgo/debug"
	"llvm.org/llgo/driver"
	"llvm.org/llgo/irgen"
	"llvm.org/llvm/bindings/go/llvm"
)

const LibDirSuffix = C.LLVM_LIBDIR_SUFFIX

func report(err error) {
	if list, ok := err.(scanner.ErrorList); ok {
		for _, e := range list {
			fmt.Fprintf(os.Stderr, "%s\n", e)
		}
	} else if err != nil {
		fmt.Fprintf(os.Stderr, "gllgo: error: %s\n", err)
	}
}

func llvmVersion() string {
	return strings.Replace(llvm.Version, "svn", "", 1)
}

func displayVersion() {
	fmt.Printf("llgo version %s (%s)\n\n", llvmVersion(), irgen.GoVersion())
	os.Exit(0)
}

func initCompiler(opts *driverOptions) (*irgen.Compiler, error) {
	importPaths := make([]string, len(opts.importPaths)+len(opts.libPaths))
	copy(importPaths, opts.importPaths)
	copy(importPaths[len(opts.importPaths):], opts.libPaths)
	if opts.prefix != "" {
		importPaths = append(importPaths, filepath.Join(opts.prefix, "lib"+LibDirSuffix, "go", "llgo-"+llvmVersion()))
	}
	copts := irgen.CompilerOptions{
		TargetTriple:       opts.triple,
		GenerateDebug:      opts.generateDebug,
		DebugPrefixMaps:    opts.debugPrefixMaps,
		DumpSSA:            opts.dumpSSA,
		GccgoPath:          opts.gccgoPath,
		GccgoABI:           opts.gccgoPath != "",
		ImportPaths:        importPaths,
		SanitizerAttribute: opts.sanitizer.getAttribute(),
	}
	if opts.dumpTrace {
		copts.Logger = log.New(os.Stderr, "", 0)
	}
	return irgen.NewCompiler(copts)
}

type actionKind int

const (
	actionAssemble = actionKind(iota)
	actionCompile
	actionLink
	actionPrint
)

type action struct {
	kind   actionKind
	inputs []string
}

type sanitizerOptions struct {
	blacklist string
	crtPrefix string

	address, thread, memory, dataflow bool
}

func (san *sanitizerOptions) resourcePath() string {
	return filepath.Join(san.crtPrefix, "lib"+LibDirSuffix, "clang", llvmVersion())
}

func (san *sanitizerOptions) isPIEDefault() bool {
	return san.thread || san.memory || san.dataflow
}

func (san *sanitizerOptions) addPasses(mpm, fpm llvm.PassManager) {
	switch {
	case san.address:
		mpm.AddAddressSanitizerModulePass()
		fpm.AddAddressSanitizerFunctionPass()
	case san.thread:
		mpm.AddThreadSanitizerPass()
	case san.memory:
		mpm.AddMemorySanitizerLegacyPassPass()
	case san.dataflow:
		blacklist := san.blacklist
		if blacklist == "" {
			blacklist = filepath.Join(san.resourcePath(), "dfsan_abilist.txt")
		}
		mpm.AddDataFlowSanitizerPass([]string{blacklist})
	}
}

func (san *sanitizerOptions) libPath(triple, sanitizerName string) string {
	s := strings.Split(triple, "-")
	return filepath.Join(san.resourcePath(), "lib", s[2], "libclang_rt."+sanitizerName+"-"+s[0]+".a")
}

func (san *sanitizerOptions) addLibsForSanitizer(flags []string, triple, sanitizerName string) []string {
	return append(flags, san.libPath(triple, sanitizerName),
		"-Wl,--no-as-needed", "-lpthread", "-lrt", "-lm", "-ldl")
}

func (san *sanitizerOptions) addLibs(triple string, flags []string) []string {
	switch {
	case san.address:
		flags = san.addLibsForSanitizer(flags, triple, "asan")
	case san.thread:
		flags = san.addLibsForSanitizer(flags, triple, "tsan")
	case san.memory:
		flags = san.addLibsForSanitizer(flags, triple, "msan")
	case san.dataflow:
		flags = san.addLibsForSanitizer(flags, triple, "dfsan")
	}

	return flags
}

func (san *sanitizerOptions) getAttribute() llvm.Attribute {
	var attrKind uint

	switch {
	case san.address:
		attrKind = llvm.AttributeKindID("sanitize_address")
	case san.thread:
		attrKind = llvm.AttributeKindID("sanitize_thread")
	case san.memory:
		attrKind = llvm.AttributeKindID("sanitize_memory")
	default:
		attrKind = 0
	}

	ctx := llvm.GlobalContext()
	return ctx.CreateEnumAttribute(attrKind, 0)
}

type driverOptions struct {
	actions []action
	output  string

	bprefix         string
	debugPrefixMaps []debug.PrefixMap
	dumpSSA         bool
	dumpTrace       bool
	emitIR          bool
	gccgoPath       string
	generateDebug   bool
	importPaths     []string
	libPaths        []string
	llvmArgs        []string
	lto             bool
	optLevel        int
	pic             bool
	pieLink         bool
	pkgpath         string
	plugins         []string
	prefix          string
	sanitizer       sanitizerOptions
	sizeLevel       int
	staticLibgcc    bool
	staticLibgo     bool
	staticLink      bool
	triple          string
}

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

func parseArguments(args []string) (opts driverOptions, err error) {
	var goInputs, otherInputs []string
	hasOtherNonFlagInputs := false
	noPrefix := false
	actionKind := actionLink
	opts.triple = llvm.DefaultTargetTriple()

	for len(args) > 0 {
		consumedArgs := 1

		switch {
		case !strings.HasPrefix(args[0], "-"):
			if strings.HasSuffix(args[0], ".go") {
				goInputs = append(goInputs, args[0])
			} else {
				hasOtherNonFlagInputs = true
				otherInputs = append(otherInputs, args[0])
			}

		case strings.HasPrefix(args[0], "-Wl,"), strings.HasPrefix(args[0], "-l"), strings.HasPrefix(args[0], "--sysroot="):
			// TODO(pcc): Handle these correctly.
			otherInputs = append(otherInputs, args[0])

		case args[0] == "-B":
			if len(args) == 1 {
				return opts, errors.New("missing argument after '-B'")
			}
			opts.bprefix = args[1]
			consumedArgs = 2

		case args[0] == "-D":
			if len(args) == 1 {
				return opts, errors.New("missing argument after '-D'")
			}
			otherInputs = append(otherInputs, args[0], args[1])
			consumedArgs = 2

		case strings.HasPrefix(args[0], "-D"):
			otherInputs = append(otherInputs, args[0])

		case args[0] == "-I":
			if len(args) == 1 {
				return opts, errors.New("missing argument after '-I'")
			}
			opts.importPaths = append(opts.importPaths, args[1])
			consumedArgs = 2

		case strings.HasPrefix(args[0], "-I"):
			opts.importPaths = append(opts.importPaths, args[0][2:])

		case args[0] == "-isystem":
			if len(args) == 1 {
				return opts, errors.New("missing argument after '-isystem'")
			}
			otherInputs = append(otherInputs, args[0], args[1])
			consumedArgs = 2

		case args[0] == "-L":
			if len(args) == 1 {
				return opts, errors.New("missing argument after '-L'")
			}
			opts.libPaths = append(opts.libPaths, args[1])
			consumedArgs = 2

		case strings.HasPrefix(args[0], "-L"):
			opts.libPaths = append(opts.libPaths, args[0][2:])

		case args[0] == "-O0":
			opts.optLevel = 0

		case args[0] == "-O1", args[0] == "-O":
			opts.optLevel = 1

		case args[0] == "-O2":
			opts.optLevel = 2

		case args[0] == "-Os":
			opts.optLevel = 2
			opts.sizeLevel = 1

		case args[0] == "-O3":
			opts.optLevel = 3

		case args[0] == "-S":
			actionKind = actionAssemble

		case args[0] == "-c":
			actionKind = actionCompile

		case strings.HasPrefix(args[0], "-fcompilerrt-prefix="):
			opts.sanitizer.crtPrefix = args[0][20:]

		case strings.HasPrefix(args[0], "-fdebug-prefix-map="):
			split := strings.SplitN(args[0][19:], "=", 2)
			if len(split) < 2 {
				return opts, fmt.Errorf("argument '%s' must be of form '-fdebug-prefix-map=SOURCE=REPLACEMENT'", args[0])
			}
			opts.debugPrefixMaps = append(opts.debugPrefixMaps, debug.PrefixMap{split[0], split[1]})

		case args[0] == "-fdump-ssa":
			opts.dumpSSA = true

		case args[0] == "-fdump-trace":
			opts.dumpTrace = true

		case strings.HasPrefix(args[0], "-fgccgo-path="):
			opts.gccgoPath = args[0][13:]

		case strings.HasPrefix(args[0], "-fgo-pkgpath="):
			opts.pkgpath = args[0][13:]

		case strings.HasPrefix(args[0], "-fgo-relative-import-path="):
			// TODO(pcc): Handle this.

		case strings.HasPrefix(args[0], "-fstack-protector"):
			// TODO(axw) set ssp function attributes. This can be useful
			// even for Go, if it interfaces with code written in a non-
			// memory safe language (e.g. via cgo).

		case strings.HasPrefix(args[0], "-W"):
			// Go doesn't do warnings. Ignore.

		case args[0] == "-fload-plugin":
			if len(args) == 1 {
				return opts, errors.New("missing argument after '-fload-plugin'")
			}
			opts.plugins = append(opts.plugins, args[1])
			consumedArgs = 2

		case args[0] == "-fno-toplevel-reorder":
			// This is a GCC-specific code generation option. Ignore.

		case args[0] == "-emit-llvm":
			opts.emitIR = true

		case args[0] == "-flto":
			opts.lto = true

		case args[0] == "-fPIC":
			opts.pic = true

		case strings.HasPrefix(args[0], "-fsanitize-blacklist="):
			opts.sanitizer.blacklist = args[0][21:]

		// TODO(pcc): Enforce mutual exclusion between sanitizers.

		case args[0] == "-fsanitize=address":
			opts.sanitizer.address = true

		case args[0] == "-fsanitize=thread":
			opts.sanitizer.thread = true

		case args[0] == "-fsanitize=memory":
			opts.sanitizer.memory = true

		case args[0] == "-fsanitize=dataflow":
			opts.sanitizer.dataflow = true

		case args[0] == "-g":
			opts.generateDebug = true

		case args[0] == "-mllvm":
			if len(args) == 1 {
				return opts, errors.New("missing argument after '-mllvm'")
			}
			opts.llvmArgs = append(opts.llvmArgs, args[1])
			consumedArgs = 2

		case strings.HasPrefix(args[0], "-m"), args[0] == "-funsafe-math-optimizations", args[0] == "-ffp-contract=off":
			// TODO(pcc): Handle code generation options.

		case args[0] == "-no-prefix":
			noPrefix = true

		case args[0] == "-o":
			if len(args) == 1 {
				return opts, errors.New("missing argument after '-o'")
			}
			opts.output = args[1]
			consumedArgs = 2

		case args[0] == "-pie":
			opts.pieLink = true

		case args[0] == "-dumpversion",
			args[0] == "-print-libgcc-file-name",
			args[0] == "-print-multi-os-directory",
			args[0] == "--version":
			actionKind = actionPrint
			opts.output = args[0]

		case args[0] == "-static":
			opts.staticLink = true

		case args[0] == "-static-libgcc":
			opts.staticLibgcc = true

		case args[0] == "-static-libgo":
			opts.staticLibgo = true

		default:
			return opts, fmt.Errorf("unrecognized command line option '%s'", args[0])
		}

		args = args[consumedArgs:]
	}

	if actionKind != actionPrint && len(goInputs) == 0 && !hasOtherNonFlagInputs {
		return opts, errors.New("no input files")
	}

	if !noPrefix {
		opts.prefix, err = getInstPrefix()
		if err != nil {
			return opts, err
		}
	}

	if opts.sanitizer.crtPrefix == "" {
		opts.sanitizer.crtPrefix = opts.prefix
	}

	if opts.sanitizer.isPIEDefault() {
		// This should really only be turning on -fPIE, but this isn't
		// easy to do from Go, and -fPIC is a superset of it anyway.
		opts.pic = true
		opts.pieLink = true
	}

	switch actionKind {
	case actionLink:
		if len(goInputs) != 0 {
			opts.actions = []action{action{actionCompile, goInputs}}
		}
		opts.actions = append(opts.actions, action{actionLink, otherInputs})

	case actionCompile, actionAssemble:
		if len(goInputs) != 0 {
			opts.actions = []action{action{actionKind, goInputs}}
		}

	case actionPrint:
		opts.actions = []action{action{actionKind, nil}}
	}

	if opts.output == "" && len(opts.actions) != 0 {
		switch actionKind {
		case actionCompile, actionAssemble:
			base := filepath.Base(goInputs[0])
			base = base[0 : len(base)-3]
			if actionKind == actionCompile {
				opts.output = base + ".o"
			} else {
				opts.output = base + ".s"
			}

		case actionLink:
			opts.output = "a.out"
		}
	}

	return opts, nil
}

func runPasses(opts *driverOptions, tm llvm.TargetMachine, m llvm.Module) {
	fpm := llvm.NewFunctionPassManagerForModule(m)
	defer fpm.Dispose()

	mpm := llvm.NewPassManager()
	defer mpm.Dispose()

	pmb := llvm.NewPassManagerBuilder()
	defer pmb.Dispose()

	pmb.SetOptLevel(opts.optLevel)
	pmb.SetSizeLevel(opts.sizeLevel)

	tm.AddAnalysisPasses(mpm)
	tm.AddAnalysisPasses(fpm)

	mpm.AddVerifierPass()
	fpm.AddVerifierPass()

	pmb.Populate(mpm)
	pmb.PopulateFunc(fpm)

	if opts.optLevel == 0 {
		// Remove references (via the descriptor) to dead functions,
		// for compatibility with other compilers.
		mpm.AddGlobalDCEPass()
	}

	opts.sanitizer.addPasses(mpm, fpm)

	fpm.InitializeFunc()
	for fn := m.FirstFunction(); !fn.IsNil(); fn = llvm.NextFunction(fn) {
		fpm.RunFunc(fn)
	}
	fpm.FinalizeFunc()

	mpm.Run(m)
}

func getMetadataSectionInlineAsm(name string) string {
	// ELF: creates a non-allocated excluded section.
	return ".section \"" + name + "\", \"e\"\n"
}

func getDataInlineAsm(data []byte) string {
	edata := make([]byte, 0, len(data)*4+10)

	edata = append(edata, ".ascii \""...)
	for i := range data {
		switch data[i] {
		case '\000':
			edata = append(edata, "\\000"...)
			continue
		case '\n':
			edata = append(edata, "\\n"...)
			continue
		case '"', '\\':
			edata = append(edata, '\\')
		}
		edata = append(edata, data[i])
	}
	edata = append(edata, "\"\n"...)
	return string(edata)
}

// Get the lib path to the standard libraries for the given driver options.
// This is normally 'lib' but can vary for cross compilation, LTO, sanitizers
// etc.
func getLibDir(opts *driverOptions) string {
	lib := "lib" + LibDirSuffix
	switch {
	case opts.lto:
		return filepath.Join(lib, "llvm-lto.0")
	case opts.sanitizer.address:
		return filepath.Join(lib, "llvm-asan.0")
	case opts.sanitizer.thread:
		return filepath.Join(lib, "llvm-tsan.0")
	case opts.sanitizer.memory:
		return filepath.Join(lib, "llvm-msan.0")
	case opts.sanitizer.dataflow:
		return filepath.Join(lib, "llvm-dfsan.0")
	default:
		return lib
	}
}

func performAction(opts *driverOptions, kind actionKind, inputs []string, output string) error {
	switch kind {
	case actionPrint:
		switch opts.output {
		case "-dumpversion":
			fmt.Println("llgo-" + llvmVersion())
			return nil
		case "-print-libgcc-file-name":
			cmd := exec.Command(opts.bprefix+"gcc", "-print-libgcc-file-name")
			out, err := cmd.CombinedOutput()
			os.Stdout.Write(out)
			return err
		case "-print-multi-os-directory":
			fmt.Println(filepath.Join("..", getLibDir(opts)))
			return nil
		case "--version":
			displayVersion()
			return nil
		default:
			panic("unexpected print command")
		}

	case actionCompile, actionAssemble:
		compiler, err := initCompiler(opts)
		if err != nil {
			return err
		}

		fset := token.NewFileSet()
		files, err := driver.ParseFiles(fset, inputs)
		if err != nil {
			return err
		}

		module, err := compiler.Compile(fset, files, opts.pkgpath)
		if err != nil {
			return err
		}

		defer module.Dispose()

		target, err := llvm.GetTargetFromTriple(opts.triple)
		if err != nil {
			return err
		}

		optLevel := [...]llvm.CodeGenOptLevel{
			llvm.CodeGenLevelNone,
			llvm.CodeGenLevelLess,
			llvm.CodeGenLevelDefault,
			llvm.CodeGenLevelAggressive,
		}[opts.optLevel]

		relocMode := llvm.RelocStatic
		if opts.pic {
			relocMode = llvm.RelocPIC
		}

		tm := target.CreateTargetMachine(opts.triple, "", "", optLevel,
			relocMode, llvm.CodeModelDefault)
		defer tm.Dispose()

		runPasses(opts, tm, module.Module)

		var file *os.File
		if output == "-" {
			file = os.Stdout
		} else {
			file, err = os.Create(output)
			if err != nil {
				return err
			}
			defer file.Close()
		}

		switch {
		case !opts.lto && !opts.emitIR:
			if module.ExportData != nil {
				asm := getMetadataSectionInlineAsm(".go_export")
				asm += getDataInlineAsm(module.ExportData)
				module.Module.SetInlineAsm(asm)
			}

			fileType := llvm.AssemblyFile
			if kind == actionCompile {
				fileType = llvm.ObjectFile
			}
			mb, err := tm.EmitToMemoryBuffer(module.Module, fileType)
			if err != nil {
				return err
			}
			defer mb.Dispose()

			bytes := mb.Bytes()
			_, err = file.Write(bytes)
			return err

		case opts.lto:
			bcmb := llvm.WriteBitcodeToMemoryBuffer(module.Module)
			defer bcmb.Dispose()

			// This is a bit of a hack. We just want an object file
			// containing some metadata sections. This might be simpler
			// if we had bindings for the MC library, but for now we create
			// a fresh module containing only inline asm that creates the
			// sections.
			outmodule := llvm.NewModule("")
			defer outmodule.Dispose()
			asm := getMetadataSectionInlineAsm(".llvmbc")
			asm += getDataInlineAsm(bcmb.Bytes())
			if module.ExportData != nil {
				asm += getMetadataSectionInlineAsm(".go_export")
				asm += getDataInlineAsm(module.ExportData)
			}
			outmodule.SetInlineAsm(asm)

			fileType := llvm.AssemblyFile
			if kind == actionCompile {
				fileType = llvm.ObjectFile
			}
			mb, err := tm.EmitToMemoryBuffer(outmodule, fileType)
			if err != nil {
				return err
			}
			defer mb.Dispose()

			bytes := mb.Bytes()
			_, err = file.Write(bytes)
			return err

		case kind == actionCompile:
			err := llvm.WriteBitcodeToFile(module.Module, file)
			return err

		case kind == actionAssemble:
			_, err := file.WriteString(module.Module.String())
			return err

		default:
			panic("unexpected action kind")
		}

	case actionLink:
		// TODO(pcc): Teach this to do LTO.
		args := []string{"-o", output}
		if opts.pic {
			args = append(args, "-fPIC")
		}
		if opts.pieLink {
			args = append(args, "-pie")
		}
		if opts.staticLink {
			args = append(args, "-static")
		}
		if opts.staticLibgcc {
			args = append(args, "-static-libgcc")
		}
		for _, p := range opts.libPaths {
			args = append(args, "-L", p)
		}
		for _, p := range opts.importPaths {
			args = append(args, "-I", p)
		}
		args = append(args, inputs...)
		var linkerPath string
		if opts.gccgoPath == "" {
			// TODO(pcc): See if we can avoid calling gcc here.
			// We currently rely on it to find crt*.o and compile
			// any C source files passed as arguments.
			linkerPath = opts.bprefix + "gcc"

			if opts.prefix != "" {
				libdir := filepath.Join(opts.prefix, getLibDir(opts))
				args = append(args, "-L", libdir)
				if !opts.staticLibgo {
					args = append(args, "-Wl,-rpath,"+libdir)
				}
			}

			args = append(args, "-lgobegin-llgo")
			if opts.staticLibgo {
				args = append(args, "-Wl,-Bstatic", "-lgo-llgo", "-Wl,-Bdynamic", "-lpthread", "-lm")
			} else {
				args = append(args, "-lgo-llgo", "-lm")
			}
		} else {
			linkerPath = opts.gccgoPath
			if opts.staticLibgo {
				args = append(args, "-static-libgo")
			}
		}

		args = opts.sanitizer.addLibs(opts.triple, args)

		cmd := exec.Command(linkerPath, args...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			os.Stderr.Write(out)
		}
		return err

	default:
		panic("unexpected action kind")
	}
}

func performActions(opts *driverOptions) error {
	var extraInput string

	for _, plugin := range opts.plugins {
		err := llvm.LoadLibraryPermanently(plugin)
		if err != nil {
			return err
		}
	}

	llvm.ParseCommandLineOptions(append([]string{"llgo"}, opts.llvmArgs...), "llgo (LLVM option parsing)\n")

	for i, action := range opts.actions {
		var output string
		if i == len(opts.actions)-1 {
			output = opts.output
		} else {
			tmpfile, err := ioutil.TempFile("", "llgo")
			if err != nil {
				return err
			}
			output = tmpfile.Name() + ".o"
			tmpfile.Close()
			err = os.Remove(tmpfile.Name())
			if err != nil {
				return err
			}
			defer os.Remove(output)
		}

		inputs := action.inputs
		if extraInput != "" {
			inputs = append([]string{extraInput}, inputs...)
		}

		err := performAction(opts, action.kind, inputs, output)
		if err != nil {
			return err
		}

		extraInput = output
	}

	return nil
}

func main() {
	llvm.InitializeAllTargets()
	llvm.InitializeAllTargetMCs()
	llvm.InitializeAllTargetInfos()
	llvm.InitializeAllAsmParsers()
	llvm.InitializeAllAsmPrinters()

	opts, err := parseArguments(os.Args[1:])
	if err != nil {
		report(err)
		os.Exit(1)
	}

	err = performActions(&opts)
	if err != nil {
		report(err)
		os.Exit(1)
	}
}
