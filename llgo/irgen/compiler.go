//===- compiler.go - IR generator entry point -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main IR generator entry point, (*Compiler).Compile.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"bytes"
	"go/ast"
	"go/token"
	"log"
	"sort"
	"strconv"

	llgobuild "llvm.org/llgo/build"
	"llvm.org/llgo/debug"
	"llvm.org/llvm/bindings/go/llvm"

	"llvm.org/llgo/third_party/gotools/go/gccgoimporter"
	"llvm.org/llgo/third_party/gotools/go/importer"
	"llvm.org/llgo/third_party/gotools/go/loader"
	"llvm.org/llgo/third_party/gotools/go/ssa"
	"llvm.org/llgo/third_party/gotools/go/types"
)

type Module struct {
	llvm.Module
	Path       string
	ExportData []byte
	Package    *types.Package
	disposed   bool
}

func (m *Module) Dispose() {
	if m.disposed {
		return
	}
	m.Module.Dispose()
	m.disposed = true
}

///////////////////////////////////////////////////////////////////////////////

type CompilerOptions struct {
	// TargetTriple is the LLVM triple for the target.
	TargetTriple string

	// GenerateDebug decides whether debug data is
	// generated in the output module.
	GenerateDebug bool

	// DebugPrefixMaps is a list of mappings from source prefixes to
	// replacement prefixes, to be applied in debug info.
	DebugPrefixMaps []debug.PrefixMap

	// Logger is a logger used for tracing compilation.
	Logger *log.Logger

	// DumpSSA is a debugging option that dumps each SSA function
	// to stderr before generating code for it.
	DumpSSA bool

	// GccgoPath is the path to the gccgo binary whose libgo we read import
	// data from. If blank, the caller is expected to supply an import
	// path in ImportPaths.
	GccgoPath string

	// Whether to use the gccgo ABI.
	GccgoABI bool

	// ImportPaths is the list of additional import paths
	ImportPaths []string

	// SanitizerAttribute is an attribute to apply to functions to enable
	// dynamic instrumentation using a sanitizer.
	SanitizerAttribute llvm.Attribute

	// Importer is the importer. If nil, the compiler will set this field
	// automatically using MakeImporter().
	Importer types.Importer

	// InitMap is the init map used by Importer. If Importer is nil, the
	// compiler will set this field automatically using MakeImporter().
	// If Importer is non-nil, InitMap must be non-nil also.
	InitMap map[*types.Package]gccgoimporter.InitData

	// PackageCreated is a hook passed to the go/loader package via
	// loader.Config, see the documentation for that package for more
	// information.
	PackageCreated func(*types.Package)

	// DisableUnusedImportCheck disables the unused import check performed
	// by go/types if set to true.
	DisableUnusedImportCheck bool

	// Packages is used by go/types as the imported package map if non-nil.
	Packages map[string]*types.Package
}

type Compiler struct {
	opts       CompilerOptions
	dataLayout string
}

func NewCompiler(opts CompilerOptions) (*Compiler, error) {
	compiler := &Compiler{opts: opts}
	dataLayout, err := llvmDataLayout(compiler.opts.TargetTriple)
	if err != nil {
		return nil, err
	}
	compiler.dataLayout = dataLayout
	return compiler, nil
}

func (c *Compiler) Compile(fset *token.FileSet, astFiles []*ast.File, importpath string) (m *Module, err error) {
	target := llvm.NewTargetData(c.dataLayout)
	compiler := &compiler{
		CompilerOptions: c.opts,
		dataLayout:      c.dataLayout,
		target:          target,
		llvmtypes:       NewLLVMTypeMap(llvm.GlobalContext(), target),
	}
	return compiler.compile(fset, astFiles, importpath)
}

type compiler struct {
	CompilerOptions

	module     *Module
	dataLayout string
	target     llvm.TargetData
	fileset    *token.FileSet

	runtime   *runtimeInterface
	llvmtypes *llvmTypeMap
	types     *TypeMap

	debug *debug.DIBuilder
}

func (c *compiler) logf(format string, v ...interface{}) {
	if c.Logger != nil {
		c.Logger.Printf(format, v...)
	}
}

func (c *compiler) addCommonFunctionAttrs(fn llvm.Value) {
	fn.AddTargetDependentFunctionAttr("disable-tail-calls", "true")
	fn.AddTargetDependentFunctionAttr("split-stack", "")
	if c.SanitizerAttribute.GetEnumKind() != 0 {
		fn.AddFunctionAttr(c.SanitizerAttribute)
	}
}

// MakeImporter sets CompilerOptions.Importer to an appropriate importer
// for the search paths given in CompilerOptions.ImportPaths, and sets
// CompilerOptions.InitMap to an init map belonging to that importer.
// If CompilerOptions.GccgoPath is non-empty, the importer will also use
// the search paths for that gccgo installation.
func (opts *CompilerOptions) MakeImporter() error {
	opts.InitMap = make(map[*types.Package]gccgoimporter.InitData)
	if opts.GccgoPath == "" {
		paths := append(append([]string{}, opts.ImportPaths...), ".")
		opts.Importer = gccgoimporter.GetImporter(paths, opts.InitMap)
	} else {
		var inst gccgoimporter.GccgoInstallation
		err := inst.InitFromDriver(opts.GccgoPath)
		if err != nil {
			return err
		}
		opts.Importer = inst.GetImporter(opts.ImportPaths, opts.InitMap)
	}
	return nil
}

func (compiler *compiler) compile(fset *token.FileSet, astFiles []*ast.File, importpath string) (m *Module, err error) {
	buildctx, err := llgobuild.ContextFromTriple(compiler.TargetTriple)
	if err != nil {
		return nil, err
	}

	if compiler.Importer == nil {
		err = compiler.MakeImporter()
		if err != nil {
			return nil, err
		}
	}

	impcfg := &loader.Config{
		Fset: fset,
		TypeChecker: types.Config{
			Packages: compiler.Packages,
			Import:   compiler.Importer,
			Sizes:    compiler.llvmtypes,
			DisableUnusedImportCheck: compiler.DisableUnusedImportCheck,
		},
		ImportFromBinary: true,
		Build:            &buildctx.Context,
		PackageCreated:   compiler.PackageCreated,
	}
	// If no import path is specified, then set the import
	// path to be the same as the package's name.
	if importpath == "" {
		importpath = astFiles[0].Name.String()
	}
	impcfg.CreateFromFiles(importpath, astFiles...)
	iprog, err := impcfg.Load()
	if err != nil {
		return nil, err
	}
	program := ssa.Create(iprog, ssa.BareInits)
	mainPkginfo := iprog.InitialPackages()[0]
	mainPkg := program.CreatePackage(mainPkginfo)

	// Create a Module, which contains the LLVM module.
	modulename := importpath
	compiler.module = &Module{Module: llvm.NewModule(modulename), Path: modulename, Package: mainPkg.Object}
	compiler.module.SetTarget(compiler.TargetTriple)
	compiler.module.SetDataLayout(compiler.dataLayout)

	// Create a new translation unit.
	unit := newUnit(compiler, mainPkg)

	// Create the runtime interface.
	compiler.runtime, err = newRuntimeInterface(compiler.module.Module, compiler.llvmtypes)
	if err != nil {
		return nil, err
	}

	mainPkg.Build()

	// Create a struct responsible for mapping static types to LLVM types,
	// and to runtime/dynamic type values.
	compiler.types = NewTypeMap(
		mainPkg,
		compiler.llvmtypes,
		compiler.module.Module,
		compiler.runtime,
		MethodResolver(unit),
	)

	if compiler.GenerateDebug {
		compiler.debug = debug.NewDIBuilder(
			types.Sizes(compiler.llvmtypes),
			compiler.module.Module,
			impcfg.Fset,
			compiler.DebugPrefixMaps,
		)
		defer compiler.debug.Destroy()
		defer compiler.debug.Finalize()
	}

	unit.translatePackage(mainPkg)
	compiler.processAnnotations(unit, mainPkginfo)

	if importpath == "main" {
		compiler.createInitMainFunction(mainPkg)
	} else {
		compiler.module.ExportData = compiler.buildExportData(mainPkg)
	}

	return compiler.module, nil
}

type byPriorityThenFunc []gccgoimporter.PackageInit

func (a byPriorityThenFunc) Len() int      { return len(a) }
func (a byPriorityThenFunc) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a byPriorityThenFunc) Less(i, j int) bool {
	switch {
	case a[i].Priority < a[j].Priority:
		return true
	case a[i].Priority > a[j].Priority:
		return false
	case a[i].InitFunc < a[j].InitFunc:
		return true
	default:
		return false
	}
}

func (c *compiler) buildPackageInitData(mainPkg *ssa.Package) gccgoimporter.InitData {
	var inits []gccgoimporter.PackageInit
	for _, imp := range mainPkg.Object.Imports() {
		inits = append(inits, c.InitMap[imp].Inits...)
	}
	sort.Sort(byPriorityThenFunc(inits))

	// Deduplicate init entries. We want to preserve the entry with the highest priority.
	// Normally a package's priorities will be consistent among its dependencies, but it is
	// possible for them to be different. For example, if a standard library test augments a
	// package which is a dependency of 'regexp' (which is imported by every test main package)
	// with additional dependencies, those dependencies may cause the package under test to
	// receive a higher priority than indicated by its init clause in 'regexp'.
	uniqinits := make([]gccgoimporter.PackageInit, len(inits))
	uniqinitpos := len(inits)
	uniqinitnames := make(map[string]struct{})
	for i, _ := range inits {
		init := inits[len(inits)-1-i]
		if _, ok := uniqinitnames[init.InitFunc]; !ok {
			uniqinitnames[init.InitFunc] = struct{}{}
			uniqinitpos--
			uniqinits[uniqinitpos] = init
		}
	}
	uniqinits = uniqinits[uniqinitpos:]

	ourprio := 1
	if len(uniqinits) != 0 {
		ourprio = uniqinits[len(uniqinits)-1].Priority + 1
	}

	if imp := mainPkg.Func("init"); imp != nil {
		impname := c.types.mc.mangleFunctionName(imp)
		uniqinits = append(uniqinits, gccgoimporter.PackageInit{mainPkg.Object.Name(), impname, ourprio})
	}

	return gccgoimporter.InitData{ourprio, uniqinits}
}

func (c *compiler) createInitMainFunction(mainPkg *ssa.Package) {
	int8ptr := llvm.PointerType(c.types.ctx.Int8Type(), 0)
	ftyp := llvm.FunctionType(llvm.VoidType(), []llvm.Type{int8ptr}, false)
	initMain := llvm.AddFunction(c.module.Module, "__go_init_main", ftyp)
	c.addCommonFunctionAttrs(initMain)
	entry := llvm.AddBasicBlock(initMain, "entry")

	builder := llvm.GlobalContext().NewBuilder()
	defer builder.Dispose()
	builder.SetInsertPointAtEnd(entry)

	args := []llvm.Value{llvm.Undef(int8ptr)}

	if !c.GccgoABI {
		initfn := c.module.Module.NamedFunction("main..import")
		if !initfn.IsNil() {
			builder.CreateCall(initfn, args, "")
		}
		builder.CreateRetVoid()
		return
	}

	initdata := c.buildPackageInitData(mainPkg)

	for _, init := range initdata.Inits {
		initfn := c.module.Module.NamedFunction(init.InitFunc)
		if initfn.IsNil() {
			initfn = llvm.AddFunction(c.module.Module, init.InitFunc, ftyp)
		}
		builder.CreateCall(initfn, args, "")
	}

	builder.CreateRetVoid()
}

func (c *compiler) buildExportData(mainPkg *ssa.Package) []byte {
	exportData := importer.ExportData(mainPkg.Object)
	b := bytes.NewBuffer(exportData)

	b.WriteString("v1;\n")
	if !c.GccgoABI {
		return b.Bytes()
	}

	initdata := c.buildPackageInitData(mainPkg)
	b.WriteString("priority ")
	b.WriteString(strconv.Itoa(initdata.Priority))
	b.WriteString(";\n")

	if len(initdata.Inits) != 0 {
		b.WriteString("init")
		for _, init := range initdata.Inits {
			b.WriteRune(' ')
			b.WriteString(init.Name)
			b.WriteRune(' ')
			b.WriteString(init.InitFunc)
			b.WriteRune(' ')
			b.WriteString(strconv.Itoa(init.Priority))
		}
		b.WriteString(";\n")
	}

	return b.Bytes()
}

// vim: set ft=go :
