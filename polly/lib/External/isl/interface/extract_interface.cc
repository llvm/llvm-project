/*
 * Copyright 2011 Sven Verdoolaege. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 * 
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY SVEN VERDOOLAEGE ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SVEN VERDOOLAEGE OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation
 * are those of the authors and should not be interpreted as
 * representing official policies, either expressed or implied, of
 * Sven Verdoolaege.
 */ 

#include "isl_config.h"
#undef PACKAGE

#include <assert.h>
#include <iostream>
#include <stdlib.h>
#ifdef HAVE_ADT_OWNINGPTR_H
#include <llvm/ADT/OwningPtr.h>
#else
#include <memory>
#endif
#ifdef HAVE_LLVM_OPTION_ARG_H
#include <llvm/Option/Arg.h>
#endif
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/ManagedStatic.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/Basic/Builtins.h>
#include <clang/Basic/FileSystemOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/Version.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#ifdef HAVE_BASIC_DIAGNOSTICOPTIONS_H
#include <clang/Basic/DiagnosticOptions.h>
#else
#include <clang/Frontend/DiagnosticOptions.h>
#endif
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Lex/HeaderSearch.h>
#ifdef HAVE_LEX_PREPROCESSOROPTIONS_H
#include <clang/Lex/PreprocessorOptions.h>
#else
#include <clang/Frontend/PreprocessorOptions.h>
#endif
#include <clang/Lex/Preprocessor.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Sema/Sema.h>

#include "extract_interface.h"
#include "generator.h"
#include "python.h"
#include "cpp.h"
#include "cpp_conversion.h"

using namespace std;
using namespace clang;
using namespace clang::driver;
#ifdef HAVE_LLVM_OPTION_ARG_H
using namespace llvm::opt;
#endif

#ifdef HAVE_ADT_OWNINGPTR_H
#define unique_ptr	llvm::OwningPtr
#endif

static llvm::cl::opt<string> InputFilename(llvm::cl::Positional,
			llvm::cl::Required, llvm::cl::desc("<input file>"));
static llvm::cl::list<string> Includes("I",
			llvm::cl::desc("Header search path"),
			llvm::cl::value_desc("path"), llvm::cl::Prefix);

static llvm::cl::opt<string> OutputLanguage(llvm::cl::Required,
	llvm::cl::ValueRequired, "language",
	llvm::cl::desc("Bindings to generate"),
	llvm::cl::value_desc("name"));

static const char *ResourceDir =
	CLANG_PREFIX "/lib/clang/" CLANG_VERSION_STRING;

/* Does decl have an attribute of the following form?
 *
 *	__attribute__((annotate("name")))
 */
bool has_annotation(Decl *decl, const char *name)
{
	if (!decl->hasAttrs())
		return false;

	AttrVec attrs = decl->getAttrs();
	for (AttrVec::const_iterator i = attrs.begin() ; i != attrs.end(); ++i) {
		const AnnotateAttr *ann = dyn_cast<AnnotateAttr>(*i);
		if (!ann)
			continue;
		if (ann->getAnnotation().str() == name)
			return true;
	}

	return false;
}

/* Is decl marked as exported?
 */
static bool is_exported(Decl *decl)
{
	return has_annotation(decl, "isl_export");
}

/* Collect all types and functions that are annotated "isl_export"
 * in "exported_types" and "exported_function".  Collect all function
 * declarations in "functions".
 *
 * We currently only consider single declarations.
 */
struct MyASTConsumer : public ASTConsumer {
	set<RecordDecl *> exported_types;
	set<FunctionDecl *> exported_functions;
	set<FunctionDecl *> functions;

	virtual HandleTopLevelDeclReturn HandleTopLevelDecl(DeclGroupRef D) {
		Decl *decl;

		if (!D.isSingleDecl())
			return HandleTopLevelDeclContinue;
		decl = D.getSingleDecl();
		if (isa<FunctionDecl>(decl))
			functions.insert(cast<FunctionDecl>(decl));
		if (!is_exported(decl))
			return HandleTopLevelDeclContinue;
		switch (decl->getKind()) {
		case Decl::Record:
			exported_types.insert(cast<RecordDecl>(decl));
			break;
		case Decl::Function:
			exported_functions.insert(cast<FunctionDecl>(decl));
			break;
		default:
			break;
		}
		return HandleTopLevelDeclContinue;
	}
};

#ifdef USE_ARRAYREF

#ifdef HAVE_CXXISPRODUCTION
static Driver *construct_driver(const char *binary, DiagnosticsEngine &Diags)
{
	return new Driver(binary, llvm::sys::getDefaultTargetTriple(),
			    "", false, false, Diags);
}
#elif defined(HAVE_ISPRODUCTION)
static Driver *construct_driver(const char *binary, DiagnosticsEngine &Diags)
{
	return new Driver(binary, llvm::sys::getDefaultTargetTriple(),
			    "", false, Diags);
}
#elif defined(DRIVER_CTOR_TAKES_DEFAULTIMAGENAME)
static Driver *construct_driver(const char *binary, DiagnosticsEngine &Diags)
{
	return new Driver(binary, llvm::sys::getDefaultTargetTriple(),
			    "", Diags);
}
#else
static Driver *construct_driver(const char *binary, DiagnosticsEngine &Diags)
{
	return new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags);
}
#endif

namespace clang { namespace driver { class Job; } }

/* Clang changed its API from 3.5 to 3.6 and once more in 3.7.
 * We fix this with a simple overloaded function here.
 */
struct ClangAPI {
	static Job *command(Job *J) { return J; }
	static Job *command(Job &J) { return &J; }
	static Command *command(Command &C) { return &C; }
};

#ifdef CREATE_FROM_ARGS_TAKES_ARRAYREF

/* Call CompilerInvocation::CreateFromArgs with the right arguments.
 * In this case, an ArrayRef<const char *>.
 */
static void create_from_args(CompilerInvocation &invocation,
	const ArgStringList *args, DiagnosticsEngine &Diags)
{
	CompilerInvocation::CreateFromArgs(invocation, *args, Diags);
}

#else

/* Call CompilerInvocation::CreateFromArgs with the right arguments.
 * In this case, two "const char *" pointers.
 */
static void create_from_args(CompilerInvocation &invocation,
	const ArgStringList *args, DiagnosticsEngine &Diags)
{
	CompilerInvocation::CreateFromArgs(invocation, args->data() + 1,
						args->data() + args->size(),
						Diags);
}

#endif

#ifdef CLANG_SYSROOT
/* Set sysroot if required.
 *
 * If CLANG_SYSROOT is defined, then set it to this value.
 */
static void set_sysroot(ArgStringList &args)
{
	args.push_back("-isysroot");
	args.push_back(CLANG_SYSROOT);
}
#else
/* Set sysroot if required.
 *
 * If CLANG_SYSROOT is not defined, then it does not need to be set.
 */
static void set_sysroot(ArgStringList &args)
{
}
#endif

/* Create a CompilerInvocation object that stores the command line
 * arguments constructed by the driver.
 * The arguments are mainly useful for setting up the system include
 * paths on newer clangs and on some platforms.
 */
static CompilerInvocation *construct_invocation(const char *filename,
	DiagnosticsEngine &Diags)
{
	const char *binary = CLANG_PREFIX"/bin/clang";
	const unique_ptr<Driver> driver(construct_driver(binary, Diags));
	std::vector<const char *> Argv;
	Argv.push_back(binary);
	Argv.push_back(filename);
	const unique_ptr<Compilation> compilation(
		driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv)));
	JobList &Jobs = compilation->getJobs();

	Command *cmd = cast<Command>(ClangAPI::command(*Jobs.begin()));
	if (strcmp(cmd->getCreator().getName(), "clang"))
		return NULL;

	ArgStringList args = cmd->getArguments();
	set_sysroot(args);

	CompilerInvocation *invocation = new CompilerInvocation;
	create_from_args(*invocation, &args, Diags);
	return invocation;
}

#else

static CompilerInvocation *construct_invocation(const char *filename,
	DiagnosticsEngine &Diags)
{
	return NULL;
}

#endif

#ifdef HAVE_BASIC_DIAGNOSTICOPTIONS_H

static TextDiagnosticPrinter *construct_printer(void)
{
	return new TextDiagnosticPrinter(llvm::errs(), new DiagnosticOptions());
}

#else

static TextDiagnosticPrinter *construct_printer(void)
{
	DiagnosticOptions DO;
	return new TextDiagnosticPrinter(llvm::errs(), DO);
}

#endif

#ifdef CREATETARGETINFO_TAKES_SHARED_PTR

static TargetInfo *create_target_info(CompilerInstance *Clang,
	DiagnosticsEngine &Diags)
{
	shared_ptr<TargetOptions> TO = Clang->getInvocation().TargetOpts;
	TO->Triple = llvm::sys::getDefaultTargetTriple();
	return TargetInfo::CreateTargetInfo(Diags, TO);
}

#elif defined(CREATETARGETINFO_TAKES_POINTER)

static TargetInfo *create_target_info(CompilerInstance *Clang,
	DiagnosticsEngine &Diags)
{
	TargetOptions &TO = Clang->getTargetOpts();
	TO.Triple = llvm::sys::getDefaultTargetTriple();
	return TargetInfo::CreateTargetInfo(Diags, &TO);
}

#else

static TargetInfo *create_target_info(CompilerInstance *Clang,
	DiagnosticsEngine &Diags)
{
	TargetOptions &TO = Clang->getTargetOpts();
	TO.Triple = llvm::sys::getDefaultTargetTriple();
	return TargetInfo::CreateTargetInfo(Diags, TO);
}

#endif

#ifdef CREATEDIAGNOSTICS_TAKES_ARG

static void create_diagnostics(CompilerInstance *Clang)
{
	Clang->createDiagnostics(0, NULL, construct_printer());
}

#else

static void create_diagnostics(CompilerInstance *Clang)
{
	Clang->createDiagnostics(construct_printer());
}

#endif

#ifdef CREATEPREPROCESSOR_TAKES_TUKIND

static void create_preprocessor(CompilerInstance *Clang)
{
	Clang->createPreprocessor(TU_Complete);
}

#else

static void create_preprocessor(CompilerInstance *Clang)
{
	Clang->createPreprocessor();
}

#endif

#ifdef ADDPATH_TAKES_4_ARGUMENTS

/* Add "Path" to the header search options.
 *
 * Do not take into account sysroot, i.e., set ignoreSysRoot to true.
 */
void add_path(HeaderSearchOptions &HSO, string Path)
{
	HSO.AddPath(Path, frontend::Angled, false, true);
}

#else

/* Add "Path" to the header search options.
 *
 * Do not take into account sysroot, i.e., set IsSysRootRelative to false.
 */
void add_path(HeaderSearchOptions &HSO, string Path)
{
	HSO.AddPath(Path, frontend::Angled, true, false, false);
}

#endif

#ifdef HAVE_SETMAINFILEID

static void create_main_file_id(SourceManager &SM, const FileEntry *file)
{
	SM.setMainFileID(SM.createFileID(file, SourceLocation(),
					SrcMgr::C_User));
}

#else

static void create_main_file_id(SourceManager &SM, const FileEntry *file)
{
	SM.createMainFileID(file);
}

#endif

#ifdef SETLANGDEFAULTS_TAKES_5_ARGUMENTS

static void set_lang_defaults(CompilerInstance *Clang)
{
	PreprocessorOptions &PO = Clang->getPreprocessorOpts();
	TargetOptions &TO = Clang->getTargetOpts();
	llvm::Triple T(TO.Triple);
	CompilerInvocation::setLangDefaults(Clang->getLangOpts(), IK_C, T, PO,
					    LangStandard::lang_unspecified);
}

#else

static void set_lang_defaults(CompilerInstance *Clang)
{
	CompilerInvocation::setLangDefaults(Clang->getLangOpts(), IK_C,
					    LangStandard::lang_unspecified);
}

#endif

#ifdef SETINVOCATION_TAKES_SHARED_PTR

static void set_invocation(CompilerInstance *Clang,
	CompilerInvocation *invocation)
{
	Clang->setInvocation(std::make_shared<CompilerInvocation>(*invocation));
}

#else

static void set_invocation(CompilerInstance *Clang,
	CompilerInvocation *invocation)
{
	Clang->setInvocation(invocation);
}

#endif

/* Helper function for ignore_error that only gets enabled if T
 * (which is either const FileEntry * or llvm::ErrorOr<const FileEntry *>)
 * has getError method, i.e., if it is llvm::ErrorOr<const FileEntry *>.
 */
template <class T>
static const FileEntry *ignore_error_helper(const T obj, int,
	int[1][sizeof(obj.getError())])
{
	return *obj;
}

/* Helper function for ignore_error that is always enabled,
 * but that only gets selected if the variant above is not enabled,
 * i.e., if T is const FileEntry *.
 */
template <class T>
static const FileEntry *ignore_error_helper(const T obj, long, void *)
{
	return obj;
}

/* Given either a const FileEntry * or a llvm::ErrorOr<const FileEntry *>,
 * extract out the const FileEntry *.
 */
template <class T>
static const FileEntry *ignore_error(const T obj)
{
	return ignore_error_helper(obj, 0, NULL);
}

/* Return the FileEntry corresponding to the given file name
 * in the given compiler instances, ignoring any error.
 */
static const FileEntry *getFile(CompilerInstance *Clang, std::string Filename)
{
	return ignore_error(Clang->getFileManager().getFile(Filename));
}

/* Create an interface generator for the selected language and
 * then use it to generate the interface.
 */
static void generate(MyASTConsumer &consumer, SourceManager &SM)
{
	generator *gen;

	if (OutputLanguage.compare("python") == 0) {
		gen = new python_generator(SM, consumer.exported_types,
			consumer.exported_functions, consumer.functions);
	} else if (OutputLanguage.compare("cpp") == 0) {
		gen = new cpp_generator(SM, consumer.exported_types,
			consumer.exported_functions, consumer.functions);
	} else if (OutputLanguage.compare("cpp-checked") == 0) {
		gen = new cpp_generator(SM, consumer.exported_types,
			consumer.exported_functions, consumer.functions, true);
	} else if (OutputLanguage.compare("cpp-checked-conversion") == 0) {
		gen = new cpp_conversion_generator(SM, consumer.exported_types,
			consumer.exported_functions, consumer.functions);
	} else {
		cerr << "Language '" << OutputLanguage
		     << "' not recognized." << endl
		     << "Not generating bindings." << endl;
		exit(EXIT_FAILURE);
	}

	gen->generate();
}

int main(int argc, char *argv[])
{
	llvm::cl::ParseCommandLineOptions(argc, argv);

	CompilerInstance *Clang = new CompilerInstance();
	create_diagnostics(Clang);
	DiagnosticsEngine &Diags = Clang->getDiagnostics();
	Diags.setSuppressSystemWarnings(true);
	TargetInfo *target = create_target_info(Clang, Diags);
	Clang->setTarget(target);
	set_lang_defaults(Clang);
	CompilerInvocation *invocation =
		construct_invocation(InputFilename.c_str(), Diags);
	if (invocation)
		set_invocation(Clang, invocation);
	Clang->createFileManager();
	Clang->createSourceManager(Clang->getFileManager());
	HeaderSearchOptions &HSO = Clang->getHeaderSearchOpts();
	LangOptions &LO = Clang->getLangOpts();
	PreprocessorOptions &PO = Clang->getPreprocessorOpts();
	HSO.ResourceDir = ResourceDir;

	for (llvm::cl::list<string>::size_type i = 0; i < Includes.size(); ++i)
		add_path(HSO, Includes[i]);

	PO.addMacroDef("__isl_give=__attribute__((annotate(\"isl_give\")))");
	PO.addMacroDef("__isl_keep=__attribute__((annotate(\"isl_keep\")))");
	PO.addMacroDef("__isl_take=__attribute__((annotate(\"isl_take\")))");
	PO.addMacroDef("__isl_export=__attribute__((annotate(\"isl_export\")))");
	PO.addMacroDef("__isl_overload="
	    "__attribute__((annotate(\"isl_overload\"))) "
	    "__attribute__((annotate(\"isl_export\")))");
	PO.addMacroDef("__isl_constructor=__attribute__((annotate(\"isl_constructor\"))) __attribute__((annotate(\"isl_export\")))");
	PO.addMacroDef("__isl_subclass(super)=__attribute__((annotate(\"isl_subclass(\" #super \")\"))) __attribute__((annotate(\"isl_export\")))");

	create_preprocessor(Clang);
	Preprocessor &PP = Clang->getPreprocessor();

	PP.getBuiltinInfo().initializeBuiltins(PP.getIdentifierTable(), LO);

	const FileEntry *file = getFile(Clang, InputFilename);
	assert(file);
	create_main_file_id(Clang->getSourceManager(), file);

	Clang->createASTContext();
	MyASTConsumer consumer;
	Sema *sema = new Sema(PP, Clang->getASTContext(), consumer);

	Diags.getClient()->BeginSourceFile(LO, &PP);
	ParseAST(*sema);
	Diags.getClient()->EndSourceFile();

	generate(consumer, Clang->getSourceManager());

	delete sema;
	delete Clang;
	llvm::llvm_shutdown();

	if (Diags.hasErrorOccurred())
		return EXIT_FAILURE;
	return EXIT_SUCCESS;
}
