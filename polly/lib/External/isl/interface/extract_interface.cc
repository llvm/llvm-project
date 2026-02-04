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
#include <type_traits>
#include <memory>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ManagedStatic.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileSystemOptions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Sema/Sema.h>

#include "isl-interface/clang_wrap.h"

#include "extract_interface.h"
#include "generator.h"
#include "python.h"
#include "plain_cpp.h"
#include "cpp_conversion.h"
#include "template_cpp.h"

using namespace std;
using namespace clang;
using namespace clang::driver;
#ifdef HAVE_LLVM_OPTION_ARG_H
using namespace llvm::opt;
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

/* A class specializing the Wrap helper class for
 * extracting the isl interface.
 */
struct Extractor : public isl::clang::Wrap {
	virtual TextDiagnosticPrinter *construct_printer() override;
	virtual void suppress_errors(DiagnosticsEngine &Diags) override;
	virtual void add_paths(HeaderSearchOptions &HSO) override;
	virtual void add_macros(PreprocessorOptions &PO) override;
	virtual void handle_error() override;
	virtual bool handle(CompilerInstance *Clang) override;
};

/* Construct a TextDiagnosticPrinter.
 */
TextDiagnosticPrinter *Extractor::construct_printer(void)
{
	return new TextDiagnosticPrinter(llvm::errs(), getDiagnosticOptions());
}

/* Suppress any errors, if needed.
 */
void Extractor::suppress_errors(DiagnosticsEngine &Diags)
{
}

/* Add required search paths to "HSO".
 */
void Extractor::add_paths(HeaderSearchOptions &HSO)
{
	for (llvm::cl::list<string>::size_type i = 0; i < Includes.size(); ++i)
		isl::clang::add_path(HSO, Includes[i]);
}

/* Add required macro definitions to "PO".
 */
void Extractor::add_macros(PreprocessorOptions &PO)
{
	PO.addMacroDef("__isl_give=__attribute__((annotate(\"isl_give\")))");
	PO.addMacroDef("__isl_keep=__attribute__((annotate(\"isl_keep\")))");
	PO.addMacroDef("__isl_take=__attribute__((annotate(\"isl_take\")))");
	PO.addMacroDef("__isl_export=__attribute__((annotate(\"isl_export\")))");
	PO.addMacroDef("__isl_overload="
	    "__attribute__((annotate(\"isl_overload\"))) "
	    "__attribute__((annotate(\"isl_export\")))");
	PO.addMacroDef("__isl_constructor=__attribute__((annotate(\"isl_constructor\"))) __attribute__((annotate(\"isl_export\")))");
	PO.addMacroDef("__isl_subclass(super)=__attribute__((annotate(\"isl_subclass(\" #super \")\"))) __attribute__((annotate(\"isl_export\")))");
}

/* Handle an error opening the file.
 */
void Extractor::handle_error()
{
	assert(false);
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
		gen = new plain_cpp_generator(SM, consumer.exported_types,
			consumer.exported_functions, consumer.functions);
	} else if (OutputLanguage.compare("cpp-checked") == 0) {
		gen = new plain_cpp_generator(SM, consumer.exported_types,
			consumer.exported_functions, consumer.functions, true);
	} else if (OutputLanguage.compare("cpp-checked-conversion") == 0) {
		gen = new cpp_conversion_generator(SM, consumer.exported_types,
			consumer.exported_functions, consumer.functions);
	} else if (OutputLanguage.compare("template-cpp") == 0) {
		gen = new template_cpp_generator(SM, consumer.exported_types,
			consumer.exported_functions, consumer.functions);
	} else {
		cerr << "Language '" << OutputLanguage
		     << "' not recognized." << endl
		     << "Not generating bindings." << endl;
		exit(EXIT_FAILURE);
	}

	gen->generate();
}

/* Parse the current source file, returning true if no error was encountered.
 */
bool Extractor::handle(CompilerInstance *Clang)
{
	Preprocessor &PP = Clang->getPreprocessor();
	MyASTConsumer consumer;
	Sema *sema = new Sema(PP, Clang->getASTContext(), consumer);

	DiagnosticsEngine &Diags = Clang->getDiagnostics();
	Diags.getClient()->BeginSourceFile(Clang->getLangOpts(), &PP);
	ParseAST(*sema);
	Diags.getClient()->EndSourceFile();

	generate(consumer, Clang->getSourceManager());

	delete sema;

	return !Diags.hasErrorOccurred();
}

int main(int argc, char *argv[])
{
	llvm::cl::ParseCommandLineOptions(argc, argv);

	Extractor extractor;
	bool ok = extractor.invoke(InputFilename.c_str());

	llvm::llvm_shutdown();

	if (!ok)
		return EXIT_FAILURE;
	return EXIT_SUCCESS;
}
