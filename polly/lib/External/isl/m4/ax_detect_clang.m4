AC_DEFUN([AX_DETECT_CLANG], [
AC_SUBST(CLANG_CXXFLAGS)
AC_SUBST(CLANG_LDFLAGS)
AC_SUBST(CLANG_RFLAG)
AC_SUBST(CLANG_LIBS)
AC_PROG_GREP
AC_PROG_SED
if test "x$with_clang_prefix" != "x"; then
	LLVM_CONFIG="$with_clang_prefix/bin/llvm-config"
fi
AC_PATH_PROG([LLVM_CONFIG], ["llvm-config"])
if test -z "$LLVM_CONFIG" || test ! -x "$LLVM_CONFIG"; then
	AC_MSG_ERROR([llvm-config not found])
fi
CLANG_CXXFLAGS=`$LLVM_CONFIG --cxxflags | \
	$SED -e 's/-Wcovered-switch-default//;s/-gsplit-dwarf//'`
CLANG_LDFLAGS=`$LLVM_CONFIG --ldflags`
# Construct a -R argument for libtool.
# This is needed in case some of the clang libraries are shared libraries.
CLANG_RFLAG=`echo "$CLANG_LDFLAGS" | $SED -e 's/-L/-R/g'`
targets=`$LLVM_CONFIG --targets-built`
components="$targets asmparser bitreader support mc"
$LLVM_CONFIG --components | $GREP option > /dev/null 2> /dev/null
if test $? -eq 0; then
	components="$components option"
fi
CLANG_LIBS=`$LLVM_CONFIG --libs $components`
systemlibs=`$LLVM_CONFIG --system-libs 2> /dev/null | tail -1`
if test $? -eq 0; then
	CLANG_LIBS="$CLANG_LIBS $systemlibs"
fi
CLANG_PREFIX=`$LLVM_CONFIG --prefix`
AC_DEFINE_UNQUOTED(CLANG_PREFIX, ["$CLANG_PREFIX"], [Clang installation prefix])

SAVE_CPPFLAGS="$CPPFLAGS"
CPPFLAGS="$CLANG_CXXFLAGS $CPPFLAGS"
AC_LANG_PUSH(C++)
AC_CHECK_HEADER([clang/Basic/SourceLocation.h], [],
	[AC_ERROR([clang header file not found])])
AC_EGREP_HEADER([getDefaultTargetTriple], [llvm/Support/Host.h], [],
	[AC_DEFINE([getDefaultTargetTriple], [getHostTriple],
	[Define to getHostTriple for older versions of clang])])
AC_EGREP_HEADER([getExpansionLineNumber], [clang/Basic/SourceLocation.h], [],
	[AC_DEFINE([getExpansionLineNumber], [getInstantiationLineNumber],
	[Define to getInstantiationLineNumber for older versions of clang])])
AC_EGREP_HEADER([getImmediateExpansionRange], [clang/Basic/SourceManager.h],
	[],
	[AC_DEFINE([getImmediateExpansionRange],
	[getImmediateInstantiationRange],
	[Define to getImmediateInstantiationRange for older versions of clang])]
)
AC_EGREP_HEADER([DiagnosticsEngine], [clang/Basic/Diagnostic.h], [],
	[AC_DEFINE([DiagnosticsEngine], [Diagnostic],
	[Define to Diagnostic for older versions of clang])])
AC_EGREP_HEADER([ArrayRef], [clang/Driver/Driver.h],
	[AC_DEFINE([USE_ARRAYREF], [],
		[Define if Driver::BuildCompilation takes ArrayRef])
	AC_EGREP_HEADER([ArrayRef.*CommandLineArgs],
		[clang/Frontend/CompilerInvocation.h],
		[AC_DEFINE([CREATE_FROM_ARGS_TAKES_ARRAYREF], [],
			[Define if CompilerInvocation::CreateFromArgs takes
			 ArrayRef])
		])
	])
AC_EGREP_HEADER([CXXIsProduction], [clang/Driver/Driver.h],
	[AC_DEFINE([HAVE_CXXISPRODUCTION], [],
		[Define if Driver constructor takes CXXIsProduction argument])])
AC_EGREP_HEADER([ IsProduction], [clang/Driver/Driver.h],
	[AC_DEFINE([HAVE_ISPRODUCTION], [],
		[Define if Driver constructor takes IsProduction argument])])
AC_TRY_COMPILE([#include <clang/Driver/Driver.h>], [
	using namespace clang;
	DiagnosticsEngine *Diags;
	new driver::Driver("", "", "", *Diags);
], [AC_DEFINE([DRIVER_CTOR_TAKES_DEFAULTIMAGENAME], [],
	      [Define if Driver constructor takes default image name])])
AC_EGREP_HEADER([void HandleTopLevelDecl\(], [clang/AST/ASTConsumer.h],
	[AC_DEFINE([HandleTopLevelDeclReturn], [void],
		   [Return type of HandleTopLevelDeclReturn])
	 AC_DEFINE([HandleTopLevelDeclContinue], [],
		   [Return type of HandleTopLevelDeclReturn])],
	[AC_DEFINE([HandleTopLevelDeclReturn], [bool],
		   [Return type of HandleTopLevelDeclReturn])
	 AC_DEFINE([HandleTopLevelDeclContinue], [true],
		   [Return type of HandleTopLevelDeclReturn])])
AC_CHECK_HEADER([clang/Basic/DiagnosticOptions.h],
	[AC_DEFINE([HAVE_BASIC_DIAGNOSTICOPTIONS_H], [],
		   [Define if clang/Basic/DiagnosticOptions.h exists])])
AC_CHECK_HEADER([clang/Lex/PreprocessorOptions.h],
	[AC_DEFINE([HAVE_LEX_PREPROCESSOROPTIONS_H], [],
		   [Define if clang/Lex/PreprocessorOptions.h exists])], [],
	[#include <clang/Basic/LLVM.h>])
AC_TRY_COMPILE([#include <clang/Basic/TargetInfo.h>], [
	using namespace clang;
	std::shared_ptr<TargetOptions> TO;
	DiagnosticsEngine *Diags;
	TargetInfo::CreateTargetInfo(*Diags, TO);
], [AC_DEFINE([CREATETARGETINFO_TAKES_SHARED_PTR], [],
	      [Define if TargetInfo::CreateTargetInfo takes shared_ptr])])
AC_TRY_COMPILE([#include <clang/Basic/TargetInfo.h>], [
	using namespace clang;
	TargetOptions *TO;
	DiagnosticsEngine *Diags;
	TargetInfo::CreateTargetInfo(*Diags, TO);
], [AC_DEFINE([CREATETARGETINFO_TAKES_POINTER], [],
	      [Define if TargetInfo::CreateTargetInfo takes pointer])])
AC_TRY_COMPILE([#include <clang/Frontend/CompilerInstance.h>], [
	using namespace clang;
	DiagnosticConsumer *client;
	CompilerInstance *Clang;
	Clang->createDiagnostics(client);
], [], [AC_DEFINE([CREATEDIAGNOSTICS_TAKES_ARG], [],
	[Define if CompilerInstance::createDiagnostics takes argc and argv])])
AC_TRY_COMPILE([#include <clang/Lex/HeaderSearchOptions.h>], [
	using namespace clang;
	HeaderSearchOptions HSO;
	HSO.AddPath("", frontend::Angled, false, false);
], [AC_DEFINE([ADDPATH_TAKES_4_ARGUMENTS], [],
	[Define if HeaderSearchOptions::AddPath takes 4 arguments])])
AC_EGREP_HEADER([getNumParams],
	[clang/AST/CanonicalType.h],
	[AC_DEFINE([getNumArgs], [getNumParams],
	    [Define to getNumParams for newer versions of clang])
	 AC_DEFINE([getArgType], [getParamType],
	    [Define to getParamType for newer versions of clang])])
AC_EGREP_HEADER([getReturnType],
	[clang/AST/CanonicalType.h], [],
	[AC_DEFINE([getReturnType], [getResultType],
	    [Define to getResultType for older versions of clang])])
AC_TRY_COMPILE([#include <clang/Frontend/CompilerInstance.h>], [
	using namespace clang;
	CompilerInstance *Clang;
	Clang->createPreprocessor(TU_Complete);
], [AC_DEFINE([CREATEPREPROCESSOR_TAKES_TUKIND], [],
[Define if CompilerInstance::createPreprocessor takes TranslationUnitKind])])
AC_EGREP_HEADER([setMainFileID], [clang/Basic/SourceManager.h],
	[AC_DEFINE([HAVE_SETMAINFILEID], [],
	[Define if SourceManager has a setMainFileID method])])
AC_CHECK_HEADER([llvm/ADT/OwningPtr.h],
	[AC_DEFINE([HAVE_ADT_OWNINGPTR_H], [],
		   [Define if llvm/ADT/OwningPtr.h exists])])
AC_EGREP_HEADER([initializeBuiltins],
	[clang/Basic/Builtins.h], [],
	[AC_DEFINE([initializeBuiltins], [InitializeBuiltins],
		[Define to InitializeBuiltins for older versions of clang])])
AC_EGREP_HEADER([IK_C], [clang/Frontend/FrontendOptions.h], [],
	[AC_CHECK_HEADER([clang/Basic/LangStandard.h],
		[IK_C=Language::C], [IK_C=InputKind::C])
	 AC_DEFINE_UNQUOTED([IK_C], [$IK_C],
	 [Define to Language::C or InputKind::C for newer versions of clang])
	])
AC_TRY_COMPILE([
	#include <clang/Basic/TargetOptions.h>
	#include <clang/Lex/PreprocessorOptions.h>
	#include <clang/Frontend/CompilerInstance.h>
], [
	using namespace clang;
	CompilerInstance *Clang;
	TargetOptions TO;
	llvm::Triple T(TO.Triple);
	PreprocessorOptions PO;
	CompilerInvocation::setLangDefaults(Clang->getLangOpts(), IK_C,
			T, PO, LangStandard::lang_unspecified);
], [AC_DEFINE([SETLANGDEFAULTS_TAKES_5_ARGUMENTS], [],
	[Define if CompilerInvocation::setLangDefaults takes 5 arguments])])
AC_TRY_COMPILE([
	#include <clang/Frontend/CompilerInstance.h>
	#include <clang/Frontend/CompilerInvocation.h>
], [
	using namespace clang;
	CompilerInvocation *invocation;
	CompilerInstance *Clang;
	Clang->setInvocation(std::make_shared<CompilerInvocation>(*invocation));
], [AC_DEFINE([SETINVOCATION_TAKES_SHARED_PTR], [],
	[Defined if CompilerInstance::setInvocation takes a shared_ptr])])
AC_CHECK_HEADER([llvm/Option/Arg.h],
	[AC_DEFINE([HAVE_LLVM_OPTION_ARG_H], [],
		   [Define if llvm/Option/Arg.h exists])])
AC_LANG_POP
CPPFLAGS="$SAVE_CPPFLAGS"

SAVE_LDFLAGS="$LDFLAGS"
LDFLAGS="$CLANG_LDFLAGS $LDFLAGS"
AC_SUBST(LIB_CLANG_EDIT)
AC_CHECK_LIB([clangEdit], [main], [LIB_CLANG_EDIT=-lclangEdit], [])
LDFLAGS="$SAVE_LDFLAGS"
])
