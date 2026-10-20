#include <isl-interface/config.h>

#include <memory>

#ifdef HAVE_LLVM_OPTION_ARG_H
#include <llvm/Option/Arg.h>
#endif
#ifdef HAVE_TARGETPARSER_HOST_H
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#include <clang/AST/ASTContext.h>
#include <clang/Basic/Builtins.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/Version.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Lex/Preprocessor.h>

namespace clang { namespace driver { class Job; } }

namespace isl {
namespace clang {

using namespace ::clang;
using namespace ::clang::driver;
#ifdef HAVE_LLVM_OPTION_ARG_H
using namespace llvm::opt;
#endif

#ifndef ISL_CLANG_RESOURCE_DIR
#define ISL_CLANG_RESOURCE_DIR \
	ISL_CLANG_PREFIX "/lib/clang/" CLANG_VERSION_STRING
#endif
static const char *ResourceDir = ISL_CLANG_RESOURCE_DIR;

static Driver *construct_driver(const char *binary, DiagnosticsEngine &Diags)
{
	return new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags);
}

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

#ifdef ISL_CLANG_SYSROOT
/* Set sysroot if required.
 *
 * If ISL_CLANG_SYSROOT is defined, then set it to this value.
 */
static void set_sysroot(ArgStringList &args)
{
	args.push_back("-isysroot");
	args.push_back(ISL_CLANG_SYSROOT);
}
#else
/* Set sysroot if required.
 *
 * If ISL_CLANG_SYSROOT is not defined, then it does not need to be set.
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
static void construct_invocation(CompilerInstance *Clang,
	const char *filename, DiagnosticsEngine &Diags)
{
	const char *binary = ISL_CLANG_PREFIX"/bin/clang";
	const std::unique_ptr<Driver> driver(construct_driver(binary, Diags));
	std::vector<const char *> Argv;
	Argv.push_back(binary);
	Argv.push_back(filename);
	const std::unique_ptr<Compilation> compilation(
		driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv)));
	JobList &Jobs = compilation->getJobs();

	Command *cmd = cast<Command>(ClangAPI::command(*Jobs.begin()));
	if (strcmp(cmd->getCreator().getName(), "clang"))
		return;

	ArgStringList args = cmd->getArguments();
	set_sysroot(args);

	create_from_args(Clang->getInvocation(), &args, Diags);
}

#ifdef CREATETARGETINFO_TAKES_SHARED_PTR

static TargetInfo *create_target_info(CompilerInstance *Clang,
	DiagnosticsEngine &Diags)
{
	std::shared_ptr<TargetOptions> TO = Clang->getInvocation().TargetOpts;
	TO->Triple = llvm::sys::getDefaultTargetTriple();
	return TargetInfo::CreateTargetInfo(Diags, TO);
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

#ifdef CREATEDIAGNOSTICS_TAKES_VFS

static void create_diagnostics(CompilerInstance *Clang)
{
	Clang->createDiagnostics(*llvm::vfs::getRealFileSystem());
}

#else

static void create_diagnostics(CompilerInstance *Clang)
{
	Clang->createDiagnostics();
}

#endif

static void create_preprocessor(CompilerInstance *Clang)
{
	Clang->createPreprocessor(TU_Complete);
}

/* Add "Path" to the header search options.
 *
 * Do not take into account sysroot, i.e., set ignoreSysRoot to true.
 */
static void add_path(HeaderSearchOptions &HSO, std::string Path)
{
	HSO.AddPath(Path, frontend::Angled, false, true);
}

template <typename T>
static void create_main_file_id(SourceManager &SM, const T &file)
{
	SM.setMainFileID(SM.createFileID(file, SourceLocation(),
					SrcMgr::C_User));
}

#ifdef SETLANGDEFAULTS_TAKES_5_ARGUMENTS

#include "isl-interface/set_lang_defaults_arg4.h"

static void set_lang_defaults(CompilerInstance *Clang)
{
	PreprocessorOptions &PO = Clang->getPreprocessorOpts();
	TargetOptions &TO = Clang->getTargetOpts();
	llvm::Triple T(TO.Triple);
	SETLANGDEFAULTS::setLangDefaults(Clang->getLangOpts(), IK_C, T,
					    setLangDefaultsArg4(PO),
					    LangStandard::lang_unspecified);
}

#else

static void set_lang_defaults(CompilerInstance *Clang)
{
	CompilerInvocation::setLangDefaults(Clang->getLangOpts(), IK_C,
					    LangStandard::lang_unspecified);
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

/* Define a template class "CLASS" with value true
 * if "EXPR" is valid for its template argument (available as "U" in "EXPR").
 */
#define ISL_VALID_EXPR_FOR_TEMPLATE_ARG(CLASS, EXPR)			\
template <typename T>							\
struct CLASS {								\
private:								\
    template <typename U>						\
    static auto test(int) -> decltype(EXPR, std::true_type());		\
									\
    template <typename>							\
    static std::false_type test(...);					\
									\
public:									\
    using type = decltype(test<T>(0));					\
    static constexpr bool value = type::value;				\
};

/* A template class with value true if the template argument
 * has a getFileRef method.
 */
ISL_VALID_EXPR_FOR_TEMPLATE_ARG(HasGetFileRef,
	std::declval<U>().getFileRef(std::declval<const std::string &>()))

/* Return a wrapper around the FileEntryRef/FileEntry
 * corresponding to the given file name.  The wrapper evaluates
 * to false if an error occurs.
 *
 * If T (= FileManager) has a getFileRef method, then call that and
 * return an llvm::Expected<clang::FileEntryRef>.
 * Otherwise, call getFile and return a FileEntry (pointer) embedded
 * in an llvm::ErrorOr.
 */
template <typename T,
	typename std::enable_if<HasGetFileRef<T>::value, bool>::type = true>
static auto getFile(T& obj, const std::string &filename)
	-> decltype(obj.getFileRef(filename))
{
	return obj.getFileRef(filename);
}
template <typename T,
	typename std::enable_if<!HasGetFileRef<T>::value, bool>::type = true>
static llvm::ErrorOr<const FileEntry *> getFile(T& obj,
	const std::string &filename)
{
	const FileEntry *file = ignore_error(obj.getFile(filename));
	if (!file)
	    return std::error_code();
	return file;
}

/* A template class with value true if the constructor of the template argument
 * takes a reference to a DiagnosticOptions object.
 */
ISL_VALID_EXPR_FOR_TEMPLATE_ARG(TakesDiagnosticOptionsRef,
	new U(llvm::errs(), std::declval<DiagnosticOptions &>()))

/* Return the type of the DiagnosticOptions argument of the constructor of "T".
 *
 * If TakesDiagnosticOptionsRef holds, then this is a reference
 * to a DiagnosticOptions object.
 * Otherwise, it is a pointer to such an object.
 */
template <typename T,
	typename std::enable_if<TakesDiagnosticOptionsRef<T>::value,
				bool>::type = true>
static DiagnosticOptions &diag_opts_type();
template <typename T,
	typename std::enable_if<!TakesDiagnosticOptionsRef<T>::value,
				bool>::type = true>
static DiagnosticOptions *diag_opts_type();

/* A helper class handling the invocation of clang on a file and
 * allowing a derived class to perform the actual parsing
 * in an overridden handle() method.
 * Other virtual methods allow different kinds of configuration.
 */
struct Wrap {
	/* The type of the DiagnosticOptions argument
	 * of the TextDiagnosticPrinter constructor.
	 */
	using DiagOptsType = decltype(diag_opts_type<TextDiagnosticPrinter>());
	/* A local DiagnosticOptions object that is used
	 * if TextDiagnosticPrinter takes a reference
	 * to a DiagnosticOptions object.
	 */
	DiagnosticOptions DiagOpts;

	/* Return a valid DiagnosticOptions argument
	 * for the constructor of "T".
	 * If "T" takes a reference, then
	 * this is a reference to the DiagOpts field.
	 * Otherwise, it is a pointer to a new object.
	 */
	template <typename T,
		typename std::enable_if<TakesDiagnosticOptionsRef<T>::value,
					bool>::type = true>
	DiagnosticOptions &getDiagOpts() {
		return DiagOpts;
	}
	template <typename T,
		typename std::enable_if<!TakesDiagnosticOptionsRef<T>::value,
					bool>::type = true>
	DiagnosticOptions *getDiagOpts() {
		return new DiagnosticOptions();
	}

	/* Return a valid DiagnosticOptions argument for
	 * the TextDiagnosticPrinter constructor.
	 * If TextDiagnosticPrinter takes a reference, then
	 * this is a reference to the DiagOpts field.
	 * Otherwise, it is a pointer to a new object.
	 */
	DiagOptsType getDiagnosticOptions() {
		return getDiagOpts<TextDiagnosticPrinter>();
	}

	/* Construct a TextDiagnosticPrinter. */
	virtual TextDiagnosticPrinter *construct_printer() = 0;
	/* Suppress any errors, if needed. */
	virtual void suppress_errors(DiagnosticsEngine &Diags) = 0;
	/* Add required search paths to "HSO". */
	virtual void add_paths(HeaderSearchOptions &HSO) = 0;
	/* Add required macro definitions to "PO". */
	virtual void add_macros(PreprocessorOptions &PO) = 0;
	/* Handle an error opening the file. */
	virtual void handle_error() = 0;
	/* Parse the file, returning true if no error was encountered. */
	virtual bool handle(CompilerInstance *Clang) = 0;

	/* Invoke clang on "filename", passing control to the handle() method
	 * for parsing.
	 */
	bool invoke(const char *filename) {
		CompilerInstance *Clang = new CompilerInstance();
		create_diagnostics(Clang);
		DiagnosticsEngine &Diags = Clang->getDiagnostics();
		Diags.setSuppressSystemWarnings(true);
		suppress_errors(Diags);
		TargetInfo *target = create_target_info(Clang, Diags);
		Clang->setTarget(target);
		set_lang_defaults(Clang);
		construct_invocation(Clang, filename, Diags);
		Diags.setClient(construct_printer());
		Clang->createFileManager();
		Clang->createSourceManager(Clang->getFileManager());
		HeaderSearchOptions &HSO = Clang->getHeaderSearchOpts();
		LangOptions &LO = Clang->getLangOpts();
		PreprocessorOptions &PO = Clang->getPreprocessorOpts();
		HSO.ResourceDir = ResourceDir;
		HSO.AddPath(ISL_CLANG_RESOURCE_DIR "/include",
			clang::frontend::System, false, false);

		add_paths(HSO);
		add_macros(PO);

		create_preprocessor(Clang);
		Preprocessor &PP = Clang->getPreprocessor();

		PP.getBuiltinInfo().initializeBuiltins(PP.getIdentifierTable(),
							LO);
		auto file = getFile(Clang->getFileManager(), filename);
		if (!file) {
			handle_error();
			delete Clang;
			return false;
		}
		create_main_file_id(Clang->getSourceManager(), *file);

		Clang->createASTContext();
		bool ok = handle(Clang);
		delete Clang;

		return ok;
	}
};

}
}
