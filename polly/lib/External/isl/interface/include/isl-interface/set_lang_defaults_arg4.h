#include <string>
#include <vector>

#include <clang/Lex/PreprocessorOptions.h>

/* Convert a clang::PreprocessorOptions to the fourth argument
 * of CompilerInvocation::setLangDefaults, which may be either
 * a clang::PreprocessorOptions itself or its Includes.
 */
struct setLangDefaultsArg4 {
	setLangDefaultsArg4(clang::PreprocessorOptions &PO) : PO(PO) {}
	operator clang::PreprocessorOptions &() { return PO; }
	operator std::vector<std::string> &() { return PO.Includes; }

	clang::PreprocessorOptions &PO;
};
