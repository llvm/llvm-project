#ifndef ISL_INTERFACE_GENERATOR_H
#define ISL_INTERFACE_GENERATOR_H

#include <map>
#include <set>
#include <string>

#include <clang/AST/Decl.h>

using namespace std;
using namespace clang;

/* isl_class collects all constructors and methods for an isl "class".
 * "name" is the name of the class.
 * "type" is the declaration that introduces the type.
 * "methods" contains the set of methods, grouped by method name.
 * "fn_to_str" is a reference to the *_to_str method of this class, if any.
 * "fn_copy" is a reference to the *_copy method of this class, if any.
 * "fn_free" is a reference to the *_free method of this class, if any.
 */
struct isl_class {
	string name;
	RecordDecl *type;
	set<FunctionDecl *> constructors;
	map<string, set<FunctionDecl *> > methods;
	FunctionDecl *fn_to_str;
	FunctionDecl *fn_copy;
	FunctionDecl *fn_free;

	/* Return name of "fd" without type suffix, if any. */
	static string name_without_type_suffix(FunctionDecl *fd);
	/* Extract the method name corresponding to "fd". */
	string method_name(FunctionDecl *fd) const {
		string m_name = name_without_type_suffix(fd);
		return m_name.substr(name.length() + 1);
	}
};

/* Base class for interface generators.
 */
class generator {
protected:
	SourceManager &SM;
	map<string,isl_class> classes;
	map<string, FunctionDecl *> functions_by_name;

public:
	generator(SourceManager &SM, set<RecordDecl *> &exported_types,
		set<FunctionDecl *> exported_functions,
		set<FunctionDecl *> functions);

	virtual void generate() = 0;
	virtual ~generator() {};

protected:
	isl_class *method2class(FunctionDecl *fd);
	bool callback_takes_argument(ParmVarDecl *param, int pos);
	FunctionDecl *find_by_name(const string &name, bool required);
public:
	static void die(const char *msg) __attribute__((noreturn));
	static void die(string msg) __attribute__((noreturn));
	static vector<string> find_superclasses(RecordDecl *decl);
	static bool is_overload(Decl *decl);
	static bool is_constructor(Decl *decl);
	static bool takes(Decl *decl);
	static bool keeps(Decl *decl);
	static bool gives(Decl *decl);
	static bool is_isl_ctx(QualType type);
	static bool first_arg_is_isl_ctx(FunctionDecl *fd);
	static bool is_isl_type(QualType type);
	static bool is_isl_bool(QualType type);
	static bool is_isl_stat(QualType type);
	static bool is_long(QualType type);
	static bool is_callback(QualType type);
	static bool is_string(QualType type);
	static bool is_static(const isl_class &clazz, FunctionDecl *method);
	static string extract_type(QualType type);
	static const FunctionProtoType *extract_prototype(QualType type);
};

#endif /* ISL_INTERFACE_GENERATOR_H */
