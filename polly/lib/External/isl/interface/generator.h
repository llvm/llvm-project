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
};

/* Base class for interface generators.
 */
class generator {
protected:
	map<string,isl_class> classes;
	map<string, FunctionDecl *> functions_by_name;

public:
	generator(set<RecordDecl *> &exported_types,
		set<FunctionDecl *> exported_functions,
		set<FunctionDecl *> functions);

	virtual void generate() = 0;
	virtual ~generator() {};

protected:
	void print_class_header(const isl_class &clazz, const string &name,
		const vector<string> &super);
	string drop_type_suffix(string name, FunctionDecl *method);
	void die(const char *msg) __attribute__((noreturn));
	void die(string msg) __attribute__((noreturn));
	vector<string> find_superclasses(RecordDecl *decl);
	bool is_overload(Decl *decl);
	bool is_constructor(Decl *decl);
	bool takes(Decl *decl);
	bool keeps(Decl *decl);
	bool gives(Decl *decl);
	isl_class *method2class(FunctionDecl *fd);
	bool is_isl_ctx(QualType type);
	bool first_arg_is_isl_ctx(FunctionDecl *fd);
	bool is_isl_type(QualType type);
	bool is_isl_bool(QualType type);
	bool is_isl_stat(QualType type);
	bool is_long(QualType type);
	bool is_callback(QualType type);
	bool is_string(QualType type);
	bool is_static(const isl_class &clazz, FunctionDecl *method);
	string extract_type(QualType type);
	FunctionDecl *find_by_name(const string &name, bool required);
};

#endif /* ISL_INTERFACE_GENERATOR_H */
