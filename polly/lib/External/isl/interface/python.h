#include <set>
#include <clang/AST/Decl.h>
#include "generator.h"

using namespace std;
using namespace clang;

class python_generator : public generator {
private:
	set<string> done;

public:
	python_generator(SourceManager &SM, set<RecordDecl *> &exported_types,
		set<FunctionDecl *> exported_functions,
		set<FunctionDecl *> functions) :
		generator(SM, exported_types, exported_functions, functions) {}

	virtual void generate();

private:
	void print(const isl_class &clazz);
	void print_method_arguments(int first, int n_arg);
	void print_method_header(bool is_static, const string &name, int n_arg);
	void print_class_header(const isl_class &clazz, const string &name,
		const vector<string> &super);
	void print_type_check(int indent, const string &type, const char *fmt,
		int pos, bool upcast, const string &super,
		const string &name, int n);
	void print_type_checks(const string &cname, FunctionDecl *method,
		bool first_is_ctx, int n, const vector<string> &super);
	void print_copy(QualType type);
	void print_callback(ParmVarDecl *param, int arg);
	void print_arg_in_call(FunctionDecl *fd, const char *fmt, int arg,
		int skip);
	void print_argtypes(FunctionDecl *fd);
	void print_method_return(int indent, const isl_class &clazz,
		FunctionDecl *method, const char *fmt);
	void print_restype(FunctionDecl *fd);
	void print(map<string, isl_class> &classes, set<string> &done);
	void print_constructor(const isl_class &clazz, FunctionDecl *method);
	void print_special_constructors(const isl_class &clazz);
	void print_special_methods(const isl_class &clazz);
	void print_upcast_constructors(const isl_class &clazz);
	void print_new(const isl_class &clazz,
		const string &python_name);
	void print_representation(const isl_class &clazz,
		const string &python_name);
	void print_copy_callbacks(const isl_class &clazz);
	void print_method_type(FunctionDecl *fd);
	void print_method_types(const isl_class &clazz);
	void print_get_method(const isl_class &clazz, FunctionDecl *fd);
	void print_method(const isl_class &clazz, FunctionDecl *method,
		vector<string> super);
	void print_method_call(int indent, const isl_class &clazz,
		FunctionDecl *method, const char *fmt,
		int drop_ctx);
	void print_argument_checks(const isl_class &clazz, FunctionDecl *fd,
		int drop_ctx);
	void print_method_overload(const isl_class &clazz,
		FunctionDecl *method);
	void print_method(const isl_class &clazz, const string &fullname,
		const function_set &methods, vector<string> super);
	void print_set_enum(const isl_class &clazz, FunctionDecl *fd,
		int value, const string &name, const vector<string> &super);
	void print_set_enum(const isl_class &clazz, FunctionDecl *fd,
		const vector<string> &super);

};
