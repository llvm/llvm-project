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
	void print_method_header(bool is_static, const string &name, int n_arg);
	void print_class_header(const isl_class &clazz, const string &name,
		const vector<string> &super);
	void print_type_check(const string &type, int pos, bool upcast,
		const string &super, const string &name, int n);
	void print_copy(QualType type);
	void print_callback(ParmVarDecl *param, int arg);
	void print_arg_in_call(FunctionDecl *fd, int arg, int skip);
	void print_argtypes(FunctionDecl *fd);
	void print_method_return(FunctionDecl *method);
	void print_restype(FunctionDecl *fd);
	void print(map<string, isl_class> &classes, set<string> &done);
	void print_constructor(const isl_class &clazz, FunctionDecl *method);
	void print_representation(const isl_class &clazz,
		const string &python_name);
	void print_method_type(FunctionDecl *fd);
	void print_method_types(const isl_class &clazz);
	void print_method(const isl_class &clazz, FunctionDecl *method,
		vector<string> super);
	void print_method_overload(const isl_class &clazz,
		FunctionDecl *method);
	void print_method(const isl_class &clazz, const string &fullname,
		const set<FunctionDecl *> &methods, vector<string> super);

};
