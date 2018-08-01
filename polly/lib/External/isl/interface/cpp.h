#include "generator.h"

using namespace std;
using namespace clang;

/* Generator for C++ bindings.
 *
 * "checked" is set if C++ bindings should be generated
 * that rely on the user to check for error conditions.
 */
class cpp_generator : public generator {
protected:
	bool checked;
public:
	cpp_generator(set<RecordDecl *> &exported_types,
		set<FunctionDecl *> exported_functions,
		set<FunctionDecl *> functions,
		bool checked = false) :
		generator(exported_types, exported_functions, functions),
		checked(checked) {}

	enum function_kind {
		function_kind_static_method,
		function_kind_member_method,
		function_kind_constructor,
	};

	virtual void generate();
private:
	void print_file(ostream &os, std::string filename);
	void print_forward_declarations(ostream &os);
	void print_declarations(ostream &os);
	void print_class(ostream &os, const isl_class &clazz);
	void print_class_forward_decl(ostream &os, const isl_class &clazz);
	void print_class_factory_decl(ostream &os, const isl_class &clazz,
		const std::string &prefix = std::string());
	void print_private_constructors_decl(ostream &os,
		const isl_class &clazz);
	void print_copy_assignment_decl(ostream &os, const isl_class &clazz);
	void print_public_constructors_decl(ostream &os,
		const isl_class &clazz);
	void print_constructors_decl(ostream &os, const isl_class &clazz);
	void print_destructor_decl(ostream &os, const isl_class &clazz);
	void print_ptr_decl(ostream &os, const isl_class &clazz);
	void print_get_ctx_decl(ostream &os);
	void print_methods_decl(ostream &os, const isl_class &clazz);
	void print_method_group_decl(ostream &os, const isl_class &clazz,
		const string &fullname, const set<FunctionDecl *> &methods);
	void print_method_decl(ostream &os, const isl_class &clazz,
		const string &fullname, FunctionDecl *method,
		function_kind kind);
	void print_implementations(ostream &os);
	void print_class_impl(ostream &os, const isl_class &clazz);
	void print_class_factory_impl(ostream &os, const isl_class &clazz);
	void print_private_constructors_impl(ostream &os,
		const isl_class &clazz);
	void print_public_constructors_impl(ostream &os,
		const isl_class &clazz);
	void print_constructors_impl(ostream &os, const isl_class &clazz);
	void print_copy_assignment_impl(ostream &os, const isl_class &clazz);
	void print_destructor_impl(ostream &os, const isl_class &clazz);
	void print_ptr_impl(ostream &os, const isl_class &clazz);
	void print_get_ctx_impl(ostream &os, const isl_class &clazz);
	void print_methods_impl(ostream &os, const isl_class &clazz);
	void print_method_group_impl(ostream &os, const isl_class &clazz,
		const string &fullname, const set<FunctionDecl *> &methods);
	void print_argument_validity_check(ostream &os, FunctionDecl *method,
		function_kind kind);
	void print_save_ctx(ostream &os, FunctionDecl *method,
		function_kind kind);
	void print_on_error_continue(ostream &os);
	void print_exceptional_execution_check(ostream &os,
		FunctionDecl *method);
	void print_method_impl(ostream &os, const isl_class &clazz,
		const string &fullname,	FunctionDecl *method,
		function_kind kind);
	void print_method_param_use(ostream &os, ParmVarDecl *param,
		bool load_from_this_ptr);
	void print_method_header(ostream &os, const isl_class &clazz,
		FunctionDecl *method, const string &fullname,
		bool is_declaration, function_kind kind);
	string generate_callback_args(QualType type, bool cpp);
	string generate_callback_type(QualType type);
	void print_wrapped_call_checked(std::ostream &os,
		const std::string &call);
	void print_wrapped_call(std::ostream &os, const std::string &call);
	void print_callback_local(ostream &os, ParmVarDecl *param);
	std::string rename_method(std::string name);
	string type2cpp(const isl_class &clazz);
	string type2cpp(QualType type);
	bool is_implicit_conversion(const isl_class &clazz, FunctionDecl *cons);
	bool is_subclass(QualType subclass_type, const isl_class &class_type);
	function_kind get_method_kind(const isl_class &clazz,
		FunctionDecl *method);
public:
	static string type2cpp(string type_string);
};
