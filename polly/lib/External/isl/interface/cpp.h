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
	cpp_generator(SourceManager &SM, set<RecordDecl *> &exported_types,
		set<FunctionDecl *> exported_functions,
		set<FunctionDecl *> functions,
		bool checked = false) :
		generator(SM, exported_types, exported_functions, functions),
		checked(checked) {}

	enum function_kind {
		function_kind_static_method,
		function_kind_member_method,
		function_kind_constructor,
	};
	enum method_part {
		decl,
		impl,
	};

	virtual void generate();
private:
	void print_forward_declarations(ostream &os);
	void print_declarations(ostream &os);
	void print_class(ostream &os, const isl_class &clazz);
	void print_subclass_type(ostream &os, const isl_class &clazz);
	void print_class_forward_decl(ostream &os, const isl_class &clazz);
	void print_class_factory_decl(ostream &os, const isl_class &clazz,
		const std::string &prefix = std::string());
	void print_protected_constructors_decl(ostream &os,
		const isl_class &clazz);
	void print_copy_assignment_decl(ostream &os, const isl_class &clazz);
	void print_public_constructors_decl(ostream &os,
		const isl_class &clazz);
	void print_constructors_decl(ostream &os, const isl_class &clazz);
	void print_destructor_decl(ostream &os, const isl_class &clazz);
	void print_ptr_decl(ostream &os, const isl_class &clazz);
	void print_isa_type_template(ostream &os, int indent,
		const isl_class &super);
	void print_downcast_decl(ostream &os, const isl_class &clazz);
	void print_ctx_decl(ostream &os);
	void print_persistent_callback_prototype(ostream &os,
		const isl_class &clazz, FunctionDecl *method,
		bool is_declaration);
	void print_persistent_callback_setter_prototype(ostream &os,
		const isl_class &clazz, FunctionDecl *method,
		bool is_declaration);
	void print_persistent_callback_data(ostream &os, const isl_class &clazz,
		FunctionDecl *method);
	void print_persistent_callbacks_decl(ostream &os,
		const isl_class &clazz);
	void print_methods_decl(ostream &os, const isl_class &clazz);
	bool next_variant(FunctionDecl *fd, std::vector<bool> &convert);
	template <enum method_part>
	void print_method_variants(ostream &os, const isl_class &clazz,
		FunctionDecl *fd);
	void print_method_group_decl(ostream &os, const isl_class &clazz,
		const function_set &methods);
	void print_named_method_decl(ostream &os, const isl_class &clazz,
		FunctionDecl *fd, const string &name, function_kind kind,
		const std::vector<bool> &convert = {});
	template <enum method_part>
	void print_method(ostream &os, const isl_class &clazz,
		FunctionDecl *method, function_kind kind);
	template <enum method_part>
	void print_method(ostream &os, const isl_class &clazz,
		FunctionDecl *method, function_kind kind,
		const std::vector<bool> &convert);
	void print_set_enum_decl(ostream &os, const isl_class &clazz,
		FunctionDecl *fd, const string &name);
	void print_set_enums_decl(ostream &os, const isl_class &clazz,
		FunctionDecl *fd);
	void print_set_enums_decl(ostream &os, const isl_class &clazz);
	void print_implementations(ostream &os);
	void print_class_impl(ostream &os, const isl_class &clazz);
	void print_check_ptr(ostream &os, const char *ptr);
	void print_check_ptr_start(ostream &os, const isl_class &clazz,
		const char *ptr);
	void print_check_ptr_end(ostream &os, const char *ptr);
	void print_class_factory_impl(ostream &os, const isl_class &clazz);
	void print_protected_constructors_impl(ostream &os,
		const isl_class &clazz);
	void print_public_constructors_impl(ostream &os,
		const isl_class &clazz);
	void print_constructors_impl(ostream &os, const isl_class &clazz);
	void print_copy_assignment_impl(ostream &os, const isl_class &clazz);
	void print_destructor_impl(ostream &os, const isl_class &clazz);
	void print_check_no_persistent_callback(ostream &os,
		const isl_class &clazz, FunctionDecl *fd);
	void print_ptr_impl(ostream &os, const isl_class &clazz);
	void print_downcast_impl(ostream &os, const isl_class &clazz);
	void print_ctx_impl(ostream &os, const isl_class &clazz);
	void print_persistent_callbacks_impl(ostream &os,
		const isl_class &clazz);
	void print_methods_impl(ostream &os, const isl_class &clazz);
	void print_method_group_impl(ostream &os, const isl_class &clazz,
		const function_set &methods);
	void print_argument_validity_check(ostream &os, FunctionDecl *method,
		function_kind kind);
	void print_save_ctx(ostream &os, FunctionDecl *method,
		function_kind kind);
	void print_on_error_continue(ostream &os);
	void print_exceptional_execution_check(ostream &os,
		const isl_class &clazz, FunctionDecl *method,
		function_kind kind);
	void print_set_persistent_callback(ostream &os, const isl_class &clazz,
		FunctionDecl *method, function_kind kind);
	void print_method_return(ostream &os, const isl_class &clazz,
		FunctionDecl *method);
	void print_set_enum_impl(ostream &os, const isl_class &clazz,
		FunctionDecl *fd, const string &enum_name,
		const string &method_name);
	void print_set_enums_impl(ostream &os, const isl_class &clazz,
		FunctionDecl *fd);
	void print_set_enums_impl(ostream &os, const isl_class &clazz);
	template <enum method_part>
	void print_get_method(ostream &os, const isl_class &clazz,
		FunctionDecl *fd);
	void print_invalid(ostream &os, int indent, const char *msg,
		const char *checked_code);
	void print_stream_insertion(ostream &os, const isl_class &clazz);
	void print_method_param_use(ostream &os, ParmVarDecl *param,
		bool load_from_this_ptr);
	std::string get_return_type(const isl_class &clazz, FunctionDecl *fd);
	ParmVarDecl *get_param(FunctionDecl *fd, int pos,
		const std::vector<bool> &convert);
	void print_method_header(ostream &os, const isl_class &clazz,
		FunctionDecl *method, const string &cname, int num_params,
		bool is_declaration, function_kind kind,
		const std::vector<bool> &convert = {});
	void print_named_method_header(ostream &os, const isl_class &clazz,
		FunctionDecl *method, string name, bool is_declaration,
		function_kind kind, const std::vector<bool> &convert = {});
	void print_method_header(ostream &os, const isl_class &clazz,
		FunctionDecl *method, bool is_declaration, function_kind kind);
	string generate_callback_args(QualType type, bool cpp);
	string generate_callback_type(QualType type);
	void print_wrapped_call_checked(std::ostream &os, int indent,
		const std::string &call);
	void print_wrapped_call(std::ostream &os, int indent,
		const std::string &call, QualType rtype);
	void print_callback_data_decl(ostream &os, ParmVarDecl *param,
		const string &name);
	void print_callback_body(ostream &os, int indent, ParmVarDecl *param,
		const string &name);
	void print_callback_local(ostream &os, ParmVarDecl *param);
	std::string rename_method(std::string name);
	string isl_bool2cpp();
	string isl_namespace();
	string type2cpp(QualType type);
	bool is_implicit_conversion(const isl_class &clazz, FunctionDecl *cons);
	bool is_subclass(QualType subclass_type, const isl_class &class_type);
	function_kind get_method_kind(const isl_class &clazz,
		FunctionDecl *method);
public:
	static string type2cpp(const isl_class &clazz);
	static string type2cpp(string type_string);
};
