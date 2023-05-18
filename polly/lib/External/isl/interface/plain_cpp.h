#ifndef ISL_INTERFACE_PLAIN_CPP_H
#define ISL_INTERFACE_PLAIN_CPP_H

#include <functional>
#include <memory>

#include "cpp.h"
#include "generator.h"

using namespace std;
using namespace clang;

/* A type printer for converting argument and return types
 * to string representations of the corresponding types
 * in the checked C++ interface.
 */
struct checked_cpp_type_printer : public cpp_type_printer {
	virtual std::string isl_bool() const override;
	virtual std::string isl_stat() const override;
	virtual std::string isl_size() const override;
	virtual std::string isl_namespace() const override;
};

/* Generator for plain C++ bindings.
 *
 * "checked" is set if C++ bindings should be generated
 * that rely on the user to check for error conditions.
 */
class plain_cpp_generator : public cpp_generator {
	struct plain_printer;
	struct decl_printer;
	struct impl_printer;
protected:
	bool checked;
public:
	plain_cpp_generator(SourceManager &SM,
		set<RecordDecl *> &exported_types,
		set<FunctionDecl *> exported_functions,
		set<FunctionDecl *> functions,
		bool checked = false);

	virtual void generate();
private:
	void print_forward_declarations(ostream &os);
	void print_declarations(ostream &os);
	void print_class(ostream &os, const isl_class &clazz);
	void print_class_forward_decl(ostream &os, const isl_class &clazz);
	void print_implementations(ostream &os);
	void print_class_impl(ostream &os, const isl_class &clazz);
	void print_check_no_persistent_callback(ostream &os,
		const isl_class &clazz, FunctionDecl *fd);
	void print_invalid(ostream &os, int indent, const char *msg,
		const char *checked_code);
	void print_method_param_use(ostream &os, ParmVarDecl *param,
		bool load_from_this_ptr);
	std::unique_ptr<cpp_type_printer> type_printer();
	std::string get_return_type(const Method &method);
	string generate_callback_args(QualType type, bool cpp);
	string generate_callback_type(QualType type);
	string isl_bool2cpp();
	string isl_namespace();
	string param2cpp(QualType type);
};

/* A helper class for printing method declarations and definitions
 * of a class for the plain C++ interface.
 *
 * "generator" is the C++ interface generator printing the classes.
 */
struct plain_cpp_generator::plain_printer : public cpp_generator::class_printer {
	plain_cpp_generator &generator;

	plain_printer(std::ostream &os, const isl_class &clazz,
			plain_cpp_generator &generator, bool is_declaration) :
		class_printer(os, clazz, generator, is_declaration),
		generator(generator) {}

	void print_persistent_callback_prototype(FunctionDecl *method);
	void print_persistent_callback_setter_prototype(FunctionDecl *method);
	void print_full_method_header(const Method &method);
	void print_callback_data_decl(ParmVarDecl *param, const string &name);
	virtual bool want_descendent_overloads(const function_set &methods)
		override;
	virtual void print_public_constructors() = 0;
	virtual void print_copy_assignment() = 0;
	virtual void print_destructor() = 0;
	virtual void print_ptr() = 0;
	virtual void print_downcast() = 0;
	virtual void print_ctx() = 0;
	virtual void print_method_separator() = 0;
	virtual void print_persistent_callbacks() = 0;
	void print_public_methods();
	void print_id_constructor_user_header();
	void print_id_user_header(bool optional);
	virtual void print_id_constructor_user() = 0;
	virtual void print_id_user(bool optional) = 0;
	void print_special_id();
	void print_special();
};

/* A helper class for printing method declarations of a class.
 */
struct plain_cpp_generator::decl_printer :
	public plain_cpp_generator::plain_printer
{
	decl_printer(std::ostream &os, const isl_class &clazz,
			plain_cpp_generator &generator) :
		plain_printer(os, clazz, generator, true) {}

	void print_subclass_type();
	void print_class_factory(const std::string &prefix = std::string());
	void print_protected_constructors();
	virtual void print_copy_assignment() override;
	virtual void print_public_constructors() override;
	virtual void print_destructor() override;
	virtual void print_ptr() override;
	void print_isa_type_template(int indent, const isl_class &super);
	virtual void print_downcast() override;
	virtual void print_ctx() override;
	virtual void print_method_separator() override;
	void print_persistent_callback_data(FunctionDecl *method);
	virtual void print_persistent_callbacks() override;
	virtual void print_method(const Method &method) override;
	virtual void print_method(const ConversionMethod &method) override;
	virtual void print_get_method(FunctionDecl *fd) override;
	virtual void print_id_constructor_user() override;
	virtual void print_id_user(bool optional) override;
};

/* A helper class for printing method definitions of a class.
 */
struct plain_cpp_generator::impl_printer :
	public plain_cpp_generator::plain_printer
{
	impl_printer(std::ostream &os, const isl_class &clazz,
			plain_cpp_generator &generator) :
		plain_printer(os, clazz, generator, false) {}

	void print_arg_conversion(ParmVarDecl *dst, ParmVarDecl *src);
	virtual void print_method(const Method &method) override;
	virtual void print_method(const ConversionMethod &method) override;
	virtual void print_get_method(FunctionDecl *fd) override;
	void print_check_ptr(const char *ptr);
	void print_check_ptr_start(const char *ptr);
	void print_check_ptr_end(const char *ptr);
	void print_class_factory();
	void print_protected_constructors();
	virtual void print_public_constructors() override;
	virtual void print_copy_assignment() override;
	virtual void print_destructor() override;
	virtual void print_ptr() override;
	virtual void print_downcast() override;
	virtual void print_ctx() override;
	virtual void print_method_separator() override;
	void print_set_persistent_callback(const Method &method);
	virtual void print_persistent_callbacks() override;
	void print_argument_validity_check(const Method &method);
	void print_save_ctx(const std::string &ctx);
	void print_save_ctx(const Method &method);
	void print_on_error_continue();
	void print_exceptional_execution_check(const Method &method);
	void print_method_return(const Method &method);
	void print_stream_insertion();
	void print_wrapped_call_checked(int indent, const std::string &call);
	void print_wrapped_call(int indent, const std::string &call,
		QualType rtype);
	void print_callback_body(int indent, ParmVarDecl *param,
		const string &name);
	void print_callback_local(ParmVarDecl *param);
	virtual void print_id_constructor_user() override;
	virtual void print_id_user(bool optional) override;
};

#endif
