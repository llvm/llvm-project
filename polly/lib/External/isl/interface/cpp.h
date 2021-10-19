#ifndef ISL_INTERFACE_CPP_H
#define ISL_INTERFACE_CPP_H

#include <iostream>
#include <string>
#include <vector>

#include "generator.h"

/* A generated C++ method derived from an isl function.
 *
 * "clazz" is the class to which the method belongs.
 * "fd" is the original isl function.
 * "name" is the name of the method, which may be different
 * from the default name derived from "fd".
 * "kind" is the type of the method.
 * "callback" stores the callback argument, if any, or NULL.
 */
struct Method {
	enum Kind {
		static_method,
		member_method,
		constructor,
	};

	Method(const isl_class &clazz, FunctionDecl *fd,
		const std::string &name);
	Method(const isl_class &clazz, FunctionDecl *fd);

	int c_num_params() const;
	virtual int num_params() const;
	virtual bool param_needs_copy(int pos) const;
	virtual clang::ParmVarDecl *get_param(int pos) const;
	virtual void print_param_use(ostream &os, int pos) const;
	bool is_subclass_mutator() const;
	static void print_arg_list(std::ostream &os, int start, int end,
		const std::function<void(int i)> &print_arg);
	void print_cpp_arg_list(std::ostream &os,
		const std::function<void(int i)> &print_arg) const;

	const isl_class &clazz;
	FunctionDecl *const fd;
	const std::string name;
	const enum Kind kind;
	ParmVarDecl *const callback;
};

/* A method that does not require its isl type parameters to be a copy.
 */
struct NoCopyMethod : Method {
	NoCopyMethod(const Method &method) : Method(method) {}

	virtual bool param_needs_copy(int pos) const override;
};

/* A generated method that performs one or more argument conversions and
 * then calls the original method.
 *
 * A ConversionMethod inherits from a NoCopyMethod, because
 * unlike methods that call an isl C function,
 * a conversion method never calls release() on an isl type argument,
 * so they can all be passed as const references.
 *
 * "this_type" is the name of the type to which "this" should be converted
 * (if different from clazz.name).
 * "get_param_fn" returns the method argument at position "pos".
 */
struct ConversionMethod : NoCopyMethod {
	ConversionMethod(const Method &method, const std::string &this_type,
		const std::function<clang::ParmVarDecl *(int pos)> &get_param);
	ConversionMethod(const Method &method, const std::string &this_type);
	ConversionMethod(const Method &method,
		const std::function<clang::ParmVarDecl *(int pos)> &get_param);
	virtual clang::ParmVarDecl *get_param(int pos) const override;

	void print_call(std::ostream &os, const std::string &ns) const;

	const std::string this_type;
	const std::function<clang::ParmVarDecl *(int pos)> get_param_fn;
};

/* A specialized generated C++ method for setting an enum.
 *
 * "enum_name" is a string representation of the enum value
 * set by this method.
 */
struct EnumMethod : public Method {
	EnumMethod(const isl_class &clazz, FunctionDecl *fd,
		const std::string &method_name, const std::string &enum_name);

	virtual int num_params() const override;
	virtual void print_param_use(ostream &os, int pos) const override;

	std::string enum_name;
};

/* A type printer for converting argument and return types,
 * as well as the class type,
 * to string representations of the corresponding types
 * in the C++ interface.
 */
struct cpp_type_printer {
	cpp_type_printer() {}

	virtual std::string isl_bool() const;
	virtual std::string isl_stat() const;
	virtual std::string isl_size() const;
	virtual std::string isl_namespace() const;
	virtual std::string class_type(const std::string &cpp_name) const;
	virtual std::string qualified(int arg, const std::string &cpp_type)
		const;
	std::string isl_type(int arg, QualType type) const;
	std::string generate_callback_args(int arg, QualType type, bool cpp)
		const;
	std::string generate_callback_type(int arg, QualType type) const;
	std::string param(int arg, QualType type) const;
	std::string return_type(const Method &method) const;
};

/* Generator for C++ bindings.
 */
class cpp_generator : public generator {
protected:
	struct class_printer;
public:
	cpp_generator(SourceManager &SM, set<RecordDecl *> &exported_types,
		set<FunctionDecl *> exported_functions,
		set<FunctionDecl *> functions);
private:
	void set_class_construction_types(isl_class &clazz);
	void set_construction_types();
	void copy_methods(isl_class &clazz, const std::string &name,
		const isl_class &super, const function_set &methods);
	void copy_super_methods(isl_class &clazz, const isl_class &super);
	void copy_super_methods(isl_class &clazz, set<string> &done);
	void copy_super_methods();
	bool is_implicit_conversion(const Method &cons);
	bool is_subclass(QualType subclass_type, const isl_class &class_type);
public:
	static string type2cpp(const isl_class &clazz);
	static string type2cpp(string type_string);
};

/* A helper class for printing method declarations and definitions
 * of a class.
 *
 * "os" is the stream onto which the methods are printed.
 * "clazz" describes the methods of the class.
 * "cppstring" is the C++ name of the class.
 * "generator" is the C++ interface generator printing the classes.
 * "declarations" is set if this object is used to print declarations.
 */
struct cpp_generator::class_printer {
	std::ostream &os;
	const isl_class &clazz;
	const std::string cppstring;
	cpp_generator &generator;
	const bool declarations;

	class_printer(std::ostream &os, const isl_class &clazz,
			cpp_generator &generator, bool declarations);

	void print_constructors();
	void print_methods();
	bool next_variant(FunctionDecl *fd, std::vector<bool> &convert);
	void print_method_variants(FunctionDecl *fd, const std::string &name);
	virtual bool want_descendent_overloads(const function_set &methods) = 0;
	void print_descendent_overloads(FunctionDecl *fd,
		const std::string &name);
	void print_method_group(const function_set &methods,
		const std::string &name);
	virtual void print_method(const Method &method) = 0;
	virtual void print_method(const ConversionMethod &method) = 0;
	virtual void print_get_method(FunctionDecl *fd) = 0;
	void print_set_enums(FunctionDecl *fd);
	void print_set_enums();
	ParmVarDecl *get_param(FunctionDecl *fd, int pos,
		const std::vector<bool> &convert);
	void print_method_header(const Method &method,
		const cpp_type_printer &type_printer);
};

#endif
