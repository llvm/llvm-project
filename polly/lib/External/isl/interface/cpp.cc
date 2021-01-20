/*
 * Copyright 2016, 2017 Tobias Grosser. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY TOBIAS GROSSER ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SVEN VERDOOLAEGE OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation
 * are those of the authors and should not be interpreted as
 * representing official policies, either expressed or implied, of
 * Tobias Grosser.
 */

#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "cpp.h"
#include "isl_config.h"

/* Print string formatted according to "fmt" to ostream "os".
 *
 * This osprintf method allows us to use printf style formatting constructs when
 * writing to an ostream.
 */
static void osprintf(ostream &os, const char *format, va_list arguments)
{
	va_list copy;
	char *string_pointer;
	size_t size;

	va_copy(copy, arguments);
	size = vsnprintf(NULL, 0, format, copy);
	string_pointer = new char[size + 1];
	va_end(copy);
	vsnprintf(string_pointer, size + 1, format, arguments);
	os << string_pointer;
	delete[] string_pointer;
}

/* Print string formatted according to "fmt" to ostream "os".
 *
 * This osprintf method allows us to use printf style formatting constructs when
 * writing to an ostream.
 */
static void osprintf(ostream &os, const char *format, ...)
{
	va_list arguments;

	va_start(arguments, format);
	osprintf(os, format, arguments);
	va_end(arguments);
}

/* Print string formatted according to "fmt" to ostream "os"
 * with the given indentation.
 *
 * This osprintf method allows us to use printf style formatting constructs when
 * writing to an ostream.
 */
static void osprintf(ostream &os, int indent, const char *format, ...)
{
	va_list arguments;

	osprintf(os, "%*s", indent, " ");
	va_start(arguments, format);
	osprintf(os, format, arguments);
	va_end(arguments);
}

/* Convert "l" to a string.
 */
static std::string to_string(long l)
{
	std::ostringstream strm;
	strm << l;
	return strm.str();
}

/* Generate a cpp interface based on the extracted types and functions.
 *
 * Print first a set of forward declarations for all isl wrapper
 * classes, then the declarations of the classes, and at the end all
 * implementations.
 *
 * If checked C++ bindings are being generated,
 * then wrap them in a namespace to avoid conflicts
 * with the default C++ bindings (with automatic checks using exceptions).
 */
void cpp_generator::generate()
{
	ostream &os = cout;

	osprintf(os, "\n");
	osprintf(os, "namespace isl {\n\n");
	if (checked)
		osprintf(os, "namespace checked {\n\n");

	print_forward_declarations(os);
	osprintf(os, "\n");
	print_declarations(os);
	osprintf(os, "\n");
	print_implementations(os);

	if (checked)
		osprintf(os, "} // namespace checked\n");
	osprintf(os, "} // namespace isl\n");
}

/* Print forward declarations for all classes to "os".
*/
void cpp_generator::print_forward_declarations(ostream &os)
{
	map<string, isl_class>::iterator ci;

	osprintf(os, "// forward declarations\n");

	for (ci = classes.begin(); ci != classes.end(); ++ci)
		print_class_forward_decl(os, ci->second);
}

/* Print all declarations to "os".
 */
void cpp_generator::print_declarations(ostream &os)
{
	map<string, isl_class>::iterator ci;
	bool first = true;

	for (ci = classes.begin(); ci != classes.end(); ++ci) {
		if (first)
			first = false;
		else
			osprintf(os, "\n");

		print_class(os, ci->second);
	}
}

/* Print all implementations to "os".
 */
void cpp_generator::print_implementations(ostream &os)
{
	map<string, isl_class>::iterator ci;
	bool first = true;

	for (ci = classes.begin(); ci != classes.end(); ++ci) {
		if (first)
			first = false;
		else
			osprintf(os, "\n");

		print_class_impl(os, ci->second);
	}
}

/* If "clazz" is a subclass that is based on a type function,
 * then introduce a "type" field that holds the value of the type
 * corresponding to the subclass and make the fields of the class
 * accessible to the "isa" and "as" methods of the (immediate) superclass.
 * In particular, "isa" needs access to the type field itself,
 * while "as" needs access to the private constructor.
 * In case of the "isa" method, all instances are made friends
 * to avoid access right confusion.
 */
void cpp_generator::print_subclass_type(ostream &os, const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	std::string super;
	const char *cppname = cppstring.c_str();
	const char *supername;

	if (!clazz.is_type_subclass())
		return;

	super = type2cpp(clazz.superclass_name);
	supername = super.c_str();
	osprintf(os, "  template <class T>\n");
	osprintf(os, "  friend %s %s::isa() const;\n",
		isl_bool2cpp().c_str(), supername);
	osprintf(os, "  friend %s %s::as<%s>() const;\n",
		cppname, supername, cppname);
	osprintf(os, "  static const auto type = %s;\n",
		clazz.subclass_name.c_str());
}

/* Print declarations for class "clazz" to "os".
 *
 * If "clazz" is a subclass based on a type function,
 * then it is made to inherit from the (immediate) superclass and
 * a "type" attribute is added for use in the "as" and "isa"
 * methods of the superclass.
 *
 * Conversely, if "clazz" is a superclass with a type function,
 * then declare those "as" and "isa" methods.
 *
 * The pointer to the isl object is only added for classes that
 * are not subclasses, since subclasses refer to the same isl object.
 */
void cpp_generator::print_class(ostream &os, const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "// declarations for isl::%s\n", cppname);

	print_class_factory_decl(os, clazz);
	osprintf(os, "\n");
	osprintf(os, "class %s ", cppname);
	if (clazz.is_type_subclass())
		osprintf(os, ": public %s ",
			type2cpp(clazz.superclass_name).c_str());
	osprintf(os, "{\n");
	print_subclass_type(os, clazz);
	print_class_factory_decl(os, clazz, "  friend ");
	osprintf(os, "\n");
	osprintf(os, "protected:\n");
	if (!clazz.is_type_subclass()) {
		osprintf(os, "  %s *ptr = nullptr;\n", name);
		osprintf(os, "\n");
	}
	print_protected_constructors_decl(os, clazz);
	osprintf(os, "\n");
	osprintf(os, "public:\n");
	print_public_constructors_decl(os, clazz);
	print_constructors_decl(os, clazz);
	print_copy_assignment_decl(os, clazz);
	print_destructor_decl(os, clazz);
	print_ptr_decl(os, clazz);
	print_downcast_decl(os, clazz);
	print_ctx_decl(os);
	osprintf(os, "\n");
	print_persistent_callbacks_decl(os, clazz);
	print_methods_decl(os, clazz);
	print_set_enums_decl(os, clazz);

	osprintf(os, "};\n");
}

/* Print forward declaration of class "clazz" to "os".
 */
void cpp_generator::print_class_forward_decl(ostream &os,
	const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "class %s;\n", cppname);
}

/* Print global factory functions to "os".
 *
 * Each class has two global factory functions:
 *
 * 	set manage(__isl_take isl_set *ptr);
 * 	set manage_copy(__isl_keep isl_set *ptr);
 *
 * A user can construct isl C++ objects from a raw pointer and indicate whether
 * they intend to take the ownership of the object or not through these global
 * factory functions. This ensures isl object creation is very explicit and
 * pointers are not converted by accident. Thanks to overloading, manage() and
 * manage_copy() can be called on any isl raw pointer and the corresponding
 * object is automatically created, without the user having to choose the right
 * isl object type.
 *
 * For a subclass based on a type function, no factory functions
 * are introduced because they share the C object type with
 * the superclass.
 */
void cpp_generator::print_class_factory_decl(ostream &os,
	const isl_class &clazz, const std::string &prefix)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	if (clazz.is_type_subclass())
		return;

	os << prefix;
	osprintf(os, "inline %s manage(__isl_take %s *ptr);\n", cppname, name);
	os << prefix;
	osprintf(os, "inline %s manage_copy(__isl_keep %s *ptr);\n",
		cppname, name);
}

/* Print declarations of protected constructors for class "clazz" to "os".
 *
 * Each class has currently one protected constructor:
 *
 * 	1) Constructor from a plain isl_* C pointer
 *
 * Example:
 *
 * 	set(__isl_take isl_set *ptr);
 *
 * The raw pointer constructor is kept protected. Object creation is only
 * possible through manage() or manage_copy().
 */
void cpp_generator::print_protected_constructors_decl(ostream &os,
	const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "  inline explicit %s(__isl_take %s *ptr);\n", cppname,
		 name);
}

/* Print declarations of public constructors for class "clazz" to "os".
 *
 * Each class currently has two public constructors:
 *
 * 	1) A default constructor
 * 	2) A copy constructor
 *
 * Example:
 *
 *	set();
 *	set(const set &set);
 */
void cpp_generator::print_public_constructors_decl(ostream &os,
	const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();
	osprintf(os, "  inline /* implicit */ %s();\n", cppname);

	osprintf(os, "  inline /* implicit */ %s(const %s &obj);\n",
		 cppname, cppname);
}

/* Print declarations for "method" in class "clazz" to "os".
 *
 * "kind" specifies the kind of method that should be generated.
 *
 * "convert" specifies which of the method arguments should
 * be automatically converted.
 */
template <>
void cpp_generator::print_method<cpp_generator::decl>(ostream &os,
	const isl_class &clazz, FunctionDecl *method, function_kind kind,
	const std::vector<bool> &convert)
{
	string name = clazz.method_name(method);

	print_named_method_decl(os, clazz, method, name, kind, convert);
}

/* Print declarations for "method" in class "clazz" to "os",
 * without any argument conversions.
 *
 * "kind" specifies the kind of method that should be generated.
 */
template <>
void cpp_generator::print_method<cpp_generator::decl>(ostream &os,
	const isl_class &clazz, FunctionDecl *method, function_kind kind)
{
	print_method<decl>(os, clazz,method, kind, {});
}

/* Print declarations for constructors for class "class" to "os".
 *
 * For each isl function that is marked as __isl_constructor,
 * add a corresponding C++ constructor.
 *
 * Example:
 *
 * 	inline /\* implicit *\/ union_set(basic_set bset);
 * 	inline /\* implicit *\/ union_set(set set);
 * 	inline explicit val(ctx ctx, long i);
 * 	inline explicit val(ctx ctx, const std::string &str);
 */
void cpp_generator::print_constructors_decl(ostream &os,
	const isl_class &clazz)
{
	function_set::const_iterator in;
	const function_set &constructors = clazz.constructors;

	for (in = constructors.begin(); in != constructors.end(); ++in) {
		FunctionDecl *cons = *in;

		print_method<decl>(os, clazz, cons, function_kind_constructor);
	}
}

/* Print declarations of copy assignment operator for class "clazz"
 * to "os".
 *
 * Each class has one assignment operator.
 *
 * 	isl:set &set::operator=(set obj)
 *
 */
void cpp_generator::print_copy_assignment_decl(ostream &os,
	const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "  inline %s &operator=(%s obj);\n", cppname, cppname);
}

/* Print declaration of destructor for class "clazz" to "os".
 *
 * No explicit destructor is needed for type based subclasses.
 */
void cpp_generator::print_destructor_decl(ostream &os, const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	if (clazz.is_type_subclass())
		return;

	osprintf(os, "  inline ~%s();\n", cppname);
}

/* Print declaration of pointer functions for class "clazz" to "os".
 * Since type based subclasses share the pointer with their superclass,
 * they can also reuse these functions from the superclass.
 *
 * To obtain a raw pointer three functions are provided:
 *
 * 	1) __isl_give isl_set *copy()
 *
 * 	  Returns a pointer to a _copy_ of the internal object
 *
 * 	2) __isl_keep isl_set *get()
 *
 * 	  Returns a pointer to the internal object
 *
 * 	3) __isl_give isl_set *release()
 *
 * 	  Returns a pointer to the internal object and resets the
 * 	  internal pointer to nullptr.
 *
 * We also provide functionality to explicitly check if a pointer is
 * currently managed by this object.
 *
 * 	4) bool is_null()
 *
 * 	  Check if the current object is a null pointer.
 *
 * The functions get() and release() model the value_ptr proposed in
 * http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3339.pdf.
 * The copy() function is an extension to allow the user to explicitly
 * copy the underlying object.
 *
 * Also generate a declaration to delete copy() for r-values, for
 * r-values release() should be used to avoid unnecessary copies.
 */
void cpp_generator::print_ptr_decl(ostream &os, const isl_class &clazz)
{
	const char *name = clazz.name.c_str();

	if (clazz.is_type_subclass())
		return;

	osprintf(os, "  inline __isl_give %s *copy() const &;\n", name);
	osprintf(os, "  inline __isl_give %s *copy() && = delete;\n", name);
	osprintf(os, "  inline __isl_keep %s *get() const;\n", name);
	osprintf(os, "  inline __isl_give %s *release();\n", name);
	osprintf(os, "  inline bool is_null() const;\n");
}

/* Print a template declaration with given indentation
 * for the "isa_type" method that ensures it is only enabled
 * when called with a template argument
 * that represents a type that is equal to that
 * of the return type of the type function of "super".
 * In particular, "isa_type" gets called from "isa"
 * with as template argument the type of the "type" field
 * of the subclass.
 * The check ensures that this subclass is in fact a direct subclass
 * of "super".
 */
void cpp_generator::print_isa_type_template(ostream &os, int indent,
	const isl_class &super)
{
	osprintf(os, indent,
		"template <typename T,\n");
	osprintf(os, indent,
		"        typename = typename std::enable_if<std::is_same<\n");
	osprintf(os, indent,
		"                const decltype(%s(NULL)),\n",
		super.fn_type->getNameAsString().c_str());
	osprintf(os, indent,
		"                const T>::value>::type>\n");
}

/* Print declarations for the "as" and "isa" methods, if "clazz"
 * is a superclass with a type function.
 *
 * "isa" checks whether an object is of a given subclass type.
 * "isa_type" does the same, but gets passed the value of the type field
 * of the subclass as a function argument and the type of this field
 * as a template argument.
 * "as" tries to cast an object to a given subclass type, returning
 * an invalid object if the object is not of the given type.
 */
void cpp_generator::print_downcast_decl(ostream &os, const isl_class &clazz)
{
	if (!clazz.fn_type)
		return;

	osprintf(os, "private:\n");
	print_isa_type_template(os, 2, clazz);
	osprintf(os, "  inline %s isa_type(T subtype) const;\n",
		isl_bool2cpp().c_str());
	osprintf(os, "public:\n");
	osprintf(os, "  template <class T> inline %s isa() const;\n",
		isl_bool2cpp().c_str());
	osprintf(os, "  template <class T> inline T as() const;\n");
}

/* Print the declaration of the ctx method.
 */
void cpp_generator::print_ctx_decl(ostream &os)
{
	std::string ns = isl_namespace();

	osprintf(os, "  inline %sctx ctx() const;\n", ns.c_str());
}

/* Add a space to the return type "type" if needed,
 * i.e., if it is not the type of a pointer.
 */
static string add_space_to_return_type(const string &type)
{
	if (type[type.size() - 1] == '*')
		return type;
	return type + " ";
}

/* Print the prototype of the static inline method that is used
 * as the C callback of "clazz" set by "method" to "os".
 */
void cpp_generator::print_persistent_callback_prototype(ostream &os,
	const isl_class &clazz, FunctionDecl *method, bool is_declaration)
{
	string callback_name, rettype, c_args;
	ParmVarDecl *param = persistent_callback_arg(method);
	const FunctionProtoType *callback;
	QualType ptype;
	string classname;

	ptype = param->getType();
	callback = extract_prototype(ptype);

	rettype = callback->getReturnType().getAsString();
	rettype = add_space_to_return_type(rettype);
	callback_name = clazz.persistent_callback_name(method);
	c_args = generate_callback_args(ptype, false);

	if (!is_declaration)
		classname = type2cpp(clazz) + "::";

	osprintf(os, "%s%s%s(%s)",
		 rettype.c_str(), classname.c_str(),
		 callback_name.c_str(), c_args.c_str());
}

/* Print the prototype of the method for setting the callback function
 * of "clazz" set by "method" to "os".
 */
void cpp_generator::print_persistent_callback_setter_prototype(ostream &os,
	const isl_class &clazz, FunctionDecl *method, bool is_declaration)
{
	string classname, callback_name, cpptype;
	ParmVarDecl *param = persistent_callback_arg(method);

	if (!is_declaration)
		classname = type2cpp(clazz) + "::";

	cpptype = type2cpp(param->getOriginalType());
	callback_name = clazz.persistent_callback_name(method);
	osprintf(os, "void %sset_%s_data(const %s &%s)",
		classname.c_str(), callback_name.c_str(), cpptype.c_str(),
		param->getName().str().c_str());
}

/* Given a function "method" for setting a "clazz" persistent callback,
 * print the fields that are needed for marshalling the callback to "os".
 *
 * In particular, print
 * - the declaration of a data structure for storing the C++ callback function
 * - a shared pointer to such a data structure
 * - the declaration of a static inline method
 *   for use as the C callback function
 * - the declaration of a private method for setting the callback function
 */
void cpp_generator::print_persistent_callback_data(ostream &os,
	const isl_class &clazz, FunctionDecl *method)
{
	string callback_name;
	ParmVarDecl *param = persistent_callback_arg(method);

	callback_name = clazz.persistent_callback_name(method);
	print_callback_data_decl(os, param, callback_name);
	osprintf(os, ";\n");
	osprintf(os, "  std::shared_ptr<%s_data> %s_data;\n",
		callback_name.c_str(), callback_name.c_str());
	osprintf(os, "  static inline ");
	print_persistent_callback_prototype(os, clazz, method, true);
	osprintf(os, ";\n");
	osprintf(os, "  inline ");
	print_persistent_callback_setter_prototype(os, clazz, method, true);
	osprintf(os, ";\n");
}

/* Print declarations needed for the persistent callbacks of "clazz".
 *
 * In particular, if there are any persistent callbacks, then
 * print a private method for copying callback data from
 * one object to another,
 * private data for keeping track of the persistent callbacks and
 * public methods for setting the persistent callbacks.
 */
void cpp_generator::print_persistent_callbacks_decl(ostream &os,
	const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();
	set<FunctionDecl *>::const_iterator in;
	const set<FunctionDecl *> &callbacks = clazz.persistent_callbacks;

	if (!clazz.has_persistent_callbacks())
		return;

	osprintf(os, "private:\n");
	osprintf(os, "  inline %s &copy_callbacks(const %s &obj);\n",
		cppname, cppname);
	for (in = callbacks.begin(); in != callbacks.end(); ++in)
		print_persistent_callback_data(os, clazz, *in);

	osprintf(os, "public:\n");
	for (in = callbacks.begin(); in != callbacks.end(); ++in)
		print_method<decl>(os, clazz, *in, function_kind_member_method);
}

/* Print declarations for methods in class "clazz" to "os".
 */
void cpp_generator::print_methods_decl(ostream &os, const isl_class &clazz)
{
	map<string, function_set >::const_iterator it;

	for (it = clazz.methods.begin(); it != clazz.methods.end(); ++it)
		print_method_group_decl(os, clazz, it->second);
}

/* Print a declaration for a method "name" in "clazz" derived
 * from "fd", which sets an enum, to "os".
 *
 * The last argument is removed because it is replaced by
 * a break-up into several methods.
 */
void cpp_generator::print_set_enum_decl(ostream &os, const isl_class &clazz,
	FunctionDecl *fd, const string &name)
{
	int n = fd->getNumParams();

	print_method_header(os, clazz, fd, name, n - 1, true,
				function_kind_member_method);
}

/* Print declarations for the methods in "clazz" derived from "fd",
 * which sets an enum, to "os".
 *
 * A method is generated for each value in the enum, setting
 * the enum to that value.
 */
void cpp_generator::print_set_enums_decl(ostream &os, const isl_class &clazz,
	FunctionDecl *fd)
{
	vector<set_enum>::const_iterator it;
	const vector<set_enum> &set_enums = clazz.set_enums.at(fd);

	for (it = set_enums.begin(); it != set_enums.end(); ++it)
		print_set_enum_decl(os, clazz, fd, it->method_name);
}

/* Print declarations for methods in "clazz" derived from functions
 * that set an enum, to "os".
 */
void cpp_generator::print_set_enums_decl(ostream &os, const isl_class &clazz)
{
	map<FunctionDecl *, vector<set_enum> >::const_iterator it;

	for (it = clazz.set_enums.begin(); it != clazz.set_enums.end(); ++it)
		print_set_enums_decl(os, clazz, it->first);
}

/* Print a declaration for the "get" method "fd" in class "clazz",
 * using a name that includes the "get_" prefix, to "os".
 */
template<>
void cpp_generator::print_get_method<cpp_generator::decl>(ostream &os,
	const isl_class &clazz, FunctionDecl *fd)
{
	function_kind kind = function_kind_member_method;
	string base = clazz.base_method_name(fd);

	print_named_method_decl(os, clazz, fd, base, kind);
}

/* Update "convert" to reflect the next combination of automatic conversions
 * for the arguments of "fd",
 * returning false if there are no more combinations.
 *
 * In particular, find the last argument for which an automatic
 * conversion function is available mapping to the type of this argument and
 * that is not already marked for conversion.
 * Mark this argument, if any, for conversion and clear the markings
 * of all subsequent arguments.
 * Repeated calls to this method therefore run through
 * all possible combinations.
 *
 * Note that the first function argument is never considered
 * for automatic conversion since this is the argument
 * from which the isl_ctx used in the conversion is extracted.
 */
bool cpp_generator::next_variant(FunctionDecl *fd, std::vector<bool> &convert)
{
	size_t n = convert.size();

	for (int i = n - 1; i >= 1; --i) {
		ParmVarDecl *param = fd->getParamDecl(i);
		const Type *type = param->getOriginalType().getTypePtr();

		if (conversions.count(type) == 0)
			continue;
		if (convert[i])
			continue;
		convert[i] = true;
		for (size_t j = i + 1; j < n; ++j)
			convert[j] = false;
		return true;
	}

	return false;
}

/* Print a declaration or definition for method "fd" in class "clazz"
 * to "os".
 *
 * For methods that are identified as "get" methods, also
 * print a declaration or definition for the method
 * using a name that includes the "get_" prefix.
 *
 * If the generated method is an object method, then check
 * whether any of its arguments can be automatically converted
 * from something else, and, if so, generate a method
 * for each combination of converted arguments.
 */
template <enum cpp_generator::method_part part>
void cpp_generator::print_method_variants(ostream &os, const isl_class &clazz,
	FunctionDecl *fd)
{
	function_kind kind = get_method_kind(clazz, fd);
	std::vector<bool> convert(fd->getNumParams());

	print_method<part>(os, clazz, fd, kind);
	if (clazz.is_get_method(fd))
		print_get_method<part>(os, clazz, fd);
	if (kind == function_kind_member_method)
		while (next_variant(fd, convert))
			print_method<part>(os, clazz, fd, kind, convert);
}

/* Print declarations for methods "methods" in class "clazz" to "os".
 */
void cpp_generator::print_method_group_decl(ostream &os, const isl_class &clazz,
	const function_set &methods)
{
	function_set::const_iterator it;

	for (it = methods.begin(); it != methods.end(); ++it)
		print_method_variants<decl>(os, clazz, *it);
}

/* Print a declaration for a method called "name" in class "clazz"
 * derived from "fd" to "os".
 *
 * "kind" specifies the kind of method that should be generated.
 *
 * "convert" specifies which of the method arguments should
 * be automatically converted.
 */
void cpp_generator::print_named_method_decl(ostream &os, const isl_class &clazz,
	FunctionDecl *fd, const string &name, function_kind kind,
	const std::vector<bool> &convert)
{
	print_named_method_header(os, clazz, fd, name, true, kind, convert);
}

/* Print implementations for class "clazz" to "os".
 */
void cpp_generator::print_class_impl(ostream &os, const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "// implementations for isl::%s", cppname);

	print_class_factory_impl(os, clazz);
	print_public_constructors_impl(os, clazz);
	print_protected_constructors_impl(os, clazz);
	print_constructors_impl(os, clazz);
	print_copy_assignment_impl(os, clazz);
	print_destructor_impl(os, clazz);
	print_ptr_impl(os, clazz);
	print_downcast_impl(os, clazz);
	print_ctx_impl(os, clazz);
	print_persistent_callbacks_impl(os, clazz);
	print_methods_impl(os, clazz);
	print_set_enums_impl(os, clazz);
	print_stream_insertion(os, clazz);
}

/* Print code for throwing an exception corresponding to the last error
 * that occurred on "saved_ctx".
 * This assumes that a valid isl::ctx is available in the "saved_ctx" variable,
 * e.g., through a prior call to print_save_ctx.
 */
static void print_throw_last_error(ostream &os)
{
	osprintf(os, "    exception::throw_last_error(saved_ctx);\n");
}

/* Print code with the given indentation
 * for throwing an exception_invalid with the given message.
 */
static void print_throw_invalid(ostream &os, int indent, const char *msg)
{
	osprintf(os, indent,
		"exception::throw_invalid(\"%s\", __FILE__, __LINE__);\n", msg);
}

/* Print code for throwing an exception on NULL input.
 */
static void print_throw_NULL_input(ostream &os)
{
	print_throw_invalid(os, 4, "NULL input");
}

/* Print code with the given indentation
 * for acting on an invalid error with message "msg".
 * In particular, throw an exception_invalid.
 * In the checked C++ bindings, isl_die is called instead with the code
 * in "checked_code".
 */
void cpp_generator::print_invalid(ostream &os, int indent, const char *msg,
	const char *checked_code)
{
	if (checked)
		osprintf(os, indent,
			"isl_die(ctx().get(), isl_error_invalid, "
			"\"%s\", %s);\n", msg, checked_code);
	else
		print_throw_invalid(os, indent, msg);
}

/* Print an operator for inserting objects of "class"
 * into an output stream.
 *
 * Unless checked C++ bindings are being generated,
 * the operator requires its argument to be non-NULL.
 * An exception is thrown if anything went wrong during the printing.
 * During this printing, isl is made not to print any error message
 * because the error message is included in the exception.
 *
 * If checked C++ bindings are being generated and anything went wrong,
 * then record this failure in the output stream.
 */
void cpp_generator::print_stream_insertion(ostream &os, const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	if (!clazz.fn_to_str)
		return;

	osprintf(os, "\n");
	osprintf(os, "inline std::ostream &operator<<(std::ostream &os, ");
	osprintf(os, "const %s &obj)\n", cppname);
	osprintf(os, "{\n");
	print_check_ptr_start(os, clazz, "obj.get()");
	osprintf(os, "  char *str = %s_to_str(obj.get());\n", name);
	print_check_ptr_end(os, "str");
	if (checked) {
		osprintf(os, "  if (!str) {\n");
		osprintf(os, "    os.setstate(std::ios_base::badbit);\n");
		osprintf(os, "    return os;\n");
		osprintf(os, "  }\n");
	}
	osprintf(os, "  os << str;\n");
	osprintf(os, "  free(str);\n");
	osprintf(os, "  return os;\n");
	osprintf(os, "}\n");
}

/* Print code that checks that "ptr" is not NULL at input.
 *
 * Omit the check if checked C++ bindings are being generated.
 */
void cpp_generator::print_check_ptr(ostream &os, const char *ptr)
{
	if (checked)
		return;

	osprintf(os, "  if (!%s)\n", ptr);
	print_throw_NULL_input(os);
}

/* Print code that checks that "ptr" is not NULL at input and
 * that saves a copy of the isl_ctx of "ptr" for a later check.
 *
 * Omit the check if checked C++ bindings are being generated.
 */
void cpp_generator::print_check_ptr_start(ostream &os, const isl_class &clazz,
	const char *ptr)
{
	if (checked)
		return;

	print_check_ptr(os, ptr);
	osprintf(os, "  auto saved_ctx = %s_get_ctx(%s);\n",
		clazz.name.c_str(), ptr);
	print_on_error_continue(os);
}

/* Print code that checks that "ptr" is not NULL at the end.
 * A copy of the isl_ctx is expected to have been saved by
 * code generated by print_check_ptr_start.
 *
 * Omit the check if checked C++ bindings are being generated.
 */
void cpp_generator::print_check_ptr_end(ostream &os, const char *ptr)
{
	if (checked)
		return;

	osprintf(os, "  if (!%s)\n", ptr);
	print_throw_last_error(os);
}

/* Print implementation of global factory functions to "os".
 *
 * Each class has two global factory functions:
 *
 * 	set manage(__isl_take isl_set *ptr);
 * 	set manage_copy(__isl_keep isl_set *ptr);
 *
 * Unless checked C++ bindings are being generated,
 * both functions require the argument to be non-NULL.
 * An exception is thrown if anything went wrong during the copying
 * in manage_copy.
 * During the copying, isl is made not to print any error message
 * because the error message is included in the exception.
 *
 * For a subclass based on a type function, no factory functions
 * are introduced because they share the C object type with
 * the superclass.
 */
void cpp_generator::print_class_factory_impl(ostream &os,
	const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	if (clazz.is_type_subclass())
		return;

	osprintf(os, "\n");
	osprintf(os, "%s manage(__isl_take %s *ptr) {\n", cppname, name);
	print_check_ptr(os, "ptr");
	osprintf(os, "  return %s(ptr);\n", cppname);
	osprintf(os, "}\n");

	osprintf(os, "%s manage_copy(__isl_keep %s *ptr) {\n", cppname,
		name);
	print_check_ptr_start(os, clazz, "ptr");
	osprintf(os, "  ptr = %s_copy(ptr);\n", name);
	print_check_ptr_end(os, "ptr");
	osprintf(os, "  return %s(ptr);\n", cppname);
	osprintf(os, "}\n");
}

/* Print implementations of protected constructors for class "clazz" to "os".
 *
 * The pointer to the isl object is either initialized directly or
 * through the (immediate) superclass.
 */
void cpp_generator::print_protected_constructors_impl(ostream &os,
	const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();
	bool subclass = clazz.is_type_subclass();

	osprintf(os, "\n");
	osprintf(os, "%s::%s(__isl_take %s *ptr)\n", cppname, cppname, name);
	if (subclass)
		osprintf(os, "    : %s(ptr) {}\n",
			type2cpp(clazz.superclass_name).c_str());
	else
		osprintf(os, "    : ptr(ptr) {}\n");
}

/* Print implementations of public constructors for class "clazz" to "os".
 *
 * The pointer to the isl object is either initialized directly or
 * through the (immediate) superclass.
 *
 * If the class has any persistent callbacks, then copy them
 * from the original object in the copy constructor.
 * If the class is a subclass, then the persistent callbacks
 * are assumed to be copied by the copy constructor of the superclass.
 *
 * Throw an exception from the copy constructor if anything went wrong
 * during the copying or if the input is NULL, if any copying is performed.
 * During the copying, isl is made not to print any error message
 * because the error message is included in the exception.
 * No exceptions are thrown if checked C++ bindings
 * are being generated,
 */
void cpp_generator::print_public_constructors_impl(ostream &os,
	const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	std::string super;
	const char *cppname = cppstring.c_str();
	bool subclass = clazz.is_type_subclass();

	osprintf(os, "\n");
	if (subclass)
		super = type2cpp(clazz.superclass_name);
	osprintf(os, "%s::%s()\n", cppname, cppname);
	if (subclass)
		osprintf(os, "    : %s() {}\n\n", super.c_str());
	else
		osprintf(os, "    : ptr(nullptr) {}\n\n");
	osprintf(os, "%s::%s(const %s &obj)\n", cppname, cppname, cppname);
	if (subclass)
		osprintf(os, "    : %s(obj)\n", super.c_str());
	else
		osprintf(os, "    : ptr(nullptr)\n");
	osprintf(os, "{\n");
	if (!subclass) {
		print_check_ptr_start(os, clazz, "obj.ptr");
		osprintf(os, "  ptr = obj.copy();\n");
		if (clazz.has_persistent_callbacks())
			osprintf(os, "  copy_callbacks(obj);\n");
		print_check_ptr_end(os, "ptr");
	}
	osprintf(os, "}\n");
}

/* Print definition for "method" in class "clazz" to "os",
 * without any automatic type conversions.
 *
 * "kind" specifies the kind of method that should be generated.
 *
 * This method distinguishes three kinds of methods: member methods, static
 * methods, and constructors.
 *
 * Member methods call "method" by passing to the underlying isl function the
 * isl object belonging to "this" as first argument and the remaining arguments
 * as subsequent arguments.
 *
 * Static methods call "method" by passing all arguments to the underlying isl
 * function, as no this-pointer is available. The result is a newly managed
 * isl C++ object.
 *
 * Constructors create a new object from a given set of input parameters. They
 * do not return a value, but instead update the pointer stored inside the
 * newly created object.
 *
 * If the method has a callback argument, we reduce the number of parameters
 * that are exposed by one to hide the user pointer from the interface. On
 * the C++ side no user pointer is needed, as arguments can be forwarded
 * as part of the std::function argument which specifies the callback function.
 *
 * Unless checked C++ bindings are being generated,
 * the inputs of the method are first checked for being valid isl objects and
 * a copy of the associated isl::ctx is saved (if needed).
 * If any failure occurs, either during the check for the inputs or
 * during the isl function call, an exception is thrown.
 * During the function call, isl is made not to print any error message
 * because the error message is included in the exception.
 */
template<>
void cpp_generator::print_method<cpp_generator::impl>(ostream &os,
	const isl_class &clazz, FunctionDecl *method, function_kind kind)
{
	string methodname = method->getName().str();
	int num_params = method->getNumParams();

	osprintf(os, "\n");
	print_method_header(os, clazz, method, false, kind);
	osprintf(os, "{\n");
	print_argument_validity_check(os, method, kind);
	print_save_ctx(os, method, kind);
	print_on_error_continue(os);

	for (int i = 0; i < num_params; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		if (is_callback(param->getType())) {
			num_params -= 1;
			print_callback_local(os, param);
		}
	}

	osprintf(os, "  auto res = %s(", methodname.c_str());

	for (int i = 0; i < num_params; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		bool load_from_this_ptr = false;

		if (i == 0 && kind == function_kind_member_method)
			load_from_this_ptr = true;

		print_method_param_use(os, param, load_from_this_ptr);

		if (i != num_params - 1)
			osprintf(os, ", ");
	}
	osprintf(os, ");\n");

	print_exceptional_execution_check(os, clazz, method, kind);
	if (kind == function_kind_constructor) {
		osprintf(os, "  ptr = res;\n");
	} else {
		print_method_return(os, clazz, method);
	}

	osprintf(os, "}\n");
}

/* Print a definition for "method" in class "clazz" to "os",
 * where at least one of the argument types needs to be converted,
 * as specified by "convert".
 *
 * "kind" specifies the kind of method that should be generated and
 * is assumed to be set to function_kind_member_method.
 *
 * The generated method performs the required conversion(s) and
 * calls the method generated without conversions.
 *
 * Each conversion is performed by calling the conversion function
 * with as arguments the isl_ctx of the object and the argument
 * to the generated method.
 * In order to be able to use this isl_ctx, the current object needs
 * to valid.  The validity of other arguments is checked
 * by the called method.
 */
template<>
void cpp_generator::print_method<cpp_generator::impl>(ostream &os,
	const isl_class &clazz, FunctionDecl *method, function_kind kind,
	const std::vector<bool> &convert)
{
	string name = clazz.method_name(method);
	int num_params = method->getNumParams();

	if (kind != function_kind_member_method)
		die("Automatic conversion currently only supported "
		    "for object methods");

	osprintf(os, "\n");
	print_named_method_header(os, clazz, method, name, false,
				  kind, convert);
	osprintf(os, "{\n");
	print_check_ptr(os, "ptr");
	osprintf(os, "  return this->%s(", name.c_str());
	for (int i = 1; i < num_params; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		std::string name = param->getName().str();

		if (i != 1)
			osprintf(os, ", ");
		if (convert[i]) {
			QualType type = param->getOriginalType();
			string cpptype = type2cpp(type);
			osprintf(os, "%s(ctx(), %s)",
				cpptype.c_str(), name.c_str());
		} else {
			osprintf(os, "%s", name.c_str());
		}
	}
	osprintf(os, ");\n");
	osprintf(os, "}\n");
}

/* Print implementations of constructors for class "clazz" to "os".
 */
void cpp_generator::print_constructors_impl(ostream &os,
	const isl_class &clazz)
{
	function_set::const_iterator in;
	const function_set constructors = clazz.constructors;

	for (in = constructors.begin(); in != constructors.end(); ++in) {
		FunctionDecl *cons = *in;

		print_method<impl>(os, clazz, cons, function_kind_constructor);
	}
}

/* Print implementation of copy assignment operator for class "clazz" to "os".
 *
 * If the class has any persistent callbacks, then copy them
 * from the original object.
 */
void cpp_generator::print_copy_assignment_impl(ostream &os,
	const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "\n");
	osprintf(os, "%s &%s::operator=(%s obj) {\n", cppname,
		 cppname, cppname);
	osprintf(os, "  std::swap(this->ptr, obj.ptr);\n", name);
	if (clazz.has_persistent_callbacks())
		osprintf(os, "  copy_callbacks(obj);\n");
	osprintf(os, "  return *this;\n");
	osprintf(os, "}\n");
}

/* Print implementation of destructor for class "clazz" to "os".
 *
 * No explicit destructor is needed for type based subclasses.
 */
void cpp_generator::print_destructor_impl(ostream &os,
	const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	if (clazz.is_type_subclass())
		return;

	osprintf(os, "\n");
	osprintf(os, "%s::~%s() {\n", cppname, cppname);
	osprintf(os, "  if (ptr)\n");
	osprintf(os, "    %s_free(ptr);\n", name);
	osprintf(os, "}\n");
}

/* Print a check that the persistent callback corresponding to "fd"
 * is not set, throwing an exception (or printing an error message
 * and returning nullptr) if it is set.
 */
void cpp_generator::print_check_no_persistent_callback(ostream &os,
	const isl_class &clazz, FunctionDecl *fd)
{
	string callback_name = clazz.persistent_callback_name(fd);

	osprintf(os, "  if (%s_data)\n", callback_name.c_str());
	print_invalid(os, 4, "cannot release object with persistent callbacks",
			    "return nullptr");
}

/* Print implementation of ptr() functions for class "clazz" to "os".
 * Since type based subclasses share the pointer with their superclass,
 * they can also reuse these functions from the superclass.
 *
 * If an object has persistent callbacks set, then the underlying
 * C object pointer cannot be released because it references data
 * in the C++ object.
 */
void cpp_generator::print_ptr_impl(ostream &os, const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();
	set<FunctionDecl *>::const_iterator in;
	const set<FunctionDecl *> &callbacks = clazz.persistent_callbacks;

	if (clazz.is_type_subclass())
		return;

	osprintf(os, "\n");
	osprintf(os, "__isl_give %s *%s::copy() const & {\n", name, cppname);
	osprintf(os, "  return %s_copy(ptr);\n", name);
	osprintf(os, "}\n\n");
	osprintf(os, "__isl_keep %s *%s::get() const {\n", name, cppname);
	osprintf(os, "  return ptr;\n");
	osprintf(os, "}\n\n");
	osprintf(os, "__isl_give %s *%s::release() {\n", name, cppname);
	for (in = callbacks.begin(); in != callbacks.end(); ++in)
		print_check_no_persistent_callback(os, clazz, *in);
	osprintf(os, "  %s *tmp = ptr;\n", name);
	osprintf(os, "  ptr = nullptr;\n");
	osprintf(os, "  return tmp;\n");
	osprintf(os, "}\n\n");
	osprintf(os, "bool %s::is_null() const {\n", cppname);
	osprintf(os, "  return ptr == nullptr;\n");
	osprintf(os, "}\n");
}

/* Print implementations for the "as" and "isa" methods, if "clazz"
 * is a superclass with a type function.
 *
 * "isa" checks whether an object is of a given subclass type.
 * "isa_type" does the same, but gets passed the value of the type field
 * of the subclass as a function argument and the type of this field
 * as a template argument.
 * "as" casts an object to a given subclass type, erroring out
 * if the object is not of the given type.
 *
 * If the input is an invalid object, then these methods raise
 * an exception.
 * If checked bindings are being generated,
 * then an invalid boolean or object is returned instead.
 */
void cpp_generator::print_downcast_impl(ostream &os, const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	if (!clazz.fn_type)
		return;

	osprintf(os, "\n");
	osprintf(os, "template <typename T, typename>\n");
	osprintf(os, "%s %s::isa_type(T subtype) const\n",
		isl_bool2cpp().c_str(), cppname);
	osprintf(os, "{\n");
	osprintf(os, "  if (is_null())\n");
	if (checked)
		osprintf(os, "    return boolean();\n");
	else
		print_throw_NULL_input(os);
	osprintf(os, "  return %s(get()) == subtype;\n",
		clazz.fn_type->getNameAsString().c_str());
	osprintf(os, "}\n");

	osprintf(os, "template <class T>\n");
	osprintf(os, "%s %s::isa() const\n", isl_bool2cpp().c_str(), cppname);
	osprintf(os, "{\n");
	osprintf(os, "  return isa_type<decltype(T::type)>(T::type);\n");
	osprintf(os, "}\n");

	osprintf(os, "template <class T>\n");
	osprintf(os, "T %s::as() const\n", cppname);
	osprintf(os, "{\n");
	if (checked)
		osprintf(os, " if (isa<T>().is_false())\n");
	else
		osprintf(os, " if (!isa<T>())\n");
	print_invalid(os, 4, "not an object of the requested subtype",
		    "return T()");
	osprintf(os, "  return T(copy());\n");
	osprintf(os, "}\n");
}

/* Print the implementation of the ctx method.
 */
void cpp_generator::print_ctx_impl(ostream &os, const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();
	std::string ns = isl_namespace();

	osprintf(os, "\n");
	osprintf(os, "%sctx %s::ctx() const {\n", ns.c_str(), cppname);
	osprintf(os, "  return %sctx(%s_get_ctx(ptr));\n", ns.c_str(), name);
	osprintf(os, "}\n");
}

/* Print the implementations of the methods needed for the persistent callbacks
 * of "clazz".
 */
void cpp_generator::print_persistent_callbacks_impl(ostream &os,
	const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();
	string classname = type2cpp(clazz);
	set<FunctionDecl *>::const_iterator in;
	const set<FunctionDecl *> &callbacks = clazz.persistent_callbacks;

	if (!clazz.has_persistent_callbacks())
		return;

	osprintf(os, "\n");
	osprintf(os, "%s &%s::copy_callbacks(const %s &obj)\n",
		cppname, classname.c_str(), cppname);
	osprintf(os, "{\n");
	for (in = callbacks.begin(); in != callbacks.end(); ++in) {
		string callback_name = clazz.persistent_callback_name(*in);

		osprintf(os, "  %s_data = obj.%s_data;\n",
			callback_name.c_str(), callback_name.c_str());
	}
	osprintf(os, "  return *this;\n");
	osprintf(os, "}\n");

	for (in = callbacks.begin(); in != callbacks.end(); ++in) {
		function_kind kind = function_kind_member_method;

		print_set_persistent_callback(os, clazz, *in, kind);
	}
}

/* Print definitions for methods of class "clazz" to "os".
 */
void cpp_generator::print_methods_impl(ostream &os, const isl_class &clazz)
{
	map<string, function_set>::const_iterator it;

	for (it = clazz.methods.begin(); it != clazz.methods.end(); ++it)
		print_method_group_impl(os, clazz, it->second);
}

/* Print the definition for a method "method_name" in "clazz" derived
 * from "fd", which sets an enum, to "os".
 * In particular, the method "method_name" sets the enum to "enum_name".
 *
 * The last argument of the C function does not appear in the method call,
 * but is fixed to "enum_name" instead.
 * Other than that, the method printed here is similar to one
 * printed by cpp_generator::print_method_impl, except that
 * some of the special cases do not occur.
 */
void cpp_generator::print_set_enum_impl(ostream &os, const isl_class &clazz,
	FunctionDecl *fd, const string &enum_name, const string &method_name)
{
	string c_name = fd->getName().str();
	int n = fd->getNumParams();
	function_kind kind = function_kind_member_method;

	osprintf(os, "\n");
	print_method_header(os, clazz, fd, method_name, n - 1, false, kind);
	osprintf(os, "{\n");

	print_argument_validity_check(os, fd, kind);
	print_save_ctx(os, fd, kind);
	print_on_error_continue(os);

	osprintf(os, "  auto res = %s(", c_name.c_str());

	for (int i = 0; i < n - 1; ++i) {
		ParmVarDecl *param = fd->getParamDecl(i);

		if (i > 0)
			osprintf(os, ", ");
		print_method_param_use(os, param, i == 0);
	}
	osprintf(os, ", %s", enum_name.c_str());
	osprintf(os, ");\n");

	print_exceptional_execution_check(os, clazz, fd, kind);
	print_method_return(os, clazz, fd);

	osprintf(os, "}\n");
}

/* Print definitions for the methods in "clazz" derived from "fd",
 * which sets an enum, to "os".
 *
 * A method is generated for each value in the enum, setting
 * the enum to that value.
 */
void cpp_generator::print_set_enums_impl(ostream &os, const isl_class &clazz,
	FunctionDecl *fd)
{
	vector<set_enum>::const_iterator it;
	const vector<set_enum> &set_enums = clazz.set_enums.at(fd);

	for (it = set_enums.begin(); it != set_enums.end(); ++it) {
		print_set_enum_impl(os, clazz, fd, it->name, it->method_name);
	}
}

/* Print definitions for methods in "clazz" derived from functions
 * that set an enum, to "os".
 */
void cpp_generator::print_set_enums_impl(ostream &os, const isl_class &clazz)
{
	map<FunctionDecl *, vector<set_enum> >::const_iterator it;

	for (it = clazz.set_enums.begin(); it != clazz.set_enums.end(); ++it)
		print_set_enums_impl(os, clazz, it->first);
}

/* Print a definition for the "get" method "fd" in class "clazz",
 * using a name that includes the "get_" prefix, to "os".
 *
 * This definition simply calls the variant without the "get_" prefix and
 * returns its result.
 * Note that static methods are not considered to be "get" methods.
 */
template<>
void cpp_generator::print_get_method<cpp_generator::impl>(ostream &os,
	const isl_class &clazz, FunctionDecl *fd)
{
	string get_name = clazz.base_method_name(fd);
	string name = clazz.method_name(fd);
	function_kind kind = function_kind_member_method;
	int num_params = fd->getNumParams();

	osprintf(os, "\n");
	print_named_method_header(os, clazz, fd, get_name, false, kind);
	osprintf(os, "{\n");
	osprintf(os, "  return %s(", name.c_str());
	for (int i = 1; i < num_params; ++i) {
		ParmVarDecl *param = fd->getParamDecl(i);

		if (i != 1)
			osprintf(os, ", ");
		osprintf(os, "%s", param->getName().str().c_str());
	}
	osprintf(os, ");\n");
	osprintf(os, "}\n");
}

/* Print definitions for methods "methods" in class "clazz" to "os".
 */
void cpp_generator::print_method_group_impl(ostream &os, const isl_class &clazz,
	const function_set &methods)
{
	function_set::const_iterator it;

	for (it = methods.begin(); it != methods.end(); ++it)
		print_method_variants<impl>(os, clazz, *it);
}

/* Print the use of "param" to "os".
 *
 * "load_from_this_ptr" specifies whether the parameter should be loaded from
 * the this-ptr.  In case a value is loaded from a this pointer, the original
 * value must be preserved and must consequently be copied.  Values that are
 * loaded from parameters do not need to be preserved, as such values will
 * already be copies of the actual parameters.  It is consequently possible
 * to directly take the pointer from these values, which saves
 * an unnecessary copy.
 *
 * In case the parameter is a callback function, two parameters get printed,
 * a wrapper for the callback function and a pointer to the actual
 * callback function.  The wrapper is expected to be available
 * in a previously declared variable <name>_lambda, while
 * the actual callback function is expected to be stored
 * in a structure called <name>_data.
 * The caller of this function must ensure that these variables exist.
 */
void cpp_generator::print_method_param_use(ostream &os, ParmVarDecl *param,
	bool load_from_this_ptr)
{
	string name = param->getName().str();
	const char *name_str = name.c_str();
	QualType type = param->getOriginalType();

	if (type->isIntegerType()) {
		osprintf(os, "%s", name_str);
		return;
	}

	if (is_string(type)) {
		osprintf(os, "%s.c_str()", name_str);
		return;
	}

	if (is_callback(type)) {
		osprintf(os, "%s_lambda, ", name_str);
		osprintf(os, "&%s_data", name_str);
		return;
	}

	if (!load_from_this_ptr)
		osprintf(os, "%s.", name_str);

	if (keeps(param)) {
		osprintf(os, "get()");
	} else {
		if (load_from_this_ptr)
			osprintf(os, "copy()");
		else
			osprintf(os, "release()");
	}
}

/* Print code that checks that all isl object arguments to "method" are valid
 * (not NULL) and throws an exception if they are not.
 * "kind" specifies the kind of method that is being generated.
 *
 * If checked bindings are being generated,
 * then no such check is performed.
 */
void cpp_generator::print_argument_validity_check(ostream &os,
	FunctionDecl *method, function_kind kind)
{
	int n;
	bool first = true;

	if (checked)
		return;

	n = method->getNumParams();
	for (int i = 0; i < n; ++i) {
		bool is_this;
		ParmVarDecl *param = method->getParamDecl(i);
		string name = param->getName().str();
		const char *name_str = name.c_str();
		QualType type = param->getOriginalType();

		is_this = i == 0 && kind == function_kind_member_method;
		if (!is_this && (is_isl_ctx(type) || !is_isl_type(type)))
			continue;

		if (first)
			osprintf(os, "  if (");
		else
			osprintf(os, " || ");

		if (is_this)
			osprintf(os, "!ptr");
		else
			osprintf(os, "%s.is_null()", name_str);

		first = false;
	}
	if (first)
		return;
	osprintf(os, ")\n");
	print_throw_NULL_input(os);
}

/* Print code for saving a copy of the isl::ctx available at the start
 * of the method "method" in a "saved_ctx" variable,
 * for use in exception handling.
 * "kind" specifies what kind of method "method" is.
 *
 * If checked bindings are being generated,
 * then the "saved_ctx" variable is not needed.
 * If "method" is a member function, then obtain the isl_ctx from
 * the "this" object.
 * If the first argument of the method is an isl::ctx, then use that one.
 * Otherwise, save a copy of the isl::ctx associated to the first argument
 * of isl object type.
 */
void cpp_generator::print_save_ctx(ostream &os, FunctionDecl *method,
	function_kind kind)
{
	int n;
	ParmVarDecl *param = method->getParamDecl(0);
	QualType type = param->getOriginalType();

	if (checked)
		return;
	if (kind == function_kind_member_method) {
		osprintf(os, "  auto saved_ctx = ctx();\n");
		return;
	}
	if (is_isl_ctx(type)) {
		const char *name;

		name = param->getName().str().c_str();
		osprintf(os, "  auto saved_ctx = %s;\n", name);
		return;
	}
	n = method->getNumParams();
	for (int i = 0; i < n; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		QualType type = param->getOriginalType();

		if (!is_isl_type(type))
			continue;
		osprintf(os, "  auto saved_ctx = %s.ctx();\n",
			param->getName().str().c_str());
		return;
	}
}

/* Print code to make isl not print an error message when an error occurs
 * within the current scope (if exceptions are available),
 * since the error message will be included in the exception.
 * If exceptions are not available, then exception::on_error
 * is set to ISL_ON_ERROR_ABORT and isl is therefore made to abort instead.
 *
 * If checked bindings are being generated,
 * then leave it to the user to decide what isl should do on error.
 * Otherwise, assume that a valid isl::ctx is available
 * in the "saved_ctx" variable,
 * e.g., through a prior call to print_save_ctx.
 */
void cpp_generator::print_on_error_continue(ostream &os)
{
	if (checked)
		return;
	osprintf(os, "  options_scoped_set_on_error saved_on_error(saved_ctx, "
		     "exception::on_error);\n");
}

/* Print code to "os" that checks whether any of the persistent callbacks
 * of "clazz" is set and if it failed with an exception.  If so, the "eptr"
 * in the corresponding data structure contains the exception
 * that was caught and that needs to be rethrown.
 * This field is cleared because the callback and its data may get reused.
 *
 * The check only needs to be generated for member methods since
 * an object is needed for any of the persistent callbacks to be set.
 */
static void print_persistent_callback_exceptional_execution_check(ostream &os,
	const isl_class &clazz, cpp_generator::function_kind kind)
{
	const set<FunctionDecl *> &callbacks = clazz.persistent_callbacks;
	set<FunctionDecl *>::const_iterator in;

	if (kind != cpp_generator::function_kind_member_method)
		return;

	for (in = callbacks.begin(); in != callbacks.end(); ++in) {
		string callback_name = clazz.persistent_callback_name(*in);

		osprintf(os, "  if (%s_data && %s_data->eptr) {\n",
			callback_name.c_str(), callback_name.c_str());
		osprintf(os, "    std::exception_ptr eptr = %s_data->eptr;\n",
			callback_name.c_str());
		osprintf(os, "    %s_data->eptr = nullptr;\n",
			callback_name.c_str());
		osprintf(os, "    std::rethrow_exception(eptr);\n");
		osprintf(os, "  }\n");
	}
}

/* Print code that checks whether the execution of the core of "method"
 * of class "clazz" was successful.
 * "kind" specifies what kind of method "method" is.
 *
 * If checked bindings are being generated,
 * then no checks are performed.
 *
 * Otherwise, first check if any of the callbacks failed with
 * an exception.  If so, the "eptr" in the corresponding data structure
 * contains the exception that was caught and that needs to be rethrown.
 * Then check if the function call failed in any other way and throw
 * the appropriate exception.
 * In particular, if the return type is isl_stat, isl_bool or isl_size,
 * then a negative value indicates a failure.  If the return type
 * is an isl type, then a NULL value indicates a failure.
 * Assume print_save_ctx has made sure that a valid isl::ctx
 * is available in the "ctx" variable.
 */
void cpp_generator::print_exceptional_execution_check(ostream &os,
	const isl_class &clazz, FunctionDecl *method, function_kind kind)
{
	int n;
	bool check_null, check_neg;
	QualType return_type = method->getReturnType();

	if (checked)
		return;

	print_persistent_callback_exceptional_execution_check(os, clazz, kind);

	n = method->getNumParams();
	for (int i = 0; i < n; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		const char *name;

		if (!is_callback(param->getOriginalType()))
			continue;
		name = param->getName().str().c_str();
		osprintf(os, "  if (%s_data.eptr)\n", name);
		osprintf(os, "    std::rethrow_exception(%s_data.eptr);\n",
			name);
	}

	check_neg = is_isl_neg_error(return_type);
	check_null = is_isl_type(return_type);
	if (!check_null && !check_neg)
		return;

	if (check_neg)
		osprintf(os, "  if (res < 0)\n");
	else
		osprintf(os, "  if (!res)\n");
	print_throw_last_error(os);
}

/* Does "fd" modify an object of a subclass based on a type function?
 */
static bool is_subclass_mutator(const isl_class &clazz, FunctionDecl *fd)
{
	return clazz.is_type_subclass() && generator::is_mutator(clazz, fd);
}

/* Return the C++ return type of the method corresponding to "fd" in "clazz".
 *
 * If "fd" modifies an object of a subclass, then return
 * the type of this subclass.
 * Otherwise, return the C++ counterpart of the actual return type.
 */
std::string cpp_generator::get_return_type(const isl_class &clazz,
	FunctionDecl *fd)
{
	if (is_subclass_mutator(clazz, fd))
		return type2cpp(clazz);
	else
		return type2cpp(fd->getReturnType());
}

/* Given a function "method" for setting a "clazz" persistent callback,
 * print the implementations of the methods needed for that callback.
 *
 * In particular, print
 * - the implementation of a static inline method
 *   for use as the C callback function
 * - the definition of a private method for setting the callback function
 * - the public method for constructing a new object with the callback set.
 */
void cpp_generator::print_set_persistent_callback(ostream &os,
	const isl_class &clazz, FunctionDecl *method,
	function_kind kind)
{
	string fullname = method->getName().str();
	ParmVarDecl *param = persistent_callback_arg(method);
	string classname = type2cpp(clazz);
	string pname;
	string callback_name = clazz.persistent_callback_name(method);

	osprintf(os, "\n");
	print_persistent_callback_prototype(os, clazz, method, false);
	osprintf(os, "\n");
	osprintf(os, "{\n");
	print_callback_body(os, 2, param, callback_name);
	osprintf(os, "}\n\n");

	pname = param->getName().str();
	print_persistent_callback_setter_prototype(os, clazz, method, false);
	osprintf(os, "\n");
	osprintf(os, "{\n");
	print_check_ptr_start(os, clazz, "ptr");
	osprintf(os, "  %s_data = std::make_shared<struct %s_data>();\n",
		callback_name.c_str(), callback_name.c_str());
	osprintf(os, "  %s_data->func = %s;\n",
		callback_name.c_str(), pname.c_str());
	osprintf(os, "  ptr = %s(ptr, &%s, %s_data.get());\n",
		fullname.c_str(), callback_name.c_str(), callback_name.c_str());
	print_check_ptr_end(os, "ptr");
	osprintf(os, "}\n\n");

	print_method_header(os, clazz, method, false, kind);
	osprintf(os, "{\n");
	osprintf(os, "  auto copy = *this;\n");
	osprintf(os, "  copy.set_%s_data(%s);\n",
		callback_name.c_str(), pname.c_str());
	osprintf(os, "  return copy;\n");
	osprintf(os, "}\n");
}

/* Print the return statement of the C++ method corresponding
 * to the C function "method" in class "clazz" to "os".
 *
 * The result of the isl function is returned as a new
 * object if the underlying isl function returns an isl_* ptr, as a bool
 * if the isl function returns an isl_bool, as void if the isl functions
 * returns an isl_stat,
 * as std::string if the isl function returns 'const char *', and as
 * unmodified return value otherwise.
 * If checked C++ bindings are being generated,
 * then an isl_bool return type is transformed into a boolean and
 * an isl_stat into a stat since no exceptions can be generated
 * on negative results from the isl function.
 * If the method returns a new instance of the same object type and
 * if the class has any persistent callbacks, then the data
 * for these callbacks are copied from the original to the new object.
 * If "clazz" is a subclass that is based on a type function and
 * if the return type corresponds to the superclass data type,
 * then it is replaced by the subclass data type.
 */
void cpp_generator::print_method_return(ostream &os, const isl_class &clazz,
	FunctionDecl *method)
{
	QualType return_type = method->getReturnType();
	string rettype_str = get_return_type(clazz, method);
	bool returns_super = is_subclass_mutator(clazz, method);

	if (is_isl_type(return_type) ||
		    (checked && is_isl_neg_error(return_type))) {
		osprintf(os, "  return manage(res)");
		if (is_mutator(clazz, method) &&
		    clazz.has_persistent_callbacks())
			osprintf(os, ".copy_callbacks(*this)");
		if (returns_super)
			osprintf(os, ".as<%s>()", rettype_str.c_str());
		osprintf(os, ";\n");
	} else if (is_isl_stat(return_type)) {
		osprintf(os, "  return;\n");
	} else if (is_string(return_type)) {
		osprintf(os, "  std::string tmp(res);\n");
		if (gives(method))
			osprintf(os, "  free(res);\n");
		osprintf(os, "  return tmp;\n");
	} else {
		osprintf(os, "  return res;\n");
	}
}

/* Return the formal parameter at position "pos" of "fd".
 * However, if this parameter should be converted, as indicated
 * by "convert", then return the second formal parameter
 * of the conversion function instead.
 *
 * If "convert" is empty, then it is assumed that
 * none of the arguments should be converted.
 */
ParmVarDecl *cpp_generator::get_param(FunctionDecl *fd, int pos,
	const std::vector<bool> &convert)
{
	ParmVarDecl *param = fd->getParamDecl(pos);

	if (convert.size() == 0)
		return param;
	if (!convert[pos])
		return param;
	return conversions[param->getOriginalType().getTypePtr()];
}

/* Print the header for "method" in class "clazz", with name "cname" and
 * "num_params" number of arguments, to "os".
 *
 * Print the header of a declaration if "is_declaration" is set, otherwise print
 * the header of a method definition.
 *
 * "kind" specifies the kind of method that should be generated.
 *
 * "convert" specifies which of the method arguments should
 * be automatically converted.
 *
 * This function prints headers for member methods, static methods, and
 * constructors, either for their declaration or definition.
 *
 * Member functions are declared as "const", as they do not change the current
 * object, but instead create a new object. They always retrieve the first
 * parameter of the original isl function from the this-pointer of the object,
 * such that only starting at the second parameter the parameters of the
 * original function become part of the method's interface.
 *
 * A function
 *
 * 	__isl_give isl_set *isl_set_intersect(__isl_take isl_set *s1,
 * 		__isl_take isl_set *s2);
 *
 * is translated into:
 *
 * 	inline set intersect(set set2) const;
 *
 * For static functions and constructors all parameters of the original isl
 * function are exposed.
 *
 * Parameters that are defined as __isl_keep, are of type string or
 * are callbacks, are passed
 * as const reference, which allows the compiler to optimize the parameter
 * transfer.
 *
 * Constructors are marked as explicit using the C++ keyword 'explicit' or as
 * implicit using a comment in place of the explicit keyword. By annotating
 * implicit constructors with a comment, users of the interface are made
 * aware of the potential danger that implicit construction is possible
 * for these constructors, whereas without a comment not every user would
 * know that implicit construction is allowed in absence of an explicit keyword.
 *
 * If any of the arguments needs to be converted, then the argument
 * of the method is changed to that of the source of the conversion.
 * The name of the argument is, however, derived from the original
 * function argument.
 */
void cpp_generator::print_method_header(ostream &os, const isl_class &clazz,
	FunctionDecl *method, const string &cname, int num_params,
	bool is_declaration, function_kind kind,
	const std::vector<bool> &convert)
{
	string rettype_str = get_return_type(clazz, method);
	string classname = type2cpp(clazz);
	int first_param = 0;

	if (kind == function_kind_member_method)
		first_param = 1;

	if (is_declaration) {
		osprintf(os, "  ");

		if (kind == function_kind_static_method)
			osprintf(os, "static ");

		osprintf(os, "inline ");

		if (kind == function_kind_constructor) {
			if (is_implicit_conversion(clazz, method))
				osprintf(os, "/* implicit */ ");
			else
				osprintf(os, "explicit ");
		}
	}

	if (kind != function_kind_constructor)
		osprintf(os, "%s ", rettype_str.c_str());

	if (!is_declaration)
		osprintf(os, "%s::", classname.c_str());

	if (kind != function_kind_constructor)
		osprintf(os, "%s", cname.c_str());
	else
		osprintf(os, "%s", classname.c_str());

	osprintf(os, "(");

	for (int i = first_param; i < num_params; ++i) {
		std::string name = method->getParamDecl(i)->getName().str();
		ParmVarDecl *param = get_param(method, i, convert);
		QualType type = param->getOriginalType();
		string cpptype = type2cpp(type);

		if (is_callback(type))
			num_params--;

		if (keeps(param) || is_string(type) || is_callback(type))
			osprintf(os, "const %s &%s", cpptype.c_str(),
				 name.c_str());
		else
			osprintf(os, "%s %s", cpptype.c_str(), name.c_str());

		if (i != num_params - 1)
			osprintf(os, ", ");
	}

	osprintf(os, ")");

	if (kind == function_kind_member_method)
		osprintf(os, " const");

	if (is_declaration)
		osprintf(os, ";");
	osprintf(os, "\n");
}

/* Print the header for a method called "name" in class "clazz"
 * derived from "method" to "os".
 *
 * Print the header of a declaration if "is_declaration" is set, otherwise print
 * the header of a method definition.
 *
 * "kind" specifies the kind of method that should be generated.
 *
 * "convert" specifies which of the method arguments should
 * be automatically converted.
 */
void cpp_generator::print_named_method_header(ostream &os,
	const isl_class &clazz, FunctionDecl *method, string name,
	bool is_declaration, function_kind kind,
	const std::vector<bool> &convert)
{
	int num_params = method->getNumParams();

	name = rename_method(name);
	print_method_header(os, clazz, method, name, num_params,
			    is_declaration, kind, convert);
}

/* Print the header for "method" in class "clazz" to "os"
 * using its default name.
 *
 * Print the header of a declaration if "is_declaration" is set, otherwise print
 * the header of a method definition.
 *
 * "kind" specifies the kind of method that should be generated.
 */
void cpp_generator::print_method_header(ostream &os, const isl_class &clazz,
	FunctionDecl *method, bool is_declaration, function_kind kind)
{
	string name = clazz.method_name(method);

	print_named_method_header(os, clazz, method, name, is_declaration,
				  kind);
}

/* Generate the list of argument types for a callback function of
 * type "type".  If "cpp" is set, then generate the C++ type list, otherwise
 * the C type list.
 *
 * For a callback of type
 *
 *      isl_stat (*)(__isl_take isl_map *map, void *user)
 *
 * the following C++ argument list is generated:
 *
 *      map
 */
string cpp_generator::generate_callback_args(QualType type, bool cpp)
{
	std::string type_str;
	const FunctionProtoType *callback;
	int num_params;

	callback = extract_prototype(type);
	num_params = callback->getNumArgs();
	if (cpp)
		num_params--;

	for (long i = 0; i < num_params; i++) {
		QualType type = callback->getArgType(i);

		if (cpp)
			type_str += type2cpp(type);
		else
			type_str += type.getAsString();

		if (!cpp)
			type_str += "arg_" + ::to_string(i);

		if (i != num_params - 1)
			type_str += ", ";
	}

	return type_str;
}

/* Generate the full cpp type of a callback function of type "type".
 *
 * For a callback of type
 *
 *      isl_stat (*)(__isl_take isl_map *map, void *user)
 *
 * the following type is generated:
 *
 *      std::function<stat(map)>
 */
string cpp_generator::generate_callback_type(QualType type)
{
	std::string type_str;
	const FunctionProtoType *callback = extract_prototype(type);
	QualType return_type = callback->getReturnType();
	string rettype_str = type2cpp(return_type);

	type_str = "std::function<";
	type_str += rettype_str;
	type_str += "(";
	type_str += generate_callback_args(type, true);
	type_str += ")>";

	return type_str;
}

/* Print the call to the C++ callback function "call",
 * with the given indentation, wrapped
 * for use inside the lambda function that is used as the C callback function,
 * in the case where checked C++ bindings are being generated.
 *
 * In particular, print
 *
 *        auto ret = @call@;
 *        return ret.release();
 */
void cpp_generator::print_wrapped_call_checked(ostream &os, int indent,
	const string &call)
{
	osprintf(os, indent, "auto ret = %s;\n", call.c_str());
	osprintf(os, indent, "return ret.release();\n");
}

/* Print the call to the C++ callback function "call",
 * with the given indentation and with return type "rtype", wrapped
 * for use inside the lambda function that is used as the C callback function.
 *
 * In particular, print
 *
 *        ISL_CPP_TRY {
 *          @call@;
 *          return isl_stat_ok;
 *        } ISL_CPP_CATCH_ALL {
 *          data->eptr = std::current_exception();
 *          return isl_stat_error;
 *        }
 * or
 *        ISL_CPP_TRY {
 *          auto ret = @call@;
 *          return ret ? isl_bool_true : isl_bool_false;
 *        } ISL_CPP_CATCH_ALL {
 *          data->eptr = std::current_exception();
 *          return isl_bool_error;
 *        }
 * or
 *        ISL_CPP_TRY {
 *          auto ret = @call@;
 *          return ret.release();
 *        } ISL_CPP_CATCH_ALL {
 *          data->eptr = std::current_exception();
 *          return NULL;
 *        }
 *
 * depending on the return type.
 *
 * where ISL_CPP_TRY is defined to "try" and ISL_CPP_CATCH_ALL to "catch (...)"
 * (if exceptions are available).
 *
 * If checked C++ bindings are being generated, then
 * the call is wrapped differently.
 */
void cpp_generator::print_wrapped_call(ostream &os, int indent,
	const string &call, QualType rtype)
{
	if (checked)
		return print_wrapped_call_checked(os, indent, call);

	osprintf(os, indent, "ISL_CPP_TRY {\n");
	if (is_isl_stat(rtype))
		osprintf(os, indent, "  %s;\n", call.c_str());
	else
		osprintf(os, indent, "  auto ret = %s;\n", call.c_str());
	if (is_isl_stat(rtype))
		osprintf(os, indent, "  return isl_stat_ok;\n");
	else if (is_isl_bool(rtype))
		osprintf(os, indent,
			"  return ret ? isl_bool_true : isl_bool_false;\n");
	else
		osprintf(os, indent, "  return ret.release();\n");
	osprintf(os, indent, "} ISL_CPP_CATCH_ALL {\n");
	osprintf(os, indent, "  data->eptr = std::current_exception();\n");
	if (is_isl_stat(rtype))
		osprintf(os, indent, "  return isl_stat_error;\n");
	else if (is_isl_bool(rtype))
		osprintf(os, indent, "  return isl_bool_error;\n");
	else
		osprintf(os, indent, "  return NULL;\n");
	osprintf(os, indent, "}\n");
}

/* Print the declaration for a "prefix"_data data structure
 * that can be used for passing to a C callback function
 * containing a copy of the C++ callback function "param",
 * along with an std::exception_ptr that is used to store any
 * exceptions thrown in the C++ callback.
 *
 * If the C callback is of the form
 *
 *      isl_stat (*fn)(__isl_take isl_map *map, void *user)
 *
 * then the following declaration is printed:
 *
 *      struct <prefix>_data {
 *        std::function<stat(map)> func;
 *        std::exception_ptr eptr;
 *      }
 *
 * (without a newline or a semicolon).
 *
 * The std::exception_ptr object is not added to "prefix"_data
 * if checked C++ bindings are being generated.
 */
void cpp_generator::print_callback_data_decl(ostream &os, ParmVarDecl *param,
	const string &prefix)
{
	string cpp_args;

	cpp_args = generate_callback_type(param->getType());

	osprintf(os, "  struct %s_data {\n", prefix.c_str());
	osprintf(os, "    %s func;\n", cpp_args.c_str());
	if (!checked)
		osprintf(os, "    std::exception_ptr eptr;\n");
	osprintf(os, "  }");
}

/* Print the body of C function callback with the given indentation
 * that can be use as an argument to "param" for marshalling
 * the corresponding C++ callback.
 * The data structure that contains the C++ callback is of type
 * "prefix"_data.
 *
 * For a callback of the form
 *
 *      isl_stat (*fn)(__isl_take isl_map *map, void *user)
 *
 * the following code is generated:
 *
 *        auto *data = static_cast<struct <prefix>_data *>(arg_1);
 *        ISL_CPP_TRY {
 *          stat ret = (data->func)(manage(arg_0));
 *          return isl_stat_ok;
 *        } ISL_CPP_CATCH_ALL {
 *          data->eptr = std::current_exception();
 *          return isl_stat_error;
 *        }
 *
 * If checked C++ bindings are being generated, then
 * generate the following code:
 *
 *        auto *data = static_cast<struct <prefix>_data *>(arg_1);
 *        stat ret = (data->func)(manage(arg_0));
 *        return isl_stat(ret);
 */
void cpp_generator::print_callback_body(ostream &os, int indent,
	ParmVarDecl *param, const string &prefix)
{
	QualType ptype, rtype;
	string call, last_idx;
	const FunctionProtoType *callback;
	int num_params;

	ptype = param->getType();

	callback = extract_prototype(ptype);
	rtype = callback->getReturnType();
	num_params = callback->getNumArgs();

	last_idx = ::to_string(num_params - 1);

	call = "(data->func)(";
	for (long i = 0; i < num_params - 1; i++) {
		if (!callback_takes_argument(param, i))
			call += "manage_copy";
		else
			call += "manage";
		call += "(arg_" + ::to_string(i) + ")";
		if (i != num_params - 2)
			call += ", ";
	}
	call += ")";

	osprintf(os, indent,
		 "auto *data = static_cast<struct %s_data *>(arg_%s);\n",
		 prefix.c_str(), last_idx.c_str());
	print_wrapped_call(os, indent, call, rtype);
}

/* Print the local variables that are needed for a callback argument,
 * in particular, print a lambda function that wraps the callback and
 * a pointer to the actual C++ callback function.
 *
 * For a callback of the form
 *
 *      isl_stat (*fn)(__isl_take isl_map *map, void *user)
 *
 * the following lambda function is generated:
 *
 *      auto fn_lambda = [](isl_map *arg_0, void *arg_1) -> isl_stat {
 *        auto *data = static_cast<struct fn_data *>(arg_1);
 *        try {
 *          stat ret = (data->func)(manage(arg_0));
 *          return isl_stat_ok;
 *        } catch (...) {
 *          data->eptr = std::current_exception();
 *          return isl_stat_error;
 *        }
 *      };
 *
 * A copy of the std::function C++ callback function is stored in
 * a fn_data data structure for passing to the C callback function,
 * along with an std::exception_ptr that is used to store any
 * exceptions thrown in the C++ callback.
 *
 *      struct fn_data {
 *        std::function<stat(map)> func;
 *        std::exception_ptr eptr;
 *      } fn_data = { fn };
 *
 * This std::function object represents the actual user
 * callback function together with the locally captured state at the caller.
 *
 * The lambda function is expected to be used as a C callback function
 * where the lambda itself is provided as the function pointer and
 * where the user void pointer is a pointer to fn_data.
 * The std::function object is extracted from the pointer to fn_data
 * inside the lambda function.
 *
 * The std::exception_ptr object is not added to fn_data
 * if checked C++ bindings are being generated.
 * The body of the generated lambda function then is as follows:
 *
 *        stat ret = (data->func)(manage(arg_0));
 *        return isl_stat(ret);
 *
 * If the C callback does not take its arguments, then
 * manage_copy is used instead of manage.
 */
void cpp_generator::print_callback_local(ostream &os, ParmVarDecl *param)
{
	string pname;
	QualType ptype, rtype;
	string c_args, cpp_args, rettype;
	const FunctionProtoType *callback;

	pname = param->getName().str();
	ptype = param->getType();

	c_args = generate_callback_args(ptype, false);

	callback = extract_prototype(ptype);
	rtype = callback->getReturnType();
	rettype = rtype.getAsString();

	print_callback_data_decl(os, param, pname);
	osprintf(os, " %s_data = { %s };\n", pname.c_str(), pname.c_str());
	osprintf(os, "  auto %s_lambda = [](%s) -> %s {\n",
		 pname.c_str(), c_args.c_str(), rettype.c_str());
	print_callback_body(os, 4, param, pname);
	osprintf(os, "  };\n");
}

/* An array listing functions that must be renamed and the function name they
 * should be renamed to. We currently rename functions in case their name would
 * match a reserved C++ keyword, which is not allowed in C++.
 */
static const char *rename_map[][2] = {
	{ "union", "unite" },
};

/* Rename method "name" in case the method name in the C++ bindings should not
 * match the name in the C bindings. We do this for example to avoid
 * C++ keywords.
 */
std::string cpp_generator::rename_method(std::string name)
{
	for (size_t i = 0; i < sizeof(rename_map) / sizeof(rename_map[0]); i++)
		if (name.compare(rename_map[i][0]) == 0)
			return rename_map[i][1];

	return name;
}

/* Translate isl class "clazz" to its corresponding C++ type.
 * Use the name of the type based subclass, if any.
 */
string cpp_generator::type2cpp(const isl_class &clazz)
{
	return type2cpp(clazz.subclass_name);
}

/* Translate type string "type_str" to its C++ name counterpart.
*/
string cpp_generator::type2cpp(string type_str)
{
	return type_str.substr(4);
}

/* Return the C++ counterpart to the isl_bool type.
 * If checked C++ bindings are being generated,
 * then this is "boolean".  Otherwise, it is simply "bool".
 */
string cpp_generator::isl_bool2cpp()
{
	return checked ? "boolean" : "bool";
}

/* Return the namespace of the generated C++ bindings.
 */
string cpp_generator::isl_namespace()
{
	return checked ? "isl::checked::" : "isl::";
}

/* Translate QualType "type" to its C++ name counterpart.
 *
 * An isl_bool return type is translated into "bool",
 * while an isl_stat is translated into "void" and
 * an isl_size is translated to "unsigned".
 * The exceptional cases are handled through exceptions.
 * If checked C++ bindings are being generated, then
 * C++ counterparts of isl_bool, isl_stat and isl_size need to be used instead.
 */
string cpp_generator::type2cpp(QualType type)
{
	if (is_isl_type(type))
		return isl_namespace() +
				type2cpp(type->getPointeeType().getAsString());

	if (is_isl_bool(type))
		return isl_bool2cpp();

	if (is_isl_stat(type))
		return checked ? "stat" : "void";

	if (is_isl_size(type))
		return checked ? "class size" : "unsigned";

	if (type->isIntegerType())
		return type.getAsString();

	if (is_string(type))
		return "std::string";

	if (is_callback(type))
		return generate_callback_type(type);

	die("Cannot convert type to C++ type");
}

/* Check if "subclass_type" is a subclass of "class_type".
 */
bool cpp_generator::is_subclass(QualType subclass_type,
	const isl_class &class_type)
{
	std::string type_str = subclass_type->getPointeeType().getAsString();
	std::vector<std::string> superclasses;
	std::vector<const isl_class *> parents;
	std::vector<std::string>::iterator ci;

	superclasses = generator::find_superclasses(classes[type_str].type);

	for (ci = superclasses.begin(); ci < superclasses.end(); ci++)
		parents.push_back(&classes[*ci]);

	while (!parents.empty()) {
		const isl_class *candidate = parents.back();

		parents.pop_back();

		if (&class_type == candidate)
			return true;

		superclasses = generator::find_superclasses(candidate->type);

		for (ci = superclasses.begin(); ci < superclasses.end(); ci++)
			parents.push_back(&classes[*ci]);
	}

	return false;
}

/* Check if "cons" is an implicit conversion constructor of class "clazz".
 *
 * An implicit conversion constructor is generated in case "cons" has a single
 * parameter, where the parameter type is a subclass of the class that is
 * currently being generated.
 */
bool cpp_generator::is_implicit_conversion(const isl_class &clazz,
	FunctionDecl *cons)
{
	ParmVarDecl *param = cons->getParamDecl(0);
	QualType type = param->getOriginalType();

	int num_params = cons->getNumParams();
	if (num_params != 1)
		return false;

	if (is_isl_type(type) && !is_isl_ctx(type) && is_subclass(type, clazz))
		return true;

	return false;
}

/* Get kind of "method" in "clazz".
 *
 * Given the declaration of a static or member method, returns its kind.
 */
cpp_generator::function_kind cpp_generator::get_method_kind(
	const isl_class &clazz, FunctionDecl *method)
{
	if (is_static(clazz, method))
		return function_kind_static_method;
	else
		return function_kind_member_method;
}
