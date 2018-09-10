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
static void osprintf(ostream &os, const char *format, ...)
{
	va_list arguments;
	char *string_pointer;
	size_t size;

	va_start(arguments, format);
	size = vsnprintf(NULL, 0, format, arguments);
	string_pointer = new char[size + 1];
	va_end(arguments);
	va_start(arguments, format);
	vsnprintf(string_pointer, size + 1, format, arguments);
	va_end(arguments);
	os << string_pointer;
	delete[] string_pointer;
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

/* Print declarations for class "clazz" to "os".
 */
void cpp_generator::print_class(ostream &os, const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "// declarations for isl::%s\n", cppname);

	print_class_factory_decl(os, clazz);
	osprintf(os, "\n");
	osprintf(os, "class %s {\n", cppname);
	print_class_factory_decl(os, clazz, "  friend ");
	osprintf(os, "\n");
	osprintf(os, "  %s *ptr = nullptr;\n", name);
	osprintf(os, "\n");
	print_private_constructors_decl(os, clazz);
	osprintf(os, "\n");
	osprintf(os, "public:\n");
	print_public_constructors_decl(os, clazz);
	print_constructors_decl(os, clazz);
	print_copy_assignment_decl(os, clazz);
	print_destructor_decl(os, clazz);
	print_ptr_decl(os, clazz);
	print_get_ctx_decl(os);
	osprintf(os, "\n");
	print_methods_decl(os, clazz);

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
 */
void cpp_generator::print_class_factory_decl(ostream &os,
	const isl_class &clazz, const std::string &prefix)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	os << prefix;
	osprintf(os, "inline %s manage(__isl_take %s *ptr);\n", cppname, name);
	os << prefix;
	osprintf(os, "inline %s manage_copy(__isl_keep %s *ptr);\n",
		cppname, name);
}

/* Print declarations of private constructors for class "clazz" to "os".
 *
 * Each class has currently one private constructor:
 *
 * 	1) Constructor from a plain isl_* C pointer
 *
 * Example:
 *
 * 	set(__isl_take isl_set *ptr);
 *
 * The raw pointer constructor is kept private. Object creation is only
 * possible through manage() or manage_copy().
 */
void cpp_generator::print_private_constructors_decl(ostream &os,
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
	set<FunctionDecl *>::const_iterator in;
	const set<FunctionDecl *> &constructors = clazz.constructors;

	for (in = constructors.begin(); in != constructors.end(); ++in) {
		FunctionDecl *cons = *in;

		print_method_decl(os, clazz, cons, function_kind_constructor);
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
 */
void cpp_generator::print_destructor_decl(ostream &os, const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "  inline ~%s();\n", cppname);
}

/* Print declaration of pointer functions for class "clazz" to "os".
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

	osprintf(os, "  inline __isl_give %s *copy() const &;\n", name);
	osprintf(os, "  inline __isl_give %s *copy() && = delete;\n", name);
	osprintf(os, "  inline __isl_keep %s *get() const;\n", name);
	osprintf(os, "  inline __isl_give %s *release();\n", name);
	osprintf(os, "  inline bool is_null() const;\n");
}

/* Print the declaration of the get_ctx method.
 */
void cpp_generator::print_get_ctx_decl(ostream &os)
{
	osprintf(os, "  inline ctx get_ctx() const;\n");
}

/* Print declarations for methods in class "clazz" to "os".
 */
void cpp_generator::print_methods_decl(ostream &os, const isl_class &clazz)
{
	map<string, set<FunctionDecl *> >::const_iterator it;

	for (it = clazz.methods.begin(); it != clazz.methods.end(); ++it)
		print_method_group_decl(os, clazz, it->second);
}

/* Print declarations for methods "methods" in class "clazz" to "os".
 */
void cpp_generator::print_method_group_decl(ostream &os, const isl_class &clazz,
	const set<FunctionDecl *> &methods)
{
	set<FunctionDecl *>::const_iterator it;

	for (it = methods.begin(); it != methods.end(); ++it) {
		function_kind kind = get_method_kind(clazz, *it);
		print_method_decl(os, clazz, *it, kind);
	}
}

/* Print declarations for "method" in class "clazz" to "os".
 *
 * "kind" specifies the kind of method that should be generated.
 */
void cpp_generator::print_method_decl(ostream &os, const isl_class &clazz,
	FunctionDecl *method, function_kind kind)
{
	print_method_header(os, clazz, method, true, kind);
}

/* Print implementations for class "clazz" to "os".
 */
void cpp_generator::print_class_impl(ostream &os, const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "// implementations for isl::%s\n", cppname);

	print_class_factory_impl(os, clazz);
	osprintf(os, "\n");
	print_public_constructors_impl(os, clazz);
	osprintf(os, "\n");
	print_private_constructors_impl(os, clazz);
	osprintf(os, "\n");
	print_constructors_impl(os, clazz);
	osprintf(os, "\n");
	print_copy_assignment_impl(os, clazz);
	osprintf(os, "\n");
	print_destructor_impl(os, clazz);
	osprintf(os, "\n");
	print_ptr_impl(os, clazz);
	osprintf(os, "\n");
	print_get_ctx_impl(os, clazz);
	osprintf(os, "\n");
	print_methods_impl(os, clazz);
}

/* Print code for throwing an exception corresponding to the last error
 * that occurred on "ctx".
 * This assumes that a valid isl::ctx is available in the "ctx" variable,
 * e.g., through a prior call to print_save_ctx.
 */
static void print_throw_last_error(ostream &os)
{
	osprintf(os, "    exception::throw_last_error(ctx);\n");
}

/* Print code for throwing an exception on NULL input.
 */
static void print_throw_NULL_input(ostream &os)
{
	osprintf(os, "    exception::throw_NULL_input(__FILE__, __LINE__);\n");
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
	osprintf(os, "  auto ctx = %s_get_ctx(%s);\n", clazz.name.c_str(), ptr);
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
 */
void cpp_generator::print_class_factory_impl(ostream &os,
	const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

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

/* Print implementations of private constructors for class "clazz" to "os".
 */
void cpp_generator::print_private_constructors_impl(ostream &os,
	const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "%s::%s(__isl_take %s *ptr)\n    : ptr(ptr) {}\n",
		 cppname, cppname, name);
}

/* Print implementations of public constructors for class "clazz" to "os".
 *
 * Throw an exception from the copy constructor if anything went wrong
 * during the copying or if the input is NULL.
 * During the copying, isl is made not to print any error message
 * because the error message is included in the exception.
 * No exceptions are thrown if checked C++ bindings
 * are being generated,
 */
void cpp_generator::print_public_constructors_impl(ostream &os,
	const isl_class &clazz)
{
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "%s::%s()\n    : ptr(nullptr) {}\n\n", cppname, cppname);
	osprintf(os, "%s::%s(const %s &obj)\n    : ptr(nullptr)\n",
		 cppname, cppname, cppname);
	osprintf(os, "{\n");
	print_check_ptr_start(os, clazz, "obj.ptr");
	osprintf(os, "  ptr = obj.copy();\n");
	print_check_ptr_end(os, "ptr");
	osprintf(os, "}\n");
}

/* Print implementations of constructors for class "clazz" to "os".
 */
void cpp_generator::print_constructors_impl(ostream &os,
       const isl_class &clazz)
{
	set<FunctionDecl *>::const_iterator in;
	const set<FunctionDecl *> constructors = clazz.constructors;

	for (in = constructors.begin(); in != constructors.end(); ++in) {
		FunctionDecl *cons = *in;

		print_method_impl(os, clazz, cons, function_kind_constructor);
	}
}

/* Print implementation of copy assignment operator for class "clazz" to "os".
 */
void cpp_generator::print_copy_assignment_impl(ostream &os,
	const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "%s &%s::operator=(%s obj) {\n", cppname,
		 cppname, cppname);
	osprintf(os, "  std::swap(this->ptr, obj.ptr);\n", name);
	osprintf(os, "  return *this;\n");
	osprintf(os, "}\n");
}

/* Print implementation of destructor for class "clazz" to "os".
 */
void cpp_generator::print_destructor_impl(ostream &os,
	const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "%s::~%s() {\n", cppname, cppname);
	osprintf(os, "  if (ptr)\n");
	osprintf(os, "    %s_free(ptr);\n", name);
	osprintf(os, "}\n");
}

/* Print implementation of ptr() functions for class "clazz" to "os".
 */
void cpp_generator::print_ptr_impl(ostream &os, const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "__isl_give %s *%s::copy() const & {\n", name, cppname);
	osprintf(os, "  return %s_copy(ptr);\n", name);
	osprintf(os, "}\n\n");
	osprintf(os, "__isl_keep %s *%s::get() const {\n", name, cppname);
	osprintf(os, "  return ptr;\n");
	osprintf(os, "}\n\n");
	osprintf(os, "__isl_give %s *%s::release() {\n", name, cppname);
	osprintf(os, "  %s *tmp = ptr;\n", name);
	osprintf(os, "  ptr = nullptr;\n");
	osprintf(os, "  return tmp;\n");
	osprintf(os, "}\n\n");
	osprintf(os, "bool %s::is_null() const {\n", cppname);
	osprintf(os, "  return ptr == nullptr;\n");
	osprintf(os, "}\n");
}

/* Print the implementation of the get_ctx method.
 */
void cpp_generator::print_get_ctx_impl(ostream &os, const isl_class &clazz)
{
	const char *name = clazz.name.c_str();
	std::string cppstring = type2cpp(clazz);
	const char *cppname = cppstring.c_str();

	osprintf(os, "ctx %s::get_ctx() const {\n", cppname);
	osprintf(os, "  return ctx(%s_get_ctx(ptr));\n", name);
	osprintf(os, "}\n");
}

/* Print definitions for methods of class "clazz" to "os".
 */
void cpp_generator::print_methods_impl(ostream &os, const isl_class &clazz)
{
	map<string, set<FunctionDecl *> >::const_iterator it;
	bool first = true;

	for (it = clazz.methods.begin(); it != clazz.methods.end(); ++it) {
		if (first)
			first = false;
		else
			osprintf(os, "\n");
		print_method_group_impl(os, clazz, it->second);
	}
}

/* Print definitions for methods "methods" in class "clazz" to "os".
 *
 * "kind" specifies the kind of method that should be generated.
 */
void cpp_generator::print_method_group_impl(ostream &os, const isl_class &clazz,
	const set<FunctionDecl *> &methods)
{
	set<FunctionDecl *>::const_iterator it;
	bool first = true;

	for (it = methods.begin(); it != methods.end(); ++it) {
		function_kind kind;
		if (first)
			first = false;
		else
			osprintf(os, "\n");
		kind = get_method_kind(clazz, *it);
		print_method_impl(os, clazz, *it, kind);
	}
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

	if (!load_from_this_ptr && !is_callback(type))
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
 * of the method "method" in a "ctx" variable, for use in exception handling.
 * "kind" specifies what kind of method "method" is.
 *
 * If checked bindings are being generated,
 * then the "ctx" variable is not needed.
 * If "method" is a member function, then obtain the isl_ctx from
 * the "this" object.
 * If the first argument of the method is an isl::ctx, then use that one,
 * assuming it is not already called "ctx".
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
		osprintf(os, "  auto ctx = get_ctx();\n");
		return;
	}
	if (is_isl_ctx(type)) {
		const char *name;

		name = param->getName().str().c_str();
		if (strcmp(name, "ctx") != 0)
			osprintf(os, "  auto ctx = %s;\n", name);
		return;
	}
	n = method->getNumParams();
	for (int i = 0; i < n; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		QualType type = param->getOriginalType();

		if (!is_isl_type(type))
			continue;
		osprintf(os, "  auto ctx = %s.get_ctx();\n",
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
 * Otherwise, assume that a valid isl::ctx is available in the "ctx" variable,
 * e.g., through a prior call to print_save_ctx.
 */
void cpp_generator::print_on_error_continue(ostream &os)
{
	if (checked)
		return;
	osprintf(os, "  options_scoped_set_on_error saved_on_error(ctx, "
		     "exception::on_error);\n");
}

/* Print code that checks whether the execution of the core of "method"
 * was successful.
 *
 * If checked bindings are being generated,
 * then no checks are performed.
 *
 * Otherwise, first check if any of the callbacks failed with
 * an exception.  If so, the "eptr" in the corresponding data structure
 * contains the exception that was caught and that needs to be rethrown.
 * Then check if the function call failed in any other way and throw
 * the appropriate exception.
 * In particular, if the return type is isl_stat or isl_bool,
 * then a negative value indicates a failure.  If the return type
 * is an isl type, then a NULL value indicates a failure.
 * Assume print_save_ctx has made sure that a valid isl::ctx
 * is available in the "ctx" variable.
 */
void cpp_generator::print_exceptional_execution_check(ostream &os,
	FunctionDecl *method)
{
	int n;
	bool check_null, check_neg;
	QualType return_type = method->getReturnType();

	if (checked)
		return;

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

	check_neg = is_isl_stat(return_type) || is_isl_bool(return_type);
	check_null = is_isl_type(return_type);
	if (!check_null && !check_neg)
		return;

	if (check_neg)
		osprintf(os, "  if (res < 0)\n");
	else
		osprintf(os, "  if (!res)\n");
	print_throw_last_error(os);
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
 */
void cpp_generator::print_method_return(ostream &os, const isl_class &clazz,
	FunctionDecl *method)
{
	QualType return_type = method->getReturnType();

	if (is_isl_type(return_type) ||
		    (checked &&
		     (is_isl_bool(return_type) || is_isl_stat(return_type)))) {
		osprintf(os, "  return manage(res);\n");
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

/* Print definition for "method" in class "clazz" to "os".
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
void cpp_generator::print_method_impl(ostream &os, const isl_class &clazz,
	FunctionDecl *method, function_kind kind)
{
	string methodname = method->getName();
	int num_params = method->getNumParams();

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

	print_exceptional_execution_check(os, method);
	if (kind == function_kind_constructor) {
		osprintf(os, "  ptr = res;\n");
	} else {
		print_method_return(os, clazz, method);
	}

	osprintf(os, "}\n");
}

/* Print the header for "method" in class "clazz" to "os".
 *
 * Print the header of a declaration if "is_declaration" is set, otherwise print
 * the header of a method definition.
 *
 * "kind" specifies the kind of method that should be generated.
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
 * Parameters that are defined as __isl_keep or are of type string, are passed
 * as const reference, which allows the compiler to optimize the parameter
 * transfer.
 *
 * Constructors are marked as explicit using the C++ keyword 'explicit' or as
 * implicit using a comment in place of the explicit keyword. By annotating
 * implicit constructors with a comment, users of the interface are made
 * aware of the potential danger that implicit construction is possible
 * for these constructors, whereas without a comment not every user would
 * know that implicit construction is allowed in absence of an explicit keyword.
 */
void cpp_generator::print_method_header(ostream &os, const isl_class &clazz,
	FunctionDecl *method, bool is_declaration, function_kind kind)
{
	string cname = clazz.method_name(method);
	string rettype_str = type2cpp(method->getReturnType());
	string classname = type2cpp(clazz);
	int num_params = method->getNumParams();
	int first_param = 0;

	cname = rename_method(cname);
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
		ParmVarDecl *param = method->getParamDecl(i);
		QualType type = param->getOriginalType();
		string cpptype = type2cpp(type);

		if (is_callback(type))
			num_params--;

		if (keeps(param) || is_string(type) || is_callback(type))
			osprintf(os, "const %s &%s", cpptype.c_str(),
				 param->getName().str().c_str());
		else
			osprintf(os, "%s %s", cpptype.c_str(),
				 param->getName().str().c_str());

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

/* Print the call to the C++ callback function "call", wrapped
 * for use inside the lambda function that is used as the C callback function,
 * in the case where checked C++ bindings are being generated.
 *
 * In particular, print
 *
 *        stat ret = @call@;
 *        return ret.release();
 */
void cpp_generator::print_wrapped_call_checked(ostream &os,
	const string &call)
{
	osprintf(os, "    stat ret = %s;\n", call.c_str());
	osprintf(os, "    return ret.release();\n");
}

/* Print the call to the C++ callback function "call", wrapped
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
 *
 * where ISL_CPP_TRY is defined to "try" and ISL_CPP_CATCH_ALL to "catch (...)"
 * (if exceptions are available).
 *
 * If checked C++ bindings are being generated, then
 * the call is wrapped differently.
 */
void cpp_generator::print_wrapped_call(ostream &os, const string &call)
{
	if (checked)
		return print_wrapped_call_checked(os, call);

	osprintf(os, "    ISL_CPP_TRY {\n");
	osprintf(os, "      %s;\n", call.c_str());
	osprintf(os, "      return isl_stat_ok;\n");
	osprintf(os, "    } ISL_CPP_CATCH_ALL {\n"
		     "      data->eptr = std::current_exception();\n");
	osprintf(os, "      return isl_stat_error;\n");
	osprintf(os, "    }\n");
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
 *          stat ret = (*data->func)(manage(arg_0));
 *          return isl_stat_ok;
 *        } catch (...) {
 *          data->eptr = std::current_exception();
 *          return isl_stat_error;
 *        }
 *      };
 *
 * The pointer to the std::function C++ callback function is stored in
 * a fn_data data structure for passing to the C callback function,
 * along with an std::exception_ptr that is used to store any
 * exceptions thrown in the C++ callback.
 *
 *      struct fn_data {
 *        const std::function<stat(map)> *func;
 *        std::exception_ptr eptr;
 *      } fn_data = { &fn };
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
 *        stat ret = (*data->func)(manage(arg_0));
 *        return isl_stat(ret);
 *
 * If the C callback does not take its arguments, then
 * manage_copy is used instead of manage.
 */
void cpp_generator::print_callback_local(ostream &os, ParmVarDecl *param)
{
	string pname;
	QualType ptype;
	string call, c_args, cpp_args, rettype, last_idx;
	const FunctionProtoType *callback;
	int num_params;

	pname = param->getName().str();
	ptype = param->getType();

	c_args = generate_callback_args(ptype, false);
	cpp_args = generate_callback_type(ptype);

	callback = extract_prototype(ptype);
	rettype = callback->getReturnType().getAsString();
	num_params = callback->getNumArgs();

	last_idx = ::to_string(num_params - 1);

	call = "(*data->func)(";
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

	osprintf(os, "  struct %s_data {\n", pname.c_str());
	osprintf(os, "    const %s *func;\n", cpp_args.c_str());
	if (!checked)
		osprintf(os, "    std::exception_ptr eptr;\n");
	osprintf(os, "  } %s_data = { &%s };\n", pname.c_str(), pname.c_str());
	osprintf(os, "  auto %s_lambda = [](%s) -> %s {\n",
		 pname.c_str(), c_args.c_str(), rettype.c_str());
	osprintf(os,
		 "    auto *data = static_cast<struct %s_data *>(arg_%s);\n",
		 pname.c_str(), last_idx.c_str());
	print_wrapped_call(os, call);
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
 */
string cpp_generator::type2cpp(const isl_class &clazz)
{
	return type2cpp(clazz.name);
}

/* Translate type string "type_str" to its C++ name counterpart.
*/
string cpp_generator::type2cpp(string type_str)
{
	return type_str.substr(4);
}

/* Translate QualType "type" to its C++ name counterpart.
 *
 * An isl_bool return type is translated into "bool",
 * while an isl_stat is translated into "void".
 * The exceptional cases are handled through exceptions.
 * If checked C++ bindings are being generated, then
 * C++ counterparts of isl_bool and isl_stat need to be used instead.
 */
string cpp_generator::type2cpp(QualType type)
{
	if (is_isl_type(type))
		return type2cpp(type->getPointeeType().getAsString());

	if (is_isl_bool(type))
		return checked ? "boolean" : "bool";

	if (is_isl_stat(type))
		return checked ? "stat" : "void";

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
