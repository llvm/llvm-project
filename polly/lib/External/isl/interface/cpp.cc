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
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL TOBIAS GROSSER OR
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

#include <iostream>
#include <string>
#include <vector>

#include "cpp.h"
#include "isl_config.h"

/* Determine the isl types from which the given class can be implicitly
 * constructed using a unary constructor.
 *
 * Look through all constructors for implicit conversion constructors that take
 * an isl type and add those types, along with the corresponding
 * constructor argument.
 */
void cpp_generator::set_class_construction_types(isl_class &clazz)
{
	for (const auto &cons : clazz.constructors) {
		ParmVarDecl *param;
		QualType type;
		std::string arg_type;

		if (!is_implicit_conversion(Method(clazz, cons)))
			continue;

		param = cons->getParamDecl(0);
		type = param->getOriginalType();
		arg_type = extract_type(type);
		clazz.construction_types.emplace(arg_type, param);
	}
}

/* Determine the isl types from which any (proper) class can be constructed
 * using a unary constructor.
 */
void cpp_generator::set_construction_types()
{
	for (auto &kvp : classes) {
		auto &clazz = kvp.second;
		set_class_construction_types(clazz);
	}
}

/* Construct a generator for C++ bindings.
 *
 * The classes and methods are extracted by the constructor
 * of the generator superclass.
 *
 * Additionally extract information about types
 * that can be converted to a class and copy all methods
 * from superclasses that can be converted to a given class
 * to that class.
 */
cpp_generator::cpp_generator(SourceManager &SM,
	set<RecordDecl *> &exported_types,
	set<FunctionDecl *> exported_functions, set<FunctionDecl *> functions) :
		generator(SM, exported_types, exported_functions, functions)
{
	set_construction_types();
	copy_super_methods();
}

/* Copy the method called "name" described by "fd" from "super" to "clazz"
 * with the distance to the original ancestor given by "depth".
 *
 * In particular, keep track of "fd" as well as the superclass
 * from which it was copied and the distance to the original ancestor.
 */
static void copy_method(isl_class &clazz, const isl_class &super,
	const std::string &name, FunctionDecl *fd, int depth)
{
	clazz.methods[name].insert(fd);
	clazz.copied_from.emplace(fd, super);
	clazz.copy_depth.emplace(fd, depth);
}

/* Do "fd1" and "fd2" have the same signature (ignoring the first argument
 * which represents the object class on which the corresponding method
 * gets called).
 */
static bool same_signature(FunctionDecl *fd1, FunctionDecl *fd2)
{
	int n1 = fd1->getNumParams();
	int n2 = fd2->getNumParams();

	if (n1 != n2)
		return false;

	for (int i = 1; i < n1; ++i) {
		ParmVarDecl *p1 = fd1->getParamDecl(i);
		ParmVarDecl *p2 = fd2->getParamDecl(i);

		if (p1->getOriginalType() != p2->getOriginalType())
			return false;
	}

	return true;
}

/* Return the distance between "clazz" and the ancestor
 * from which "fd" got copied.
 * If no distance was recorded, then the method has not been copied
 * but appears in "clazz" itself and so the distance is zero.
 */
static int copy_depth(const isl_class &clazz, FunctionDecl *fd)
{
	if (clazz.copy_depth.count(fd) == 0)
		return 0;
	return clazz.copy_depth.at(fd);
}

/* Is the method derived from "fd", with method name "name" and
 * with distance to the original ancestor "depth",
 * overridden by a method already in "clazz"?
 *
 * A method is considered to have been overridden if there
 * is a method with the same name in "clazz" that has the same signature and
 * that comes from an ancestor closer to "clazz",
 * where an ancestor is closer if the distance in the class hierarchy
 * is smaller or the distance is the same and the ancestor appears
 * closer in the declaration of the type (in which case it gets added first).
 *
 * If a method with the same signature has already been added,
 * but it does not override the method derived from "fd",
 * then this method is removed since it is overridden by "fd".
 */
static bool is_overridden(FunctionDecl *fd, isl_class &clazz,
	const std::string &name, int depth)
{
	if (clazz.methods.count(name) == 0)
		return false;

	for (const auto &m : clazz.methods.at(name)) {
		if (!same_signature(fd, m))
			continue;
		if (copy_depth(clazz, m) <= depth)
			return true;
		clazz.methods[name].erase(m);
		return false;
	}
	return false;
}

/* Add the methods "methods" with method name "name" from "super" to "clazz"
 * provided they have not been overridden by a method already in "clazz".
 *
 * Methods that are static in their original class are not copied.
 */
void cpp_generator::copy_methods(isl_class &clazz, const std::string &name,
	const isl_class &super, const function_set &methods)
{
	for (auto fd : methods) {
		int depth;

		if (method2class(fd)->is_static(fd))
			continue;
		depth = copy_depth(super, fd) + 1;
		if (is_overridden(fd, clazz, name, depth))
			continue;
		copy_method(clazz, super, name, fd, depth);
	}
}

/* Add all methods from "super" to "clazz" that have not been overridden
 * by a method already in "clazz".
 *
 * Look through all groups of methods with the same name.
 */
void cpp_generator::copy_super_methods(isl_class &clazz, const isl_class &super)
{
	for (const auto &kvp : super.methods) {
		const auto &name = kvp.first;
		const auto &methods = kvp.second;

		copy_methods(clazz, name, super, methods);
	}
}

/* Copy methods from the superclasses of "clazz"
 * if an object of this class can be implicitly converted to an object
 * from the superclass, keeping track
 * of the classes that have already been handled in "done".
 *
 * Make sure the superclasses have copied methods from their superclasses first
 * since those methods could be copied further down to this class.
 *
 * Consider the superclass that appears closest to the subclass first.
 */
void cpp_generator::copy_super_methods(isl_class &clazz, set<string> &done)
{
	auto supers = find_superclasses(clazz.type);

	for (const auto &super : supers)
		if (done.count(super) == 0)
			copy_super_methods(classes[super], done);
	done.insert(clazz.name);

	for (const auto &super_name : supers) {
		const auto &super = classes[super_name];

		if (super.construction_types.count(clazz.name) == 0)
			continue;
		copy_super_methods(clazz, super);
	}
}

/* For each (proper) class, copy methods from its superclasses,
 * if an object from the class can be converted to an object
 * from the superclass.
 *
 * Type based subclasses are not considered for now since
 * they do not have any explicit superclasses.
 *
 * Iterate through all (proper) classes and copy methods
 * from their superclasses,
 * unless they have already been determined by a recursive call.
 */
void cpp_generator::copy_super_methods()
{
	set<string> done;

	for (auto &kvp : classes) {
		auto &clazz = kvp.second;

		if (clazz.is_type_subclass())
			continue;
		if (done.count(clazz.name) != 0)
			continue;
		copy_super_methods(clazz, done);
	}
}

/* Print declarations or implementations of constructors.
 *
 * For each isl function that is marked as __isl_constructor,
 * add a corresponding C++ constructor.
 *
 * Example of declarations:
 *
 * 	inline /\* implicit *\/ union_set(basic_set bset);
 * 	inline /\* implicit *\/ union_set(set set);
 * 	inline explicit val(ctx ctx, long i);
 * 	inline explicit val(ctx ctx, const std::string &str);
 */
void cpp_generator::class_printer::print_constructors()
{
	for (const auto &cons : clazz.constructors)
		print_method(Method(clazz, cons));
}

/* Print declarations or definitions for methods in the class.
 */
void cpp_generator::class_printer::print_methods()
{
	for (const auto &kvp : clazz.methods)
		print_method_group(kvp.second, kvp.first);
}

/* Print declarations or implementations for the methods derived from "fd",
 * which sets an enum.
 *
 * A method is generated for each value in the enum, setting
 * the enum to that value.
 */
void cpp_generator::class_printer::print_set_enums(FunctionDecl *fd)
{
	for (const auto &set : clazz.set_enums.at(fd)) {
		EnumMethod method(clazz, fd, set.method_name, set.name);

		print_method(method);
	}
}

/* Print declarations or implementations for methods derived from functions
 * that set an enum.
 */
void cpp_generator::class_printer::print_set_enums()
{
	for (const auto &kvp : clazz.set_enums)
		print_set_enums(kvp.first);
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
bool cpp_generator::class_printer::next_variant(FunctionDecl *fd,
	std::vector<bool> &convert)
{
	size_t n = convert.size();

	for (int i = n - 1; i >= 1; --i) {
		ParmVarDecl *param = fd->getParamDecl(i);
		const Type *type = param->getOriginalType().getTypePtr();

		if (generator.conversions.count(type) == 0)
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

/* Print a declaration or definition for a method called "name"
 * derived from "fd".
 *
 * If the method was copied from a superclass, then print a definition
 * that calls the corresponding method in the superclass.
 * Otherwise, for methods that are identified as "get" methods, also
 * print a declaration or definition for the method
 * using a name that includes the "get_" prefix.
 *
 * If the generated method is an object method, then check
 * whether any of its arguments can be automatically converted
 * from something else, and, if so, generate a method
 * for each combination of converted arguments.
 * Do so by constructing a ConversionMethod that changes the converted arguments
 * to those of the sources of the conversions.
 *
 * Note that a method may be both copied from a superclass and
 * have arguments that can be automatically converted.
 * In this case, the conversion methods for the arguments
 * call the corresponding method in this class, which
 * in turn will call the method in the superclass.
 */
void cpp_generator::class_printer::print_method_variants(FunctionDecl *fd,
	const std::string &name)
{
	Method method(clazz, fd, name);
	std::vector<bool> convert(method.num_params());

	if (method.clazz.copied_from.count(method.fd) == 0) {
		print_method(method);
		if (clazz.is_get_method(fd))
			print_get_method(fd);
	} else {
		auto super = method.clazz.copied_from.at(method.fd);
		print_method(ConversionMethod(method, super.name));
	}
	if (method.kind != Method::Kind::member_method)
		return;
	while (next_variant(fd, convert)) {
		print_method(ConversionMethod(method, [&] (int pos) {
			return get_param(fd, pos, convert);
		}));
	}
}

/* Given a function declaration representing a method,
 * does this method have a single argument (beyond the object
 * on which the method is called) that corresponds to
 * an isl object?
 */
static bool has_single_isl_argument(FunctionDecl *fd)
{
	ParmVarDecl *param;

	if (fd->getNumParams() != 2)
		return false;

	param = fd->getParamDecl(1);
	return generator::is_isl_type(param->getOriginalType());
}

/* Does the set "methods" contain exactly one function declaration
 * that corresponds to a method of "clazz" itself (i.e., that
 * was not copied from an ancestor)?
 */
static FunctionDecl *single_local(const isl_class &clazz,
	const function_set &methods)
{
	int count = 0;
	FunctionDecl *local;

	for (const auto &fn : methods) {
		if (!clazz.first_arg_matches_class(fn))
			continue;
		++count;
		local = fn;
	}

	return count == 1 ? local : NULL;
}

/* Given a function declaration "fd" for a method called "name"
 * with a single argument representing an isl object,
 * generate declarations or definitions for methods with the same name,
 * but with as argument an isl object of a class that can be implicitly
 * converted to that of the original argument.
 * In particular, generate methods for converting this argument.
 */
void cpp_generator::class_printer::print_descendent_overloads(
	FunctionDecl *fd, const std::string &name)
{
	Method method(clazz, fd, name);
	ParmVarDecl *param = fd->getParamDecl(1);
	QualType type = param->getOriginalType();
	std::string arg = type->getPointeeType().getAsString();

	for (const auto &kvp : generator.classes[arg].construction_types) {
		const auto sub = kvp.second;
		print_method(ConversionMethod(method, [&] (int pos) {
			return sub;
		}));
	}
}

/* Print declarations or definitions for methods called "name"
 * derived from "methods".
 *
 * If want_descendent_overloads signals that variants should be added that take
 * as arguments those types that can be converted to the original argument type
 * through a unary constructor and if only one of the methods in the group
 * was originally defined in "clazz", then effectively add those variants.
 * Only do this for methods with a single (isl object) argument.
 */
void cpp_generator::class_printer::print_method_group(
	const function_set &methods, const std::string &name)
{
	FunctionDecl *local;

	for (const auto &fd : methods)
		print_method_variants(fd, name);
	if (!want_descendent_overloads(methods))
		return;
	local = single_local(clazz, methods);
	if (!local)
		return;
	if (!has_single_isl_argument(local))
		return;
	print_descendent_overloads(local, name);
}

/* Print the use of the argument at position "pos" to "os".
 *
 * Member methods pass the isl object corresponding to "this"
 * as first argument (at position 0).
 * Any other arguments are passed along from the method arguments.
 *
 * If the argument value is loaded from a this pointer, the original
 * value must be preserved and must consequently be copied.  Values that are
 * loaded from method parameters do not need to be preserved, as such values
 * will already be copies of the actual parameters.  It is consequently possible
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
void Method::print_param_use(ostream &os, int pos) const
{
	ParmVarDecl *param = fd->getParamDecl(pos);
	bool load_from_this_ptr = pos == 0 && kind == member_method;
	string name = param->getName().str();
	QualType type = param->getOriginalType();

	if (type->isIntegerType()) {
		os << name;
		return;
	}

	if (generator::is_string(type)) {
		os << name << ".c_str()";
		return;
	}

	if (generator::is_callback(type)) {
		os << name << "_lambda, ";
		os << "&" << name << "_data";
		return;
	}

	if (!load_from_this_ptr)
		os << name << ".";

	if (generator::keeps(param)) {
		os << "get()";
	} else {
		if (load_from_this_ptr)
			os << "copy()";
		else
			os << "release()";
	}
}

/* Does the isl function from which this method is derived
 * modify an object of a subclass based on a type function?
 */
bool Method::is_subclass_mutator() const
{
	return clazz.is_type_subclass() && generator::is_mutator(clazz, fd);
}

/* Return the C++ return type of the method "method".
 *
 * If the corresponding function modifies an object of a subclass, then return
 * the type of this subclass.
 * Otherwise, return the C++ counterpart of the actual return type.
 */
std::string cpp_type_printer::return_type(const Method &method) const
{
	if (method.is_subclass_mutator())
		return cpp_generator::type2cpp(method.clazz);
	else
		return param(-1, method.fd->getReturnType());
}

/* Return the formal parameter at position "pos" of "fd".
 * However, if this parameter should be converted, as indicated
 * by "convert", then return the second formal parameter
 * of the conversion function instead.
 */
ParmVarDecl *cpp_generator::class_printer::get_param(FunctionDecl *fd,
	int pos, const std::vector<bool> &convert)
{
	ParmVarDecl *param = fd->getParamDecl(pos);

	if (!convert[pos])
		return param;
	return generator.conversions[param->getOriginalType().getTypePtr()];
}

/* Print the header for "method", without newline or semicolon,
 * using "type_printer" to print argument and return types.
 *
 * Print the header of a declaration if this->declarations is set,
 * otherwise print the header of a method definition.
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
 * 	inline set intersect(set set2) const
 *
 * For static functions and constructors all parameters of the original isl
 * function are exposed.
 *
 * Parameters of which no copy is required, are passed
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
 * Note that in case "method" is a ConversionMethod, the argument returned
 * by Method::get_param may be different from the original argument.
 * The name of the argument is, however, derived from the original
 * function argument.
 */
void cpp_generator::class_printer::print_method_header(
	const Method &method, const cpp_type_printer &type_printer)
{
	string rettype_str = type_printer.return_type(method);

	if (declarations) {
		os << "  ";

		if (method.kind == Method::Kind::static_method)
			os << "static ";

		os << "inline ";

		if (method.kind == Method::Kind::constructor) {
			if (generator.is_implicit_conversion(method))
				os << "/* implicit */ ";
			else
				os << "explicit ";
		}
	}

	if (method.kind != Method::Kind::constructor)
		os << rettype_str << " ";

	if (!declarations)
		os << type_printer.class_type(cppstring) << "::";

	if (method.kind != Method::Kind::constructor)
		os << method.name;
	else
		os << cppstring;

	method.print_cpp_arg_list(os, [&] (int i, int arg) {
		std::string name = method.fd->getParamDecl(i)->getName().str();
		ParmVarDecl *param = method.get_param(i);
		QualType type = param->getOriginalType();
		string cpptype = type_printer.param(arg, type);

		if (!method.param_needs_copy(i))
			os << "const " << cpptype << " &" << name;
		else
			os << cpptype << " " << name;
	});

	if (method.kind == Method::Kind::member_method)
		os << " const";
}

/* Generate the list of argument types for a callback function of
 * type "type", appearing in argument position "arg".
 * If "cpp" is set, then generate the C++ type list, otherwise
 * the C type list.
 *
 * For a callback of type
 *
 *      isl_stat (*)(__isl_take isl_map *map, void *user)
 *
 * the following C++ argument list is generated:
 *
 *      map
 *
 * The arguments of the callback are considered to appear
 * after the position of the callback itself.
 */
std::string cpp_type_printer::generate_callback_args(int arg, QualType type,
	bool cpp) const
{
	std::string type_str;
	const FunctionProtoType *callback;
	int num_params;

	callback = generator::extract_prototype(type);
	num_params = callback->getNumArgs();
	if (cpp)
		num_params--;

	for (long i = 0; i < num_params; i++) {
		QualType type = callback->getArgType(i);

		if (cpp)
			type_str += param(arg + 1 + i, type);
		else
			type_str += type.getAsString();

		if (!cpp)
			type_str += "arg_" + ::to_string(i);

		if (i != num_params - 1)
			type_str += ", ";
	}

	return type_str;
}

/* Generate the full cpp type of a callback function of type "type",
 * appearing in argument position "arg".
 *
 * For a callback of type
 *
 *      isl_stat (*)(__isl_take isl_map *map, void *user)
 *
 * the following type is generated:
 *
 *      std::function<stat(map)>
 */
std::string cpp_type_printer::generate_callback_type(int arg, QualType type)
	const
{
	std::string type_str;
	const FunctionProtoType *callback = generator::extract_prototype(type);
	QualType return_type = callback->getReturnType();
	string rettype_str = param(arg, return_type);

	type_str = "std::function<";
	type_str += rettype_str;
	type_str += "(";
	type_str += generate_callback_args(arg, type, true);
	type_str += ")>";

	return type_str;
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
static std::string rename_method(std::string name)
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
 *
 * By default, this is simply "bool" since
 * the exceptional case is handled through exceptions.
 */
std::string cpp_type_printer::isl_bool() const
{
	return "bool";
}

/* Return the C++ counterpart to the isl_stat type.
 *
 * By default, this is simply "void" since
 * the exceptional case is handled through exceptions.
 */
string cpp_type_printer::isl_stat() const
{
	return "void";
}

/* Return the C++ counterpart to the isl_size type.
 *
 * By default, this is simply "unsigned" since
 * the exceptional case is handled through exceptions.
 */
string cpp_type_printer::isl_size() const
{
	return "unsigned";
}

/* Return the namespace of the generated C++ bindings.
 *
 * By default, this is "isl::".
 */
std::string cpp_type_printer::isl_namespace() const
{
	return "isl::";
}

/* Return the class type given the C++ name.
 *
 * By default, directly use the C++ name.
 */
std::string cpp_type_printer::class_type(const std::string &cpp_name) const
{
	return cpp_name;
}

/* Return the qualified form of the given C++ isl type name appearing
 * in argument position "arg" (-1 for return type).
 *
 * By default, the argument position is ignored.
 */
std::string cpp_type_printer::qualified(int arg, const std::string &cpp_type)
	const
{
	return isl_namespace() + cpp_type;
}

/* Return the C++ counterpart to the given isl type appearing
 * in argument position "arg" (-1 for return type).
 */
std::string cpp_type_printer::isl_type(int arg, QualType type) const
{
	auto name = type->getPointeeType().getAsString();
	return qualified(arg, cpp_generator::type2cpp(name));
}

/* Translate parameter or return type "type" to its C++ name counterpart.
 * "arg" is the position of the argument, or -1 in case of the return type.
 * If any callback is involved, then the return type and arguments types
 * of the callback are considered to start at the position of the callback.
 */
std::string cpp_type_printer::param(int arg, QualType type) const
{
	if (cpp_generator::is_isl_type(type))
		return isl_type(arg, type);

	if (cpp_generator::is_isl_bool(type))
		return isl_bool();

	if (cpp_generator::is_isl_stat(type))
		return isl_stat();

	if (cpp_generator::is_isl_size(type))
		return isl_size();

	if (type->isIntegerType())
		return type.getAsString();

	if (cpp_generator::is_string(type))
		return "std::string";

	if (cpp_generator::is_callback(type))
		return generate_callback_type(arg, type);

	generator::die("Cannot convert type to C++ type");
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
bool cpp_generator::is_implicit_conversion(const Method &cons)
{
	const auto &clazz = cons.clazz;
	ParmVarDecl *param = cons.fd->getParamDecl(0);
	QualType type = param->getOriginalType();

	int num_params = cons.fd->getNumParams();
	if (num_params != 1)
		return false;

	if (is_isl_type(type) && !is_isl_ctx(type) && is_subclass(type, clazz))
		return true;

	return false;
}

/* Construct a list combiner for printing a list.
 */
Method::list_combiner Method::print_combiner(std::ostream &os)
{
	return {
		[&] () { os << "("; },
		[&] () { os << ", "; },
		[&] () { os << ")"; }
	};
}

/* Construct a list combiner for simply iterating over a list.
 */
Method::list_combiner Method::empty_combiner()
{
	return { [&] () { }, [&] () { }, [&] () { } };
}

/* Get kind of "method" in "clazz".
 *
 * Given the declaration of a static or member method, returns its kind.
 */
static Method::Kind get_kind(const isl_class &clazz, FunctionDecl *method)
{
	if (generator::is_constructor(method))
		return Method::Kind::constructor;
	else if (generator::is_static(clazz, method))
		return Method::Kind::static_method;
	else
		return Method::Kind::member_method;
}

/* Return the callback arguments of "fd".
 */
static std::vector<ParmVarDecl *> find_callback_args(FunctionDecl *fd)
{
	std::vector<ParmVarDecl *> callbacks;
	int num_params = fd->getNumParams();

	for (int i = 0; i < num_params; ++i) {
		ParmVarDecl *param = fd->getParamDecl(i);
		if (generator::is_callback(param->getType()))
			callbacks.emplace_back(param);
	}

	return callbacks;
}

/* Construct a C++ method object from the class to which is belongs,
 * the isl function from which it is derived and the method name.
 *
 * Perform any renaming of the method that may be required and
 * determine the type of the method.
 */
Method::Method(const isl_class &clazz, FunctionDecl *fd,
	const std::string &name) :
		clazz(clazz), fd(fd), name(rename_method(name)),
		kind(get_kind(clazz, fd)),
		callbacks(find_callback_args(fd))
{
}

/* Construct a C++ method object from the class to which is belongs and
 * the isl function from which it is derived.
 *
 * Obtain the default method name and continue
 * with the generic constructor.
 */
Method::Method(const isl_class &clazz, FunctionDecl *fd) :
	Method(clazz, fd, clazz.method_name(fd))
{
}

/* Return the number of parameters of the corresponding C function.
 *
 * This number includes any possible user pointers that follow callback
 * arguments.  These are skipped by Method::print_fd_arg_list
 * during the actual argument printing.
 */
int Method::c_num_params() const
{
	return fd->getNumParams();
}

/* Return the number of parameters of the method
 * (including the implicit "this").
 *
 * By default, it is the same as the number of parameters
 * of the corresponding C function.
 */
int Method::num_params() const
{
	return c_num_params();
}

/* Call "on_arg_skip_next" on the arguments from "start" (inclusive)
 * to "end" (exclusive), calling the methods of "combiner"
 * before, between and after the arguments.
 * If "on_arg_skip_next" returns true then the next argument is skipped.
 */
void Method::on_arg_list(int start, int end,
	const Method::list_combiner &combiner,
	const std::function<bool(int i)> &on_arg_skip_next)
{
	combiner.before();
	for (int i = start; i < end; ++i) {
		if (i != start)
			combiner.between();
		if (on_arg_skip_next(i))
			++i;
	}
	combiner.after();
}

/* Print the arguments from "start" (inclusive) to "end" (exclusive)
 * as arguments to a method of C function call, using "print_arg_skip_next"
 * to print each individual argument.  If this callback return true
 * then the next argument is skipped.
 */
void Method::print_arg_list(std::ostream &os, int start, int end,
	const std::function<bool(int i)> &print_arg_skip_next)
{
	on_arg_list(start, end, print_combiner(os), [&] (int i) {
		return print_arg_skip_next(i);
	});
}

/* Call "on_arg" on the arguments from "start" (inclusive) to "end" (exclusive),
 * calling the methods of "combiner" before, between and after the arguments.
 * The first argument to "on_arg" is the position of the argument
 * in this->fd.
 * The second argument is the (first) position in the list of arguments
 * with all callback arguments spliced in.
 *
 * Call on_arg_list to do the actual iteration over the arguments, skipping
 * the user argument that comes after every callback argument.
 * On the C++ side no user pointer is needed, as arguments can be forwarded
 * as part of the std::function argument which specifies the callback function.
 * The user pointer is also removed from the number of parameters
 * of the C function because the pair of callback and user pointer
 * is considered as a single argument that is printed as a whole
 * by Method::print_param_use.
 *
 * In case of a callback argument, the second argument to "print_arg"
 * is also adjusted to account for the spliced-in arguments of the callback.
 * The return value takes the place of the callback itself,
 * while the arguments (excluding the final user pointer)
 * take the following positions.
 */
void Method::on_fd_arg_list(int start, int end,
	const Method::list_combiner &combiner,
	const std::function<void(int i, int arg)> &on_arg) const
{
	int arg = start;

	on_arg_list(start, end, combiner, [this, &on_arg, &arg] (int i) {
		auto type = fd->getParamDecl(i)->getType();

		on_arg(i, arg++);
		if (!generator::is_callback(type))
			return false;
		arg += generator::prototype_n_args(type) - 1;
		return true;
	});
}

/* Print the arguments from "start" (inclusive) to "end" (exclusive)
 * as arguments to a method of C function call, using "print_arg"
 * to print each individual argument.
 * The first argument to this callback is the position of the argument
 * in this->fd.
 * The second argument is the (first) position in the list of arguments
 * with all callback arguments spliced in.
 */
void Method::print_fd_arg_list(std::ostream &os, int start, int end,
	const std::function<void(int i, int arg)> &print_arg) const
{
	on_fd_arg_list(start, end, print_combiner(os), print_arg);
}

/* Call "on_arg" on the arguments to the method call,
 * calling the methods of "combiner" before, between and after the arguments.
 * The first argument to "on_arg" is the position of the argument
 * in this->fd.
 * The second argument is the (first) position in the list of arguments
 * with all callback arguments spliced in.
 */
void Method::on_cpp_arg_list(const Method::list_combiner &combiner,
	const std::function<void(int i, int arg)> &on_arg) const
{
	int first_param = kind == member_method ? 1 : 0;
	on_fd_arg_list(first_param, num_params(), combiner, on_arg);
}

/* Call "on_arg" on the arguments to the method call.
 * The first argument to "on_arg" is the position of the argument
 * in this->fd.
 * The second argument is the (first) position in the list of arguments
 * with all callback arguments spliced in.
 */
void Method::on_cpp_arg_list(
	const std::function<void(int i, int arg)> &on_arg) const
{
	on_cpp_arg_list(empty_combiner(), on_arg);
}

/* Print the arguments to the method call, using "print_arg"
 * to print each individual argument.
 * The first argument to this callback is the position of the argument
 * in this->fd.
 * The second argument is the (first) position in the list of arguments
 * with all callback arguments spliced in.
 */
void Method::print_cpp_arg_list(std::ostream &os,
	const std::function<void(int i, int arg)> &print_arg) const
{
	on_cpp_arg_list(print_combiner(os), print_arg);
}

/* Should the parameter at position "pos" be a copy (rather than
 * a const reference)?
 *
 * Strictly speaking, a copy is only needed on isl types that are
 * not marked __isl_keep, since those will be release()'d
 * by code printed by Method::print_param_use.
 *
 * However, there may be other arguments such as integer types
 * that are more naturally passed as a copy.
 * The default is therefore to require a copy, except for
 * arguments marked __isl_keep, string arguments or callback arguments.
 */
bool Method::param_needs_copy(int pos) const
{
	ParmVarDecl *param = get_param(pos);
	QualType type = param->getOriginalType();

	if (generator::keeps(param))
		return false;
	if (generator::is_string(type) || generator::is_callback(type))
		return false;
	return true;
}

/* Return the method argument at position "pos".
 */
clang::ParmVarDecl *Method::get_param(int pos) const
{
	return fd->getParamDecl(pos);
}

/* Construct a method that performs one or more conversions
 * from the original Method (without conversions),
 * the name of the type to which "this" should be converted and
 * a function for determining the arguments of the constructed method.
 */
ConversionMethod::ConversionMethod(const Method &method,
	const std::string &this_type,
	const std::function<clang::ParmVarDecl *(int pos)> &get_param) :
		NoCopyMethod(method), this_type(this_type),
		get_param_fn(get_param)
{
}

/* Construct a method that only performs a conversion on "this"
 * from the original Method (without conversions) and
 * the name of the type to which "this" should be converted.
 *
 * Call the generic constructor with
 * a function for determining the arguments of the constructed method
 * that performs no conversion.
 */
ConversionMethod::ConversionMethod(const Method &method,
	const std::string &this_type) :
		ConversionMethod(method, this_type, [this] (int pos) {
			return Method::get_param(pos);
		})
{
}

/* Construct a method that performs one or more argument conversions
 * from the original Method (without conversions) and
 * a function for determining the arguments of the constructed method.
 *
 * Call the generic constructor with method.clazz.name as "this" type,
 * indicating that "this" should not be converted.
 */
ConversionMethod::ConversionMethod(const Method &method,
	const std::function<clang::ParmVarDecl *(int pos)> &get_param) :
		ConversionMethod(method, method.clazz.name, get_param)
{
}

/* Should the parameter at position "pos" be a copy (rather than
 * a const reference)?
 *
 * Parameters of isl type do not need to be a copy.
 * For other types, use the same defaults as Method.
 */
bool NoCopyMethod::param_needs_copy(int pos) const
{
	ParmVarDecl *param = get_param(pos);
	QualType type = param->getOriginalType();

	if (generator::is_isl_type(type))
		return false;

	return Method::param_needs_copy(pos);
}

/* Return the method argument at position "pos".
 *
 * Call get_param_fn to determine this argument.
 */
clang::ParmVarDecl *ConversionMethod::get_param(int pos) const
{
	return get_param_fn(pos);
}

/* Print a call to the method (without the arguments),
 * with "ns" the namespace of the generated C++ bindings.
 *
 * If "this_type" is different from the name of the class of the method,
 * then "this" needs to be converted to that type before
 * the call is performed.
 */
void ConversionMethod::print_call(std::ostream &os, const std::string &ns) const
{
	if (clazz.name == this_type) {
		os << "this->";
	} else {
		auto cpp_type = ns + cpp_generator::type2cpp(this_type);
		os << cpp_type << "(*this).";
	}
	os << name;
}

/* Construct an object representing a C++ method for setting an enum
 * from the class to which is belongs,
 * the isl function from which it is derived and the method and enum names.
 */
EnumMethod::EnumMethod(const isl_class &clazz, FunctionDecl *fd,
	const std::string &method_name, const std::string &enum_name) :
		Method(clazz, fd, method_name), enum_name(enum_name)
{
}

/* Print the use of the argument at position "pos" to "os".
 *
 * If the position is beyond the number of method arguments,
 * then it corresponds to the enum value corresponding to this EnumMethod.
 * Otherwise, delegate to Method::print_param_use.
 */
void EnumMethod::print_param_use(ostream &os, int pos) const
{
	if (pos == num_params())
		os << enum_name;
	else
		Method::print_param_use(os, pos);
}

/* Return the number of parameters of the method
 * (including the implicit "this").
 *
 * The last argument of the C function does not appear in the method call,
 * because it is replaced by a break-up into several methods.
 */
int EnumMethod::num_params() const
{
	return Method::num_params() - 1;
}

/* Initialize a class method printer from the stream onto which the methods
 * are printed, the class method description and the C++ interface generator.
 */
cpp_generator::class_printer::class_printer(std::ostream &os,
		const isl_class &clazz, cpp_generator &generator,
		bool declarations) :
	os(os), clazz(clazz), cppstring(type2cpp(clazz)), generator(generator),
	declarations(declarations)
{
}
