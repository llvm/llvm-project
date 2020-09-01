/*
 * Copyright 2011,2015 Sven Verdoolaege. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY SVEN VERDOOLAEGE ''AS IS'' AND ANY
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
 * Sven Verdoolaege.
 */ 

#include "isl_config.h"

#include <stdarg.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include "python.h"
#include "generator.h"

/* Argument format for Python methods with a fixed number of arguments.
 */
static const char *fixed_arg_fmt = "arg%d";
/* Argument format for Python methods with a variable number of arguments.
 */
static const char *var_arg_fmt = "args[%d]";

/* Drop the "isl_" initial part of the type name "name".
 */
static string type2python(string name)
{
	return name.substr(4);
}

/* Print the arguments of a method with "n_arg" arguments, starting at "first".
 */
void python_generator::print_method_arguments(int first, int n_arg)
{
	for (int i = first; i < n_arg; ++i) {
		if (i > first)
			printf(", ");
		printf("arg%d", i);
	}
}

/* Print the start of a definition for method "name"
 * (without specifying the arguments).
 * If "is_static" is set, then mark the python method as static.
 *
 * If the method is called "from", then rename it to "convert_from"
 * because "from" is a python keyword.
 */
static void print_method_def(bool is_static, const string &name)
{
	const char *s;

	if (is_static)
		printf("    @staticmethod\n");

	s = name.c_str();
	if (name == "from")
		s = "convert_from";

	printf("    def %s", s);
}

/* Print the header of the method "name" with "n_arg" arguments.
 * If "is_static" is set, then mark the python method as static.
 */
void python_generator::print_method_header(bool is_static, const string &name,
	int n_arg)
{
	print_method_def(is_static, name);
	printf("(");
	print_method_arguments(0, n_arg);
	printf("):\n");
}

/* Print formatted output with the given indentation.
 */
static void print_indent(int indent, const char *format, ...)
{
	va_list args;

	printf("%*s", indent, " ");
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
}

/* Print a check that the argument in position "pos" is of type "type"
 * with the given indentation.
 * If this fails and if "upcast" is set, then convert the first
 * argument to "super" and call the method "name" on it, passing
 * the remaining of the "n" arguments.
 * If the check fails and "upcast" is not set, then simply raise
 * an exception.
 * If "upcast" is not set, then the "super", "name" and "n" arguments
 * to this function are ignored.
 * "fmt" is the format for printing Python method arguments.
 */
void python_generator::print_type_check(int indent, const string &type,
	const char *fmt, int pos, bool upcast, const string &super,
	const string &name, int n)
{
	print_indent(indent, "try:\n");
	print_indent(indent, "    if not ");
	printf(fmt, pos);
	printf(".__class__ is %s:\n", type.c_str());
	print_indent(indent, "        ");
	printf(fmt, pos);
	printf(" = %s(", type.c_str());
	printf(fmt, pos);
	printf(")\n");
	print_indent(indent, "except:\n");
	if (upcast) {
		print_indent(indent, "    return %s(",
			type2python(super).c_str());
		printf(fmt, 0);
		printf(").%s(", name.c_str());
		for (int i = 1; i < n; ++i) {
			if (i != 1)
				printf(", ");
			printf(fmt, i);
		}
		printf(")\n");
	} else
		print_indent(indent, "    raise\n");
}

/* For each of the "n" initial arguments of the function "method"
 * that refer to an isl structure,
 * including the object on which the method is called,
 * check if the corresponding actual argument is of the right type.
 * If not, try and convert it to the right type.
 * If that doesn't work and if "super" contains at least one element,
 * try and convert self to the type of the first superclass in "super" and
 * call the corresponding method.
 * If "first_is_ctx" is set, then the first argument is skipped.
 */
void python_generator::print_type_checks(const string &cname,
	FunctionDecl *method, bool first_is_ctx, int n,
	const vector<string> &super)
{
	for (int i = first_is_ctx; i < n; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		string type;

		if (!is_isl_type(param->getOriginalType()))
			continue;
		type = type2python(extract_type(param->getOriginalType()));
		if (!first_is_ctx && i > 0 && super.size() > 0)
			print_type_check(8, type, fixed_arg_fmt,
					i - first_is_ctx, true,
					super[0], cname, n);
		else
			print_type_check(8, type, fixed_arg_fmt,
					i - first_is_ctx, false, "", cname, -1);
	}
}

/* Print a call to the *_copy function corresponding to "type".
 */
void python_generator::print_copy(QualType type)
{
	string type_s = extract_type(type);

	printf("isl.%s_copy", type_s.c_str());
}

/* Construct a wrapper for callback argument "param" (at position "arg").
 * Assign the wrapper to "cb".  We assume here that a function call
 * has at most one callback argument.
 *
 * The wrapper converts the arguments of the callback to python types,
 * taking a copy if the C callback does not take its arguments.
 * If any exception is thrown, the wrapper keeps track of it in exc_info[0]
 * and returns a value indicating an error.  Otherwise the wrapper
 * returns a value indicating success.
 * In case the C callback is expected to return an isl_stat,
 * the error value is -1 and the success value is 0.
 * In case the C callback is expected to return an isl_bool,
 * the error value is -1 and the success value is 1 or 0 depending
 * on the result of the Python callback.
 * Otherwise, None is returned to indicate an error and
 * a copy of the object in case of success.
 */
void python_generator::print_callback(ParmVarDecl *param, int arg)
{
	QualType type = param->getOriginalType();
	const FunctionProtoType *fn = extract_prototype(type);
	QualType return_type = fn->getReturnType();
	unsigned n_arg = fn->getNumArgs();

	printf("        exc_info = [None]\n");
	printf("        fn = CFUNCTYPE(");
	if (is_isl_stat(return_type) || is_isl_bool(return_type))
		printf("c_int");
	else
		printf("c_void_p");
	for (unsigned i = 0; i < n_arg - 1; ++i) {
		if (!is_isl_type(fn->getArgType(i)))
			die("Argument has non-isl type");
		printf(", c_void_p");
	}
	printf(", c_void_p)\n");
	printf("        def cb_func(");
	for (unsigned i = 0; i < n_arg; ++i) {
		if (i)
			printf(", ");
		printf("cb_arg%d", i);
	}
	printf("):\n");
	for (unsigned i = 0; i < n_arg - 1; ++i) {
		string arg_type;
		arg_type = type2python(extract_type(fn->getArgType(i)));
		printf("            cb_arg%d = %s(ctx=arg0.ctx, ptr=",
			i, arg_type.c_str());
		if (!callback_takes_argument(param, i))
			print_copy(fn->getArgType(i));
		printf("(cb_arg%d))\n", i);
	}
	printf("            try:\n");
	if (is_isl_stat(return_type))
		printf("                arg%d(", arg);
	else
		printf("                res = arg%d(", arg);
	for (unsigned i = 0; i < n_arg - 1; ++i) {
		if (i)
			printf(", ");
		printf("cb_arg%d", i);
	}
	printf(")\n");
	printf("            except:\n");
	printf("                import sys\n");
	printf("                exc_info[0] = sys.exc_info()\n");
	if (is_isl_stat(return_type) || is_isl_bool(return_type))
		printf("                return -1\n");
	else
		printf("                return None\n");
	if (is_isl_stat(return_type)) {
		printf("            return 0\n");
	} else if (is_isl_bool(return_type)) {
		printf("            return 1 if res else 0\n");
	} else {
		printf("            return ");
		print_copy(return_type);
		printf("(res.ptr)\n");
	}
	printf("        cb = fn(cb_func)\n");
}

/* Print the argument at position "arg" in call to "fd".
 * "fmt" is the format for printing Python method arguments.
 * "skip" is the number of initial arguments of "fd" that are
 * skipped in the Python method.
 *
 * If the (first) argument is an isl_ctx, then print "ctx",
 * assuming that the caller has made the context available
 * in a "ctx" variable.
 * Otherwise, if the argument is a callback, then print a reference to
 * the callback wrapper "cb".
 * Otherwise, if the argument is marked as consuming a reference,
 * then pass a copy of the pointer stored in the corresponding
 * argument passed to the Python method.
 * Otherwise, if the argument is a string, then the python string is first
 * encoded as a byte sequence, using 'ascii' as encoding.  This assumes
 * that all strings passed to isl can be converted to 'ascii'.
 * Otherwise, if the argument is a pointer, then pass this pointer itself.
 * Otherwise, pass the argument directly.
 */
void python_generator::print_arg_in_call(FunctionDecl *fd, const char *fmt,
	int arg, int skip)
{
	ParmVarDecl *param = fd->getParamDecl(arg);
	QualType type = param->getOriginalType();
	if (is_isl_ctx(type)) {
		printf("ctx");
	} else if (is_callback(type)) {
		printf("cb");
	} else if (takes(param)) {
		print_copy(type);
		printf("(");
		printf(fmt, arg - skip);
		printf(".ptr)");
	} else if (is_string(type)) {
		printf(fmt, arg - skip);
		printf(".encode('ascii')");
	} else if (type->isPointerType()) {
		printf(fmt, arg - skip);
		printf(".ptr");
	} else {
		printf(fmt, arg - skip);
	}
}

/* Generate code that raises the exception captured in "exc_info", if any,
 * with the given indentation.
 */
static void print_rethrow(int indent, const char *exc_info)
{
	print_indent(indent, "if %s != None:\n", exc_info);
	print_indent(indent, "    raise (%s[0], %s[1], %s[2])\n",
		exc_info, exc_info, exc_info);
}

/* Print code with the given indentation that checks
 * whether any of the persistent callbacks of "clazz"
 * is set and if it failed with an exception.  If so, the 'exc_info'
 * field contains the exception and is raised again.
 * The field is cleared because the callback and its data may get reused.
 * "fmt" is the format for printing Python method arguments.
 */
static void print_persistent_callback_failure_check(int indent,
	const isl_class &clazz, const char *fmt)
{
	const set<FunctionDecl *> &callbacks = clazz.persistent_callbacks;
	set<FunctionDecl *>::const_iterator in;

	for (in = callbacks.begin(); in != callbacks.end(); ++in) {
		string callback_name = clazz.persistent_callback_name(*in);

		print_indent(indent, "if hasattr(");
		printf(fmt, 0);
		printf(", '%s') and ", callback_name.c_str());
		printf(fmt, 0);
		printf(".%s['exc_info'] != None:\n", callback_name.c_str());
		print_indent(indent, "    exc_info = ");
		printf(fmt, 0);
		printf(".%s['exc_info'][0]\n", callback_name.c_str());
		print_indent(indent, "    ");
		printf(fmt, 0);
		printf(".%s['exc_info'][0] = None\n", callback_name.c_str());
		print_rethrow(indent + 4, "exc_info");
	}
}

/* Print the return statement of the python method corresponding
 * to the C function "method" with the given indentation.
 * If the object on which the method was called
 * may have a persistent callback, then first check if any of those failed.
 * "fmt" is the format for printing Python method arguments.
 *
 * If the method returns a new instance of the same object type and
 * if the class has any persistent callbacks, then the data
 * for these callbacks are copied from the original to the new object.
 * If the method it itself setting a persistent callback,
 * then keep track of the constructed C callback (such that it doesn't
 * get destroyed) and the data structure that holds the captured exception
 * (such that it can be raised again).
 *
 * If the return type is a (const) char *, then convert the result
 * to a Python string, raising an error on NULL and freeing
 * the C string if needed.  For python 3 compatibility, the string returned
 * by isl is explicitly decoded as an 'ascii' string.  This is correct
 * as all strings returned by isl are expected to be 'ascii'.
 *
 * If the return type is isl_stat, isl_bool or isl_size, then
 * raise an error on isl_stat_error, isl_bool_error or isl_size_error.
 * In case of isl_bool, the result is converted to
 * a Python boolean.
 * In case of isl_size, the result is converted to a Python int.
 */
void python_generator::print_method_return(int indent, const isl_class &clazz,
	FunctionDecl *method, const char *fmt)
{
	QualType return_type = method->getReturnType();

	if (!is_static(clazz, method))
		print_persistent_callback_failure_check(indent, clazz, fmt);

	if (is_isl_type(return_type)) {
		string type;

		type = type2python(extract_type(return_type));
		print_indent(indent,
			"obj = %s(ctx=ctx, ptr=res)\n", type.c_str());
		if (is_mutator(clazz, method) &&
		    clazz.has_persistent_callbacks())
			print_indent(indent, "obj.copy_callbacks(arg0)\n");
		if (clazz.persistent_callbacks.count(method)) {
			string callback_name;

			callback_name = clazz.persistent_callback_name(method);
			print_indent(indent, "obj.%s = { 'func': cb, "
				"'exc_info': exc_info }\n",
				callback_name.c_str());
		}
		print_indent(indent, "return obj\n");
	} else if (is_string(return_type)) {
		print_indent(indent, "if res == 0:\n");
		print_indent(indent, "    raise\n");
		print_indent(indent, "string = "
		       "cast(res, c_char_p).value.decode('ascii')\n");

		if (gives(method))
			print_indent(indent, "libc.free(res)\n");

		print_indent(indent, "return string\n");
	} else if (is_isl_neg_error(return_type)) {
		print_indent(indent, "if res < 0:\n");
		print_indent(indent, "    raise\n");
		if (is_isl_bool(return_type))
			print_indent(indent, "return bool(res)\n");
		else if (is_isl_size(return_type))
			print_indent(indent, "return int(res)\n");
	} else {
		print_indent(indent, "return res\n");
	}
}

/* Print a python "get" method corresponding to the C function "fd"
 * in class "clazz" using a name that includes the "get_" prefix.
 *
 * This method simply calls the variant without the "get_" prefix and
 * returns its result.
 * Note that static methods are not considered to be "get" methods.
 */
void python_generator::print_get_method(const isl_class &clazz,
	FunctionDecl *fd)
{
	string get_name = clazz.base_method_name(fd);
	string name = clazz.method_name(fd);
	int num_params = fd->getNumParams();

	print_method_header(false, get_name, num_params);
	printf("        return arg0.%s(", name.c_str());
	print_method_arguments(1, num_params);
	printf(")\n");
}

/* Print a call to "method", along with the corresponding
 * return statement, with the given indentation.
 * "drop_ctx" is set if the first argument is an isl_ctx.
 * "drop_user" is set if the last argument is a "user" argument
 * corresponding to a callback argument.
 *
 * A "ctx" variable is first initialized as it may be needed
 * in the first call to print_arg_in_call and in print_method_return.
 *
 * If the method has a callback function, then any exception
 * thrown in the callback also need to be rethrown.
 */
void python_generator::print_method_call(int indent, const isl_class &clazz,
	FunctionDecl *method, const char *fmt, int drop_ctx, int drop_user)
{
	string fullname = method->getName().str();
	int num_params = method->getNumParams();

	if (drop_ctx) {
		print_indent(indent, "ctx = Context.getDefaultInstance()\n");
	} else {
		print_indent(indent, "ctx = ");
		printf(fmt, 0);
		printf(".ctx\n");
	}
	print_indent(indent, "res = isl.%s(", fullname.c_str());
	for (int i = 0; i < num_params - drop_user; ++i) {
		if (i > 0)
			printf(", ");
		print_arg_in_call(method, fmt, i, drop_ctx);
	}
	if (drop_user)
		printf(", None");
	printf(")\n");

	if (drop_user)
		print_rethrow(indent, "exc_info[0]");

	print_method_return(indent, clazz, method, fmt);
}

/* Print a python method corresponding to the C function "method".
 * "super" contains the superclasses of the class to which the method belongs,
 * with the first element corresponding to the annotation that appears
 * closest to the annotated type.  This superclass is the least
 * general extension of the annotated type in the linearization
 * of the class hierarchy.
 *
 * If the first argument of "method" is something other than an instance
 * of the class, then mark the python method as static.
 * If, moreover, this first argument is an isl_ctx, then remove
 * it from the arguments of the Python method.
 *
 * If the function has a callback argument, then it also has a "user"
 * argument.  Since Python has closures, there is no need for such
 * a user argument in the Python interface, so we simply drop it.
 * We also create a wrapper ("cb") for the callback.
 *
 * If the function consumes a reference, then we pass it a copy of
 * the actual argument.
 *
 * For methods that are identified as "get" methods, also
 * print a variant of the method using a name that includes
 * the "get_" prefix.
 */
void python_generator::print_method(const isl_class &clazz,
	FunctionDecl *method, vector<string> super)
{
	string cname = clazz.method_name(method);
	int num_params = method->getNumParams();
	int drop_user = 0;
	int drop_ctx = first_arg_is_isl_ctx(method);

	for (int i = 1; i < num_params; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		QualType type = param->getOriginalType();
		if (is_callback(type))
			drop_user = 1;
	}

	print_method_header(is_static(clazz, method), cname,
			    num_params - drop_ctx - drop_user);

	print_type_checks(cname, method, drop_ctx,
			    num_params - drop_user, super);
	for (int i = 1; i < num_params; ++i) {
		ParmVarDecl *param = method->getParamDecl(i);
		QualType type = param->getOriginalType();
		if (!is_callback(type))
			continue;
		print_callback(param, i - drop_ctx);
	}
	print_method_call(8, clazz, method, fixed_arg_fmt, drop_ctx, drop_user);

	if (clazz.is_get_method(method))
		print_get_method(clazz, method);
}

/* Print a condition that checks whether Python method argument "i"
 * corresponds to the C function argument type "type".
 */
static void print_argument_check(QualType type, int i)
{
	if (generator::is_isl_type(type)) {
		string type_str;
		type_str = generator::extract_type(type);
		type_str = type2python(type_str);
		printf("args[%d].__class__ is %s", i, type_str.c_str());
	} else if (type->isPointerType()) {
		printf("type(args[%d]) == str", i);
	} else {
		printf("type(args[%d]) == int", i);
	}
}

/* Print a test that checks whether the arguments passed
 * to the Python method correspond to the arguments
 * expected by "fd".
 * "drop_ctx" is set if the first argument of "fd" is an isl_ctx,
 * which does not appear as an argument to the Python method.
 *
 * If an automatic conversion function is available for any
 * of the argument types, then also allow the argument
 * to be of the type as prescribed by the second input argument
 * of the conversion function.
 * The corresponding arguments are then converted to the expected types
 * if needed.  The argument tuple first needs to be converted to a list
 * in order to be able to modify the entries.
 */
void python_generator::print_argument_checks(const isl_class &clazz,
	FunctionDecl *fd, int drop_ctx)
{
	int num_params = fd->getNumParams();
	int first = generator::is_static(clazz, fd) ? drop_ctx : 1;
	std::vector<bool> convert(num_params);

	printf("        if len(args) == %d", num_params - drop_ctx);
	for (int i = first; i < num_params; ++i) {
		ParmVarDecl *param = fd->getParamDecl(i);
		QualType type = param->getOriginalType();
		const Type *ptr = type.getTypePtr();

		printf(" and ");
		if (conversions.count(ptr) == 0) {
			print_argument_check(type, i - drop_ctx);
		} else {
			QualType type2 = conversions.at(ptr)->getOriginalType();
			convert[i] = true;
			printf("(");
			print_argument_check(type, i - drop_ctx);
			printf(" or ");
			print_argument_check(type2, i - drop_ctx);
			printf(")");
		}
	}
	printf(":\n");

	if (std::find(convert.begin(), convert.end(), true) == convert.end())
		return;
	print_indent(12, "args = list(args)\n");
	for (int i = first; i < num_params; ++i) {
		ParmVarDecl *param = fd->getParamDecl(i);
		string type;

		if (!convert[i])
			continue;
		type = type2python(extract_type(param->getOriginalType()));
		print_type_check(12, type, var_arg_fmt,
				i - drop_ctx, false, "", "", -1);
	}
}

/* Print part of an overloaded python method corresponding to the C function
 * "method".
 * "drop_ctx" is set if the first argument of "method" is an isl_ctx.
 *
 * In particular, print code to test whether the arguments passed to
 * the python method correspond to the arguments expected by "method"
 * and to call "method" if they do.
 */
void python_generator::print_method_overload(const isl_class &clazz,
	FunctionDecl *method)
{
	int drop_ctx = first_arg_is_isl_ctx(method);

	print_argument_checks(clazz, method, drop_ctx);
	print_method_call(12, clazz, method, var_arg_fmt, drop_ctx, 0);
}

/* Print a python method with a name derived from "fullname"
 * corresponding to the C functions "methods".
 * "super" contains the superclasses of the class to which the method belongs.
 *
 * If "methods" consists of a single element that is not marked overloaded,
 * the use print_method to print the method.
 * Otherwise, print an overloaded method with pieces corresponding
 * to each function in "methods".
 */
void python_generator::print_method(const isl_class &clazz,
	const string &fullname, const function_set &methods,
	vector<string> super)
{
	string cname;
	function_set::const_iterator it;
	FunctionDecl *any_method;

	any_method = *methods.begin();
	if (methods.size() == 1 && !is_overload(any_method)) {
		print_method(clazz, any_method, super);
		return;
	}

	cname = clazz.method_name(any_method);

	print_method_def(is_static(clazz, any_method), cname);
	printf("(*args):\n");

	for (it = methods.begin(); it != methods.end(); ++it)
		print_method_overload(clazz, *it);
	printf("        raise Error\n");
}

/* Print a python method "name" corresponding to "fd" setting
 * the enum value "value".
 * "super" contains the superclasses of the class to which the method belongs,
 * with the first element corresponding to the annotation that appears
 * closest to the annotated type.
 *
 * The last argument of the C function does not appear in the method call,
 * but is fixed to "value" instead.
 * Other than that, the method printed here is similar to one
 * printed by python_generator::print_method, except that
 * some of the special cases do not occur.
 */
void python_generator::print_set_enum(const isl_class &clazz,
	FunctionDecl *fd, int value, const string &name,
	const vector<string> &super)
{
	string fullname = fd->getName().str();
	int num_params = fd->getNumParams();

	print_method_header(is_static(clazz, fd), name, num_params - 1);

	print_type_checks(name, fd, false, num_params - 1, super);
	printf("        ctx = arg0.ctx\n");
	printf("        res = isl.%s(", fullname.c_str());
	for (int i = 0; i < num_params - 1; ++i) {
		if (i)
			printf(", ");
		print_arg_in_call(fd, fixed_arg_fmt, i, 0);
	}
	printf(", %d", value);
	printf(")\n");
	print_method_return(8, clazz, fd, fixed_arg_fmt);
}

/* Print python methods corresponding to "fd", which sets an enum.
 * "super" contains the superclasses of the class to which the method belongs,
 * with the first element corresponding to the annotation that appears
 * closest to the annotated type.
 *
 * A method is generated for each value in the enum, setting
 * the enum to that value.
 */
void python_generator::print_set_enum(const isl_class &clazz,
	FunctionDecl *fd, const vector<string> &super)
{
	vector<set_enum>::const_iterator it;
	const vector<set_enum> &set_enums = clazz.set_enums.at(fd);

	for (it = set_enums.begin(); it != set_enums.end(); ++it)
		print_set_enum(clazz, fd, it->value, it->method_name, super);
}

/* Print part of the constructor for this isl_class.
 *
 * In particular, check if the actual arguments correspond to the
 * formal arguments of "cons" and if so call "cons" and put the
 * result in self.ptr and a reference to the default context in self.ctx.
 */
void python_generator::print_constructor(const isl_class &clazz,
	FunctionDecl *cons)
{
	string fullname = cons->getName().str();
	string cname = clazz.method_name(cons);
	int num_params = cons->getNumParams();
	int drop_ctx = first_arg_is_isl_ctx(cons);

	print_argument_checks(clazz, cons, drop_ctx);
	printf("            self.ctx = Context.getDefaultInstance()\n");
	printf("            self.ptr = isl.%s(", fullname.c_str());
	if (drop_ctx)
		printf("self.ctx");
	for (int i = drop_ctx; i < num_params; ++i) {
		if (i)
			printf(", ");
		print_arg_in_call(cons, var_arg_fmt, i, drop_ctx);
	}
	printf(")\n");
	printf("            return\n");
}

/* If "clazz" has a type function describing subclasses,
 * then add constructors that allow each of these subclasses
 * to be treated as an object to the superclass.
 */
void python_generator::print_upcast_constructors(const isl_class &clazz)
{
	map<int, string>::const_iterator i;

	if (!clazz.fn_type)
		return;

	for (i = clazz.type_subclasses.begin();
	     i != clazz.type_subclasses.end(); ++i) {
		printf("        if len(args) == 1 and "
						"isinstance(args[0], %s):\n",
			 type2python(i->second).c_str());
		printf("            self.ctx = args[0].ctx\n");
		printf("            self.ptr = isl.%s_copy(args[0].ptr)\n",
			clazz.name.c_str());
		printf("            return\n");
	}
}

/* Print the header of the class "name" with superclasses "super".
 * The order of the superclasses is the opposite of the order
 * in which the corresponding annotations appear in the source code.
 * If "clazz" is a subclass derived from a type function,
 * then the immediate superclass is recorded in "clazz" itself.
 */
void python_generator::print_class_header(const isl_class &clazz,
	const string &name, const vector<string> &super)
{
	printf("class %s", name.c_str());
	if (super.size() > 0) {
		printf("(");
		for (unsigned i = 0; i < super.size(); ++i) {
			if (i > 0)
				printf(", ");
			printf("%s", type2python(super[i]).c_str());
		}
		printf(")");
	} else if (clazz.is_type_subclass()) {
		printf("(%s)", type2python(clazz.superclass_name).c_str());
	} else {
		printf("(object)");
	}
	printf(":\n");
}

/* Tell ctypes about the return type of "fd".
 * In particular, if "fd" returns a pointer to an isl object,
 * then tell ctypes it returns a "c_void_p".
 * If "fd" returns a char *, then simply tell ctypes.
 *
 * Nothing needs to be done for functions returning
 * isl_bool, isl_stat or isl_size since they are represented by an int and
 * ctypes assumes that a function returns int by default.
 */
void python_generator::print_restype(FunctionDecl *fd)
{
	string fullname = fd->getName().str();
	QualType type = fd->getReturnType();
	if (is_isl_type(type))
		printf("isl.%s.restype = c_void_p\n", fullname.c_str());
	else if (is_string(type))
		printf("isl.%s.restype = POINTER(c_char)\n", fullname.c_str());
}

/* Tell ctypes about the types of the arguments of the function "fd".
 */
void python_generator::print_argtypes(FunctionDecl *fd)
{
	string fullname = fd->getName().str();
	int n = fd->getNumParams();
	int drop_user = 0;

	printf("isl.%s.argtypes = [", fullname.c_str());
	for (int i = 0; i < n - drop_user; ++i) {
		ParmVarDecl *param = fd->getParamDecl(i);
		QualType type = param->getOriginalType();
		if (is_callback(type))
			drop_user = 1;
		if (i)
			printf(", ");
		if (is_isl_ctx(type))
			printf("Context");
		else if (is_isl_type(type) || is_callback(type))
			printf("c_void_p");
		else if (is_string(type))
			printf("c_char_p");
		else if (is_long(type))
			printf("c_long");
		else
			printf("c_int");
	}
	if (drop_user)
		printf(", c_void_p");
	printf("]\n");
}

/* Print type definitions for the method 'fd'.
 */
void python_generator::print_method_type(FunctionDecl *fd)
{
	print_restype(fd);
	print_argtypes(fd);
}

/* If "clazz" has a type function describing subclasses or
 * if it is one of those type subclasses, then print a __new__ method.
 *
 * In the superclass, the __new__ method constructs an object
 * of the subclass type specified by the type function.
 * In the subclass, the __new__ method reverts to the original behavior.
 */
void python_generator::print_new(const isl_class &clazz,
	const string &python_name)
{
	if (!clazz.fn_type && !clazz.is_type_subclass())
		return;

	printf("    def __new__(cls, *args, **keywords):\n");

	if (clazz.fn_type) {
		map<int, string>::const_iterator i;

		printf("        if \"ptr\" in keywords:\n");
		printf("            type = isl.%s(keywords[\"ptr\"])\n",
			clazz.fn_type->getNameAsString().c_str());

		for (i = clazz.type_subclasses.begin();
		     i != clazz.type_subclasses.end(); ++i) {
			printf("            if type == %d:\n", i->first);
			printf("                return %s(**keywords)\n",
				type2python(i->second).c_str());
		}
		printf("            raise\n");
	}

	printf("        return super(%s, cls).__new__(cls)\n",
		python_name.c_str());
}

/* Print declarations for methods printing the class representation,
 * provided there is a corresponding *_to_str function.
 *
 * In particular, provide an implementation of __str__ and __repr__ methods to
 * override the default representation used by python. Python uses __str__ to
 * pretty print the class (e.g., when calling print(obj)) and uses __repr__
 * when printing a precise representation of an object (e.g., when dumping it
 * in the REPL console).
 *
 * Check the type of the argument before calling the *_to_str function
 * on it in case the method was called on an object from a subclass.
 *
 * The return value of the *_to_str function is decoded to a python string
 * assuming an 'ascii' encoding.  This is necessary for python 3 compatibility.
 */
void python_generator::print_representation(const isl_class &clazz,
	const string &python_name)
{
	if (!clazz.fn_to_str)
		return;

	printf("    def __str__(arg0):\n");
	print_type_check(8, python_name, fixed_arg_fmt, 0, false, "", "", -1);
	printf("        ptr = isl.%s(arg0.ptr)\n",
		string(clazz.fn_to_str->getName()).c_str());
	printf("        res = cast(ptr, c_char_p).value.decode('ascii')\n");
	printf("        libc.free(ptr)\n");
	printf("        return res\n");
	printf("    def __repr__(self):\n");
	printf("        s = str(self)\n");
	printf("        if '\"' in s:\n");
	printf("            return 'isl.%s(\"\"\"%%s\"\"\")' %% s\n",
		python_name.c_str());
	printf("        else:\n");
	printf("            return 'isl.%s(\"%%s\")' %% s\n",
		python_name.c_str());
}

/* If "clazz" has any persistent callbacks, then print the definition
 * of a "copy_callbacks" function that copies the persistent callbacks
 * from one object to another.
 */
void python_generator::print_copy_callbacks(const isl_class &clazz)
{
	const set<FunctionDecl *> &callbacks = clazz.persistent_callbacks;
	set<FunctionDecl *>::const_iterator in;

	if (!clazz.has_persistent_callbacks())
		return;

	printf("    def copy_callbacks(self, obj):\n");
	for (in = callbacks.begin(); in != callbacks.end(); ++in) {
		string callback_name = clazz.persistent_callback_name(*in);

		printf("        if hasattr(obj, '%s'):\n",
			callback_name.c_str());
		printf("            self.%s = obj.%s\n",
			callback_name.c_str(), callback_name.c_str());
	}
}

/* Print code to set method type signatures.
 *
 * To be able to call C functions it is necessary to explicitly set their
 * argument and result types.  Do this for all exported constructors and
 * methods (including those that set a persistent callback and
 * those that set an enum value),
 * as well as for the *_to_str and the type function, if they exist.
 * Assuming each exported class has a *_copy and a *_free method,
 * also unconditionally set the type of such methods.
 */
void python_generator::print_method_types(const isl_class &clazz)
{
	function_set::const_iterator in;
	map<string, function_set>::const_iterator it;
	map<FunctionDecl *, vector<set_enum> >::const_iterator ie;
	const set<FunctionDecl *> &callbacks = clazz.persistent_callbacks;

	for (in = clazz.constructors.begin(); in != clazz.constructors.end();
		++in)
		print_method_type(*in);

	for (in = callbacks.begin(); in != callbacks.end(); ++in)
		print_method_type(*in);
	for (it = clazz.methods.begin(); it != clazz.methods.end(); ++it)
		for (in = it->second.begin(); in != it->second.end(); ++in)
			print_method_type(*in);
	for (ie = clazz.set_enums.begin(); ie != clazz.set_enums.end(); ++ie)
		print_method_type(ie->first);

	print_method_type(clazz.fn_copy);
	print_method_type(clazz.fn_free);
	if (clazz.fn_to_str)
		print_method_type(clazz.fn_to_str);
	if (clazz.fn_type)
		print_method_type(clazz.fn_type);
}

/* Print out the definition of this isl_class.
 *
 * We first check if this isl_class is a subclass of one or more other classes.
 * If it is, we make sure those superclasses are printed out first.
 *
 * Then we print a constructor with several cases, one for constructing
 * a Python object from a return value, one for each function that
 * was marked as a constructor and for each type based subclass.
 *
 * Next, we print out some common methods and the methods corresponding
 * to functions that are not marked as constructors, including those
 * that set a persistent callback and those that set an enum value.
 *
 * Finally, we tell ctypes about the types of the arguments of the
 * constructor functions and the return types of those function returning
 * an isl object.
 */
void python_generator::print(const isl_class &clazz)
{
	string p_name = type2python(clazz.subclass_name);
	function_set::const_iterator in;
	map<string, function_set>::const_iterator it;
	map<FunctionDecl *, vector<set_enum> >::const_iterator ie;
	vector<string> super = find_superclasses(clazz.type);
	const set<FunctionDecl *> &callbacks = clazz.persistent_callbacks;

	for (unsigned i = 0; i < super.size(); ++i)
		if (done.find(super[i]) == done.end())
			print(classes[super[i]]);
	if (clazz.is_type_subclass() && done.find(clazz.name) == done.end())
		print(classes[clazz.name]);
	done.insert(clazz.subclass_name);

	printf("\n");
	print_class_header(clazz, p_name, super);
	printf("    def __init__(self, *args, **keywords):\n");

	printf("        if \"ptr\" in keywords:\n");
	printf("            self.ctx = keywords[\"ctx\"]\n");
	printf("            self.ptr = keywords[\"ptr\"]\n");
	printf("            return\n");

	for (in = clazz.constructors.begin(); in != clazz.constructors.end();
		++in)
		print_constructor(clazz, *in);
	print_upcast_constructors(clazz);
	printf("        raise Error\n");
	printf("    def __del__(self):\n");
	printf("        if hasattr(self, 'ptr'):\n");
	printf("            isl.%s_free(self.ptr)\n", clazz.name.c_str());

	print_new(clazz, p_name);
	print_representation(clazz, p_name);
	print_copy_callbacks(clazz);

	for (in = callbacks.begin(); in != callbacks.end(); ++in)
		print_method(clazz, *in, super);
	for (it = clazz.methods.begin(); it != clazz.methods.end(); ++it)
		print_method(clazz, it->first, it->second, super);
	for (ie = clazz.set_enums.begin(); ie != clazz.set_enums.end(); ++ie)
		print_set_enum(clazz, ie->first, super);

	printf("\n");

	print_method_types(clazz);
}

/* Generate a python interface based on the extracted types and
 * functions.
 *
 * Print out each class in turn.  If one of these is a subclass of some
 * other class, make sure the superclass is printed out first.
 * functions.
 */
void python_generator::generate()
{
	map<string, isl_class>::iterator ci;

	for (ci = classes.begin(); ci != classes.end(); ++ci) {
		if (done.find(ci->first) == done.end())
			print(ci->second);
	}
}
