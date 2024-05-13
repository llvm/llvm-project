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

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <iostream>

#include <clang/AST/Attr.h>
#include <clang/Basic/SourceManager.h>

#include "isl_config.h"
#include "extract_interface.h"
#include "generator.h"

const char *isl_class::get_prefix = "get_";
const char *isl_class::set_callback_prefix = "set_";

/* Is the first argument an instance of the class?
 */
bool isl_class::first_arg_matches_class(FunctionDecl *method) const
{
	ParmVarDecl *param;
	QualType type;

	if (method->getNumParams() < 1)
		return false;

	param = method->getParamDecl(0);
	type = param->getOriginalType();
	if (!generator::is_isl_type(type))
		return false;
	return generator::extract_type(type) == name;
}

/* Should "method" be considered to be a static method?
 * That is, is the first argument something other than
 * an instance of the class?
 *
 * If this method was copied from a superclass, then check
 * whether the method is static with respect to this superclass.
 */
bool isl_class::is_static(FunctionDecl *method) const
{
	if (copied_from.count(method) != 0)
		return copied_from.at(method).is_static(method);
	return !first_arg_matches_class(method);
}

/* Should "method" be considered to be a static method?
 * That is, is the first argument something other than
 * an instance of the class?
 */
bool generator::is_static(const isl_class &clazz, FunctionDecl *method)
{
	return clazz.is_static(method);
}

/* Does "fd" modify an object of "clazz"?
 * That is, is it an object method that takes the object and
 * returns (gives) an object of the same type?
 */
bool generator::is_mutator(const isl_class &clazz, FunctionDecl *fd)
{
	ParmVarDecl *param;
	QualType type, return_type;

	if (fd->getNumParams() < 1)
		return false;
	if (is_static(clazz, fd))
		return false;

	if (!gives(fd))
		return false;
	param = fd->getParamDecl(0);
	if (!takes(param))
		return false;
	type = param->getOriginalType();
	return_type = fd->getReturnType();
	return return_type == type;
}

/* Find the FunctionDecl with name "name",
 * returning NULL if there is no such FunctionDecl.
 * If "required" is set, then error out if no FunctionDecl can be found.
 */
FunctionDecl *generator::find_by_name(const string &name, bool required)
{
	map<string, FunctionDecl *>::iterator i;

	i = functions_by_name.find(name);
	if (i != functions_by_name.end())
		return i->second;
	if (required)
		die("No " + name + " function found");
	return NULL;
}

/* List of conversion functions that are used to automatically convert
 * the second argument of the conversion function to its function result.
 */
const std::set<std::string> generator::automatic_conversion_functions = {
	"isl_id_read_from_str",
	"isl_val_int_from_si",
};

/* Extract information about the automatic conversion function "fd",
 * storing the results in this->conversions.
 *
 * A function used for automatic conversion has exactly two arguments,
 * an isl_ctx and a non-isl object, and it returns an isl object.
 * Store a mapping from the isl object return type
 * to the non-isl object source type.
 */
void generator::extract_automatic_conversion(FunctionDecl *fd)
{
	QualType return_type = fd->getReturnType();
	const Type *type = return_type.getTypePtr();

	if (fd->getNumParams() != 2)
		die("Expecting two arguments");
	if (!is_isl_ctx(fd->getParamDecl(0)->getOriginalType()))
		die("Expecting isl_ctx first argument");
	if (!is_isl_type(return_type))
		die("Expecting isl object return type");
	conversions[type] = fd->getParamDecl(1);
}

/* Extract information about all automatic conversion functions
 * for the given class, storing the results in this->conversions.
 *
 * In particular, look through all exported constructors for the class and
 * check if any of them is explicitly marked as a conversion function.
 */
void generator::extract_class_automatic_conversions(const isl_class &clazz)
{
	const function_set &constructors = clazz.constructors;
	function_set::iterator fi;

	for (fi = constructors.begin(); fi != constructors.end(); ++fi) {
		FunctionDecl *fd = *fi;
		string name = fd->getName().str();
		if (automatic_conversion_functions.count(name) != 0)
			extract_automatic_conversion(fd);
	}
}

/* Extract information about all automatic conversion functions,
 * storing the results in this->conversions.
 */
void generator::extract_automatic_conversions()
{
	map<string, isl_class>::iterator ci;

	for (ci = classes.begin(); ci != classes.end(); ++ci)
		extract_class_automatic_conversions(ci->second);
}

/* Add a subclass derived from "decl" called "sub_name" to the set of classes,
 * keeping track of the _to_str, _copy and _free functions, if any, separately.
 * "sub_name" is either the name of the class itself or
 * the name of a type based subclass.
 * If the class is a proper subclass, then "super_name" is the name
 * of its immediate superclass.
 */
void generator::add_subclass(RecordDecl *decl, const string &super_name,
	const string &sub_name)
{
	string name = decl->getName().str();

	classes[sub_name].name = name;
	classes[sub_name].superclass_name = super_name;
	classes[sub_name].subclass_name = sub_name;
	classes[sub_name].type = decl;
	classes[sub_name].fn_to_str = find_by_name(name + "_to_str", false);
	classes[sub_name].fn_copy = find_by_name(name + "_copy", true);
	classes[sub_name].fn_free = find_by_name(name + "_free", true);
}

/* Add a class derived from "decl" to the set of classes,
 * keeping track of the _to_str, _copy and _free functions, if any, separately.
 */
void generator::add_class(RecordDecl *decl)
{
	return add_subclass(decl, "", decl->getName().str());
}

/* Given a function "fn_type" that returns the subclass type
 * of a C object, create subclasses for each of the (non-negative)
 * return values.
 *
 * The function "fn_type" is also stored in the superclass,
 * along with all pairs of type values and subclass names.
 */
void generator::add_type_subclasses(FunctionDecl *fn_type)
{
	QualType return_type = fn_type->getReturnType();
	const EnumType *enum_type = return_type->getAs<EnumType>();
	EnumDecl *decl = enum_type->getDecl();
	isl_class *c = method2class(fn_type);
	DeclContext::decl_iterator i;

	c->fn_type = fn_type;
	for (i = decl->decls_begin(); i != decl->decls_end(); ++i) {
		EnumConstantDecl *ecd = dyn_cast<EnumConstantDecl>(*i);
		int val = (int) ecd->getInitVal().getSExtValue();
		string name = ecd->getNameAsString();

		if (val < 0)
			continue;
		c->type_subclasses[val] = name;
		add_subclass(c->type, c->subclass_name, name);
	}
}

/* Add information about the enum values in "decl", set by "fd",
 * to c->set_enums. "prefix" is the prefix of the generated method names.
 * In particular, it has the name of the enum type removed.
 *
 * In particular, for each non-negative enum value, keep track of
 * the value, the name and the corresponding method name.
 */
static void add_set_enum(isl_class *c, const string &prefix, EnumDecl *decl,
	FunctionDecl *fd)
{
	DeclContext::decl_iterator i;

	for (i = decl->decls_begin(); i != decl->decls_end(); ++i) {
		EnumConstantDecl *ecd = dyn_cast<EnumConstantDecl>(*i);
		int val = (int) ecd->getInitVal().getSExtValue();
		string name = ecd->getNameAsString();
		string method_name;

		if (val < 0)
			continue;
		method_name = prefix + name.substr(4);
		c->set_enums[fd].push_back(set_enum(val, name, method_name));
	}
}

/* Check if "fd" sets an enum value and, if so, add information
 * about the enum values to c->set_enums.
 *
 * A function is considered to set an enum value if:
 * - the function returns an object of the same type
 * - the last argument is of type enum
 * - the name of the function ends with the name of the enum
 */
static bool handled_sets_enum(isl_class *c, FunctionDecl *fd)
{
	unsigned n;
	ParmVarDecl *param;
	const EnumType *enum_type;
	EnumDecl *decl;
	string enum_name;
	string fd_name;
	string prefix;
	size_t pos;

	if (!generator::is_mutator(*c, fd))
		return false;
	n = fd->getNumParams();
	if (n < 2)
		return false;
	param = fd->getParamDecl(n - 1);
	enum_type = param->getType()->getAs<EnumType>();
	if (!enum_type)
		return false;
	decl = enum_type->getDecl();
	enum_name = decl->getName().str();
	enum_name = enum_name.substr(4);
	fd_name = c->method_name(fd);
	pos = fd_name.find(enum_name);
	if (pos == std::string::npos)
		return false;
	prefix = fd_name.substr(0, pos);

	add_set_enum(c, prefix, decl, fd);

	return true;
}

/* Return the callback argument of a function setting
 * a persistent callback.
 * This callback is in the second argument (position 1).
 */
ParmVarDecl *generator::persistent_callback_arg(FunctionDecl *fd)
{
	return fd->getParamDecl(1);
}

/* Does the given function set a persistent callback?
 * The following heuristics are used to determine this property:
 * - the function returns an object of the same type
 * - its name starts with "set_"
 * - it has exactly three arguments
 * - the second (position 1) of which is a callback
 */
static bool sets_persistent_callback(isl_class *c, FunctionDecl *fd)
{
	ParmVarDecl *param;

	if (!generator::is_mutator(*c, fd))
		return false;
	if (fd->getNumParams() != 3)
		return false;
	param = generator::persistent_callback_arg(fd);
	if (!generator::is_callback(param->getType()))
		return false;
	return prefixcmp(c->method_name(fd).c_str(),
			 c->set_callback_prefix) == 0;
}

/* Does this function take any enum arguments?
 */
static bool takes_enums(FunctionDecl *fd)
{
	unsigned n;

	n = fd->getNumParams();
	for (unsigned i = 0; i < n; ++i) {
		ParmVarDecl *param = fd->getParamDecl(i);
		if (param->getType()->getAs<EnumType>())
			return true;
	}
	return false;
}

/* Sorting function that places declaration of functions
 * with a shorter name first.
 */
static bool less_name(const FunctionDecl *a, const FunctionDecl *b)
{
	return a->getName().size() < b->getName().size();
}

/* Collect all functions that belong to a certain type, separating
 * constructors from methods that set an enum value,
 * methods that set a persistent callback and
 * from regular methods, while keeping track of the _to_str,
 * _copy and _free functions, if any, separately.
 * Methods that accept any enum arguments that are not specifically handled
 * are not supported.
 * If there are any overloaded
 * functions, then they are grouped based on their name after removing the
 * argument type suffix.
 * Check for functions that describe subclasses before considering
 * any other functions in order to be able to detect those other
 * functions as belonging to the subclasses.
 * Sort the names of the functions based on their lengths
 * to ensure that nested subclasses are handled later.
 *
 * Also extract information about automatic conversion functions.
 */
generator::generator(SourceManager &SM, set<RecordDecl *> &exported_types,
	set<FunctionDecl *> exported_functions, set<FunctionDecl *> functions) :
	SM(SM)
{
	set<FunctionDecl *>::iterator in;
	set<RecordDecl *>::iterator it;
	vector<FunctionDecl *> type_subclasses;
	vector<FunctionDecl *>::iterator iv;

	for (in = functions.begin(); in != functions.end(); ++in) {
		FunctionDecl *decl = *in;
		functions_by_name[decl->getName().str()] = decl;
	}

	for (it = exported_types.begin(); it != exported_types.end(); ++it)
		add_class(*it);

	for (in = exported_functions.begin(); in != exported_functions.end();
	     ++in) {
		if (is_subclass(*in))
			type_subclasses.push_back(*in);
	}
	std::sort(type_subclasses.begin(), type_subclasses.end(), &less_name);
	for (iv = type_subclasses.begin(); iv != type_subclasses.end(); ++iv) {
		add_type_subclasses(*iv);
	}

	for (in = exported_functions.begin(); in != exported_functions.end();
	     ++in) {
		FunctionDecl *method = *in;
		isl_class *c;

		if (is_subclass(method))
			continue;

		c = method2class(method);
		if (!c)
			continue;
		if (is_constructor(method)) {
			c->constructors.insert(method);
		} else if (handled_sets_enum(c, method)) {
		} else if (sets_persistent_callback(c, method)) {
			c->persistent_callbacks.insert(method);
		} else if (takes_enums(method)) {
			std::string name = method->getName().str();
			die(name + " has unhandled enum argument");
		} else {
			string name = c->method_name(method);
			c->methods[name].insert(method);
		}
	}

	extract_automatic_conversions();
}

/* Print error message "msg" and abort.
 */
void generator::die(const char *msg)
{
	fprintf(stderr, "%s\n", msg);
	abort();
}

/* Print error message "msg" and abort.
 */
void generator::die(string msg)
{
	die(msg.c_str());
}

/* Return a sequence of the types of which the given type declaration is
 * marked as being a subtype.
 * The order of the types is the opposite of the order in which they
 * appear in the source.  In particular, the first annotation
 * is the one that is closest to the annotated type and the corresponding
 * type is then also the first that will appear in the sequence of types.
 * This is also the order in which the annotations appear
 * in the AttrVec returned by Decl::getAttrs() in older versions of clang.
 * In newer versions of clang, the order is that in which
 * the attribute appears in the source.
 * Use the position of the "isl_export" attribute to determine
 * whether this is an old (with reversed order) or a new version.
 * The "isl_export" attribute is automatically added
 * after each "isl_subclass" attribute.  If it appears in the list before
 * any "isl_subclass" is encountered, then this must be a reversed list.
 */
std::vector<string> generator::find_superclasses(Decl *decl)
{
	vector<string> super;
	bool reversed = false;

	if (!decl->hasAttrs())
		return super;

	string sub = "isl_subclass";
	size_t len = sub.length();
	AttrVec attrs = decl->getAttrs();
	for (AttrVec::const_iterator i = attrs.begin(); i != attrs.end(); ++i) {
		const AnnotateAttr *ann = dyn_cast<AnnotateAttr>(*i);
		if (!ann)
			continue;
		string s = ann->getAnnotation().str();
		if (s == "isl_export" && super.size() == 0)
			reversed = true;
		if (s.substr(0, len) == sub) {
			s = s.substr(len + 1, s.length() - len  - 2);
			if (reversed)
				super.push_back(s);
			else
				super.insert(super.begin(), s);
		}
	}

	return super;
}

/* Is "decl" marked as describing subclasses?
 */
bool generator::is_subclass(FunctionDecl *decl)
{
	return find_superclasses(decl).size() > 0;
}

/* Is decl marked as being part of an overloaded method?
 */
bool generator::is_overload(Decl *decl)
{
	return has_annotation(decl, "isl_overload");
}

/* Is decl marked as a constructor?
 */
bool generator::is_constructor(Decl *decl)
{
	return has_annotation(decl, "isl_constructor");
}

/* Is decl marked as consuming a reference?
 */
bool generator::takes(Decl *decl)
{
	return has_annotation(decl, "isl_take");
}

/* Is decl marked as preserving a reference?
 */
bool generator::keeps(Decl *decl)
{
	return has_annotation(decl, "isl_keep");
}

/* Is decl marked as returning a reference that is required to be freed.
 */
bool generator::gives(Decl *decl)
{
	return has_annotation(decl, "isl_give");
}

/* Return the class that has a name that best matches the initial part
 * of the name of function "fd" or NULL if no such class could be found.
 */
isl_class *generator::method2class(FunctionDecl *fd)
{
	string best;
	map<string, isl_class>::iterator ci;
	string name = fd->getNameAsString();

	for (ci = classes.begin(); ci != classes.end(); ++ci) {
		size_t len = ci->first.length();
		if (len > best.length() && name.substr(0, len) == ci->first &&
		    name[len] == '_')
			best = ci->first;
	}

	if (classes.find(best) == classes.end()) {
		cerr << "Unable to find class of " << name << endl;
		return NULL;
	}

	return &classes[best];
}

/* Is "type" the type "isl_ctx *"?
 */
bool generator::is_isl_ctx(QualType type)
{
	if (!type->isPointerType())
		return false;
	type = type->getPointeeType();
	if (type.getAsString() != "isl_ctx")
		return false;

	return true;
}

/* Is the first argument of "fd" of type "isl_ctx *"?
 */
bool generator::first_arg_is_isl_ctx(FunctionDecl *fd)
{
	ParmVarDecl *param;

	if (fd->getNumParams() < 1)
		return false;

	param = fd->getParamDecl(0);
	return is_isl_ctx(param->getOriginalType());
}

namespace {

struct ClangAPI {
	/* Return the first location in the range returned by
	 * clang::SourceManager::getImmediateExpansionRange.
	 * Older versions of clang return a pair of SourceLocation objects.
	 * More recent versions return a CharSourceRange.
	 */
	static SourceLocation range_begin(
			const std::pair<SourceLocation,SourceLocation> &p) {
		return p.first;
	}
	static SourceLocation range_begin(const CharSourceRange &range) {
		return range.getBegin();
	}
};

}

/* Does the callback argument "param" take its argument at position "pos"?
 *
 * The memory management annotations of arguments to function pointers
 * are not recorded by clang, so the information cannot be extracted
 * from the type of "param".
 * Instead, go to the location in the source where the callback argument
 * is declared, look for the right argument of the callback itself and
 * then check if it has an "__isl_take" memory management annotation.
 *
 * If the return value of the function has a memory management annotation,
 * then the spelling of "param" will point to the spelling
 * of this memory management annotation.  Since the macro is defined
 * on the command line (in main), this location does not have a file entry.
 * In this case, move up one level in the macro expansion to the location
 * where the memory management annotation is used.
 */
bool generator::callback_takes_argument(ParmVarDecl *param,
	int pos)
{
	SourceLocation loc;
	const char *s, *end, *next;
	bool takes, keeps;

	loc = param->getSourceRange().getBegin();
	if (!SM.getFileEntryForID(SM.getFileID(SM.getSpellingLoc(loc))))
		loc = ClangAPI::range_begin(SM.getImmediateExpansionRange(loc));
	s = SM.getCharacterData(loc);
	if (!s)
		die("No character data");
	s = strchr(s, '(');
	if (!s)
		die("Cannot find function pointer");
	s = strchr(s + 1, '(');
	if (!s)
		die("Cannot find function pointer arguments");
	end = strchr(s + 1, ')');
	if (!end)
		die("Cannot find end of function pointer arguments");
	while (pos-- > 0) {
		s = strchr(s + 1, ',');
		if (!s || s > end)
			die("Cannot find function pointer argument");
	}
	next = strchr(s + 1, ',');
	if (next && next < end)
		end = next;
	s = strchr(s + 1, '_');
	if (!s || s > end)
		die("Cannot find function pointer argument annotation");
	takes = prefixcmp(s, "__isl_take") == 0;
	keeps = prefixcmp(s, "__isl_keep") == 0;
	if (!takes && !keeps)
		die("Cannot find function pointer argument annotation");

	return takes;
}

/* Is "type" that of a pointer to an isl_* structure?
 */
bool generator::is_isl_type(QualType type)
{
	if (type->isPointerType()) {
		string s;

		type = type->getPointeeType();
		if (type->isFunctionType())
			return false;
		s = type.getAsString();
		return s.substr(0, 4) == "isl_";
	}

	return false;
}

/* Is "type" one of the integral types with a negative value
 * indicating an error condition?
 */
bool generator::is_isl_neg_error(QualType type)
{
	return is_isl_bool(type) || is_isl_stat(type) || is_isl_size(type);
}

/* Is "type" the primitive type with the given name?
 */
static bool is_isl_primitive(QualType type, const char *name)
{
	string s;

	if (type->isPointerType())
		return false;

	s = type.getAsString();
	return s == name;
}

/* Is "type" the type isl_bool?
 */
bool generator::is_isl_bool(QualType type)
{
	return is_isl_primitive(type, "isl_bool");
}

/* Is "type" the type isl_stat?
 */
bool generator::is_isl_stat(QualType type)
{
	return is_isl_primitive(type, "isl_stat");
}

/* Is "type" the type isl_size?
 */
bool generator::is_isl_size(QualType type)
{
	return is_isl_primitive(type, "isl_size");
}

/* Is "type" that of a pointer to a function?
 */
bool generator::is_callback(QualType type)
{
	if (!type->isPointerType())
		return false;
	type = type->getPointeeType();
	return type->isFunctionType();
}

/* Is the parameter at position "i" of "fd" a pointer to a function?
 */
bool generator::is_callback_arg(FunctionDecl *fd, int i)
{
	ParmVarDecl *param = fd->getParamDecl(i);
	QualType type = param->getOriginalType();

	return is_callback(type);
}

/* Is "type" that of "char *" of "const char *"?
 */
bool generator::is_string(QualType type)
{
	if (type->isPointerType()) {
		string s = type->getPointeeType().getAsString();
		return s == "const char" || s == "char";
	}

	return false;
}

/* Is "type" that of "long"?
 */
bool generator::is_long(QualType type)
{
	const BuiltinType *builtin = type->getAs<BuiltinType>();
	return builtin && builtin->getKind() == BuiltinType::Long;
}

/* Is "type" that of "unsigned int"?
 */
static bool is_unsigned_int(QualType type)
{
	const BuiltinType *builtin = type->getAs<BuiltinType>();
	return builtin && builtin->getKind() == BuiltinType::UInt;
}

/* Return the name of the type that "type" points to.
 * The input "type" is assumed to be a pointer type.
 */
string generator::extract_type(QualType type)
{
	if (type->isPointerType())
		return type->getPointeeType().getAsString();
	die("Cannot extract type from non-pointer type");
}

/* Given the type of a function pointer, return the corresponding
 * function prototype.
 */
const FunctionProtoType *generator::extract_prototype(QualType type)
{
	return type->getPointeeType()->getAs<FunctionProtoType>();
}

/* Given the type of a function pointer, return the number of arguments
 * of the corresponding function prototype.
 */
int generator::prototype_n_args(QualType type)
{
	return extract_prototype(type)->getNumArgs();
}

/* Return the function name suffix for the type of "param".
 *
 * If the type of "param" is an isl object type,
 * then the suffix is the name of the type with the "isl" prefix removed,
 * but keeping the "_".
 * If the type is an unsigned integer, then the type suffix is "_ui".
 */
static std::string type_suffix(ParmVarDecl *param)
{
	QualType type;

	type = param->getOriginalType();
	if (generator::is_isl_type(type))
		return generator::extract_type(type).substr(3);
	else if (is_unsigned_int(type))
		return "_ui";
	generator::die("Unsupported type suffix");
}

/* If "suffix" is a suffix of "s", then return "s" with the suffix removed.
 * Otherwise, simply return "s".
 */
std::string generator::drop_suffix(const std::string &s,
	const std::string &suffix)
{
	size_t len, suffix_len;

	len = s.length();
	suffix_len = suffix.length();

	if (len >= suffix_len && s.substr(len - suffix_len) == suffix)
		return s.substr(0, len - suffix_len);
	else
		return s;
}

/* If "method" is overloaded, then return its name with the suffixes
 * corresponding to the types of the final arguments removed.
 * Otherwise, simply return the name of the function.
 * Start from the final argument and keep removing suffixes
 * matching arguments, independently of whether previously considered
 * arguments matched.
 */
string isl_class::name_without_type_suffixes(FunctionDecl *method)
{
	int num_params;
	string name;

	name = method->getName().str();
	if (!generator::is_overload(method))
		return name;

	num_params = method->getNumParams();
	for (int i = num_params - 1; i >= 0; --i) {
		ParmVarDecl *param;
		string type;

		param = method->getParamDecl(i);
		type = type_suffix(param);

		name = generator::drop_suffix(name, type);
	}

	return name;
}

/* Is function "fd" with the given name a "get" method?
 *
 * A "get" method is an instance method
 * with a name that starts with the get method prefix.
 */
bool isl_class::is_get_method_name(FunctionDecl *fd, const string &name) const
{
	return !is_static(fd) && prefixcmp(name.c_str(), get_prefix) == 0;
}

/* Extract the method name corresponding to "fd".
 *
 * If "fd" is a "get" method, then drop the "get" method prefix.
 */
string isl_class::method_name(FunctionDecl *fd) const
{
      string base = base_method_name(fd);

      if (is_get_method_name(fd, base))
	      return base.substr(strlen(get_prefix));
      return base;
}
