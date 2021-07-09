#ifndef ISL_INTERFACE_GENERATOR_H
#define ISL_INTERFACE_GENERATOR_H

#include <map>
#include <set>
#include <string>
#include <vector>

#include <clang/AST/Decl.h>

using namespace std;
using namespace clang;

/* Compare the prefix of "s" to "prefix" up to the length of "prefix".
 */
inline int prefixcmp(const char *s, const char *prefix)
{
	return strncmp(s, prefix, strlen(prefix));
}

/* Information about a single enum value of an enum set by a function.
 * "value" is the enum value.
 * "name" is the corresponding name.
 * "method_name" is the the name of the method that sets this value.
 */
struct set_enum {
	int	value;
	string	name;
	string	method_name;
	set_enum(int value, string name, string method_name) :
		value(value), name(name), method_name(method_name) {}
};

/* Helper structure for sorting FunctionDecl pointers
 * on the corresponding function names.
 */
struct function_name_less {
	bool operator()(FunctionDecl *x, FunctionDecl *y) const {
		return x->getName() < y->getName();
	}
};

/* Set of FunctionDecl pointers sorted on function name.
 */
typedef std::set<FunctionDecl *, function_name_less> function_set;

/* isl_class collects all constructors and methods for an isl "class".
 * "name" is the name of the class.
 * If this object describes a subclass of a C type, then
 * "subclass_name" is the name of that subclass and "superclass_name"
 * is the name of the immediate superclass of that subclass.  Otherwise,
 * "subclass_name" is equal to "name" and "superclass_name" is undefined.
 * "type" is the declaration that introduces the type.
 * "persistent_callbacks" contains the set of functions that
 * set a persistent callback.
 * "set_enums" maps the set of functions that set an enum value
 * to information associated to each value.
 * A function is considered to set an enum value if it returns
 * an object of the same type and if its last argument is of an enum type.
 * "methods" contains the set of methods, grouped by method name.
 * "fn_to_str" is a reference to the *_to_str method of this class, if any.
 * "fn_copy" is a reference to the *_copy method of this class, if any.
 * "fn_free" is a reference to the *_free method of this class, if any.
 * "fn_type" is a reference to a function that described subclasses, if any.
 * If "fn_type" is set, then "type_subclasses" maps the values returned
 * by that function to the names of the corresponding subclasses.
 *
 * The following fields are only used for the C++ bindings.
 * For methods that are not derived from a function that applies
 * directly to this class, but are rather copied from some ancestor,
 * "copied_from" records the direct superclass from which the method
 * was copied (where it may have been copied from a further ancestor) and
 * "copy_depth" records the distance to the ancestor to which
 * the function applies.
 * "construction_types" contains the set of isl classes that can be
 * implicitly converted to this class through a unary constructor,
 * mapped to the single argument
 * of this unary constructor.
 */
struct isl_class {
	string name;
	string superclass_name;
	string subclass_name;
	RecordDecl *type;
	function_set constructors;
	set<FunctionDecl *> persistent_callbacks;
	map<FunctionDecl *, vector<set_enum> > set_enums;
	map<string, function_set> methods;
	map<int, string> type_subclasses;
	FunctionDecl *fn_type;
	FunctionDecl *fn_to_str;
	FunctionDecl *fn_copy;
	FunctionDecl *fn_free;

	std::map<clang::FunctionDecl *, const isl_class &> copied_from;
	std::map<clang::FunctionDecl *, int> copy_depth;
	std::map<std::string, clang::ParmVarDecl *> construction_types;

	/* Is the first argument an instance of the class? */
	bool first_arg_matches_class(FunctionDecl *method) const;
	/* Does "method" correspond to a static method? */
	bool is_static(FunctionDecl *method) const;
	/* Is this class a subclass based on a type function? */
	bool is_type_subclass() const { return name != subclass_name; }
	/* Return name of "fd" without type suffixes, if any. */
	static string name_without_type_suffixes(FunctionDecl *fd);
	/* Extract the method name corresponding to "fd"
	 * (including "get" method prefix if any).
	 */
	string base_method_name(FunctionDecl *fd) const {
		string m_name = name_without_type_suffixes(fd);
		return m_name.substr(subclass_name.length() + 1);
	}
	/* The prefix of a "get" method. */
	static const char *get_prefix;
	/* Is function "fd" with the given name a "get" method? */
	bool is_get_method_name(FunctionDecl *fd, const string &name) const;
	/* Is function "fd" a "get" method? */
	bool is_get_method(FunctionDecl *fd) const {
		return is_get_method_name(fd, base_method_name(fd));
	}
	/* Extract the method name corresponding to "fd". */
	string method_name(FunctionDecl *fd) const;
	/* The prefix of any method that may set a (persistent) callback. */
	static const char *set_callback_prefix;
	/* Given a function that sets a persistent callback,
	 * return the name of the callback.
	 */
	string persistent_callback_name(FunctionDecl *fd) const {
		return method_name(fd).substr(strlen(set_callback_prefix));
	}
	/* Does this class have any functions that set a persistent callback?
	 */
	bool has_persistent_callbacks() const {
		return persistent_callbacks.size() != 0;
	}
};

/* Base class for interface generators.
 *
 * "conversions" maps the target type of automatic conversion
 * to the second input argument of the conversion function.
 */
class generator {
protected:
	SourceManager &SM;
	map<string,isl_class> classes;
	map<string, FunctionDecl *> functions_by_name;

public:
	generator(SourceManager &SM, set<RecordDecl *> &exported_types,
		set<FunctionDecl *> exported_functions,
		set<FunctionDecl *> functions);

	virtual void generate() = 0;
	virtual ~generator() {};

protected:
	void add_subclass(RecordDecl *decl, const string &name,
		const string &sub_name);
	void add_class(RecordDecl *decl);
	void add_type_subclasses(FunctionDecl *method);
	isl_class *method2class(FunctionDecl *fd);
	bool callback_takes_argument(ParmVarDecl *param, int pos);
	FunctionDecl *find_by_name(const string &name, bool required);
	std::map<const Type *, ParmVarDecl *> conversions;
private:
	static const std::set<std::string> automatic_conversion_functions;
	void extract_automatic_conversion(FunctionDecl *fd);
	void extract_class_automatic_conversions(const isl_class &clazz);
	void extract_automatic_conversions();
public:
	static std::string drop_suffix(const std::string &s,
		const std::string &suffix);
	static void die(const char *msg) __attribute__((noreturn));
	static void die(string msg) __attribute__((noreturn));
	static vector<string> find_superclasses(Decl *decl);
	static bool is_subclass(FunctionDecl *decl);
	static bool is_overload(Decl *decl);
	static bool is_constructor(Decl *decl);
	static bool takes(Decl *decl);
	static bool keeps(Decl *decl);
	static bool gives(Decl *decl);
	static bool is_isl_ctx(QualType type);
	static bool first_arg_is_isl_ctx(FunctionDecl *fd);
	static bool is_isl_type(QualType type);
	static bool is_isl_neg_error(QualType type);
	static bool is_isl_bool(QualType type);
	static bool is_isl_stat(QualType type);
	static bool is_isl_size(QualType type);
	static bool is_long(QualType type);
	static bool is_callback(QualType type);
	static bool is_string(QualType type);
	static bool is_static(const isl_class &clazz, FunctionDecl *method);
	static bool is_mutator(const isl_class &clazz, FunctionDecl *fd);
	static string extract_type(QualType type);
	static const FunctionProtoType *extract_prototype(QualType type);
	static ParmVarDecl *persistent_callback_arg(FunctionDecl *fd);
};

#endif /* ISL_INTERFACE_GENERATOR_H */
