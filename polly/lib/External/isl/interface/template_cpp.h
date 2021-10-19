#ifndef ISL_INTERFACE_TEMPLATE_CPP_H
#define ISL_INTERFACE_TEMPLATE_CPP_H

#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "cpp.h"

struct Fixed;

struct TupleKind;

/* A shared pointer to a TupleKind.
 */
struct TupleKindPtr : public std::shared_ptr<const TupleKind> {
  using Base = std::shared_ptr<const TupleKind>;
  TupleKindPtr() = default;
  TupleKindPtr(Fixed);
  TupleKindPtr(Base base) : Base(base) {}
  TupleKindPtr(const std::string &name);
  TupleKindPtr(const TupleKindPtr &left, const TupleKindPtr &right);
};

/* A substitution mapping leaf tuple kind names to tuple kinds.
 */
using Substitution = std::unordered_map<std::string, TupleKindPtr>;

/* A representation of a (possibly improper) tuple kind.
 * That is, this also includes tuple kinds for types
 * that do not have any tuples.
 *
 * The kind could be a name (the base case) or
 * a (currently) unnamed nested pair of tuple kinds.
 */
struct TupleKind {
	TupleKind(const std::string &name) : name(name) {}

	virtual std::string to_string() const;
	virtual std::vector<std::string> params() const;
	virtual TupleKindPtr apply(const Substitution &subs,
		const TupleKindPtr &self) const;
	virtual TupleKindPtr left() const;
	virtual TupleKindPtr right() const;

	const std::string name;
};

/* A sequence of tuple kinds, representing a kind of objects.
 */
struct Kind : public std::vector<TupleKindPtr> {
	Kind() {}
	Kind(std::initializer_list<TupleKindPtr> list) : vector(list) {}

	bool is_anon() const;
	bool is_set() const;
	bool is_anon_set() const;
	std::vector<std::string> params() const;
	Kind apply(const Substitution &subs) const;
};

/* A representation of a template class.
 *
 * "class_name" is the name of the template class.
 * "super_name" is the (fully qualified) name of the corresponding
 * plain C++ interface class, from which this template class derives.
 * "clazz" describes the plain class.
 *
 * "class_tuples" contains the specializations.
 * It is initialized with a predefined set of specializations,
 * but may be extended during the generations of the specializations.
 */
struct template_class {
	const std::string class_name;
	const std::string super_name;
	const isl_class &clazz;

	std::vector<Kind> class_tuples;

	bool is_anon() const;
	bool is_anon_set() const;
	void add_specialization(const Kind &kind);
};

/* A generator for templated C++ bindings.
 *
 * "template_classes" contains all generated template classes,
 * keyed on their names.
 */
class template_cpp_generator : public cpp_generator {
	struct class_printer;
	struct method_decl_printer;
	struct method_impl_printer;
	struct class_decl_printer;
	struct class_impl_printer;

	void add_template_class(const isl_class &clazz, const std::string &name,
		const std::vector<Kind> &base_kinds);
public:
	template_cpp_generator(clang::SourceManager &SM,
		std::set<clang::RecordDecl *> &exported_types,
		std::set<clang::FunctionDecl *> exported_functions,
		std::set<clang::FunctionDecl *> functions);

	virtual void generate() override;
	void foreach_template_class(
		const std::function<void(const template_class &)> &fn) const;
	void print_forward_declarations(std::ostream &os);
	void print_friends(std::ostream &os);

	std::map<std::string, template_class> template_classes;
};

#endif
