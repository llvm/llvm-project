/*
 * Copyright 2020 Cerebras Systems. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY CEREBRAS SYSTEMS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CEREBRAS SYSTEMS OR
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
 * Cerebras Systems.
 */

#include <ctype.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "template_cpp.h"
#include "isl_config.h"

/* The textual representation of this tuple kind.
 *
 * By default, the textual representation is just the name.
 */
std::string TupleKind::to_string() const
{
	return name;
}

/* Return the parameters of this tuple kind.
 *
 * By default, there are no parameters.
 */
std::vector<std::string> TupleKind::params() const
{
	return { };
}

/* Apply the substitution "subs" to this tuple kind and return the result.
 * "self" is a shared pointer to this.
 *
 * If the name of this tuple kind appears in the substitution,
 * then return the corresponding tuple kind pointer.
 * Otherwise, return "self".
 */
TupleKindPtr TupleKind::apply(const Substitution &subs,
	const TupleKindPtr &self) const
{
	if (subs.count(name) != 0)
		return subs.at(name);
	return self;
}

/* Apply the substitution "subs" to "tuple" and return the result.
 */
static TupleKindPtr apply(const TupleKindPtr tuple, const Substitution &subs)
{
	return tuple->apply(subs, tuple);
}

/* Return the left child of this tuple kind.
 *
 * Since this is not a pair, there is no left child.
 */
TupleKindPtr TupleKind::left() const
{
	return TupleKindPtr();
}

/* Return the right child of this tuple kind.
 *
 * Since this is not a pair, there is no right child.
 */
TupleKindPtr TupleKind::right() const
{
	return TupleKindPtr();
}

/* Helper class used to construct a pointer to a tuple kind
 * that refers to a non-template type.
 */
struct Fixed {
};

/* Construct a pointer to a tuple kind that refers to a non-template type.
 *
 * Use an empty string as name.  Since this is a non-template type,
 * the kind name will never appear in the generated code.
 */
TupleKindPtr::TupleKindPtr(Fixed) : Base(std::make_shared<TupleKind>(""))
{
}

/* Tuple pointers for non-template types.
 */
static TupleKindPtr Ctx{Fixed()};
static TupleKindPtr Integer{Fixed()};
static TupleKindPtr Str{Fixed()};
static TupleKindPtr Res{Fixed()};

/* Special tuple pointers.
 * Anonymous appears in the generated code but cannot be unified
 * with anything else since it is a predefined template argument.
 * Leaf can only be unified with something that is not a pair and
 * does not appear in the generated code.
 */
static TupleKindPtr Anonymous("Anonymous");
static TupleKindPtr Leaf("Leaf");

/* Placeholder tuple pointers that refer to (part of) the domain or range.
 */
static TupleKindPtr Domain("Domain");
static TupleKindPtr Domain2("Domain2");
static TupleKindPtr Domain3("Domain3");
static TupleKindPtr Range("Range");
static TupleKindPtr Range2("Range2");
static TupleKindPtr Range3("Range3");

/* A representation of a proper tuple kind that is used as a template
 * parameter or a template argument.
 */
struct ProperTupleKind : public TupleKind {
	ProperTupleKind(const std::string &name) : TupleKind(name) {}

	virtual std::vector<std::string> params() const override;
};

/* Return the parameters of this tuple kind.
 *
 * Return the name of this tuple kind, unless it is the special Anonymous
 * predefined template argument.
 */
std::vector<std::string> ProperTupleKind::params() const
{
	if (Anonymous.get() == this)
		return { };
	return { name };
}

/* Construct a pointer to a tuple kind that refers
 * to a proper tuple kind with the given name.
 */
TupleKindPtr::TupleKindPtr(const std::string &name) :
	Base(std::make_shared<ProperTupleKind>(name))
{
}

/* A tuple kind that represents an anonymous pair of nested tuple kinds.
 */
struct Pair : public TupleKind {
	Pair(const TupleKindPtr &tuple1, const TupleKindPtr &tuple2) :
		TupleKind(""), tuple1(tuple1), tuple2(tuple2) {}

	virtual std::string to_string() const override;
	virtual std::vector<std::string> params() const override;
	virtual TupleKindPtr apply(const Substitution &match,
		const TupleKindPtr &self) const override;
	virtual TupleKindPtr left() const override;
	virtual TupleKindPtr right() const override;

	const TupleKindPtr tuple1;
	const TupleKindPtr tuple2;
};

/* The textual representation of this tuple kind.
 *
 * The textual representation of a pair is of the form "pair<tuple1, tuple2>".
 */
std::string Pair::to_string() const
{
	return std::string("pair<") + tuple1->to_string() + ", " +
					tuple2->to_string() + ">";
}

/* Add the elements of "vec2" that do not already appear in "vec1"
 * at the end of "vec1".
 *
 * The two vectors are assumed not to have any repeated elements.
 * The updated vector will then also not have repeated elements.
 */
static void combine(std::vector<std::string> &vec1,
	const std::vector<std::string> &vec2)
{
	for (const auto &s : vec2)
		if (std::find(vec1.begin(), vec1.end(), s) == vec1.end())
			vec1.emplace_back(s);
}

/* Return the parameters of this tuple kind.
 *
 * Combine the parameters of the two nested tuple kinds.
 */
std::vector<std::string> Pair::params() const
{
	auto names1 = tuple1->params();
	auto names2 = tuple2->params();

	combine(names1, names2);

	return names1;
}

/* Apply the substitution "subs" to this tuple kind and return the result.
 * "self" is a shared pointer to this.
 *
 * Construct a new tuple kind consisting of the result of applying
 * the substitution to the two nested tuple kinds.
 */
TupleKindPtr Pair::apply(const Substitution &subs, const TupleKindPtr &self)
	const
{
	return TupleKindPtr(::apply(tuple1, subs), ::apply(tuple2, subs));
}

/* Return the left child of this tuple kind.
 */
TupleKindPtr Pair::left() const
{
	return tuple1;
}

/* Return the right child of this tuple kind.
 */
TupleKindPtr Pair::right() const
{
	return tuple2;
}

/* Construct a pointer to a tuple kind that refers
 * to the given pair of nested tuple kinds.
 */
TupleKindPtr::TupleKindPtr(const TupleKindPtr &left, const TupleKindPtr &right)
	: Base(std::make_shared<Pair>(left, right))
{
}

/* Is this a kind of object representing an anonymous function?
 */
bool Kind::is_anon() const
{
	return size() != 0 && back() == Anonymous;
}

/* Is this a kind of object with a single tuple?
 */
bool Kind::is_set() const
{
	return size() == 1;
}

/* Is this a kind of object with a single, anonymous tuple?
 */
bool Kind::is_anon_set() const
{
	return is_set() && is_anon();
}

/* Return the parameters of this kind.
 *
 * Collect the parameters of the tuple kinds in the sequence.
 */
std::vector<std::string> Kind::params() const
{
	std::vector<std::string> params;

	for (const auto &tuple : *this)
		combine(params, tuple->params());

	return params;
}

/* Apply the substitution "subs" to this kind and return the result.
 *
 * Apply the substitution to each of the tuple kinds in the sequence.
 */
Kind Kind::apply(const Substitution &subs) const
{
	Kind applied;

	for (const auto &tuple : *this)
		applied.emplace_back(::apply(tuple, subs));

	return applied;
}

/* A signature of a method in terms of kinds,
 * consisting of a return kind and a sequence of argument kinds.
 */
struct Signature {
	Kind ret;
	std::vector<Kind> args;

	std::vector<std::string> params() const;
	Signature apply(const Substitution &match) const;
};

/* Return the parameters of this signature.
 *
 * Collect the parameters of the argument kinds and the return kind.
 */
std::vector<std::string> Signature::params() const
{
	std::vector<std::string> params;

	for (const auto &arg : args)
		combine(params, arg.params());
	combine(params, ret.params());

	return params;
}

/* Apply the substitution "subs" to this kind and return the result.
 *
 * Apply the substitution to the argument kinds and the return kind.
 */
Signature Signature::apply(const Substitution &subs) const
{
	std::vector<Kind> applied_args;

	for (const auto &arg : args)
		applied_args.emplace_back(arg.apply(subs));

	return { ret.apply(subs), applied_args };
}

/* Return a renaming substitution that renames the elements of "params"
 * using names starting with "prefix".
 */
static Substitution param_renamer(const std::vector<std::string> &params,
	const std::string &prefix)
{
	Substitution renamer;
	int n = 0;

	for (const auto &name : params) {
		auto suffix = std::to_string(++n);
		auto arg_name = prefix + suffix;
		auto arg = TupleKindPtr(arg_name);

		if (name == Leaf->name)
			generator::die("Leaf cannot be renamed");

		renamer.emplace(name, arg);
	}

	return renamer;
}

/* Does the vector "v" contain the element "el"?
 */
static bool contains(const std::vector<std::string> &v, const std::string &el)
{
	 return find(v.begin(), v.end(), el) != v.end();
 }


/* Return the shared elements of "v1" and "v2", preserving the order
 * of those elements in "v1".
 */
static std::vector<std::string> intersect(const std::vector<std::string> &v1,
	const std::vector<std::string> &v2)
{
	std::vector<std::string> intersection;

	for (const auto &el : v1)
		if (contains(v2, el))
			intersection.push_back(el);

	return intersection;
}

/* Return a renaming substitution that renames
 * any parameters that appears in both "sig" and "kind".
 */
static Substitution shared_param_renamer(const Signature &sig, const Kind &kind)
{
	return param_renamer(intersect(sig.params(), kind.params()), "Arg");
}

/* Signatures for unary operations.
 * Functions have at least one tuple.
 */
static Signature un_params = { { }, { { } } };
static Signature un_set = { { Domain }, { { Domain } } };
static Signature un_map = { { Domain, Range }, { { Domain, Range } } };
static std::vector<Signature> un_op = { un_params, un_set, un_map };
static std::vector<Signature> fn_un_op = { un_set, un_map };

/* Signatures for binary operations, with the second argument
 * possibly referring to part of the first argument.
 * Functions have at least one tuple.
 */
static Signature bin_params = { { }, { { }, { } } };
static Signature bin_set = { { Domain }, { { Domain }, { Domain } } };
static Signature bin_map =
	{ { Domain, Range }, { { Domain, Range }, { Domain, Range } } };
static std::vector<Signature> bin_op = { bin_params, bin_set, bin_map };
static std::vector<Signature> fn_bin_op = { bin_set, bin_map };
static Signature bin_set_params = { { Domain }, { { Domain }, { } } };
static Signature bin_map_params =
	{ { Domain, Range }, { { Domain, Range }, { } } };
static Signature bin_map_domain =
	{ { Domain, Range }, { { Domain, Range }, { Domain } } };
static Signature bin_map_range =
	{ { Domain, Range }, { { Domain, Range }, { Range } } };
static Signature bin_map_domain_wrapped_domain =
	{ { { Domain, Domain2 }, Range },
	  { { { Domain, Domain2 }, Range }, { Domain } } };
static Signature bin_map_range_wrapped_domain =
	{ { Domain, { Range, Range2 } },
	  { { Domain, { Range, Range2 } }, { Range } } };

/* Signatures for binary operations, where the second argument
 * is an identifier (with an anonymous tuple).
 */
static Signature bin_params_anon = { { }, { { }, { Anonymous } } };
static Signature bin_set_anon = { { Domain }, { { Domain }, { Anonymous } } };
static Signature bin_map_anon =
	{ { Domain, Range }, { { Domain, Range }, { Anonymous } } };
static std::vector<Signature> bin_op_anon =
	{ bin_params_anon, bin_set_anon, bin_map_anon };

/* Signatures for ternary operations, where the last two arguments are integers.
 */
static Signature ter_params_int_int =
	{ { }, { { }, { Integer }, { Integer } } };
static Signature ter_set_int_int =
	{ { Domain }, { { Domain }, { Integer }, { Integer } } };
static Signature ter_map_int_int =
	{ { Domain, Range }, { { Domain, Range }, { Integer }, { Integer } } };
static std::vector<Signature> ter_int_int =
	{ ter_params_int_int, ter_set_int_int, ter_map_int_int };

/* Signatures for ternary operations.
 * Functions have at least one tuple.
 */
static Signature ter_set =
	{ { Domain }, { { Domain }, { Domain }, { Domain } } };
static Signature ter_map =
	{ { Domain, Range },
	  { { Domain, Range }, { Domain, Range }, { Domain, Range } } };
static std::vector<Signature> fn_ter_op = { ter_set, ter_map };

/* Signatures for naming a leaf tuple using an identifier (with an anonymous
 * tuple).
 */
static Signature update_set = { { Domain2 }, { { Leaf }, { Anonymous } } };
static Signature update_domain =
	{ { Domain2, Range }, { { Leaf, Range }, { Anonymous } } };
static Signature update_range =
	{ { Domain, Range2 }, { { Domain, Leaf }, { Anonymous } } };

/* Signatures for the functions "min" and "max", which can be either
 * unary or binary operations.
 */
static std::vector<Signature> min_max = { un_set, bin_set, un_map, bin_map };

/* Signatures for adding an unnamed tuple to an object with zero or one tuple.
 */
static Signature to_set = { { Domain }, { { }, { Integer } } };
static Signature add_range = { { Domain, Range }, { { Domain }, { Integer } } };
/* Signatures for adding a named tuple to an object with zero or one tuple.
 */
static Signature to_set_named =
	{ { Domain }, { { }, { Anonymous }, { Integer } } };
static Signature add_range_named =
	{ { Domain, Range }, { { Domain }, { Anonymous }, { Integer } } };

/* Signatures for methods applying a map to a set, a function or
 * part of a map.
 */
static Signature set_forward = { { Range }, { { Domain }, { Domain, Range } } };
static Signature domain_forward =
	{ { Domain2, Range }, { { Domain, Range }, { Domain, Domain2 } } };
static Signature range_forward =
	{ { Domain, Range2 }, { { Domain, Range }, { Range, Range2 } } };

/* Signatures for methods plugging in a function into a set, a function or
 * part of a map.
 */
static Signature set_backward =
	{ { Domain2 }, { { Domain }, { Domain2, Domain } } };
static Signature domain_backward =
	{ { Domain2, Range }, { { Domain, Range }, { Domain2, Domain } } };
static Signature range_backward =
	{ { Domain, Range2 }, { { Domain, Range }, { Range2, Range } } };
static Signature domain_wrapped_domain_backward =
	{ { { Domain3, Domain2 }, Range },
	  { { { Domain, Domain2 }, Range }, { Domain3, Domain } } };

/* Signatures for methods binding a set, a function,
 * or (part of) a map to parameters or an object of the same kind.
 */
static Signature bind_set = { { }, { { Domain }, { Domain } } };
static Signature bind_domain = { { Range }, { { Domain, Range }, { Domain } } };
static Signature bind_range = { { Domain }, { { Domain, Range }, { Range } } };
static Signature bind_domain_wrapped_domain =
	{ { Range2, Range }, { { { Domain2, Range2 }, Range }, { Domain2 } } };

/* Signatures for functions that take a callback accepting
 * objects of the same kind (but a different type).
 *
 * The return and argument kinds of the callback appear
 * at the position of the callback.
 */
static Signature each_params = { { Res }, { { }, { Res }, { } } };
static Signature each_set = { { Res }, { { Domain }, { Res }, { Domain } } };
static Signature each_map =
	{ { Res }, { { Domain, Range }, { Res }, { Domain, Range } } };
static std::vector<Signature> each = { each_params, each_set, each_map };

/* Signatures for isl_*_list_foreach_scc.
 *
 * The first callback takes two elements with the same tuple kinds.
 * The second callback takes a list with the same tuple kinds.
 */
static Signature each_scc_params =
	{ { Res }, { { }, { Res }, { }, { }, { Res }, { } } };
static Signature each_scc_set =
	{ { Res }, { { Domain },
		     { Res }, { Domain }, { Domain },
		     { Res }, { Domain } } };
static Signature each_scc_map =
	{ { Res }, { { Domain, Range },
		     { Res }, { Domain, Range }, { Domain, Range },
		     { Res }, { Domain, Range } } };
static std::vector<Signature> each_scc =
	{ each_scc_params, each_scc_set, each_scc_map };

/* Signature for creating a map from a range,
 * where the domain is given by an extra argument.
 */
static Signature map_from_range_and_domain =
	{ { Domain, Range }, { { Range }, { Domain } } };

/* Signature for creating a map from a domain,
 * where the range is given by an extra argument.
 */
static Signature map_from_domain_and_range =
	{ { Domain, Range }, { { Domain }, { Range } } };

/* Signatures for creating an anonymous set from a parameter set
 * or a map from a domain, where the range is anonymous.
 */
static Signature anonymous_set_from_params = { { Anonymous }, { { } } };
static Signature anonymous_map_from_domain =
	{ { Domain, Anonymous }, { { Domain } } };
static std::vector<Signature> anonymous_from_domain =
	{ anonymous_set_from_params, anonymous_map_from_domain };

/* Signature for creating a set from a parameter set,
 * where the domain is given by an extra argument.
 */
static Signature set_from_params = { { Domain }, { { }, { Domain } } };

/* Signatures for creating an anonymous function from a domain,
 * where the second argument is an identifier (with an anonymous tuple).
 */
static Signature anonymous_set_from_params_bin_anon =
	{ { Anonymous }, { { }, { Anonymous } } };
static Signature anonymous_map_from_domain_bin_anon =
	{ { Domain, Anonymous }, { { Domain }, { Anonymous } } };
static std::vector<Signature> anonymous_from_domain_bin_anon = {
	  anonymous_set_from_params_bin_anon,
	  anonymous_map_from_domain_bin_anon
	};

/* Signature for creating a map from a domain,
 * where the range tuple is equal to the domain tuple.
 */
static Signature set_to_map = { { Domain, Domain }, { { Domain } } };

/* Signatures for obtaining the range or the domain of a map.
 * In case of a transformation, the domain and range are the same.
 */
static Signature domain = { { Domain }, { { Domain, Range } } };
static Signature range = { { Range }, { { Domain, Range } } };
static Signature transformation_domain = { { Domain }, { { Domain, Domain } } };

/* Signatures for obtaining the parameter domain of a set or map.
 */
static Signature set_params = { { }, { { Domain } } };
static Signature map_params = { { }, { { Domain, Range } } };

/* Signatures for obtaining the domain of a function.
 */
static std::vector<Signature> fn_domain = { domain, set_params };

/* Signatures for interchanging (wrapped) domain and range.
 */
static Signature map_reverse = { { Range, Domain }, { { Domain, Range } } };
static Signature map_range_reverse =
	{ { Domain, { Range2, Range } }, { { Domain, { Range, Range2 } } } };

/* Signatures for constructing products.
 */
static Signature set_product =
	{ { { Domain, Range } }, { { Domain }, { Range } } };
static Signature map_product =
	{ { { Domain, Domain2 }, { Range, Range2 } },
	  { { Domain, Range }, { Domain2, Range2 } } };
static Signature domain_product =
	{ { { Domain, Domain2 }, Range },
	  { { Domain, Range }, { Domain2, Range } } };
static Signature range_product =
	{ { Domain, { Range, Range2 } },
	  { { Domain, Range }, { Domain, Range2 } } };

/* Signatures for obtaining factors from a product.
 */
static Signature domain_factor_domain =
	{ { Domain, Range }, { { { Domain, Domain2 }, Range } } };
static Signature domain_factor_range =
	{ { Domain2, Range }, { { { Domain, Domain2 }, Range } } };
static Signature range_factor_domain =
	{ { Domain, Range }, { { Domain, { Range, Range2 } } } };
static Signature range_factor_range =
	{ { Domain, Range2 }, { { Domain, { Range, Range2 } } } };

/* Signatures for (un)currying.
 */
static Signature curry =
	{ { Domain, { Range, Range2 } },
	  { { { Domain, Range }, Range2 } } };
static Signature uncurry =
	{ { { Domain, Range }, Range2 },
	  { { Domain, { Range, Range2 } } } };

/* Signatures for (un)wrapping.
 */
static Signature wrap = { { { Domain, Range } }, { { Domain, Range } } };
static Signature unwrap = { { Domain, Range }, { { { Domain, Range } } } };

/* Signatures for constructing objects that map to the domain or range
 * of a map.
 */
static Signature domain_map =
	{ { { Domain, Range }, Domain }, { { Domain, Range } } };
static Signature range_map =
	{ { { Domain, Range }, Range }, { { Domain, Range } } };

/* Signature for applying a comparison between the domain and the range
 * of a map.
 */
static Signature map_cmp =
	{ { Domain, Domain }, { { Domain, Domain }, { Domain, Range } } };

/* Signature for creating a set corresponding to the domains
 * of two functions.
 */
static Signature set_join =
	{ { Domain }, { { Domain, Range }, { Domain, Range } } };

/* Signatures for flattening the domain or range of a map,
 * replacing it with either an anonymous tuple or a tuple with a given name.
 */
static Signature anonymize_nested_domain =
	{ { Anonymous, Range2 }, { { { Domain, Range }, Range2 } } };
static Signature anonymize_nested_range =
	{ { Domain, Anonymous }, { { Domain, { Range, Range2 } } } };
static Signature replace_nested_domain =
	{ { Domain2, Range2 },
	  { { { Domain, Range }, Range2 }, { Anonymous} } };
static Signature replace_nested_range =
	{ { Domain, Range3 }, { { Domain, { Range, Range2 } }, { Anonymous} } };
static std::vector<Signature> flatten_domain =
	{ anonymize_nested_domain, replace_nested_domain };
static std::vector<Signature> flatten_range =
	{ anonymize_nested_range, replace_nested_range };

/* Signatures for "set_at" methods.
 */
static Signature set_at_set =
	{ { Domain }, { { Domain }, { Integer }, { Anonymous } } };
static Signature set_at_map =
	{ { Domain, Range },
	  { { Domain, Range }, { Integer }, { Domain, Anonymous } } };
static std::vector<Signature> set_at = { set_at_set, set_at_map };

/* Signatures for "list" methods, extracting a list
 * from a multi-expression.
 */
static Signature to_list_set = { { Anonymous }, { { Domain } } };
static Signature to_list_map = { { Domain, Anonymous }, { { Domain, Range } } };

/* Signatures for functions constructing an object from only an isl::ctx.
 */
static Signature ctx_params = { { }, { { Ctx } } };
static Signature ctx_set = { { Domain }, { { Ctx } } };
static Signature ctx_map = { { Domain, Range }, { { Ctx } } };

/* Helper structure for sorting the keys of static_methods and
 * special_member_methods such that the larger keys appear first.
 * In particular, a key should appear before any key that appears
 * as a substring in the key.
 * Note that this sorting is currently only important
 * for special_member_methods.
 */
struct larger_infix {
	bool operator()(const std::string &x, const std::string &y) const {
		if (x.length() > y. length())
			return true;
		return x < y;
	}
};

/* A map from part of a type name to a sequence of signatures.
 */
typedef std::map<std::string, std::vector<Signature>, larger_infix> infix_map;

/* A map from a method name to a map from part of a type name
 * to a sequence of signatures.
 */
typedef std::map<std::string, infix_map> infix_map_map;

/* Signatures for static methods.
 *
 * The "unit" static method is only available in a 0-tuple space.
 *
 * The "empty" static method creates union objects with the relevant
 * number of tuples.
 *
 * The "universe" static methods create objects from the corresponding spaces.
 */
static const infix_map_map static_methods {
	{ "unit",
	  { { "space",			{ ctx_params } } }
	},
	{ "empty",
	  {
	    { "union_set",		{ ctx_params, ctx_set } },
	    { "union_map",		{ ctx_map } },
	    { "union_pw_multi_aff",	{ ctx_set, ctx_map } },
	  }
	},
	{ "universe",
	  {
	    { "set",			{ un_params, un_set } },
	    { "map",			{ un_map } },
	  }
	},
};

/* Signatures for unary operations that either take something in a set space
 * and return something in the same space or take something in a map space
 * and return something in the range of that space.
 */
static std::vector<Signature> range_op = { un_set, range };

/* Signatures for binary operations where the second argument
 * is a (multi-)value.
 */
static std::vector<Signature> bin_val = { bin_set, bin_map_range };

/* The (default) signatures for methods with a given name.
 * Some of these are overridden by special_member_methods.
 */
static const std::unordered_map<std::string, std::vector<Signature>>
member_methods {
	{ "add",		bin_op },
	{ "add_constant",	bin_val },
	{ "add_named_tuple",	{ to_set_named, add_range_named } },
	{ "add_param",		bin_op_anon },
	{ "add_unnamed_tuple",	{ to_set, add_range } },
	{ "apply",		{ set_forward, range_forward } },
	{ "apply_domain",	{ domain_forward } },
	{ "apply_range",	{ range_forward } },
	{ "as",			un_op },
	{ "as_map",		{ un_map } },
	{ "as_union_map",	{ un_map } },
	{ "as_set",		{ un_set } },
	{ "bind",		{ bind_set, bind_range } },
	{ "bind_domain",	{ bind_domain } },
	{ "bind_range",		{ bind_range } },
	{ "bind_domain_wrapped_domain",
				{ bind_domain_wrapped_domain } },
	{ "ceil",		fn_un_op },
	{ "coalesce",		un_op },
	{ "cond",		fn_ter_op },
	{ "constant",		range_op },
	{ "curry",		{ curry } },
	{ "deltas",		{ transformation_domain } },
	{ "detect_equalities",	un_op },
	{ "domain",		fn_domain },
	{ "domain_factor_domain",
				{ domain_factor_domain } },
	{ "domain_factor_range",
				{ domain_factor_range } },
	{ "domain_map",		{ domain_map } },
	{ "domain_product",	{ domain_product } },
	{ "drop",		ter_int_int },
	{ "eq_at",		{ map_cmp } },
	{ "every",		each },
	{ "extract",		bin_op },
	{ "flatten_domain",	flatten_domain },
	{ "flatten_range",	flatten_range },
	{ "floor",		fn_un_op },
	{ "foreach",		each },
	{ "foreach_scc",	each_scc },
	{ "ge_set",		{ set_join } },
	{ "gt_set",		{ set_join } },
	{ "gist",		bin_op },
	{ "gist_domain",	{ bin_map_domain } },
	{ "gist_params",	{ bin_set_params, bin_map_params } },
	{ "identity",		{ un_map, set_to_map } },
	{ "identity_on_domain",	{ set_to_map } },
	{ "indicator_function",	anonymous_from_domain },
	{ "insert_domain",	{ map_from_range_and_domain } },
	{ "intersect",		bin_op },
	{ "intersect_params",	{ bin_set_params, bin_map_params } },
	{ "intersect_domain",	{ bin_map_domain } },
	{ "intersect_domain_wrapped_domain",
				{ bin_map_domain_wrapped_domain } },
	{ "intersect_range",	{ bin_map_range } },
	{ "intersect_range_wrapped_domain",
				{ bin_map_range_wrapped_domain } },
	{ "lattice_tile",	{ un_set } },
	{ "le_set",		{ set_join } },
	{ "lt_set",		{ set_join } },
	{ "lex_le_at",		{ map_cmp } },
	{ "lex_lt_at",		{ map_cmp } },
	{ "lex_ge_at",		{ map_cmp } },
	{ "lex_gt_at",		{ map_cmp } },
	{ "lexmin",		fn_un_op },
	{ "lexmax",		fn_un_op },
	{ "list",		{ to_list_set, to_list_map } },
	{ "lower_bound",	fn_bin_op },
	{ "map_from_set",	{ set_to_map } },
	{ "max",		min_max },
	{ "max_val",		range_op },
	{ "max_multi_val",	range_op },
	{ "min",		min_max },
	{ "min_val",		range_op },
	{ "min_multi_val",	range_op },
	{ "mod",		bin_val },
	{ "on_domain",		{ map_from_domain_and_range } },
	{ "neg",		fn_un_op },
	{ "offset",		fn_un_op },
	{ "param_on_domain",	anonymous_from_domain_bin_anon },
	{ "params",		{ set_params, map_params } },
	{ "plain_multi_val_if_fixed",
				{ un_set } },
	{ "preimage",		{ set_backward } },
	{ "preimage_domain",	{ domain_backward } },
	{ "preimage_domain_wrapped_domain",
				{ domain_wrapped_domain_backward } },
	{ "preimage_range",	{ range_backward } },
	{ "product",		{ set_product, map_product } },
	{ "project_out_param",	bin_op_anon },
	{ "project_out_all_params",
				un_op },
	{ "pullback",		{ domain_backward, bind_domain } },
	{ "range",		{ range } },
	{ "range_factor_domain",
				{ range_factor_domain } },
	{ "range_factor_range",	{ range_factor_range } },
	{ "range_lattice_tile",	{ un_map } },
	{ "range_map",		{ range_map } },
	{ "range_product",	{ range_product } },
	{ "range_reverse",	{ map_range_reverse } },
	{ "range_simple_fixed_box_hull",
				{ un_map } },
	{ "reverse",		{ map_reverse } },
	{ "scale",		bin_val },
	{ "scale_down",		bin_val },
	{ "set_at",		set_at },
	{ "set_domain_tuple",	{ update_domain } },
	{ "set_range_tuple",	{ update_set, update_range } },
	{ "simple_fixed_box_hull",
				{ un_set } },
	{ "sub",		fn_bin_op },
	{ "subtract",		bin_op },
	{ "subtract_domain",	{ bin_map_domain } },
	{ "subtract_range",	{ bin_map_range } },
	{ "translation",	{ set_to_map } },
	{ "to",			un_op },
	{ "unbind_params",	{ set_from_params } },
	{ "unbind_params_insert_domain",
				{ map_from_range_and_domain } },
	{ "uncurry",		{ uncurry } },
	{ "union_add",		fn_bin_op },
	{ "unite",		bin_op },
	{ "universe",		un_op },
	{ "unwrap",		{ unwrap } },
	{ "upper_bound",	fn_bin_op },
	{ "wrap",		{ wrap } },
	{ "zero",		fn_un_op },
	{ "zero_on_domain",	{ anonymous_map_from_domain } },
};

/* Signatures for methods of types containing a given substring
 * that override the default signatures, where larger substrings
 * appear first.
 *
 * In particular, "gist" is usually a regular binary operation,
 * but for any type derived from "aff", the argument refers
 * to the domain of the function.
 *
 * The "size" method can usually simply be inherited from
 * the corresponding plain C++ type, but for a "fixed_box",
 * the size lives in the space of the box or its range.
 *
 * The "space" method is usually a regular unary operation
 * that returns the single space of the elements in the object,
 * with the same number of tuples.
 * However, a "union" object may contain elements from many spaces and
 * therefore its space only refers to the symbolic constants and
 * has zero tuples, except if it is also a "multi_union" object,
 * in which case it has a fixed range space and the space of the object
 * has a single tuple.
 * Note that since "space' is also the name of a template class,
 * the default space method is handled by print_type_named_member_method.
 */
static const infix_map_map special_member_methods {
	{ "gist",
	  { { "aff",		{ bin_set_params, bin_map_domain } } }
	},
	{ "size",
	  { { "fixed_box",	range_op } },
	},
	{ "space",
	  {
	    { "multi_union",	range_op },
	    { "union",		{ un_params, set_params, map_params } },
	  }
	},
};

/* Generic kinds for objects with zero, one or two tuples,
 * the last of which may be anonymous.
 */
static Kind params{};
static Kind set_type{ Domain };
static Kind set_anon{ Anonymous };
static Kind map_type{ Domain, Range };
static Kind map_anon{ Domain, Anonymous };

/* The initial sequence of specialization kinds for base types.
 * The specialization kinds for other types are derived
 * from the corresponding base types.
 *
 * In particular, this sequence specifies how many tuples
 * a given type can have and whether it is anonymous.
 *
 * "space" can have any number of tuples.
 * "set" and "point" can have zero or one tuple.
 * "map" can only have two tuples.
 * "aff" can have one or two tuples, the last of which is anonymous.
 * "fixed_box" can represent a (proper) set) or a map.
 * "val" and "id" are treated as anonymous sets so that
 * they can form the basis of "multi_val" and "multi_id".
 */
static const std::unordered_map<std::string, std::vector<Kind>> base_kinds {
	{ "space",	{ params, set_type, map_type } },
	{ "set",	{ params, set_type } },
	{ "point",	{ params, set_type } },
	{ "map",	{ map_type } },
	{ "aff",	{ set_anon, map_anon } },
	{ "fixed_box",	{ set_type, map_type } },
	{ "val",	{ set_anon } },
	{ "id",		{ set_anon } },
};

/* Prefixes introduced by type constructors.
 */
static const std::unordered_set<std::string> type_prefixes {
	"basic",
	"multi",
	"pw",
	"union",
};

/* If "type" has a "_list" suffix, then return "type" with this suffix removed.
 * Otherwise, simply return "type".
 */
static std::string drop_list(const std::string &type)
{
	size_t pos = type.rfind('_');

	if (pos == std::string::npos)
		return type;
	if (type.substr(pos + 1) == "list")
		return type.substr(0, pos);
	return type;
}

/* Given the name of a plain C++ type, return the base type
 * from which it was derived using type constructors.
 *
 * In particular, drop any "list" suffix and
 * drop any prefixes from type_prefixes, stopping
 * as soon as a base type is found for which kinds have been registered
 * in base_kinds.
 */
static std::string base_type(const std::string &type)
{
	auto base = type;
	size_t pos;

	base = drop_list(base);
	while (base_kinds.count(base) == 0 &&
			(pos = base.find('_')) != std::string::npos &&
			type_prefixes.count(base.substr(0, pos)) != 0) {
		base = base.substr(pos + 1);
	}

	return base;
}

/* A mapping from anonymous kinds to named kinds.
 */
static std::map<Kind, Kind> anon_to_named {
	{ set_anon, set_type },
	{ map_anon, map_type },
};

/* Given a sequence of anonymous kinds, replace them
 * by the corresponding named kinds.
 */
static std::vector<Kind> add_name(const std::vector<Kind> &tuples)
{
	std::vector<Kind> named;

	for (const auto &tuple : tuples)
		named.emplace_back(anon_to_named.at(tuple));

	return named;
}

/* Look up the (initial) specializations of the class called "name".
 * If no specializations have been defined, then return an empty vector.
 *
 * Start from the initial specializations of the corresponding base type.
 * If this template class is a multi-expression, then it was derived
 * from an anonymous function type.  Replace the final Anonymous
 * tuple kind by a placeholder in this case.
 */
static std::vector<Kind> lookup_class_tuples(const std::string &name)
{
	std::string base = base_type(name);

	if (base_kinds.count(base) == 0)
		return { };
	if (name.find("multi_") != std::string::npos)
		return add_name(base_kinds.at(base));
	return base_kinds.at(base);
}

/* Add a template class called "name", of which the methods are described
 * by "clazz" and the initial specializations by "class_tuples".
 */
void template_cpp_generator::add_template_class(const isl_class &clazz,
	const std::string &name, const std::vector<Kind> &class_tuples)
{
	auto isl_namespace = cpp_type_printer().isl_namespace();
	auto super = isl_namespace + name;

	template_classes.emplace(name,
		template_class{name, super, clazz, class_tuples});
}

/* Construct a templated C++ bindings generator from
 * the exported types and functions and the set of all declared functions.
 *
 * On top of the initialization of the shared parts
 * of C++ bindings generators, add a template class
 * for each plain C++ class for which template kinds
 * have been defined.
 * In particular, determine the base type from which the plain C++ class
 * was derived using type constructors and check if any template kinds
 * have been registered for this base type.
 */
template_cpp_generator::template_cpp_generator(clang::SourceManager &SM,
	std::set<clang::RecordDecl *> &exported_types,
	std::set<clang::FunctionDecl *> exported_functions,
	std::set<clang::FunctionDecl *> functions) :
		cpp_generator(SM, exported_types, exported_functions,
			functions)
{
	for (const auto &kvp : classes) {
		const auto &clazz = kvp.second;
		std::string name = type2cpp(clazz);
		const auto &class_tuples = lookup_class_tuples(name);

		if (class_tuples.empty())
			continue;
		add_template_class(clazz, name, class_tuples);
	}
}

/* Call "fn" on each template class.
 */
void template_cpp_generator::foreach_template_class(
	const std::function<void(const template_class &)> &fn) const
{
	for (const auto &kvp : template_classes)
		fn(kvp.second);
}

/* Print forward declarations for all template classes to "os".
 *
 * For template classes that represent an anonymous function
 * that can also have a domain tuple, provide an <name>_on alias
 * that adds the fixed Anonymous tuple kind.
 */
void template_cpp_generator::print_forward_declarations(std::ostream &os)
{
	foreach_template_class([&os] (const template_class &template_class) {
		auto name = template_class.class_name;

		os << "\n";
		os << "template <typename...>\n";
		os << "struct " << name << ";\n";

		if (!template_class.is_anon())
			return;
		if (template_class.is_anon_set())
			return;

		os << "\n";
		os << "template <typename...Ts>\n";
		os << "using " << name << "_on = "
		   << name << "<Ts..., Anonymous>;\n";
	});
}

/* Print friend declarations for all template classes to "os".
 */
void template_cpp_generator::print_friends(std::ostream &os)
{
	foreach_template_class([&os] (const template_class &template_class) {
		os << "  template <typename...>\n";
		os << "  friend struct " << template_class.class_name << ";\n";
	});
}

/* Print a template parameter or argument.
 * In case of a std::string, it's a template parameter
 * that needs to be declared.
 */
static void print_template_arg(std::ostream &os, const std::string &arg)
{
	os << "typename " << arg;
}

/* Print a template parameter or argument.
 * In case of a TupleKindPtr, it's a template argument.
 */
static void print_template_arg(std::ostream &os, const TupleKindPtr &kind)
{
	os << kind->to_string();
}

/* Print a sequence of template parameters (std::string) or
 * arguments (TupleKindPtr) "args", without the enclosing angle brackets.
 */
template <typename List>
static void print_pure_template_args(std::ostream &os, const List &args)
{
	for (size_t i = 0; i < args.size(); ++i) {
		if (i != 0)
			os << ", ";
		print_template_arg(os, args[i]);
	}
}

/* Print a sequence of template parameters (std::string) or
 * arguments (TupleKindPtr) "args".
 */
template <typename List>
static void print_template_args(std::ostream &os, const List &args)
{
	os << "<";
	print_pure_template_args(os, args);
	os << ">";
}

/* Print a declaration of the template parameters "params".
 */
static void print_template(std::ostream &os,
	const std::vector<std::string> &params)
{
	os << "template ";
	print_template_args(os, params);
	os << "\n";
}

/* Print a declaration of the template parameters "params",
 * if there are any.
 */
static void print_non_empty_template(std::ostream &os,
	const std::vector<std::string> &params)
{
	if (params.size() > 0)
		print_template(os, params);
}

/* Print a bare template type, i.e., without namespace,
 * consisting of the type "type" and the kind "kind" to "os".
 *
 * In particular, print "type" followed by the template arguments
 * as specified by "kind".
 */
static void print_bare_template_type(std::ostream &os, const std::string &type,
	const Kind &kind)
{
	os << type;
	print_template_args(os, kind);
}

/* A specific instance of "template_class", with tuple kinds given by "kind".
 */
struct specialization {
	struct template_class &template_class;
	Kind kind;

	const std::string &base_name() const;
	const std::string &class_name() const;
};

/* The name of the plain C++ interface class
 * from which this template class (instance) derives.
 */
const std::string &specialization::base_name() const
{
	return template_class.super_name;
}

/* The name of the template class.
 */
const std::string &specialization::class_name() const
{
	return template_class.class_name;
}

/* Helper class for printing the specializations of template classes
 * that is used to print both the class declarations and the class definitions.
 *
 * "os" is the stream onto which the classes should be printed.
 * "generator" is the templated C++ interface generator printing the classes.
 */
struct specialization_printer {
	specialization_printer(std::ostream &os,
			template_cpp_generator &generator) :
		os(os), generator(generator) {}

	virtual void print_class(const specialization &instance) const = 0;
	void print_classes() const;

	std::ostream &os;
	template_cpp_generator &generator;
};

/* Print all specializations of all template classes.
 *
 * Each class has a predefined set of initial specializations,
 * but while such a specialization is being printed,
 * the need for other specializations may arise and
 * these are added at the end of the list of specializations.
 * That is, class_tuples.size() may change during the execution
 * of the loop.
 *
 * For each specialization of a template class, call
 * the print_class virtual method.
 */
void specialization_printer::print_classes() const
{
	for (auto &kvp : generator.template_classes) {
		auto &template_class = kvp.second;
		const auto &class_tuples = template_class.class_tuples;

		for (size_t i = 0; i < class_tuples.size(); ++i)
			print_class({ template_class, class_tuples[i] });
	}
}

/* A helper class for printing method declarations and definitions
 * of a template class specialization.
 *
 * "instance" is the template class specialization for which methods
 * are printed.
 * "generator" is the templated C++ interface generator printing the classes.
 */
struct template_cpp_generator::class_printer :
		public cpp_generator::class_printer {
	class_printer(const specialization &instance,
			const specialization_printer &instance_printer,
			bool is_declaration);

	void print_return_type(const Method &method, const Kind &kind)
		const;
	void print_method_template_arguments(const Signature &sig);
	void print_method_header(const Method &method, const Signature &sig);
	bool print_special_method(const Method &method,
		const infix_map_map &special_methods);
	void print_static_method(const Method &method);
	void print_constructor(const Method &method);
	bool is_return_kind(const Method &method, const Kind &return_kind);
	void add_specialization(const Kind &kind);
	bool print_matching_method(const Method &method, const Signature &sig,
		const Kind &match_arg);
	bool print_matching_method(const Method &method, const Signature &sig);
	void print_matching_method(const Method &method,
		const std::vector<Signature> &signatures);
	void print_at_method(const Method &method);
	bool print_special_member_method(const Method &method);
	bool print_type_named_member_method(const Method &method);
	bool print_member_method_with_name(const Method &method,
		const std::string &name);
	void print_member_method(const Method &method);
	void print_any_method(const Method &method);
	virtual void print_method(const Method &method) override;
	virtual void print_method(const ConversionMethod &method) override;
	virtual void print_method_sig(const Method &method,
		const Signature &sig, bool deleted) = 0;
	virtual bool want_descendent_overloads(const function_set &methods)
		override;
	void print_all_methods();

	const specialization &instance;
	template_cpp_generator &generator;
};

/* Construct a class_printer from the template class specialization
 * for which methods are printed and
 * the printer of the template class.
 *
 * The template class printer is only used to obtain the output stream and
 * the templated C++ interface generator printing the classes.
 */
template_cpp_generator::class_printer::class_printer(
		const specialization &instance,
		const specialization_printer &instance_printer,
		bool is_declaration) :
	cpp_generator::class_printer(instance_printer.os,
		instance.template_class.clazz, instance_printer.generator,
		is_declaration),
	instance(instance), generator(instance_printer.generator)
{
}

/* An abstract template type printer, where the way of obtaining
 * the argument kind is specified by the subclasses.
 */
struct template_cpp_type_printer : public cpp_type_printer {
	template_cpp_type_printer() {}

	std::string base(const std::string &type, const Kind &kind) const;
	virtual Kind kind(int arg) const = 0;
	virtual std::string qualified(int arg, const std::string &cpp_type)
		const override;
};

/* Print a template type consisting of the type "type" and the kind "kind",
 * including the "typed::" namespace specifier.
 */
std::string template_cpp_type_printer::base(const std::string &type,
	const Kind &kind) const
{
	std::ostringstream ss;

	ss << "typed::";
	print_bare_template_type(ss, type, kind);
	return ss.str();
}

/* Return the qualified form of the given C++ isl type name appearing
 * in argument position "arg" (-1 for return type).
 *
 * isl::ctx is not templated, so if "cpp_type" is "ctx",
 * then print a non-templated version.
 * Otherwise, look up the kind of the argument and print
 * the corresponding template type.
 */
std::string template_cpp_type_printer::qualified(int arg,
	const std::string &cpp_type) const
{
	if (cpp_type == "ctx")
		return cpp_type_printer::qualified(arg, cpp_type);

	return base(cpp_type, kind(arg));
}

/* A template type printer for printing types with a fixed kind.
 *
 * "fixed_kind" is the fixed kind.
 */
struct template_cpp_kind_type_printer : public template_cpp_type_printer {
	template_cpp_kind_type_printer(const Kind &kind) :
		template_cpp_type_printer(), fixed_kind(kind) {}

	virtual Kind kind(int arg) const override;

	const Kind &fixed_kind;
};

/* Return the kind of the argument at position "arg",
 * where position -1 refers to the return type.
 *
 * Always use the fixed kind.
 */
Kind template_cpp_kind_type_printer::kind(int arg) const
{
	return fixed_kind;
}

/* A template type printer for printing a method with a given signature.
 *
 * "sig" is the signature of the method being printed.
 */
struct template_cpp_arg_type_printer : public template_cpp_type_printer {
	template_cpp_arg_type_printer(const Signature &sig) :
		template_cpp_type_printer(), sig(sig) {}

	virtual Kind kind(int arg) const override;

	const Signature &sig;
};

/* Return the kind of the argument at position "arg",
 * where position -1 refers to the return type.
 *
 * Look up the kind in the signature.
 */
Kind template_cpp_arg_type_printer::kind(int arg) const
{
	int n_args = sig.args.size();

	if (arg < 0)
		return sig.ret;
	if (arg >= n_args)
		generator::die("argument out of bounds");
	return sig.args[arg];
}

/* A template type printer for printing a method with a given signature
 * as part of a template class specialization of a given kind.
 *
 * "class_kind" is the template class specialization kind.
 */
struct template_method_type_printer : public template_cpp_arg_type_printer {
	template_method_type_printer(const Signature &sig,
			const Kind &class_kind) :
		template_cpp_arg_type_printer(sig),
		class_kind(class_kind) {}

	virtual std::string class_type(const std::string &cpp_name)
		const override;

	const Kind &class_kind;
};

/* Print the class type "cpp_name".
 *
 * Print the templated version using the template class specialization kind.
 */
std::string template_method_type_printer::class_type(
	const std::string &cpp_name) const
{
	return base(cpp_name, class_kind);
}

/* Print the templated return type of "method" of the kind "return_kind".
 *
 * Construct a type printer with "return_kind" as fixed kind and
 * use it to print the return type.
 */
void template_cpp_generator::class_printer::print_return_type(
	const Method &method, const Kind &return_kind) const
{
	template_cpp_kind_type_printer printer(return_kind);

	os << printer.return_type(method);
}

/* Remove the initial "n" elements from "v".
 */
template <typename T>
static void drop_initial(std::vector<T> &v, size_t n)
{
	v.erase(v.begin(), v.begin() + n);
}

/* If a method with signature "sig" requires additional template parameters
 * compared to those of the class, then print a declaration for them.
 * If this->declarations is set, then this will be part of a method declaration,
 * requiring extra indentation.
 *
 * Construct the sequence of all required template parameters
 * with those of the template class appearing first.
 * If this sequence has any parameters not induced by the template class itself,
 * then print a declaration for these extra parameters.
 */
void template_cpp_generator::class_printer::print_method_template_arguments(
	const Signature &sig)
{
	std::vector<std::string> class_params, method_params;

	class_params = instance.kind.params();
	method_params = class_params;
	combine(method_params, sig.params());

	if (class_params.size() == method_params.size())
		return;

	drop_initial(method_params, class_params.size());

	if (declarations)
		os << "  ";
	print_template(os, method_params);
}

/* Print the header for "method" with signature "sig".
 *
 * First print any additional template parameters that may be required and
 * then print a regular method header, using a template type printer.
 */
void template_cpp_generator::class_printer::print_method_header(
	const Method &method, const Signature &sig)
{
	template_method_type_printer type_printer(sig, instance.kind);

	print_method_template_arguments(sig);
	cpp_generator::class_printer::print_method_header(method,
							type_printer);
}

/* Given a group of methods with the same name,
 * should extra methods be added that take as arguments
 * those types that can be converted to the original argument type
 * through a unary constructor?
 *
 * Since type deduction does not consider implicit conversions,
 * these extra methods should always be printed.
 */
bool template_cpp_generator::class_printer::want_descendent_overloads(
	const function_set &methods)
{
	return true;
}

/* Print all constructors and methods that forward
 * to the corresponding methods in the plain C++ interface class.
 */
void template_cpp_generator::class_printer::print_all_methods()
{
	print_constructors();
	print_methods();
}

/* A helper class for printing method declarations
 * of a template class specialization.
 */
struct template_cpp_generator::method_decl_printer :
		public template_cpp_generator::class_printer {
	method_decl_printer(const specialization &instance,
			const struct specialization_printer &instance_printer) :
		class_printer(instance, instance_printer, true) {}

	virtual void print_method_sig(const Method &method,
		const Signature &sig, bool deleted) override;
	virtual void print_get_method(FunctionDecl *fd) override;
};

/* Print a declaration of the method "method" with signature "sig".
 * Mark is "delete" if "deleted" is set.
 */
void template_cpp_generator::method_decl_printer::print_method_sig(
	const Method &method, const Signature &sig, bool deleted)
{
	print_method_header(method, sig);
	if (deleted)
		os << " = delete";
	os << ";\n";
}

/* Return the total number of arguments in the signature for "method",
 * taking into account any possible callback arguments.
 *
 * In particular, if the method has a callback argument,
 * then the return kind of the callback appears at the position
 * of the callback and the kinds of the arguments (except
 * the user pointer argument) appear in the following positions.
 * The user pointer argument that follows the callback argument
 * is also removed.
 */
static int total_params(const Method &method)
{
	int n = method.num_params();

	for (const auto &callback : method.callbacks) {
		auto callback_type = callback->getType();
		auto proto = generator::extract_prototype(callback_type);

		n += proto->getNumArgs() - 1;
		n -= 1;
	}

	return n;
}

/* Return a signature for "method" that matches "instance".
 */
static Signature instance_sig(const Method &method,
	const specialization &instance)
{
	std::vector<Kind> args(total_params(method));

	args[0] = instance.kind;
	return { instance.kind, args };
}

/* Print a declaration for the "get" method "fd",
 * using a name that includes the "get_" prefix.
 *
 * These methods are only included in the plain interface.
 * Explicitly delete them from the templated interface.
 */
void template_cpp_generator::method_decl_printer::print_get_method(
	FunctionDecl *fd)
{
	Method method(clazz, fd, clazz.base_method_name(fd));

	print_method_sig(method, instance_sig(method, instance), true);
}

/* A helper class for printing method definitions
 * of a template class specialization.
 */
struct template_cpp_generator::method_impl_printer :
		public template_cpp_generator::class_printer {
	method_impl_printer(const specialization &instance,
			const struct specialization_printer &instance_printer) :
		class_printer(instance, instance_printer, false) {}

	void print_callback_method_body(const Method &method,
		const Signature &sig);
	void print_method_body(const Method &method, const Signature &sig);
	void print_constructor_body(const Method &method, const Signature &sig);
	virtual void print_method_sig(const Method &method,
		const Signature &sig, bool deleted) override;
	virtual void print_get_method(FunctionDecl *fd) override;
};

/* Print a definition of the constructor "method" with signature "sig".
 *
 * Simply pass all arguments to the constructor of the corresponding
 * plain type.
 */
void template_cpp_generator::method_impl_printer::print_constructor_body(
	const Method &method, const Signature &sig)
{
	const auto &base_name = instance.base_name();

	os << "  : " << base_name;
	method.print_cpp_arg_list(os, [&] (int i, int arg) {
		os << method.fd->getParamDecl(i)->getName().str();
	});
	os << "\n";

	os << "{\n";
	os << "}\n";
}

/* Print the arguments of the callback function "callback" to "os",
 * calling "print_arg" with the type and the name of the arguments,
 * where the type is obtained from "type_printer" with argument positions
 * shifted by "shift".
 * None of the arguments should be skipped.
 */
static void print_callback_args(std::ostream &os,
	const FunctionProtoType *callback, const cpp_type_printer &type_printer,
	int shift,
	const std::function<void(const std::string &type,
		const std::string &name)> &print_arg)
{
	auto n_arg = callback->getNumArgs() - 1;

	Method::print_arg_list(os, 0, n_arg, [&] (int i) {
		auto type = callback->getArgType(i);
		auto name = "arg" + std::to_string(i);
		auto cpptype = type_printer.param(shift + i, type);

		print_arg(cpptype, name);

		return false;
	});
}

/* Print a lambda corresponding to "callback"
 * with signature "sig" and argument positions shifted by "shift".
 *
 * The lambda takes arguments with plain isl types and
 * calls the callback of "method" with templated arguments.
 */
static void print_callback_lambda(std::ostream &os, ParmVarDecl *callback,
	const Signature &sig, int shift)
{
	auto callback_type = callback->getType();
	auto callback_name = callback->getName().str();
	auto proto = generator::extract_prototype(callback_type);

	os << "  auto lambda_" << callback_name << " = [&] ";
	print_callback_args(os, proto, cpp_type_printer(), shift,
		[&] (const std::string &type, const std::string &name) {
			os << type << " " << name;
		});
	os << " {\n";

	os << "    return " << callback_name;
	print_callback_args(os, proto, template_cpp_arg_type_printer(sig),
		shift,
		[&] (const std::string &type, const std::string &name) {
			os << type << "(" << name << ")";
		});
	os << ";\n";

	os << "  };\n";
}

/* Print lambdas for passing to the plain method corresponding to "method"
 * with signature "sig".
 *
 * The method is assumed to have only callbacks as argument,
 * which means the arguments of the first callback are shifted by 2
 * with respect to the arguments of the signature
 * (one for the position of the callback argument plus
 * one for the return kind of the callback).
 * The arguments of a subsequent callback are shifted by
 * the number of arguments of the previous callback minus one
 * for the user pointer plus one for the return kind.
 */
static void print_callback_lambdas(std::ostream &os, const Method &method,
	const Signature &sig)
{
	int shift;

	if (method.num_params() != 1 + 2 * method.callbacks.size())
		generator::die("callbacks are assumed to be only arguments");

	shift = 2;
	for (const auto &callback : method.callbacks) {
		print_callback_lambda(os, callback, sig, shift);
		shift += generator::prototype_n_args(callback->getType());
	}
}

/* Print a definition of the member method "method", which is known
 * to have a callback argument, with signature "sig".
 *
 * First print lambdas for passing to the corresponding plain method and
 * calling the callback of "method" with templated arguments.
 * Then call the plain method, replacing the original callbacks
 * by the lambdas.
 *
 * The return value is assumed to be isl_bool or isl_stat
 * so that no conversion to a template type is required.
 */
void template_cpp_generator::method_impl_printer::print_callback_method_body(
	const Method &method, const Signature &sig)
{
	const auto &base_name = instance.base_name();
	auto return_type = method.fd->getReturnType();

	if (!is_isl_bool(return_type) && !is_isl_stat(return_type))
		die("only isl_bool and isl_stat return types are supported");

	os << "{\n";

	print_callback_lambdas(os, method, sig);

	os << "  return ";
	os << base_name << "::" << method.name;
	method.print_cpp_arg_list(os, [&] (int i, int arg) {
		auto param = method.fd->getParamDecl(i);

		if (generator::is_callback(param->getType()))
			os << "lambda_";
		os << param->getName().str();
	});
	os << ";\n";

	os << "}\n";
}

/* Print a definition of the member or static method "method"
 * with signature "sig".
 *
 * The body calls the corresponding method of the base class
 * in the plain interface and
 * then casts the result to the templated result type.
 */
void template_cpp_generator::method_impl_printer::print_method_body(
	const Method &method, const Signature &sig)
{
	const auto &base_name = instance.base_name();

	os << "{\n";
	os << "  auto res = ";
	os << base_name << "::" << method.name;
	method.print_cpp_arg_list(os, [&] (int i, int arg) {
		os << method.fd->getParamDecl(i)->getName().str();
	});
	os << ";\n";

	os << "  return ";
	print_return_type(method, sig.ret);
	os << "(res);\n";
	os << "}\n";
}

/* Print a definition of the method "method" with signature "sig",
 * if "deleted" is not set.
 *
 * If "deleted" is set, then the corresponding declaration
 * is marked "delete" and no definition needs to be printed.
 *
 * Otherwise print the method header, preceded by the template parameters,
 * if needed.
 * The body depends on whether the method is a constructor or
 * takes any callbacks.
 */
void template_cpp_generator::method_impl_printer::print_method_sig(
	const Method &method, const Signature &sig, bool deleted)
{
	if (deleted)
		return;

	os << "\n";
	print_non_empty_template(os, instance.kind.params());
	print_method_header(method, sig);
	os << "\n";
	if (method.kind == Method::Kind::constructor)
		print_constructor_body(method, sig);
	else if (method.callbacks.size() != 0)
		print_callback_method_body(method, sig);
	else
		print_method_body(method, sig);
}

/* Print a definition for the "get" method "fd" in class "clazz",
 * using a name that includes the "get_" prefix, to "os".
 *
 * The declarations of these methods are explicitly delete'd
 * so no definition needs to be printed.
 */
void template_cpp_generator::method_impl_printer::print_get_method(
	FunctionDecl *fd)
{
}

/* Print a declaration or definition of the static method "method",
 * if it has a signature specified by static_methods.
 */
void template_cpp_generator::class_printer::print_static_method(
	const Method &method)
{
	print_special_method(method, static_methods);
}

/* Signatures for constructors of multi-expressions
 * from a space and a list.
 */
static Signature from_list_set = { { Domain }, { { Domain }, { Anonymous } } };
static Signature from_list_map =
	{ { Domain, Range }, { { Domain, Range }, { Domain, Anonymous } } };

/* Signatures for constructors from a string.
 */
static Signature params_from_str = { { }, { { Ctx }, { Str } } };
static Signature set_from_str = { { Domain }, { { Ctx }, { Str } } };
static Signature map_from_str = { { Domain, Range }, { { Ctx }, { Str } } };
static std::vector<Signature> from_str =
	{ params_from_str, set_from_str, map_from_str };

/* Signature for a constructor from an integer.
 */
static Signature int_from_si = { { Anonymous }, { { Ctx }, { Integer } } };

/* Signatures for constructors of lists from the initial number
 * of elements.
 */
static Signature alloc_params = { { }, { { Ctx }, { Integer } } };
static Signature alloc_set = { { Domain }, { { Ctx }, { Integer } } };
static Signature alloc_map = { { Domain, Range }, { { Ctx }, { Integer } } };

/* Signatures for constructors and methods named after some other class.
 *
 * Two forms of constructors are handled
 * - conversion from another object
 * - construction of a multi-expression from a space and a list
 *
 * Methods named after some other class also come in two forms
 * - extraction of information such as the space or a list
 * - construction of a multi-expression from a space and a list
 *
 * In both cases, the first form is a unary operation and
 * the second has an extra argument with a kind that is equal
 * to that of the first argument, except that the final tuple is anonymous.
 */
static std::vector<Signature> constructor_sig = {
	un_params,
	un_set,
	un_map,
	from_list_set,
	from_list_map,
};

/* Signatures for constructors derived from methods
 * with the given names that override the default signatures.
 */
static const std::unordered_map<std::string, std::vector<Signature>>
special_constructors {
	{ "alloc",		{ alloc_params, alloc_set, alloc_map } },
	{ "int_from_si",	{ int_from_si } },
	{ "read_from_str",	from_str },
};

/* Print a declaration or definition of the constructor "method".
 */
void template_cpp_generator::class_printer::print_constructor(
	const Method &method)
{
	if (special_constructors.count(method.name) != 0) {
		const auto &sigs = special_constructors.at(method.name);
		return print_matching_method(method, sigs);
	}
	print_matching_method(method, constructor_sig);
}

/* Does this template class represent an anonymous function?
 *
 * If any specialization represents an anonymous function,
 * then every specialization does, so simply check
 * the first specialization.
 */
bool template_class::is_anon() const
{
	return class_tuples[0].is_anon();
}

/* Does this template class represent an anonymous value?
 *
 * That is, is there only a single specialization that moreover
 * has a single, anonymous tuple?
 */
bool template_class::is_anon_set() const
{
	return class_tuples.size() == 1 && class_tuples[0].is_anon_set();
}

/* Update the substitution "sub" to map "general" to "specific"
 * if "specific" is a special case of "general" consistent with "sub",
 * given that "general" is not a pair and can be assigned "specific".
 * Return true if successful.
 * Otherwise, return false.
 *
 * Check whether "general" is already assigned something in "sub".
 * If so, it must be assigned "specific".
 * Otherwise, there is a conflict.
 */
static bool update_sub_base(Substitution &sub, const TupleKindPtr &general,
	const TupleKindPtr &specific)
{
	auto name = general->name;

	if (sub.count(name) != 0 && sub.at(name) != specific)
		return false;
	sub.emplace(name, specific);
	return true;
}

/* Update the substitution "sub" to map "general" to "specific"
 * if "specific" is a special case of "general" consistent with "sub".
 * Return true if successful.
 * Otherwise, return false.
 *
 * If "general" is a pair and "specific" is not,
 * then "specific" cannot be a special case.
 * If both are pairs, then update the substitution based
 * on both sides.
 * If "general" is Anonymous, then "specific" must be Anonymous as well.
 * If "general" is Leaf, then "specific" cannot be a pair.
 *
 * Otherwise, assign "specific" to "general", if possible.
 */
static bool update_sub(Substitution &sub, const TupleKindPtr &general,
	const TupleKindPtr &specific)
{
	if (general->left() && !specific->left())
		return false;
	if (general->left())
		return update_sub(sub, general->left(), specific->left()) &&
		    update_sub(sub, general->right(), specific->right());
	if (general == Anonymous && specific != Anonymous)
		return false;
	if (general == Leaf && specific->left())
		return false;

	return update_sub_base(sub, general, specific);
}

/* Check if "specific" is a special case of "general" and,
 * if so, return true along with a substitution
 * that maps "general" to "specific".
 * Otherwise return false.
 *
 * This can only happen if the number of tuple kinds is the same.
 * If so, start with an empty substitution and update it
 * for each pair of tuple kinds, checking that each update succeeds.
 */
static std::pair<bool, Substitution> specializer(const Kind &general,
	const Kind &specific)
{
	Substitution specializer;

	if (general.size() != specific.size())
		return { false, Substitution() };

	for (size_t i = 0; i < general.size(); ++i) {
		auto general_tuple = general[i];

		if (!update_sub(specializer, general[i], specific[i]))
			return { false, Substitution() };
	}

	return { true, specializer };
}

/* Is "kind1" equivalent to "kind2"?
 * That is, is each a special case of the other?
 */
static bool equivalent(const Kind &kind1, const Kind &kind2)
{
	return specializer(kind1, kind2).first &&
	       specializer(kind2, kind1).first;
}

/* Add the specialization "kind" to the sequence of specializations,
 * provided there is no equivalent specialization already in there.
 */
void template_class::add_specialization(const Kind &kind)
{
	for (const auto &special : class_tuples)
		if (equivalent(special, kind))
			return;
	class_tuples.emplace_back(kind);
}

/* A type printer that prints the plain interface type,
 * without namespace.
 */
struct plain_cpp_type_printer : public cpp_type_printer {
	plain_cpp_type_printer() {}

	virtual std::string qualified(int arg, const std::string &cpp_type)
		const override;
};

/* Return the qualified form of the given C++ isl type name appearing
 * in argument position "arg" (-1 for return type).
 *
 * For printing the plain type without namespace, no modifications
 * are required.
 */
std::string plain_cpp_type_printer::qualified(int arg,
	const std::string &cpp_type) const
{
	return cpp_type;
}

/* Return a string representation of the plain type "type".
 *
 * For the plain printer, the argument position is irrelevant,
 * so simply pass in -1.
 */
static std::string plain_type(QualType type)
{
	return plain_cpp_type_printer().param(-1, type);
}

/* Return a string representation of the plain return type of "method".
 */
static std::string plain_return_type(const Method &method)
{
	return plain_type(method.fd->getReturnType());
}

/* Return that part of the signature "sig" that should match
 * the template class specialization for the given method.
 *
 * In particular, if the method is a regular member method,
 * then the instance should match the first argument.
 * Otherwise, it should match the return kind.
 */
static const Kind &matching_kind(const Method &method, const Signature &sig)
{
	if (method.kind == Method::Kind::member_method)
		return sig.args[0];
	else
		return sig.ret;
}

/* Is it possible for "template_class" to have the given kind?
 *
 * If the template class represents an anonymous function,
 * then so must the given kind.
 * There should also be specialization with the same number of tuple kinds.
 */
static bool has_kind(const template_class &template_class, const Kind &kind)
{
	if (template_class.is_anon() && !kind.is_anon())
		return false;
	for (const auto &class_tuple : template_class.class_tuples)
		if (class_tuple.size() == kind.size())
			return true;
	return false;
}

/* Is "return_kind" a possible kind for the return type of "method"?
 *
 * If the return type is not a template class,
 * then "return_kind" should not have any template parameters.
 * Otherwise, "return_kind" should be a valid kind for the template class.
 */
bool template_cpp_generator::class_printer::is_return_kind(
	const Method &method, const Kind &return_kind)
{
	const auto &template_classes = generator.template_classes;
	auto return_type = plain_return_type(method);

	if (template_classes.count(return_type) == 0)
		return return_kind.params().size() == 0;
	return has_kind(template_classes.at(return_type), return_kind);
}

/* Is "kind" a placeholder that can be assigned something else
 * in a substitution?
 *
 * Anonymous can only be mapped to itself.  This is taken care of
 * by assign().
 * Leaf can only be assigned a placeholder, but there is no need
 * to handle this specifically since Leaf can still be assigned
 * to the placeholder.
 */
static bool assignable(const TupleKindPtr &kind)
{
	return kind != Anonymous && kind != Leaf;
}

/* Return a substitution that maps "kind1" to "kind2", if possible.
 * Otherwise return an empty substitution.
 *
 * Check if "kind1" can be assigned anything or
 * if "kind1" and "kind2" are identical.
 * The latter case handles mapping Anonymous to itself.
 */
static Substitution assign(const TupleKindPtr &kind1, const TupleKindPtr &kind2)
{
	Substitution res;

	if (assignable(kind1) || kind1 == kind2)
		res.emplace(kind1->name, kind2);
	return res;
}

/* Return a substitution that first applies "first" and then "second".
 *
 * The result consists of "second" and of "second" applied to "first".
 */
static Substitution compose(const Substitution &first,
	const Substitution &second)
{
	Substitution res = second;

	for (const auto &kvp : first)
		res.emplace(kvp.first, apply(kvp.second, second));

	return res;
}

static Substitution compute_unifier(const TupleKindPtr &kind1,
	const TupleKindPtr &kind2);

/* Try and extend "unifier" with a unifier for "kind1" and "kind2".
 * Return the resulting unifier if successful.
 * Otherwise, return an empty substitution.
 *
 * First apply "unifier" to "kind1" and "kind2".
 * Then compute a unifier for the resulting tuple kinds and
 * combine it with "unifier".
 */
static Substitution combine_unifiers(const TupleKindPtr &kind1,
	const TupleKindPtr &kind2, const Substitution &unifier)
{
	auto k1 = apply(kind1, unifier);
	auto k2 = apply(kind2, unifier);
	auto u = compute_unifier(k1, k2);
	if (u.size() == 0)
		return Substitution();
	return compose(unifier, u);
}

/* Try and compute a unifier of "kind1" and "kind2",
 * i.e., a substitution that produces the same result when
 * applied to both "kind1" and "kind2",
 * for the case where both "kind1" and "kind2" are pairs.
 * Return this unifier if it was found.
 * Return an empty substitution if no unifier can be found.
 *
 * First compute a unifier for the left parts of the pairs and,
 * if successful, combine it with a unifier for the right parts.
 */
static Substitution compute_pair_unifier(const TupleKindPtr &kind1,
	const TupleKindPtr &kind2)
{
	auto unifier_left = compute_unifier(kind1->left(), kind2->left());
	if (unifier_left.size() == 0)
		return Substitution();
	return combine_unifiers(kind1->right(), kind2->right(), unifier_left);
}

/* Try and compute a unifier of "kind1" and "kind2",
 * i.e., a substitution that produces the same result when
 * applied to both "kind1" and "kind2".
 * Return this unifier if it was found.
 * Return an empty substitution if no unifier can be found.
 *
 * If one of the tuple kinds is a pair then assign it
 * to the other tuple kind, if possible.
 * If neither is a pair, then try and assign one to the other.
 * Otherwise, let compute_pair_unifier compute a unifier.
 *
 * Note that an assignment is added to the unifier even
 * if "kind1" and "kind2" are identical.
 * This ensures that a successful substitution is never empty.
 */
static Substitution compute_unifier(const TupleKindPtr &kind1,
	const TupleKindPtr &kind2)
{
	if (kind1->left() && !kind2->left())
		return assign(kind2, kind1);
	if (!kind1->left() && kind2->left())
		return assign(kind1, kind2);
	if (!kind1->left() && !kind2->left()) {
		if (assignable(kind1))
			return assign(kind1, kind2);
		else
			return assign(kind2, kind1);
	}

	return compute_pair_unifier(kind1, kind2);
}

/* Try and compute a unifier of "kind1" and "kind2",
 * i.e., a substitution that produces the same result when
 * applied to both "kind1" and "kind2".
 * Return this unifier if it was found.
 * Return an empty substitution if no unifier can be found.
 *
 * Start with an empty substitution and compute a unifier for
 * each pair of tuple kinds, combining the results.
 * If no combined unifier can be found or
 * if the numbers of tuple kinds are different, then return
 * an empty substitution.
 * This assumes that the number of tuples is greater than zero,
 * as otherwise an empty substitution would be returned as well.
 */
static Substitution compute_unifier(const Kind &kind1, const Kind &kind2)
{
	Substitution unifier;

	if (kind1.size() != kind2.size())
		return Substitution();

	for (size_t i = 0; i < kind1.size(); ++i)
		unifier = combine_unifiers(kind1[i], kind2[i], unifier);

	return unifier;
}

/* Try and construct a Kind that is a specialization of both "general" and
 * "specific", where "specific" is known _not_ to be a specialization
 * of "general" and not to contain any Leaf.
 *
 * First check whether "general" is a specialization of "specific".
 * If so, simply return "general".
 * Otherwise, rename the placeholders in the two kinds apart and
 * try and compute a unifier.
 * If this succeeds, then return the result of applying the unifier.
 */
static std::pair<bool, Kind> unify(const Kind &general, const Kind &specific)
{
	if (specializer(specific, general).first) {
		return { true, general };
	} else {
		auto rename = param_renamer(specific.params(), "T");
		auto renamed = specific.apply(rename);
		auto unifier = compute_unifier(general, renamed);

		if (unifier.size() == 0)
			return { false, { } };

		return { true, general.apply(unifier) };
	}
}

/* Try and add a template class specialization corresponding to "kind".
 * The new specialization needs to be a specialization of both
 * the current specialization and "kind".
 *
 * The current template class specialization is known not to be a special case
 * of "kind".
 *
 * Try and unify the two kinds and, if this succeeds, add the result
 * to this list of template class specializations.
 */
void template_cpp_generator::class_printer::add_specialization(
	const Kind &kind)
{
	auto maybe_unified = unify(kind, instance.kind);

	if (!maybe_unified.first)
		return;
	instance.template_class.add_specialization(maybe_unified.second);
}

/* Does the type of the parameter at position "i" of "method" necessarily
 * have a final Anonymous tuple?
 *
 * If the parameter is not of an isl type or if no specializations
 * have been defined for the type, then it can be considered anonymous.
 * Otherwise, if any specialization represents an anonymous function,
 * then every specialization does, so simply check
 * the first specialization.
 */
static bool param_is_anon(const Method &method, int i)
{
	ParmVarDecl *param = method.get_param(i);
	QualType type = param->getOriginalType();

	if (cpp_generator::is_isl_type(type)) {
		const auto &name = type->getPointeeType().getAsString();
		const auto &cpp = cpp_generator::type2cpp(name);
		const auto &tuples = lookup_class_tuples(cpp);

		if (tuples.empty())
			return true;
		return tuples[0].is_anon();
	}

	return true;
}

/* Replace the final tuple of "arg_kind" by Anonymous in "sig" and
 * return the update signature,
 * unless this would affect the class instance "instance_kind".
 *
 * If the original "instance_kind" is a special case
 * of the result of the substitution, then "instance_kind"
 * is not affected and the substitution can be applied
 * to the entire signature.
 */
static Signature specialize_anonymous_arg(const Signature &sig,
	const Kind &arg_kind, const Kind &instance_kind)
{
	const auto &subs = compute_unifier(arg_kind.back(), Anonymous);
	const auto &specialized_instance = instance_kind.apply(subs);

	if (!specializer(specialized_instance, instance_kind).first)
		return sig;

	return sig.apply(subs);
}

/* If any of the arguments of "method" is of a type that necessarily
 * has a final Anonymous tuple, but the corresponding entry
 * in the signature "sig" is not Anonymous, then replace
 * that entry by Anonymous and return the updated signature,
 * unless this would affect the class instance "instance_kind".
 */
static Signature specialize_anonymous_args(const Signature &sig,
	const Method &method, const Kind &instance_kind)
{
	auto specialized_sig = sig;

	method.on_cpp_arg_list([&] (int i, int arg) {
		const auto &arg_kind = sig.args[arg];

		if (arg_kind.is_anon())
			return;
		if (!param_is_anon(method, i))
			return;
		specialized_sig = specialize_anonymous_arg(specialized_sig,
					arg_kind, instance_kind);
	});

	return specialized_sig;
}

/* Print a declaration or definition of the method "method"
 * if the template class specialization matches "match_arg".
 * Return true if so.
 * "sig" is the complete signature, of which "match_arg" refers
 * to the first argument or the return type.
 *
 * Since "sig" may have parameters with the same names as
 * those in instance.kind, rename them apart first.
 *
 * If the template class specialization is a special case of
 * (the renamed) "match_arg"
 * then apply the specializer to the complete (renamed) signature,
 * specialize any anonymous arguments,
 * check that the return kind is allowed and, if so,
 * print the declaration or definition using the specialized signature.
 *
 * If the template class specialization is not a special case of "match_arg"
 * then add a further specialization to the list of specializations
 * of the template class.
 */
bool template_cpp_generator::class_printer::print_matching_method(
	const Method &method, const Signature &sig, const Kind &match_arg)
{
	auto rename = shared_param_renamer(sig, instance.kind);
	auto renamed_arg = match_arg.apply(rename);
	auto maybe_specializer = specializer(renamed_arg, instance.kind);
	if (maybe_specializer.first) {
		const auto &specializer = maybe_specializer.second;
		auto specialized_sig = sig.apply(rename).apply(specializer);
		specialized_sig = specialize_anonymous_args(specialized_sig,
							method, instance.kind);
		if (!is_return_kind(method, specialized_sig.ret))
			return false;

		print_method_sig(method, specialized_sig, false);
	} else {
		add_specialization(match_arg);
	}
	return maybe_specializer.first;
}

/* Is the first argument of "method" of type "isl_ctx *"?
 */
static bool first_arg_is_ctx(const Method &method)
{
	return generator::first_arg_is_isl_ctx(method.fd);
}

/* Is the first signature argument set to { Ctx }?
 */
static bool first_kind_is_ctx(const Signature &sig)
{
	return sig.args[0].size() > 0 && sig.args[0][0] == Ctx;
}

/* Print a declaration or definition of the member method "method"
 * if it matches the signature "sig".
 * Return true if so.
 *
 * First determine the part of the signature that needs to match
 * the template class specialization and
 * check that it has the same number of template arguments.
 * Also check that the number of arguments of the signature
 * matches that of the method.
 * If there is at least one argument, then check that the first method argument
 * is an isl_ctx if and only if the first signature argument is Ctx.
 *
 * If these tests succeed, proceed with the actual matching.
 */
bool template_cpp_generator::class_printer::print_matching_method(
	const Method &method, const Signature &sig)
{
	auto match_arg = matching_kind(method, sig);
	int n_args = sig.args.size();

	if (match_arg.size() != instance.kind.size())
		return false;
	if (n_args != total_params(method))
		return false;
	if (n_args > 0 && first_arg_is_ctx(method) != first_kind_is_ctx(sig))
		return false;

	return print_matching_method(method, sig, match_arg);
}

/* Print a declaration or definition of the member method "method"
 * for each matching signature in "signatures".
 *
 * If there is no matching signature in "signatures",
 * then explicitly delete the method (using a signature based on
 * the specialization) so that it is not inherited from the base class.
 */
void template_cpp_generator::class_printer::print_matching_method(
	const Method &method, const std::vector<Signature> &signatures)
{
	auto any = false;

	for (const auto &sig : signatures)
		if (print_matching_method(method, sig))
			any = true;

	if (!any)
		print_method_sig(method, instance_sig(method, instance), true);
}

/* Signatures for "at" methods applied to a multi-expression,
 * which make the final tuple anonymous.
 */
static Signature select_set = { { Anonymous }, { { Domain }, { Integer } } };
static Signature select_map =
	{ { Domain, Anonymous }, { { Domain, Range }, { Integer } } };
static std::vector<Signature> at_select = { select_set, select_map };

/* Signatures for other "at" methods applied to a list,
 * which do not modify the tuple kind.
 */
static Signature bin_set_int = { { Domain }, { { Domain }, { Integer } } };
static Signature bin_map_int =
	{ { Domain, Range }, { { Domain, Range }, { Integer } } };
static std::vector<Signature> at_keep = { bin_set_int, bin_map_int };

/* Print a declaration or definition of the "at" member method "method".
 *
 * There are two types of methods called "at".
 * One type extracts an element from a multi-expression and
 * the other extracts an element from a list.
 *
 * In the first case, the return type is an anonymous function
 * while the object type is not.  In this case, the return kind
 * should have a final Anonymous tuple.
 * Otherwise, the return kind should be the same as the object kind.
 */
void template_cpp_generator::class_printer::print_at_method(
	const Method &method)
{
	auto anon = instance.template_class.is_anon();
	auto return_type = plain_return_type(method);
	auto return_class = generator.template_classes.at(return_type);

	if (!anon && return_class.is_anon())
		return print_matching_method(method, at_select);
	else
		return print_matching_method(method, at_keep);
}

/* Does the string "s" contain "sub" as a substring?
 */
static bool contains(const std::string &s, const std::string &sub)
{
	return s.find(sub) != std::string::npos;
}

/* Print a declaration or definition of the member method "method",
 * if it has a special signature in "special_methods".
 * Return true if this is the case.
 *
 * Check if any special signatures are specified for this method and
 * if the class name matches any of those with special signatures.
 * If so, pick the one with the best match, i.e., the first match
 * since the largest keys appear first.
 */
bool template_cpp_generator::class_printer::print_special_method(
	const Method &method, const infix_map_map &special_methods)
{
	if (special_methods.count(method.name) == 0)
		return false;

	for (const auto &kvp : special_methods.at(method.name)) {
		if (!contains(instance.template_class.class_name, kvp.first))
			continue;
		print_matching_method(method, kvp.second);
		return true;
	}

	return false;
}

/* Print a declaration or definition of the member method "method",
 * if it has a special signature specified by special_member_methods.
 * Return true if this is the case.
 */
bool template_cpp_generator::class_printer::print_special_member_method(
	const Method &method)
{
	return print_special_method(method, special_member_methods);
}

/* Print a declaration or definition of the member method "method",
 * if it is named after a template class.  Return true if this is the case.
 */
bool template_cpp_generator::class_printer::print_type_named_member_method(
	const Method &method)
{
	if (generator.template_classes.count(method.name) == 0)
		return false;

	print_matching_method(method, constructor_sig);

	return true;
}

/* Print a declaration or definition of the member method "method"
 * using a signature associated to method name "name", if there is any.
 * Return true if this is the case.
 */
bool template_cpp_generator::class_printer::print_member_method_with_name(
	const Method &method, const std::string &name)
{
	if (member_methods.count(name) == 0)
		return false;

	print_matching_method(method, member_methods.at(name));
	return true;
}

/* If "sub" appears inside "str", then remove the first occurrence and
 * return the result.  Otherwise, simply return "str".
 */
static std::string drop_occurrence(const std::string &str,
	const std::string &sub)
{
	auto res = str;
	auto pos = str.find(sub);

	if (pos != std::string::npos)
		res.erase(pos, sub.length());

	return res;
}

/* If "sub" appears in "str" next to an underscore, then remove the combination.
 * Otherwise, simply return "str".
 */
static std::string drop_underscore_occurrence(const std::string &str,
	const std::string &sub)
{
	auto res = drop_occurrence(str, sub + "_");
	if (res != str)
		return res;
	return drop_occurrence(res, std::string("_") + sub);
}

/* Return the name of "method", with the name of the return type,
 * along with an underscore, removed, if this combination appears in the name.
 * Otherwise, simply return the name.
 */
const std::string name_without_return(const Method &method)
{
	auto return_infix = plain_return_type(method);
	return drop_underscore_occurrence(method.name, return_infix);
}

/* If this method has a callback, then remove the type
 * of the first argument of the first callback from the name of the method.
 * Otherwise, simply return the name of the method.
 */
const std::string callback_name(const Method &method)
{
	if (method.callbacks.size() == 0)
		return method.name;

	auto type = method.callbacks.at(0)->getType();
	auto callback = cpp_generator::extract_prototype(type);
	auto arg_type = plain_type(callback->getArgType(0));
	return generator::drop_suffix(method.name, "_" + arg_type);
}

/* Print a declaration or definition of the member method "method".
 *
 * If the method is called "at", then it requires special treatment.
 * Otherwise, check if the signature is overridden for this class or
 * if the method is named after some other type.
 * Otherwise look for an appropriate signature using different variations
 * of the method name.  First try the method name itself,
 * then the method name with the return type removed and
 * finally the method name with the callback argument type removed.
 */
void template_cpp_generator::class_printer::print_member_method(
	const Method &method)
{
	if (method.name == "at")
		return print_at_method(method);
	if (print_special_member_method(method))
		return;
	if (print_type_named_member_method(method))
		return;
	if (print_member_method_with_name(method, method.name))
		return;
	if (print_member_method_with_name(method, name_without_return(method)))
		return;
	if (print_member_method_with_name(method, callback_name(method)))
		return;
}

/* Print a declaration or definition of "method" based on its type.
 */
void template_cpp_generator::class_printer::print_any_method(
	const Method &method)
{
	switch (method.kind) {
	case Method::Kind::static_method:
		print_static_method(method);
		break;
	case Method::Kind::constructor:
		print_constructor(method);
		break;
	case Method::Kind::member_method:
		print_member_method(method);
		break;
	}
}

/* Print a declaration or definition of "method".
 *
 * Mark the method as not requiring copies of the arguments.
 */
void template_cpp_generator::class_printer::print_method(const Method &method)
{
	print_any_method(NoCopyMethod(method));
}

/* Print a declaration or definition of "method".
 *
 * Note that a ConversionMethod is already marked
 * as not requiring copies of the arguments.
 */
void template_cpp_generator::class_printer::print_method(
	const ConversionMethod &method)
{
	print_any_method(method);
}

/* Helper class for printing the declarations for
 * template class specializations.
 */
struct template_cpp_generator::class_decl_printer :
	public specialization_printer
{
	class_decl_printer(std::ostream &os,
				template_cpp_generator &generator) :
		specialization_printer(os, generator) {}

	void print_arg_subclass_constructor(const specialization &instance,
		const std::vector<std::string> &params) const;
	void print_super_constructor(const specialization &instance) const;
	virtual void print_class(const specialization &instance) const override;
};

/* Print the declaration and definition of a constructor
 * for the template class specialization "instance" taking
 * an instance with more specialized template arguments,
 * where "params" holds the template parameters of "instance".
 * It is assumed that there is at least one template parameter as otherwise
 * there are no template arguments to be specialized and
 * no constructor needs to be printed.
 *
 * In particular, the constructor takes an object of the same instance where
 * for each template parameter, the corresponding template argument
 * of the input object is a subclass of the template argument
 * of the constructed object.
 *
 * Pick fresh names for all template parameters and
 * add a constructor with these fresh names as extra template parameters and
 * a constraint requiring that each of them is a subclass
 * of the corresponding class template parameter.
 * The plain C++ interface object of the constructed object is initialized with
 * the plain C++ interface object of the constructor argument.
 */
void template_cpp_generator::class_decl_printer::print_arg_subclass_constructor(
	const specialization &instance,
	const std::vector<std::string> &params) const
{
	const auto &class_name = instance.class_name();
	auto rename = param_renamer(params, "Arg");
	auto derived = instance.kind.apply(rename);

	os << "  template ";
	os << "<";
	print_pure_template_args(os, derived.params());
	os << ",\n";
	os << "            typename std::enable_if<\n";
	for (size_t i = 0; i < params.size(); ++i) {
		if (i != 0)
			os << " &&\n";
		os << "              std::is_base_of<"
		   << params[i] << ", "
		   << rename.at(params[i])->params()[0] << ">{}";
	}
	os << ",\n";
	os << "            bool>::type = true>";
	os << "\n";
	os << "  " << class_name << "(const ";
	print_bare_template_type(os, class_name, derived);
	os << " &obj) : " << instance.base_name() << "(obj) {}\n";
}

/* Print the declaration and definition of a constructor
 * for the template class specialization "instance" taking
 * an instance of the base class.
 *
 * If the instance kind is that of an anonymous set
 * (i.e., it has a single tuple that is set to Anonymous),
 * then allow the constructor to be called externally.
 * This is mostly useful for being able to use isl::val and
 * isl::typed::val<Anonymous> interchangeably and similarly for isl::id.
 *
 * If the instance is of any other kind, then make this constructor private
 * to avoid objects of the plain interface being converted automatically.
 * Also make sure that it does not apply to any type derived
 * from the base class.  In particular, this makes sure it does
 * not apply to any other specializations of this template class as
 * otherwise any conflict in specializations would simply point
 * to the private constructor.
 *
 * A factory method is added to be able to perform the conversion explicitly,
 * with an explicit specification of the template arguments.
 */
void template_cpp_generator::class_decl_printer::print_super_constructor(
	const specialization &instance) const
{
	bool hide = !instance.kind.is_anon_set();
	const auto &base_name = instance.base_name();
	const auto &arg_name = hide ? "base" : base_name;

	if (hide) {
		os << " private:\n";
		os << "  template <typename base,\n";
		os << "            typename std::enable_if<\n";
		os << "              std::is_same<base, " << base_name
		   << ">{}, bool>::type = true>\n";
	}
	os << "  " << instance.class_name()
	   << "(const " << arg_name << " &obj) : "
	   << base_name << "(obj) {}\n";
	if (hide)
		os << " public:\n";
	os << "  static " << instance.class_name() << " from"
	   << "(const " << base_name << " &obj) {\n";
	os << "    return " << instance.class_name() << "(obj);\n";
	os << "  }\n";
}

/* Print a "declaration" for the given template class specialization.
 * In particular, print the class definition and the method declarations.
 *
 * The template parameters are the distinct variable names
 * in the instance kind.
 *
 * Each instance of the template class derives from the corresponding
 * plain C++ interface class.
 *
 * All (other) template classes are made friends of this template class
 * to allow them to call the private constructor taking an object
 * of the plain interface.
 *
 * Besides the constructors and methods that forward
 * to the corresponding methods in the plain C++ interface class,
 * some extra constructors are defined.
 * The default zero-argument constructor is useful for declaring
 * a variable that only gets assigned a value at a later stage.
 * The constructor taking an instance with more specialized
 * template arguments is useful for lifting the class hierarchy
 * of the template arguments to the template class.
 * The constructor taking an instance of the base class
 * is useful for (explicitly) constructing a template type
 * from a plain type.
 */
void template_cpp_generator::class_decl_printer::print_class(
	const specialization &instance) const
{
	const auto &class_name = instance.class_name();
	auto params = instance.kind.params();

	os << "\n";

	print_template(os, params);

	os << "struct ";
	print_bare_template_type(os, class_name, instance.kind);
	os << " : public " << instance.base_name() << " {\n";

	generator.print_friends(os);
	os << "\n";

	os << "  " << class_name << "() = default;\n";
	if (params.size() != 0)
		print_arg_subclass_constructor(instance, params);
	print_super_constructor(instance);
	method_decl_printer(instance, *this).print_all_methods();

	os << "};\n";
}

/* Helper class for printing the definitions of template class specializations.
 */
struct template_cpp_generator::class_impl_printer :
	public specialization_printer
{
	class_impl_printer(std::ostream &os,
				template_cpp_generator &generator) :
		specialization_printer(os, generator) {}

	virtual void print_class(const specialization &instance) const override;
};

/* Print a definition for the given template class specialization.
 *
 * In particular, print definitions
 * for the constructors and methods that forward
 * to the corresponding methods in the plain C++ interface class.
 * The extra constructors declared in the class definition
 * are defined inline.
 */
void template_cpp_generator::class_impl_printer::print_class(
	const specialization &instance) const
{
	method_impl_printer(instance, *this).print_all_methods();
}

/* Generate a templated cpp interface
 * based on the extracted types and functions.
 *
 * First print forward declarations for all template classes,
 * then the declarations of the classes, and at the end all
 * method implementations.
 */
void template_cpp_generator::generate()
{
	ostream &os = std::cout;

	os << "\n";

	print_forward_declarations(os);
	class_decl_printer(os, *this).print_classes();
	class_impl_printer(os, *this).print_classes();
}
