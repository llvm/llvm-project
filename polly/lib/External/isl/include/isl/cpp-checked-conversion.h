/// These are automatically generated conversions between
/// the default and the checked C++ bindings for isl.
///
/// isl is a library for computing with integer sets and maps described by
/// Presburger formulas. On top of this, isl provides various tools for
/// polyhedral compilation, ranging from dependence analysis over scheduling
/// to AST generation.

#ifndef ISL_CPP_CHECKED_CONVERSION
#define ISL_CPP_CHECKED_CONVERSION

#include <isl/cpp.h>
#include <isl/cpp-checked.h>

namespace isl {

checked::aff check(aff obj) {
	return checked::manage(obj.copy());
}

aff uncheck(checked::aff obj) {
	return manage(obj.copy());
}

checked::ast_build check(ast_build obj) {
	return checked::manage(obj.copy());
}

ast_build uncheck(checked::ast_build obj) {
	return manage(obj.copy());
}

checked::ast_expr check(ast_expr obj) {
	return checked::manage(obj.copy());
}

ast_expr uncheck(checked::ast_expr obj) {
	return manage(obj.copy());
}

checked::ast_node check(ast_node obj) {
	return checked::manage(obj.copy());
}

ast_node uncheck(checked::ast_node obj) {
	return manage(obj.copy());
}

checked::basic_map check(basic_map obj) {
	return checked::manage(obj.copy());
}

basic_map uncheck(checked::basic_map obj) {
	return manage(obj.copy());
}

checked::basic_set check(basic_set obj) {
	return checked::manage(obj.copy());
}

basic_set uncheck(checked::basic_set obj) {
	return manage(obj.copy());
}

checked::map check(map obj) {
	return checked::manage(obj.copy());
}

map uncheck(checked::map obj) {
	return manage(obj.copy());
}

checked::multi_aff check(multi_aff obj) {
	return checked::manage(obj.copy());
}

multi_aff uncheck(checked::multi_aff obj) {
	return manage(obj.copy());
}

checked::multi_pw_aff check(multi_pw_aff obj) {
	return checked::manage(obj.copy());
}

multi_pw_aff uncheck(checked::multi_pw_aff obj) {
	return manage(obj.copy());
}

checked::multi_union_pw_aff check(multi_union_pw_aff obj) {
	return checked::manage(obj.copy());
}

multi_union_pw_aff uncheck(checked::multi_union_pw_aff obj) {
	return manage(obj.copy());
}

checked::multi_val check(multi_val obj) {
	return checked::manage(obj.copy());
}

multi_val uncheck(checked::multi_val obj) {
	return manage(obj.copy());
}

checked::point check(point obj) {
	return checked::manage(obj.copy());
}

point uncheck(checked::point obj) {
	return manage(obj.copy());
}

checked::pw_aff check(pw_aff obj) {
	return checked::manage(obj.copy());
}

pw_aff uncheck(checked::pw_aff obj) {
	return manage(obj.copy());
}

checked::pw_multi_aff check(pw_multi_aff obj) {
	return checked::manage(obj.copy());
}

pw_multi_aff uncheck(checked::pw_multi_aff obj) {
	return manage(obj.copy());
}

checked::schedule check(schedule obj) {
	return checked::manage(obj.copy());
}

schedule uncheck(checked::schedule obj) {
	return manage(obj.copy());
}

checked::schedule_constraints check(schedule_constraints obj) {
	return checked::manage(obj.copy());
}

schedule_constraints uncheck(checked::schedule_constraints obj) {
	return manage(obj.copy());
}

checked::schedule_node check(schedule_node obj) {
	return checked::manage(obj.copy());
}

schedule_node uncheck(checked::schedule_node obj) {
	return manage(obj.copy());
}

checked::set check(set obj) {
	return checked::manage(obj.copy());
}

set uncheck(checked::set obj) {
	return manage(obj.copy());
}

checked::union_access_info check(union_access_info obj) {
	return checked::manage(obj.copy());
}

union_access_info uncheck(checked::union_access_info obj) {
	return manage(obj.copy());
}

checked::union_flow check(union_flow obj) {
	return checked::manage(obj.copy());
}

union_flow uncheck(checked::union_flow obj) {
	return manage(obj.copy());
}

checked::union_map check(union_map obj) {
	return checked::manage(obj.copy());
}

union_map uncheck(checked::union_map obj) {
	return manage(obj.copy());
}

checked::union_pw_aff check(union_pw_aff obj) {
	return checked::manage(obj.copy());
}

union_pw_aff uncheck(checked::union_pw_aff obj) {
	return manage(obj.copy());
}

checked::union_pw_multi_aff check(union_pw_multi_aff obj) {
	return checked::manage(obj.copy());
}

union_pw_multi_aff uncheck(checked::union_pw_multi_aff obj) {
	return manage(obj.copy());
}

checked::union_set check(union_set obj) {
	return checked::manage(obj.copy());
}

union_set uncheck(checked::union_set obj) {
	return manage(obj.copy());
}

checked::val check(val obj) {
	return checked::manage(obj.copy());
}

val uncheck(checked::val obj) {
	return manage(obj.copy());
}

} // namespace isl

#endif /* ISL_CPP_CHECKED_CONVERSION */
