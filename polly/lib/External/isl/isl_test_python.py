# Copyright 2016-2017 Tobias Grosser
#
# Use of this software is governed by the MIT license
#
# Written by Tobias Grosser, Weststrasse 47, CH-8003, Zurich

import sys
import isl

# Test that isl objects can be constructed.
#
# This tests:
#  - construction from a string
#  - construction from an integer
#  - static constructor without a parameter
#  - conversion construction
#  - construction of empty union set
#
#  The tests to construct from integers and strings cover functionality that
#  is also tested in the parameter type tests, but here the presence of
#  multiple overloaded constructors and overload resolution is tested.
#
def test_constructors():
	zero1 = isl.val("0")
	assert(zero1.is_zero())

	zero2 = isl.val(0)
	assert(zero2.is_zero())

	zero3 = isl.val.zero()
	assert(zero3.is_zero())

	bs = isl.basic_set("{ [1] }")
	result = isl.set("{ [1] }")
	s = isl.set(bs)
	assert(s.is_equal(result))

	us = isl.union_set("{ A[1]; B[2, 3] }")
	empty = isl.union_set.empty()
	assert(us.is_equal(us.union(empty)))

# Test integer function parameters for a particular integer value.
#
def test_int(i):
	val_int = isl.val(i)
	val_str = isl.val(str(i))
	assert(val_int.eq(val_str))

# Test integer function parameters.
#
# Verify that extreme values and zero work.
#
def test_parameters_int():
	test_int(sys.maxsize)
	test_int(-sys.maxsize - 1)
	test_int(0)

# Test isl objects parameters.
#
# Verify that isl objects can be passed as lvalue and rvalue parameters.
# Also verify that isl object parameters are automatically type converted if
# there is an inheritance relation. Finally, test function calls without
# any additional parameters, apart from the isl object on which
# the method is called.
#
def test_parameters_obj():
	a = isl.set("{ [0] }")
	b = isl.set("{ [1] }")
	c = isl.set("{ [2] }")
	expected = isl.set("{ [i] : 0 <= i <= 2 }")

	tmp = a.union(b)
	res_lvalue_param = tmp.union(c)
	assert(res_lvalue_param.is_equal(expected))

	res_rvalue_param = a.union(b).union(c)
	assert(res_rvalue_param.is_equal(expected))

	a2 = isl.basic_set("{ [0] }")
	assert(a.is_equal(a2))

	two = isl.val(2)
	half = isl.val("1/2")
	res_only_this_param = two.inv()
	assert(res_only_this_param.eq(half))

# Test different kinds of parameters to be passed to functions.
#
# This includes integer and isl object parameters.
#
def test_parameters():
	test_parameters_int()
	test_parameters_obj()

# Test that isl objects are returned correctly.
#
# This only tests that after combining two objects, the result is successfully
# returned.
#
def test_return_obj():
	one = isl.val("1")
	two = isl.val("2")
	three = isl.val("3")

	res = one.add(two)

	assert(res.eq(three))

# Test that integer values are returned correctly.
#
def test_return_int():
	one = isl.val("1")
	neg_one = isl.val("-1")
	zero = isl.val("0")

	assert(one.sgn() > 0)
	assert(neg_one.sgn() < 0)
	assert(zero.sgn() == 0)

# Test that isl_bool values are returned correctly.
#
# In particular, check the conversion to bool in case of true and false.
#
def test_return_bool():
	empty = isl.set("{ : false }")
	univ = isl.set("{ : }")

	b_true = empty.is_empty()
	b_false = univ.is_empty()

	assert(b_true)
	assert(not b_false)

# Test that strings are returned correctly.
# Do so by calling overloaded isl.ast_build.from_expr methods.
#
def test_return_string():
	context = isl.set("[n] -> { : }")
	build = isl.ast_build.from_context(context)
	pw_aff = isl.pw_aff("[n] -> { [n] }")
	set = isl.set("[n] -> { : n >= 0 }")

	expr = build.expr_from(pw_aff)
	expected_string = "n"
	assert(expected_string == expr.to_C_str())

	expr = build.expr_from(set)
	expected_string = "n >= 0"
	assert(expected_string == expr.to_C_str())

# Test that return values are handled correctly.
#
# Test that isl objects, integers, boolean values, and strings are
# returned correctly.
#
def test_return():
	test_return_obj()
	test_return_int()
	test_return_bool()
	test_return_string()

# Test that foreach functions are modeled correctly.
#
# Verify that closures are correctly called as callback of a 'foreach'
# function and that variables captured by the closure work correctly. Also
# check that the foreach function handles exceptions thrown from
# the closure and that it propagates the exception.
#
def test_foreach():
	s = isl.set("{ [0]; [1]; [2] }")

	list = []
	def add(bs):
		list.append(bs)
	s.foreach_basic_set(add)

	assert(len(list) == 3)
	assert(list[0].is_subset(s))
	assert(list[1].is_subset(s))
	assert(list[2].is_subset(s))
	assert(not list[0].is_equal(list[1]))
	assert(not list[0].is_equal(list[2]))
	assert(not list[1].is_equal(list[2]))

	def fail(bs):
		raise Exception("fail")

	caught = False
	try:
		s.foreach_basic_set(fail)
	except:
		caught = True
	assert(caught)

# Test the functionality of "every" functions.
#
# In particular, test the generic functionality and
# test that exceptions are properly propagated.
#
def test_every():
	us = isl.union_set("{ A[i]; B[j] }")

	def is_empty(s):
		return s.is_empty()
	assert(not us.every_set(is_empty))

	def is_non_empty(s):
		return not s.is_empty()
	assert(us.every_set(is_non_empty))

	def in_A(s):
		return s.is_subset(isl.set("{ A[x] }"))
	assert(not us.every_set(in_A))

	def not_in_A(s):
		return not s.is_subset(isl.set("{ A[x] }"))
	assert(not us.every_set(not_in_A))

	def fail(s):
		raise Exception("fail")

	caught = False
	try:
		us.ever_set(fail)
	except:
		caught = True
	assert(caught)

# Check basic construction of spaces.
#
def test_space():
	unit = isl.space.unit()
	set_space = unit.add_named_tuple("A", 3)
	map_space = set_space.add_named_tuple("B", 2)

	set = isl.set.universe(set_space)
	map = isl.map.universe(map_space)
	assert(set.is_equal(isl.set("{ A[*,*,*] }")))
	assert(map.is_equal(isl.map("{ A[*,*,*] -> B[*,*] }")))

# Construct a simple schedule tree with an outer sequence node and
# a single-dimensional band node in each branch, with one of them
# marked coincident.
#
def construct_schedule_tree():
	A = isl.union_set("{ A[i] : 0 <= i < 10 }")
	B = isl.union_set("{ B[i] : 0 <= i < 20 }")

	node = isl.schedule_node.from_domain(A.union(B))
	node = node.child(0)

	filters = isl.union_set_list(A).add(B)
	node = node.insert_sequence(filters)

	f_A = isl.multi_union_pw_aff("[ { A[i] -> [i] } ]")
	node = node.child(0)
	node = node.child(0)
	node = node.insert_partial_schedule(f_A)
	node = node.member_set_coincident(0, True)
	node = node.ancestor(2)

	f_B = isl.multi_union_pw_aff("[ { B[i] -> [i] } ]")
	node = node.child(1)
	node = node.child(0)
	node = node.insert_partial_schedule(f_B)
	node = node.ancestor(2)

	return node.schedule()

# Test basic schedule tree functionality.
#
# In particular, create a simple schedule tree and
# - check that the root node is a domain node
# - test map_descendant_bottom_up
# - test foreach_descendant_top_down
# - test every_descendant
#
def test_schedule_tree():
	schedule = construct_schedule_tree()
	root = schedule.root()

	assert(type(root) == isl.schedule_node_domain)

	count = [0]
	def inc_count(node):
		count[0] += 1
		return node
	root = root.map_descendant_bottom_up(inc_count)
	assert(count[0] == 8)

	def fail_map(node):
		raise Exception("fail")
		return node
	caught = False
	try:
		root.map_descendant_bottom_up(fail_map)
	except:
		caught = True
	assert(caught)

	count = [0]
	def inc_count(node):
		count[0] += 1
		return True
	root.foreach_descendant_top_down(inc_count)
	assert(count[0] == 8)

	count = [0]
	def inc_count(node):
		count[0] += 1
		return False
	root.foreach_descendant_top_down(inc_count)
	assert(count[0] == 1)

	def is_not_domain(node):
		return type(node) != isl.schedule_node_domain
	assert(root.child(0).every_descendant(is_not_domain))
	assert(not root.every_descendant(is_not_domain))

	def fail(node):
		raise Exception("fail")
	caught = False
	try:
		root.every_descendant(fail)
	except:
		caught = True
	assert(caught)

	domain = root.domain()
	filters = [isl.union_set("{}")]
	def collect_filters(node):
		if type(node) == isl.schedule_node_filter:
			filters[0] = filters[0].union(node.filter())
		return True
	root.every_descendant(collect_filters)
	assert(domain.is_equal(filters[0]))

# Test marking band members for unrolling.
# "schedule" is the schedule created by construct_schedule_tree.
# It schedules two statements, with 10 and 20 instances, respectively.
# Unrolling all band members therefore results in 30 at-domain calls
# by the AST generator.
#
def test_ast_build_unroll(schedule):
	root = schedule.root()
	def mark_unroll(node):
		if type(node) == isl.schedule_node_band:
			node = node.member_set_ast_loop_unroll(0)
		return node
	root = root.map_descendant_bottom_up(mark_unroll)
	schedule = root.schedule()

	count_ast = [0]
	def inc_count_ast(node, build):
		count_ast[0] += 1
		return node

	build = isl.ast_build()
	build = build.set_at_each_domain(inc_count_ast)
	ast = build.node_from(schedule)
	assert(count_ast[0] == 30)

# Test basic AST generation from a schedule tree.
#
# In particular, create a simple schedule tree and
# - generate an AST from the schedule tree
# - test at_each_domain
# - test unrolling
#
def test_ast_build():
	schedule = construct_schedule_tree()

	count_ast = [0]
	def inc_count_ast(node, build):
		count_ast[0] += 1
		return node

	build = isl.ast_build()
	build_copy = build.set_at_each_domain(inc_count_ast)
	ast = build.node_from(schedule)
	assert(count_ast[0] == 0)
	count_ast[0] = 0
	ast = build_copy.node_from(schedule)
	assert(count_ast[0] == 2)
	build = build_copy
	count_ast[0] = 0
	ast = build.node_from(schedule)
	assert(count_ast[0] == 2)

	do_fail = True
	count_ast_fail = [0]
	def fail_inc_count_ast(node, build):
		count_ast_fail[0] += 1
		if do_fail:
			raise Exception("fail")
		return node
	build = isl.ast_build()
	build = build.set_at_each_domain(fail_inc_count_ast)
	caught = False
	try:
		ast = build.node_from(schedule)
	except:
		caught = True
	assert(caught)
	assert(count_ast_fail[0] > 0)
	build_copy = build
	build_copy = build_copy.set_at_each_domain(inc_count_ast)
	count_ast[0] = 0
	ast = build_copy.node_from(schedule)
	assert(count_ast[0] == 2)
	count_ast_fail[0] = 0
	do_fail = False
	ast = build.node_from(schedule)
	assert(count_ast_fail[0] == 2)

	test_ast_build_unroll(schedule)

# Test basic AST expression generation from an affine expression.
#
def test_ast_build_expr():
	pa = isl.pw_aff("[n] -> { [n + 1] }")
	build = isl.ast_build.from_context(pa.domain())

	op = build.expr_from(pa)
	assert(type(op) == isl.ast_expr_op_add)
	assert(op.n_arg() == 2)

# Test the isl Python interface
#
# This includes:
#  - Object construction
#  - Different parameter types
#  - Different return types
#  - Foreach functions
#  - Every functions
#  - Spaces
#  - Schedule trees
#  - AST generation
#  - AST expression generation
#
test_constructors()
test_parameters()
test_return()
test_foreach()
test_every()
test_space()
test_schedule_tree()
test_ast_build()
test_ast_build_expr()
