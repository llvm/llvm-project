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
		raise "fail"

	caught = False
	try:
		s.foreach_basic_set(fail)
	except:
		caught = True
	assert(caught)

# Test the isl Python interface
#
# This includes:
#  - Object construction
#  - Different parameter types
#  - Different return types
#  - Foreach functions
#
test_constructors()
test_parameters()
test_return()
test_foreach()
