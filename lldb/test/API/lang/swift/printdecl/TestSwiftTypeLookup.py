# TestSwiftTypeLookup.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test the ability to look for type definitions at the command line
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftTypeLookup(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @swiftTest
    def test_swift_type_lookup(self):
        """Test the ability to look for type definitions at the command line"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        # for now, type lookup won't load the AST context, so force load it
        # before testing
        self.runCmd("expr 1")

        # check that basic cases work
        self.expect(
            "type lookup String",
            substrs=[
                '(struct Swift.String)',
                'struct String {',
                'extension Swift.String : '])
        self.expect(
            "type lookup Cla1",
            substrs=[
                '(class a.Cla1)',
                'class Cla1 {',
                'func bat(_ x: Swift.Int, y: a.Str1) -> Swift.Int'])
        self.expect(
            "type lookup Str1",
            substrs=[
                '(struct a.Str1)',
                'struct Str1 {',
                'func bar()',
                'var b'])

        # Regression test. Ensure "<could not resolve type>" is not output.
        self.expect(
            "type lookup String",
            matching=False,
            substrs=["<could not resolve type>"])

        # check that specifiers are honored
        # self.expect('type lookup class Cla1', substrs=['class Cla1 {'])
        # self.expect('type lookup struct Cla1', substrs=['class Cla1 {'], matching=False, error=True)

        # check that modules are honored
        self.expect("type lookup Swift.String",
                    substrs=['(struct Swift.String)',
                             'struct String {'])
        self.expect(
            "type lookup a.String",
            substrs=['(struct Swift.String)',
                     'struct String {'],
            matching=False)

        # check that a combination of module and specifier is honored
        # self.expect('type lookup class a.Cla1', substrs=['class Cla1 {'])
        # self.expect('type lookup class Swift.Cla1', substrs=['class Cla1 {'], matching=False, error=True)
        # self.expect('type lookup struct a.Cla1', substrs=['class Cla1 {'], matching=False, error=True)
        # self.expect('type lookup struct Swift.Cla1', substrs=['class Cla1 {'], matching=False, error=True)

        # check that nested types are looked up correctly
        self.expect('type lookup Toplevel.Nested.Deeper',
                    substrs=['class a.Toplevel.Nested.Deeper',
                             'struct a.Toplevel.Nested',
                             'class a.Toplevel',
                             'class Deeper', 
                             'func foo'])

        # check that mangled name lookup works
        self.expect(
            'type lookup _$sSiD',
            substrs=[
                'struct Int',
                'extension Swift.Int'])

        # check that we can look for generic things
        self.expect('type lookup Generic', substrs=['class Generic', 'foo'])
        self.expect(
            'type lookup Generic<String>',
            substrs=[
                'bound_generic_class a.Generic',
                'struct Swift.String',
                'class Generic',
                'func foo'])

        # check that we print comment text (and let you get rid of it)
        self.expect(
            'type lookup Int',
            substrs=['Creates a new instance'],
            matching=False)
        self.expect(
            'type lookup --show-help -- Int',
            substrs=['Creates a new instance'],
            matching=True)
        self.expect(
            'type lookup foo',
            substrs=[
                'func foo',
                'Int',
                'Double'],
            matching=True,
            ordered=False)
        self.expect(
            'type lookup --show-help -- print',
            substrs=[
                '/// ',
                'func print'],
            matching=True)
