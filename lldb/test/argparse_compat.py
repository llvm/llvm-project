#!/usr/bin/env python

"""
Compatibility module to use the lldb test-suite with Python 2.6.

Warning: This may be buggy. It has not been extensively tested and should only
be used when it is impossible to use a newer Python version.
It is also a special-purpose class for lldb's test-suite.
"""

import sys

if sys.version_info >= (2, 7):
    raise "This module shouldn't be used when argparse is available (Python >= 2.7)"
else:
    print "Using Python 2.6 compatibility layer. Some command line options may not be supported"


import optparse


class ArgumentParser(object):
    def __init__(self, description="My program's description", prefix_chars='-', add_help=True):
        self.groups = []
        self.parser = optparse.OptionParser(description=description, add_help_option=add_help)
        self.prefix_chars = prefix_chars

    def add_argument_group(self, name):
        group = optparse.OptionGroup(self.parser, name)
        # Hack around our test directories argument (what's left after the
        # options)
        if name != 'Test directories':
            self.groups.append(group)
        return ArgumentGroup(group)

    def add_argument(self, *opt_strs, **kwargs):
        self.parser.add_option(*opt_strs, **kwargs)
    # def add_argument(self, opt_str, action='store', dest=None, metavar=None, help=''):
    #     if dest is None and metavar is None:
    #         self.parser.add_argument(opt_str, action=action, help=help)

    def parse_args(self, arguments=sys.argv[1:]):
        map(lambda g: self.parser.add_option_group(g), self.groups)
        (options, args) = self.parser.parse_args(arguments)
        d = vars(options)
        d['args'] = args
        return Namespace(d)

    def print_help(self):
        self.parser.print_help()


class ArgumentGroup(object):
    def __init__(self, option_group):
        self.option_group = option_group

    def add_argument(self, *opt_strs, **kwargs):
        # Hack around our positional argument (the test directories)
        if opt_strs == ('args',):
            return

        # Hack around the options that start with '+'
        if len(opt_strs) == 1 and opt_strs[0] == '+a':
            opt_strs = ('--plus_a',)
        if len(opt_strs) == 1 and opt_strs[0] == '+b':
            opt_strs = ('--plus_b',)
        self.option_group.add_option(*opt_strs, **kwargs)


class Namespace(object):
    def __init__(self, d):
        self.__dict__ = d

    def __str__(self):
        strings = []
        for (k, v) in self.__dict__.iteritems():
            strings.append(str(k) + '=' + str(v))
        strings.sort()

        return self.__class__.__name__ + '(' + ', '.join(strings) + ')'
