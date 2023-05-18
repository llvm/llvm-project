#!/usr/bin/env python

from __future__ import print_function

import ctypes
import random
import gmpapi
import wrappers
import glob
import sys
import os
from optparse import OptionParser
from gmpapi import void
from gmpapi import ilong
from gmpapi import ulong
from gmpapi import mpz_t
from gmpapi import voidp
from gmpapi import size_t
from gmpapi import size_tp
from gmpapi import iint
from gmpapi import charp
from gmpapi import mpq_t


def print_failure(line, test):
    print("FAIL: {}@{}".format(line, test))


def run_tests(test_file, options):
    passes = 0
    failures = 0
    fail_lines = []
    for (line, test) in enumerate(open(test_file), start=1):
        if test.startswith("#"):
            continue
        if options.skip > 0 and line < options.skip:
            continue
        name, args = test.split("|")
        if options.verbose or (options.progress > 0 and line % options.progress == 0):
            print("TEST: {}@{}".format(line, test), end="")
        api = gmpapi.get_api(name)
        wrapper = wrappers.get_wrapper(name)
        input_args = args.split(",")
        if len(api.params) != len(input_args):
            raise RuntimeError(
                "Mismatch in args length: {} != {}".format(
                    len(api.params), len(input_args)
                )
            )

        call_args = []
        for i in range(len(api.params)):
            param = api.params[i]
            if param == mpz_t:
                call_args.append(bytes(input_args[i]).encode("utf-8"))
            elif param == mpq_t:
                call_args.append(bytes(input_args[i]).encode("utf-8"))
            elif param == ulong:
                call_args.append(ctypes.c_ulong(int(input_args[i])))
            elif param == ilong:
                call_args.append(ctypes.c_long(int(input_args[i])))
            elif param == voidp or param == size_tp:
                call_args.append(ctypes.c_void_p(None))
            elif param == size_t:
                call_args.append(ctypes.c_size_t(int(input_args[i])))
            elif param == iint:
                call_args.append(ctypes.c_int(int(input_args[i])))
            # pass null for charp
            elif param == charp:
                if input_args[i] == "NULL":
                    call_args.append(ctypes.c_void_p(None))
                else:
                    call_args.append(bytes(input_args[i]).encode("utf-8"))
            else:
                raise RuntimeError("Unknown param type: {}".format(param))

        res = wrappers.run_test(
            wrapper, line, name, gmp_test_so, imath_test_so, *call_args
        )
        if not res:
            failures += 1
            print_failure(line, test)
            fail_lines.append((line, test))
        else:
            passes += 1
    return (passes, failures, fail_lines)


def parse_args():
    parser = OptionParser()
    parser.add_option(
        "-f",
        "--fork",
        help="fork() before each operation",
        action="store_true",
        default=False,
    )
    parser.add_option(
        "-v",
        "--verbose",
        help="print PASS and FAIL tests",
        action="store_true",
        default=False,
    )
    parser.add_option(
        "-p",
        "--progress",
        help="print progress every N tests ",
        metavar="N",
        type="int",
        default=0,
    )
    parser.add_option(
        "-s", "--skip", help="skip to test N", metavar="N", type="int", default=0
    )
    return parser.parse_args()


if __name__ == "__main__":
    (options, tests) = parse_args()
    gmp_test_so = ctypes.cdll.LoadLibrary("gmp_test.so")
    imath_test_so = ctypes.cdll.LoadLibrary("imath_test.so")

    wrappers.verbose = options.verbose
    wrappers.fork = options.fork

    total_pass = 0
    total_fail = 0
    all_fail_lines = []
    for test_file in tests:
        print("Running tests in {}".format(test_file))
        (passes, failures, fail_lines) = run_tests(test_file, options)
        print(
            "  Tests: {}. Passes: {}. Failures: {}.".format(
                passes + failures, passes, failures
            )
        )
        total_pass += passes
        total_fail += failures
        all_fail_lines += fail_lines

    print("=" * 70)
    print("Total")
    print(
        "  Tests: {}. Passes: {}. Failures: {}.".format(
            total_pass + total_fail, total_pass, total_fail
        )
    )
    if len(all_fail_lines) > 0:
        print("Failing Tests:")
        for (line, test) in all_fail_lines:
            print(test.rstrip())
