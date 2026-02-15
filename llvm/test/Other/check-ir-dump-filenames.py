#!/usr/bin/python3

import re
import sys
import os

dir = sys.argv[1]
pat = pat_arg = sys.argv[2:]
pat = "".join(pat)
pat = re.compile(pat)

filenames = os.listdir(dir)
if len(filenames) < 2:
    print(f"expecting at least 2 files in {dir} but got {len(filenames)}")
    sys.exit(1)


def fndict(fn):
    m = pat.match(fn)
    if not m:
        print(f"filename '{fn}' does not match pattern '{pat_arg}'")
        sys.exit(1)
    return m.groupdict()


first = fndict(filenames[0])
should_check_ordinal = "ordinal" in first
should_check_suffix = "suffix" in first
should_check_suffix_ordinal = should_check_suffix and "suffix_ordinal" in first
should_check_before_after = should_check_suffix and should_check_ordinal

should_check_pass_id = "pass_id" in first
should_check_kind = "kind" in first
should_check_pass_number = "pass_number" in first

if should_check_ordinal:
    filenames = sorted(filenames, key=lambda x: fndict(x)["ordinal"])
first = fndict(filenames[0])
second = fndict(filenames[1])


def failed(msg):
    print(f"error: {msg}")
    sys.exit(1)


def check(actual, expected, name):
    if actual != expected:
        failed(f"error: expected {name} '{expected}' but got '{actual}' ")


# ------------------------------------------------------------------------------
# ordinal
# ------------------------------------------------------------------------------
def check_ordinal(d, prev):
    if not should_check_ordinal:
        return
    actual = int(d["ordinal"])
    expected = int(prev["ordinal"]) + 1
    check(actual, expected, "ordinal")


# ------------------------------------------------------------------------------
# suffix
# ------------------------------------------------------------------------------
suffix_ordinal_list = ["before", "after"]
suffix_map = {}
if should_check_before_after:
    suffix_map |= {first["suffix"]: second["suffix"]}
    suffix_map |= {second["suffix"]: first["suffix"]}


def check_suffix(d, prev):
    if not should_check_suffix:
        return
    suffix = d["suffix"]
    if should_check_before_after:
        if suffix not in suffix_map:
            failed(f"suffix '{suffix}' not in {suffix_map}")
        if prev is not None:
            check(suffix, suffix_map[prev["suffix"]], "suffix")
    else:
        if suffix not in suffix_ordinal_list:
            failed(f"suffix '{suffix}' not in {suffix_ordinal_list}")
    if should_check_suffix_ordinal:
        suffix_ordinal = int(d["suffix_ordinal"])
        check(suffix_ordinal, suffix_ordinal_list.index(suffix), "suffix ordinal")


# ------------------------------------------------------------------------------
# pass number
# ------------------------------------------------------------------------------
def check_pass_number(d, prev):
    if not should_check_pass_number or not should_check_before_after:
        return
    actual = d["pass_number"]
    if d["suffix"] == "after":
        expected = prev["pass_number"]
        check(actual, expected, "pass number")


# ------------------------------------------------------------------------------
# pass id
# ------------------------------------------------------------------------------
def check_pass_id(d, prev):
    if not should_check_pass_id or not should_check_before_after:
        return
    actual = d["pass_id"]
    if d["suffix"] == "after":
        expected = prev["pass_id"]
        check(actual, expected, "pass id")


# ------------------------------------------------------------------------------
# kind
# ------------------------------------------------------------------------------
def check_kind(d, prev):
    if not should_check_kind or not should_check_before_after:
        return
    actual = d["kind"]
    if d["suffix"] == "after":
        expected = prev["kind"]
        check(actual, expected, "kind")


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
check_ordinal(first, {"ordinal": -1})

prev = first
filenames = filenames[1:]
for fn in filenames:
    d = fndict(fn)
    check_ordinal(d, prev)
    check_suffix(d, prev)
    check_pass_id(d, prev)
    check_pass_number(d, prev)
    check_kind(d, prev)
    prev = d
