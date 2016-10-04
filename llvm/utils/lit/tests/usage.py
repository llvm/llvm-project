# Basic sanity check that usage works.
#
# RUN: %{lit} --help > %t.out
# RUN: FileCheck < %t.out %s
#
# CHECK: usage: lit.py [-h]
