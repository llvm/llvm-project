# Basic sanity check that --version works.
#
# RUN: %{lit} --version 2>&1 | FileCheck %s
#
# CHECK: lit {{[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9]*}}
