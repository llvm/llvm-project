# Basic sanity check that --version works.
#
# RUN: %{lit} --version | FileCheck %s
#
# CHECK: lit {{[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9]*}}
