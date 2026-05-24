# Verify lit's --fn flag injects the select-function pass into -passes= args.

# --- --fn=foo: single function ---
# RUN: %{lit} -a --fn=foo %{inputs}/fn-selection/sample.ll \
# RUN:   | FileCheck --check-prefix=SINGLE %s
#
# SINGLE: -passes='select-function<fn=foo>,instcombine,mem2reg'
# SINGLE: -passes="select-function<fn=foo>,instcombine,mem2reg"

# --- --fn=foo,bar: multiple functions ---
# RUN: %{lit} -a --fn=foo,bar %{inputs}/fn-selection/sample.ll \
# RUN:   | FileCheck --check-prefix=MULTI %s
#
# MULTI: -passes='select-function<fn=foo;fn=bar>,instcombine,mem2reg'
# MULTI: -passes="select-function<fn=foo;fn=bar>,instcombine,mem2reg"

# --- No --fn: passes unchanged ---
# RUN: %{lit} -a %{inputs}/fn-selection/sample.ll \
# RUN:   | FileCheck --check-prefix=NONE %s
#
# NONE-NOT: select-function
# NONE: -passes='instcombine,mem2reg'
# NONE: -passes="instcombine,mem2reg"
