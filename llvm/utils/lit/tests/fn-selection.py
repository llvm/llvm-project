# Verify --param fn=NAMES splices select-function into -passes= pipelines via
# lit.llvm.fn_selection (which is also wired into llvm/test/lit.cfg.py).

# --- --param fn=foo: single function ---
# RUN: %{lit} -a --param fn=foo %{inputs}/fn-selection/sample.ll \
# RUN:   | FileCheck --check-prefix=SINGLE %s
#
# SINGLE: -passes='select-function<fn=foo>,instcombine,mem2reg'
# SINGLE: -passes="select-function<fn=foo>,instcombine,mem2reg"

# --- --param fn=foo,bar: multiple functions ---
# RUN: %{lit} -a --param fn=foo,bar %{inputs}/fn-selection/sample.ll \
# RUN:   | FileCheck --check-prefix=MULTI %s
#
# MULTI: -passes='select-function<fn=foo;fn=bar>,instcombine,mem2reg'
# MULTI: -passes="select-function<fn=foo;fn=bar>,instcombine,mem2reg"

# --- No --param: passes unchanged ---
# RUN: %{lit} -a %{inputs}/fn-selection/sample.ll \
# RUN:   | FileCheck --check-prefix=NONE %s
#
# NONE-NOT: select-function
# NONE: -passes='instcombine,mem2reg'
# NONE: -passes="instcombine,mem2reg"
