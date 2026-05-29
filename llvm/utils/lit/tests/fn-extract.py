# Verify --param fn=NAMES prepends llvm-extract to the first pipeline stage
# via lit.llvm.fn_extract (which is also wired into llvm/test/lit.cfg.py).

# --- --param fn=foo: single function ---
# RUN: %{lit} -a --param fn=foo %{inputs}/fn-extract/sample.ll \
# RUN:   | FileCheck --check-prefix=SINGLE %s
#
# Positional `%s` form:
# SINGLE: llvm-extract --func=foo {{.*}}sample.ll -o - | echo opt -S -passes='instcombine' | echo FileCheck {{.*}}sample.ll
# Redirect `< %s` form:
# SINGLE: llvm-extract --func=foo {{.*}}sample.ll -o - | echo opt -S -passes="instcombine" | echo FileCheck {{.*}}sample.ll

# --- --param fn=foo,bar: multiple functions ---
# RUN: %{lit} -a --param fn=foo,bar %{inputs}/fn-extract/sample.ll \
# RUN:   | FileCheck --check-prefix=MULTI %s
#
# MULTI: llvm-extract --func=foo --func=bar {{.*}}sample.ll -o - | echo opt -S -passes='instcombine' | echo FileCheck {{.*}}sample.ll
# MULTI: llvm-extract --func=foo --func=bar {{.*}}sample.ll -o - | echo opt -S -passes="instcombine" | echo FileCheck {{.*}}sample.ll

# --- No --param: pipeline unchanged ---
# RUN: %{lit} -a %{inputs}/fn-extract/sample.ll \
# RUN:   | FileCheck --check-prefix=NONE %s
#
# NONE-NOT: llvm-extract
# NONE: echo opt {{.*}}sample.ll -S -passes='instcombine'
# NONE: echo opt < {{.*}}sample.ll -S -passes="instcombine"
