# Verify lit's --fn flag prepends llvm-extract to pipelines.

# --- --fn=foo: single function ---
# RUN: %{lit} -a --fn=foo %{inputs}/fn-selection/sample.ll \
# RUN:   | FileCheck --check-prefix=SINGLE %s
#
# Positional %s form:
# SINGLE: llvm-extract --func=foo {{.*}}sample.ll -o - | echo opt  -S -passes='instcombine' | echo FileCheck {{.*}}sample.ll
# Redirect < %s form:
# SINGLE: llvm-extract --func=foo {{.*}}sample.ll -o - | echo opt  -S -passes="instcombine" | echo FileCheck {{.*}}sample.ll

# --- --fn=foo,bar: multiple functions ---
# RUN: %{lit} -a --fn=foo,bar %{inputs}/fn-selection/sample.ll \
# RUN:   | FileCheck --check-prefix=MULTI %s
#
# MULTI: llvm-extract --func=foo --func=bar {{.*}}sample.ll -o - | echo opt  -S -passes='instcombine' | echo FileCheck {{.*}}sample.ll
# MULTI: llvm-extract --func=foo --func=bar {{.*}}sample.ll -o - | echo opt  -S -passes="instcombine" | echo FileCheck {{.*}}sample.ll

# --- No --fn: passes unchanged ---
# RUN: %{lit} -a %{inputs}/fn-selection/sample.ll \
# RUN:   | FileCheck --check-prefix=NONE %s
#
# NONE-NOT: llvm-extract
# NONE: echo opt {{.*}}sample.ll -S -passes='instcombine'
# NONE: echo opt < {{.*}}sample.ll -S -passes="instcombine"
