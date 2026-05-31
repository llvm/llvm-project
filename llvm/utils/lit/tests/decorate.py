# Verify --param decorate=PROG[@N] prepends PROG to pipeline stage N (default 0).

# --- Default stage (0): PROG prepended to the first stage ---
# RUN: %{lit} -a --param decorate=time %{inputs}/decorate/sample.ll \
# RUN:   | FileCheck --check-prefix=STAGE0 %s
#
# STAGE0: time echo opt {{.*}}sample.ll -S -passes='instcombine' | echo FileCheck {{.*}}sample.ll

# --- Explicit @0 is the same as default ---
# RUN: %{lit} -a --param decorate=time@0 %{inputs}/decorate/sample.ll \
# RUN:   | FileCheck --check-prefix=STAGE0 %s

# --- @1: PROG prepended to the second stage ---
# RUN: %{lit} -a --param decorate=time@1 %{inputs}/decorate/sample.ll \
# RUN:   | FileCheck --check-prefix=STAGE1 %s
#
# STAGE1: echo opt {{.*}}sample.ll -S -passes='instcombine' | time echo FileCheck {{.*}}sample.ll

# --- @N past the end: no-op, pipeline unchanged ---
# RUN: %{lit} -a --param decorate=time@9 %{inputs}/decorate/sample.ll \
# RUN:   | FileCheck --check-prefix=OOB %s
#
# OOB-NOT: time
# OOB: echo opt {{.*}}sample.ll -S -passes='instcombine' | echo FileCheck {{.*}}sample.ll

# --- No --param decorate: pipeline unchanged ---
# RUN: %{lit} -a %{inputs}/decorate/sample.ll | FileCheck --check-prefix=NONE %s
#
# NONE-NOT: time
# NONE: echo opt {{.*}}sample.ll -S -passes='instcombine' | echo FileCheck {{.*}}sample.ll
