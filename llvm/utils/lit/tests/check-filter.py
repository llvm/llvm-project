# Verify lit's --check filter. --check=N drops every CommandDirective except
# the Nth (0-indexed).

# --- --check=0: keep only the first RUN ---
# RUN: %{lit} -a --check=0 %{inputs}/check-filter/sample.ll \
# RUN:   | FileCheck --check-prefix=CHECK-0 \
# RUN:       --implicit-check-not="executed command: echo SECOND-RUN" %s
#
# CHECK-0:     executed command: echo FIRST-RUN

# --- --check=1: skip the first, keep the second ---
# RUN: %{lit} -a --check=1 %{inputs}/check-filter/sample.ll \
# RUN:   | FileCheck --check-prefix=CHECK-1 \
# RUN:       --implicit-check-not="executed command: echo FIRST-RUN" %s
#
# CHECK-1:     executed command: echo SECOND-RUN

# --- No filter: both RUNs execute ---
# RUN: %{lit} -a %{inputs}/check-filter/sample.ll \
# RUN:   | FileCheck --check-prefix=NO-FILTER %s
#
# NO-FILTER:   executed command: echo FIRST-RUN
# NO-FILTER:   executed command: echo SECOND-RUN
