# Check that marking tests as UNSUPPORTED works via command line or env var.

# RUN: %{lit} --unsupported 'false.txt;false2.txt;top-level-suite :: b :: test.txt' \
# RUN:   --unsupported-not 'true-xfail.txt;top-level-suite :: a :: test-xfail.txt' \
# RUN:   %{inputs}/xfail-cl \
# RUN: | FileCheck --check-prefix=CHECK-FILTER %s

# RUN: env LIT_UNSUPPORTED='false.txt;false2.txt;top-level-suite :: b :: test.txt' \
# RUN:   LIT_UNSUPPORTED_NOT='true-xfail.txt;top-level-suite :: a :: test-xfail.txt' \
# RUN: %{lit} %{inputs}/xfail-cl \
# RUN: | FileCheck --check-prefix=CHECK-FILTER %s

# Check that --unsupported-not and LIT_UNSUPPORTED_NOT always have precedence.

# RUN: env LIT_UNSUPPORTED=true-xfail.txt \
# RUN: %{lit} --unsupported true-xfail.txt --unsupported-not true-xfail.txt \
# RUN:   --unsupported true-xfail.txt %{inputs}/xfail-cl/true-xfail.txt \
# RUN: | FileCheck --check-prefix=CHECK-OVERRIDE %s

# RUN: env LIT_UNSUPPORTED_NOT=true-xfail.txt LIT_UNSUPPORTED=true-xfail.txt \
# RUN: %{lit} --unsupported true-xfail.txt %{inputs}/xfail-cl/true-xfail.txt \
# RUN: | FileCheck --check-prefix=CHECK-OVERRIDE %s

# END.

# CHECK-FILTER: Testing: 11 tests, {{[0-9]*}} workers
# CHECK-FILTER-DAG: {{^}}PASS: top-level-suite :: a :: test.txt
# CHECK-FILTER-DAG: {{^}}UNSUPPORTED: top-level-suite :: b :: test.txt
# CHECK-FILTER-DAG: {{^}}UNSUPPORTED: top-level-suite :: a :: false.txt
# CHECK-FILTER-DAG: {{^}}UNSUPPORTED: top-level-suite :: b :: false.txt
# CHECK-FILTER-DAG: {{^}}UNSUPPORTED: top-level-suite :: false.txt
# CHECK-FILTER-DAG: {{^}}UNSUPPORTED: top-level-suite :: false2.txt
# CHECK-FILTER-DAG: {{^}}PASS: top-level-suite :: true.txt
# CHECK-FILTER-DAG: {{^}}PASS: top-level-suite :: true-xfail.txt
# CHECK-FILTER-DAG: {{^}}PASS: top-level-suite :: a :: test-xfail.txt
# CHECK-FILTER-DAG: {{^}}UNSUPPORTED: top-level-suite :: b :: test-xfail.txt

# CHECK-OVERRIDE: Testing: 1 tests, {{[0-9]*}} workers
# CHECK-OVERRIDE: {{^}}PASS: top-level-suite :: true-xfail.txt
