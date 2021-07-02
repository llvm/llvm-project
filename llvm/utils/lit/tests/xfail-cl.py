# Check that XFAILing works via command line or env var.

# RUN: %{lit} --xfail 'false.txt;false2.txt;top-level-suite :: b :: test.txt' \
# RUN:        %{inputs}/xfail-cl \
# RUN: | FileCheck --check-prefix=CHECK-FILTER %s

# RUN: env LIT_XFAIL='false.txt;false2.txt;top-level-suite :: b :: test.txt' \
# RUN: %{lit} %{inputs}/xfail-cl \
# RUN: | FileCheck --check-prefix=CHECK-FILTER %s

# END.

# CHECK-FILTER: Testing: 7 tests, {{[1-7]}} workers
# CHECK-FILTER-DAG: {{^}}PASS: top-level-suite :: a :: test.txt
# CHECK-FILTER-DAG: {{^}}XFAIL: top-level-suite :: b :: test.txt
# CHECK-FILTER-DAG: {{^}}XFAIL: top-level-suite :: a :: false.txt
# CHECK-FILTER-DAG: {{^}}XFAIL: top-level-suite :: b :: false.txt
# CHECK-FILTER-DAG: {{^}}XFAIL: top-level-suite :: false.txt
# CHECK-FILTER-DAG: {{^}}XFAIL: top-level-suite :: false2.txt
# CHECK-FILTER-DAG: {{^}}PASS: top-level-suite :: true.txt
