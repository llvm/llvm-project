# Check that regex-XFAILing works and can be configured via env var.
#
# RUN: %{lit} --xfail 'false.txt;false2.txt' %{inputs}/xfail-cl | FileCheck --check-prefix=CHECK-FILTER %s
# RUN: env LIT_XFAIL='false.txt;false2.txt' %{lit} %{inputs}/xfail-cl | FileCheck --check-prefix=CHECK-FILTER %s
# END.
# CHECK-FILTER: Testing: 3 tests, {{[1-3]}} workers
# CHECK-FILTER-DAG: XFAIL: top-level-suite :: false.txt
# CHECK-FILTER-DAG: XFAIL: top-level-suite :: false2.txt
# CHECK-FILTER-DAG: PASS: top-level-suite :: true.txt
