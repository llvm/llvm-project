# Check that marking tests as UNSUPPORTED works via command line or env var.

# RUN: %{lit} --unsupported 'true.txt' \
# RUN:   %{inputs}/xfail-cl/true.txt \
# RUN: | FileCheck --check-prefix=CHECK-UNSUPPORTED %s

# RUN: env LIT_UNSUPPORTED='true.txt' \
# RUN: %{lit} %{inputs}/xfail-cl/true.txt \
# RUN: | FileCheck --check-prefix=CHECK-UNSUPPORTED %s

# Check that --unsupported-not and LIT_UNSUPPORTED_NOT override --unsupported.

# RUN: %{lit} --unsupported 'true.txt' --unsupported-not 'true.txt' \
# RUN:   %{inputs}/xfail-cl/true.txt \
# RUN: | FileCheck --check-prefix=CHECK-NOT-UNSUPPORTED %s

# RUN: env LIT_UNSUPPORTED='true.txt' LIT_UNSUPPORTED_NOT='true.txt' \
# RUN: %{lit} %{inputs}/xfail-cl/true.txt \
# RUN: | FileCheck --check-prefix=CHECK-NOT-UNSUPPORTED %s

# END.

# CHECK-UNSUPPORTED: Testing: 1 tests, {{[0-9]*}} workers
# CHECK-UNSUPPORTED: {{^}}UNSUPPORTED: top-level-suite :: true.txt

# CHECK-NOT-UNSUPPORTED: Testing: 1 tests, {{[0-9]*}} workers
# CHECK-NOT-UNSUPPORTED: {{^}}PASS: top-level-suite :: true.txt
