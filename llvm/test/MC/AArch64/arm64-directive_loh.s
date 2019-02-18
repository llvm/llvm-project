# RUN: not llvm-mc -triple arm64-apple-darwin < %s 2> %t | FileCheck %s
# RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s
# RUN: not llvm-mc -triple aarch64-linux-gnu < %s 2>&1 | FileCheck --check-prefix=UNKNOWN %s
# RUN: not llvm-mc -triple aarch64-win32-gnu < %s 2>&1 | FileCheck --check-prefix=UNKNOWN %s

.globl _fct1
_fct1:
  L1:
  L2:
  L3:
  L4:
  ret lr;

# Known LOHs with:
# - Regular syntax.
# - Alternative syntax.

# CHECK: .loh AdrpAdrp L1, L2
# CHECK: .loh AdrpAdrp L1, L2
# UNKNOWN: error: unknown directive
# UNKNOWN-NEXT: .loh AdrpAdrp L1, L2
.loh AdrpAdrp L1, L2
.loh 1 L1, L2

# CHECK: .loh AdrpLdr L1, L2
# CHECK: .loh AdrpLdr L1, L2
.loh AdrpLdr L1, L2
.loh 2 L1, L2

# CHECK: .loh AdrpAddLdr L1, L2, L3
# CHECK: .loh AdrpAddLdr L1, L2, L3
.loh AdrpAddLdr L1, L2, L3
.loh 3 L1, L2, L3

# CHECK: .loh AdrpLdrGotLdr L1, L2, L3
# CHECK: .loh AdrpLdrGotLdr L1, L2, L3
.loh AdrpLdrGotLdr L1, L2, L3
.loh 4 L1, L2, L3

# CHECK: .loh AdrpAddStr L1, L2, L3
# CHECK: .loh AdrpAddStr L1, L2, L3
.loh AdrpAddStr L1, L2, L3
.loh 5 L1, L2, L3

# CHECK: .loh AdrpLdrGotStr L1, L2, L3
# CHECK: .loh AdrpLdrGotStr L1, L2, L3
.loh AdrpLdrGotStr L1, L2, L3
.loh 6 L1, L2, L3

# CHECK: .loh AdrpAdd L1, L2
# CHECK: .loh AdrpAdd L1, L2
.loh AdrpAdd L1, L2
.loh 7 L1, L2

# CHECK: .loh AdrpLdrGot L1, L2
# CHECK: .loh AdrpLdrGot L1, L2
.loh AdrpLdrGot L1, L2
.loh 8 L1, L2

# End Known LOHs.

### Errors Check ####

# Unknown textual identifier.
# CHECK-ERRORS: error: invalid identifier in directive
# CHECK-ERRORS-NEXT: .loh Unknown
# CHECK-ERRORS-NEXT:      ^
.loh Unknown
# Unknown numeric identifier.
# CHECK-ERRORS: error: invalid numeric identifier in directive
# CHECK-ERRORS-NEXT: .loh 153, L1
# CHECK-ERRORS-NEXT:      ^
.loh 153, L1

# Too much arguments.
# CHECK-ERRORS: error: unexpected token in '.loh' directive
# CHECK-ERRORS-NEXT: .loh AdrpAdrp L1, L2, L3
# CHECK-ERRORS-NEXT:                     ^
.loh AdrpAdrp L1, L2, L3

# Too much arguments with alternative syntax.
# CHECK-ERRORS: error: unexpected token in '.loh' directive
# CHECK-ERRORS-NEXT: .loh 1 L1, L2, L3
# CHECK-ERRORS-NEXT:              ^
.loh 1 L1, L2, L3

# Too few argumets.
# CHECK-ERRORS: error: unexpected token in '.loh' directive
# CHECK-ERRORS-NEXT: .loh AdrpAdrp L1
# CHECK-ERRORS-NEXT:                 ^
.loh AdrpAdrp L1

# Too few argumets with alternative syntax.
# CHECK-ERRORS: error: unexpected token in '.loh' directive
# CHECK-ERRORS-NEXT: .loh 1 L1
# CHECK-ERRORS-NEXT:          ^
.loh 1 L1
