# RUN: not llvm-mc -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

#      CHECK:{{.*}}.s:[[#@LINE+3]]:21: error: vararg parameter 'a' should be the last parameter
# CHECK-NEXT:.macro two a:vararg b:vararg
# CHECK-NEXT:                    ^
.macro two a:vararg b:vararg

#      CHECK:{{.*}}.s:[[#@LINE+3]]:17: error: expected identifier in '.macro' directive
# CHECK-NEXT:.macro one a:req:vararg
# CHECK-NEXT:                ^
.macro one a:req:vararg

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:32: warning: pointless default value for required parameter 'a' in macro 'pointless_default'
# CHECK-NEXT:.macro pointless_default a:req=default
# CHECK-NEXT:                               ^
.macro pointless_default a:req=default
.endm
