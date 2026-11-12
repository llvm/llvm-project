# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

# CHECK: {{.*}}.s:[[#@LINE+3]]:1: error: local changed binding to STB_GLOBAL
local:
.local local
.globl local

## `.globl x; .weak x` matches the GNU as behavior. We issue a warning for now.
# CHECK: {{.*}}.s:[[#@LINE+3]]:1: warning: global changed binding to STB_WEAK
global:
.global global
.weak global

# CHECK: {{.*}}.s:[[#@LINE+3]]:1: error: weak changed binding to STB_LOCAL
weak:
.weak weak
.local weak

# CHECK-NOT: error:
multi_local:
.local multi_local
.local multi_local
multi_global:
.global multi_global
.global multi_global
multi_weak:
.weak multi_weak
.weak multi_weak
