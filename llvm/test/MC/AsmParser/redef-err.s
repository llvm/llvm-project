## Test redefinition errors. MCAsmStreamer::emitLabel is different from MCObjectStreamer. Test both streamers.
# RUN: not llvm-mc -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

l:
.set l, .
# CHECK: [[#@LINE-1]]:9: error: redefinition of 'l'

.equiv a, undef
.set a, 3
# CHECK: [[#@LINE-1]]:9: error: redefinition of 'a'

.equiv a, undef
# CHECK: [[#@LINE-1]]:11: error: redefinition of 'a'

.equiv b, undef
b:
# CHECK: [[#@LINE-1]]:1: error: symbol 'b' is already defined
