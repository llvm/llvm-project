# RUN: not llvm-mc -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

#      CHECK:{{.*}}.s:[[#@LINE+3]]:36: error: missing parameter qualifier for 'parameter' in macro 'missing_qualifier'
# CHECK-NEXT:.macro missing_qualifier parameter:
# CHECK-NEXT:                                   ^
.macro missing_qualifier parameter:

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:43: error: missing parameter qualifier for 'parameter' in macro 'non_identifier_qualifier'
# CHECK-NEXT:.macro non_identifier_qualifier parameter:0
# CHECK-NEXT:                                          ^
.macro non_identifier_qualifier parameter:0

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:36: error: invalid_qualifier is not a valid parameter qualifier for 'parameter' in macro 'invalid_qualifier'
# CHECK-NEXT:.macro invalid_qualifier parameter:invalid_qualifier
# CHECK-NEXT:                                   ^
.macro invalid_qualifier parameter:invalid_qualifier

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:40: warning: pointless default value for required parameter 'parameter' in macro 'pointless_default'
# CHECK-NEXT:.macro pointless_default parameter:req=default
# CHECK-NEXT:                                       ^
.macro pointless_default parameter:req=default
.endm

# CHECK-NEXT:{{.*}}.s:[[#@LINE+5]]:17: error: missing value for required parameter 'parameter' in macro 'missing_required'
# CHECK-NEXT:missing_required
# CHECK-NEXT:                ^
.macro missing_required parameter:req
.endm
missing_required

# CHECK-NEXT:{{.*}}.s:[[#@LINE+5]]:24: error: missing value for required parameter 'second' in macro 'missing_second_required'
# CHECK-NEXT:missing_second_required
# CHECK-NEXT:                       ^
.macro missing_second_required first=0 second:req
.endm
missing_second_required

# CHECK-NEXT:{{.*}}.s:[[#@LINE+8]]:24: error: missing value for required parameter 'second' in macro 'second_third_required'
# CHECK-NEXT:second_third_required 0
# CHECK-NEXT:                       ^
# CHECK-NEXT:{{.*}}.s:[[#@LINE+5]]:24: error: missing value for required parameter 'third' in macro 'second_third_required'
# CHECK-NEXT:second_third_required 0
# CHECK-NEXT:                       ^
.macro second_third_required first=0 second:req third:req
.endm
second_third_required 0

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:38: error: missing value for required parameter 'second' in macro 'second_third_required'
# CHECK-NEXT:second_third_required third=3 first=1
# CHECK-NEXT:                                     ^
second_third_required third=3 first=1
