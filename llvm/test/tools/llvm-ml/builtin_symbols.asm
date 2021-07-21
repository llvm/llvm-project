; RUN: llvm-ml -filetype=s %s /I %S /Fo /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-REALTIME
; RUN: llvm-ml -filetype=s %s /I %S /Fo /dev/null --timestamp=0 --utc 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-FIXEDTIME

.code

version_val TEXTEQU %@Version

ECHO t1:
%ECHO @Version = version_val
; CHECK-LABEL: t1:
; CHECK-NEXT: 1427

ECHO

ECHO t2:
if @Version gt 510
ECHO @Version gt 510
endif
; CHECK-LABEL: t2:
; CHECK-NEXT: @Version gt 510

ECHO

ECHO t3:
if @Version le 510
ECHO le 510
endif
; CHECK-LABEL: t3:
; CHECK-NOT: @Version le 510

ECHO

line_val TEXTEQU %@Line

ECHO t4:
%ECHO @Line = line_val
; CHECK-LABEL: t4:
; CHECK-NEXT: @Line = [[# @LINE - 5]]

ECHO t5:
include builtin_symbols_t5.inc
; CHECK-LABEL: t5:
; CHECK: FileCur = {{.*}}builtin_symbols_t5.inc
; CHECK: FileName = BUILTIN_SYMBOLS
; CHECK-NOT: _T5

ECHO t6:
%ECHO Date = @Date
%ECHO Time = @Time

; CHECK-LABEL: t6:
; CHECK-REALTIME: Date = {{([[:digit:]]{2}/[[:digit:]]{2}/[[:digit:]]{2})}}
; CHECK-FIXEDTIME: Date = 01/01/70
; CHECK-NOT: {{[[:digit:]]}}
; CHECK-REALTIME: Time = {{([[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2})}}
; CHECK-FIXEDTIME: Time = 00:00:00
; CHECK-NOT: {{[[:digit:]]}}

end
