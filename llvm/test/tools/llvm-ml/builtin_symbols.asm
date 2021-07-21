; RUN: llvm-ml -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s

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
