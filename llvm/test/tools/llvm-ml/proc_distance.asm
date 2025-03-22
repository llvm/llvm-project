; RUN: llvm-ml -m32 -filetype=s %s /Fo - | FileCheck %s

.code

DefaultProc PROC
  ret
DefaultProc ENDP
; CHECK: DefaultProc:
; CHECK: {{^ *}}ret{{ *$}}

t1:
call DefaultProc
; CHECK: t1:
; CHECK-NEXT: call DefaultProc

NearProc PROC NEAR
  ret
NearProc ENDP
; CHECK: NearProc:
; CHECK: {{^ *}}ret{{ *$}}

t2:
call NearProc
; CHECK: t2:
; CHECK-NEXT: call NearProc

FarProcInCode PROC FAR
  ret
FarProcInCode ENDP
; CHECK: FarProcInCode:
; CHECK: {{^ *}}retf{{ *$}}

t3:
call FarProcInCode
; CHECK: t3:
; CHECK-NEXT: push cs
; CHECK-NEXT: call FarProcInCode

FarCode SEGMENT SHARED NOPAGE NOCACHE INFO READ WRITE EXECUTE DISCARD

FarProcInFarCode PROC FAR
  ret
FarProcInFarCode ENDP
; CHECK: FarProcInFarCode:
; CHECK: {{^ *}}retf{{ *$}}

FarCode ENDS

.code

t4:
call FarProcInFarCode
; CHECK: t4:
; CHECK-NEXT: call FarProcInFarCode

END
