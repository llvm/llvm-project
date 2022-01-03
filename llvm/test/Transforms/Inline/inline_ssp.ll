; RUN: opt -inline %s -S | FileCheck %s
; RUN: opt -passes='cgscc(inline)' %s -S | FileCheck %s
; Ensure SSP attributes are propagated correctly when inlining.

@.str = private unnamed_addr constant [11 x i8] c"fun_nossp\0A\00", align 1
@.str1 = private unnamed_addr constant [9 x i8] c"fun_ssp\0A\00", align 1
@.str2 = private unnamed_addr constant [15 x i8] c"fun_sspstrong\0A\00", align 1
@.str3 = private unnamed_addr constant [12 x i8] c"fun_sspreq\0A\00", align 1

; These first four functions (@fun_sspreq, @fun_sspstrong, @fun_ssp, @fun_nossp)
; are used by the remaining functions to ensure that the SSP attributes are
; propagated correctly.  If the caller had an SSP attribute before inlining, it
; should have its new SSP attribute set as:
; strictest(caller-ssp-attr, callee-ssp-attr), where strictness is ordered as:
;  sspreq > sspstrong > ssp

define internal void @fun_sspreq() sspreq {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str3, i32 0, i32 0))
  ret void
}

define internal void @fun_sspreq_alwaysinline() sspreq alwaysinline {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str3, i32 0, i32 0))
  ret void
}

define internal void @fun_sspstrong() sspstrong {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str2, i32 0, i32 0))
  ret void
}

define internal void @fun_ssp() ssp {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str1, i32 0, i32 0))
  ret void
}

define internal void @fun_nossp() {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0))
  ret void
}

; Tests start below.

define void @inline_req_req() sspreq {
entry:
; CHECK: @inline_req_req() #[[SSPREQ:[0-9]]]
  call void @fun_sspreq()
  ret void
}

define void @inline_req_strong() sspstrong {
entry:
; CHECK: @inline_req_strong() #[[SSPREQ]]
  call void @fun_sspreq()
  ret void
}

define void @inline_req_ssp() ssp {
entry:
; CHECK: @inline_req_ssp() #[[SSPREQ]]
  call void @fun_sspreq()
  ret void
}

define void @inline_req_nossp() {
entry:
; CHECK: @inline_req_nossp() {
  call void @fun_sspreq()
  ret void
}

define void @alwaysinline_req_nossp() {
entry:
; CHECK: @alwaysinline_req_nossp() {
  call void @fun_sspreq_alwaysinline()
  ret void
}

define void @inline_strong_req() sspreq {
entry:
; CHECK: @inline_strong_req() #[[SSPREQ]]
  call void @fun_sspstrong()
  ret void
}


define void @inline_strong_strong() sspstrong {
entry:
; CHECK: @inline_strong_strong() #[[SSPSTRONG:[0-9]]]
  call void @fun_sspstrong()
  ret void
}

define void @inline_strong_ssp() ssp {
entry:
; CHECK: @inline_strong_ssp() #[[SSPSTRONG]]
  call void @fun_sspstrong()
  ret void
}

define void @inline_strong_nossp() {
entry:
; CHECK: @inline_strong_nossp() {
  call void @fun_sspstrong()
  ret void
}

define void @inline_ssp_req() sspreq {
entry:
; CHECK: @inline_ssp_req() #[[SSPREQ]]
  call void @fun_ssp()
  ret void
}


define void @inline_ssp_strong() sspstrong {
entry:
; CHECK: @inline_ssp_strong() #[[SSPSTRONG]]
  call void @fun_ssp()
  ret void
}

define void @inline_ssp_ssp() ssp {
entry:
; CHECK: @inline_ssp_ssp() #[[SSP:[0-9]]]
  call void @fun_ssp()
  ret void
}

define void @inline_ssp_nossp() {
entry:
; CHECK: @inline_ssp_nossp() {
  call void @fun_ssp()
  ret void
}

define void @inline_nossp_req() sspreq {
entry:
; CHECK: @inline_nossp_req() #[[SSPREQ]]
  call void @fun_nossp()
  ret void
}


define void @inline_nossp_strong() sspstrong {
entry:
; CHECK: @inline_nossp_strong() #[[SSPSTRONG]]
  call void @fun_nossp()
  ret void
}

define void @inline_nossp_ssp() ssp {
entry:
; CHECK: @inline_nossp_ssp() #[[SSP]]
  call void @fun_nossp()
  ret void
}

define void @inline_nossp_nossp() {
entry:
; CHECK: @inline_nossp_nossp() {
  call void @fun_nossp()
  ret void
}

declare i32 @printf(i8*, ...)

; CHECK: attributes #[[SSPREQ]] = { sspreq }
; CHECK: attributes #[[SSPSTRONG]] = { sspstrong }
; CHECK: attributes #[[SSP]] = { ssp }
