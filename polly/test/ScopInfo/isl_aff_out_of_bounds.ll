; RUN: opt %loadPolly -basic-aa -polly-detect < %s

; Used to fail with:
; ../../isl/isl_aff.c:591: position out of bounds

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare double @frexp(double)

define void @vorbis_lsp_to_curve(float* %lsp, i32 %m) {
entry:
  %q.1.reg2mem = alloca float, align 4
  br i1 undef, label %do.body, label %while.end

do.body:                                          ; preds = %do.body, %entry
  %ftmp.0 = phi float* [ %add.ptr, %do.body ], [ %lsp, %entry ]
  %add.ptr = getelementptr inbounds float, float* %ftmp.0, i64 2
  br i1 true, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  br i1 false, label %if.end.single_exit, label %if.then

if.then:                                          ; preds = %do.end
  %0 = load float, float* %add.ptr, align 4
  store float %0, float* %q.1.reg2mem, align 4
  br label %if.end.single_exit

if.end.single_exit:                               ; preds = %do.end, %if.then
  br label %if.end

if.end:                                           ; preds = %if.end.single_exit
  %q.1.reload = load float, float* %q.1.reg2mem, align 4
  %conv31 = fpext float %q.1.reload to double
  %call32 = call double @frexp(double %conv31)
  unreachable

while.end:                                        ; preds = %entry
  ret void
}
