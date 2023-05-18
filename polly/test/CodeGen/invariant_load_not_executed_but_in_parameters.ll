; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true -disable-output < %s
;
; Check that this does not crash as the invariant load is not executed (thus
; not preloaded) but still referenced by one of the parameters.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.Exp.204.248.358 = type { ptr, i32, i32, i32, ptr, ptr, %union.anon.2.201.245.355, %union.anon.3.202.246.356, %union.anon.4.203.247.357 }
%struct.Exp_.200.244.354 = type { i32, i32, i32, i32, ptr }
%struct.Id.199.243.353 = type { ptr, i32, i32, i32, %union.anon.1.198.242.352 }
%union.anon.1.198.242.352 = type { [2 x i64] }
%union.anon.2.201.245.355 = type { ptr }
%union.anon.3.202.246.356 = type { i32 }
%union.anon.4.203.247.357 = type { ptr }
%struct.Classfile.218.262.372 = type { ptr, ptr, i32, i32, i32, %struct.ClassVersion.207.251.361, %struct.ConstPool.210.254.364, %struct.AccessFlags.211.255.365, i16, ptr, ptr, i16, ptr, i16, ptr, i16, ptr, i16, ptr, ptr, i16, ptr, ptr }
%struct._IO_FILE.206.250.360 = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, [1 x i8], ptr, i64, ptr, ptr, ptr, ptr, i64, i32, [20 x i8] }
%struct._IO_marker.205.249.359 = type { ptr, ptr, i32 }
%struct.ClassVersion.207.251.361 = type { i16, i16 }
%struct.ConstPool.210.254.364 = type { i16, ptr }
%struct.cp_info.209.253.363 = type { i8, %union.anon.208.252.362 }
%union.anon.208.252.362 = type { i64 }
%struct.AccessFlags.211.255.365 = type { i16 }
%struct.field_info.212.256.366 = type <{ %struct.AccessFlags.211.255.365, [6 x i8], ptr, ptr, i32, i16, [2 x i8] }>
%struct.method_info.217.261.371 = type { %struct.AccessFlags.211.255.365, ptr, ptr, i8, i8, i32, ptr, i16, ptr, i16, ptr, i16, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr }
%struct.Block.214.258.368 = type { i32, i16, i16, %union.anon.0.213.257.367, i16, ptr }
%union.anon.0.213.257.367 = type { i32 }
%struct.LineNumberTableEntry.215.259.369 = type { i16, i16 }
%struct.LocalVariableTableEntry.216.260.370 = type { i16, i16, i16, i16, i16 }
%struct.Case.219.263.373 = type { i64, i64 }

@currpc = external global i32, align 4
@bufflength = external global i32, align 4
@inbuff = external global ptr, align 8
@stkptr = external global ptr, align 8
@donestkptr = external global ptr, align 8

; Function Attrs: uwtable
define i32 @_Z13dotableswitchP9Classfile(ptr %c) #0 personality ptr @__gxx_personality_v0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %sub = add i32 0, -1
  %tobool.5 = icmp eq i32 0, 0
  br i1 %tobool.5, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry.split
  br label %while.body

while.body:                                       ; preds = %while.body, %while.body.lr.ph
  %0 = load i32, ptr @currpc, align 4
  %rem = and i32 %0, 3
  %tobool = icmp eq i32 %rem, 0
  br i1 %tobool, label %while.cond.while.end_crit_edge, label %while.body

while.cond.while.end_crit_edge:                   ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.cond.while.end_crit_edge, %entry.split
  invoke void @_ZN3ExpC2Ejj7Exptype4Type2OpPS_jjP4Case(ptr nonnull undef, i32 %sub, i32 undef, i32 9, i32 0, i32 39, ptr undef, i32 undef, i32 undef, ptr nonnull undef)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %while.end
  br i1 undef, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %invoke.cont
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  br i1 undef, label %for.cond.for.end_crit_edge, label %for.body

lpad:                                             ; preds = %while.end
  %1 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } undef

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %invoke.cont
  ret i32 0
}

; Function Attrs: nounwind readnone
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) #1

; Function Attrs: nobuiltin
declare noalias ptr @_Znam(i64) #2

; Function Attrs: nobuiltin
declare noalias ptr @_Znwm(i64) #2

; Function Attrs: uwtable
declare void @_ZN3ExpC2Ejj7Exptype4Type2OpPS_jjP4Case(ptr, i32, i32, i32, i32, i32, ptr, i32, i32, ptr) unnamed_addr #0 align 2

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(ptr) #3

; Function Attrs: uwtable
declare i32 @_Z10doluswitchP9Classfile(ptr) #0

; Function Attrs: nounwind uwtable
declare void @_ZN4Exp_C2E7Exptype4Type2Op(ptr, i32, i32, i32) unnamed_addr #4 align 2

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nobuiltin "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nobuiltin nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0"}
