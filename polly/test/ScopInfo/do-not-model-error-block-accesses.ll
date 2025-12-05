; RUN: opt %loadNPMPolly '-passes=polly-custom<scops>' -polly-print-scops -disable-output < %s

; Check that we do not crash on this input. Earlier this indeed crashed as
; we tried to model the access functions in an error block.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @FORMAT3_4() {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br i1 false, label %if.end.38, label %if.else

if.else:                                          ; preds = %entry.split
  call void (i32, i32, ptr, ...) @BYTES_TO_BITS(i32 undef, i32 1, ptr undef)
  %0 = load i32, ptr null, align 4
  br label %if.end.38

if.end.38:                                        ; preds = %if.else, %entry.split
  unreachable
}

declare void @BYTES_TO_BITS(...)

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0"}
