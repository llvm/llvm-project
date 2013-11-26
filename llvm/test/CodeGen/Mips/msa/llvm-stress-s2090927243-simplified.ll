; RUN: llc -march=mips < %s
; RUN: llc -march=mips -mattr=+msa,+fp64 < %s
; RUN: llc -march=mipsel < %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s

; This test originally failed for MSA with a "Cannot select ..." error.
; This was because undef's are ignored when checking if a vector constant is a
; splat, but are legalized to zero if left in the DAG which changes the constant
; into a non-splat.
;
; It should at least successfully build.

define void @autogen_SD2090927243() {
BB:
  br label %CF77

CF77:                                             ; preds = %CF77, %CF80
  %Shuff27 = shufflevector <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>,
                           <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>,
                           <16 x i32> <i32 27, i32 29, i32 31, i32 1, i32 3, i32 5, i32 undef, i32 9, i32 11, i32 13, i32 undef, i32 17, i32 19, i32 21, i32 23, i32 undef>
  %ZE30 = zext <16 x i8> %Shuff27 to <16 x i32>
  %Cmp32 = fcmp ueq float undef, 0x3CDA6E5E40000000
  br i1 %Cmp32, label %CF77, label %CF

CF:                                               ; preds = %CF, %CF81
  %E48 = extractelement <16 x i32> %ZE30, i32 14
  br i1 undef, label %CF, label %CF78

CF78:                                             ; preds = %CF
  ret void
}
