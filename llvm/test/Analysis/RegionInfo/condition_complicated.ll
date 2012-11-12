; RUN: opt -regions -analyze < %s | FileCheck %s
; RUN: opt -regions -stats < %s 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -regions -print-region-style=bb  -analyze < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define internal fastcc zeroext i8 @handle_compress() nounwind {
end165:
  br i1 1, label %false239, label %true181

true181:
  br i1 1, label %then187, label %else232

then187:
  br label %end265

else232:
  br i1 1, label %false239, label %then245

false239:
  br i1 1, label %then245, label %else259

then245:
  br i1 1, label %then251, label %end253

then251:
  br label %end253

end253:
  br label %end265

else259:
  br label %end265

end265:
  br i1 1, label %then291, label %end298

then291:
  br label %end298

end298:
  ret i8 1
}

; CHECK-NOT: =>
; CHECK: [0] end165 => <Function Return>
; CHECK-NEXT: [1] end165 => end265
; CHECK-NEXT: [2] then245 => end253
; CHECK-NEXT: [1] end265 => end298

; STAT: 4 region - The # of regions

; BBIT: end165, false239, then245, then251, end253, end265, then291, end298, else259, true181, then187, else232,
; BBIT: end165, false239, then245, then251, end253, else259, true181, then187, else232,
; BBIT: then245, then251,
; BBIT: end265, then291,

; RNIT: end165 => end265, end265 => end298, end298,
; RNIT: end165, false239, then245 => end253, end253, else259, true181, then187, else232,
; RNIT: then245, then251,
; RNIT: end265, then291,
