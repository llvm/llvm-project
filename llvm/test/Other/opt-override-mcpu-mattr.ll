; RUN: opt < %s -mtriple=x86_64-apple-darwin -mcpu=broadwell -mattr=+avx2 -S | FileCheck %s

; Check that opt can add but not rewrite function attributes
; target-cpu and target-features using command line options -mcpu and
; -mattr.

; CHECK: attributes #0 = { nounwind readnone ssp uwtable "target-cpu"="broadwell" "target-features"="+ssse3,+cx16,+sse,+sse2,+sse3,+avx2" "use-soft-float"="false" }
; CHECK: attributes #1 = { nounwind readnone ssp uwtable "target-cpu"="core2" "target-features"="+ssse3,+cx16,+sse,+sse2,+sse3,+avx2" "use-soft-float"="false" }

define i32 @no_target_cpu() #0 {
entry:
  ret i32 0
}

define i32 @has_targe_cpu() #1 {
entry:
  ret i32 0
}

attributes #0 = { nounwind readnone ssp uwtable "target-features"="+ssse3,+cx16,+sse,+sse2,+sse3" "use-soft-float"="false" }
attributes #1 = { nounwind readnone ssp uwtable "target-cpu"="core2" "target-features"="+ssse3,+cx16,+sse,+sse2,+sse3" "use-soft-float"="false" }
